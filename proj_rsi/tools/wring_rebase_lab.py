"""Root-only QKV128 separate-W-ring lab. No CUDA calls at import; no official admission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'candidates/pw36_wring_rebase'
LABELS=('incumbent','coupled','equal','split')
ARM_NAMES=('coupled_x5w5','split_x5w5','split_x6w4')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_frozen():
    m=json.loads((HERE/'manifest.json').read_text())
    for path,digest in m['files'].items():assert sha(HERE/path)==digest,('candidate drift',path)
    for path,digest in m['external_files'].items():assert sha(ROOT/path)==digest,('driver drift',path)
    assert sha(ROOT/'candidates/pw36/submission.py')==m['parent_source_sha256']
    return m

def terminal_gate(outputs,x,fixtures,seeds,last,checker,integrity):
    # Read only: this must execute before any subsequent replay/write/profile.
    assert set(outputs)==set(seeds)==set(LABELS)
    assert torch.equal(x,fixtures[last['seed']]),'current X differs from last actual call'
    checks=[]
    for label,out in outputs.items():
        assert bool(torch.isfinite(out).all()),(label,'nonfinite terminal output')
        ok,message=checker(fixtures[seeds[label]],out)
        assert ok,(label,'terminal own-fixture mismatch',seeds[label],message)
        checks.append(dict(label=label,input_seed=seeds[label],result=message))
    integrity()
    return dict(checks=checks,current_X_matches_last_call=True,before_any_replay_or_mutation=True)

def measure(graphs,x,fixtures,strict_gate):
    flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),dtype=torch.float32,device='cuda')
    calls=[];seeds={};boundaries={}
    def one(label,seed,mode):
        x.copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graphs[label].replay();end.record();torch.cuda.synchronize()
        us=start.elapsed_time(end)*1000
        calls.append(dict(label=label,seed=seed,mode=mode,us=us))
        if mode!='warmup':seeds[label]=seed
        return us
    for label in LABELS:
        for _ in range(5):one(label,600000,'warmup')
    raw={};block=0
    for label in LABELS[:3]:
        raw[label+'_before']=[one(label,601000+21*block+i,'F1') for i in range(21)];block+=1
    raw['split']=[one('split',601000+21*block+i,'F1') for i in range(21)];block+=1
    for label in reversed(LABELS[:3]):
        raw[label+'_after']=[one(label,601000+21*block+i,'F1') for i in range(21)];block+=1
    boundaries['F1']=strict_gate(dict(seeds),dict(calls[-1]))
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[i:]+seq[:i] for i in range(4)])
    pairs=[]
    for repetition in range(32):
        row={}
        for index in orders[repetition%8]:row[LABELS[index]]=one(LABELS[index],602000+repetition,'paired')
        pairs.append(row)
    boundaries['paired']=strict_gate(dict(seeds),dict(calls[-1]))
    med={key:statistics.median(values) for key,values in raw.items()}
    comparisons={}
    # Report every causal contrast, including equal-depth split vs coupled.
    for a,b in (('incumbent','split'),('coupled','equal'),('coupled','split'),('equal','split'),('incumbent','equal'),('incumbent','coupled')):
        f1a=med[a] if a=='split' else min(med[a+'_before'],med[a+'_after'])
        f1b=med[b] if b=='split' else min(med[b+'_before'],med[b+'_after'])
        comparisons[a+'_minus_'+b]=dict(F1_gain_us=f1a-f1b,paired_median_gain_us=statistics.median(row[a]-row[b] for row in pairs),paired_wins=sum(row[a]>row[b] for row in pairs))
    return dict(F1_raw_us=raw,F1_medians_us=med,paired_raw_us=pairs,comparisons=comparisons,all_calls=calls,terminal_checks=boundaries,
                policy='All295rawcalls retained,147F1/128paired/20warm,32pairedrounds/eightbalancedorders; no tiny-gain retention cutoff or promotion inference')

def storage_contract(tensor,active=None):
    assert tensor.is_contiguous()
    storage=tensor.untyped_storage();raw=torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(storage,0,(storage.nbytes(),),(1,))
    baseline=raw.clone();identity=(str(tensor.device),storage.data_ptr(),storage.nbytes(),tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))
    if active is not None:
        assert tensor.dtype==torch.int32 and tensor.ndim==1 and active<=tensor.numel()
        assert not int(torch.count_nonzero(tensor[:active]))
    def check():
        now=tensor.untyped_storage()
        assert (str(tensor.device),now.data_ptr(),now.nbytes(),tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))==identity,'counter allocation identity changed'
        if active is None:assert torch.equal(raw,baseline),'unrelated counter storage changed'
        else:
            lo=tensor.storage_offset()*tensor.element_size();hi=lo+active*tensor.element_size()
            assert not int(torch.count_nonzero(tensor[:active])),'active RQ counter not reset'
            assert torch.equal(raw[:lo],baseline[:lo]) and torch.equal(raw[hi:],baseline[hi:]),'dormant counter tail/guards changed'
    return check,identity[:3]

def collect_counters(obj,prefix='',seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return []
    seen.add(id(obj));out=[]
    if isinstance(obj,dict):
        for name,value in obj.items():
            path=prefix+'/'+str(name)
            if torch.is_tensor(value) and ('cnt' in str(name).lower() or 'counter' in str(name).lower()):out.append((path,value))
            elif isinstance(value,(dict,list,tuple)):out.extend(collect_counters(value,path,seen))
    elif isinstance(obj,(list,tuple)):
        for i,value in enumerate(obj):out.extend(collect_counters(value,prefix+'/'+str(i),seen))
    return out

def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn(x,out)
    torch.cuda.synchronize();return g

def main():
    global torch
    import torch
    from torch.utils.cpp_extension import load_inline
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);args=ap.parse_args();args.json.parent.mkdir(parents=True,exist_ok=True)
    frozen=verify_frozen();sys.path.insert(0,str(HERE));import preflight as gates
    P=load(HERE/'submission.py','wring_candidate');B=load(ROOT/'candidates/pw36/submission.py','wring_frozen_parent')
    sys.path.insert(0,str(ROOT.parents[0]/'muse_bf16_proj'));import reference
    extension_source=gates.extension_source((HERE/'submission.py').read_text())
    assert extension_source==(HERE/'extension_source.cu').read_text()
    result=dict(status='STARTED_NO_TIMING_ADMISSION',source_sha256=frozen['files']['submission.py'],manifest_sha256=sha(HERE/'manifest.json'),GPU='ROOT_ONLY',attrs={})
    try:
        ext=load_inline('pw36_wring_'+hashlib.sha256(extension_source.encode()).hexdigest()[:16],
           cpp_sources='int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t); int64_t wrinfo(int64_t,int64_t);',
           cuda_sources=extension_source,functions=['rqrun','wrinfo'],extra_cuda_cflags=gates.FLAGS,verbose=True)
    except Exception as error:
        result.update(status='INVALID_EXTENSION_BUILD',error=repr(error),no_candidate_math_launch=True,graph_cleanup='NO_GRAPHS_CREATED')
        args.json.write_text(json.dumps(result,indent=2)+'\n');raise
    graphs={}
    def save():args.json.write_text(json.dumps(result,indent=2)+'\n')
    try:
        binary=Path(ext.__file__).resolve();sass=subprocess.check_output([shutil.which('cuobjdump') or '/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True,timeout=30)
        artifact=args.json.with_suffix('.jit.sass');artifact.write_text(sass)
        actual,all_canonical=gates.streams(sass),gates.streams((HERE/'sass.txt').read_text())
        canonical={args:all_canonical[args] for args in gates.ARMS.values()}
        assert actual==canonical,'ACTUAL_JIT_MAIN_SEMANTIC_DRIFT'
        result['actual_JIT']=dict(binary=str(binary),sass=str(artifact),sha256=hashlib.sha256(sass.encode()).hexdigest(),physical=gates.physical(sass,expected=3),exact_all_main_streams=True)
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        for i,label in enumerate(LABELS[1:]):
            attrs={key:int(ext.wrinfo(i,code)) for key,code in [('regs',1),('local',2),('smem',3),('occupancy',4),('max_dynamic',5),('max_threads',6)]}
            assert attrs['regs']==168 and attrs['local']==0 and attrs['max_threads']>=384 and attrs['occupancy']>=1
            assert attrs['smem']=={'coupled':144520,'equal':144560,'split':148656}[label]
            assert attrs['max_dynamic']>=attrs['smem']
            assert ((attrs['regs']+7)//8)*8*384<=65536
            assert 18*7<=props.multi_processor_count*attrs['occupancy'],'ZI splitK grid must be co-resident'
            result['attrs'][label]=attrs
        save() # no candidate math has launched before JIT/resource gates.
        weights=reference.generate_weights(reference.WEIGHTS_SEED);weight_before={name:t.clone() for name,t in weights.items()}
        state=B.prepare(weights);w=weights['qkv_gate'];selected=state.get('rq',{}).get(('qkv_gate',128))
        assert selected is not None and selected['n']==2304 and selected['KS']==7,'UNSUPPORTED_INCUMBENT_SELECTION_NO_ADMISSION'
        names=B.RQPROD2 if 'mod' in selected else B.RQPROD;selected_name=names[selected['i']];variant=B.RQV[B.RQI[selected_name]]
        assert variant[1]==128 and variant[-1] in (2,3),'incumbent active counter contract requires current full-output ZI variant'
        result['actual_incumbent']=dict(variant=selected_name,KS=selected['KS'],not_forced=True)
        original_call=B._rqcall;guarded_calls=[0]
        def guarded_call(st,p,x,out):
            assert p is selected,'incumbent selected plan changed'
            ret=original_call(st,p,x,out);assert ret==0,'INCUMBENT_FALLBACK_NO_ADMISSION';guarded_calls[0]+=1;return ret
        B._rqcall=guarded_call
        incumbent_ext=selected.get('mod',B._rqmod[0]);incumbent_binary=Path(incumbent_ext.__file__).resolve()
        incumbent_sass=subprocess.check_output([shutil.which('cuobjdump') or '/usr/local/cuda/bin/cuobjdump','--dump-sass',str(incumbent_binary)],text=True,timeout=30)
        incumbent_artifact=args.json.with_suffix('.incumbent.jit.sass');incumbent_artifact.write_text(incumbent_sass)
        selected_args=tuple(variant[1:8]);original_control=gates.streams((HERE/'control-sass.txt').read_text())
        assert selected_args in original_control and gates.streams(incumbent_sass)[selected_args]==original_control[selected_args],'INCUMBENT_ACTUAL_JIT_CONTROL_DRIFT'
        result['actual_incumbent']['exact_selected_parent_JIT']=True;result['actual_incumbent']['sass']=str(incumbent_artifact)
        lay,tab,esc6=P._rqpack(w);assert torch.equal(P._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
        esc,_=P._rqesc(esc6,18,128,7,'cuda');empty=torch.empty(0,dtype=torch.int64,device='cuda');sink=torch.zeros(1,dtype=torch.int32,device='cuda')
        counters={};counter_checks=[];storage_ids=set()
        for label in LABELS[1:]:
            # Visible72ints:36active,36dormant;8guardints on each side.
            backing=torch.full((88,),0x6a17c25,dtype=torch.int32,device='cuda');counter=backing[8:80];counter[:36].zero_();counters[label]=counter
            check,identity=storage_contract(counter,36);counter_checks.append(check);storage_ids.add(identity)
        check,identity=storage_contract(selected['cnt'],36);counter_checks.append(check);storage_ids.add(identity)
        for name,tensor in collect_counters(state):
            st=tensor.untyped_storage();identity=(str(tensor.device),st.data_ptr(),st.nbytes())
            if identity not in storage_ids:
                check,identity=storage_contract(tensor);counter_checks.append(check);storage_ids.add(identity)
        immutable=[(t,weight_before[name]) for name,t in weights.items()]+[(lay,lay.clone()),(esc,esc.clone()),(selected['lay'],selected['lay'].clone()),(selected['esc'],selected['esc'].clone())]
        def integrity():
            assert all(torch.equal(a,b) for a,b in immutable),'weights/packed/CSR immutable failure'
            for check in counter_checks:check()
        funcs={}
        for i,label in enumerate(LABELS[1:]):
            def fn(x,out,i=i,label=label):
                code=int(ext.rqrun(i,lay,x,out,esc,counters[label],7,tab,0,empty,sink,0))
                assert code==0,(label,'launch',code);return 0
            funcs[label]=fn
        config={'layer':'qkv_gate','m':128,'n':2304,'k':8192};hold=[None]
        def incumbent(x,out):
            before=guarded_calls[0];hold[0]=B.kernel_fn({'x':x,'config':config},state)
            assert guarded_calls[0]==before+1,'INCUMBENT_SELECTED_RQ_NOT_EXECUTED'
            return hold[0]
        def incumbent_validate(x,out):out.copy_(incumbent(x,out));return 0
        result['validation']={}
        for label,fn in {'incumbent':incumbent_validate,**funcs}.items():
            gen=torch.Generator(device='cuda').manual_seed(7401)
            ok,message=P._rqval(fn,w,128,2304,gen,4e-3);assert ok,(label,message)
            checks=[dict(all8192_basis='EXACT',FP64='PASS',detail=message)]
            for dist in ('normal','outliers'):
                data=reference.generate_input(128,'qkv_gate',7403,dist=dist);before=data['x'].clone();out=torch.full((128,2304),float('nan'),dtype=torch.bfloat16,device='cuda')
                fn(data['x'],out);ok,message=reference.check_implementation(data,out,{'qkv_gate':w});assert ok and torch.equal(data['x'],before),(label,message)
                checks.append(dict(task_distribution=dist,result=message,fresh_current_X_and_output=True));integrity()
            result['validation'][label]=checks;save()
        data=reference.generate_input(128,'qkv_gate',7411);x=data['x'];outputs={label:torch.empty((128,2304),dtype=torch.bfloat16,device='cuda') for label in LABELS[1:]}
        graphs['incumbent']=capture(incumbent,x,outputs['coupled']);outputs={'incumbent':hold[0],**outputs}
        for label,fn in funcs.items():graphs[label]=capture(fn,x,outputs[label])
        assert len({out.data_ptr() for out in outputs.values()})==4
        seeds=[600000]+list(range(601000,601147))+list(range(602000,602032));fixtures={seed:reference.generate_input(128,'qkv_gate',seed)['x'] for seed in seeds}
        def checker(fresh,out):return reference.check_implementation({'x':fresh,'config':config},out,{'qkv_gate':w})
        def strict(last_seeds,last):return terminal_gate(outputs,x,fixtures,last_seeds,last,checker,integrity)
        result['timing']=measure(graphs,x,fixtures,strict);save()
        probes=torch.zeros_like(x);rows=torch.arange(128,device=x.device);probes[rows,(37*rows+19)%8192]=((2.**((rows%5)+3))*(2*(rows%2)-1)).to(torch.bfloat16)
        mutations=[('original',data['x'].clone()),('zero',torch.zeros_like(x)),('normal',reference.generate_input(128,'qkv_gate',7501)['x']),('outliers',reference.generate_input(128,'qkv_gate',7503,dist='outliers')['x']),('all_rows',probes)]
        result['actual_graph_mutations']=[]
        for kind,fresh in mutations:
            x.copy_(fresh)
            for label,g in graphs.items():
                before=x.clone();outputs[label].fill_(float('nan'));g.replay();torch.cuda.synchronize()
                assert torch.equal(x,before) and bool(torch.isfinite(outputs[label]).all())
                if kind=='zero':assert not int(torch.count_nonzero(outputs[label]));message='exact complete zero'
                else:ok,message=checker(x,outputs[label]);assert ok,(label,kind,message)
                integrity();result['actual_graph_mutations'].append(dict(label=label,input=kind,oracle=message,NaN_poison_overwritten=True))
        from torch.profiler import profile,ProfilerActivity
        result['profile']={}
        for label,g in graphs.items():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
            events=[e.name() for e in prof.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]
            result['profile'][label]=events
            assert len(events)==1 and sum('rq<' in name for name in events)==1,(label,events)
        integrity();result['status']='VALID_RESEARCH_RESULT_NO_PROMOTION';result['next']='Retain every contrast including tiny gains; diagnose split-barrier versus depth effect; no family-wide kill or assumed additivity.';save()
    except Exception as error:result.update(status='INVALID_SETUP_RUNTIME_OR_CORRECTNESS',error=repr(error));save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for g in graphs.values():g.reset()
            graphs.clear()
        result['graph_cleanup']='DRAINED_AND_RESET_ALL_CREATED_GRAPHS';save()
if __name__=='__main__':main()
