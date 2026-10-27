"""Root-only narrow PW42 discriminator. Import does not create a CUDA context."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LABELS=('incumbent','generic_a','guarded_a','generic_b','guarded_b')
BASES=((1,1,1,2,16,8,1,1,2,1),(1,1,1,1,16,8,1,1,2,8193))
TRAIT=1<<30

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def source():
    text=(HERE/'submission.py').read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GVCU' for t in n.targets));cu=ast.literal_eval(node.value)
    args=[BASES[0],BASES[0][:-1]+(BASES[0][-1]|TRAIT,),BASES[1],BASES[1][:-1]+(BASES[1][-1]|TRAIT,)]
    cases='\n'.join('case %d:return gvl<%s>(p,st,occ);'%(i,','.join(map(str,a))) for i,a in enumerate(args));cu=cu.replace('@@CASES@@',cases)
    cu+='\ntemplate<int FMT,int MT,int ST,int WR,int WK,int KU,int S,int RT,int XS,int V> int64_t eeattr(int code){auto k=[](){if constexpr(V&1073741824)return gv_empty_guard<FMT,MT,ST,WR,WK,KU,S,RT,XS,(V&~1073741824)>;else return gv<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;}();cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;return code==1?a.numRegs:code==2?(int)a.localSizeBytes:code==3?a.maxThreadsPerBlock:a.sharedSizeBytes;}\n'
    cu+='int64_t eeinfo(int64_t inst,int64_t code){switch(inst){\n'
    for i,a in enumerate(args):cu+='case %d:return eeattr<%s>(code);\n'%(i,','.join(map(str,a)))
    cu+='}return -2;}\n'
    cu+='int64_t eeoccupancy(int64_t inst){G p={};p.m=1;p.N=2304;p.K=8192;switch(inst){\n'
    for i,a in enumerate(args):cu+='case %d:return gvl<%s>(p,nullptr,1);\n'%(i,','.join(map(str,a)))
    return cu+'}return -2;}\n'

def verify_files():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((HERE.parent/'pw38/submission.py').read_bytes()).hexdigest()==manifest['parent_source_sha256']
    return manifest

def gv_sass(sass):
    return ''.join('\n Function : '+t for t in re.split(r'\n\s*Function : ',sass)[1:] if re.match(r'_Z(?:2gvI|14gv_empty_guardI)',t.splitlines()[0]))

def actual_binary_gate(sass):
    sys.path.insert(0,str(HERE))
    import preflight as gate
    import physical_proof
    actual=gv_sass(sass)
    assert gate.tables(actual)==gate.tables((HERE/'sass.txt').read_text()),'ACTUAL_JIT_MAIN_DRIFT'
    return physical_proof.proof(actual)

def terminal(outputs,x,fixtures,last_seeds,last,checker,integrity):
    # Strictly reads retained actual timed outputs; no replay or tensor repair.
    assert set(outputs)==set(last_seeds)==set(LABELS)
    assert torch.equal(x,fixtures[last['seed']]),'current X differs from lastactualcall'
    checks=[]
    for label,out in outputs.items():
        assert bool(torch.isfinite(out).all()),(label,'terminal nonfinite')
        ok,message=checker(fixtures[last_seeds[label]],out);assert ok,(label,'terminal ownfixture',message)
        checks.append(dict(label=label,seed=last_seeds[label],oracle=message))
    integrity();return dict(checks=checks,before_replay_mutation_profile=True)

def measure(graphs,x,fixtures,gate,calls=None):
    controls=('incumbent','generic_a','generic_b');new=('guarded_a','guarded_b')
    flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),dtype=torch.float32,device='cuda')
    calls=[] if calls is None else calls;seeds={};checks={}
    def one(label,seed,mode):
        x.copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        us=a.elapsed_time(b)*1000;calls.append(dict(label=label,seed=seed,mode=mode,us=us))
        assert math.isfinite(us) and us>0,('INVALID_ELAPSED_US',label,seed,mode,us)
        if mode!='warmup':seeds[label]=seed
        return us
    for label in LABELS:
        for _ in range(3):one(label,800000,'warmup')
    raw={};block=0
    for label in controls:
        raw[label+'_before']=[one(label,801000+21*block+i,'F1') for i in range(21)];block+=1
    for label in new:
        raw[label]=[one(label,801000+21*block+i,'F1') for i in range(21)];block+=1
    for label in reversed(controls):
        raw[label+'_after']=[one(label,801000+21*block+i,'F1') for i in range(21)];block+=1
    checks['F1']=gate(dict(seeds),dict(calls[-1]))
    orders=[]
    for row in ((0,1,2,3,4),(4,3,2,1,0)):orders.extend(row[i:]+row[:i] for i in range(5))
    pairs=[]
    for r in range(20):
        row={}
        for index in orders[r%10]:row[LABELS[index]]=one(LABELS[index],802000+r,'paired')
        pairs.append(row)
    checks['paired']=gate(dict(seeds),dict(calls[-1]))
    med={key:statistics.median(values) for key,values in raw.items()};contrasts={}
    for before,after in (('generic_a','guarded_a'),('generic_b','guarded_b'),('incumbent','guarded_a'),('incumbent','guarded_b')):
        contrasts[before+'_minus_'+after]=dict(F1_gain_us=min(med[before+'_before'],med[before+'_after'])-med[after],paired_median_gain_us=statistics.median(r[before]-r[after] for r in pairs),wins=sum(r[before]>r[after] for r in pairs))
    return dict(raw_F1=raw,raw_paired=pairs,all_calls=calls,medians_F1=med,contrasts=contrasts,terminal_checks=checks,
                scope='283calls:15warm/168F1/100paired,20rounds/10balancedorders; every signed result retained, no gain cutoff')

def storage(tensor):
    raw=torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(tensor.untyped_storage(),0,(tensor.untyped_storage().nbytes(),),(1,));baseline=raw.clone()
    identity=(str(tensor.device),tensor.untyped_storage().data_ptr(),tensor.untyped_storage().nbytes(),tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))
    def check():
        assert (str(tensor.device),tensor.untyped_storage().data_ptr(),tensor.untyped_storage().nbytes(),tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))==identity
        assert torch.equal(raw,baseline),'dormant fullstorage changed'
    return check,identity[:3]

def counters(state,prefix='',seen=None):
    seen=set() if seen is None else seen
    if id(state) in seen:return []
    seen.add(id(state));out=[]
    if isinstance(state,dict):
        for name,value in state.items():
            if torch.is_tensor(value) and ('cnt' in str(name).lower() or 'counter' in str(name).lower()):out.append((prefix+'/'+str(name),value))
            elif isinstance(value,(dict,list,tuple)):out.extend(counters(value,prefix+'/'+str(name),seen))
    elif isinstance(state,(list,tuple)):
        for i,value in enumerate(state):out.extend(counters(value,prefix+'/'+str(i),seen))
    return out

def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn(x,out)
    torch.cuda.synchronize();return graph

def validate_hardware(record):
    assert re.search(r'\bH(?:100|200)\b',record['name']),('UNSUPPORTED_GPU_NAME',record)
    assert tuple(record['capability'])==(9,0),('UNSUPPORTED_COMPUTE_CAPABILITY',record)
    assert isinstance(record['SM_count'],int) and record['SM_count']>0,('INVALID_SM_COUNT',record)
    assert record['UUID'] and record['UUID'].lower() not in ('none','unknown'),('MISSING_GPU_UUID',record)
    return record

def main():
    global torch
    import torch
    from torch.utils.cpp_extension import load_inline
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);args=ap.parse_args();args.json.parent.mkdir(parents=True,exist_ok=True)
    frozen=verify_files();sys.path.insert(0,str(HERE));import preflight as gate;import physical_proof;from csr_proof import validate_rows
    result=dict(status='STARTED_NO_GAIN_ADMISSION',candidate_source_sha256=frozen['source_sha256'],actual_empty_prevalence={},runtime={},contexts=[])
    def save():args.json.write_text(json.dumps(result,indent=2)+'\n')
    graphs={}
    try:
        device=torch.cuda.current_device();properties=torch.cuda.get_device_properties(device)
        result['hardware']=dict(device_index=device,name=properties.name,UUID=str(getattr(properties,'uuid','')),capability=list(torch.cuda.get_device_capability(device)),SM_count=int(properties.multi_processor_count),total_memory_bytes=int(properties.total_memory))
        save();validate_hardware(result['hardware'])
        cu=source();assert cu==(HERE/'lab_source.cu').read_text()
        ext=load_inline('pw42_narrow_'+hashlib.sha256(cu.encode()).hexdigest()[:16],cpp_sources='int64_t gvrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t); int64_t gvregs(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t); int64_t gvocc(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t); int64_t eeinfo(int64_t,int64_t); int64_t eeoccupancy(int64_t);',cuda_sources=cu,functions=['gvrun','gvregs','gvocc','eeinfo','eeoccupancy'],extra_cuda_cflags=gate.FLAGS,verbose=True)
        dump=shutil.which('cuobjdump') or '/usr/local/cuda/bin/cuobjdump';binary=Path(ext.__file__).resolve();sass=subprocess.check_output([dump,'--dump-sass',str(binary)],text=True,timeout=30)
        artifact=args.json.with_suffix('.jit.sass');artifact.write_text(sass);actual=gv_sass(sass)
        physical=actual_binary_gate(sass)
        result['actual_JIT']=dict(path=str(artifact),sha256=hashlib.sha256(sass.encode()).hexdigest(),physical=physical)
        for i,label in enumerate(LABELS[1:]):
            attrs=dict(regs=int(ext.eeinfo(i,1)),local=int(ext.eeinfo(i,2)),max_threads=int(ext.eeinfo(i,3)),static_shared=int(ext.eeinfo(i,4)))
            threads=1024 if i<2 else 512
            assert attrs['regs']==64 and attrs['local']==0 and attrs['max_threads']>=threads
            occupancy=int(ext.eeoccupancy(i));sms=result['hardware']['SM_count']
            assert occupancy>0 and occupancy%sms==0 and occupancy//sms>=(1 if i<2 else 2),('PREPREPARE_OCCUPANCY',label,occupancy,sms)
            attrs['active_CTAs_per_SM']=occupancy//sms;attrs['occupancy_before_prepare']=True
            result['runtime'][label]=attrs
        save() # Actual JIT/resources before anycandidate math.
        B=load(HERE.parent/'pw38/submission.py','pw42_frozen_PW38');sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        weights=reference.generate_weights(reference.WEIGHTS_SEED);weight_before={key:w.clone() for key,w in weights.items()};state=B.prepare(weights);w=weights['qkv_gate']
        assert state.get('sel',{}).get(('qkv_gate',1)) is None and state.get('sel',{}).get(('qkv_gate',8)) is None,'UNSUPPORTED_NON_GV_INCUMBENT'
        selected={m:state.get('gv',{}).get(('qkv_gate',m)) for m in (1,8)}
        assert all(p is not None and tuple(B.GVI[p[0]]) in BASES for p in selected.values()),'ACTUAL_POLICY_OUTSIDE_REVIEWED_TWO_GEOMETRIES'
        old_call=B._gvcall;calls=[0]
        def guarded(st,p,x,out):
            assert p is selected[x.shape[0]],'selected incumbent plan changed'
            code=old_call(st,p,x,out);assert code==0,'INCUMBENT_FALLBACK';calls[0]+=1;return code
        B._gvcall=guarded
        def incumbent(x,out):
            before=calls[0];value=B.kernel_fn({'x':x,'config':{'layer':'qkv_gate','m':x.shape[0],'n':2304,'k':8192}},state);assert calls[0]==before+1,'INCUMBENT_SELECTED_GV_NOT_USED';holder[0]=value;return value
        base_ext=B._gvext();old_sass=subprocess.check_output([dump,'--dump-sass',str(Path(base_ext.__file__).resolve())],text=True,timeout=30)
        old_artifact=args.json.with_suffix('.incumbent.jit.sass');old_artifact.write_text(old_sass);old_tables=gate.tables(gv_sass(old_sass));canonical=gate.tables((HERE/'control-sass.txt').read_text())
        for m,p in selected.items():assert old_tables[('generic',tuple(B.GVI[p[0]]))]==canonical[('generic',tuple(B.GVI[p[0]]))]
        result['incumbent_selection']={str(m):list(B.GVI[p[0]]) for m,p in selected.items()}
        # ActualCUDA weight codec and escape lists, not aCPU-fixture histogram.
        fragments=state['qkv_gate'][0];packed=B._c12(fragments,1);assert packed is not None
        W,table,esc6=packed;assert torch.equal(B._unc12(W,table,esc6,1),fragments.view(torch.int16).int()&65535)
        esc7=torch.cat((esc6,torch.zeros_like(esc6[:,:1])),1);lists=[]
        for j,base in enumerate(BASES):
            escaped=B._gvesc(esc7,144,base[3:9]);assert escaped is not None
            stats=validate_rows(escaped.reshape(-1,32).detach().cpu().tolist());stats['empty_fraction']=stats['empty_groups']/stats['groups'];result['actual_empty_prevalence'][str(j)]=stats;lists.append(escaped)
        save() # First discriminator retained before candidate math/timing.
        if all(record['empty_groups']==0 for record in result['actual_empty_prevalence'].values()):
            result['status']='VALID_METADATA_DIAGNOSTIC_EMPTY_PATH_UNUSED_NO_TIMING';save();return
        fixed_ws=torch.full((64,),17.25,dtype=torch.float32,device='cuda');fixed_cnt=torch.full((64,),12345,dtype=torch.int32,device='cuda')
        immutable=[(v,weight_before[key]) for key,v in weights.items()]+[(W,W.clone())]+[(e,e.clone()) for e in lists]+[(state['gvws'],state['gvws'].clone()),(fixed_ws,fixed_ws.clone()),(fixed_cnt,fixed_cnt.clone())]
        for p in selected.values():immutable.extend([(p[1],p[1].clone()),(p[2],p[2].clone())])
        checks=[];ids=set()
        for name,tensor in counters(state):
            check,identity=storage(tensor)
            if identity not in ids:checks.append(check);ids.add(identity)
        def integrity():
            assert all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in immutable),'W/pack/CSR/workspace/counter changed'
            for check in checks:check()
        funcs={}
        for i,label in enumerate(LABELS[1:]):
            def fn(x,out,i=i):
                code=int(ext.gvrun(i,W,W,x,out,lists[i//2],fixed_ws,fixed_cnt,table,table));assert code==0,(i,code);return 0
            funcs[label]=fn
        sample=torch.empty((1,8192),dtype=torch.bfloat16,device='cuda');sample_out=torch.empty((1,2304),dtype=torch.bfloat16,device='cuda');sms=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        for i,label in enumerate(LABELS[1:]):
            occupancy=int(ext.gvocc(i,W,W,sample,sample_out,lists[i//2],fixed_ws,fixed_cnt,table,table));assert occupancy%sms==0 and occupancy//sms>=(1 if i<2 else 2)
            assert result['runtime'][label]['active_CTAs_per_SM']==occupancy//sms,('PREPARE_OCCUPANCY_DRIFT',label)
        # Complete fullK basis at8rows, independent FP64, then odd-M task probes.
        basis=torch.eye(8192,dtype=torch.bfloat16,device='cuda');result['validation']={}
        for label,fn in funcs.items():
            out=torch.empty((8,2304),dtype=torch.bfloat16,device='cuda')
            for start in range(0,8192,8):fn(basis[start:start+8],out);assert torch.equal(out.float(),w[:,start:start+8].t().float())
            rows=[]
            for m in (1,2,3,5,7,8):
                for dist in ('normal','outliers'):
                    data=reference.generate_input(m,'qkv_gate',830003,dist=dist);before=data['x'].clone();output=torch.full((m,2304),float('nan'),dtype=torch.bfloat16,device='cuda');fn(data['x'],output)
                    ok,message=reference.check_implementation(data,output,{'qkv_gate':w});assert ok and torch.equal(data['x'],before),(label,m,message)
                    fp64=data['x'].double()@w.double().t();rel=float((output.double()-fp64).norm()/fp64.norm());assert rel<4e-3
                    rows.append(dict(m=m,dist=dist,oracle=message,FP64_relative=rel));integrity()
            result['validation'][label]=dict(all8192_basis='EXACT',odd_M_task_FP64=rows);save()
        del basis
        for m in (1,8):
            data=reference.generate_input(m,'qkv_gate',830011);x=data['x'];config=data['config']
            buffers=[torch.empty((m,2304),dtype=torch.bfloat16,device='cuda') for _ in range(4)]
            for role in (0,1):
                context=dict(M=m,output_role_swap=role);result['contexts'].append(context);holder=[None]
                indices=(0,1,2,3) if role==0 else (1,0,3,2)
                outputs={label:buffers[indices[i]] for i,label in enumerate(LABELS[1:])}
                graphs['incumbent']=capture(incumbent,x,outputs['generic_a']);outputs={'incumbent':holder[0],**outputs}
                for label,fn in funcs.items():graphs[label]=capture(fn,x,outputs[label])
                assert len({out.data_ptr() for out in outputs.values()})==5
                seeds=[800000]+list(range(801000,801168))+list(range(802000,802020));fixtures={seed:reference.generate_input(m,'qkv_gate',seed)['x'] for seed in seeds}
                def checker(fresh,out):return reference.check_implementation({'x':fresh,'config':config},out,{'qkv_gate':w})
                def strict(last_seeds,last):return terminal(outputs,x,fixtures,last_seeds,last,checker,integrity)
                context['timing_calls']=[]
                context['timing']=measure(graphs,x,fixtures,strict,calls=context['timing_calls']);save()
                probe=torch.zeros_like(x);rr=torch.arange(m,device=x.device);probe[rr,(37*rr+19)%8192]=8.
                context['graph_mutations']=[]
                for name,fresh in [('zero',torch.zeros_like(x)),('normal',reference.generate_input(m,'qkv_gate',840001)['x']),('outliers',reference.generate_input(m,'qkv_gate',840003,dist='outliers')['x']),('allrows',probe)]:
                    x.copy_(fresh)
                    for label,g in graphs.items():
                        outputs[label].fill_(float('nan'));before=x.clone();g.replay();torch.cuda.synchronize();assert torch.equal(x,before) and bool(torch.isfinite(outputs[label]).all())
                        if name=='zero':assert not int(torch.count_nonzero(outputs[label]));message='complete exact zero'
                        else:ok,message=checker(x,outputs[label]);assert ok,(label,name,message)
                        integrity();context['graph_mutations'].append(dict(label=label,input=name,oracle=message,NaN_overwrite=True))
                from torch.profiler import profile,ProfilerActivity
                context['profiles']={}
                for label,g in graphs.items():
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
                    events=[e.name() for e in prof.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA];assert len(events)==1,(label,events);context['profiles'][label]=events
                    if label!='incumbent':assert ('gv_empty_guard' in events[0])==label.startswith('guarded')
                integrity();torch.cuda.synchronize()
                for g in graphs.values():g.reset()
                graphs.clear();context['cleanup']='DRAINED_RESET_ALL5';save()
        result['status']='VALID_NARROW_DISCRIMINATOR_NO_CUMULATIVE_ADMISSION';result['next']='Compare both output roles, actual empty fractions and static code cost; retain every tiny signed result; full actualPW42 prepare/30shape cumulative test is separate.';save()
    except Exception as error:result.update(status='INVALID_SETUP_OR_CORRECTNESS',error=repr(error));save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for g in graphs.values():g.reset()
            graphs.clear()
        result['graph_cleanup']='DRAINED_RESET_ALL_CREATED_GRAPHS';save()
if __name__=='__main__':main()
