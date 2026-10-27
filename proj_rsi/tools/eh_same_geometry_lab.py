"""Root-only fixed-geometry EH research; import creates no CUDA context or extension."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'proj_rsi/candidates/eh_same_geometry_trial'
sys.path.insert(0,str(HERE))
import support as S
LABELS=S.LABELS


def terminal_gate(outputs,x,config,w,fixtures,last,final,immutable,counters):
    # Read only: no replay, poison, mutation or repair before this gate succeeds.
    assert set(last)==set(outputs)==set(LABELS)
    assert len({o.data_ptr() for o in outputs.values()})==4,'output alias'
    assert torch.equal(x,fixtures[final['input_seed']]),'current X changed'
    checks=[]
    for label,out in outputs.items():
        fresh=fixtures[last[label]]
        assert bool(torch.isfinite(out).all()),(label,'nonfinite actual timed output')
        ok,msg=reference.check_implementation({'x':fresh,'config':config},out,{config['layer']:w})
        assert ok,(label,'actual terminal output',last[label],msg)
        checks.append({'label':label,'input_seed':last[label],'actual_terminal_output_before_any_replay':'PASS','oracle':msg})
    assert all(torch.equal(a,b) for a,b in immutable),'immutable W/packed/escape changed'
    assert all(not int(c.abs().sum()) for c in counters.values()),'counter not reset'
    return checks


def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn(x,out)
    return graph


def measure(graphs,x,fixtures,strict):
    flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda')
    calls=[];last={}
    def one(label,seed,mode,warmup=False):
        x.copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        us=a.elapsed_time(b)*1000
        calls.append(dict(label=label,input_seed=seed,mode=mode,warmup=warmup,us=us))
        if not warmup:last[label]=seed
        return us
    for label in LABELS:
        for _ in range(5):one(label,239999,'warmup',True)
    raw={};block=0
    for parent,new in [('parent2','eh2'),('parent3','eh3')]:
        for key,label in [(parent+'_before',parent),(new,new),(parent+'_after',parent)]:
            raw[key]=[one(label,240000+21*block+i,'F1') for i in range(21)];block+=1
    checks={'F1':strict(fixtures,dict(last),dict(calls[-1]))}
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
    paired=[]
    for repetition in range(32):
        row={};seed=250000+repetition
        for index in orders[repetition%8]:row[LABELS[index]]=one(LABELS[index],seed,'paired')
        paired.append(row)
    checks['paired']=strict(fixtures,dict(last),dict(calls[-1]))
    med={k:statistics.median(v) for k,v in raw.items()};comparisons={}
    for parent,new in [('parent2','eh2'),('parent3','eh3')]:
        gains=[r[parent]-r[new] for r in paired]
        comparisons[new]={'control':parent,'FL_control':S.FLAGS[parent],'FL_EH':S.FLAGS[new],
            'F1_gain_us':min(med[parent+'_before'],med[parent+'_after'])-med[new],
            'paired_median_gain_us':statistics.median(gains),'paired_all_gains_us':gains,
            'paired_wins':sum(g>0 for g in gains),'paired_median_ratio':statistics.median(r[parent]/r[new] for r in paired),
            'gain_threshold_us':0,'serving_incumbent_win_claim':False}
    assert len(calls)==274
    return dict(F1_samples_us=raw,F1_medians_us=med,paired_samples_us=paired,comparisons=comparisons,
                strict_terminal_checks=checks,all_raw_calls=calls,terminal_input_seed_per_policy=last,
                policy='21 F1 in matched control/EH/control brackets;32 balanced4arm fresh-input pairs; report every sample; no minimum retained gain')


def validate(fn,w,layer,n,ks,P):
    gen=torch.Generator(device='cuda').manual_seed(6715)
    ok,msg=P._rqval(fn,w,128,n,gen,P.RQTOL[ks]);assert ok,msg
    records=[dict(all_K_basis='EXACT',random_FP64='PASS',tolerance=P.RQTOL[ks],detail=msg)]
    for dist in ['normal','outliers']:
        data=reference.generate_input(128,layer,7289,dist=dist);before=data['x'].clone()
        out=torch.full((128,n),float('nan'),dtype=torch.bfloat16,device='cuda');fn(data['x'],out);torch.cuda.synchronize()
        ok,msg=reference.check_implementation(data,out,{layer:w});assert ok,msg
        assert torch.equal(before,data['x']);records.append(dict(task_dist=dist,oracle=msg))
    return records


def mutation_checks(graphs,outputs,x,config,w,immutable,counters):
    original=x.clone();probes=torch.zeros_like(x);rows=torch.arange(128,device='cuda')
    probes[rows,(37*rows+19)%x.shape[1]]=(8*(2*(rows%2)-1)).to(torch.bfloat16)
    records=[]
    for kind,fresh in [('original',original),('zero',torch.zeros_like(x)),('fresh_normal',reference.generate_input(128,config['layer'],9911)['x']),('fresh_outliers',reference.generate_input(128,config['layer'],9913,dist='outliers')['x']),('probes',probes)]:
        x.copy_(fresh)
        for label,g in graphs.items():
            outputs[label].fill_(float('nan'));g.replay();torch.cuda.synchronize()
            assert torch.equal(x,fresh) and bool(torch.isfinite(outputs[label]).all())
            if kind=='zero':assert not int(torch.count_nonzero(outputs[label]));msg='exact zero'
            else:
                ok,msg=reference.check_implementation({'x':x,'config':config},outputs[label],{config['layer']:w});assert ok,msg
            assert all(torch.equal(a,b) for a,b in immutable)
            assert all(not int(c.abs().sum()) for c in counters.values())
            records.append(dict(label=label,input=kind,NaN_poison_overwritten=True,oracle=msg))
    x.copy_(original)
    return records


def main():
    global torch,reference
    parser=argparse.ArgumentParser();parser.add_argument('--json',type=Path,required=True);args=parser.parse_args()
    S.verify_frozen();S.verify_lab()
    import torch
    from torch.utils.cpp_extension import load_inline
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    result={'status':'STARTED_NO_TIMING_ADMISSION','scope':'EH_FIXED_GEOMETRY_MECHANISM_ONLY_NOT_SELECTED_SERVING_POLICY',
            'source_pins':S.PINS,'source_roles':S.SOURCES,'shapes':{},'JIT':{},'attrs':{},
            'full_stack_or_actual_selected_incumbent_win_claim':False}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():args.json.write_text(json.dumps(result,indent=2)+'\n')
    exts={}
    try:
        for role in ['parent','eh']:
            cu=S.cuda_source(role);tag='eh_same_geometry_'+role+'_'+hashlib.sha256(cu.encode()).hexdigest()[:12]
            ext=load_inline(tag,cpp_sources='int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t); int64_t labattrs(int64_t,int64_t);',cuda_sources=cu,functions=['rqrun','labattrs'],extra_cuda_cflags=S.CUDA_FLAGS,verbose=True)
            exts[role]=ext;binary=Path(ext.__file__).resolve()
            sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True,timeout=20)
            path=args.json.with_suffix('.'+role+'.jit.sass');path.write_text(sass)
            result['JIT'][role]={'module_sha256':S.sha(binary),'artifact':str(path),'sha256':S.sha(path),'gate':S.binary_gate(role,sass)}
            for index,label in enumerate(['parent2','parent3'] if role=='parent' else ['eh2','eh3']):
                attr={key:int(ext.labattrs(index,code)) for key,code in [('regs',1),('local',2),('shared',3),('occupancy',4),('max_threads',5)]}
                assert attr==dict(regs=168,local=0,shared=144520,occupancy=1,max_threads=384),(label,attr)
                attr.update(threads=384,static_rounded_register_pool=64512,dynamic_register_pool=64512)
                result['attrs'][label]=attr
        result['all_candidate_JIT_and_resources_before_first_math']='PASS';save()
    except Exception as error:
        result.update(status='INVALID_JIT_OR_RESOURCE_BEFORE_MATH',error=repr(error),graph_cleanup='NO_GRAPHS_CREATED');save();raise
    P=S.load_module('eh');weights=reference.generate_weights(reference.WEIGHTS_SEED)
    weight_copies={name:w.clone() for name,w in weights.items()}
    for layer,(n,k,ks) in S.SHAPES.items():
        graphs={};record={'geometry':dict(m=128,n=n,k=k,KS=ks,NT=128,ST=5,GS=1,PRB=0,WM=0,INS=0)}
        result['shapes'][layer]=record
        try:
            w=weights[layer];assert tuple(w.shape)==(n,k)
            packed=P._rqpack(w);assert packed is not None
            lay,tab,esc6=packed
            assert torch.equal(P._rqunpack(*packed,n,k),w.view(torch.int16).int()&65535)
            esc,_=P._rqesc(esc6,lay.shape[0],lay.shape[1],ks,'cuda')
            counters={label:torch.zeros(2*lay.shape[0],dtype=torch.int32,device='cuda') for label in LABELS}
            empty=torch.empty(0,dtype=torch.int64,device='cuda');sink=torch.zeros(4,dtype=torch.int32,device='cuda')
            immutable=[(a,weight_copies[name]) for name,a in weights.items()]+[(lay,lay.clone()),(esc,esc.clone())]
            funcs={}
            for label in LABELS:
                ext=exts['parent' if label.startswith('parent') else 'eh'];index=0 if label.endswith('2') else 1
                def fn(x,out,ext=ext,index=index,label=label):
                    code=int(ext.rqrun(index,lay,x,out,esc,counters[label],ks,tab,0,empty,sink,0))
                    if code:raise RuntimeError((layer,label,'launch',code))
                    return 0
                funcs[label]=fn
            record['validation']={label:validate(fn,w,layer,n,ks,P) for label,fn in funcs.items()}
            assert all(not int(c.abs().sum()) for c in counters.values())
            data=reference.generate_input(128,layer,5237);x=data['x'];outputs={label:torch.empty((128,n),dtype=torch.bfloat16,device='cuda') for label in LABELS}
            for label in LABELS:graphs[label]=capture(funcs[label],x,outputs[label])
            seeds=[239999]+list(range(240000,240126))+list(range(250000,250032))
            fixtures={seed:reference.generate_input(128,layer,seed)['x'] for seed in seeds}
            def strict(fixtures,last,final):return terminal_gate(outputs,x,data['config'],w,fixtures,last,final,immutable,counters)
            record['timing']=measure(graphs,x,fixtures,strict);save()
            record['mutations']=mutation_checks(graphs,outputs,x,data['config'],w,immutable,counters)
            # Only after both immediate timed-output gates and mutation validation.
            from torch.profiler import profile,ProfilerActivity
            record['graph_nodes']={}
            for label,g in graphs.items():
                with profile(activities=[ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
                events=[e.name() for e in prof.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]
                assert len(events)==1 and 'rq<128, 5, 1, 0, 0, 0, '+str(S.FLAGS[label])+'>' in events[0],(label,events)
                record['graph_nodes'][label]=events
            assert all(torch.equal(a,b) for a,b in immutable)
            assert all(not int(c.abs().sum()) for c in counters.values())
            record['status']='VALID_FIXED_GEOMETRY_RESULT_NO_PRODUCTION_ADMISSION';save()
        except Exception as error:
            record.update(status='INVALID_RUNTIME_OR_CORRECTNESS',error=repr(error));result['status']='INVALID_NO_PERFORMANCE_ADMISSION';save();raise
        finally:
            if graphs:
                torch.cuda.synchronize()
                for g in graphs.values():g.reset()
                graphs.clear()
            record['graph_cleanup']='DRAINED_RESET_ALL_FOUR';save()
    result['status']='VALID_EH_SAME_GEOMETRY_MECHANISM_ONLY'
    result['next']='Retain every signed tiny delta; selected prepared serving policy and cumulative full stack need separate validated comparisons.'
    save();print('EH_SAME_GEOMETRY_COMPLETE '+json.dumps({layer:r['timing']['comparisons'] for layer,r in result['shapes'].items()}),flush=True)

if __name__=='__main__':main()
