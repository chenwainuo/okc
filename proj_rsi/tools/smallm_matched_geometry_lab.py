"""Root-only matched cursor mechanism plus actual policy references; CPU import safe."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'proj_rsi/candidates/smallm_matched_geometry_trial'
spec=importlib.util.spec_from_file_location('matched_cursor_support',HERE/'support.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
LABELS=S.LABELS


def terminal_gate(outputs,x,config,w,fixtures,last,final,immutable,invariants):
    # Read-only gate: no replay, input/output repair, poison or mutation here.
    assert set(outputs)==set(last)==set(LABELS)
    assert len({out.data_ptr() for out in outputs.values()})==len(LABELS),'output alias'
    assert torch.equal(x,fixtures[final['input_seed']]),'current X changed'
    records=[]
    for label,out in outputs.items():
        assert bool(torch.isfinite(out).all()),(label,'nonfinite terminal output')
        fresh=fixtures[last[label]]
        ok,msg=reference.check_implementation({'x':fresh,'config':config},out,{config['layer']:w})
        assert ok,(label,last[label],'actual final timed output',msg)
        records.append({'label':label,'seed':last[label],'actual_final_output_before_any_replay':'PASS','oracle':msg})
    assert all(torch.equal(t,baseline) for _,t,baseline in immutable),'immutable state changed'
    invariants()
    return records


def measure(graphs,x,fixtures,strict,journal,shape):
    flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda')
    calls=[];last={};f1=collections.defaultdict(list);paired=collections.defaultdict(dict);terminals={}
    for mode,label,seed,key in S.schedule():
        if mode=='terminal':
            terminals[label]=strict(dict(last),dict(calls[-1]));continue
        x.copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        us=a.elapsed_time(b)*1000
        assert us>0 and us<float('inf')
        row={'shape':shape,'mode':mode,'label':label,'input_seed':seed,'block':key,'us':us}
        calls.append(row);journal.write(json.dumps(row)+'\n');journal.flush()
        if mode!='warmup':last[label]=seed
        if mode=='F1':f1[key].append(us)
        if mode=='paired':paired[seed][label]=us
    assert len(calls)==462 and len(paired)==28
    med={key:statistics.median(values) for key,values in f1.items()};comparisons={}
    for name,(parent,new) in S.COMPARISONS.items():
        gains=[row[parent]-row[new] for row in paired.values()]
        comparisons[name]={'control':parent,'candidate':new,'F1_conservative_gain_us':min(med[parent+'_before'],med[parent+'_after'])-med[new],
            'paired_median_gain_us':statistics.median(gains),'paired_median_control_over_candidate_ratio':statistics.median(row[parent]/row[new] for row in paired.values()),
            'paired_all_signed_gains_us':gains,'paired_win_tie_loss':[sum(g>0 for g in gains),sum(g==0 for g in gains),sum(g<0 for g in gains)],'gain_threshold_us':0}
    return {'raw_calls':calls,'F1_samples_us':dict(f1),'F1_medians_us':med,'paired_samples_us':list(paired.values()),'comparisons':comparisons,'strict_terminal_checks':terminals}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--root-review-token')
    args=ap.parse_args()
    if args.cpu_check:
        proof=S.module(HERE/'cpu_proof.py','matched_cursor_cpu_proof').proof()
        print(json.dumps(proof,indent=2));return
    manifest=S.verify();assert args.output and args.root_review_token
    review=ROOT/args.root_review_token;assert review.is_file()
    args.output.mkdir(parents=True,exist_ok=False)
    global torch,reference
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    M=S.driver();M.verify()
    result={'status':'RUNNING_NO_PERFORMANCE_ADMISSION','scope':list(S.SHAPES),'source_version':manifest['version'],
            'review_sha256':S.sha(review),'source_pins':M.PINS,'phases':[],'JIT':{},'prepared_policy':{},'shapes':{},
            'mechanism_comparisons':'matched ST5/ST6 same KS parent/cursor; actual prepared policy arms are separately labeled',
            'excluded_coverage':'qkv16/o32 were GV in prior attempt; this mechanism diagnostic claims no improvement there','started':time.time()}
    def save(): (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    def phase(name):result['phases'].append({'phase':name,'time':time.time()});save();print(name,flush=True)
    graphs={};modules={};states={}
    try:
        weights=reference.generate_weights(reference.WEIGHTS_SEED);registry={}
        for role in ('pw36','rs32','increment'):
            phase('actual_JIT_before_math_'+role);mod=modules[role]=M.load(role)
            result['JIT'][role]=M.jit_gate(mod,role,args.output,registry)
            binary=Path(mod._rqmod2[0].__file__);target=args.output/(role+'.actual.so');shutil.copyfile(binary,target)
            assert S.sha(target)==result['JIT'][role]['so_sha256'];result['JIT'][role]['archived_so']=target.name
            mod._T0=time.time();phase('actual_full_prepare_'+role);states[role]=mod.prepare(weights);torch.cuda.synchronize()
        result['counter_source_bindings']=M.contract_source_check()
        immutable=[];counters=[];seen=set()
        for role,state in states.items():
            for name,t in M.tensors(state):
                if not t.numel():continue
                ident=M.tensor_identity(t)
                if 'cnt' in name:counters.append((role,name,t,M.counter_snapshot(role+name,t)));continue
                view=(ident,t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                if view in seen:continue
                seen.add(view)
                if 'ws' not in name and 'sink' not in name:immutable.append((role+name,t,t.clone()))
            original=modules[role]._rqcall
            def guarded(st,p,x,out,original=original,role=role):
                code=original(st,p,x,out);assert code==0,('HOLD actual RQ fallback',role,code);return code
            modules[role]._rqcall=guarded
        result['counter_prepare_baseline']=[{k:v for k,v in snap.items() if k!='values'} for _,_,_,snap in counters]
        for shape,(layer,m,n,k,ks) in S.SHAPES.items():
            graphs={};entry=result['shapes'][shape]={'config':{'layer':layer,'m':m,'n':n,'k':k},'fixed_KS':ks,'picks':{},'validation':{}}
            cfg=entry['config'];active={}
            for role,state in states.items():
                path,counter=M.select_counter_contract(modules[role],state,layer,m,n)
                active[role]={M.tensor_identity(counter)} if counter is not None else set()
                rp=state.get('rq',{}).get((layer,m));entry['picks'][role]={'counter_path':path,'rq':{key:value for key,value in (rp or {}).items() if isinstance(value,(int,float,str))},'rq_name':modules[role].RQPROD2[rp['i']] if rp else None,'sel':str(state.get('sel',{}).get((layer,modules[role]._bk(m)))),'gv':str(state.get('gv',{}).get((layer,modules[role]._bk(m))))}
            P=modules['rs32'];w=weights[layer];packed=P._rqpack(w);assert packed is not None
            lay,tab,esc6=packed;assert torch.equal(P._rqunpack(*packed,n,k),w.view(torch.int16).int()&65535)
            esc,_=P._rqesc(esc6,lay.shape[0],lay.shape[1],ks,'cuda')
            fixedcnt={label:torch.zeros(4*lay.shape[0],dtype=torch.int32,device='cuda') for label in LABELS if not label.startswith('actual_')}
            local_immutable=immutable+[('fixed W',w,w.clone()),('fixed packed',lay,lay.clone()),('fixed escape',esc,esc.clone())]
            def invariants():
                assert all(torch.equal(t,baseline) for _,t,baseline in local_immutable),'W/pack/escape/immutable changed'
                failed=[]
                for role,name,t,initial in counters:
                    final=M.counter_snapshot(role+name,t);contract=M.counter_storage_contract(t,active[role])
                    if not M.counter_verdict(contract,initial['values'],final['values']):failed.append({'name':role+name,'contract':contract,'initial':initial,'final':final})
                for label,t in fixedcnt.items():
                    if int(torch.count_nonzero(t)):failed.append({'name':label,'contract':'reset0','final':M.counter_snapshot(label,t)})
                if failed:entry['counter_failure']=failed;save();raise AssertionError('counter contract failed')
            invariants();funcs={};entry['fixed_attrs']={}
            for label in fixedcnt:
                role='increment' if label.startswith('cursor') else 'rs32';stage=int(label[-1]);mod=modules[role]
                name='t32s%dhze'%stage;i=mod.RQPROD2.index(name);ext=mod._rqmod2[0]
                rp={'i':i,'lay':lay,'esc':esc,'cnt':fixedcnt[label],'KS':ks,'tab':tab}
                xprobe=torch.zeros((m,k),dtype=torch.bfloat16,device='cuda');oprobe=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
                attrs=M.selected_attributes(ext,rp,xprobe,oprobe,states[role]);assert attrs==[168,0,83080 if stage==5 else 99488]
                entry['fixed_attrs'][label]={'name':name,'KS':ks,'regs_local_shared':attrs,'generated_source_sha256':result['JIT'][role]['source_sha256']}
                def fn(x,out,ext=ext,rp=rp,role=role):
                    code=ext.rqrun(rp['i'],lay,x,out,esc,rp['cnt'],ks,tab,0,states[role]['rq_e'],states[role]['rq_sink'],0)
                    assert int(code)==0,('fixed launch declined',role,int(code));return 0
                funcs[label]=fn
            for label,role in [('actual_pw36','pw36'),('actual_rs32','rs32'),('actual_cursor','increment')]:
                def fn(x,out,role=role):out.copy_(modules[role].kernel_fn({'x':x,'config':cfg},states[role]));return 0
                funcs[label]=fn
            for label,fn in funcs.items():
                phase(shape+'_full_K_basis_FP64_'+label)
                ok,msg=P._rqval(fn,w,m,n,torch.Generator(device='cuda').manual_seed(79001),4e-3);assert ok,(label,msg)
                invariants();entry['validation'][label]={'full_K_basis_FP64':msg}
            seeds={seed for mode,label,seed,key in S.schedule() if seed is not None}
            fixtures={seed:reference.generate_input(m,layer,seed)['x'] for seed in seeds}
            x=fixtures[239999].clone();outputs={}
            for label in LABELS:
                if label.startswith('actual_'):
                    role={'actual_pw36':'pw36','actual_rs32':'rs32','actual_cursor':'increment'}[label]
                    for _ in range(2):modules[role].kernel_fn({'x':x,'config':cfg},states[role])
                    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):out=modules[role].kernel_fn({'x':x,'config':cfg},states[role])
                else:
                    out=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
                    for _ in range(2):funcs[label](x,out)
                    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):funcs[label](x,out)
                outputs[label]=out;graphs[label]=g
            def strict(last,final):return terminal_gate(outputs,x,cfg,w,fixtures,last,final,local_immutable,invariants)
            phase(shape+'_timing')
            with (args.output/'raw.jsonl').open('a') as journal:entry['timing']=measure(graphs,x,fixtures,strict,journal,shape)
            save();entry['mutations']=[]
            probes=torch.zeros_like(x);r=torch.arange(m,device='cuda');probes[r,(r*37+19)%k]=8
            mutations=[('normal',fixtures[239999]),('zero',torch.zeros_like(x)),('fresh_normal',reference.generate_input(m,layer,279001)['x']),('outliers',reference.generate_input(m,layer,279002,dist='outliers')['x']),('probes',probes)]
            for kind,fresh in mutations:
                x.copy_(fresh)
                for label in LABELS:
                    outputs[label].fill_(float('nan'))
                    if label.startswith('actual_'):
                        role={'actual_pw36':'pw36','actual_rs32':'rs32','actual_cursor':'increment'}[label]
                        for name,t in M.tensors(states[role]):
                            if name in ('/ws','/gvws','/rs_ws') and t.is_floating_point():t.fill_(float('nan'))
                    graphs[label].replay();torch.cuda.synchronize()
                    assert bool(torch.isfinite(outputs[label]).all())
                    ok,msg=reference.check_implementation({'x':fresh,'config':cfg},outputs[label],{layer:w});assert ok,(kind,label,msg)
                    assert torch.equal(x,fresh)
                    if kind=='zero':assert not int(torch.count_nonzero(outputs[label]))
                    invariants();entry['mutations'].append({'label':label,'kind':kind,'PASS':True})
            entry['graph_nodes']={}
            from torch.profiler import profile,ProfilerActivity
            for label,g in graphs.items():
                with profile(activities=[ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
                events=[event.name() for event in prof.profiler.kineto_results.events() if event.device_type()==torch.autograd.DeviceType.CUDA]
                assert events,(label,'no actual graph device node')
                if not label.startswith('actual_'):assert len(events)==1 and 'rq<' in events[0],(label,events)
                entry['graph_nodes'][label]=events
            invariants();torch.cuda.synchronize()
            for g in graphs.values():g.reset()
            graphs.clear();entry['graph_cleanup']='DRAINED_RESET_ALL_SEVEN';entry['status']='VALID_FIXED_AND_ACTUAL_POLICY_REFERENCES';save()
        result['status']='VALID_MATCHED_CURSOR_MECHANISM_AND_ACTUAL_POLICY_REFERENCES_NO_OFFICIAL_ADMISSION';phase('complete')
    except BaseException:
        result['status']='INVALID_NO_PERFORMANCE_ADMISSION';result['failure']=traceback.format_exc();save();raise
    finally:
        if graphs:
            try:
                torch.cuda.synchronize()
                for g in graphs.values():g.reset()
                graphs.clear();result['graph_cleanup']='DRAINED_RESET'
            except BaseException:result['graph_cleanup_failure']=traceback.format_exc();result['status']='INVALID_NO_PERFORMANCE_ADMISSION'
        result['ended']=time.time();result['cleanup']='root guard independently verifies process/container/compute absence';save()

if __name__=='__main__':main()
