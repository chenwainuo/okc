"""Root-only unchangedPW38 serving hotpath timeline; no NCU replay/hardware counters.
CPU-check performs no CUDA/NCU/prepare/context/extension operation.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'proj_rsi/candidates/pw38_hotpath_trace_trial'
spec=importlib.util.spec_from_file_location('selected_profile_support',HERE/'support.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


def final_oracle(outputs,inputs,fixtures,configs,weights,immutable,counter_check):
    # No replay/repair/poison/input mutation before all actual outputs pass.
    expected={name for name,_,_ in S.CASES}
    assert set(outputs)==set(inputs)==set(fixtures)==set(configs)==expected,'missing actual case'
    ptrs=[t.data_ptr()for t in list(outputs.values())+list(inputs.values())]
    assert len(set(ptrs))==len(ptrs),'input/output alias'
    records=[]
    for name,layer,m in S.CASES:
        x=inputs[name];out=outputs[name]
        assert torch.equal(x.view(torch.uint8),fixtures[name].view(torch.uint8)),(name,'current input changed')
        assert bool(torch.isfinite(out).all()),(name,'actual output poison/nonfinite')
        ok,msg=reference.check_implementation({'x':fixtures[name],'config':configs[name]},out,weights)
        assert ok,(name,'actual final selected output',msg)
        records.append({'case':name,'current_X':'BYTE_IMMUTABLE','actual_retained_output_before_any_replay':'PASS','oracle':msg})
    assert all(same_bytes(t,b)for _,t,b in immutable),'W/packed/escape state changed'
    counter_check()
    return records


def same_bytes(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def library_identities():
    found={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if not fields or not fields[-1].startswith('/'):continue
        p=Path(fields[-1]);name=p.name
        if any(token in name for token in ('libcuda.so','libcudart.so','libcublas.so','libcublasLt.so','libnvJitLink.so')):
            resolved=p.resolve()
            if str(resolved)not in found:found[str(resolved)]={'path':str(resolved),'bytes':resolved.stat().st_size if resolved.is_file()else None,'sha256':S.sha(resolved)if resolved.is_file()else None,'source':'CLOSED_LIBRARY_SOURCE_UNKNOWN'if'cublas'in name else'installed runtime library'}
    return list(found.values())


def selected_lt_config(mod,state,layer,m,n,k):
    code=state.get('lt',{}).get((layer,m))
    if code is None:return {'selected_code':None,'selected_algorithm':'UNKNOWN_TORCH_DEFAULT_CLOSED_DISPATCH','default_algorithm':'UNKNOWN','read_only':True}
    assert isinstance(code,int)and code>=0,'unrecognized selected LT code'
    values=[int(v)for v in mod._ext().lt_cfg(m,n,k,int(layer=='router'),code)]
    assert len(values)==9
    return {'selected_code':code,'layout_code':code//100000,'config_raw':values,'config_field_order':['ID','TILE_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION','STAGES_ID','INNER_SHAPE_ID','CLUSTER_SHAPE_ID'],
      'default_algorithm':'UNKNOWN_TORCH_DEFAULT_CLOSED_DISPATCH','read_only':True}


def validate_case(mod,state,stack,layer,m,config,weights,plan,immutable):
    w=weights[layer];n=config['n'];k=config['k'];before=stack.counter_arena(state);calls=0;maximum=0.0
    records=[]
    for dist in ('normal','outliers'):
        data=reference.generate_input(m,layer,88100,dist=dist);snapshot=data['x'].clone()
        out=mod.kernel_fn(data,state);calls+=1
        assert bool(torch.isfinite(out).all())
        ok,msg=reference.check_implementation(data,out,weights);assert ok,(layer,dist,msg)
        assert same_bytes(data['x'],snapshot)
        records.append({'dist':dist,'task_FP64_row_and_probe_oracle':msg})
    torch.cuda.synchronize();stack.counter_check(before,stack.counter_arena(state),plan,calls)
    assert all(same_bytes(t,b)for _,t,b in immutable)
    return {'full_K_columns':None,'basis_batches':0,'basis':'PRIOR_UNCHANGED_SOURCE_VALIDITY_REUSED_NOT_RERUN','router_max_basis_relative_error':None,'task':records,'logical_direct_calls':calls,'source_bound_counter_contract':'PASS'}


def establish_metadata_context(torch_module):
    # init alone may leave CUDA12.8 primary context noncurrent. Allocation and
    # current-stream runtime access establish it without launching model math.
    torch_module.cuda.init()
    anchor=torch_module.empty((1,),device='cuda',dtype=torch_module.uint8)
    torch_module.cuda.current_stream().synchronize()
    assert anchor.numel()==1 and anchor.element_size()==1 and anchor.is_cuda
    return anchor


def run(args):
    S.verify();assert args.review and args.review.is_file()
    assert args.output.is_absolute() and args.output.resolve()==args.output and (ROOT/'logs/scheduler') in args.output.parents
    args.output.mkdir(parents=True,exist_ok=False)
    global torch,reference
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    metadata=S.module(HERE/'graph_metadata.py','pw38_graph_metadata');inspector=None
    stack=S.stack();stack.verify();graphs=[];result={'status':'CAPTURE_STARTED_NO_COUNTER_ADMISSION','source_sha256':S.PW38_SHA,'JIT':[],'jit_errors':[],'cases':{},'profile_order':[name for name,_,_ in S.CASES],'scope':args.scope,'review_sha256':S.sha(args.review),'trace_kind':'projection_microbenchmark_not_full_model_traffic','scored_context':{'path':'proj_rsi/audits/selected-hotpaths-pw38/report.json','sha256':S.sha(ROOT/'proj_rsi/audits/selected-hotpaths-pw38/report.json'),'join_keys':['layer','m'],'limits':'Official H200 scoredcontext is separate from freshH100 picks/cold-event timings/instrumentedCUDA traces; no productionrequest frequencies or sampledinnerloop costs.'}}
    def save():stack.dump(args.output/'capture.json',result)
    lock=threading.RLock()
    try:
        # Retain tiny runtime allocation so the query has a current primary context.
        metadata_context_anchor=establish_metadata_context(torch)
        inspector=metadata.Inspector()
        result['hardware']=inspector.hardware();save()
        metadata.require_prepare_hardware(result['hardware'])
        result['context_establishment']={'method':'retained_one_byte_cuda_allocation_and_current_stream_synchronize','allocation_bytes':1,'model_kernel_launches':0,'anchor_device':str(metadata_context_anchor.device)}
        result['preprepare_hardware_gate']='PASS_SM90_AT_LEAST_128_SMS';save()
        weights=reference.generate_weights(reference.WEIGHTS_SEED)
        immutable=[('originalW/'+name,t,t.clone())for name,t in weights.items()]
        mod=S.module(ROOT/'proj_rsi/candidates/pw38/submission.py','actual_fresh_pw38_profile')
        original=mod.load_inline;registry={}
        def checked_build(*a,**kw):
            try:
                name=kw.get('name',a[0]if a else None);cu=kw['cuda_sources'];assert isinstance(cu,str)
                digest=hashlib.sha256(cu.encode()).hexdigest()
                with lock:
                    assert name not in registry or registry[name]==digest,'extension name collision';registry[name]=digest
                ext=original(*a,**kw);binary=Path(ext.__file__).resolve()
                sass=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True,timeout=60)
                resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(binary)],text=True,timeout=60)
                with lock:
                    prefix='JIT%02d.'%len(result['JIT'])+re.sub(r'[^A-Za-z0-9_.-]','_',str(name))
                    for suffix,text in (('.cu',cu),('.sass',sass),('.resources',resources)):(args.output/(prefix+suffix)).write_text(text)
                    shutil.copyfile(binary,args.output/(prefix+'.so'))
                    assert S.sha(binary)==S.sha(args.output/(prefix+'.so')),'actual SO copy changed'
                    gate=None
                    if 'struct RqP'in cu:
                        gate=S.P45.canonical_gate(cu,sass,resources)
                        gate['EH']=stack.actual_rq_gate(sass,resources,'rq_go<128,5,1,0,0,0,65538>' in re.sub(r'\s+','',cu))
                        gate['flags']=gate['EH']['flags']
                    record={'prefix':prefix,'name':name,'loaded_SO_path':str(binary),'archived_SO':prefix+'.so','CPP_source_argument_sha256':hashlib.sha256(str(kw.get('cpp_sources')).encode()).hexdigest(),'CUDA_source_sha256':digest,'SO_sha256':S.sha(binary),'SASS_sha256':S.sha(args.output/(prefix+'.sass')),'resource_sha256':S.sha(args.output/(prefix+'.resources')),
                        'functions':re.findall(r'Function : (\S+)',sass),'CUDA_flags':kw.get('extra_cuda_cflags'),'before_return_to_prepare_math':True,'RQ_gate':gate}
                    result['JIT'].append(record);save()
                return ext
            except BaseException:
                with lock:result['jit_errors'].append(traceback.format_exc());save()
                raise
        mod.load_inline=checked_build
        threads=[];start_thread=mod._rqstart
        def recorded_start(*a,**kw):
            t=start_thread(*a,**kw);threads.append(t);return t
        mod._rqstart=recorded_start
        begin=time.time();state=mod.prepare(weights);torch.cuda.synchronize()
        for t in threads:t.join(timeout=max(0,300-(time.time()-mod._T0)))
        assert not any(t.is_alive()for t in threads)and not result['jit_errors'],'prepare/JIT thread failure'
        assert not state.get('cpu')and time.time()-mod._T0<300,'original prepare overrun/CPU fallback'
        assert any(r.get('RQ_gate',{}).get('EH',{}).get('flags')==[2,3,65538,65539]for r in result['JIT']if r.get('RQ_gate')),'EH/control JIT never gated'
        actual_args={tuple(a) for r in result['JIT'] for a in (r.get('RQ_gate') or {}).get('args',[])}
        expected_args=set(S.P45.streams((S.P45.CAND/'sass.txt').read_text()))-{(32,4,1,0,0,0,52),(32,5,1,0,0,0,52)}
        assert actual_args==expected_args and len(actual_args)==19,'complete actual PW38 RQ catalog required'
        result['actual_RQ_functions']=len(actual_args)
        result['prepare']={'wall_seconds':time.time()-begin,'original_import_clock_elapsed':time.time()-mod._T0,'clock_reset':False,'original_PW38_source_unchanged':True}
        seen=set()
        for name,t in stack.tensors(state):
            if not t.numel()or any(word in name for word in ('cnt','ws','sink','prof')):continue
            view=(stack.tensor_id(t),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
            if view not in seen:seen.add(view);immutable.append((name,t,t.clone()))
        assert all(same_bytes(t,b)for _,t,b in immutable)
        original_rq=mod._rqcall
        def guarded(*a,**kw):
            rc=original_rq(*a,**kw);assert rc==0,('HOLD actual RQ fallback',rc);return rc
        mod._rqcall=guarded
        original_gv=mod._gvcall
        def guarded_gv(*a,**kw):
            rc=original_gv(*a,**kw);assert rc==0,('serving GV/router fallback',rc);return rc
        mod._gvcall=guarded_gv
        native_ext=mod._ext
        mod._ext=lambda:S.P45.ServingStatus(native_ext())
        assert inspector.hardware()==result['hardware'],'hardware identity changed after prepare'
        result['software']={'torch':torch.__version__,'torch_CUDA_build':torch.version.cuda,'Python':sys.version,'libraries':library_identities()}
        plans={};configs={};inputs={};fixtures={};outputs={};objects={}
        # Admit lazy selected counter allocations before any checked intervals.
        for index,(name,layer,m)in enumerate(S.CASES):
            data=reference.generate_input(m,layer,89100+index)
            mod.kernel_fn(data,state);torch.cuda.synchronize()
        for index,(name,layer,m)in enumerate(S.CASES):
            data=reference.generate_input(m,layer,89110+index);cfg=data['config'];configs[name]=cfg
            info,plan=stack.counter_plan(mod,state,layer,m,cfg['n'],cfg['k']);plans[name]=plan
            record=result['cases'][name]={'layer':layer,'m':m,'config':cfg,'actual_selected_policy':info,'counter_plan':S.serial_plan(plan)}
            if info['family']=='LT':record['selected_LT_configuration']=selected_lt_config(mod,state,layer,m,cfg['n'],cfg['k'])
            record['numerics']=validate_case(mod,state,stack,layer,m,cfg,weights,plan,immutable);save()
            x=data['x'];inputs[name]=x;before=stack.counter_arena(state)
            for _ in range(2):mod.kernel_fn({'x':x,'config':cfg},state)
            torch.cuda.synchronize();stack.counter_check(before,stack.counter_arena(state),plan,2)
            before=stack.counter_arena(state)
            g=torch.cuda.CUDAGraph(keep_graph=True);graphs.append(g)
            with torch.cuda.graph(g):out=mod.kernel_fn({'x':x,'config':cfg},state)
            torch.cuda.synchronize();stack.counter_check(before,stack.counter_arena(state),plan,0)
            nodes=inspector.inspect(g.raw_cuda_graph());kernel_nodes=[n for n in nodes if n['node_type']==0];assert kernel_nodes,'no actual kernel nodes';g.instantiate()
            node=kernel_nodes[0]
            stack.counter_check(before,stack.counter_arena(state),plan,0)
            for node in kernel_nodes:
                custom=[r for r in result['JIT'] if node['function_mangled'] in r['functions']]
                if info['family']!='LT':assert custom,('custom graph kernel lacks actual source/SASS',name,node)
                node['source_binding']={'custom_JIT_records':[{'prefix':r['prefix'],'SO_sha256':r['SO_sha256'],'CUDA_source_sha256':r['CUDA_source_sha256'],'SASS_sha256':r['SASS_sha256']} for r in custom],
                    'closed_library_PC_source':'UNKNOWN_CLOSED_LIBRARY' if not custom else None}
                node['demangled']=subprocess.check_output(['c++filt',node['function_mangled']],text=True,timeout=5).strip()
            record['actual_graph_nodes']=nodes;objects[name]=g;outputs[name]=out
            # A fresh current fixture is validated before the selected range.
            fresh=reference.generate_input(m,layer,89200+index)['x'];x.copy_(fresh);fixtures[name]=fresh.clone();out.fill_(float('nan'))
            before=stack.counter_arena(state);g.replay();torch.cuda.synchronize();stack.counter_check(before,stack.counter_arena(state),plan,1)
            assert bool(torch.isfinite(out).all());ok,msg=reference.check_implementation({'x':x,'config':cfg},out,weights);assert ok,(name,msg)
            assert same_bytes(x,fresh);record['fresh_pre_range_oracle']=msg;save()
        assert len(graphs)==len(S.CASES) and len(set(t.data_ptr()for t in list(outputs.values())+list(inputs.values())))==2*len(S.CASES)
        baseline=stack.counter_arena(state);counts={name:0 for name,_,_ in S.CASES}
        result['pre_trace_counters']=S.serial_counters(baseline)
        result['trace_cases']={}
        def check_counters():stack.counter_check(baseline,stack.counter_arena(state),S.weighted_plan(plans,counts),1)
        def terminal(name):
            x=inputs[name];out=outputs[name]
            assert same_bytes(x,fixtures[name]),(name,'current input changed')
            assert bool(torch.isfinite(out).all()),(name,'nonfinite actual own output')
            ok,msg=reference.check_implementation({'x':fixtures[name],'config':configs[name]},out,weights)
            assert ok,(name,'own retained output before replay/mutation',msg)
            assert all(same_bytes(t,b) for _,t,b in immutable),'immutable state changed'
            check_counters()
            return {'case':name,'current_X':'BYTE_IMMUTABLE','actual_retained_output_before_any_replay':'PASS','oracle':msg}
        flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
        from torch.profiler import profile,ProfilerActivity,record_function
        for index,(name,layer,m) in enumerate(S.CASES):
            x=inputs[name];out=outputs[name];g=objects[name];cfg=configs[name]
            entry=result['trace_cases'][name]={'pick':result['cases'][name]['actual_selected_policy'],'benchmarks':[],'mutations':[]}
            for rep in range(11):
                fresh=reference.generate_input(m,layer,92100+100*index+rep)['x'];x.copy_(fresh);fixtures[name]=fresh.clone();out.fill_(float('nan'))
                torch.sum(flush,0,out=sink);a,b=torch.cuda.Event(True),torch.cuda.Event(True)
                a.record();g.replay();b.record();torch.cuda.synchronize();counts[name]+=1
                us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                entry['benchmarks'].append({'replay':rep,'seed':92100+100*index+rep,'cold_L2_event_us':us})
            entry['benchmark_terminal']=terminal(name);save()
            probes=torch.zeros_like(x);rows=torch.arange(m,device='cuda');probes[rows,(rows*37+19)%cfg['k']]=8
            mutations=[('zero',torch.zeros_like(x)),('outliers',reference.generate_input(m,layer,95100+index,dist='outliers')['x']),('all_probes',probes),('fresh_normal',reference.generate_input(m,layer,95200+index)['x'])]
            for kind,fresh in mutations:
                x.copy_(fresh);fixtures[name]=fresh.clone();out.fill_(float('nan'))
                info=entry['pick'];assert not(info['family']=='router' and info['id'] in (8,11)),'HOLD persistent-zero RED2 workspace'
                seen=set();poisoned=[]
                for path,t in stack.tensors(state):
                    if 'ws' in path and t.numel() and t.is_floating_point() and stack.tensor_id(t) not in seen:
                        seen.add(stack.tensor_id(t));t.fill_(float('nan'));poisoned.append(path)
                g.replay();torch.cuda.synchronize();counts[name]+=1
                record={'kind':kind,'terminal':terminal(name),'poisoned_float_scratch':poisoned}
                if kind=='zero':assert not int(torch.count_nonzero(out))
                entry['mutations'].append(record);save()
            fresh=reference.generate_input(m,layer,96100+index)['x'];torch.cuda.synchronize()
            trace_path=args.output/(name+'.chrome.json')
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                with record_function('PW38_TRACE_SETUP'):
                    x.copy_(fresh);fixtures[name]=fresh.clone();out.fill_(float('nan'));torch.cuda.synchronize()
                for rep in range(3):
                    with record_function('PW38_TRACE_REPLAY/'+str(rep)):
                        g.replay();torch.cuda.synchronize();counts[name]+=1
            # Read OWN last output/current input/counters before any further replay/mutation.
            entry['profile_terminal']=terminal(name);entry['profile_replays']=3;save()
            trace_path.parent.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(trace_path))
            entry['trace']=S.parse_trace(json.loads(trace_path.read_text()),result['cases'][name]['actual_graph_nodes'],3)
            entry['trace_path']=str(trace_path.relative_to(args.output));entry['trace_sha256']=S.sha(trace_path)
            entry['counter_calls_since_snapshot']=counts[name]
            entry['latency_boundaries']='cold_L2_event_us is unprofiled benchmark; trace CUDA durations and CPU inclusive scopes are instrumented diagnostics, not scored latency'
            save()
        result['final_all_retained_output_oracles']=final_oracle(outputs,inputs,fixtures,configs,weights,immutable,check_counters)
        result['post_trace_counters']=S.serial_counters(stack.counter_arena(state));check_counters()
        result['phase']='ALL_ACTUAL_PROFILE_TERMINALS_PASSED'
        result['status']='VALID_PW38_HOTPATH_TRACE_ROOT_REVIEW_REQUIRED';save()
    except BaseException:
        result['status']='INVALID_SELECTED_PROFILE_NO_COUNTER_ADMISSION';result['failure']=traceback.format_exc();save();raise
    finally:
        cleanup=[]
        for g in graphs:
            try:torch.cuda.synchronize();g.reset();cleanup.append('DRAINED_RESET')
            except BaseException:cleanup.append(traceback.format_exc());result['status']='INVALID_GRAPH_CLEANUP'
        result['graph_cleanup']=cleanup;save()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--scope',choices=('top4','all30'),default='all30');ap.add_argument('--output',type=Path);ap.add_argument('--review',type=Path);args=ap.parse_args()
    S.CASES=S.case_scope(args.scope)
    if args.cpu_check:print(json.dumps(S.module(HERE/'cpu_proof.py','hotpath_cpu').proof(),indent=2));return
    assert args.output;run(args)
if __name__=='__main__':main()
