"""Threearm fixed M8 sparse-header cell, no full prepare/profiler/SASS campaign; root-only GPU run."""
import argparse,ast,hashlib,importlib.util,itertools,json,statistics,sys,time,traceback,subprocess,os
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
CAND=D.parent/"gv53_quarter_sparse_header_v1"
TRANSPORT=D.parent/"gv53_quarter_stream_v1"
OLD_SO=ROOT/'results/proj-rsi/scheduler/artifacts/research.gv53.quarter.stream.fixed.01/collection-1793140874912701666/logs/scheduler/gv53-quarter-stream-01/run/actual.so'
OLD_SO_SHA='4b85c059c953e91989f6d613e8148269cfbb8e0480ab120585abcac492c3e44c'
OLD_INIT='gv53_quarter_607e6faec9d5b5e706ac'
OLD_CUDA_SHA='607e6faec9d5b5e706ac3109c9a83af3ab8ba6e7b17e2bbea7a7de1fb7316038'
NEW_RUN=ROOT/'results/proj-rsi/scheduler/artifacts/research.gv53.quarter.sparse.fixed.01/collection-1793141738325901851/logs/scheduler/gv53-quarter-sparse-01/run'
NEW_SO_SHA='2eb511c0af3c4ce5bd3990e45d376c64d12016b876a2109ac0f263eafc8c17a7'
CLEANUP_HOOK=None
LABELS=('control','quarter','sparse')
CONTRASTS=(('control','quarter'),('control','sparse'),('quarter','sparse'))
DECL='std::vector<int64_t> clear_blas_cache();\nint64_t attrs(int64_t,int64_t);\nint64_t run(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def verify():
    m=json.loads((D/'runtime-manifest.json').read_text())
    for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
    return m

def codec_nodes(source):
    names={'_pack','_c12','_unc12','_gvesc','_frag_bits'};out=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef)and node.name in names:out.append(node)
        elif isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ('_OS','_SS')for t in node.targets):out.append(node)
    assert len(out)==7
    return out

def basis_columns():
    columns=[0,1,31,63,64,511,512,8191]
    assert len(columns)==len(set(columns))==8
    return columns

def schedule():
    return [dict(arm=arm,seed=91600,mode='warmup')for arm in LABELS for _ in range(5)]+[dict(arm=arm,seed=91600,mode='profile')for arm in LABELS]

def terminal(last,raw,oracle,integrity):
    assert set(last)==set(LABELS)
    records=[]
    for arm in LABELS:
        seed=next(r['seed']for r in reversed(raw)if r['arm']==arm and r['mode']!='warmup')
        assert seed==last[arm]
        records.append(dict(arm=arm,seed=seed,oracle=oracle(arm,seed),before_any_replay_or_mutation=True))
    integrity();return records

def cpu_check():
    verify();assert len(schedule())==18 and sum(r['mode']=='profile'for r in schedule())==3
    assert sha(OLD_SO)==OLD_SO_SHA and sha(NEW_RUN/'actual.so')==NEW_SO_SHA
    assert not any(isinstance(n,ast.ImportFrom)and n.module=='torch.utils.cpp_extension'for n in ast.walk(ast.parse(Path(__file__).read_text())))
    return {'state':'CPU_EXACTSO_PROFILE_DRIVER_PASS_GPU_UNRUN','JIT_calls':0,'profiled_launches':3,'warmup_calls':15,'fresh_weights_seed':5344,'X_seed':91600,'old_and_new_SO_hashes':'PASS','prior_math_and_transport_proofs_reused':True}

def run(args):
    global CLEANUP_HOOK
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    assert args.output.is_dir() and os.environ.get('GV53_COUNTER_TOKEN')==json.loads((args.output/'session.json').read_text())['capture_token']
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'pilot.cu'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':3,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'resource_gate':'legal residency only; actual registers/local/static recorded'}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<240,'inner240s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        import torch
        # Exact collected SOs only: no extension build
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('load_exact_collected_SOs_no_JIT')
        assert sha(NEW_RUN/'actual.so')==NEW_SO_SHA
        cu=(NEW_RUN/'actual.cu').read_text();digest=hashlib.sha256(cu.encode()).hexdigest();assert digest=='0e17b3c566c46a7b6981135a5aadf49a88e454ed1cd3fa1c07e60e56c50d6aa1'
        begin=time.monotonic();spec=importlib.util.spec_from_file_location('gv53_sparse_'+digest[:20],NEW_RUN/'actual.so');ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
        CLEANUP_HOOK=ext.clear_blas_cache
        assert sha(OLD_SO)==OLD_SO_SHA
        oldspec=importlib.util.spec_from_file_location(OLD_INIT,OLD_SO);oldext=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(oldext)
        assert sha(Path(ext.__file__))==NEW_SO_SHA and sha(Path(oldext.__file__))==OLD_SO_SHA
        modules={'control':(ext,0),'quarter':(oldext,1),'sparse':(ext,1)}
        result['arm_bindings']={arm:{'SO_sha256':OLD_SO_SHA if arm=='quarter'else NEW_SO_SHA,'CU_sha256':OLD_CUDA_SHA if arm=='quarter'else digest,'native_arm':modules[arm][1]}for arm in LABELS}
        result['cost']['native_load_seconds']=time.monotonic()-begin;result['cost']['native_JIT_builds']=0
        import shutil
        shutil.copy2(NEW_RUN/'actual.so',args.output/'sparse.so');shutil.copy2(NEW_RUN/'actual.cu',args.output/'sparse.cu');shutil.copy2(OLD_SO,args.output/'quarter.so')
        shutil.copy2(OLD_SO.with_name('actual.cu'),args.output/'quarter.cu')
        result['build']={'CUDA_sha256':digest,'SO_sha256':NEW_SO_SHA,'JIT':False}
        fields=['registers','local','user_static','max_dynamic','max_threads','occupancy','device_reserved','arm_echo']
        # Save all raw fields before asserting any expected value.
        result['raw_attributes']={arm:{key:int(modules[arm][0].attrs(modules[arm][1],j+1))for j,key in enumerate(fields)}for arm in LABELS};save()
        for i,arm in enumerate(LABELS):
            a=result['raw_attributes'][arm]
            assert 0<a['registers']<=255 and a['local']>=0 and a['user_static']>=0 and a['max_dynamic']>=17408 and a['max_threads']>=1024 and a['occupancy']>=1 and a['arm_echo']==modules[arm][1],(arm,a)
        phase('actual_task_weight_and_pack')
        begin=time.monotonic();w_seed=reference.WEIGHTS_SEED+193;weights=reference.generate_weights(w_seed)
        env={'torch':torch};exec(compile(ast.Module(body=codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<exact-original-codec-AST>','exec'),env)
        fr=env['_pack'](weights['pooler']);encoded=env['_c12'](fr,1);assert encoded is not None
        packed,table,esc6=encoded
        esc7=torch.cat((esc6,torch.full_like(esc6[:,:1],0)),1)
        escape=env['_gvesc'](esc7,256,(2,16,8,1,1,0));assert escape is not None
        sys.path.insert(0,str(TRANSPORT));from transport import pack_bank,unpack_group
        old_words=escape.detach().cpu().tolist()
        begin_transport=time.monotonic();new_words=pack_bank(old_words);assert new_words is not None
        assert len(new_words)==len(old_words)==131072
        record_count=0;unit_hist={str(i):0 for i in range(9)};full32=0
        for offset in range(0,len(old_words),32):
            accepted=[int(v)&0xffffffff for v in old_words[offset:offset+32]if (int(v)&0xffffffff)!=0xffffffff]
            expected=sorted(accepted,key=lambda v:(v>>20,(v>>13)&3))
            assert unpack_group(new_words[offset:offset+32])==expected,('actual accepted escape mapping',offset)
            record_count+=len(accepted);unit_hist[str(len({v>>20 for v in accepted}))]+=1;full32+=len(accepted)==32
        quarter_escape=torch.tensor([v if v<2**31 else v-2**32 for v in new_words],dtype=torch.int32,device=escape.device)
        assert quarter_escape.data_ptr()!=escape.data_ptr() and quarter_escape.numel()==escape.numel()
        escapes={'control':escape,'quarter':quarter_escape,'sparse':quarter_escape}
        result['escape_work_predictions']={'unit_occupancy_histogram':unit_hist,'full32_groups':full32,'nonempty_units':sum(int(k)*v for k,v in unit_hist.items()),'source_expected_parent_SHFL':4096+record_count-full32,'source_expected_quarter_header_SHFL':32768,'source_expected_quarter_sparse_data_SHFL':record_count,'scope':'Source-level logical per-warpgroup counts; profiler dynamicSIMT metric normalization differs, not a new validity gate'}
        result['transport']={'bytes_per_warpgroup':128,'groups':len(old_words)//32,'capacity_per_group':32,'record_count':record_count,'actual_accepted_records_exact_after_stable_bucketing':True,'original_overflow_rejection_preserved':True,'parent_bank_sha256':hashlib.sha256(escape.cpu().numpy().tobytes()).hexdigest(),'quarter_bank_sha256':hashlib.sha256(quarter_escape.cpu().numpy().tobytes()).hexdigest(),'separate_bank_pointers':True,'outside_timed_region':True,'host_transform_and_actual_mapping_check_seconds':time.monotonic()-begin_transport}
        immutable=[('pooler',weights['pooler'],weights['pooler'].clone()),('packed',packed,packed.clone()),('parent_escape',escape,escape.clone()),('quarter_escape',quarter_escape,quarter_escape.clone())]
        result['pack']={'W_seed':w_seed,'actual_W_bit_reconstruction':'EXACT','packed_bytes':packed.numel(),'escape_words':escape.numel(),'geometry':[2,16,8,1,1,0],'packing_functions':'Exact original source AST; prepare and extension factories not executed'}
        result['cost']['task_weight_pack_seconds']=time.monotonic()-begin
        flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda',dtype=torch.float32)
        def immutables():
            for name,t,copy in immutable:assert torch.equal(t.view(torch.uint8),copy.view(torch.uint8)),('immutable changed',name)
        for m in (8,):
            phase('fixed_pooler_M'+str(m));config={'layer':'pooler','m':m,'n':4096,'k':8192}
            entry=result['fixed_cells'][str(m)]={'config':config,'raw':[],'validation':{},'terminals':{},'mutations':[]};work={};arenas={}
            fixtures={seed:reference.generate_input(m,'pooler',seed)['x']for seed in {r['seed']for r in schedule()}}
            for index,arm in enumerate(LABELS):
                xb=torch.full((m*8192+32,),11.5,dtype=torch.bfloat16,device='cuda');x=xb[16:-16].reshape(m,8192)
                ob=torch.full((m*4096+32,),17.25,dtype=torch.bfloat16,device='cuda');out=ob[16:-16].reshape(m,4096)
                cb=torch.full((65536+32,),0x19a75d3,dtype=torch.int32,device='cuda');wb=torch.full((65536+32,),17.25,dtype=torch.float32,device='cuda')
                arenas[arm]=(xb,ob,cb,wb,cb.clone(),wb.clone());work[arm]=(x,out)
            assert len({v[0].data_ptr()for v in work.values()})==3 and len({v[1].data_ptr()for v in work.values()})==3
            def integrity():
                immutables()
                for xb,ob,cb,wb,cb0,wb0 in arenas.values():
                    assert bool((xb[:16]==11.5).all())and bool((xb[-16:]==11.5).all())
                    assert bool((ob[:16]==17.25).all())and bool((ob[-16:]==17.25).all())
                    assert torch.equal(cb,cb0)and torch.equal(wb.view(torch.uint8),wb0.view(torch.uint8)),'S1 counter/scratch full backing changed'
            def invoke(arm):
                x,out=work[arm];_,_,cb,wb,_,_=arenas[arm]
                assert int(modules[arm][0].run(modules[arm][1],packed,escapes[arm],x,out,wb[16:-16],cb[16:-16],table))==0,('launch declined',arm,m)
            def oracle_x(arm,fresh):
                x,out=work[arm];assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
                ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(arm,m,msg)
                ref=fresh.double()@weights['pooler'].double().T
                rel=float((out.double()-ref).norm()/ref.norm().clamp(min=1e-30));row=float(((out.double()-ref).norm(dim=-1)/ref.norm(dim=-1).clamp(min=1e-30)).max())
                assert rel<=4e-3 and row<=2e-2,(arm,m,rel,row)
                return {'task':msg,'FP64_L2':rel,'worst_row':row,'current_X':'EXACT'}
            for arm in LABELS:
                x,out=work[arm];checks=[]
                for kind,fresh in [('normal',reference.generate_input(m,'pooler',91301)['x']),('outliers',reference.generate_input(m,'pooler',91303,dist='outliers')['x']),('zero',torch.zeros_like(x))]:
                    x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize();checks.append(dict(kind=kind,oracle=oracle_x(arm,fresh)));integrity();budget()
                columns=basis_columns()
                for start in range(0,8,m):
                    selected=columns[start:start+m];fresh=torch.zeros_like(x);fresh[torch.arange(m,device=x.device),torch.tensor(selected,device=x.device)]=8
                    x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize()
                    expected=(weights['pooler'][:,selected].T*8).contiguous()
                    assert torch.equal(out.float(),expected.float())and torch.equal(x,fresh),('selected basis',arm,m,selected)
                integrity();entry['validation'][arm]={'task':checks,'basis_columns':columns,'basis':'EXACT'};save();budget()
            live={}
            for arm in LABELS:
                x,out=work[arm];x.copy_(fixtures[91600]);g=torch.cuda.CUDAGraph(keep_graph=True);graphs.append(g);live[arm]=g
                with torch.cuda.graph(g):invoke(arm)
            # Query actual graph attributes before any profiled launch (S1 alone is insufficient).
            metadata=importlib.util.spec_from_file_location('gv53_node_metadata',ROOT/'proj_rsi/audits/hotpath-fresh-counter-v2/selected_graph_metadata.py');meta=importlib.util.module_from_spec(metadata);metadata.loader.exec_module(meta);inspector=meta.Inspector()
            result['actual_graph_nodes']={arm:inspector.inspect(live[arm].raw_cuda_graph())for arm in LABELS};save()
            expected={'control':'gv_predxor','quarter':'gv_quarter','sparse':'gv_sparse_quarter'}
            for arm in LABELS:
                nodes=result['actual_graph_nodes'][arm];assert len(nodes)==1 and nodes[0]['node_type']==0
                node=nodes[0];assert node['cluster_query_returncode']==0 and node['cluster_dimension']in([0,0,0],[1,1,1]),('unsupported actualcluster',arm,node)
                assert node['grid']==[1,128,1] and node['block']==[1024,1,1] and node['dynamic_shared_bytes']==17408 and node['attributes']['local_bytes']==0
                assert expected[arm]in subprocess.check_output(['c++filt',node['function_mangled']],text=True,timeout=5)
                so=OLD_SO if arm=='quarter'else NEW_RUN/'actual.so'
                sass=subprocess.check_output(['cuobjdump','--dump-sass','--function',node['function_mangled'],str(so)],text=True,timeout=15)
                assert set(__import__('re').findall(r'Function : (\S+)',sass))=={node['function_mangled']}
                (args.output/(arm+'.actual.sass')).write_text(sass);node['SASS_sha256']=hashlib.sha256(sass.encode()).hexdigest()
                live[arm].instantiate()
            integrity();last={}
            def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
            result['clock_context']=[]
            def clock_context(arm,position):
                argv=['nvidia-smi','--query-gpu=timestamp,index,uuid,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw','--format=csv,noheader,nounits']
                begin=time.monotonic_ns()
                try:
                    q=subprocess.run(argv,capture_output=True,text=True,timeout=5);rec={'argv':argv,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
                except Exception as e:rec={'argv':argv,'unavailable_reason':repr(e)}
                result['clock_context'].append(dict(arm=arm,position=position,host_monotonic_ns=begin,**rec));save()
            # Cold read flush remains outside profile NVTX range. NCU cache-control all additionally flushes each replay pass.
            for r in schedule():
                arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);out.fill_(float('nan'))
                if r['mode']=='profile':clock_context(arm,'before_flush')
                torch.sum(flush,0,out=sink);torch.cuda.synchronize()
                if r['mode']=='profile':torch.cuda.nvtx.range_push('profile')
                try:live[arm].replay();torch.cuda.synchronize()
                finally:
                    if r['mode']=='profile':torch.cuda.nvtx.range_pop()
                last[arm]=r['seed'];row=dict(r);entry['raw'].append(row)
                with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
                if r['mode']=='profile':
                    # Own retained output checked before any replay/poison/repair of this arm.
                    entry['terminals'][arm]={'oracle':oracle(arm,r['seed']),'current_seed':r['seed'],'before_any_replay_or_mutation':True};integrity();clock_context(arm,'after_oracle')
                budget()
            assert set(entry['terminals'])==set(LABELS);result['profile_order']=list(LABELS);save()
            probe=torch.zeros_like(fixtures[91600]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns()[i]for i in range(m)],device=probe.device)]=8
            mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,'pooler',91401)['x']),('outliers',reference.generate_input(m,'pooler',91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,'pooler',91407)['x'])]
            for kind,fresh in mutations:
                for arm in LABELS:
                    x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                    message=oracle_x(arm,fresh)
                    if kind=='zero':assert not int(torch.count_nonzero(out))
                    integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
            for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
            entry['state']='VALID_EXACTSO_PROFILE_CELL_NO_TIMING_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['state']='THREE_ARM_EXACTSO_PROFILE_COMPLETE_ROOT_CLEANUP_REQUIRED';result['performance_scope']='Instrumented aggregate counters only, no performance gain or official score' ;phase('complete')
    except BaseException:
        result['state']='INVALID_NO_PERFORMANCE_ADMISSION';result['error']=traceback.format_exc();save();raise
    finally:
        cleanup=[]
        for g in graphs:
            try:torch.cuda.synchronize();g.reset();cleanup.append('RESET_AFTER_DRAIN')
            except BaseException as e:cleanup.append(repr(e));result['state']='INVALID_CLEANUP'
        result['cleanup']=cleanup;result['finished_at']=time.time();result['cost']['run_elapsed_seconds']=time.monotonic()-started;save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);args=p.parse_args()
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output and args.review
    error=None
    try:run(args)
    except BaseException as exc:error=repr(exc)
    import gc
    gc.collect();torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize();before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
        evidence=None;clear_error=None
        try:
            if CLEANUP_HOOK is not None:evidence=list(map(int,CLEANUP_HOOK()))
        except BaseException as exc:clear_error=repr(exc)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        cleanup={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'before_blas_clear':before,'blas_map_entries_workspace_bytes_after':evidence,'clear_error':clear_error,'after_run_scope_and_drain':True}
        if (args.output/'result.json').exists():
            result=json.loads((args.output/'result.json').read_text());result['post_scope_cleanup']=cleanup
            if cleanup['allocated']!=0 or clear_error is not None:result['state']='INVALID_CLEANUP'
            dump(args.output/'result.json',result)
        assert cleanup['allocated']==0 and clear_error is None,cleanup
        if error is None:assert evidence is not None and len(evidence)==3 and evidence[2]==0,cleanup
    if error:raise RuntimeError(error)
if __name__=='__main__':main()
