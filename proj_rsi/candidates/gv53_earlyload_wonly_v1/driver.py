"""One fixed M8 incumbent earlier-load cell, no full prepare/profiler/SASS campaign; root-only GPU run."""
import argparse,ast,hashlib,json,statistics,sys,time,traceback,importlib.util,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
CAND=D
CLEANUP_HOOK=None
LABELS=('parent','W_X','W_only')
DECL='std::vector<int64_t> clear_blas_cache();\nint64_t attrs(int64_t,int64_t);\nint64_t run(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def verify():
    m=json.loads((D/'manifest.json').read_text())
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
    import itertools
    rows=[]
    for arm in LABELS:rows +=[dict(arm=arm,seed=95000,mode='warmup')for _ in range(5)]
    for block,arm in enumerate(('parent','W_X','W_only','parent')):
        rows +=[dict(arm=arm,seed=95100+i,mode='F1',block=block,terminal=(i==10))for i in range(11)]
    for seed in range(95200,95208):
        for order_index,order in enumerate(itertools.permutations(LABELS)):
            for position,arm in enumerate(order):
                rows.append(dict(arm=arm,seed=seed,mode='paired',order=order_index,position=position,terminal=(seed==95207 and order_index==5)))
    assert len(rows)==203
    return rows

def terminal(arm,seed,last,raw,oracle,integrity):
    assert last[arm]==seed==next(r['seed']for r in reversed(raw)if r['arm']==arm)
    record=dict(arm=arm,seed=seed,oracle=oracle(arm,seed),before_any_next_replay_or_mutation=True)
    integrity();return record

def cpu_check():
    verify();rows=schedule();assert len(rows)==203 and sum(r['mode']=='warmup'for r in rows)==15
    paired=[r for r in rows if r['mode']=='paired'];assert len(paired)==144
    for seed in range(95200,95208):
        bank=[r for r in paired if r['seed']==seed];assert len(bank)==18
        for arm in LABELS:assert sum(r['arm']==arm for r in bank)==6
        for order in range(6):assert {r['arm']for r in bank if r['order']==order}==set(LABELS)
    assert sum(r.get('terminal',False)for r in rows)==7
    last={'parent':7};raw=[dict(arm='parent',seed=7)]
    assert terminal('parent',7,last,raw,lambda a,s:'PASS',lambda:None)['seed']==7
    for case in('stale_seed','invalid_output','counter_mutation'):
        try:
            def oracle(a,s):assert case!='invalid_output';return 'PASS'
            def integrity():assert case!='counter_mutation'
            terminal('parent',8 if case=='stale_seed'else 7,last,raw,oracle,integrity)
        except AssertionError:pass
        else:raise AssertionError(case)
    return {'state':'PASS_CPU_THREE_ARM_PROTOCOL_GPU_UNRUN','fixed_cells':1,'warmups':15,'clean_calls':188,'paired_groups':48,'paired_calls':144,'F1_calls':44,'own_terminals':7,'mutations':15,'same_actual_W_and_frozen_X_per_arm':True,'six_triple_orders_per_seed':True,'basis_columns_per_arm':8,'codec_AST_reused':True}

def run(args):
    global CLEANUP_HOOK
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'submission.py'),'pilot_sha256':sha(CAND/'pilot.cu'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'realized_scope':'W-only source Xsnapshot removed; compiler partly hoists X,392slots versus WX384/parent360;64regs/no spill; no cheaper-staging or gain assumption','resource_gate':'legal residency only; actual registers/local/static recorded'}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<60,'inner60s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        import torch
        # Validated thin C ABI: no Torch C++ extension build
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('load_cleanup_bridge_and_compile_thin')
        sys.path.insert(0,str(D));from adapter import ThinGV53
        manifest=verify();prior=ROOT/manifest['prior_directory'];assert sha(prior/'actual.so')==manifest['prior_SO_sha256']
        spec=importlib.util.spec_from_file_location(manifest['prior_module_name'],prior/'actual.so');oldext=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldext);CLEANUP_HOOK=oldext.clear_blas_cache
        cu=(CAND/'pilot.cu').read_text()+'\n'+(D/'bridge.cuh').read_text();(args.output/'actual.cu').write_text(cu);digest=hashlib.sha256(cu.encode()).hexdigest()
        stages=[];nvcc='/usr/local/cuda/bin/nvcc';begin=time.monotonic()
        for label,cmd in [('native_compile',[nvcc,'-c','-std=c++17','-arch=sm_90a','-O3','-lineinfo','-Xptxas=-v','-Xcompiler','-fPIC',str(args.output/'actual.cu'),'-o',str(args.output/'actual.o')]),('host_link',[nvcc,'-shared','-cudart','shared','-Xcompiler','-fPIC',str(args.output/'actual.o'),'-o',str(args.output/'actual.so')])]:
            budget();t=time.monotonic();q=subprocess.run(cmd,capture_output=True,text=True,timeout=35);stages.append(dict(stage=label,argv=cmd,seconds=time.monotonic()-t,returncode=q.returncode));(args.output/(label+'.stdout')).write_text(q.stdout);(args.output/(label+'.stderr')).write_text(q.stderr);assert q.returncode==0,q.stderr
        result['cost']['native_compile_and_link_seconds']=time.monotonic()-begin
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(args.output/'actual.so')],text=True,timeout=15);(args.output/'actual.sass').write_text(sass)
        def native_rows(text):
            import re
            return {c.splitlines()[0].strip():re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',c)for c in text.split('Function : ')[1:]}
        actual_rows=native_rows(sass);expected_rows=native_rows((D/'pilot.sass').read_text())
        names=['_Z10gv_predxorILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G','_Z12gv_earlyloadILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G','_Z18gv_earlyload_wonlyILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G']
        identity={n:bool(actual_rows[n])and actual_rows[n]==expected_rows[n]for n in names};assert all(identity.values()),identity
        thin=ThinGV53(args.output/'actual.so',sha(args.output/'actual.so'),torch,oldext.clear_blas_cache)
        class Native:
            def __init__(self):self.bounds={}
            def attrs(self,arm,field):return thin.attrs(arm,field,torch.cuda.current_device())
            def run(self,arm,W,esc,x,out,ws,cnt,table):
                key=tuple(t.data_ptr()for t in(W,esc,x,out,ws,cnt))+(table,)
                if key not in self.bounds:self.bounds[key]=thin.bind(W,esc,x,out,ws,cnt,table)
                return self.bounds[key].run(arm)
        ext=Native()
        result['build']={'CUDA_sha256':digest,'SO_sha256':sha(args.output/'actual.so'),'stages':stages,'CXX_Torch_compile_calls':0,'native_compile_calls':1,'actual_fullPC_predicate_encoding_equals_pilot':identity,'ABI':'validatedthinCABIwithcurrentstreamandstrongowners'}
        fields=['registers','local','user_static','max_dynamic','max_threads','occupancy','device_reserved','arm_echo']
        # Save all raw fields before asserting any expected value.
        result['raw_attributes']={arm:{key:int(ext.attrs(i,j+1))for j,key in enumerate(fields)}for i,arm in enumerate(LABELS)};save()
        for i,arm in enumerate(LABELS):
            a=result['raw_attributes'][arm]
            assert 0<a['registers']<=255 and a['local']>=0 and a['user_static']>=0 and a['max_dynamic']>=17408 and a['max_threads']>=1024 and a['occupancy']>=1 and a['arm_echo']==i,(arm,a)
        phase('actual_task_weight_and_pack')
        begin=time.monotonic();w_seed=reference.WEIGHTS_SEED+193;weights=reference.generate_weights(w_seed)
        env={'torch':torch};exec(compile(ast.Module(body=codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<exact-original-codec-AST>','exec'),env)
        fr=env['_pack'](weights['pooler']);encoded=env['_c12'](fr,1);assert encoded is not None
        packed,table,esc6=encoded
        esc7=torch.cat((esc6,torch.full_like(esc6[:,:1],0)),1)
        escape=env['_gvesc'](esc7,256,(2,16,8,1,1,0));assert escape is not None
        escapes={arm:escape for arm in LABELS}
        result['escape_contract']={'three_arms_same_original_bank':True,'SHA256':hashlib.sha256(escape.cpu().numpy().tobytes()).hexdigest(),'transport_repack_calls':0}
        immutable=[('pooler',weights['pooler'],weights['pooler'].clone()),('packed',packed,packed.clone()),('escape',escape,escape.clone())]
        result['pack']={'W_seed':w_seed,'actual_W_bit_reconstruction':'EXACT','packed_bytes':packed.numel(),'escape_words':escape.numel(),'geometry':[2,16,8,1,1,0],'packing_functions':'Exact original source AST; prepare and extension factories not executed'}
        result['cost']['task_weight_pack_seconds']=time.monotonic()-begin
        flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda',dtype=torch.float32)
        def immutables():
            for name,t,copy in immutable:assert torch.equal(t.view(torch.uint8),copy.view(torch.uint8)),('immutable changed',name)
        for m in (8,):
            phase('fixed_pooler_M'+str(m));config={'layer':'pooler','m':m,'n':4096,'k':8192}
            entry=result['fixed_cells'][str(m)]={'config':config,'raw':[],'validation':{},'terminals':[],'mutations':[]};work={};arenas={}
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
                assert int(ext.run(LABELS.index(arm),packed,escapes[arm],x,out,wb[16:-16],cb[16:-16],table))==0,('launch declined',arm,m)
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
                x,out=work[arm];x.copy_(fixtures[95000]);g=torch.cuda.CUDAGraph();graphs.append(g);live[arm]=g
                with torch.cuda.graph(g):invoke(arm)
            integrity();last={};timing_events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
            for i,r in enumerate(schedule()):
                arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);torch.sum(flush,0,out=sink)
                a,b=timing_events;t_submit=time.monotonic_ns();a.record();live[arm].replay();b.record();t_enqueued=time.monotonic_ns();torch.cuda.synchronize();t_synced=time.monotonic_ns()
                us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                row=dict(r,us=us,host_before_start_ns=t_submit,host_after_end_enqueue_ns=t_enqueued,host_after_sync_ns=t_synced);entry['raw'].append(row);last[arm]=r['seed']
                t_journal=time.monotonic_ns()
                with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(row,m=m))+'\n');f.flush()
                entry.setdefault('host_journal_spans_ns',[]).append([t_journal,time.monotonic_ns()])
                if r.get('terminal'):entry['terminals'].append(dict(mode=r['mode'],receipt=terminal(arm,r['seed'],last,entry['raw'],oracle,integrity)));save()
                budget()
            f1=[r for r in entry['raw']if r['mode']=='F1'];med=[statistics.median(r['us']for r in f1 if r['block']==i)for i in range(4)]
            entry['F1_medians_us']=dict(zip(('parent_before','W_X','W_only','parent_after'),med))
            groups=[]
            for seed in range(95200,95208):
                for order in range(6):
                    group={r['arm']:r['us']for r in entry['raw']if r['mode']=='paired'and r['seed']==seed and r['order']==order};assert set(group)==set(LABELS);groups.append(dict(seed=seed,order=order,times_us=group))
            entry['paired']=groups;entry['contrasts']={}
            for left,right in [('parent','W_X'),('parent','W_only'),('W_X','W_only')]:
                diffs=[g['times_us'][left]-g['times_us'][right]for g in groups]
                entry['contrasts'][left+'_minus_'+right]={'paired_median_us':statistics.median(diffs),'paired_mean_us':statistics.mean(diffs),'all_signed_us':diffs,'wins_ties_losses':[sum(x>0 for x in diffs),sum(x==0 for x in diffs),sum(x<0 for x in diffs)]}
            entry['F1_contrasts']={arm:{'parent_before_minus_arm_us':med[0]-med[i],'parent_after_minus_arm_us':med[3]-med[i],'endpoint_average_minus_arm_us':(med[0]+med[3])/2-med[i]}for i,arm in[(1,'W_X'),(2,'W_only')]};save()
            probe=torch.zeros_like(fixtures[95000]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns()[i]for i in range(m)],device=probe.device)]=8
            mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,'pooler',91401)['x']),('outliers',reference.generate_input(m,'pooler',91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,'pooler',91407)['x'])]
            for kind,fresh in mutations:
                for arm in LABELS:
                    x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                    message=oracle_x(arm,fresh)
                    if kind=='zero':assert not int(torch.count_nonzero(out))
                    integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
            for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
            for binding in ext.bounds.values():binding.close_after_graph_reset()
            ext.bounds.clear()
            entry['state']='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['state']='THREE_ARM_W_ONLY_FALSIFIER_COMPLETE_ROOT_REVIEW_REQUIRED';result['performance_scope']='Fixed M8 parent/W+X/W-only realized schedule comparison, not pure fixed-X isolation, selected policy or official score';phase('complete')
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
