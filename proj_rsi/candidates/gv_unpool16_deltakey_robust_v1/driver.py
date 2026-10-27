"""One official selected unpoolerM16 fixedpair, no fullprepare/profiler; root-only GPU."""
import argparse,ast,hashlib,json,statistics,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
CAND=ROOT/'proj_rsi/candidates/gv_unpool16_deltakey_v1'
CLEANUP_HOOK=None
LABELS=('control','deltakey')
WEIGHT_SEEDS=(6203,6257,6287)
X_SEEDS=tuple(range(92100,92108))
def xdist(seed):return 'normal' if seed<92104 else 'outliers'
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
    columns=[0,1,15,16,31,32,63,64,127,128,511,512,1023,1024,4094,4095]
    assert len(columns)==len(set(columns))==16
    return columns

def schedule(weight_index=0):
    rows=[]
    for arm in LABELS:rows += [dict(arm=arm,seed=X_SEEDS[0],mode='warmup')for _ in range(5)]
    a,b=LABELS if weight_index%2==0 else LABELS[::-1]
    for phase,arms in [('F1_pre',(a,b,a)),('interleaved',()),('F1_post',(b,a,b))]:
        if phase=='interleaved':
            for order_index,order in enumerate((LABELS,LABELS[::-1])):
                for seed in X_SEEDS:
                    for position,arm in enumerate(order):
                        rows.append(dict(arm=arm,seed=seed,mode=phase,order=order_index,position=position,terminal_arms=list(LABELS)if seed==X_SEEDS[-1]and position==1 else []))
        else:
            for block,arm in enumerate(arms):
                for seed in X_SEEDS:rows.append(dict(arm=arm,seed=seed,mode=phase,block=block,terminal_arms=[arm]if seed==X_SEEDS[-1]else []))
    assert len(rows)==90
    return rows

def terminal(last,raw,oracle,integrity):
    assert set(last)==set(LABELS)
    records=[]
    for arm in LABELS:
        seed=next(r['seed']for r in reversed(raw)if r['arm']==arm and r['mode']!='warmup')
        assert seed==last[arm]
        records.append(dict(arm=arm,seed=seed,oracle=oracle(arm,seed),before_any_replay_or_mutation=True))
    integrity();return records

def cpu_check():
    verify()
    for wi in range(3):
        rows=schedule(wi);assert len(rows)==90 and sum(r['mode']!='warmup'for r in rows)==80
        assert sum(len(r.get('terminal_arms',[]))for r in rows)==10
        inter=[r for r in rows if r['mode']=='interleaved']
        for seed in X_SEEDS:
            group=[r for r in inter if r['seed']==seed];assert [r['arm']for r in group]==['control','deltakey','deltakey','control']
        for arm in LABELS:assert sum(r['arm']==arm and r['mode']!='warmup'for r in rows)==40
    return {'state':'PASS_CPU_SAME_BANK_WEIGHT_SCHEDULE_GPU_UNRUN','fresh_weights':list(WEIGHT_SEEDS),'X_seeds':list(X_SEEDS),'unique_X_fixtures':8,'normal_X':4,'outlier_X':4,'clean_calls':240,'warmups':30,'own_terminal_oracles':30,'native_build_calls':0,'profiler_calls':0,'blocked_both_orientations':True,'same_bank_both_styles_and_all_weights_bytes_bound':True}

def run(args):
    global CLEANUP_HOOK
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'submission.py'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':0,'full_prepare_calls':0}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<60,'inner60s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        import torch
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('load_exact_collected_native_no_JIT')
        manifest=verify();prior=json.loads((args.prior/'result.json').read_text())
        assert sha(args.prior/'result.json')==manifest['prior_result_sha256']
        assert sha(args.prior/'actual.so')==manifest['SO_sha256']==prior['build']['SO_sha256']
        assert prior['source_sha256']==manifest['candidate_source_sha256']
        spec=__import__('importlib.util',fromlist=['x']).spec_from_file_location(manifest['module_name'],args.prior/'actual.so')
        ext=__import__('importlib.util',fromlist=['x']).module_from_spec(spec);sys.modules[manifest['module_name']]=ext;begin=time.monotonic();spec.loader.exec_module(ext)
        CLEANUP_HOOK=ext.clear_blas_cache
        __import__('shutil').copy2(args.prior/'actual.so',args.output/'actual.so')
        result['build']={'SO_sha256':sha(args.output/'actual.so'),'exact_prior_SO_reused':True,'module_name':manifest['module_name'],'prior_result_sha256':manifest['prior_result_sha256'],'native_compile_calls':0}
        result['cost']['native_load_seconds']=time.monotonic()-begin
        fields=['registers','local','user_static','max_dynamic','max_threads','occupancy','device_reserved','arm_echo']
        # Save all raw fields before asserting any expected value.
        result['raw_attributes']={arm:{key:int(ext.attrs(i,j+1))for j,key in enumerate(fields)}for i,arm in enumerate(LABELS)};save()
        for i,arm in enumerate(LABELS):
            a=result['raw_attributes'][arm]
            assert 0<a['registers']<=128 and a['local']>=0 and a['user_static']>=0 and a['max_dynamic']>=5120 and a['max_threads']>=128 and a['occupancy']>=1 and a['arm_echo']==i,(arm,a)
        for weight_index,weight_seed in enumerate(WEIGHT_SEEDS):
            phase('actual_task_weight_and_pack')
            begin=time.monotonic();weights=reference.generate_weights(weight_seed)
            env={'torch':torch};exec(compile(ast.Module(body=codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<exact-original-codec-AST>','exec'),env)
            fr=env['_pack'](weights['unpooler']);encoded=env['_c12'](fr,1);assert encoded is not None
            packed,table,esc6=encoded
            esc7=torch.cat((esc6,torch.full_like(esc6[:,:1],0)),1)
            escape=env['_gvesc'](esc7,512,(1,4,16,1,1,2));assert escape is not None
            immutable=[(key,t,t.clone())for key,t in weights.items()]+[('packed',packed,packed.clone()),('escape',escape,escape.clone())]
            result['pack']={'W_seed':weight_seed,'actual_W_bit_reconstruction':'EXACT','packed_bytes':packed.numel(),'escape_words':escape.numel(),'geometry':[1,4,16,1,1,2],'packing_functions':'Exact original source AST; prepare and extension factories not executed'}
            result['cost']['task_weight_pack_seconds']=time.monotonic()-begin
            flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda',dtype=torch.float32)
            def immutables():
                for name,t,copy in immutable:assert torch.equal(t.view(torch.uint8),copy.view(torch.uint8)),('immutable changed',name)
            for m in (16,):
                phase('fixed_unpooler_M'+str(m));config={'layer':'unpooler','m':m,'n':8192,'k':4096}
                entry=result['fixed_cells'][str(weight_seed)]={'config':config,'raw':[],'validation':{},'terminals':{},'mutations':[]};work={};arenas={}
                entry['weight_seed']=weight_seed;entry['pack']=dict(result['pack'])
                fixtures={seed:reference.generate_input(m,'unpooler',seed,dist=xdist(seed))['x']for seed in X_SEEDS}
                for index,arm in enumerate(LABELS):
                    xb=torch.full((m*4096+32,),11.5,dtype=torch.bfloat16,device='cuda');x=xb[16:-16].reshape(m,4096)
                    ob=torch.full((m*8192+32,),17.25,dtype=torch.bfloat16,device='cuda');out=ob[16:-16].reshape(m,8192)
                    cb=torch.full((65536+32,),0x19a75d3,dtype=torch.int32,device='cuda');wb=torch.full((65536+32,),17.25,dtype=torch.float32,device='cuda')
                    arenas[arm]=(xb,ob,cb,wb,cb.clone(),wb.clone());work[arm]=(x,out)
                assert len({v[0].data_ptr()for v in work.values()})==2 and len({v[1].data_ptr()for v in work.values()})==2
                def integrity():
                    immutables()
                    for xb,ob,cb,wb,cb0,wb0 in arenas.values():
                        assert bool((xb[:16]==11.5).all())and bool((xb[-16:]==11.5).all())
                        assert bool((ob[:16]==17.25).all())and bool((ob[-16:]==17.25).all())
                        assert torch.equal(cb,cb0)and torch.equal(wb.view(torch.uint8),wb0.view(torch.uint8)),'S1 counter/scratch full backing changed'
                def invoke(arm):
                    x,out=work[arm];_,_,cb,wb,_,_=arenas[arm]
                    assert int(ext.run(LABELS.index(arm),packed,escape,x,out,wb[16:-16],cb[16:-16],table))==0,('launch declined',arm,m)
                def oracle_x(arm,fresh):
                    x,out=work[arm];assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
                    ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(arm,m,msg)
                    ref=fresh.double()@weights['unpooler'].double().T
                    rel=float((out.double()-ref).norm()/ref.norm().clamp(min=1e-30));row=float(((out.double()-ref).norm(dim=-1)/ref.norm(dim=-1).clamp(min=1e-30)).max())
                    assert rel<=4e-3 and row<=2e-2,(arm,m,rel,row)
                    return {'task':msg,'FP64_L2':rel,'worst_row':row,'current_X':'EXACT'}
                for arm in LABELS:
                    x,out=work[arm];checks=[]
                    for kind,fresh in [('normal',reference.generate_input(m,'unpooler',91301)['x']),('outliers',reference.generate_input(m,'unpooler',91303,dist='outliers')['x']),('zero',torch.zeros_like(x))]:
                        x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize();checks.append(dict(kind=kind,oracle=oracle_x(arm,fresh)));integrity();budget()
                    columns=basis_columns()
                    for start in range(0,len(columns),m):
                        selected=columns[start:start+m];fresh=torch.zeros_like(x);fresh[torch.arange(m,device=x.device),torch.tensor(selected,device=x.device)]=8
                        x.copy_(fresh);out.fill_(float('nan'));invoke(arm);torch.cuda.synchronize()
                        expected=(weights['unpooler'][:,selected].T*8).contiguous()
                        assert torch.equal(out.float(),expected.float())and torch.equal(x,fresh),('selected basis',arm,m,selected)
                    integrity();entry['validation'][arm]={'task':checks,'basis_columns':columns,'basis':'EXACT'};save();budget()
                live={}
                for arm in LABELS:
                    x,out=work[arm];x.copy_(fixtures[X_SEEDS[0]]);g=torch.cuda.CUDAGraph();graphs.append(g);live[arm]=g
                    with torch.cuda.graph(g):invoke(arm)
                integrity();last={}
                def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
                entry['fixture_hashes']={str(seed):hashlib.sha256(fixtures[seed].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()for seed in X_SEEDS}
                if 'shared_X_hashes'not in result:result['shared_X_hashes']=entry['fixture_hashes']
                else:assert result['shared_X_hashes']==entry['fixture_hashes'],'X bank differs across weights'
                entry['terminals']=[]
                for r in schedule(weight_index):
                    arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);torch.sum(flush,0,out=sink)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();live[arm].replay();b.record();torch.cuda.synchronize()
                    us=a.elapsed_time(b)*1000;assert 0<us<float('inf');row=dict(r,us=us,dist=xdist(r['seed']));entry['raw'].append(row);last[arm]=r['seed']
                    with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(row,m=m,weight_seed=weight_seed))+'\n');f.flush()
                    for checked in r.get('terminal_arms',[]):
                        expected=next(q['seed']for q in reversed(entry['raw'])if q['arm']==checked and q['mode']!='warmup');assert last[checked]==expected
                        entry['terminals'].append(dict(arm=checked,seed=expected,phase=r['mode'],oracle=oracle(checked,expected),before_any_replay_or_mutation=True));integrity()
                    budget()
                blocked={}
                for tag in('F1_pre','F1_post'):
                    for block in range(3):
                        group=[r for r in entry['raw']if r['mode']==tag and r['block']==block];blocked[tag+str(block)]={'arm':group[0]['arm'],'samples_us':[r['us']for r in group],'median_us':statistics.median(r['us']for r in group)}
                pairs=[]
                for order in range(2):
                    for seed in X_SEEDS:pairs.append({'seed':seed,'dist':xdist(seed),'order':order,'times':{r['arm']:r['us']for r in entry['raw']if r['mode']=='interleaved'and r['order']==order and r['seed']==seed}})
                entry['blocked']=blocked;entry['paired']=pairs
                entry['contrasts']={'paired_control_minus_new_us':statistics.median(r['times']['control']-r['times']['deltakey']for r in pairs),'paired_ratio':statistics.median(r['times']['control']/r['times']['deltakey']for r in pairs),'per_seed':{str(seed):{'dist':xdist(seed),'interleaved_control_minus_new_us':statistics.median(r['times']['control']-r['times']['deltakey']for r in pairs if r['seed']==seed),'blocked_control_minus_new_us':statistics.median(r['us']for r in entry['raw']if r['arm']=='control'and r['mode'].startswith('F1')and r['seed']==seed)-statistics.median(r['us']for r in entry['raw']if r['arm']=='deltakey'and r['mode'].startswith('F1')and r['seed']==seed)}for seed in X_SEEDS}};save()
                probe=torch.zeros_like(fixtures[X_SEEDS[0]]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns()[i]for i in range(m)],device=probe.device)]=8
                mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,'unpooler',91401)['x']),('outliers',reference.generate_input(m,'unpooler',91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,'unpooler',91407)['x'])]
                for kind,fresh in mutations:
                    for arm in LABELS:
                        x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                        message=oracle_x(arm,fresh)
                        if kind=='zero':assert not int(torch.count_nonzero(out))
                        integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
                for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
                entry['state']='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['state']='THREE_FRESH_WEIGHTS_SAME_BANK_COMPLETE_ROOT_REVIEW_REQUIRED';result['performance_scope']='Fixed exact current-selected unpoolerM16 template implementation comparison, not fresh policy or official score';phase('complete')
    except BaseException:
        result['state']='INVALID_NO_PERFORMANCE_ADMISSION';result['error']=traceback.format_exc();save();raise
    finally:
        cleanup=[]
        for g in graphs:
            try:torch.cuda.synchronize();g.reset();cleanup.append('RESET_AFTER_DRAIN')
            except BaseException as e:cleanup.append(repr(e));result['state']='INVALID_CLEANUP'
        result['cleanup']=cleanup;result['finished_at']=time.time();result['cost']['run_elapsed_seconds']=time.monotonic()-started;save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--prior',type=Path);p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);args=p.parse_args()
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output and args.review and args.prior
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
