"""One fixed M8 quarter-stream cell, no full prepare/profiler/SASS campaign; root-only GPU run."""
import argparse,ast,hashlib,json,statistics,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
CAND=D.parent/"gv53_quarter_stream_v1"
CLEANUP_HOOK=None
LABELS=('control','quarter')
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
    rows=[]
    for arm in LABELS:rows += [dict(arm=arm,seed=91000,mode='warmup')for _ in range(5)]
    for arm in ('control','quarter','control'):rows += [dict(arm=arm,seed=91100+i,mode='F1')for i in range(11)]
    for i in range(24):
        rows += [dict(arm=arm,seed=91200+i,mode='paired',pair=i)for arm in (LABELS if i%2==0 else LABELS[::-1])]
    assert len(rows)==91
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
    m=verify();rows=schedule();assert sum(r['mode']!='warmup'for r in rows)==81
    blocks=[rows[43+2*i:45+2*i]for i in range(24)]
    assert sum(b[0]['arm']=='control'for b in blocks)==12
    for b in blocks:assert len({r['seed']for r in b})==1
    raw=[dict(arm=a,seed=i,mode='F1')for i,a in enumerate(LABELS)];last=dict(zip(LABELS,range(2)))
    assert len(terminal(last,raw,lambda a,s:'PASS',lambda:None))==2
    caught=[]
    for name in ('stale_seed','missing_arm','invalid_output','counter_mutation'):
        state=dict(last)
        if name=='stale_seed':state['control']=99
        if name=='missing_arm':state.pop('quarter')
        def oracle(a,s):assert name!='invalid_output';return 'PASS'
        def integrity():assert name!='counter_mutation'
        try:terminal(state,raw,oracle,integrity)
        except AssertionError:caught.append(name)
        else:raise AssertionError(name)
    assert len(codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()))==7
    assert len(basis_columns())==8
    return {'state':'PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN','pins_verified':True,'fixed_cells':1,'timed_calls':81,'warmup_calls':10,'own_terminal_sites':4,'mutations':10,'terminal_negatives':caught,'original_codec_AST_functions':5,'codec_literals':2,'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'resource_gate':'legal residency only; actual registers/local/static recorded'}

def run(args):
    global CLEANUP_HOOK
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'pilot.cu'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'resource_gate':'legal residency only; actual registers/local/static recorded'}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<90,'inner90s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        import torch
        from torch.utils.cpp_extension import load_inline
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('compile_two_functions')
        cu=(CAND/'pilot.cu').read_text()+'\n'+(D/'binding.cu').read_text()
        (args.output/'actual.cu').write_text(cu)
        digest=hashlib.sha256(cu.encode()).hexdigest();begin=time.monotonic()
        ext=load_inline(name='gv53_quarter_'+digest[:20],cpp_sources=DECL,cuda_sources=cu,functions=['run','attrs','clear_blas_cache'],extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],verbose=True)
        CLEANUP_HOOK=ext.clear_blas_cache
        result['cost']['compile_and_load_seconds']=time.monotonic()-begin
        __import__('shutil').copy2(ext.__file__,args.output/'actual.so')
        result['build']={'CUDA_sha256':digest,'CPP_declarations_sha256':hashlib.sha256(DECL.encode()).hexdigest(),'SO_sha256':sha(ext.__file__),'CUDA_flags':['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],'native_symbols':['gv_predxor<1,1,1,2,16,8,1,1,0,3>','gv_quarter<1,1,1,2,16,8,1,1,0,3>'],'field_query_calls':16}
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
        sys.path.insert(0,str(CAND));from transport import pack_bank,unpack_group
        old_words=escape.detach().cpu().tolist()
        begin_transport=time.monotonic();new_words=pack_bank(old_words);assert new_words is not None
        assert len(new_words)==len(old_words)==131072
        record_count=0
        for offset in range(0,len(old_words),32):
            accepted=[int(v)&0xffffffff for v in old_words[offset:offset+32]if (int(v)&0xffffffff)!=0xffffffff]
            expected=sorted(accepted,key=lambda v:(v>>20,(v>>13)&3))
            assert unpack_group(new_words[offset:offset+32])==expected,('actual accepted escape mapping',offset)
            record_count+=len(accepted)
        quarter_escape=torch.tensor([v if v<2**31 else v-2**32 for v in new_words],dtype=torch.int32,device=escape.device)
        assert quarter_escape.data_ptr()!=escape.data_ptr() and quarter_escape.numel()==escape.numel()
        escapes={'control':escape,'quarter':quarter_escape}
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
            assert len({v[0].data_ptr()for v in work.values()})==2 and len({v[1].data_ptr()for v in work.values()})==2
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
                x,out=work[arm];x.copy_(fixtures[91000]);g=torch.cuda.CUDAGraph();graphs.append(g);live[arm]=g
                with torch.cuda.graph(g):invoke(arm)
            integrity();last={}
            def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
            for i,r in enumerate(schedule()):
                if i==43:entry['terminals']['F1']=terminal(last,entry['raw'],oracle,integrity);save()
                arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);torch.sum(flush,0,out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();live[arm].replay();b.record();torch.cuda.synchronize()
                us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                row=dict(r,us=us);entry['raw'].append(row);last[arm]=r['seed']
                with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(row,m=m))+'\n');f.flush()
                budget()
            entry['terminals']['paired']=terminal(last,entry['raw'],oracle,integrity)
            f1=entry['raw'][10:43];med=[statistics.median(r['us']for r in f1[i*11:(i+1)*11])for i in range(3)]
            paired=[{r['arm']:r['us']for r in entry['raw'][43+2*i:45+2*i]}for i in range(24)]
            entry['F1_medians_us']=dict(zip(('control_before','quarter','control_after'),med));entry['paired']=paired
            entry['contrasts']={'paired_median_control_minus_new_us':statistics.median(r['control']-r['quarter']for r in paired),'F1_endpoint_average_control_minus_new_us':(med[0]+med[2])/2-med[1],'paired_new_wins':sum(r['control']>r['quarter']for r in paired)};save()
            probe=torch.zeros_like(fixtures[91000]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns()[i]for i in range(m)],device=probe.device)]=8
            mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,'pooler',91401)['x']),('outliers',reference.generate_input(m,'pooler',91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,'pooler',91407)['x'])]
            for kind,fresh in mutations:
                for arm in LABELS:
                    x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                    message=oracle_x(arm,fresh)
                    if kind=='zero':assert not int(torch.count_nonzero(out))
                    integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
            for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
            entry['state']='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['state']='ONE_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED';result['performance_scope']='Fixed GV53 implementation comparison only, not selected policy or official score';phase('complete')
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
