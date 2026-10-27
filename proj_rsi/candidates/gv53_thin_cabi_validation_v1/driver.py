"""Two fixed GV53 cells, no full prepare/profiler/SASS campaign; root-only GPU run."""
import argparse,ast,hashlib,json,statistics,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
CAND=ROOT/'proj_rsi/candidates/gv53_keyed_half_v1'
CLEANUP_HOOK=None
LABELS=('pybind','cabi')
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
    for arm in ('pybind','cabi','pybind'):rows += [dict(arm=arm,seed=91100+i,mode='F1')for i in range(11)]
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
    assert sum(b[0]['arm']=='pybind'for b in blocks)==12
    for b in blocks:assert len({r['seed']for r in b})==1
    raw=[dict(arm=a,seed=i,mode='F1')for i,a in enumerate(LABELS)];last=dict(zip(LABELS,range(2)))
    assert len(terminal(last,raw,lambda a,s:'PASS',lambda:None))==2
    caught=[]
    for name in ('stale_seed','missing_arm','invalid_output','counter_mutation'):
        state=dict(last)
        if name=='stale_seed':state['pybind']=99
        if name=='missing_arm':state.pop('cabi')
        def oracle(a,s):assert name!='invalid_output';return 'PASS'
        def integrity():assert name!='counter_mutation'
        try:terminal(state,raw,oracle,integrity)
        except AssertionError:caught.append(name)
        else:raise AssertionError(name)
    assert len(codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()))==7
    assert len(basis_columns())==8
    return {'state':'PASS_FOCUSED_HARNESS_CPU_GPU_UNRUN','pins_verified':True,'fixed_cells':4,'timed_calls':324,'warmup_calls':40,'own_terminal_sites':16,'mutations':40,'terminal_negatives':caught,'original_codec_AST_functions':5,'codec_literals':2,'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'resource_gate':'legal residency only; actual registers/local/static recorded'}

def parse_sass(text):
    import re
    result={};name=None
    for line in text.splitlines():
        if 'Function :'in line:name=line.split('Function :',1)[1].strip();result[name]=[]
        elif name and (re.search(r'/\*[0-9a-f]{4,}\*/',line)or re.search(r'/\* 0x[0-9a-f]+ \*/',line)):result[name].append(' '.join(line.split()))
    return result

class Facade:
    def __init__(self,old,thin):self.old=old;self.thin=thin;self.native_arm=0;self.bound={};self.last_bound=None
    def run(self,api,W,esc,x,out,ws,cnt,table):
        if api==0:return self.old.run(self.native_arm,W,esc,x,out,ws,cnt,table)
        key=tuple(t.data_ptr()for t in(W,esc,x,out,ws,cnt))+(int(table),)
        if key not in self.bound:self.bound[key]=self.thin.bind(W,esc,x,out,ws,cnt,table)
        self.last_bound=self.bound[key];return self.last_bound.run(self.native_arm)
    def release_after_graph_reset(self):
        for b in self.bound.values():b.close_after_graph_reset()
        self.bound.clear();self.last_bound=None

def lifetime_probe(torch,thin,W,esc,table,weights,reference):
    import weakref,gc
    x=torch.zeros((1,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((1,4096),device='cuda',dtype=torch.bfloat16)
    ws=torch.full((65536,),17.25,device='cuda');cnt=torch.full((65536,),0x19a75d3,device='cuda',dtype=torch.int32)
    refs=[weakref.ref(z)for z in(x,out,ws,cnt)];bound=thin.bind(W,esc,x,out,ws,cnt,table);g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.graph(g,stream=stream):bound.run(0)
        assert bound.last_stream==stream.cuda_stream
        del x,out,ws,cnt;gc.collect();assert all(r()is not None for r in refs),'bound lifetime lost owners'
        for seed in(91501,91503):
            fresh=reference.generate_input(1,'pooler',seed)['x'];refs[0]().copy_(fresh);refs[1]().fill_(float('nan'))
            replay=torch.cuda.Stream();replay.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(replay):g.replay()
            torch.cuda.synchronize();ok,msg=reference.check_implementation({'x':fresh,'config':{'layer':'pooler','m':1,'n':4096,'k':8192}},refs[1](),weights);assert ok and torch.isfinite(refs[1]()).all(),msg
            assert torch.equal(refs[0](),fresh)and bool((refs[2]()==17.25).all())and bool((refs[3]()==0x19a75d3).all())
    finally:
        torch.cuda.synchronize();g.reset();bound.close_after_graph_reset()
    gc.collect();assert all(r()is None for r in refs),'retired lifetime leaked tensor owners'
    return {'strong_owners_after_caller_refs_dropped':True,'nondefault_capture_and_different_replay_streams':True,'fresh_current_X_own_output_checks':2,'graph_reset_before_owner_release':True,'weak_owners_released_after_drain':True}

def run(args):
    global CLEANUP_HOOK
    started=time.monotonic();verify();assert args.review.is_file(),'independent frozen review required'
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','review_sha256':sha(args.review),'source_sha256':sha(CAND/'submission.py'),'control_sha256':sha(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),'fixed_cells':{},'started_at':time.time(),'cost':{},'profiler_calls':0,'full_prepare_calls':0,'basis_columns_per_arm_cell':8,'resource_gate':'legal residency only; actual registers/local/static recorded'}
    graphs=[]
    def save():dump(args.output/'result.json',result)
    def budget():assert time.monotonic()-started<120,'inner120s admission exhausted'
    def phase(name):result['phase']=name;save();print(name,flush=True);budget()
    try:
        phase('torch_import_and_device')
        begin=time.monotonic();import torch
        result['cost']['torch_import_seconds']=time.monotonic()-begin
        import importlib.util,subprocess,ctypes,gc,weakref
        from adapter import ThinGV53,ThinError
        sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and any(x in props.name.upper()for x in('H100','H200'))
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'cc':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE'))}
        phase('load_exact_collected_pybind_and_compile_thin')
        manifest=verify();prior=ROOT/manifest['prior_directory'];old_result=json.loads((prior/'result.json').read_text())
        assert sha(prior/'result.json')==manifest['prior_result_sha256'] and sha(prior/'actual.so')==manifest['prior_SO_sha256']==old_result['build']['SO_sha256']
        begin=time.monotonic();spec=importlib.util.spec_from_file_location(manifest['prior_module_name'],prior/'actual.so')
        oldext=importlib.util.module_from_spec(spec);sys.modules[manifest['prior_module_name']]=oldext;spec.loader.exec_module(oldext)
        CLEANUP_HOOK=oldext.clear_blas_cache;result['cost']['prior_pybind_import_seconds']=time.monotonic()-begin
        cu=(CAND/'pilot.cu').read_text()+'\n'+(D/'bridge.cuh').read_text();(args.output/'thin.cu').write_text(cu)
        stages=[];nvcc='/usr/local/cuda/bin/nvcc'
        for name,cmd in [('native_compile',[nvcc,'-c','-std=c++17','-arch=sm_90a','-O3','-lineinfo','-Xptxas=-v','-Xcompiler','-fPIC',str(args.output/'thin.cu'),'-o',str(args.output/'thin.o')]),('host_link',[nvcc,'-shared','-cudart','shared','-Xcompiler','-fPIC',str(args.output/'thin.o'),'-o',str(args.output/'thin.so')])]:
            budget();begin=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
            stages.append({'name':name,'seconds':time.monotonic()-begin,'exit':p.returncode,'argv':cmd});(args.output/(name+'.stdout')).write_text(p.stdout);(args.output/(name+'.stderr')).write_text(p.stderr);assert p.returncode==0,p.stderr
        def sass(path,out):
            p=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],capture_output=True,text=True,timeout=15,check=True);out.write_text(p.stdout);return parse_sass(p.stdout)
        old_sass=sass(prior/'actual.so',args.output/'pybind.sass');new_sass=sass(args.output/'thin.so',args.output/'thin.sass')
        names=['_Z10gv_predxorILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G','_Z8gv_keyedILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G']
        identity={n:bool(old_sass[n])and old_sass[n]==new_sass[n]for n in names};assert all(identity.values()),identity
        begin=time.monotonic();thin=ThinGV53(args.output/'thin.so',sha(args.output/'thin.so'),torch,oldext.clear_blas_cache)
        result['cost']['thin_dlopen_seconds']=time.monotonic()-begin;result['build']={'stages':stages,'thin_SO_sha256':sha(args.output/'thin.so'),'prior_SO_sha256':sha(prior/'actual.so'),'native_identity':identity,'prior_compile_and_load_seconds_descriptive_only':old_result['cost']['compile_and_load_seconds'],'CXX_Torch_compile_calls':0,'native_compile_calls':1}
        fields=['registers','local','user_static','max_dynamic','max_threads','occupancy','device_reserved','arm_echo'];attrs={}
        for native_arm in(0,1):
            attrs[str(native_arm)]={api:{key:int(oldext.attrs(native_arm,j+1)if api=='pybind'else thin.attrs(native_arm,j+1,torch.cuda.current_device()))for j,key in enumerate(fields)}for api in LABELS}
            assert attrs[str(native_arm)]['pybind']==attrs[str(native_arm)]['cabi'],attrs[str(native_arm)]
            q=attrs[str(native_arm)]['cabi'];assert q['max_dynamic']>=17408 and q['max_threads']>=1024 and q['occupancy']>=1
        result['raw_attributes']=attrs;result['ABI_errors']={}
        try:thin.check(thin.lib.gv53_run(-1,*([None]*6),0,1,None))
        except ThinError as e:result['ABI_errors']['invalid_native_arm']=str(e)
        else:raise AssertionError('invalid nativearm accepted')
        result['ABI_errors']['unsafe_asynchronous_fault_injection']='NOT_ATTEMPTED; async errors remain checked by synchronization/own-terminal gates'
        facade=Facade(oldext,thin);ext=facade;save()
        phase('actual_task_weight_and_pack')
        begin=time.monotonic();weights=reference.generate_weights(reference.WEIGHTS_SEED)
        env={'torch':torch};exec(compile(ast.Module(body=codec_nodes((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()),type_ignores=[]),'<exact-original-codec-AST>','exec'),env)
        fr=env['_pack'](weights['pooler']);encoded=env['_c12'](fr,1);assert encoded is not None
        packed,table,esc6=encoded
        esc7=torch.cat((esc6,torch.full_like(esc6[:,:1],0)),1)
        escape=env['_gvesc'](esc7,256,(2,16,8,1,1,0));assert escape is not None
        immutable=[('pooler',weights['pooler'],weights['pooler'].clone()),('packed',packed,packed.clone()),('escape',escape,escape.clone())]
        result['pack']={'W_seed':reference.WEIGHTS_SEED,'actual_W_bit_reconstruction':'EXACT','packed_bytes':packed.numel(),'escape_words':escape.numel(),'geometry':[2,16,8,1,1,0],'packing_functions':'Exact original source AST; prepare and extension factories not executed'}
        result['cost']['task_weight_pack_seconds']=time.monotonic()-begin
        flush=torch.ones(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda',dtype=torch.float32)
        def immutables():
            for name,t,copy in immutable:assert torch.equal(t.view(torch.uint8),copy.view(torch.uint8)),('immutable changed',name)
        for native_arm in(0,1):
            for m in (1,8):
                phase('native'+str(native_arm)+'_pooler_M'+str(m));facade.native_arm=native_arm;config={'layer':'pooler','m':m,'n':4096,'k':8192}
                entry=result['fixed_cells'][str(native_arm)+':'+str(m)]={'config':config,'raw':[],'validation':{},'terminals':{},'mutations':[]};work={};arenas={}
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
                    assert int(ext.run(LABELS.index(arm),packed,escape,x,out,wb[16:-16],cb[16:-16],table))==0,('launch declined',arm,m)
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
                    capture_stream=torch.cuda.Stream();capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(g,stream=capture_stream):invoke(arm)
                    if arm=='cabi':assert facade.last_bound.last_stream==capture_stream.cuda_stream
                    assert capture_stream.cuda_stream!=torch.cuda.default_stream().cuda_stream
                    entry.setdefault('capture_streams',{})[arm]=int(capture_stream.cuda_stream)
                integrity();last={};replay_stream=torch.cuda.Stream();entry['replay_stream']=int(replay_stream.cuda_stream)
                assert all(replay_stream.cuda_stream!=x for x in entry['capture_streams'].values())
                def oracle(arm,seed):return oracle_x(arm,fixtures[seed])
                for i,r in enumerate(schedule()):
                    if i==43:entry['terminals']['F1']=terminal(last,entry['raw'],oracle,integrity);save()
                    arm=r['arm'];x,out=work[arm];x.copy_(fixtures[r['seed']]);torch.sum(flush,0,out=sink)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);replay_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(replay_stream):a.record();live[arm].replay();b.record()
                    torch.cuda.synchronize()
                    us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                    row=dict(r,us=us);entry['raw'].append(row);last[arm]=r['seed']
                    with(args.output/'raw.jsonl').open('a')as f:f.write(json.dumps(dict(row,m=m,native_arm=native_arm))+'\n');f.flush()
                    budget()
                entry['terminals']['paired']=terminal(last,entry['raw'],oracle,integrity)
                assert torch.equal(work['pybind'][1].view(torch.uint8),work['cabi'][1].view(torch.uint8))
                entry['paired_terminal_interfaces_bitwise_equal']=True
                f1=entry['raw'][10:43];med=[statistics.median(r['us']for r in f1[i*11:(i+1)*11])for i in range(3)]
                paired=[{r['arm']:r['us']for r in entry['raw'][43+2*i:45+2*i]}for i in range(24)]
                entry['F1_medians_us']=dict(zip(('pybind_before','cabi','pybind_after'),med));entry['paired']=paired
                entry['contrasts']={'paired_median_control_minus_new_us':statistics.median(r['pybind']-r['cabi']for r in paired),'F1_endpoint_average_control_minus_new_us':(med[0]+med[2])/2-med[1],'paired_new_wins':sum(r['pybind']>r['cabi']for r in paired)};save()
                probe=torch.zeros_like(fixtures[91000]);rr=torch.arange(m,device=probe.device);probe[rr,torch.tensor([basis_columns()[i]for i in range(m)],device=probe.device)]=8
                mutations=[('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,'pooler',91401)['x']),('outliers',reference.generate_input(m,'pooler',91403,dist='outliers')['x']),('boundary_probes',probe),('fresh_normal',reference.generate_input(m,'pooler',91407)['x'])]
                for kind,fresh in mutations:
                    for arm in LABELS:
                        x,out=work[arm];x.copy_(fresh);out.fill_(float('nan'));live[arm].replay();torch.cuda.synchronize()
                        message=oracle_x(arm,fresh)
                        if kind=='zero':assert not int(torch.count_nonzero(out))
                        integrity();entry['mutations'].append(dict(arm=arm,kind=kind,oracle=message,NaN_output_overwritten=True));budget()
                    assert torch.equal(work['pybind'][1].view(torch.uint8),work['cabi'][1].view(torch.uint8)),('interface bitwise mismatch',kind)
                entry['five_mutation_fixtures_interfaces_bitwise_equal']=True
                for g in live.values():torch.cuda.synchronize();g.reset();graphs.remove(g)
                facade.release_after_graph_reset();entry['native_arm']=native_arm;entry['state']='VALID_INTERFACE_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION';entry['cleanup']='RESET_AFTER_DRAIN';save()
        result['lifetime_probe']=lifetime_probe(torch,thin,packed,escape,table,weights,reference);immutables();save()
        result['state']='FOUR_INTERFACE_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED';result['performance_scope']='Same-native-kernel PyBind versus C ABI validation only; no kernel speedup, policy or score claim';phase('complete')
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
