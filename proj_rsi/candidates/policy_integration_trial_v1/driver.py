"""Pinned PW45/GVpredxor fullpolicy plus matched fixed-template lab; root-only GPU execution.
--cpu-check uses only stdlib/AST. Fixed mechanism arms are separate from actual policy; no clock reset or selector mutation.
"""
import argparse, ast, hashlib, importlib.util, json, re, statistics, subprocess, sys, threading, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
TRIAL=ROOT/'proj_rsi/candidates/policy_integration_trial_v1'
PINS={'pw45':'835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3','gv_predxor_pw45_v2':'501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'}
sys.path.insert(0,str(TRIAL))
import support as S
import gv_support as G
LABELS=tuple(PINS)
LOADS={'decode1':1,'decode16':16,'decode32':32,'verify1':8,'verify16':128,'verify32':256}
LAYERS=['qkv_gate','o','pooler','unpooler','router']
MS=[1,5,7,8,9,16,32,33,128,130,256,512,1000]
OFFICIAL=[{'m':1,'layer':'qkv_gate','dist':'normal'},{'m':7,'layer':'pooler','dist':'normal'},{'m':33,'layer':'o','dist':'normal'},{'m':130,'layer':'unpooler','dist':'normal'},{'m':5,'layer':'unpooler','dist':'normal'},{'m':8,'layer':'router','dist':'normal'},{'m':9,'layer':'router','dist':'outliers'},{'m':512,'layer':'qkv_gate','dist':'outliers'},{'m':1000,'layer':'unpooler','dist':'outliers'}]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def verify():
    manifest=json.loads((TRIAL/'manifest.json').read_text())
    for rel,pin in manifest['files'].items():assert sha(ROOT/rel)==pin,('source drift',rel)
    for label,pin in PINS.items():assert sha(ROOT/f'proj_rsi/candidates/{label}/submission.py')==pin
    return manifest

def tensor_id(t):return(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())
def tensors(obj,path=''):
    if hasattr(obj,'untyped_storage'):yield path,obj
    elif isinstance(obj,dict):
        for k,v in obj.items():yield from tensors(v,path+'/'+str(k))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj):yield from tensors(v,path+'/'+str(i))

def counter_values(t):
    assert str(t.dtype)=='torch.int32'
    return t.as_strided((t.untyped_storage().nbytes()//4,),(1,),0).detach().cpu().tolist()
def counter_arena(state):
    result={}
    for name,t in tensors(state):
        if 'cnt' in name and t.numel():
            key=tensor_id(t)
            if key not in result:result[key]={'name':name,'tensor':t,'values':counter_values(t)}
    return result

def pick(mod,state,layer,m,n):
    rp=state.get('rq',{}).get((layer,m))
    if rp is not None and rp['n']==n:
        names=mod.RQPROD2 if 'mod' in rp else mod.RQPROD
        name=names[rp['i']];v=mod.RQV[mod.RQI[name]]
        return {'family':'RQ','name':name,'variant':list(v),'KS':rp['KS']},rp
    p=state['sel'].get((layer,mod._bk(m)))
    origin='sel'
    if p is None or p[4]!=n:p=state['gv'].get((layer,mod._bk(m)));origin='gv'
    if p is not None and p[4]==n:
        i=p[0]
        if i==-7:return {'family':'router','variant':i,'id':p[1],'origin':origin},p
        if i>=1000:return {'family':'RR','id':i,'origin':origin},p
        return {'family':'GV','id':i,'variant':list(mod.GVI[i]),'origin':origin},p
    return {'family':'LT' if mod._cublas(layer,m) or(layer=='router' and m>32) else 'V12','lt':state['lt'].get((layer,m))},None

def router_cases(mod):
    # Bind slot and geometry to the actual generated switch passed to load_inline.
    source=mod._rwsrc()
    found={}
    for match in re.finditer(r'case (\d+): return (rw|rt)_go<([^>]+)>\(xp, h, l, o, m, N, K, c, (\d+), st\);',source):
        ident=int(match[1]);assert ident not in found
        found[ident]=(match[2],tuple(int(v.strip()) for v in match[3].split(',')),int(match[4]))
    assert found
    return found

def counter_plan(mod,state,layer,m,n,k):
    info,p=pick(mod,state,layer,m,n);family=info['family'];plan={}
    def use(t,mode,changes=None,reset_indices=None):
        key=tensor_id(t);entry={'mode':mode,'offset':t.storage_offset(),'changes':changes or {},'reset_indices':reset_indices}
        assert key not in plan;plan[key]=entry
    if family=='RQ':
        fl=info['variant'][7]
        if fl&2:use(p['cnt'],'reset0',reset_indices=list(range(2*(n//128)*(2 if fl&16 else 1))))
    elif family=='V12':
        indices=list(range(2048,2048+n//16)) if layer=='router' and m<=32 and mod.RTR else list(range(((n+15)//16)*((m+31)//32)))
        use(state['cnt'],'reset0',reset_indices=indices)
    elif family=='GV':
        fmt,mt,st,wr,wk,ku,s,rt,xs,v=info['variant']
        if(v>>27)&5:
            assert not(v&(1<<28)),'unreviewed PDL counter path'
            t=state.get('p1cnt',{}).get(p[0]);assert t is not None,'warm selected path before snapshot'
            tiles=((m+8*mt-1)//(8*mt))*(n//(16*wr*rt))
            use(t,'progress',{2*i:s for i in range(tiles)}|{2*i+1:1 for i in range(tiles)})
        elif s>1 and(v&4) and ((v>>26)&15)==0:use(state['gvcnt'],'reset0',reset_indices=list(range(((m+8*mt-1)//(8*mt))*(n//(16*wr*rt)))))
        # S1 and sm90 cluster branches do not use global counters.
    elif family=='router':
        ident=info['id']
        if ident<100:raise AssertionError('HOLD unreviewed legacy RSK counter range')
        else:
            kind,geom,slot=router_cases(mod)[ident]
            if kind=='rt':
                kr,nbuf,toks,wn,minb,mode=geom
                assert mode==4,('unreviewed RTK mode',mode)
            else:
                kr,nstg,nbuf,toks,wn,minb,mode,cl=geom
                assert cl==1 and mode in(0,4,12),('unreviewed RW mode',mode,cl)
            if mode&4:
                tiles=(n//64)*((m+toks-1)//toks)
                use(state['rw_cnt'],'progress',{2*(slot*64+i):k//kr for i in range(tiles)})
    return info,plan

def counter_check(before,after,plan,calls):
    assert set(before)==set(after),'counter allocation/storage alias changed in checked interval'
    for key,a in before.items():
        b=after[key];rule=plan.get(key,{'mode':'unchanged','offset':0,'changes':{}})
        expected=list(a['values'])
        expected=S.counter_expected(expected,rule['mode'],rule.get('offset',0),rule.get('reset_indices') or (),rule.get('changes'),calls)
        assert b['values']==expected,{'counter':a['name'],'rule':rule,'calls':calls,'before':a['values'],'expected':expected,'actual':b['values']}

def rq_targets(sass):
    found={}
    for table in sass.split('Function : ')[1:]:
        symbol=table.splitlines()[0].strip()
        match=re.search(r'rqILi128ELi5ELi1ELi0ELi0ELi0ELi(2|3|65538|65539)EE',symbol)
        if match and 'HGMMA.' in table:
            flag=int(match[1]);assert flag not in found
            found[flag]={'symbol':symbol,'table':table,'stream':[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]}
    return found

def rq_resource_gate(resource,symbol):
    start=resource.find(symbol);assert start>=0,('missing actual resource symbol',symbol)
    block=resource[start+len(symbol):].split('Function',1)[0]
    assert re.search(r'REG\s*[:=]\s*168\b',block),('register gate',symbol,block[:500])
    assert re.search(r'LOCAL\s*[:=]\s*0\b',block),('local/spill gate',symbol,block[:500])

def eh_physical(sass):
    env={'re':re}
    source=(ROOT/'proj_rsi/candidates/rs32_eh_backport/preflight.py').read_text()
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('tables','physical'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-pure-EH-physical>','exec'),env)
    return env['physical'](sass)

def actual_rq_gate(sass,resources,expect_eh):
    canonical=rq_targets((ROOT/'proj_rsi/candidates/rs32_eh_backport/sass.txt').read_text())
    actual=rq_targets(sass)
    if expect_eh:assert set(actual)=={2,3,65538,65539},'actual EH/control function set incomplete'
    for flag,entry in actual.items():
        assert entry['stream']==canonical[flag]['stream'],('actual main PC/predicate/opcode drift',flag)
        rq_resource_gate(resources,entry['symbol'])
    physical=None
    if expect_eh:physical=eh_physical('\n'.join('\n Function : '+actual[flag]['table'] for flag in sorted(actual)))
    return {'flags':sorted(actual),'resources':'168_REG_0_LOCAL_ALL_TARGETS','EH_physical':physical}

def router_cpu_proof():
    import types
    proof={}
    allowed={'RWCU','RWV','RWST','RTV','RWLAD','RWALL','RSPALL','RSP'}
    for label in LABELS:
        source=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text();chosen=[]
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in allowed for t in node.targets):chosen.append(node)
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id in ('RWV','RTV') and node.value.func.attr=='update':chosen.append(node)
            elif isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='RWLAD':chosen.append(node)
            elif isinstance(node,ast.FunctionDef) and node.name in ('_rwwg','_rwsrc'):chosen.append(node)
        env={};exec(compile(ast.Module(body=chosen,type_ignores=[]),'<pure-router-source-factory>','exec'),env)
        cases=router_cases(types.SimpleNamespace(**env))
        assert set(cases)=={p[1] for p in env['RSP'] if p[1]>=100}
        for ident,(kind,geom,slot) in cases.items():
            mapping=env['RTV'] if kind=='rt' else env['RWV']
            assert geom==mapping[ident] and slot==sorted(mapping).index(ident)+(64 if kind=='rt' else 0)
        proof[label]={str(ident):{'kind':kind,'geometry':geom,'source_generated_slot':slot} for ident,(kind,geom,slot) in cases.items()}
    return proof

def cpu_check(output=None):
    verify()
    import focused_check
    return focused_check.check()

def run(args):
    verify();assert args.review and Path(args.review).is_file(),'independent frozen driver review required'
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    args.output.mkdir(parents=True,exist_ok=False)
    result={'_output':str(args.output),'state':'RUNNING_NO_PERFORMANCE_ADMISSION','sources':PINS,'review_sha256':sha(args.review),'prepare':{},'jit':[],'jit_errors':[],'checks':[],'shapes':{},'raw':[]}
    lock=threading.RLock();graphs_live=[]
    def save():
        with lock:dump(args.output/'result.json',result)
    def phase(name):result['phase']=name;save();print(name,flush=True)
    modules={};states={};immutable=[];weights=None
    original_weights=[]
    def immutable_check():
        for name,t,original in original_weights+immutable:
            assert torch.equal(t.view(torch.uint8),original.view(torch.uint8)),('immutable changed',name)
    registry={};ext_geometries={}
    try:
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        assert props.major==9 and props.minor==0 and props.multi_processor_count>=128 and any(x in props.name.upper() for x in ('H100','H200')),'HOLD incompatible hardware before weights/prepare'
        result['hardware']={'name':props.name,'SM_count':props.multi_processor_count,'compute_capability':[props.major,props.minor],'UUID':str(getattr(props,'uuid','UNAVAILABLE_FROM_TORCH_PROPERTIES')),'before_weights_and_prepare':True}
        weights=reference.generate_weights(reference.WEIGHTS_SEED);original_weights.extend((key,t,t.clone()) for key,t in weights.items());save()
        for label in LABELS:
            phase('prepare_'+label)
            spec=importlib.util.spec_from_file_location('stack_frozen_'+label,ROOT/f'proj_rsi/candidates/{label}/submission.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules[label]=mod
            original_build=mod.load_inline
            def checked_build(*a,_build=original_build,_label=label,**kw):
                name=kw.get('name',a[0] if a else None);original_cu=str(kw['cuda_sources']);cu=original_cu
                rq='struct RqP' in cu;gv='static int64_t gvl(' in cu;dispatch=[];build_kw=dict(kw);build_a=list(a)
                if rq:
                    cu,dispatch=S.append_attrs(cu);build_kw['cuda_sources']=cu
                    cpp=build_kw['cpp_sources'];declaration='int64_t pw45_labattrs(int64_t,int64_t);'
                    build_kw['cpp_sources']=cpp+'\n'+declaration if isinstance(cpp,str) else list(cpp)+[declaration]
                    build_kw['functions']=list(build_kw['functions'])+['pw45_labattrs']
                    name='pw45_gate_'+hashlib.sha256((cu+str(build_kw['cpp_sources'])+repr(build_kw.get('extra_cuda_cflags'))).encode()).hexdigest()[:24]
                    if build_a:build_a[0]=name
                    else:build_kw['name']=name
                if gv:
                    cu,helper_info=G.append_helpers(original_cu,mod,_label)
                    build_kw['cuda_sources']=cu
                    cpp=build_kw['cpp_sources'];decl=G.DECL
                    build_kw['cpp_sources']=cpp+'\n'+decl if isinstance(cpp,str) else list(cpp)+[decl]
                    build_kw['functions']=list(build_kw['functions'])+['gvpred_attrs','gvpred_oldrun']
                    name='gvpred_lab_'+_label+'_'+hashlib.sha256((cu+str(build_kw['cpp_sources'])+repr(build_kw.get('extra_cuda_cflags'))).encode()).hexdigest()[:20]
                    if build_a:build_a[0]=name
                    else:build_kw['name']=name
                digest=hashlib.sha256(cu.encode()).hexdigest()
                with lock:
                    assert name not in registry or registry[name]==digest,('extension name collision',name)
                    registry[name]=digest
                ext=_build(*build_a,**build_kw)
                sass=subprocess.check_output(['cuobjdump','--dump-sass',str(ext.__file__)],text=True,timeout=60)
                resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(ext.__file__)],text=True,timeout=60)
                prefix=_label+'.'+str(name)
                (args.output/(prefix+'.sass')).write_text(sass);(args.output/(prefix+'.resources')).write_text(resources)
                (args.output/(prefix+'.original.cu')).write_text(original_cu);(args.output/(prefix+'.gated.cu')).write_text(cu)
                __import__('shutil').copy2(ext.__file__,args.output/(prefix+'.so'))
                actual_gate=None;live=[]
                if rq:
                    actual_gate=S.canonical_gate(original_cu,sass,resources)
                    with lock:ext_geometries[id(ext)]=set(dispatch)
                    for index,geometry in enumerate(dispatch):
                        attr={key:int(ext.pw45_labattrs(index,field)) for key,field in [('regs',1),('local',2),('shared',3),('occupancy',4),('max_dynamic',5),('max_threads',6)]}
                        live.append(S.check_attrs(geometry,attr))
                gv_gate=None
                if gv:gv_gate=G.actual_gate(original_cu,cu,sass,resources,ext,mod,_label)
                with lock:
                    result['jit'].append({'label':_label,'name':name,'original_source_sha256':hashlib.sha256(original_cu.encode()).hexdigest(),'source_sha256':digest,'so_sha256':sha(ext.__file__),'SASS_sha256':sha(args.output/(prefix+'.sass')),'before_return_to_prepare_math':True,'RQ_gate':actual_gate,'live_RQ_attrs':live,'GV_gate':gv_gate});save()
                return ext
            checked_build_body=checked_build
            def checked_build(*a,_body=checked_build_body,_label=label,**kw):
                try:return _body(*a,**kw)
                except BaseException:
                    with lock:result['jit_errors'].append({'label':_label,'error':traceback.format_exc()});save()
                    raise
            mod.load_inline=checked_build
            build_threads=[];original_start=mod._rqstart
            def record_start(*a,_start=original_start,**kw):
                thread=_start(*a,**kw);build_threads.append(thread);return thread
            mod._rqstart=record_start
            start=time.time();state=states[label]=mod.prepare(weights);torch.cuda.synchronize()
            for thread in build_threads:thread.join(timeout=max(0,300-(time.time()-mod._T0)))
            assert not any(thread.is_alive() for thread in build_threads),'prepare background build still live after300s'
            assert not any(error['label']==label for error in result['jit_errors']),('actual JIT/build gate failed',label)
            actual_args={tuple(a) for record in result['jit'] if record['label']==label for a in (record.get('RQ_gate') or {}).get('args',[])}
            canonical=set(S.streams((S.CAND/'sass.txt').read_text()))
            expected=canonical
            assert actual_args==expected,('complete actual RQ functions must gate before serving',label,len(actual_args),len(expected))

            result['prepare'][label]={'wall_seconds':time.time()-start,'import_clock_elapsed':time.time()-mod._T0,'budget_clock_reset':False,'serving_CPU_fallback':bool(state.get('cpu')),'build_threads_joined':len(build_threads)}
            assert not state.get('cpu') and time.time()-mod._T0<300,('prepare overrun',label)
            # Prevent silent RQ fallback during serving; preparation retains its original decline logic.
            original_rq=mod._rqcall
            def guarded(*a,_fn=original_rq,_label=label,**kw):
                code=_fn(*a,**kw);assert code==0,('serving RQ declined',_label,code);return code
            mod._rqcall=guarded
            original_gv=mod._gvcall
            def guarded_gv(*a,_fn=original_gv,_label=label,**kw):
                code=_fn(*a,**kw);assert code==0,('serving GV/router declined',_label,code);return code
            mod._gvcall=guarded_gv
            original_ext=mod._ext
            def serving_ext(_fn=original_ext):return S.ServingStatus(_fn())
            mod._ext=serving_ext
            # Full physical sink backing is dormant for admitted production paths.
            sink_seen=set()
            for name,t in tensors(state):
                if 'sink' in name and t.numel() and tensor_id(t) not in sink_seen:
                    sink_seen.add(tensor_id(t));physical=t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)
                    immutable.append((label+name+' FULL_PHYSICAL_SINK',physical,physical.clone()))
            seen=set()
            for name,t in tensors(state):
                if not t.numel() or any(word in name for word in ('cnt','ws','sink','prof')):continue
                view=(tensor_id(t),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                if view not in seen:seen.add(view);immutable.append((label+name,t,t.clone()))
            save()
        # Every state must own distinct physical counter/workspace arenas.
        allocations={}
        for label,state in states.items():
            for name,t in tensors(state):
                if t.numel() and any(word in name for word in ('cnt','ws')):
                    identity=tensor_id(t)
                    assert identity not in allocations or allocations[identity]==label,('cross-policy mutable alias',name)
                    allocations[identity]=label
        # Independent task oracles on all65 legacy normal cases AND exact9 official distributions.
        for label in LABELS:
            mod,state=modules[label],states[label]
            cases=[{'m':m,'layer':layer,'dist':'normal'} for layer in LAYERS for m in MS]+OFFICIAL
            for index,case in enumerate(cases):
                phase('correctness_'+label+'_'+str(index))
                data=reference.generate_input(case['m'],case['layer'],6151+index,dist=case['dist']);x_before=data['x'].clone()
                # Warm call admits lazy selected counter allocation before the checked interval.
                mod.kernel_fn(data,state);torch.cuda.synchronize()
                info,plan=counter_plan(mod,state,case['layer'],case['m'],data['config']['n'],data['config']['k'])
                before=counter_arena(state);out=mod.kernel_fn(data,state);torch.cuda.synchronize()
                ok,message=reference.check_implementation(data,out,weights);assert ok,(label,case,message)
                assert torch.equal(data['x'].view(torch.uint8),x_before.view(torch.uint8))
                counter_check(before,counter_arena(state),plan,1);immutable_check()
                result['checks'].append({'label':label,'case':case,'official':index>=65,'oracle':message,'pick':info,'counter_contract':'PASS'});save()
        # Full o32 K=1024 basis and FP64 on each ACTUAL serving policy.
        result['actual_o32_basis']={}
        for label in LABELS:
            mod,state=modules[label],states[label];config={'layer':'o','m':32,'n':8192,'k':1024}
            warm=reference.generate_input(32,'o',71501);mod.kernel_fn(warm,state);torch.cuda.synchronize()
            info,plan=counter_plan(mod,state,'o',32,8192,1024);before=counter_arena(state);calls=[0]
            def actual_fn(x,out):
                calls[0]+=1;xb=x.clone();out.copy_(mod.kernel_fn({'x':x,'config':config},state));assert torch.equal(x,xb);return 0
            gen=torch.Generator(device=weights['o'].device).manual_seed(71503)
            ok,msg=mod._rqval(actual_fn,weights['o'],32,8192,gen,1e-2);assert ok,(label,msg)
            torch.cuda.synchronize();counter_check(before,counter_arena(state),plan,calls[0]);immutable_check()
            result['actual_o32_basis'][label]={'pick':info,'all1024_basis':'EXACT','FP64_normal_outlier':msg,'calls':calls[0]};save()
        immutable_check()
        flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
        orders=[(0,1),(1,0)]
        for load,m in LOADS.items():
            for layer in LAYERS:
                name=load+'/'+layer;phase('timing_'+name)
                config={'m':m,'n':reference.out_features(layer,m),'k':reference.LAYERS[layer][1],'layer':layer}
                entry=result['shapes'][name]={'picks':{},'terminal_oracles':[],'mutations':[]}
                work={};counts={label:0 for label in LABELS};baseline={};plans={}
                fixtures={seed:reference.generate_input(m,layer,seed)['x'] for seed in range(9000,9064)}
                for label in LABELS:
                    mod,state=modules[label],states[label];x=fixtures[9000].clone()
                    for _ in range(2):mod.kernel_fn({'x':x,'config':config},state)
                    torch.cuda.synchronize();info,plan=counter_plan(mod,state,layer,m,config['n'],config['k'])
                    entry['picks'][label]=info;plans[label]=plan
                    if info['family']=='RQ':
                        rp=state['rq'][(layer,m)];ext=rp.get('mod',mod._rqmod[0])
                        query_out=torch.empty((m,config['n']),device='cuda',dtype=torch.bfloat16)
                        attrs=[int(ext.rqrun(rp['i'],rp['lay'],x,query_out,rp['esc'],rp['cnt'],rp['KS'],rp['tab'],0,state['rq_e'],state['rq_sink'],code)) for code in (1,2,3)]
                        assert 0<attrs[0]<=168 and attrs[1]==0,('selected RQ resources',label,info,attrs)
                        v=info['variant'];nt,st,fl=v[1],v[2],v[7]
                        assert attrs[2]==1024+st*(nt*128+(6144 if fl&16 else 12288))+24*st+16
                        entry['picks'][label]['actual_regs_local_shared']=attrs
                    baseline[label]=counter_arena(state)
                    g=torch.cuda.CUDAGraph();graphs_live.append(g)
                    with torch.cuda.graph(g):out=mod.kernel_fn({'x':x,'config':config},state)
                    torch.cuda.synchronize();counter_check(baseline[label],counter_arena(state),plan,0)
                    work[label]=(x,out,g)
                assert len({item[0].data_ptr() for item in work.values()})==2
                assert len({item[1].data_ptr() for item in work.values()})==2
                last={}
                def oracle(label,fresh,kind):
                    x,out,g=work[label]
                    assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
                    ok,message=reference.check_implementation({'x':fresh,'config':config},out,weights)
                    assert ok and bool(torch.isfinite(out).all()),(name,label,kind,message)
                    counter_check(baseline[label],counter_arena(states[label]),plans[label],counts[label]);immutable_check()
                    return message
                def one(label,seed,mode):
                    x,out,g=work[label];x.copy_(fixtures[seed]);torch.sum(flush,0,out=sink)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();g.replay();b.record();torch.cuda.synchronize();counts[label]+=1
                    us=a.elapsed_time(b)*1000;assert 0<us<float('inf')
                    row={'shape':name,'label':label,'seed':seed,'mode':mode,'us':us};result['raw'].append(row)
                    with (args.output/'raw.jsonl').open('a') as journal:journal.write(json.dumps(row)+'\n');journal.flush()
                    last[label]=seed;return us
                def terminal(kind):
                    # Read exact terminal outputs before any graph replay or X/out mutation.
                    for label in LABELS:entry['terminal_oracles'].append({'label':label,'phase':kind,'seed':last[label],'oracle':oracle(label,fixtures[last[label]],kind)})
                    save()
                f1={}
                for label in LABELS[:-1]:f1[label+'_before']=[one(label,9000+i,'F1') for i in range(args.f1_reps)]
                f1[LABELS[-1]]=[one(LABELS[-1],9000+i,'F1') for i in range(args.f1_reps)]
                for label in reversed(LABELS[:-1]):f1[label+'_after']=[one(label,9000+i,'F1') for i in range(args.f1_reps)]
                terminal('F1');entry['F1_medians_us']={key:statistics.median(values) for key,values in f1.items()}
                paired=[]
                for i in range(args.pairs):
                    row={}
                    for index in orders[i%2]:row[LABELS[index]]=one(LABELS[index],9020+i,'paired')
                    paired.append(row)
                terminal('paired');entry['paired_all_samples']=paired
                entry['paired_control_minus_candidate_us']={label:statistics.median(row[label]-row[LABELS[-1]] for row in paired) for label in LABELS[:-1]}
                entry['paired_control_over_candidate_ratio']={label:statistics.median(row[label]/row[LABELS[-1]] for row in paired) for label in LABELS[:-1]}
                # Separate retained-graph mutations AFTER both terminal gates; never repair timed outputs.
                probes=torch.zeros_like(fixtures[9000]);rows=torch.arange(m,device='cuda');probes[rows,(rows*37+19)%config['k']]=8
                mutations=[('zero',torch.zeros_like(probes)),('outliers',reference.generate_input(m,layer,9999,dist='outliers')['x']),('all_probes',probes),('fresh_normal',fixtures[9063])]
                for kind,fresh in mutations:
                    for label in LABELS:
                        x,out,g=work[label];x.copy_(fresh);out.fill_(float('nan'))
                        # Mutable float partial buffers must overwrite stale storage. Never poison counters.
                        # RED2 persistent-zero rs_ws is not an admitted selected path for this poison gate.
                        selected=entry['picks'][label]
                        assert not(selected['family']=='router' and selected['id'] in (8,11)),'HOLD active persistent-zero RED2 workspace'
                        poisoned=[];seen_scratch=set()
                        for path,t in tensors(states[label]):
                            if 'ws' in path and t.numel() and t.is_floating_point() and tensor_id(t) not in seen_scratch:
                                seen_scratch.add(tensor_id(t));t.fill_(float('nan'));poisoned.append(path)
                        g.replay();torch.cuda.synchronize();counts[label]+=1
                        message=oracle(label,fresh,kind)
                        if kind=='zero':assert not int(torch.count_nonzero(out))
                        entry['mutations'].append({'label':label,'kind':kind,'oracle':message,'output_poison_overwritten':True,'float_scratch_poisoned':poisoned})
                for x,out,g in work.values():torch.cuda.synchronize();g.reset();graphs_live.remove(g)
                entry['graphs']='RESET_AFTER_DRAIN';save()
        result['candidate_predxor_selected']=[name for name,e in result['shapes'].items() if G.selected_is_new(e['picks'][LABELS[-1]],LOADS[name.split('/')[0]],name.split('/')[1])]
        result['profiling']='NOT_REQUESTED_NO_PROFILER_DEPENDENCY';result['fixed_mechanism']='OMITTED_ACTUAL_POLICY_ONLY'
        result['all30_package_controls_required_for_promotion']=True
        result['state']='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED';result['official_score_claim']=False;phase('complete')
    except BaseException:
        result['state']='INVALID_NO_PERFORMANCE_ADMISSION';result['error']=traceback.format_exc();save();raise
    finally:
        cleanup=[]
        for graph in graphs_live:
            try:torch.cuda.synchronize();graph.reset();cleanup.append('RESET_AFTER_DRAIN')
            except BaseException as error:cleanup.append(repr(error));result['state']='INVALID_CLEANUP'
        result['graph_cleanup']=cleanup;result['finished_at']=time.time();save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path)
    p.add_argument('--pairs',type=int,default=32);p.add_argument('--f1-reps',type=int,default=11);p.add_argument('--cpu-artifacts',type=Path);args=p.parse_args()
    assert args.pairs==32 and args.f1_reps==11,'freeze balanced declared sample counts'
    if args.cpu_check:print(json.dumps(cpu_check(args.cpu_artifacts),indent=2));return
    assert args.cpu_artifacts is None
    assert args.output;run(args)
if __name__=='__main__':main()
