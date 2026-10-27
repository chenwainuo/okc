"""Pinned full-scope RS32/PW38/PW39 serving-policy lab; root-only GPU execution.
--cpu-check uses only stdlib/AST. No forced EH, clock reset or selector mutation.
"""
import argparse, ast, hashlib, importlib.util, json, re, statistics, subprocess, sys, threading, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TRIAL=ROOT/'proj_rsi/candidates/pw39_trial'
PINS={'rs32':'a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad','pw38':'29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e','pw39':'8a13389137add37b6d363685bfec98a7c83d48704cab2a72444129efc0c5520a'}
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
    def use(t,mode,changes=None):
        key=tensor_id(t);entry={'mode':mode,'offset':t.storage_offset(),'changes':changes or {}}
        assert key not in plan;plan[key]=entry
    if family=='RQ':use(p['cnt'],'reset0')
    elif family=='V12':use(state['cnt'],'reset0')
    elif family=='GV':
        fmt,mt,st,wr,wk,ku,s,rt,xs,v=info['variant']
        if(v>>27)&5:
            assert not(v&(1<<28)),'unreviewed PDL counter path'
            t=state.get('p1cnt',{}).get(p[0]);assert t is not None,'warm selected path before snapshot'
            tiles=((m+8*mt-1)//(8*mt))*(n//(16*wr*rt))
            use(t,'progress',{2*i:s for i in range(tiles)}|{2*i+1:1 for i in range(tiles)})
        elif s>1 and(v&4):use(state['gvcnt'],'reset0')
        # S1 and sm90 cluster branches do not use global counters.
    elif family=='router':
        ident=info['id']
        if ident<100:use(state['rs_cnt'],'reset0')
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
        if rule['mode']=='reset0':expected=[0]*len(expected)
        elif rule['mode']=='progress':
            for relative,delta in rule['changes'].items():
                index=rule['offset']+relative;assert 0<=index<len(expected)
                expected[index]+=delta*calls;assert expected[index]<2**31,'counter wrap not admitted'
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

def plain_metadata(value):
    if isinstance(value,float) and not __import__('math').isfinite(value):return {'nonfinite':repr(value)}
    if isinstance(value,dict):return {str(key):plain_metadata(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain_metadata(v) for v in value]
    assert value is None or isinstance(value,(str,int,float,bool)),('unexpected confirmation metadata',type(value))
    return value

def instrument_time_ast(node):
    import copy
    changed=copy.deepcopy(node);indices=[]
    for index,statement in enumerate(changed.body):
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and isinstance(statement.value.func.value,ast.Name) and statement.value.func.value.id=='times' and statement.value.func.attr=='sort':indices.append(index)
    assert len(indices)==1
    index=indices[0]
    observer=ast.Expr(value=ast.Call(func=ast.Name(id='_stack_time_observer',ctx=ast.Load()),args=[ast.Name(id='times',ctx=ast.Load()),ast.Name(id='it',ctx=ast.Load())],keywords=[]))
    changed.body.insert(index,ast.copy_location(observer,changed.body[index]));ast.fix_missing_locations(changed)
    restored=copy.deepcopy(changed);restored.body.pop(index)
    assert ast.dump(restored,include_attributes=False)==ast.dump(node,include_attributes=False),'observer inverse changed original timing algorithm'
    return changed

def install_prepare_observer(mod,label):
    audit={'label':label,'invocations':[],'instrumentation':'one in-memory observer before original times.sort; no extra event, replay, timer or CUDA call; charged to original prepare/confirmation clocks'}
    if not hasattr(mod,'_rqconfirm'):
        audit['helper_present']=False
        return audit,lambda:None
    audit['helper_present']=True;original_confirm=mod._rqconfirm;original_time=mod._time_us;original_rq=mod._rqcall
    source=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text()
    timer=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_time_us')
    active=[None]
    def observe(times,it):
        if active[0] is not None:
            active[0]['raw_timing_blocks'].append({'it':it,'raw_replay_us':plain_metadata(list(times))})
    mod.__dict__['_stack_time_observer']=observe
    exec(compile(ast.Module(body=[instrument_time_ast(timer)],type_ignores=[]),'<source-exact-timer-plus-CPU-observer>','exec'),mod.__dict__)
    def confirm(st,layer,m,cands,inc,x,out,flush,sink,screen,incumbent):
        assert active[0] is None
        entry={'layer':layer,'m':m,'screen_pick':screen[0],'incumbent':incumbent[0],'raw_timing_blocks':[]}
        audit['invocations'].append(entry);active[0]=entry
        before=len(st.get('rq_confirm',{}).get('records',[]))
        try:
            decision=original_confirm(st,layer,m,cands,inc,x,out,flush,sink,screen,incumbent)
            entry['returned_pick']=decision[0];entry['returned_us']=plain_metadata(decision[2])
            return decision
        finally:
            active[0]=None
            records=st.get('rq_confirm',{}).get('records',[])
            assert len(records)==before+1,'confirmation did not retain exactly one metadata record'
            entry['production_record']=plain_metadata(records[-1])
    def observe_rq(*args,**kwargs):
        entry=active[0]
        if entry is None:return original_rq(*args,**kwargs)
        plan=args[1];names=mod.RQPROD2 if 'mod' in plan else mod.RQPROD
        record={'block_position':len(entry['raw_timing_blocks']),'variant':names[plan['i']],'KS':plan['KS']}
        entry.setdefault('RQ_host_launch_status',[]).append(record)
        try:
            code=original_rq(*args,**kwargs);record['code']=int(code);return code
        except BaseException:
            record['exception']=traceback.format_exc();raise
    mod._rqcall=observe_rq
    mod._rqconfirm=confirm
    def restore():
        mod._rqconfirm=original_confirm;mod._time_us=original_time;mod._rqcall=original_rq
        mod.__dict__.pop('_stack_time_observer',None)
    return audit,restore

def confirmation_gate(audit):
    if not audit['helper_present']:
        assert not audit['invocations'] and audit['state_audit'] is None
        return
    records=(audit['state_audit'] or {}).get('records',[])
    assert records==[entry['production_record'] for entry in audit['invocations']]
    for entry in audit['invocations']:
        record=entry['production_record'];raw=entry['raw_timing_blocks'];blocks=record['blocks']
        assert record['layer']==entry['layer'] and record['m']==entry['m']
        assert len(raw) in (len(blocks),len(blocks)+1)
        if len(raw)>len(blocks):assert record['status']=='error' and 'attempting_block' in record
        assert all(call.get('code')==0 and 'exception' not in call for call in entry.get('RQ_host_launch_status',[])),'confirmation RQ launch failure cannot yield valid performance'
        for observed,block in zip(raw,blocks):
            assert observed['it']==block['replays']==5 and len(observed['raw_replay_us'])==5
            assert statistics.median(observed['raw_replay_us'])==block['median_us']
        if record['status']=='confirmed':
            assert len(raw)==len(blocks) and len(blocks) in (5,7)
            order=[-1,0,1,-1,1,0,-1] if len(record['candidates'])==2 else [-1,0,-1,0,-1]
            assert [block['index'] for block in blocks]==order
            controls=[i for i,block in enumerate(blocks) if block['index']<0]
            for i,candidate in enumerate(record['candidates']):
                times=[];ratios=[]
                for pos,block in enumerate(blocks):
                    if block['index']!=i:continue
                    before=max(k for k in controls if k<pos);after=min(k for k in controls if k>pos)
                    baseline=min(blocks[before]['median_us'],blocks[after]['median_us'])
                    times.append(block['median_us']);ratios.append(block['median_us']/baseline)
                assert times==candidate['confirmation_medians_us'] and ratios==candidate['bracket_ratios']
                assert max(ratios)==candidate['worst_bracket_ratio']
            winner=min(range(len(record['candidates'])),key=lambda i:record['candidates'][i]['worst_bracket_ratio'])
            selected=record['candidates'][winner]
            expected='rq:'+selected['tag'] if selected['worst_bracket_ratio']<.99 else entry['incumbent']
            assert record['decision']==expected
            expected_us=max(selected['confirmation_medians_us']) if selected['worst_bracket_ratio']<.99 else min(blocks[i]['median_us'] for i in controls)
            assert entry['returned_us']==expected_us
        else:
            assert record['status'] in ('skipped','error')
            assert record['decision']==(entry['incumbent'] if record['status']=='error' else entry['screen_pick'])
        # Preserve partial/error attempts and reasons; do not convert fallback to success.
        assert record['reason'] and record['decision']==entry.get('returned_pick')


def cpu_check():
    verify()
    # Source unchanged except reviewed host confirmation; all five GPU literals and 19 instances remain identical.
    parent=(ROOT/'proj_rsi/candidates/pw38/submission.py').read_text();new=(ROOT/'proj_rsi/candidates/pw39/submission.py').read_text()
    def literal(text,name):
        return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    assert all(literal(parent,name)==literal(new,name) for name in ('CUDA','GVCU','LTCPP','RWCU','RQCU'))
    generated={}
    for label in ('pw38','pw39'):
        text=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_time_us');instrument_time_ast(node)
        env={'RQCU':literal(text,'RQCU')}
        for factory in ast.parse(text).body:
            if isinstance(factory,ast.FunctionDef) and factory.name in ('_rqwg','_rqss','_rqsrc'):exec(compile(ast.Module(body=[factory],type_ignores=[]),'<pure-RQ-factory>','exec'),env)
        variants={v[0]:v for v in literal(text,'RQV')};small=list(dict.fromkeys(name for pairs in literal(text,'RQSBAKE').values() for name,ks in pairs))
        generated[label]={name:hashlib.sha256(env['_rqsrc']([variants[v] for v in names]).encode()).hexdigest() for name,names in [('RQ128',literal(text,'RQPROD')),('RQ32',small)]}
    assert generated['pw38']==generated['pw39']=={'RQ128':'d50202811d73e1ce12eff287d17b67ca5366db3c708829d720867521066fd9fa','RQ32':'9f187dc85102a1ca956b895a6d3168220a5eb6fce367072a90f2f7e05c29a589'}
    fake=ast.parse('def _time_us(it=5):\n times=[5.,1.,4.,2.,3.]\n times.sort()\n return times[len(times)//2]\n').body[0]
    captured=[];env={'_stack_time_observer':lambda times,it:captured.append((list(times),it))}
    exec(compile(ast.Module(body=[instrument_time_ast(fake)],type_ignores=[]),'<fake-timer-observer>','exec'),env)
    assert env['_time_us']()==3 and captured==[([5.,1.,4.,2.,3.],5)]
    source=(ROOT/'proj_rsi/correctness.py').read_text()
    official=next(ast.literal_eval(node.value) for node in ast.parse(source).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OFFICIAL_CASES' for t in node.targets))
    assert list(official)==OFFICIAL
    assert len(LOADS)*len(LAYERS)==30 and len(MS)*len(LAYERS)==65 and len(OFFICIAL)==9
    # Pure semantic positive/negative contracts, including dormant odd/sentinel words.
    before={'a':{'name':'arena','values':[12,7,0,19]}}
    counter_check(before,{'a':{'name':'arena','values':[20,8,0,19]}},{'a':{'mode':'progress','offset':0,'changes':{0:8,1:1}}},1)
    caught=[]
    for label,values in [('missing_ticket',[12,8,0,19]),('dormant_word',[20,8,1,19]),('flag',[20,9,0,19])]:
        try:counter_check(before,{'a':{'name':'arena','values':values}},{'a':{'mode':'progress','offset':0,'changes':{0:8,1:1}}},1)
        except AssertionError:caught.append(label)
        else:raise AssertionError('negative accepted')
    for label in LABELS:
        source=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text()
        assert 'tk = atomicAdd(p.cnt + 2 * tile, 1)' in source
        assert 'atomicAdd(p.cnt + 2 * tile + 1, 1)' in source
        assert 'slot * 64 + blockIdx.x + gridDim.x * blockIdx.z' in source
        assert 'atom.add.release.gpu.global.s32' in source
        assert 'enumerate(sorted(RWV.items()))' in source
        if label!='rs32':assert '64 + i) for i, (v, c) in enumerate(sorted(RTV.items()))' in source
    canonical=(ROOT/'proj_rsi/candidates/rs32_eh_backport/sass.txt').read_text()
    targets=rq_targets(canonical)
    resources='\n'.join('Function '+entry['symbol']+'\n REG:168 LOCAL:0\n' for entry in targets.values())
    actual_rq_gate(canonical,resources,True)
    negatives={'wrong_registers':(canonical,resources.replace('REG:168','REG:169',1)),
               'local_spill':(canonical,resources.replace('LOCAL:0','LOCAL:16',1)),
               'missing_EH':('\n Function : '+targets[2]['table'],resources)}
    jit_negative=[]
    for name,(sass,res) in negatives.items():
        try:actual_rq_gate(sass,res,True)
        except AssertionError:jit_negative.append(name)
        else:raise AssertionError('actual JIT negative accepted: '+name)
    return {'router_source_switch_proof':router_cpu_proof(),'jit_negative_controls':jit_negative,'state':'CPU_STACK_LAB_PASS_GPU_UNRUN' ,'sources':PINS,'scope':{'timed_shapes':30,'normal_cases_per_arm':65,'exact_official_cases_per_arm':9},'counter_negatives':caught,'CUDA_API_calls':0,'production_imports':0,'confirmation_observer_inverse_AST_and_unsorted_raw':'PASS','GPU_literals_PW38_PW39':'ALL_FIVE_EXACT','generated_19_RQ_instances':generated}

def run(args):
    verify();assert args.review and Path(args.review).is_file(),'independent frozen driver review required'
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    args.output.mkdir(parents=True,exist_ok=False)
    result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','sources':PINS,'review_sha256':sha(args.review),'prepare':{},'prepare_confirmation':{},'jit':[],'jit_errors':[],'checks':[],'shapes':{},'raw':[]}
    lock=threading.RLock();graphs_live=[]
    def save():
        with lock:dump(args.output/'result.json',result)
    def phase(name):result['phase']=name;save();print(name,flush=True)
    modules={};states={};immutable=[];weights=reference.generate_weights(reference.WEIGHTS_SEED)
    original_weights=[(key,t,t.clone()) for key,t in weights.items()]
    def immutable_check():
        for name,t,original in original_weights+immutable:
            assert torch.equal(t.view(torch.uint8),original.view(torch.uint8)),('immutable changed',name)
    registry={}
    try:
        for label in LABELS:
            phase('prepare_'+label)
            spec=importlib.util.spec_from_file_location('stack_frozen_'+label,ROOT/f'proj_rsi/candidates/{label}/submission.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules[label]=mod
            original_build=mod.load_inline
            def checked_build(*a,_build=original_build,_label=label,**kw):
                name=kw.get('name',a[0] if a else None);cu=kw['cuda_sources'];digest=hashlib.sha256(str(cu).encode()).hexdigest()
                with lock:
                    assert name not in registry or registry[name]==digest,('extension name collision',name)
                    registry[name]=digest
                ext=_build(*a,**kw)
                sass=subprocess.check_output(['cuobjdump','--dump-sass',str(ext.__file__)],text=True,timeout=60)
                resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(ext.__file__)],text=True,timeout=60)
                prefix=_label+'.'+str(name)
                (args.output/(prefix+'.sass')).write_text(sass);(args.output/(prefix+'.resources')).write_text(resources)
                (args.output/(prefix+'.cu')).write_text(str(cu))
                if 'struct RqP' in str(cu):
                    expect_eh=_label in ('pw38','pw39') and 'rq_go<128,5,1,0,0,0,65538>' in re.sub(r'\s+','',str(cu))
                    actual_gate=actual_rq_gate(sass,resources,expect_eh)
                with lock:
                    result['jit'].append({'label':_label,'name':name,'source_sha256':digest,'so_sha256':sha(ext.__file__),'SASS_sha256':sha(args.output/(prefix+'.sass')),'before_return_to_prepare_math':True,'RQ_gate':actual_gate if 'struct RqP' in str(cu) else None});save()
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
            confirmation,restore_observer=install_prepare_observer(mod,label)
            result['prepare_confirmation'][label]=confirmation
            start=time.time()
            try:
                state=states[label]=mod.prepare(weights);torch.cuda.synchronize()
            finally:restore_observer()
            confirmation['state_audit']=plain_metadata(state.get('rq_confirm'))
            confirmation['actually_invoked']=bool(confirmation['invocations'])
            confirmation['observer_detached_after_prepare']=True
            confirmation_gate(confirmation)

            for thread in build_threads:thread.join(timeout=max(0,300-(time.time()-mod._T0)))
            assert not any(thread.is_alive() for thread in build_threads),'prepare background build still live after300s'
            assert not any(error['label']==label for error in result['jit_errors']),('actual JIT/build gate failed',label)
            if label in ('pw38','pw39'):assert any(record['label']==label and record.get('RQ_gate',{}).get('flags')==[2,3,65538,65539] for record in result['jit'] if record.get('RQ_gate')),'PW39 EH actual JIT gate never completed'

            result['prepare'][label]={'wall_seconds':time.time()-start,'import_clock_elapsed':time.time()-mod._T0,'budget_clock_reset':False,'serving_CPU_fallback':bool(state.get('cpu')),'build_threads_joined':len(build_threads)}
            assert not state.get('cpu') and time.time()-mod._T0<300,('prepare overrun',label)
            # Prevent silent RQ fallback during serving; preparation retains its original decline logic.
            original_rq=mod._rqcall
            def guarded(*a,_fn=original_rq,_label=label,**kw):
                code=_fn(*a,**kw);assert code==0,('serving RQ declined',_label,code);return code
            mod._rqcall=guarded
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
        flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
        orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
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
                assert len({item[0].data_ptr() for item in work.values()})==3
                assert len({item[1].data_ptr() for item in work.values()})==3
                last={}
                def oracle(label,fresh,kind):
                    x,out,g=work[label]
                    assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
                    ok,message=reference.check_implementation({'x':fresh,'config':config},out,weights)
                    assert ok,(name,label,kind,message)
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
                f1['pw39']=[one('pw39',9000+i,'F1') for i in range(args.f1_reps)]
                for label in reversed(LABELS[:-1]):f1[label+'_after']=[one(label,9000+i,'F1') for i in range(args.f1_reps)]
                terminal('F1');entry['F1_medians_us']={key:statistics.median(values) for key,values in f1.items()}
                paired=[]
                for i in range(args.pairs):
                    row={}
                    for index in orders[i%6]:row[LABELS[index]]=one(LABELS[index],9020+i,'paired')
                    paired.append(row)
                terminal('paired');entry['paired_all_samples']=paired
                entry['paired_control_minus_pw39_us']={label:statistics.median(row[label]-row['pw39'] for row in paired) for label in LABELS[:-1]}
                entry['paired_control_over_pw39_ratio']={label:statistics.median(row[label]/row['pw39'] for row in paired) for label in LABELS[:-1]}
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
        result['EH_selected']=[name for name,e in result['shapes'].items() if e['picks']['pw39'].get('name','').endswith('EH')]
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
    p.add_argument('--pairs',type=int,default=30);p.add_argument('--f1-reps',type=int,default=11);args=p.parse_args()
    assert args.pairs==30 and args.f1_reps==11,'freeze balanced declared sample counts'
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output;run(args)
if __name__=='__main__':main()
