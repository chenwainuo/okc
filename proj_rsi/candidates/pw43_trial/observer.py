"""Stdlib-only transparent PW43 prepare observer; no added CUDA/clock calls."""
import ast,copy,inspect,json,math,statistics

def instrument(node,kind):
    old=ast.dump(node,include_attributes=False);node=copy.deepcopy(node)
    if kind=='timer':
        hits=[(body,i) for parent in ast.walk(node) for body in [getattr(parent,'body',[])] for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='ts' and n.value.func.attr=='sort']
        assert len(hits)==1;body,i=hits[0];extra=ast.parse('_stack_lt_raw(ts,it)').body[0];body.insert(i,extra)
    else:
        hits=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='explore_end' for t in n.targets)]
        assert len(hits)==1;body=node.body;i=hits[0]+1;extra=ast.parse('_stack_lt_deadline(remaining,dl,end,explore_end,target)').body[0];body.insert(i,extra)
    removed=body.pop(i);assert ast.dump(node,include_attributes=False)==old
    body.insert(i,removed);return ast.fix_missing_locations(node)

def gate(audit):
    for entry in audit['searches']:
        for timer in entry['timers']:
            assert len(timer['raw_ms'])==timer['it'] and all(math.isfinite(x) and x>0 for x in timer['raw_ms'])
            assert statistics.median(timer['raw_ms'])==timer['returned_ms']
        confirms=entry['confirmations'];assert len(confirms)<=1
        if not confirms:continue
        c=confirms[0];record=c['record'];rows=record['rows'];deltas={}
        codes=c['codes'];assert len(codes)<=2 and len(set(codes))==len(codes)
        expected_rows=[(fixture,round_id,code) for fixture in (0,1) for round_id in (0,1) for code in (codes if round_id==0 else list(reversed(codes)))]
        actual_rows=[(row['fixture'],row['round'],row['code']) for row in rows]
        assert actual_rows==expected_rows[:len(rows)] and len(rows)<=len(expected_rows),'exact ABBA fixture/round/code prefix required'
        expected_valid=[(fixture,code) for fixture in (0,1) for code in codes]
        actual_valid=[(v['fixture'],v['code']) for v in c['validations']]
        assert actual_valid==expected_valid[:len(actual_valid)] and len(actual_valid)<=len(expected_valid),'exact validation prefix required'
        assert all(v['valid'] for v in c['validations'][:-1]),'only last validation may decline'
        assert all(len(row['times_ms'])==3 for row in rows[:-1]),'only last observed row may be partial'
        if any(v['fixture']==1 for v in c['validations']):assert len(rows)>=2*len(codes) and all(len(r['times_ms'])==3 for r in rows[:2*len(codes)])
        expected_timing=[]
        for row in rows:
            assert len(row['times_ms'])<=3
            assert all(any(v['fixture']==row['fixture'] and v['code']==code and v['valid'] for v in c['validations']) for code in codes),'fixture must validate every finalist before timing'
            expected_timing += [(row['fixture'],code) for code in (-1,row['code'],-1)[:len(row['times_ms'])]]
        assert [(t['fixture'],t['code']) for t in c['timings']]==expected_timing,'timing triples must bind exact fixture and candidate/control identity'
        if record['status']=='complete':assert actual_rows==expected_rows and actual_valid==expected_valid and all(len(r['times_ms'])==3 for r in rows)

        if entry.get('deadlines'):
            confirmed_timers=[t for t in entry['timers'] if t['context']=='confirmation_timing']
            assert len(confirmed_timers)==len(c['timings'])
            assert [(t['confirmation_fixture'],t['algorithm_code'],t['returned_ms']) for t in confirmed_timers]==[(t['fixture'],t['code'],t['returned_ms']) for t in c['timings']]
            for observation in record.get('exploration_observations',[]):
                if observation['time_ms'] is not None:
                    matches=[t for t in entry['timers'] if t['it']==5 and t['algorithm_code']==observation['code']]
                    assert len(matches)==1 and matches[0]['returned_ms']==observation['time_ms']
        for r in rows:
            assert r['fixture'] in (0,1) and r['round'] in (0,1) and r['code'] in c['codes']
            if len(r['times_ms'])==3 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in r['times_ms']):
                x,y,z=r['times_ms'];deltas.setdefault(str(r['code']),[]).append((x+z)/2-y)
        if record['status']=='complete':
            assert len(rows)==4*len(c['codes'])
            assert record['deltas']==deltas
            eligible=[(sum(deltas[str(code)])/4,code) for code in c['codes'] if len(deltas.get(str(code),[]))==4 and min(deltas[str(code)])>0]
            expected=max(eligible,key=lambda pair:(pair[0],-c['codes'].index(pair[1])))[1] if eligible else None
            assert expected==c['returned']==entry['returned']
            assert len(c['timings'])==12*len(c['codes'])
        else:assert record['status']=='incomplete' and c['returned'] is None and entry['returned'] is None and record['reason']
        flat=[value for row in rows for value in row['times_ms']]
        assert flat==[r['returned_ms'] for r in c['timings']]
        if record['status']=='complete':assert all(v['valid'] for v in c['validations'])
    assert all(e['return_code']==0 for e in audit['lt_mm_status'] if e['context'] in ('timing','confirmation_timing','serving'))
    if audit['printed_records']:
        assert json.dumps(audit['printed_records'],sort_keys=True)==json.dumps([c['record'] for e in audit['searches'] for c in e['confirmations']],sort_keys=True)
    assert audit['observer_detached']

def install(mod,label):
    missing=object();names=('_tm','_ltsearch','_lt43confirm','time','print','_ext','_stack_lt_raw','_stack_lt_deadline')
    saved={name:getattr(mod,name,missing) for name in names};audit={'label':label,'searches':[],'lt_mm_status':[],'config_calls':[],'printed_records':[],'observer_detached':False,'added_GPU_calls':0,'added_clock_queries':0}
    active=[None];context=['discovery_validation'];preparing=[True]
    def raw(values,it):
        if active[0] is not None:active[0]['timers'].append({'context':context[0],'algorithm_code':active[0].get('timer_algorithm_code'),'confirmation_fixture':active[0].get('timer_fixture'),'raw_ms':list(values),'it':it})
    def deadline(remaining,dl,end,explore_end,target):
        if active[0] is not None:active[0]['deadlines'].append(dict(remaining_seconds=remaining,wall_deadline=dl,monotonic_deadline=end,exploration_monotonic_deadline=explore_end,target=target,original_T0=mod._T0,prepare_budget=mod.PREPARE_BUDGET_S))
    mod._stack_lt_raw=raw;mod._stack_lt_deadline=deadline
    tree=ast.parse(inspect.getsource(saved['_tm']));exec(compile(ast.Module(body=[instrument(tree.body[0],'timer')],type_ignores=[]),'<pure-observer-timer>', 'exec'),mod.__dict__)
    timed_original=mod._tm
    def timer(*args,**kw):
        entry=active[0];before=len(entry['timers']) if entry else 0
        value=timed_original(*args,**kw)
        if entry is not None:assert len(entry['timers'])==before+1;entry['timers'][-1]['returned_ms']=value
        return value
    mod._tm=timer
    class Clock:
        def __getattr__(self,name):return getattr(saved['time'],name)
        def query(self,kind):
            value=getattr(saved['time'],kind)()
            if active[0] is not None:active[0]['clock_queries'].append({'kind':kind,'value':value,'context':context[0]})
            return value
        def time(self):return self.query('time')
        def monotonic(self):return self.query('monotonic')
    mod.time=Clock()
    def output(*args,**kw):
        if len(args)==1 and isinstance(args[0],str) and args[0].startswith('PW43_LT '):audit['printed_records'].append(json.loads(args[0][8:]))
        return print(*args,**kw)
    mod.print=output
    native_ext=saved['_ext'];proxies={}
    class Extension:
        def __init__(self,native):self.native=native
        def __getattr__(self,name):return getattr(self.native,name)
        def lt_mm(self,*args,**kw):
            code=self.native.lt_mm(*args,**kw)
            x=args[0]
            if int(x.shape[0])==256:
                event={'m':256,'K':int(x.shape[1]),'n':int(args[4]),'algorithm_code':int(args[5]),'return_code':int(code),'context':context[0] if preparing[0] else 'serving','layer':active[0]['layer'] if active[0] else None};audit['lt_mm_status'].append(event)
                if event['context'] in ('timing','confirmation_timing','serving'):assert code==0,('Lt timing/serving declined',event)
            return code
        def lt_cfg(self,*args,**kw):
            value=self.native.lt_cfg(*args,**kw)
            if active[0] is not None:audit['config_calls'].append({'layer':active[0]['layer'],'m':int(args[0]),'n':int(args[1]),'K':int(args[2]),'f32':int(args[3]),'code':int(args[4]),'layout':int(args[4])//100000,'config':list(value)})
            return value
    def extension(*args,**kw):
        native=native_ext(*args,**kw);key=id(native)
        if key not in proxies:proxies[key]=Extension(native)
        return proxies[key]
    mod._ext=extension
    confirm=saved['_lt43confirm']
    if confirm is not missing:
        def confirmation(codes,factory,end,clock):
            entry={'codes':list(dict.fromkeys(codes))[:2],'deadline':end,'validations':[],'timings':[]};active[0]['confirmations'].append(entry)
            def fixture(i):
                context[0]='confirmation_fixture';valid,timed=factory(i)
                def validation(code):
                    context[0]='confirmation_validation';value=valid(code);entry['validations'].append({'fixture':i,'code':code,'valid':bool(value)});return value
                def timing(code):
                    context[0]='confirmation_timing';active[0]['timer_algorithm_code']=code;active[0]['timer_fixture']=i;value=timed(code);entry['timings'].append({'fixture':i,'code':code,'returned_ms':value});return value
                return validation,timing
            selected,record=confirm(codes,fixture,end,clock);entry.update(returned=selected,record=record);return selected,record
        mod._lt43confirm=confirmation
        node=ast.parse(inspect.getsource(saved['_ltsearch'])).body[0];exec(compile(ast.Module(body=[instrument(node,'deadline')],type_ignores=[]),'<pure-observer-deadlines>','exec'),mod.__dict__)
        search_native=mod._ltsearch
    else:search_native=saved['_ltsearch']
    def search(ext,layer,m,n,K,f32,ok,run,torch_fn,flush,t0,heldout=None):
        target=m==256 and not f32
        if not target:return search_native(ext,layer,m,n,K,f32,ok,run,torch_fn,flush,t0) if confirm is missing else search_native(ext,layer,m,n,K,f32,ok,run,torch_fn,flush,t0,heldout)
        entry={'layer':layer,'m':m,'n':n,'K':K,'timers':[],'deadlines':[],'clock_queries':[],'confirmations':[],'discovery_validations':[]};audit['searches'].append(entry);active[0]=entry;context[0]='discovery_validation'
        def valid(code):
            context[0]='discovery_validation';value=ok(code);entry['discovery_validations'].append({'code':code,'valid':bool(value)});return value
        def running(code):context[0]='timing';entry['timer_algorithm_code']=code;return run(code)
        def torch_timed():context[0]='timing';entry['timer_algorithm_code']=-1;return torch_fn()
        try:
            entry['returned']=search_native(ext,layer,m,n,K,f32,valid,running,torch_timed,flush,t0) if confirm is missing else search_native(ext,layer,m,n,K,f32,valid,running,torch_timed,flush,t0,heldout)
            return entry['returned']
        except BaseException as error:entry['exception']=repr(error);raise
        finally:active[0]=None;context[0]='discovery_validation'
    mod._ltsearch=search
    def detach_prepare():
        preparing[0]=False
        # Keep only transparent extension status guard for real serving calls.
        for name,value in saved.items():
            if name=='_ext':continue
            if value is missing:mod.__dict__.pop(name,None)
            else:setattr(mod,name,value)
        audit['observer_detached']=True;gate(audit)
    return audit,detach_prepare

def plain(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':repr(value)}
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def cpu_check(source):
    nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    instrument(nodes['_tm'],'timer');instrument(nodes['_ltsearch'],'deadline')
    fake=ast.parse('def _tm(it=3):\n ts=[3.,1.,2.]\n ts.sort()\n return ts[len(ts)//2]\n').body[0]
    observed=[];env={'_stack_lt_raw':lambda values,it:observed.append((list(values),it))}
    exec(compile(ast.Module(body=[instrument(fake,'timer')],type_ignores=[]),'<fake-timer>','exec'),env)
    assert env['_tm']()==2 and observed==[([3.,1.,2.],3)]
    env={};exec(compile(ast.Module(body=[nodes['_lt43confirm']],type_ignores=[]),'<real-source-helper>','exec'),env)
    def case(deadline=100):
        calls=[0];entry={'timers':[],'confirmations':[]};c={'codes':[7,8],'validations':[],'timings':[]};entry['confirmations'].append(c)
        def clock():calls[0]+=.01;return calls[0]
        def factory(fixture):
            def valid(code):c['validations'].append({'fixture':fixture,'code':code,'valid':True});return True
            def timed(code):
                value=1.0 if code<0 else .9999 if code==7 else 1.0001
                c['timings'].append({'fixture':fixture,'code':code,'returned_ms':value});entry['timers'].append({'it':3,'raw_ms':[value]*3,'returned_ms':value});return value
            return valid,timed
        selected,record=env['_lt43confirm'](c['codes'],factory,deadline,clock);c.update(returned=selected,record=record);entry['returned']=selected
        audit={'searches':[entry],'observer_detached':True,'lt_mm_status':[],'printed_records':[]};gate(audit);return audit
    complete=case();assert complete['searches'][0]['returned']==7
    partial=case(.15);assert partial['searches'][0]['confirmations'][0]['record']['status']=='incomplete'
    negatives=[]
    for name in ('wrong_raw','wrong_winner','nonzero_status','missing_row','wrong_fixture','wrong_order','duplicate_row','wrong_validation_fixture','wrong_timing_fixture'):
        bad=copy.deepcopy(complete)
        if name=='wrong_raw':bad['searches'][0]['timers'][0]['raw_ms'][:2]=[9,9]
        if name=='wrong_winner':bad['searches'][0]['returned']=8
        if name=='nonzero_status':bad['lt_mm_status']=[{'return_code':1,'context':'serving'}]
        if name=='missing_row':bad['searches'][0]['confirmations'][0]['record']['rows'].pop()
        confirmation=bad['searches'][0]['confirmations'][0]
        if name=='wrong_fixture':
            for row in confirmation['record']['rows']:row['fixture']=0;row['round']=0
        if name=='wrong_order':confirmation['record']['rows'][0],confirmation['record']['rows'][1]=confirmation['record']['rows'][1],confirmation['record']['rows'][0]
        if name=='duplicate_row':confirmation['record']['rows'][1]=copy.deepcopy(confirmation['record']['rows'][0])
        if name=='wrong_validation_fixture':confirmation['validations'][2]['fixture']=0
        if name=='wrong_timing_fixture':confirmation['timings'][-1]['fixture']=0
        try:gate(bad)
        except AssertionError:negatives.append(name)
        else:raise AssertionError('observer negative accepted '+name)
    return {'exact_AST_inverse_timer_and_deadline':True,'real_helper_tiny_gain_and_partial_PASS':True,'negative_controls':negatives,'added_GPU_calls':0,'added_clock_queries':0}

def integration_cpu(source_path):
    """Run real host search/timer and installed observer with fake CUDA/extension."""
    import contextlib,io,types
    value=[1.0]
    class Scope:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def wait_stream(self,*args):pass
        def replay(self):pass
        def record(self):pass
        def elapsed_time(self,other):return value[0]
    cuda=types.SimpleNamespace(CUDAGraph=Scope,Stream=Scope,stream=lambda s:Scope(),graph=lambda g:Scope(),current_stream=Scope,synchronize=lambda:None,Event=lambda *a,**k:Scope())
    torch=types.SimpleNamespace(cuda=cuda,sum=lambda *a,**k:None,empty=lambda *a,**k:object())
    class Times:
        def __init__(self):self.t=0
        def query(self):self.t+=.00001;return self.t
        time=query;monotonic=query
    class Tensor:shape=(256,8)
    class Native:
        def lt_algos(self,*args):return 2
        def lt_caps(self,*args):return []
        def lt_add(self,*args):return -1
        def lt_cfg(self,*args):return [66,1,1,0,0,0,35,0,0]
        def lt_mm(self,*args):value[0]=.9999 if args[5]==0 else .9998;return 0
    mod=types.ModuleType('fake_production_host');mod.__dict__.update(torch=torch,time=Times(),_FL=[None,None],_ext=lambda:Native(),_LTDONE=[],_LT0=[],_T0=0.,PREPARE_BUDGET_S=240.,LTBUDGET=22.,LTW={'qkv_gate':1.6,'o':.3,'pooler':1.,'unpooler':.7,'router':1.6},_LTSEED=[])
    for node in ast.parse(source_path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_tm','_ltsearch','_lt43confirm'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source_path),'exec'),mod.__dict__)
    original=mod._tm;original_search=mod._ltsearch;original_time=mod.time;audit,detach=install(mod,'fake');ext=mod._ext();flush=types.SimpleNamespace(dtype='float32',device='fakeCPU')
    def valid(code):return ext.lt_mm(Tensor(),None,None,None,8,code)==0
    def run(code):return ext.lt_mm(Tensor(),None,None,None,8,code)
    def baseline():value[0]=1.0
    def factory(fixture):
        def timed(code):return mod._tm(baseline if code<0 else lambda:run(code),flush,3)
        return valid,timed
    with contextlib.redirect_stdout(io.StringIO()):
        for layer in ('qkv_gate','o','pooler','unpooler'):
            selected=mod._ltsearch(ext,layer,256,8,8,0,valid,run,baseline,flush,0.,factory)
            assert selected is not None
    detach();assert mod._tm is original and mod._ltsearch is original_search and mod.time is original_time
    assert len(audit['searches'])==len(audit['printed_records'])==4
    assert all(len(e['deadlines'])==1 and e['clock_queries'] and e['timers'] for e in audit['searches'])
    assert audit['config_calls'] and audit['lt_mm_status']
    return {'real_host_install_search_timer_detach_four_targets_PASS':True,'CUDA_API_calls':0,'real_torch_imports':0,'searches':4,'raw_timers':sum(len(e['timers']) for e in audit['searches'])}
