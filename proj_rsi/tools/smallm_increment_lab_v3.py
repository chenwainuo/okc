"""Root-only actual serving comparison. --cpu-check never imports torch.
GPU entry requires independent review; no source/dispatch edits or forced picks.
"""
import argparse, ast, collections, hashlib, importlib.util, json, os
from pathlib import Path
import re, statistics, subprocess, sys, time, traceback
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'proj_rsi'
TRIAL = BASE / 'candidates/smallm_increment_trial_v3'
PINS = {
 'pw36': ('pw36','2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'),
 'rs32': ('rs32','a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'),
 'increment': ('smallm_increment','0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737')}
LABELS = tuple(PINS)
SHAPES = {'qkv32':('qkv_gate',32), 'qkv16':('qkv_gate',16), 'pooler32':('pooler',32),
          'unpooler32':('unpooler',32), 'o32':('o',32)}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data,indent=2,default=str)+'\n')
def verify():
    for _,(folder,pin) in PINS.items(): assert sha(BASE/'candidates'/folder/'submission.py') == pin
    manifest=json.loads((TRIAL/'manifest.json').read_text())
    for rel,pin in manifest['files'].items(): assert sha(ROOT/rel)==pin, rel
    return manifest

def ops(sass):
    result={}
    for segment in sass.split('Function : ')[1:]:
        name=segment.splitlines()[0].strip()
        result[name]=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',segment))
    return result

def selected_attributes(ext, rp, x, out, state):
    # rqrun(v,W,x,out,esc,cnt,KS,Tb,walias,prof,sink,info): final arg is info.
    return [int(ext.rqrun(rp['i'], rp['lay'], x, out, rp['esc'], rp['cnt'],
                          rp['KS'], rp['tab'], 0, state['rq_e'], state['rq_sink'], info))
            for info in (1, 2, 3)]

def routing_proof():
    source = (BASE/'candidates/smallm_increment/submission.py').read_text()
    signature = re.search(r'int64_t rqrun\(([^)]*)\)', source)
    assert signature
    assert re.search(r'if \(info\) \{\s*cudaFuncAttributes', source)
    assert re.search(r'if \(!info\) \{\s*if \(W.dim', source)
    names = [part.strip().split()[-1] for part in signature.group(1).split(',')]
    assert names == ['v','W','x','out','esc','cnt','KS','Tb','walias','prof','sink','info'], names
    class FakeExtension:
        def __init__(self): self.queries=[]; self.launches=0
        def rqrun(self, v, W, x, out, esc, cnt, KS, Tb, walias, prof, sink, info):
            assert walias == 0, 'walias is not the query selector'
            if info == 0: self.launches+=1; return 0
            assert info in (1,2,3)
            self.queries.append(info)
            return {1:168,2:0,3:83080}[info]
    rp={'i':7,'lay':'W','esc':'esc','cnt':'cnt','KS':7,'tab':123}
    state={'rq_e':'prof','rq_sink':'sink'}
    fake=FakeExtension()
    assert selected_attributes(fake,rp,'x','out',state)==[168,0,83080]
    assert fake.queries==[1,2,3] and fake.launches==0
    rejected=False
    try: fake.rqrun(7,'W','x','out','esc','cnt',7,123,1,'prof','sink',0)
    except AssertionError: rejected=True
    assert rejected, 'old info-in-walias call must fail the CPU contract'
    return {'signature':names,'positive_queries':[1,2,3],'positive_math_launches':fake.launches,
            'negative_old_info_in_walias_rejected':rejected}

def cpu_check():
    verify(); cases=0
    for st in (5,6):
        for kcs in range(1,129):
            slot=phase=0
            for g in range(kcs):
                assert (slot,phase)==(g%st,(g//st)&1)
                slot+=1
                if slot==st: slot=0;phase^=1
                cases+=1
    before=ops((BASE/'candidates/smallm_increment/before.sass').read_text())
    after=ops((BASE/'candidates/smallm_increment/after.sass').read_text())
    assert before.keys()==after.keys()
    for symbol in before:
        for op in set(before[symbol])|set(after[symbol]):
            if op.startswith(('LDS','SYNCS','HGMMA','WARPGROUP')):
                assert before[symbol][op]==after[symbol][op],(symbol,op)
        assert after[symbol]['UIMAD.WIDE.U32']==before[symbol]['UIMAD.WIDE.U32']-1
    return {'CPU_ONLY':True,'counter_contract_proof':counter_cpu_proof(),'rqrun_argument_routing':routing_proof(),'cursor_events':cases,'all_selected_body_barrier_math_LDS_counts_unchanged':True,
            'physical_register_budget':{'producer':40,'math':232,'threads':384,'dynamic':40*128+232*256,'static':168*384},
            'scope_GPU_pending':list(SHAPES)}

def load(label):
    folder,_=PINS[label]; path=BASE/'candidates'/folder/'submission.py'
    spec=importlib.util.spec_from_file_location('smallm_frozen_'+label,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def jit_gate(mod,label,outdir,registry):
    """Build + inspect rq32 before prepare can bake candidate math. No launch."""
    vs=[mod.RQV[mod.RQI[n]] for n in mod.RQPROD2]
    source=mod._rqsrc(vs);tag='rq32_%x'%(hash(source)&0xffffffff)
    digest=hashlib.sha256(source.encode()).hexdigest()
    assert tag not in registry or registry[tag]==digest,'32-bit extension name collision'
    registry[tag]=digest
    ext=mod._rqext(vs,slot=mod._rqmod2)
    assert mod._rqmod2[1]==source
    so=Path(ext.__file__); (outdir/(label+'.rq32.cu')).write_text(source)
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(so)],text=True)
    resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(so)],text=True)
    (outdir/(label+'.sass')).write_text(sass);(outdir/(label+'.resources')).write_text(resource)
    counts=ops(sass);expected=ops((BASE/'candidates/smallm_increment'/('after.sass' if label=='increment' else 'before.sass')).read_text())
    # Exact device opcode counts bind the runtime JIT recurrence to CPU-admitted bodies.
    # pw36 is a comparator; its resources are recorded without asserting rs32 identity.
    if label in ('increment','rs32'):
        for symbol,want in expected.items():
            assert counts[symbol]==want,(label,'actual JIT opcode drift',symbol)
            block=resource.split('Function '+symbol)[-1].split('Function ')[0]
            if block==resource: block=resource.split(symbol)[-1].split('Function')[0]
            assert re.search(r'REG\s*[:=]\s*168\b',block),('register gate',symbol,block[:500])
            assert re.search(r'LOCAL\s*[:=]\s*0\b',block),('local/spill gate',symbol,block[:500])
    return {'source_sha256':digest,'extension_name':ext.__name__,'so':str(so),'so_sha256':sha(so),
            'SASS_sha256':sha(outdir/(label+'.sass')),'gate':'before prepare / first candidate math',
            'resource_file':label+'.resources'}

def tensors(value, prefix=''):
    import torch
    if isinstance(value,torch.Tensor): yield prefix,value
    elif isinstance(value,dict):
        for k,v in value.items(): yield from tensors(v,prefix+'/'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value): yield from tensors(v,prefix+'/'+str(i))

def contract_source_check():
    """Names select tensors only after the exact source and dispatch were bound."""
    bindings={}
    for label,(folder,pin) in PINS.items():
        source=(BASE/'candidates'/folder/'submission.py').read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==pin
        tree=ast.parse(source)
        funcs={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('kernel_fn','_r6','_v12','_gvcall','_rscall','_rqcall'):
            assert name in funcs
        assert 'rp["n"] == n' in funcs['kernel_fn']
        assert '_rqcall(state, rp, x.contiguous(), out) == 0' in funcs['kernel_fn']
        assert '(GVI[p[0]][9] >> 27) & 5' in funcs['_gvcall']
        assert 'p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0;' in source
        assert 'p.cnt[bz * GX + bx] = 0;' in source
        assert 'q.cnt[tile] = 0;' in source
        bindings[label]={'source_sha256':pin,'dispatch_sha256':{name:hashlib.sha256(body.encode()).hexdigest() for name,body in funcs.items() if name in ('kernel_fn','_r6','_v12','_gvcall','_rscall','_rqcall')},
                         'RQ':'two arrival/poller arrays reset by last poller',
                         'GV_regular':'S reduction last CTA resets tile; no counter used for other regular modes',
                         'V12':'run/run2/run3 last reducer resets tile',
                         'active_persistent':'HOLD: exact launch-count/progress contract not admitted'}
    return bindings

def select_counter_contract(mod,state,layer,m,n):
    rp=state.get('rq',{}).get((layer,m))
    if rp is not None and rp['n']==n:
        return 'RQ',rp['cnt']
    p=state['sel'].get((layer,mod._bk(m)))
    origin='sel'
    if p is None or p[4]!=n:
        p=state['gv'].get((layer,mod._bk(m)));origin='gv'
    if p is not None and p[4]==n:
        i=p[0]
        if 0<=i<1000 and (mod.GVI[i][9]>>27)&5:
            raise AssertionError(('HOLD active persistent P1',origin,i))
        if i==-7:
            raise AssertionError(('HOLD active router protocol',origin,p[1]))
        if i>=1000:return 'RR_no_counter',None
        assert 0<=i<1000,('HOLD unknown GV instance',i)
        v=mod.GVI[i][9];splits=mod.GVI[i][6]
        if splits==1 or ((v>>26)&15)!=0:return 'GV_no_counter',None
        if not (v&4):
            # sm90 cluster branch has no global counter; pre-sm90 fallback differs.
            raise AssertionError(('HOLD cluster architecture-specific counter contract',origin,i))
        return 'GV_regular',state['gvcnt']
    if mod._cublas(layer,m) or (layer=='router' and m>32):return 'LT_no_counter',None
    return 'V12',state['cnt']

def counter_verdict(contract,initial,final):
    # Pure semantic gate, shared by GPU checks and CPU positive/negative tests.
    if contract=='reset0':return all(v==0 for v in final)
    if contract=='dormant_unchanged':return initial==final
    raise AssertionError(('unreviewed counter contract',contract))

def counter_storage_contract(t,active):
    return 'reset0' if tensor_identity(t) in active else 'dormant_unchanged'

def counter_cpu_proof():
    assert counter_verdict('dormant_unchanged',[19,3],[19,3])
    assert counter_verdict('reset0',[0,0],[0,0])
    negatives={'selected_RQ_nonzero':('reset0',[0,0],[1,0]),
               'dormant_mutation':('dormant_unchanged',[19,3],[20,3]),
               'unrelated_rw_change':('dormant_unchanged',[7,2],[8,2])}
    for name,args in negatives.items():assert not counter_verdict(*args),name
    class Mod:
        GVI=[(0,)*9+(1<<27,), (0,1,0,1,0,0,1,1,0,0), (0,1,0,1,0,0,2,1,0,4)]
        @staticmethod
        def _bk(m):return m
        @staticmethod
        def _cublas(layer,m):return False
    baseline={'rq':{},'sel':{},'gv':{},'cnt':'base','gvcnt':'gv'}
    rq=dict(baseline,rq={('layer',32):{'n':64,'cnt':'selected'}})
    assert select_counter_contract(Mod,rq,'layer',32,64)==('RQ','selected')
    assert select_counter_contract(Mod,rq,'layer',32,65)==('V12','base')
    p1=dict(baseline,sel={('layer',32):(0,None,None,None,64)})
    try:select_counter_contract(Mod,p1,'layer',32,64)
    except AssertionError:pass
    else:raise AssertionError('active persistent path must HOLD')
    for instance,want in ((1,('GV_no_counter',None)),(2,('GV_regular','gv'))):
        state=dict(baseline,sel={('layer',32):(instance,None,None,None,64)})
        assert select_counter_contract(Mod,state,'layer',32,64)==want
    class Storage:
        values=[0,0,0,0]
        def data_ptr(self):return 1234
        def nbytes(self):return 16
    class Tensor:
        device='cpu';dtype='torch.int32';shape=(2,)
        def __init__(self,storage,offset=0):self.storage=storage;self.offset=offset
        def untyped_storage(self):return self.storage
        def storage_offset(self):return self.offset
        def stride(self):return (1,)
        def element_size(self):return 4
        def as_strided(self,shape,stride,offset):
            assert shape==(4,) and stride==(1,) and offset==0
            return self
        def detach(self):return self
        def cpu(self):return self
        def contiguous(self):return self
        def reshape(self,*args):return self
        def tolist(self):return list(self.storage.values)
        def numpy(self):return self
        def tobytes(self):
            import struct
            return struct.pack('4i',*self.storage.values)
    storage=Storage();a=Tensor(storage);b=Tensor(storage,2)
    plans=dict(baseline,rq={('layer',16):{'n':64,'cnt':a},('layer',32):{'n':64,'cnt':b}})
    chosen=select_counter_contract(Mod,plans,'layer',32,64)[1]
    active={tensor_identity(chosen)}
    assert counter_storage_contract(a,active)==counter_storage_contract(b,active)=='reset0'
    initial=counter_snapshot('alias',a)
    assert initial['values']==[0,0,0,0]
    storage.values=[0,0,1,0]
    final=counter_snapshot('offset_alias',b)
    assert not counter_verdict(counter_storage_contract(a,active),initial['values'],final['values'])
    assert final['values']==[0,0,1,0] and initial['digest']!=final['digest']
    return {'shared_RQ_storage_and_offset_alias_nonzero_rejected':'PASS','failure_full_values_retained':'PASS','valid_nonzero_dormant' :'PASS','reset0_positive':'PASS','negative_rejections':list(negatives),
            'dispatch_n_mismatch_fallback':'PASS','active_persistent_HOLD':'PASS','source_bindings':contract_source_check()}

def tensor_identity(t):
    return (str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())

def counter_snapshot(name,t):
    assert str(t.dtype)=='torch.int32',('HOLD counter dtype',name,str(t.dtype))
    c=t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0).detach().cpu().contiguous()
    values=c.reshape(-1).tolist()
    return {'name':name,'storage':tensor_identity(t),'dtype':str(t.dtype),'shape':list(t.shape),
            'storage_offset':t.storage_offset(),'stride':list(t.stride()),
            'digest':hashlib.sha256(c.numpy().tobytes()).hexdigest(),'nonzero':sum(v!=0 for v in values),
            'small_values':values[:32],'values':values}

def run(args):
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    verify(); assert args.root_review_token, 'root must supply independent critic artifact path'
    review=ROOT/args.root_review_token; assert review.is_file()
    result={'status':'RUNNING','driver_version':'v3-source-bound-counter-contracts','scope':args.shapes,'review':str(review),'review_sha256':sha(review),
            'source_pins':PINS,'phases':[],'raw':[],'jit':{},'shapes':{},'started':time.time()}
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output/'result.json'
    def phase(s):
        result['phases'].append({'phase':s,'time':time.time()});dump(output,result);print(s,flush=True)
    try:
        weights=reference.generate_weights(reference.WEIGHTS_SEED)
        modules={};states={};registry={}
        for label in LABELS:
            phase('compile_and_gate_'+label); mod=modules[label]=load(label)
            result['jit'][label]=jit_gate(mod,label,args.output,registry)
            # Reset only import-time budget clock after compile; production prepare semantics otherwise unchanged.
            # Build is normally overlapped with prepare; removing that overlap is disclosed and all picks recorded.
            mod._T0=time.time();phase('actual_full_prepare_'+label)
            states[label]=mod.prepare(weights);torch.cuda.synchronize()
        result['counter_source_bindings']=contract_source_check()
        result['prepared_policy']={}
        for label,state in states.items():
            result['prepared_policy'][label]={}
            for shape in args.shapes:
                layer,m=SHAPES[shape];n=reference.LAYERS[layer][0]
                rp=state.get('rq',{}).get((layer,m))
                pick={'rq':{k:v for k,v in (rp or {}).items() if isinstance(v,(int,float,str))},
                      'rq_name':modules[label].RQPROD2[rp['i']] if rp else None,
                      'sel':str(state.get('sel',{}).get((layer,modules[label]._bk(m)))),
                      'gv':str(state.get('gv',{}).get((layer,modules[label]._bk(m)))),
                      'lt':str(state.get('lt',{}).get((layer,m)))}
                try:pick['counter_path']=select_counter_contract(modules[label],state,layer,m,n)[0]
                except AssertionError as error:pick['counter_path_HOLD']=str(error)
                result['prepared_policy'][label][shape]=pick
        dump(output,result)
        # Hold a declined RQ launch rather than silently accepting an unvalidated fallback.
        for label,mod in modules.items():
            original=mod._rqcall
            def guarded(st,p,x,out,original=original,label=label):
                code=original(st,p,x,out)
                assert code==0,('HOLD RQ fallback',label,code)
                return code
            mod._rqcall=guarded
        immutable=[];counters=[];scratch=[];seen={}
        for label,state in states.items():
            for name,t in tensors(state):
                if not t.numel():continue
                identity=tensor_identity(t)
                if 'cnt' in name:
                    counters.append((label,name,t,counter_snapshot(label+name,t)));continue
                # Preserve separate immutable views, because packed/raw aliases can
                # have different offsets and shapes in one physical allocation.
                view=(identity,t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                if view in seen:continue
                seen[view]=True
                if 'ws' in name or 'sink' in name:
                    scratch.append({'name':label+name,'storage':identity});continue
                immutable.append((label+name,t,t.clone()))
        result['scratch']=scratch
        result['counter_prepare_baseline']=[{k:v for k,v in initial.items() if k!='values'} for _,_,_,initial in counters]
        dump(output,result)
        active={};context={}
        def invariants():
            changed=[name for name,t,s in immutable if not torch.equal(t,s)]
            records=[];failures=[]
            for label,name,t,initial in counters:
                final=counter_snapshot(label+name,t)
                contract=counter_storage_contract(t,active.get(label,set()))
                ok=counter_verdict(contract,initial['values'],final['values'])
                record={'name':label+name,'contract':contract,'PASS':ok,'context':dict(context),
                        'initial':initial if not ok else {k:v for k,v in initial.items() if k!='values'},
                        'final':final if not ok else {k:v for k,v in final.items() if k!='values'}}
                records.append(record)
                if not ok:failures.append(record)
            if changed or failures:
                result['invariant_failure']={'immutable_changed':changed,'counter_failures':failures,'all_counter_diagnostics':records}
                dump(output,result)
                raise AssertionError(('invariants',changed,[r['name'] for r in failures]))
        for shape in args.shapes:
            layer,m=SHAPES[shape];n,k=reference.LAYERS[layer]
            config={'m':m,'n':n,'k':k,'layer':layer}
            entry=result['shapes'][shape]={'picks':{},'validation':[]}
            active={}
            for label,state in states.items():
                path,counter=select_counter_contract(modules[label],state,layer,m,n)
                active[label]={tensor_identity(counter)} if counter is not None else set()
            context.update(shape=shape,stage='prepared_baseline')
            invariants()
            for label in LABELS:
                state=states[label];mod=modules[label];rp=state.get('rq',{}).get((layer,m))
                entry['picks'][label]={'rq':{key:value for key,value in (rp or {}).items() if isinstance(value,(int,float,str))},
                    'rq_name':mod.RQPROD2[rp['i']] if rp else None,
                    'sel':str(state.get('sel',{}).get((layer,mod._bk(m)))),
                    'gv':str(state.get('gv',{}).get((layer,mod._bk(m)))), 'lt':str(state.get('lt',{}).get((layer,m)))}
                def fn(x,out,mod=mod,state=state):out.copy_(mod.kernel_fn({'x':x,'config':config},state));return 0
                phase(shape+'_basis_FP64_'+label)
                ok,msg=mod._rqval(fn,weights[layer],m,n,torch.Generator(device='cuda').manual_seed(10001),4e-3)
                assert ok,(label,msg);context.update(label=label,stage='basis_FP64');invariants();entry['validation'].append({'label':label,'full_K_basis_FP64':msg})
                if rp:
                    ext=rp.get('mod',mod._rqmod[0]);x=torch.zeros((m,k),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                    attrs=selected_attributes(ext,rp,x,out,state)
                    entry['picks'][label]['selected_attrs_regs_local_shared']=attrs
                    entry['picks'][label]['cursor_affected_selected']=name_affected = mod.RQPROD2[rp['i']] in ('t32s5hze','t32s6hze','t16s5hze')
                    name=entry['picks'][label]['rq_name']
                    if label=='increment' and name in ('t32s5hze','t32s6hze','t16s5hze'):
                        assert attrs==[168,0,72840 if m==16 else 99488 if 's6' in name else 83080],attrs
            x=torch.empty((m,k),dtype=torch.bfloat16,device='cuda');graphs={};outs={}
            fixture=lambda seed,dist='normal':reference.generate_input(m,layer,seed,dist=dist)['x']
            fixtures={seed:fixture(seed) for seed in range(20000,20100)}
            for label in LABELS:
                x.copy_(fixtures[20000]);mod,state=modules[label],states[label]
                for _ in range(2):mod.kernel_fn({'x':x,'config':config},state)
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=mod.kernel_fn({'x':x,'config':config},state)
                graphs[label]=g;outs[label]=out
            def check(label,fresh,kind):
                context.update(label=label,stage=kind)
                assert bool(torch.isfinite(outs[label]).all()),(label,kind,'nonfinite')
                ok,msg=reference.check_implementation({'x':fresh,'config':config},outs[label],weights)
                assert ok,(label,kind,msg);invariants()
            probes=torch.zeros_like(x);rows=torch.arange(m,device='cuda');probes[rows,(rows*37+19)%k]=8
            for kind,fresh in [('normal',fixtures[20000]),('outliers',fixture(20999,'outliers')),('zero',torch.zeros_like(x)),('all_probes',probes),('fresh_normal',fixtures[20001])]:
                x.copy_(fresh)
                for label in LABELS:
                    outs[label].fill_(float('nan'))
                    for name,t in tensors(states[label]):
                        if name in ('/ws','/gvws') and t.is_floating_point():t.fill_(float('nan'))
                    graphs[label].replay();torch.cuda.synchronize();check(label,fresh,kind)
                    assert torch.equal(x,fresh),'X changed'
                    if kind=='zero':assert not int(torch.count_nonzero(outs[label]))
                    entry['validation'].append({'label':label,'graph_mutation':kind,'NaN_output_fresh_workspace':'PASS'})
            flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda');last={}
            def one(label,seed,mode):
                fresh=fixtures[seed];x.copy_(fresh);torch.sum(flush,0,out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
                us=a.elapsed_time(b)*1000;assert us>0 and us<float('inf')
                record={'shape':shape,'label':label,'seed':seed,'mode':mode,'us':us}
                result['raw'].append(record)
                with (args.output/'raw.jsonl').open('a') as journal:journal.write(json.dumps(record)+'\n');journal.flush()
                last[label]=seed
                return us
            def terminal(mode):
                # First operation after final timed launch is read-only output oracle, before any replay/mutation.
                assert torch.equal(x,fixtures[result['raw'][-1]['seed']])
                for label in LABELS:check(label,fixtures[last[label]],'immediate_'+mode)
                entry['validation'].append({'immediate_final_timed_output':mode,'last_fixtures':dict(last),'PASS':True})
                dump(output,result)
            phase(shape+'_timing')
            for label in LABELS:
                for _ in range(3):one(label,20000,'warmup')
            f1={}
            for label in LABELS[:-1]: f1[label+'_before']=[one(label,20001+i,'F1') for i in range(11)]
            f1['increment']=[one('increment',20001+i,'F1') for i in range(11)]
            for label in reversed(LABELS[:-1]):f1[label+'_after']=[one(label,20001+i,'F1') for i in range(11)]
            terminal('F1');entry['F1_medians_us']={key:statistics.median(v) for key,v in f1.items()}
            entry['F1_control_over_increment']={label:min(statistics.median(f1[label+'_before']),statistics.median(f1[label+'_after']))/statistics.median(f1['increment']) for label in LABELS[:-1]}
            orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)];pairs=[]
            for i in range(30):
                row={}
                for j in orders[i%6]:row[LABELS[j]]=one(LABELS[j],20040+i,'paired')
                pairs.append(row)
            terminal('paired');entry['paired_control_over_increment_median_ratio']={label:statistics.median(row[label]/row['increment'] for row in pairs) for label in LABELS[:-1]}
            entry['paired_all_samples']=pairs
        result['status']='VALID_FOCUSED_DEVELOPMENT_ONLY_FULL_SCOPE_REQUIRED';phase('complete')
    except BaseException:
        result['status']='INVALID_NO_PERFORMANCE_ADMISSION';result['failure']=traceback.format_exc();dump(output,result);raise
    finally:
        result['ended']=time.time();result['cleanup']='process exit releases CUDA; root independently verifies container/PID/GPU absence';dump(output,result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--output',type=Path)
    ap.add_argument('--shapes',nargs='+',choices=SHAPES,default=['qkv32']);ap.add_argument('--root-review-token')
    args=ap.parse_args()
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output;run(args)
if __name__=='__main__':main()
