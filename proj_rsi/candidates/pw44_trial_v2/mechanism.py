"""Matched-pointer private-output mechanism lab; root GPU execution only."""
import ast,hashlib,json,math,statistics,time
from pathlib import Path
FLAGS={'fixedEH_z':65538,'fixedEH_hz':65539,'private_z_bar':327682,'private_hz_bar':327683,'private_z_free':851970,'private_hz_free':851971}
NAMES={'fixedEH_z':'s4s5zEH','fixedEH_hz':'s4s5hzEH','private_z_bar':'ehpriv_z_bar','private_hz_bar':'ehpriv_hz_bar','private_z_free':'ehpriv_z_free','private_hz_free':'ehpriv_hz_free'}
LABELS=tuple(FLAGS)
ORDERS=[tuple((i+shift)%6 for i in (0,1,5,2,4,3)) for shift in range(6)]
F1_BLOCKS=('fixedEH_z','private_z_bar','private_z_free','private_z_bar','fixedEH_z','fixedEH_hz','private_hz_bar','private_hz_free','private_hz_bar','fixedEH_hz')

def raw_schedule():
    return [(label,9400+i,'F1',block) for block,label in enumerate(F1_BLOCKS) for i in range(11)]+[(LABELS[index],9420+i,'paired',i) for i in range(30) for index in ORDERS[i%6]]

def drive_schedule(one,terminal):
    for label,seed,mode,block in raw_schedule():
        one(label,seed,mode,block)
        if (mode=='F1' and seed==9410) or (mode=='paired' and block==29):terminal(label,seed,mode,block)

def terminal_gate(out,fixture,checker):
    # Must run immediately at the designated arm's last timed call, before any overwrite.
    ok,message=checker(fixture,out);assert ok,message
    return message

def summarize(raw):
    assert len(raw)==290 and [(r['label'],r['seed'],r['mode'],r['block']) for r in raw]==raw_schedule()
    paired=[{r['label']:r['us'] for r in raw[110+6*i:116+6*i]} for i in range(30)]
    blocks=[statistics.median(r['us'] for r in raw[11*i:11*(i+1)]) for i in range(10)]
    comparisons={}
    for hint,base in [('z',0),('hz',5)]:
        names=('fixedEH_'+hint,'private_'+hint+'_bar','private_'+hint+'_free');eh,bar,free=names
        fs={eh:min(blocks[base],blocks[base+4]),bar:min(blocks[base+1],blocks[base+3]),free:blocks[base+2]}
        comparisons[hint]={}
        for control,candidate in ((bar,free),(eh,bar),(eh,free)):
            values=[p[control]-p[candidate] for p in paired]
            comparisons[hint][control+' minus '+candidate]={'paired_median_us':statistics.median(values),'F1_conservative_us':fs[control]-fs[candidate],'wins_ties_losses':[sum(v>0 for v in values),sum(v==0 for v in values),sum(v<0 for v in values)],'all_deltas_us':values}
    return comparisons

def cpu_check():
    schedule=raw_schedule();assert len(schedule)==290
    assert all(sum(order[position]==index for order in ORDERS)==1 for index in range(6) for position in range(6))
    predecessors={(a,b):0 for a in range(6) for b in range(6) if a!=b}
    for order in ORDERS:
        for a,b in zip(order,order[1:]):predecessors[a,b]+=1
    assert set(predecessors.values())=={1}
    # Shared-output lifetime negative: later arm overwrote previous final output.
    class Output:seed=0
    out=Output();out.seed=9410
    checker=lambda expected,actual:(expected==actual.seed,'own last output')
    terminal_gate(out,9410,checker);out.seed=9449
    try:terminal_gate(out,9410,checker)
    except AssertionError:pass
    else:raise AssertionError('overwritten stale terminal output admitted')
    observed=[];shared={}
    def one(label,seed,mode,block):shared.update(label=label,seed=seed,mode=mode,block=block)
    def terminal(label,seed,mode,block):
        assert shared==dict(label=label,seed=seed,mode=mode,block=block),'actual schedule checked overwritten output'
        observed.append((label,seed,mode,block))
    drive_schedule(one,terminal);assert len(observed)==16
    deferred=[]
    drive_schedule(one,lambda *row:deferred.append(row))
    try:
        for row in deferred:terminal(*row)
    except AssertionError:pass
    else:raise AssertionError('deferred terminal after shared-output overwrite accepted')
    mock=[dict(label=l,seed=s,mode=m,block=b,us=1.+LABELS.index(l)*.01) for l,s,m,b in schedule]
    summarize(mock)
    return {'raw_per_shape':290,'raw_total':580,'six_Williams_orders_balance_all_positions_and_predecessors':True,'immediate_terminal_overwrite_negative_rejected':True,'real_drive_schedule_deferred_terminal_negative_rejected':True,'expected_F1_terminal_oracles':20,'expected_paired_terminal_oracles':12,'expected_mutations':48,'GPU_calls':0}

def run(torch,mod,weights,reference,output,result,save,private_runtime):
    start=time.monotonic();result.update(state='RUNNING_NO_GAIN_ADMISSION',stage_cap_seconds=300,raw=[],shapes={},matched_pointer_policy='All six arms share X/output/lay/esc/cnt; serial execution with immediate arm-last-output oracle before any other replay/mutation.',limits=['Compiler epilogue rescheduling is part of barrier-removal contrast; not intrinsic barrier instruction latency.','FixedEH arms use PW44 module exact old canonical kernels; they are not the actual PW38 prepared policy.'])
    def checkpoint():
        result['elapsed_seconds']=time.monotonic()-start;save();assert result['elapsed_seconds']<300,'narrow stage exceeds quoted300s; no performance admission'
    def identity(t):return [str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes(),t.storage_offset()]
    flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
    all_graphs=[]
    try:
        for layer,n,ks in [('qkv_gate',2304,7),('pooler',4096,4)]:
            checkpoint();w=weights[layer];k=int(w.shape[1]);packed=mod._rqpack(w);assert packed is not None
            lay,tab,esc6=packed;unpacked=mod._rqunpack(lay,tab,esc6,n,k)
            assert torch.equal(unpacked,w.view(torch.int16).int()&65535),'independent full packed-weight decode mismatch'
            esc,_dummy=mod._rqesc(esc6,n//128,k//64,ks,w.device);cnt=torch.zeros(2*(n//128),device=w.device,dtype=torch.int32);own={'rq_e':torch.empty(0,device=w.device,dtype=torch.int64),'rq_sink':torch.zeros(4,device=w.device,dtype=torch.int32)}
            x=torch.empty((128,k),device=w.device,dtype=torch.bfloat16);out=torch.empty((128,n),device=w.device,dtype=torch.bfloat16)
            fixed={key:(tensor,tensor.clone()) for key,tensor in [('weight',w),('packed',lay),('escape',esc)]}
            entry=result['shapes'][layer]={'geometry':{'m':128,'n':n,'k':k,'KS':ks,'grid_CTAs':n//128*ks},'shared_physical':{key:identity(t) for key,t in [('X',x),('output',out),('packed',lay),('escape',esc),('counter',cnt)]},'variants':{},'basis':[],'terminal_oracles':[],'mutations':[],'raw':[],'graphs':'LIVE'}
            def intact():
                assert not torch.any(cnt),'private mechanism full counter arena must reset0, never repaired'
                for key,(tensor,before) in fixed.items():assert torch.equal(tensor.view(torch.uint8),before.view(torch.uint8)),('mechanism immutable drift',layer,key)
            def check(fixture,actual):return reference.check_implementation({'x':fixture,'config':{'m':128,'n':n,'k':k,'layer':layer}},actual,{layer:w})
            fns={};graphs={}
            for label in LABELS:
                nm=NAMES[label];index=mod.RQPROD.index(nm);v=mod.RQV[mod.RQI[nm]];assert tuple(v[1:8])==(128,5,1,0,0,0,FLAGS[label])
                p=dict(i=index,lay=lay,esc=esc,cnt=cnt,KS=ks,tab=tab,n=n)
                def fn(xx,oo,p=p,label=label):
                    code=mod._rqcall(own,p,xx,oo);assert code==0,('mechanism launch declined',layer,label,code);return code
                fns[label]=fn
                gen=torch.Generator(device=w.device).manual_seed(4410+ks)
                def basis_fn(xx,oo):
                    code=fn(xx,oo);torch.cuda.synchronize();intact();return code
                ok,message=mod._rqval(basis_fn,w,128,n,gen,mod.RQTOL[ks]);torch.cuda.synchronize();assert ok,message;intact()
                entry['basis'].append({'label':label,'full_K_columns':k,'source_validator':message,'launch_status':'ALL0','counter':'RESET0'});checkpoint()
                fixture=reference.generate_input(128,layer,9399)['x'];x.copy_(fixture);fn(x,out);torch.cuda.synchronize();intact()
                graph=torch.cuda.CUDAGraph();all_graphs.append(graph)
                with torch.cuda.graph(graph):fn(x,out)
                torch.cuda.synchronize();intact();graphs[label]=graph
                entry['variants'][label]={'name':nm,'FL':FLAGS[label],'dispatch_index':index,'variant':list(v),'shared_physical':entry['shared_physical'],'runtime':private_runtime[str(FLAGS[label])]}
            fixtures={seed:reference.generate_input(128,layer,seed)['x'] for seed in range(9400,9450)}
            def one(label,seed,mode,block):
                fixture=fixtures[seed];x.copy_(fixture);torch.sum(flush,0,out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
                us=a.elapsed_time(b)*1000;assert math.isfinite(us) and us>0;intact();assert torch.equal(x.view(torch.uint8),fixture.view(torch.uint8))
                row={'shape':layer,'label':label,'seed':seed,'mode':mode,'block':block,'us':us};entry['raw'].append(row);result['raw'].append(row)
                with (output/'mechanism.raw.jsonl').open('a') as journal:journal.write(json.dumps(row)+'\n');journal.flush()
            def terminal(label,seed,mode,block):
                fixture=fixtures[seed]
                message=terminal_gate(out,fixture,check);intact();entry['terminal_oracles'].append({'label':label,'seed':seed,'mode':mode,'block':block,'oracle':message,'immediate_before_any_other_replay_or_mutation':True});checkpoint()
            drive_schedule(one,terminal)
            entry['comparisons']=summarize(entry['raw'])
            for label in LABELS:
                for kind in ('zero','outliers','all_probes','fresh_normal'):
                    fixture=reference.generate_input(128,layer,9460)['x']
                    if kind=='zero':fixture=torch.zeros_like(fixture)
                    elif kind=='outliers':fixture=reference.generate_input(128,layer,9461,dist='outliers')['x']
                    elif kind=='all_probes':
                        fixture=torch.zeros_like(fixture)
                        rows=torch.arange(128,device=w.device);fixture[rows,(rows*37+19)%k]=8.
                    x.copy_(fixture);out.fill_(float('nan'));graphs[label].replay();torch.cuda.synchronize();intact();assert torch.equal(x.view(torch.uint8),fixture.view(torch.uint8))
                    message=terminal_gate(out,fixture,check);entry['mutations'].append({'label':label,'kind':kind,'oracle':message,'output_poison_overwritten':bool(torch.isfinite(out).all()),'counter':'RESET0'});assert entry['mutations'][-1]['output_poison_overwritten']
                    if kind=='zero':assert not int(torch.count_nonzero(out))
                    checkpoint()
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset();all_graphs.remove(graph)
            entry['graphs']='RESET_AFTER_DRAIN';checkpoint()
        assert len(result['raw'])==580
        assert sum(len(e['terminal_oracles']) for e in result['shapes'].values())==32
        assert sum(len(e['mutations']) for e in result['shapes'].values())==48
        result['state']='NARROW_VALID_COMPLETE_NO_INTRINSIC_BARRIER_OR_FULL_POLICY_GAIN_CLAIM';checkpoint()
    finally:
        result['graph_cleanup_errors']=[]
        try:torch.cuda.synchronize()
        except Exception as error:result['graph_cleanup_errors'].append(repr(error))
        for graph in list(all_graphs):
            try:graph.reset();all_graphs.remove(graph)
            except Exception as error:result['graph_cleanup_errors'].append(repr(error))
        result['final_graph_list_empty']=not all_graphs;save()
        assert not result['graph_cleanup_errors'],result['graph_cleanup_errors']
    return result

def integration_cpu(source_path):
    """Actual run() with fake tensors/graphs verifies plumbing, counts and cnt sizing."""
    import types,tempfile
    source=ast.parse(source_path.read_text()).body
    def literal(name):return next(ast.literal_eval(n.value) for n in source if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    serial=[0];current=[None]
    class Tensor:
        def __init__(self,shape,seed=0,device='FAKE_CPU',dtype='bf16'):
            self.shape=tuple(shape);self.seed=seed;self.device=device;self.dtype=dtype;serial[0]+=1;self.ptr=serial[0]
        def numel(self):return math.prod(self.shape)
        def untyped_storage(self):return self
        def data_ptr(self):return self.ptr
        def nbytes(self):return self.numel()*4
        def storage_offset(self):return 0
        def clone(self):return Tensor(self.shape,self.seed,self.device,self.dtype)
        def view(self,*args):return self
        def int(self):return self
        def __and__(self,other):return self
        def __mul__(self,other):return self
        def __add__(self,other):return self
        def __mod__(self,other):return self
        def __setitem__(self,key,value):self.seed='probes'
        def copy_(self,other):self.seed=other.seed;return self
        def fill_(self,value):self.seed='POISON' if isinstance(value,float) and math.isnan(value) else value;return self
    def tensor(shape,*args,**kw):
        if isinstance(shape,int):shape=(shape,)
        return Tensor(shape,device=kw.get('device','FAKE_CPU'),dtype=kw.get('dtype','bf16'))
    class Graph:
        def __enter__(self):current[0]=self;return self
        def __exit__(self,*args):current[0]=None
        def replay(self):self.fn()
        def reset(self):pass
    class Event:
        def __init__(self,*args,**kw):pass
        def record(self):pass
        def elapsed_time(self,other):return .01
    class Generator:
        def __init__(self,*args,**kw):pass
        def manual_seed(self,*args):return self
    class Finite:
        def __init__(self,t):self.t=t
        def all(self):return self.t.seed!='POISON'
    torch=types.SimpleNamespace(empty=tensor,ones=tensor,zeros=tensor,empty_like=lambda t:t.clone(),zeros_like=lambda t:Tensor(t.shape,0),arange=lambda n,**kw:Tensor((n,)),sum=lambda *a,**k:None,any=lambda t:False,equal=lambda a,b:a.shape==b.shape and a.seed==b.seed,isfinite=Finite,count_nonzero=lambda t:0 if t.seed==0 else 1,Generator=Generator,int64='int64',int32='int32',int16='int16',uint8='uint8',bfloat16='bf16',float32='f32',cuda=types.SimpleNamespace(Event=Event,CUDAGraph=Graph,graph=lambda graph:graph,synchronize=lambda:None))
    weights={'qkv_gate':Tensor((2304,8192),'qkv_gate'),'pooler':Tensor((4096,8192),'pooler')}
    def packing(w):return Tensor((w.shape[0]//128,w.shape[1]//64,8,1536),w.seed),1,Tensor((0,6))
    def unpack(lay,tab,esc,n,k):return Tensor((n,k),lay.seed)
    mod=types.SimpleNamespace(RQPROD=literal('RQPROD'),RQV=literal('RQV'),RQTOL=literal('RQTOL'),_rqpack=packing,_rqunpack=unpack,_rqesc=lambda esc,nb,kc,ks,dev:(Tensor((ks*nb*8*32+32,)),Tensor((1,))))
    mod.RQI={v[0]:i for i,v in enumerate(mod.RQV)}
    def call(st,p,x,out):
        if p['cnt'].numel()<2*p['lay'].shape[0]:return -13
        if current[0] is not None:current[0].fn=lambda:call(st,p,x,out)
        out.seed=x.seed;return 0
    mod._rqcall=call
    def validate(fn,w,m,n,gen,tol):
        code=fn(Tensor((m,w.shape[1]),'basis'),Tensor((m,n)));return code==0,'fake CPU basis plumbing'
    mod._rqval=validate
    reference=types.SimpleNamespace(generate_input=lambda m,layer,seed,**kw:{'x':Tensor((m,8192),seed)},check_implementation=lambda data,out,weights:(data['x'].seed==out.seed,'fake CPU own terminal output'))
    runtime={str(flag):{'CPU_FAKE':True} for flag in FLAGS.values()};result={}
    with tempfile.TemporaryDirectory() as folder:run(torch,mod,weights,reference,Path(folder),result,lambda:None,runtime)
    assert len(result['raw'])==580 and len(result['shapes'])==2 and result['final_graph_list_empty']
    assert sum(len(e['terminal_oracles']) for e in result['shapes'].values())==32
    assert sum(len(e['mutations']) for e in result['shapes'].values())==48
    assert call({},dict(cnt=Tensor((1,)),lay=Tensor((18,128,8,1536))),Tensor((128,8192)),Tensor((128,2304)))==-13
    return {'actual_run_fake_tensor_graph_transport_PASS':True,'raw':580,'immediate_terminal_oracles':32,'mutations':48,'dummy_oneword_counter_negative_declines':True,'GPU_calls':0,'real_torch_imports':0}
