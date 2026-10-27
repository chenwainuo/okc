"""Root runtime focused o32 experiment. Import is stdlib-only; no GPU/prepare calls."""
import statistics
LABELS=('old_st5_early_zi','half_st4_fullk','half_st5_fullk')
def current_x_bytes(torch,x,expected):
    assert torch.equal(x.view(torch.uint8),expected.view(torch.uint8)),'fixed current X bytes changed'
def plan():
    result=[]
    for label in LABELS:
        result.extend(dict(label=label,seed=810000,mode='warmup') for _ in range(5))
    for block,label in enumerate((LABELS[0],LABELS[1],LABELS[2],LABELS[1],LABELS[0])):
        result.extend(dict(label=label,seed=811000+21*block+i,mode='F1') for i in range(21))
    orders=((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
    for rep in range(30):result.extend(dict(label=LABELS[i],seed=812000+rep,mode='paired') for i in orders[rep%6])
    assert len(result)==210;return result

def terminal(outputs,x,fixtures,last,raw,check,integrity,torch):
    assert set(outputs)==set(last)==set(LABELS)
    assert len({out.data_ptr() for out in outputs.values()})==3
    current_x_bytes(torch,x,fixtures[raw[-1]['seed']])
    records=[]
    for label in LABELS:
        seed=next(r['seed'] for r in reversed(raw) if r['label']==label and r['mode']!='warmup')
        assert last[label]==seed
        assert bool(torch.isfinite(outputs[label]).all())
        records.append(dict(label=label,seed=seed,oracle=check(fixtures[seed],outputs[label]),actual_terminal_before_any_replay=True))
    integrity();return records

def run(A,torch,reference,modules,states,weights,result,graphs_live,save,ext_geometries):
    S=A.S;mod=modules['pw45'];w=weights['o'];config={'layer':'o','m':32,'n':8192,'k':1024}
    fixed=result['fixed_o32']={'scope':config,'raw':[],'validation':{},'mutations':[],'fixed_vs_actual_policy':'SEPARATE_FROM_FULL_STACK_PICKS','counter_plain':'all physical bytes unchanged','allocated_resource':'384threads/168regs/64512pool all; activeWG/shared/KS/store differ'}
    lay,tab,esc6=mod._rqpack(w);assert torch.equal(mod._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    ext=mod._rqmod2[0];assert ext is not None
    names=mod.RQPROD2;empty=torch.empty(0,dtype=torch.int64,device=w.device);sink=torch.zeros(1,dtype=torch.int32,device=w.device)
    assert type(tab) is int;table_word=tab;fixed['decode_table_scalar']=tab
    immutable=[(lay,lay.clone()),(sink,sink.clone())];funcs={};counter_states={};plans={}
    for label,(geometry,ks) in S.FIXED.items():
        matches=[i for i,name in enumerate(names) if tuple(mod.RQV[mod.RQI[name]][1:8])==geometry];assert len(matches)==1,(label,matches)
        index=matches[0];esc,_=mod._rqesc(esc6,64,16,ks,w.device);immutable.append((esc,esc.clone()))
        # Full visible256 words satisfy HALF host guard; all tail and guards remain dormant.
        backing=torch.full((272,),0x19a75d3,dtype=torch.int32,device=w.device);counter=backing[8:264]
        reset=list(range(128)) if geometry[-1]&2 else []
        if reset:counter[:128].zero_()
        counter_states[label]={'cnt':counter}
        plans[label]={A.tensor_id(counter):{'mode':'reset0','offset':8,'reset_indices':reset,'changes':{}}} if reset else {}
        assert geometry in ext_geometries[id(ext)],('fixed arm extension was not admitted',label)
        def fn(x,out,index=index,ks=ks,esc=esc,counter=counter,label=label):
            code=int(ext.rqrun(index,lay,x,out,esc,counter,ks,tab,0,empty,sink,0));assert code==0,(label,code);return 0
        funcs[label]=fn
    baselines={label:A.counter_arena(state) for label,state in counter_states.items()};calls={label:0 for label in LABELS}
    def integrity():
        assert tab==table_word
        assert all(torch.equal(a,b) for a,b in immutable)
        for label in LABELS:A.counter_check(baselines[label],A.counter_arena(counter_states[label]),plans[label],calls[label])
        assert all(torch.equal(t,reference_copy) for t,reference_copy in zip(weights.values(),fixed_weight_copies))
    fixed_weight_copies=[t.clone() for t in weights.values()]
    for label,fn in funcs.items():
        def counted(x,out,fn=fn,label=label):calls[label]+=1;before=x.clone();code=fn(x,out);current_x_bytes(torch,x,before);return code
        gen=torch.Generator(device=w.device).manual_seed(82003)
        tol=mod.RQTOL[S.FIXED[label][1]];ok,msg=mod._rqval(counted,w,32,8192,gen,tol);assert ok,(label,msg)
        fixed['validation'][label]=[dict(all1024_basis='EXACT',random_FP64='PASS',tolerance=tol,detail=msg)]
        for dist in ('normal','outliers'):
            data=reference.generate_input(32,'o',82007,dist=dist);before=data['x'].clone();out=torch.full((32,8192),float('nan'),dtype=torch.bfloat16,device=w.device)
            counted(data['x'],out);torch.cuda.synchronize();current_x_bytes(torch,data['x'],before);assert bool(torch.isfinite(out).all())
            ok,msg=reference.check_implementation(data,out,weights);assert ok,(label,msg)
            fixed['validation'][label].append(dict(dist=dist,oracle=msg,fresh_input_output=True));integrity()
    data=reference.generate_input(32,'o',810000);x=data['x'];outputs={l:torch.empty((32,8192),dtype=torch.bfloat16,device=w.device) for l in LABELS};graphs={}
    try:
        for label,fn in funcs.items():
            fn(x,outputs[label]);fn(x,outputs[label]);calls[label]+=2;torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph();graphs_live.append(graph)
            with torch.cuda.graph(graph):fn(x,outputs[label])
            graphs[label]=graph
        schedule=plan();fixtures={r['seed']:reference.generate_input(32,'o',r['seed'])['x'] for r in schedule}
        flush=torch.ones(64<<20,dtype=torch.float32,device=w.device);sink_flush=torch.empty((),dtype=torch.float32,device=w.device);last={}
        def oracle(fresh,out):
            ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok,msg;return msg
        fixed['terminal']={}
        for i,row in enumerate(schedule):
            if i==120:fixed['terminal']['F1']=terminal(outputs,x,fixtures,last,fixed['raw'],oracle,integrity,torch)
            x.copy_(fixtures[row['seed']]);torch.sum(flush,0,out=sink_flush)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[row['label']].replay();b.record();torch.cuda.synchronize();calls[row['label']]+=1
            us=a.elapsed_time(b)*1000;assert 0<us<float('inf');fixed['raw'].append(dict(row,us=us));last[row['label']]=row['seed'];save()
        fixed['terminal']['paired']=terminal(outputs,x,fixtures,last,fixed['raw'],oracle,integrity,torch)
        f1=[fixed['raw'][15+21*b:15+21*(b+1)] for b in range(5)];med=[statistics.median(r['us'] for r in b) for b in f1]
        pairs=[{r['label']:r['us'] for r in fixed['raw'][120+3*i:123+3*i]} for i in range(30)]
        fixed['F1_medians_us']=dict(zip(('old_before','st4_before','st5','st4_after','old_after'),med));fixed['paired']=pairs
        fixed['contrasts']={}
        fmed={LABELS[0]:(med[0]+med[4])/2,LABELS[1]:(med[1]+med[3])/2,LABELS[2]:med[2]}
        for control,new in ((LABELS[0],LABELS[1]),(LABELS[0],LABELS[2]),(LABELS[1],LABELS[2])):
            fixed['contrasts'][control+'_minus_'+new]={'F1_endpoint_average_gain_us':fmed[control]-fmed[new],'paired_median_gain_us':statistics.median(r[control]-r[new] for r in pairs),'paired_median_ratio':statistics.median(r[control]/r[new] for r in pairs),'wins':sum(r[control]>r[new] for r in pairs)}
        # Mutations/profile only after both actual retained terminal gates.
        probes=torch.zeros_like(x);rr=torch.arange(32,device=w.device);probes[rr,(rr*37+19)%1024]=8
        for name,fresh in [('zero',torch.zeros_like(x)),('normal',reference.generate_input(32,'o',82501)['x']),('outliers',reference.generate_input(32,'o',82503,dist='outliers')['x']),('probes',probes)]:
            x.copy_(fresh)
            for label,g in graphs.items():
                outputs[label].fill_(float('nan'));g.replay();torch.cuda.synchronize();calls[label]+=1
                current_x_bytes(torch,x,fresh);assert bool(torch.isfinite(outputs[label]).all());msg=oracle(fresh,outputs[label])
                if name=='zero':assert not int(torch.count_nonzero(outputs[label]))
                integrity();fixed['mutations'].append(dict(label=label,input=name,oracle=msg,NaN_poison_overwritten=True))
        from torch.profiler import profile,ProfilerActivity
        fixed['profiles']={}
        for label,g in graphs.items():
            expected_fixture=x.clone()
            with profile(activities=[ProfilerActivity.CUDA]) as p:g.replay();torch.cuda.synchronize();calls[label]+=1
            events=[e.name() for e in p.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]
            geometry=S.FIXED[label][0];token='rq<'+', '.join(map(str,geometry))+'>'
            assert len(events)==1 and token in events[0],(label,events);fixed['profiles'][label]={'kernels':events,'current_x_bytes_unchanged':True};current_x_bytes(torch,x,expected_fixture);oracle(expected_fixture,outputs[label]);integrity()
        fixed['state']='VALID_FIXED_MECHANISM_NO_SELECTED_POLICY_CLAIM';save()
    finally:
        for graph in graphs.values():torch.cuda.synchronize();graph.reset();graphs_live.remove(graph)
        fixed['cleanup']='DRAINED_RESET_ALL_CREATED';save()
