"""Reuse PW45 fixed-graph validation pattern for matched parent/new ST5/ST6."""
import statistics
LABELS=('parent5','new5','parent6','new6')
SHAPES={'pooler':(4096,8192,4),'unpooler':(8192,4096,2)}
BLOCKS=('parent5','new5','parent6','new6','parent6','new5','parent5')
def current_x_bytes(torch,x,expected):
    assert torch.equal(x.view(torch.uint8),expected.view(torch.uint8)),'fixed current X bytes changed'
def plan():
    rows=[]
    for label in LABELS:rows.extend(dict(label=label,seed=850000,mode='warmup') for i in range(5))
    for block,label in enumerate(BLOCKS):rows.extend(dict(label=label,seed=851000+21*block+i,mode='F1') for i in range(21))
    orders=[]
    for seq in (tuple(range(4)),tuple(reversed(range(4)))):orders.extend(seq[j:]+seq[:j] for j in range(4))
    for rep in range(32):rows.extend(dict(label=LABELS[i],seed=852000+rep,mode='paired') for i in orders[rep%8])
    assert len(rows)==295;return rows

def terminal(A,torch,S,outputs,x,fixtures,last,raw,oracle,integrity,counter_states,phase):
    assert set(outputs)==set(last)==set(LABELS) and len({o.data_ptr() for o in outputs.values()})==4
    current_x_bytes(torch,x,fixtures[raw[-1]['seed']]);records=[]
    for label in LABELS:
        seed=next(r['seed'] for r in reversed(raw) if r['label']==label and r['mode']!='warmup');assert last[label]==seed
        assert bool(torch.isfinite(outputs[label]).all());msg=oracle(fixtures[seed],outputs[label]);integrity()
        records.append(dict(label=label,phase=phase,seed=seed,oracle=msg,actual_terminal_before_any_replay=True,physical_counters=S.arena_receipt(A.counter_arena(counter_states[label])),physical_sinks=S.sink_receipt(A,counter_states[label],torch)))
    return records

def make_arm(A,torch,mod,ext,geometry,w,ks,ext_geometries):
    n,k=w.shape;lay,tab,esc6=mod._rqpack(w);assert type(tab) is int
    assert torch.equal(mod._rqunpack(lay,tab,esc6,n,k),w.view(torch.int16).int()&65535)
    matches=[i for i,name in enumerate(mod.RQPROD2) if tuple(mod.RQV[mod.RQI[name]][1:8])==geometry];assert len(matches)==1
    assert geometry in ext_geometries[id(ext)]
    esc,_=mod._rqesc(esc6,n//128,k//64,ks,w.device)
    backing=torch.full((4*(n//128)+16,),0x19a75d3,dtype=torch.int32,device=w.device);counter=backing[8:-8];active=2*(n//128);counter[:active].zero_()
    sink=torch.full((4,),0x4735,dtype=torch.int32,device=w.device);empty=torch.empty(0,dtype=torch.int64,device=w.device)
    index=matches[0]
    def fn(x,out):
        code=int(ext.rqrun(index,lay,x,out,esc,counter,ks,tab,0,empty,sink,0));assert code==0;return code
    state={'cnt':counter,'sink':sink,'lay':lay,'esc':esc,'table':tab,'source_so_sha256':A.sha(ext.__file__),'geometry':geometry};immutable=[(t,t.clone()) for t in (lay,esc,sink,w)];plan={A.tensor_id(counter):{'mode':'reset0','offset':8,'reset_indices':list(range(active)),'changes':{}}}
    return fn,state,immutable,plan,esc6

def edge_weight(torch,mod,n,k):
    # One warp's first stage exercises all lane/register/half owners, 33 windows including the initial window;
    # later stages test sequence advance/tail; every other warp has empty escape lists.
    gen=torch.Generator(device='cuda').manual_seed(86011)
    codes=torch.randint(0,512,(n//16,k//64,4,32,8),dtype=torch.int32,device='cuda',generator=gen)
    fr=(0x3f00|(codes&255)|((codes>>8)<<15)).to(torch.int16).view(torch.bfloat16)
    bits=fr.view(torch.int16);bits[0,0,:,:,:]=0x2081;bits[0,1,0,0,0]=-24447;bits[0,31,3,31,7]=0x2081
    w=fr.view(n//16,k//64,4,8,4,2,2,2).permute(0,6,3,1,2,5,4,7).reshape(n,k).contiguous()
    assert torch.equal(mod._pack_nat(w).view(torch.uint8),fr.view(torch.uint8))
    return w

def run(A,torch,reference,modules,states,weights,result,graphs_live,save,ext_geometries):
    S=A.S;result['fixed_matched']={}
    for layer,(n,k,ks) in SHAPES.items():
        config={'layer':layer,'m':32,'n':n,'k':k};w=weights[layer]
        entry=result['fixed_matched'][layer]={'scope':config,'KS':ks,'raw':[],'validation':{},'edge_validation':{},'mutations':[],'terminal':{},'fixed_vs_actual_policy':'SEPARATE_NO_FORCED_SELECTED_POLICY'}
        funcs={};counter_states={};plans={};immutable=[];calls={l:0 for l in LABELS}
        for label,(policy,geometry) in S.FIXED.items():
            mod=modules[policy];ext=mod._rqmod2[0];fn,state,im,rule,_=make_arm(A,torch,mod,ext,geometry,w,ks,ext_geometries)
            funcs[label]=fn;counter_states[label]=state;plans[label]=rule;immutable+=im
        baselines={l:A.counter_arena(s) for l,s in counter_states.items()}
        def integrity():
            for a,b in immutable:current_x_bytes(torch,a,b)
            for label in LABELS:A.counter_check(baselines[label],A.counter_arena(counter_states[label]),plans[label],calls[label])
        entry['capture_warmup']={'calls_per_label':2,'fixture_seed':850000}
        entry['baseline']={l:{'counters':S.arena_receipt(baselines[l]),'sinks':S.sink_receipt(A,counter_states[l],torch)} for l in LABELS}
        for label,fn in funcs.items():
            mod=modules[S.FIXED[label][0]]
            def counted(x,out,fn=fn,label=label):
                before=x.clone();code=fn(x,out);calls[label]+=1;current_x_bytes(torch,x,before);return code
            gen=torch.Generator(device=w.device).manual_seed(86003);tol=mod.RQTOL[ks]
            ok,msg=mod._rqval(counted,w,32,n,gen,tol);assert ok,(layer,label,msg)
            entry['validation'][label]={'all_basis_columns':k,'onehot':'EXACT','FP64_normal_outlier':msg};integrity()
        # Real CUDA edge/fresh-W validation uses a different weight allocation and all-K basis.
        ew=edge_weight(torch,modules['pw45'],n,k)
        for label,(policy,geometry) in S.FIXED.items():
            mod=modules[policy];fn,state,im,rule,esc6=make_arm(A,torch,mod,mod._rqmod2[0],geometry,ew,ks,ext_geometries)
            assert int(esc6.shape[0])==1026,('edge fixture escape coverage',layer,esc6.shape)
            rows=esc6.detach().cpu().tolist();edge_table=state['table'];coverage=S.verify_edge_rows(rows,edge_table)
            before=A.counter_arena(state);ecalls=[0]
            def counted_edge(x,out):
                xb=x.clone();code=fn(x,out);ecalls[0]+=1;current_x_bytes(torch,x,xb);return code
            gen=torch.Generator(device=w.device).manual_seed(86009);ok,msg=mod._rqval(counted_edge,ew,32,n,gen,mod.RQTOL[ks]);assert ok,(layer,label,'escape-edge',msg)
            torch.cuda.synchronize();A.counter_check(before,A.counter_arena(state),rule,ecalls[0])
            for a,b in im:current_x_bytes(torch,a,b)
            entry['edge_validation'][label]={'fresh_W':True,'all_basis_columns':k,'onehot':'EXACT','FP64':msg,'escape_entries':1026,'actual_escape_metadata_coverage':coverage,'same_stage_all_32lanes_16registers_both_halves':True,'window_boundary_entries':list((31,32,63,64,1023,1024)),'empty_other_warp_lists':True,'tail_stage':31,'physical_counters':S.arena_receipt(A.counter_arena(state)),'physical_sinks':S.sink_receipt(A,state,torch)}
        x=reference.generate_input(32,layer,850000)['x'];outputs={l:torch.empty((32,n),dtype=torch.bfloat16,device=w.device) for l in LABELS};graphs={}
        try:
            for label,fn in funcs.items():
                fn(x,outputs[label]);fn(x,outputs[label]);calls[label]+=2;torch.cuda.synchronize();g=torch.cuda.CUDAGraph();graphs_live.append(g)
                with torch.cuda.graph(g):fn(x,outputs[label])
                graphs[label]=g
            fixtures={row['seed']:reference.generate_input(32,layer,row['seed'])['x'] for row in plan()};last={}
            flush=torch.ones(64<<20,dtype=torch.float32,device=w.device);sink_flush=torch.empty((),dtype=torch.float32,device=w.device)
            def oracle(fresh,out):
                ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),msg;return msg
            for i,row in enumerate(plan()):
                if i==167:entry['terminal']['F1']=terminal(A,torch,S,outputs,x,fixtures,last,entry['raw'],oracle,integrity,counter_states,'F1')
                x.copy_(fixtures[row['seed']]);torch.sum(flush,0,out=sink_flush);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[row['label']].replay();b.record();torch.cuda.synchronize();calls[row['label']]+=1
                us=a.elapsed_time(b)*1000
                def invalid_elapsed(record):result.setdefault('invalid_measurements',[]).append(record);save()
                us=S.elapsed_gate(us,dict(row,layer=layer),invalid_elapsed);entry['raw'].append(dict(row,us=us));last[row['label']]=row['seed']
                with (S.ARTIFACT_ROOT.parent/'fixed-raw.jsonl').open('a') as journal:
                    import json;journal.write(json.dumps(dict(row,layer=layer,us=us))+'\n')
            entry['terminal']['paired']=terminal(A,torch,S,outputs,x,fixtures,last,entry['raw'],oracle,integrity,counter_states,'paired')
            med=[statistics.median(r['us'] for r in entry['raw'][20+21*i:41+21*i]) for i in range(7)]
            entry['F1_block_medians_us']=med;fm={'parent5':(med[0]+med[6])/2,'new5':(med[1]+med[5])/2,'parent6':(med[2]+med[4])/2,'new6':med[3]}
            pairs=[{r['label']:r['us'] for r in entry['raw'][167+4*i:171+4*i]} for i in range(32)];entry['paired_all_samples']=pairs;entry['contrasts']={}
            for control,new in (('parent5','new5'),('parent6','new6'),('parent5','parent6'),('new5','new6')):
                entry['contrasts'][control+'_minus_'+new]={'F1_endpoint_average_gain_us':fm[control]-fm[new],'paired_median_gain_us':statistics.median(r[control]-r[new] for r in pairs),'paired_median_ratio':statistics.median(r[control]/r[new] for r in pairs),'wins_ties_losses':[sum(r[control]>r[new] for r in pairs),sum(r[control]==r[new] for r in pairs),sum(r[control]<r[new] for r in pairs)]}
            probes=torch.zeros_like(x);rr=torch.arange(32,device=w.device);probes[rr,(rr*37+19)%k]=8
            for name,fresh in [('zero',torch.zeros_like(x)),('normal',reference.generate_input(32,layer,86501)['x']),('outliers',reference.generate_input(32,layer,86503,dist='outliers')['x']),('probes',probes)]:
                x.copy_(fresh)
                for label,g in graphs.items():
                    outputs[label].fill_(float('nan'));g.replay();torch.cuda.synchronize();calls[label]+=1;current_x_bytes(torch,x,fresh);msg=oracle(fresh,outputs[label]);integrity()
                    if name=='zero':assert not int(torch.count_nonzero(outputs[label]))
                    entry['mutations'].append(dict(label=label,input=name,oracle=msg,NaN_poison_overwritten=True))
            from torch.profiler import profile,ProfilerActivity
            entry['profiles']={}
            for label,g in graphs.items():
                expected_fixture=x.clone();torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as p:g.replay();torch.cuda.synchronize();calls[label]+=1
                events=[e.name() for e in p.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA];geometry=S.FIXED[label][1];token='rq<'+', '.join(map(str,geometry))+'>'
                assert len(events)==1 and token in events[0];current_x_bytes(torch,x,expected_fixture);oracle(expected_fixture,outputs[label]);integrity()
                entry['profiles'][label]={'kernels':events,'policy':S.FIXED[label][0],'source_so_sha256':counter_states[label]['source_so_sha256'],'expected_geometry':geometry,'logical_calls_since_baseline':calls[label],'current_X_bytes':'UNCHANGED','counters':S.arena_receipt(A.counter_arena(counter_states[label])),'sinks':S.sink_receipt(A,counter_states[label],torch)}
            entry['state']='VALID_MATCHED_FIXED_NO_SELECTED_POLICY_INFERENCE';save()
        finally:
            for g in graphs.values():torch.cuda.synchronize();g.reset();graphs_live.remove(g)
            entry['cleanup']='DRAINED_RESET_ALL_CREATED';save()
