"""Matched three-arm GV mechanism screen, stdlib-only until authorized run."""
import statistics
import gv_support as G
LABELS=('parent','wrapper_control','predxor')
ORDERS=((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
def plan():
 rows=[]
 for label in LABELS:rows.extend(dict(label=label,seed=810000,mode='warmup')for _ in range(5))
 for block,label in enumerate((LABELS[0],LABELS[1],LABELS[2],LABELS[1],LABELS[0])):rows.extend(dict(label=label,seed=811000+9*block+i,mode='F1')for i in range(9))
 for i in range(32):rows.extend(dict(label=LABELS[j],seed=812000+i,mode='paired')for j in ORDERS[i%6])
 assert len(rows)==156;return rows
def verify_terminal(last,raw,check,integrity):
 assert set(last)==set(LABELS)
 records=[]
 for label in LABELS:
  seed=next(r['seed']for r in reversed(raw)if r['label']==label and r['mode']!='warmup');assert last[label]==seed
  records.append(dict(label=label,seed=seed,oracle=check(label,seed),actual_terminal_before_any_replay=True))
 integrity();return records
def cpu_terminal_negatives():
 raw=[dict(label=l,seed=i,mode='F1')for i,l in enumerate(LABELS)];last={l:i for i,l in enumerate(LABELS)}
 assert len(verify_terminal(last,raw,lambda l,s:'PASS',lambda:None))==3
 caught=[]
 for kind in ('stale_seed','missing_arm','invalid_output','counter_mutation'):
  v=dict(last)
  if kind=='stale_seed':v['parent']=99
  if kind=='missing_arm':v.pop('predxor')
  def checker(l,s):
   assert kind!='invalid_output';return 'PASS'
  def integrity():assert kind!='counter_mutation'
  try:verify_terminal(v,raw,checker,integrity)
  except AssertionError:caught.append(kind)
  else:raise AssertionError('terminal negative accepted')
 return caught

def run(A,torch,reference,modules,states,weights,result,graphs_live,save,ext_geometries):
 parent=modules['pw45'];new=modules[G.NEW];state=states['pw45'];all_results=result['fixed_GV']={}
 fixed_weight_copies=[t.clone()for t in weights.values()]
 # Same immutable packedW/escape addresses for all three arms, private X/output/counters/workspaces.
 geometries=[]
 for t in G.TARGETS:
  if t[1]==1:geometries.extend([('pooler',m,t)for m in (1,8)])
  elif t[4]*t[5]*64==8192:geometries.append(('pooler',16,t))
  else:geometries.append(('unpooler',16,t))
 assert len(geometries)==11
 for layer,m,t in geometries:
  n=4096 if layer=='pooler'else 8192;k=8192 if layer=='pooler'else 4096
  key=layer+'/'+str(m)+'/'+'.'.join(map(str,t));entry=all_results[key]={'template':list(t),'config':{'layer':layer,'m':m,'n':n,'k':k},'raw':[],'validation':{},'mutations':[],'fixed_vs_actual_policy':'SEPARATE; does not force picker'};config=entry['config']
  packs=dict(state.get('gvc12',{}));assert layer in packs,'actual preparation did not retain the required verified pack'
  css,esc=packs[layer];cs=css[t[9]&1];fr=state[layer][0][:n//16]
  # Exact actual-W BF16 bits through the independent packdecoder.
  decoded=parent._unc12(cs[0][0],cs[0][1],cs[0][2],t[9]&1);assert torch.equal(decoded,parent._frag_bits(fr))
  p=parent._gxplan(state,layer,t,n,packs);assert p is not None and tuple(parent.GVI[p[0]])==t and parent.GVI==new.GVI
  immutable=[];seen=set()
  for tensor in (p[1],p[2],p[5]):
   identity=A.tensor_id(tensor)
   if identity not in seen:seen.add(identity);immutable.append((tensor,tensor.clone()))
  counters={};scratch={};funcs={};calls={l:0 for l in LABELS};guards={}
  oldext=parent._gvext();newext=new._gvext()
  for label in LABELS:
   # Whole physical backing includes both guard slices. S1 means no active global counter/workspace.
   cb=torch.full((65536+16,),0x19a75d3,dtype=torch.int32,device=weights[layer].device);counter=cb[8:-8]
   ws=torch.full_like(state['gvws'],17.25);counters[label]={'cnt':counter};scratch[label]=(ws,ws.clone())
   ext=oldext if label=='parent'else newext
   function=ext.gvpred_oldrun if label=='wrapper_control'else ext.gvrun
   def fn(x,out,function=function,counter=counter,ws=ws):
    code=int(function(p[0],p[1],p[5],x,out,p[2],ws,counter,p[3],p[6]));assert code==0,('fixed GV declined',key,code);return code
   funcs[label]=fn
  baselines={label:A.counter_arena(counters[label])for label in LABELS}
  def integrity():
   for a,b in immutable:assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
   for a,b in scratch.values():assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
   for label in LABELS:A.counter_check(baselines[label],A.counter_arena(counters[label]),{},calls[label])
   for xb,ob in guards.values():
    assert bool((xb[:16]==11.5).all()) and bool((xb[-16:]==11.5).all()),'fixed X backing guard changed'
    assert bool((ob[:16]==17.25).all()) and bool((ob[-16:]==17.25).all()),'fixed output backing guard changed'
   for tensor,copy in zip(weights.values(),fixed_weight_copies):assert torch.equal(tensor.view(torch.uint8),copy.view(torch.uint8))
  for label,fn in funcs.items():
   def counted(x,out,fn=fn,label=label):
    before=x.clone();code=fn(x,out);torch.cuda.synchronize();calls[label]+=1;assert torch.equal(x.view(torch.uint8),before.view(torch.uint8));return code
   gen=torch.Generator(device=weights[layer].device).manual_seed(82003)
   ok,msg=parent._rqval(counted,weights[layer],m,n,gen,4e-3);assert ok,(key,label,msg)
   entry['validation'][label]={'all_K_basis':k,'all_K_bits':'EXACT','FP64_normal_outlier':msg,'tol':4e-3,'actual_W_pack_bitproof':True};integrity()
   for dist in ('normal','outliers'):
    data=reference.generate_input(m,layer,82007,dist=dist);out=torch.full((m,n),float('nan'),dtype=torch.bfloat16,device=data['x'].device);counted(data['x'],out)
    ok,msg=reference.check_implementation(data,out,weights);assert ok and bool(torch.isfinite(out).all()),(label,key,msg);entry['validation'][label][dist]=msg;integrity()
  work={};last={};schedule=plan();fixtures={r['seed']:reference.generate_input(m,layer,r['seed'])['x']for r in schedule}
  flush=torch.ones(64<<20,dtype=torch.float32,device=weights[layer].device);sink=torch.empty((),dtype=torch.float32,device=flush.device)
  try:
   for label,fn in funcs.items():
    xb=torch.full((m*k+32,),11.5,dtype=torch.bfloat16,device=weights[layer].device);x=xb[16:-16].reshape(m,k);x.copy_(fixtures[810000])
    ob=torch.full((m*n+32,),17.25,dtype=torch.bfloat16,device=x.device);out=ob[16:-16].reshape(m,n);guards[label]=(xb,ob)
    fn(x,out);fn(x,out);calls[label]+=2;torch.cuda.synchronize();g=torch.cuda.CUDAGraph();graphs_live.append(g)
    with torch.cuda.graph(g):fn(x,out)
    torch.cuda.synchronize();A.counter_check(baselines[label],A.counter_arena(counters[label]),{},calls[label]);work[label]=(x,out,g)
   assert len({v[0].data_ptr()for v in work.values()})==3 and len({v[1].data_ptr()for v in work.values()})==3
   def oracle(label,seed):
    x,out,g=work[label];fresh=fixtures[seed];assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8))
    ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(key,label,msg);return msg
   entry['terminal']={}
   for i,row in enumerate(schedule):
    if i==60:entry['terminal']['F1']=verify_terminal(last,entry['raw'],oracle,integrity);save()
    label=row['label'];x,out,g=work[label];x.copy_(fixtures[row['seed']]);torch.sum(flush,0,out=sink)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();torch.cuda.synchronize();calls[label]+=1
    us=a.elapsed_time(b)*1000;assert 0<us<float('inf');record=dict(row,us=us);entry['raw'].append(record);last[label]=row['seed']
    with(A.Path(result['_output'])/'fixed-raw.jsonl').open('a')as f: f.write(A.json.dumps(dict(record,geometry=key))+'\n');f.flush()
    save()
   entry['terminal']['paired']=verify_terminal(last,entry['raw'],oracle,integrity)
   f1=[entry['raw'][15+9*b:15+9*(b+1)]for b in range(5)];med=[statistics.median(r['us']for r in block)for block in f1]
   pairs=[{r['label']:r['us']for r in entry['raw'][60+3*i:63+3*i]}for i in range(32)];entry['paired']=pairs;entry['F1_medians_us']=dict(zip(('parent_before','wrapper_before','predxor','wrapper_after','parent_after'),med))
   fmed={'parent':(med[0]+med[4])/2,'wrapper_control':(med[1]+med[3])/2,'predxor':med[2]}
   entry['contrasts']={a+'_minus_'+b:{'paired_median_us':statistics.median(r[a]-r[b]for r in pairs),'F1_endpoint_average_us':fmed[a]-fmed[b],'wins':sum(r[a]>r[b]for r in pairs)}for a,b in [('parent','wrapper_control'),('parent','predxor'),('wrapper_control','predxor')]}
   probe=torch.zeros_like(fixtures[810000]);rr=torch.arange(m,device=probe.device);probe[rr,(rr*37+19)%k]=8
   for kind,fresh in [('zero',torch.zeros_like(probe)),('normal',reference.generate_input(m,layer,82501)['x']),('outliers',reference.generate_input(m,layer,82503,dist='outliers')['x']),('all_probes',probe),('fresh_normal',reference.generate_input(m,layer,82507)['x'])]:
    for label,(x,out,g)in work.items():
     x.copy_(fresh);out.fill_(float('nan'));g.replay();torch.cuda.synchronize();calls[label]+=1
     assert torch.equal(x.view(torch.uint8),fresh.view(torch.uint8));ok,msg=reference.check_implementation({'x':fresh,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(key,label,kind,msg)
     if kind=='zero':assert not int(torch.count_nonzero(out))
     integrity();entry['mutations'].append({'label':label,'kind':kind,'oracle':msg,'currentX':'EXACT','NaN_output_overwritten':True})
   from torch.profiler import profile,ProfilerActivity
   entry['profiles']={}
   for label,(x,out,g)in work.items():
    current=x.clone();torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA])as prof:g.replay();torch.cuda.synchronize();calls[label]+=1
    events=[e.name()for e in prof.profiler.kineto_results.events()if e.device_type()==torch.autograd.DeviceType.CUDA]
    token=('gv_predxor<'if label=='predxor'else'gv<')+', '.join(map(str,t))+'>';assert len(events)==1 and token in events[0],(key,label,events,token)
    assert torch.equal(x.view(torch.uint8),current.view(torch.uint8));ok,msg=reference.check_implementation({'x':current,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all());integrity();entry['profiles'][label]={'kernels':events,'nodes':1,'oracle':msg}
   entry['state']='VALID_FIXED_MECHANISM_NO_POLICY_ADMISSION';save()
  finally:
   for x,out,g in work.values():torch.cuda.synchronize();g.reset();graphs_live.remove(g)
   entry['cleanup']='DRAINED_RESET_ALL_CREATED';save()
