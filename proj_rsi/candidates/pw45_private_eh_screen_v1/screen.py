"""Root-run narrow screen. CPU check imports no torch/production source; no queue/API."""
import argparse, hashlib, importlib.util, json, math, re, statistics, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import build

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def schedule():
 # Eight balanced forward/reverse Latin orders; each arm appears at every position.
 for i in range(32):
  order=list(range(4));shift=i%4;order=order[shift:]+order[:shift]
  if (i//4)%2:order.reverse()
  yield i,order

def cpu_check():
 proof=build.check()
 manifest=HERE/'screen-manifest.json'
 if manifest.exists():
  for rel,pin in json.loads(manifest.read_text())['pins'].items():assert sha(ROOT/rel)==pin,('screen dependency drift',rel)
 orders=list(schedule());assert len(orders)==32
 assert all(sorted(o)==list(range(4)) for _,o in orders)
 for position in range(4):assert [sum(o[position]==arm for _,o in orders) for arm in range(4)]==[8]*4
 proof['scope']={'shapes':['qkv_gate128/KS7','pooler128/KS4'],'arms':['actual_current_parent','actual_composed_policy','private_free_z851970','private_free_hz851971'],'paired_batches':32,'fullscope':False,'minimum_gain':None,'GPU_calls':0}
 return proof

def run(args):
 cpu_check()
 import torch
 sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
 H=load(ROOT/'proj_rsi/tools/pw44_stack_lab_v2.py','private_screen_helpers')
 S=load(ROOT/'proj_rsi/tools/pw45_stack_lab.py','policy_screen_helpers')
 device=torch.cuda.get_device_properties(torch.cuda.current_device())
 assert device.major==9 and device.minor==0 and device.multi_processor_count>=128
 args.output.mkdir(parents=True,exist_ok=False)
 result={'state':'RUNNING_NO_PERFORMANCE_ADMISSION','parent':args.parent,'device':str(device),'prepare':{},'runtime':{},'shapes':{},'raw':[],'jit_errors':[],'source_proof':cpu_check(),'limits':['Only two affected M128 shapes; no all30 or official gain claim.','Actual prepared policies are distinct from fixed private arms.','No minimum effect size or estimator agreement gate.']}
 def save():
  (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
 d=json.loads((HERE/'composition.json').read_text())['compositions'][args.parent]
 weights=reference.generate_weights(reference.WEIGHTS_SEED);modules={};states={};graphs=[]
 fixed_weights={layer:(weights[layer],weights[layer].clone()) for layer in ['qkv_gate','pooler']}
 def intact():
  for name,(t,before) in fixed_weights.items():assert torch.equal(t.view(torch.uint8),before.view(torch.uint8)),name
 try:
  for label,path in [('parent',ROOT/d['parent']),('composed',HERE/d['file'])]:
   mod=load(path,'screen_'+label);modules[label]=mod;original=mod.load_inline
   def checked(*a,_fn=original,_label=label,**kw):
    production=kw['cuda_sources'];private='rq_go<128,5,1,0,0,0,851971>' in re.sub(r'\s+','',str(production))
    try:
     if private:
      kw=dict(kw);kw['cuda_sources']=production+H.PRIVATE_INFO;kw['cpp_sources']+='\nint64_t pw44_private_info(int64_t,int64_t);\n';kw['functions']=list(kw['functions'])+['pw44_private_info']
     ext=_fn(*a,**kw)
     if private:
      prefix=args.output/('private-'+str(kw['name']));sass=subprocess.check_output(['cuobjdump','--dump-sass',ext.__file__],text=True);resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',ext.__file__],text=True)
      Path(str(prefix)+'.cu').write_text(kw['cuda_sources']);Path(str(prefix)+'.sass').write_text(sass);Path(str(prefix)+'.resources').write_text(resources)
      H.full_rq_gate(production,sass,resources)
      attrs={str(flag):{} for flag in H.PRIVATE_FLAGS};result['runtime']=attrs
      fields=['regs','local','smem','occupancy','max_dynamic','max_threads','user_static_shared','flag_echo','driver_reserved_shared','shared_per_SM','max_optin_per_block']
      for flag in H.PRIVATE_FLAGS:
       for code,key in enumerate(fields,1):attrs[str(flag)][key]=int(ext.pw44_private_info(flag,code));save()
      for flag in H.PRIVATE_FLAGS:H.private_attributes(flag,attrs[str(flag)],device.multi_processor_count,hashlib.sha256(production.encode()).hexdigest(),sha(ext.__file__),1024)
     return ext
    except BaseException as error:result['jit_errors'].append(repr(error));save();raise
   mod.load_inline=checked
   started=time.monotonic();states[label]=mod.prepare(weights);torch.cuda.synchronize()
   assert not result['jit_errors'] and not states[label].get('cpu')
   result['prepare'][label]={'seconds':time.monotonic()-started,'source_sha256':sha(path),'original_clock_elapsed':time.time()-mod._T0};save()
  assert result['runtime'],'private actual loaded resource gate missing'
  flush=torch.ones(64<<20,device='cuda',dtype=torch.float32);sink=torch.empty((),device='cuda')
  for layer,n,ks in [('qkv_gate',2304,7),('pooler',4096,4)]:
   mod=modules['composed'];w=weights[layer];k=w.shape[1];config={'m':128,'n':n,'k':k,'layer':layer}
   packed=mod._rqpack(w);assert packed is not None;lay,tab,esc6=packed
   assert torch.equal(mod._rqunpack(lay,tab,esc6,n,k),w.view(torch.int16).int()&65535)
   esc,_=mod._rqesc(esc6,n//128,k//64,ks,w.device);cnt=torch.zeros(2*(n//128),device='cuda',dtype=torch.int32)
   fixed=[(t,t.clone()) for t in (lay,esc)];own={'rq_e':torch.empty(0,device='cuda',dtype=torch.int64),'rq_sink':torch.zeros(4,device='cuda',dtype=torch.int32)}
   entry=result['shapes'][layer]={'picks':{},'terminals':[],'mutations':[],'basis':[]};work={};baselines={};plans={};calls={};labels=['parent','composed','free_z','free_hz']
   def local_intact():
    intact();assert not bool(torch.any(cnt))
    for t,before in fixed:assert torch.equal(t.view(torch.uint8),before.view(torch.uint8))
   for label in labels:
    x=reference.generate_input(128,layer,12000)['x'];out=torch.empty((128,n),device='cuda',dtype=torch.bfloat16)
    if label in modules:
     m=modules[label];st=states[label]
     def fn(xx,oo,m=m,st=st):oo.copy_(m.kernel_fn({'x':xx,'config':config},st));return 0
     fn(x,out);torch.cuda.synchronize();info,plan=S.counter_plan(m,st,layer,128,n,k);entry['picks'][label]=info;plans[label]=plan
    else:
     flag=851970 if label=='free_z' else 851971;name='ehpriv_z_free' if label=='free_z' else 'ehpriv_hz_free'
     p=dict(i=mod.RQPROD.index(name),lay=lay,tab=tab,esc=esc,cnt=cnt,KS=ks,n=n)
     def fn(xx,oo,p=p):code=mod._rqcall(own,p,xx,oo);assert code==0;return code
     entry['picks'][label]={'fixed_flag':flag,'KS':ks}
    # Existing source validator covers full K basis and FP64/random/outlier checks only on touched shapes.
    def checked_basis(xx,oo,fn=fn):code=fn(xx,oo);torch.cuda.synchronize();local_intact();return code
    ok,msg=mod._rqval(checked_basis,w,128,n,torch.Generator(device='cuda').manual_seed(12001),mod.RQTOL[ks]);assert ok,(label,msg)
    entry['basis'].append({'label':label,'message':msg})
    fn(x,out);torch.cuda.synchronize()
    if label in states:baselines[label]=S.counter_arena(states[label])
    g=torch.cuda.CUDAGraph();graphs.append(g)
    with torch.cuda.graph(g):
     if label in modules:out=m.kernel_fn({'x':x,'config':config},st)
     else:fn(x,out)
    torch.cuda.synchronize();work[label]=(x,out,g);calls[label]=0
   def oracle(label,fixture,kind):
    x,out,g=work[label];assert torch.equal(x.view(torch.uint8),fixture.view(torch.uint8));ok,msg=reference.check_implementation({'x':fixture,'config':config},out,weights);assert ok and bool(torch.isfinite(out).all()),(label,msg)
    if label in states:S.counter_check(baselines[label],S.counter_arena(states[label]),plans[label],calls[label])
    local_intact();entry['terminals'].append({'label':label,'kind':kind,'message':msg})
   def one(label,seed,kind):
    fixture=reference.generate_input(128,layer,seed)['x'];x,out,g=work[label];x.copy_(fixture);torch.sum(flush,0,out=sink)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();torch.cuda.synchronize();calls[label]+=1;us=a.elapsed_time(b)*1000;assert math.isfinite(us) and us>0
    result['raw'].append({'layer':layer,'label':label,'seed':seed,'kind':kind,'us':us});return us,fixture
   f1={}
   for block,label in enumerate(['parent','composed','free_z','free_hz','parent']):
    rows=[one(label,12100+i,'F1') for i in range(11)];oracle(label,rows[-1][1],'F1-block'+str(block));f1[str(block)+':'+label]=statistics.median(x[0] for x in rows)
   paired=[]
   for i,order in schedule():
    row={}
    for index in order:
     label=labels[index];row[label],fixture=one(label,12200+i,'paired');oracle(label,fixture,'paired'+str(i))
    paired.append(row)
   entry['F1_medians_us']=f1;entry['paired_parent_minus_arm_us']={label:statistics.median(r['parent']-r[label] for r in paired) for label in labels[1:]}
   entry['F1_parent_minus_arm_us']={label:min(f1['0:parent'],f1['4:parent'])-f1[str(index)+':'+label] for index,label in enumerate(labels[1:],1)}
   for label in labels:
    for kind,dist in [('zero','normal'),('outliers','outliers')]:
     fixture=reference.generate_input(128,layer,12300,dist=dist)['x']
     if kind=='zero':fixture.zero_()
     x,out,g=work[label];x.copy_(fixture);out.fill_(float('nan'));g.replay();torch.cuda.synchronize();calls[label]+=1;oracle(label,fixture,kind);entry['mutations'].append([label,kind])
   for x,out,g in work.values():g.reset();graphs.remove(g)
   save()
  result['state']='VALID_TWO_SHAPE_SCREEN_NO_FULL_POLICY_OR_OFFICIAL_CLAIM'
 finally:
  torch.cuda.synchronize()
  for g in graphs:g.reset()
  result['graphs_drained']=True;save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--parent',choices=list(build.PARENTS),default='pw45');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2))
 else:
  assert a.output,'--output required for root-owned GPU run';run(a)
