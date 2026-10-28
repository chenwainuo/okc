"""Actual production helper/dispatch CPU transports, no CUDA/native campaign."""
import ast,hashlib,json,pathlib,types
import torch
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];torch.set_num_threads(1)
text=(D/'submission.py').read_text();nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
parent=(D.parent/'gv53_selected_earlyload_v1/submission.py').read_text();pnodes={n.name:n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)}
for n in ['_gvkey_confirm','_gvearly_confirm','_gxbake','_rsbake','_rqbake','_rqsbake','_tm']:
 assert ast.dump(nodes[n])==ast.dump(pnodes[n]),('parent helper changed',n)
T=(1,2,2,1,4,16,1,1,2,1);P=(1,1,1,2,16,8,1,1,0,3)
base=torch.full((8192,32),.125,dtype=torch.bfloat16)
class Weight:
 shape=(8192,4096);device=base.device
 def __getitem__(self,k):return self
 def double(self):return base.double()
proxy=types.SimpleNamespace(**{n:getattr(torch,n) for n in ['Generator','bfloat16','uint8','empty','equal','isfinite']});proxy.randn=lambda shape,**kw:torch.randn((shape[0],32),**kw)
def actual(names,env):exec(compile(ast.Module(body=[nodes[n] for n in names],type_ignores=[]),'<actual host CPU>','exec'),env)
def run():
 before=torch.random.get_rng_state().clone();results=[]
 cases=[('gain',21,20,'xprefetch',{}),('loss',20,21,'parent',{}),('tie',21,21,'parent',{}),('tiny_positive',21.000001,21,'xprefetch',{}),('invalid_timing',0,21,'parent',{}),('rq',21,20,'parent',{'rq':True}),('sel',21,20,'xprefetch',{'sel':True}),('wrong_tuple',21,20,'parent',{'wrong':True}),('budget_start',21,20,'parent',{'start':True}),('partial_budget',21,20,'parent',{'partial':True}),('numeric_error',21,20,'RAISE',{'bad':True}),('no_parent',21,20,'parent',{'none':True}),('marked_parent',21,20,'parent',{'marked':True})]
 for case,old,new,expect,opt in cases:
  calls=[];p=(0,'packed','escape',7,8192,'packed2',9);owner='sel' if opt.get('sel') else 'gv';st={'sel':{},'gv':{},'rq':{},'unpooler_raw':Weight()}
  if not opt.get('none'):st[owner][('unpooler',16)]=p+('existing_marker',) if opt.get('marked') else p
  if opt.get('rq'):st['rq'][('unpooler',16)]={'n':8192}
  def launch(state,plan,x,out):
   calls.append('xprefetch' if len(plan)==8 else 'parent');out.copy_((x.double()@base.double().T).to(torch.bfloat16))
   if opt.get('bad') and len(plan)==8:out.fill_(float('nan'))
   return 0
  def tm(fn,flush,it):assert it==9;fn();return (old if calls[-1]=='parent' else new)/1000
  clock=lambda:241 if opt.get('start') or (opt.get('partial') and len(calls)>=3) else 1
  env={'torch':proxy,'GVI':[(0,)*10 if opt.get('wrong') else T],'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=clock),'_gvcall':launch,'_tm':tm}
  actual(['_bk','_gvxprefetch_confirm'],env)
  try:env['_gvxprefetch_confirm'](st,None)
  except AssertionError:
   assert expect=='RAISE';results.append({'case':case,'numeric_failure_propagated':True});continue
  assert expect!='RAISE';rec=st['gvxprefetch_choices'][('unpooler',16)];assert rec['choice']==expect,(case,rec)
  if not(opt.get('none') or opt.get('marked')):assert (len(st[owner][('unpooler',16)])==8)==(expect=='xprefetch') and st[owner][('unpooler',16)][:7]==p
  if rec['status']=='COMPLETE_ZERO_THRESHOLD':assert len(calls)==8 and [r['order']for r in rec['pairs_us']]==['AB','BA','AB','BA']
  results.append({'case':case,'choice':rec['choice'],'status':rec['status'],'calls':len(calls)})
 assert torch.equal(before,torch.random.get_rng_state()),'global RNG changed'
 calls=[]
 class Ext:
  def gvkeyrun(self,*a):calls.append('keyed');return 0
  def gvearlyrun(self,*a):calls.append('earlyload');return 0
  def gvxprefetchrun(self,*a):calls.append('xprefetch');return 0
  def gvrun(self,*a):calls.append('parent');return 0
 env={'torch':torch,'GVI':[P,T],'_gvext':lambda:Ext()};actual(['_bk','_gvcall'],env);st={'gvws':1,'gvcnt':2};count=0
 for m in range(1,33):
  for route,i,k,n in [('pooler',0,8192,4096),('unpooler',1,4096,8192)]:
   p=(i,1,2,3,n,4,5)
   for marker in [None,'gv_keyed_choice','gv_earlyload_choice','gv_xprefetch_choice']:
    plan=p+(marker,)if marker else p;x=torch.empty((m,k));out=torch.empty((m,n))
    wrong=(route=='unpooler' and ((marker=='gv_keyed_choice' and m==1) or (marker=='gv_earlyload_choice' and m==8))) or (route=='pooler' and marker=='gv_xprefetch_choice' and m==16)
    try:env['_gvcall'](st,plan,x,out)
    except AssertionError:assert wrong;count+=1;continue
    assert not wrong
    expect='keyed'if marker=='gv_keyed_choice'and m==1 else'earlyload'if marker=='gv_earlyload_choice'and m==8 else'xprefetch'if marker=='gv_xprefetch_choice'and m==16 else'parent'
    assert calls[-1]==expect,(m,route,marker,calls[-1]);count+=1
 # Exact source inverse and measured source-body identity, including unchanged inherited math.
 restored=text
 for e in reversed(json.loads((D/'delta.json').read_text())):assert restored.count(e['new'])==1;restored=restored.replace(e['new'],e['old'],1)
 assert restored==parent
 entry=next(e for e in json.loads((D/'delta.json').read_text()) if e['old']=='\n__global__ void p1zero');body=entry['new'][1:-len('\n__global__ void p1zero')]
 measured=R/'results/proj-rsi/scheduler/artifacts/research.gv.unpool16.xprefetch.fixed.01/collection-1793144168981231482/logs/scheduler/gv-unpool16-xprefetch-fixed-01/run/actual.cu'
 assert body in measured.read_text()
 assert not torch.cuda.is_initialized()
 obj={'state':'PASS_ACTUAL_PRODUCTION_HELPER_DISPATCH_CPU','source_sha256':hashlib.sha256(text.encode()).hexdigest(),'whole_source_inverse_to17d44d_exact':True,'native_body_exact_actual_measured_source':True,'M1_M8_and_original_bake_timing_helpers_AST_unchanged':True,'selection_cases':results,'actual_dispatch_cases':count,'global_RNG_unchanged':True,'CUDA_initialized':False,'transport_proxy':'CPU K32 math proxy with actual host shape metadata; not GPU arithmetic validation','reused_GPU_evidence':'Same native body and unchanged G parameter ABI/math; scoped actual fixedcell numerics/ownterminals/mutations/cleanup in gv-unpool16-xprefetch-result-v1'}
 (D/'cpu-proof.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj,indent=2))
if __name__=='__main__':run()
