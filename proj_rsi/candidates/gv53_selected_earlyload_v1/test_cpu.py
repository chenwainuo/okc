"""Execute actual late-chooser/dispatch bodies with CPU transports and native mocks."""
import ast,types,json,hashlib,re
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[2];torch.set_num_threads(1)
nodes={n.name:n for n in ast.parse((D/'submission.py').read_text()).body if isinstance(n,ast.FunctionDef)}
TARGET=(1,1,1,2,16,8,1,1,0,3);base=torch.full((4096,32),.125,dtype=torch.bfloat16)
class Weight:
 shape=(4096,8192);device=base.device
 def __getitem__(self,k):return self
 def double(self):return base.double()
proxy=types.SimpleNamespace(**{n:getattr(torch,n)for n in['Generator','bfloat16','uint8','empty','equal','isfinite']});proxy.randn=lambda shape,**kw:torch.randn((shape[0],32),**kw)
def actual(names,env):exec(compile(ast.Module(body=[nodes[n]for n in names],type_ignores=[]),'<actualCPUhost>','exec'),env)
def run():
 result=[];rng_before=torch.random.get_rng_state().clone()
 for m,helper,store,alternative in[(1,'_gvkey_confirm','gvkey_choices','keyed'),(8,'_gvearly_confirm','gvearly_choices','earlyload')]:
  for case,old,new,expect,opts in [('gain',21,20,alternative,{}),('loss',20,21,'parent',{}),('tie',21,21,'parent',{}),('tiny_positive',21.000001,21,alternative,{}),('invalid_timing',0,21,'parent',{}),('rq',21,20,'parent',{'rq':True}),('sel',21,20,alternative,{'sel':True}),('wrong_tuple',21,20,'parent',{'wrong':True}),('budget_start',21,20,'parent',{'start':True}),('partial_budget',21,20,'parent',{'partial':True}),('numeric_error',21,20,'RAISE',{'bad':True})]:
   calls=[];p=(0,'packed','escape',7,4096,'packed2',9);owner='sel'if opts.get('sel')else'gv';st={'sel':{},'gv':{},'rq':{},'pooler_raw':Weight()};st[owner][('pooler',m)]=p
   if opts.get('rq'):st['rq'][('pooler',m)]={'n':4096}
   def launch(state,plan,x,out):
    calls.append(alternative if len(plan)==8 else'parent');out.copy_((x.double()@base.double().T).to(torch.bfloat16))
    if opts.get('bad')and len(plan)==8:out.fill_(float('nan'))
    return 0
   def tm(fn,flush,it):assert it==9;fn();return(old if calls[-1]=='parent'else new)/1000
   clock=lambda:241 if opts.get('start')or(opts.get('partial')and len(calls)>=3)else 1
   env={'torch':proxy,'GVI':[(0,)*10 if opts.get('wrong')else TARGET],'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=clock),'_gvcall':launch,'_tm':tm};actual(['_bk',helper],env)
   try:env[helper](st,None)
   except AssertionError:assert expect=='RAISE';result.append({'m':m,'case':case,'failure_propagated':True});continue
   assert expect!='RAISE';rec=st[store][('pooler',m)];assert rec['choice']==expect,(m,case,rec)
   assert(len(st[owner][('pooler',m)])==8)==(expect==alternative)
   assert st[owner][('pooler',m)][:7]==p
   if rec['status']=='COMPLETE_ZERO_THRESHOLD':assert len(calls)==8 and[x['order']for x in rec['pairs_us']]==['AB','BA','AB','BA']
   result.append({'m':m,'case':case,'choice':rec['choice'],'status':rec['status'],'calls':len(calls)})
 assert torch.equal(torch.random.get_rng_state(),rng_before),'privategenerator advancedglobalRNG'
 calls=[]
 class Ext:
  def gvkeyrun(self,*a):calls.append('keyed');return 0
  def gvearlyrun(self,*a):calls.append('earlyload');return 0
  def gvrun(self,*a):calls.append('parent');return 0
 ext=Ext();env={'torch':torch,'GVI':[TARGET],'_gvext':lambda:ext};actual(['_bk','_gvcall'],env);p=(0,1,2,3,4096,4,5);st={'gvws':1,'gvcnt':2}
 for m in range(1,17):
  for marker in[None,'gv_keyed_choice','gv_earlyload_choice']:
   plan=p+(marker,)if marker else p;env['_gvcall'](st,plan,torch.empty((m,8192)),torch.empty((m,4096)))
   expected='keyed'if marker=='gv_keyed_choice'and m==1 else'earlyload'if marker=='gv_earlyload_choice'and m==8 else'parent';assert calls[-1]==expected,(m,marker,calls[-1])
 assert env['_bk'](5)==env['_bk'](7)==8
 # Whole file integration inverse plus exact measured kernel reuse.
 delta=json.loads((D/'delta.json').read_text());s=(D/'submission.py').read_text()
 for e in reversed(delta):assert s.count(e['new'])==1;s=s.replace(e['new'],e['old'],1)
 assert s==(D.parent/'gv53_keyed_choice_v1/submission.py').read_text()
 def parse(p):return{c.splitlines()[0].strip():re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',c)for c in p.read_text().split('Function : ')[1:]}
 native=parse(D/'pilot.sass');prior=parse(R/'results/proj-rsi/scheduler/artifacts/research.gv53.earlyload.fixed.01/collection-1793143210466227704/logs/scheduler/gv53-earlyload-fixed-01/run/actual.sass')
 for name,rows in prior.items():assert native[name]==rows,('early/controlnativechanged',name)
 oldkey=parse(D.parent/'gv53_keyed_choice_v1/host-target.sass');key=next(k for k in native if'gv_keyed'in k);assert native[key]==oldkey[key]
 assert not torch.cuda.is_initialized()
 out={'state':'PASS_INTEGRATED_CPU_CHOOSER_DISPATCH_NATIVE','cases':result,'actual_m1_through16_marker_routes':48,'M5_M7_actual_bucket_fallback_tested':True,'global_RNG_unchanged':True,'whole_source_inverse_to78f':True,'three_native_streams_fullPC_predicate_encoding_exact':True,'three_targets_register64_stack_spill0':'compile.stderr','CUDA_initialized':False,'math_transport_K32_shape_proxy':True}
 (D/'cpu-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
