"""Focused actual confirmation-body test on CPU Torch with mocked timing/native launch."""
import ast,json,types
from pathlib import Path
import torch
D=Path(__file__).resolve().parent
source=(D/'submission.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_gvpred_confirm')
torch.set_num_threads(1)
results=[]
for name,old_us,new_us,expected in [('win',21,20,'predxor'),('loss',20,21,'original'),('tie',21,21,'original'),('tiny_win',21.000001,21,'predxor'),('invalid_timing',0,21,'original')]:
 w=torch.full((4096,32),.125,dtype=torch.bfloat16);p=(0,1,2,3,4096,4,5)
 state={'sel':{},'gv':{('pooler',1):p},'pooler_raw':w};calls=[]
 def launch(st,plan,x,out):
  calls.append('predxor'if len(plan)==8 else'original');out.copy_((x.double()@w.double().T).to(torch.bfloat16));return 0
 def timing(fn,flush,it):
  fn();assert it==9;return(old_us if calls[-1]=='original'else new_us)/1000
 env={'torch':torch,'GVI':[(1,1,1,2,16,8,1,1,0,3)],'_bk':lambda m:m,'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=lambda:1),'_gvok':lambda *a:True,'_gvcall':launch,'_tm':timing}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-confirm-body-CPU-transport>','exec'),env);env['_gvpred_confirm'](state,None)
 record=state['gvpred_choices'][('pooler',1)];assert record['choice']==expected
 assert(len(state['gv'][('pooler',1)])==8)==(expected=='predxor')
 if old_us:
  assert [r['order']for r in record['pairs_us']]==['AB','BA','AB','BA'];assert len(calls)==8
 else:assert record['status']=='INVALID_TIMING_ORIGINAL_RETAINED'and len(calls)==1
 results.append({'case':name,'choice':record['choice'],'status':record['status'],'calls':len(calls)})
assert not torch.cuda.is_initialized()
print(json.dumps({'state':'PASS_ACTUAL_CONFIRM_BODY_CPU_TRANSPORT','cases':results,'native_launches_mocked':True,'small_K32_CPU_fixture_not_kernel_numerics':True,'CUDA_initialized':False},indent=2))
