"""Targeted actual-helper transport edge cases; CPU Tensor, mocked launch/timer."""
import ast,json,types
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
source=(ROOT/'proj_rsi/candidates/gv_predxor_keepchoice_v1/submission.py').read_text();helper=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_gvpred_confirm')
torch.set_num_threads(1)
def exercise(kind):
 w=torch.full((4096,32),.125,dtype=torch.bfloat16);pack1,pack2,esc=object(),object(),object();p=(0,pack1,esc,3,4096,pack2,5)
 state={'sel':{},'gv':{('pooler',1):p},'pooler_raw':w};calls=[];clock=[0];validation=[0];input_ptrs=[];output_ptrs=[]
 if kind=='sel_owner':state['sel'][('pooler',1)]=p
 if kind=='RQ_precedence':state['rq']={('pooler',1):{'n':4096}}
 def now():
  clock[0]+=1
  expired=(kind=='budget_before' or kind=='budget_after_validation'and clock[0]>=2 or kind=='budget_mid_pairs'and clock[0]>=4 or kind=='budget_after_pairs'and clock[0]>=7)
  return 241 if expired else 1
 def validate(*args):validation[0]+=1;return kind!='validation_decline'
 def launch(st,plan,x,out):
  assert plan[:7]==p and plan[1]is pack1 and plan[2]is esc and plan[5]is pack2
  calls.append('new'if len(plan)==8 else'old');input_ptrs.append(x.data_ptr());output_ptrs.append(out.data_ptr())
  if kind=='changed_X':x.add_(1)
  out.copy_((x.double()@w.double().T).to(torch.bfloat16))
  if kind=='bad_own_output':out.fill_(float('nan'))
  return 0
 def timing(fn,flush,it):
  fn();assert it==9
  if kind=='invalid_nan':return float('nan')
  return .021 if calls[-1]=='old'else .020
 env={'torch':torch,'GVI':[(1,1,2,2,8,16,1,1,0,1)],'_bk':lambda m:m,'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=now),'_gvok':validate,'_gvcall':launch,'_tm':timing}
 exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual-helper-edge>','exec'),env)
 raised=False
 try:env['_gvpred_confirm'](state,None)
 except AssertionError:raised=True
 if kind in ('changed_X','bad_own_output'):
  assert raised and len(state['gv'][('pooler',1)])==7
 elif kind in ('budget_before','RQ_precedence'):
  assert not raised and not calls and not validation[0]and 'gvpred_choices'not in state
 else:
  assert not raised
  record=state['gvpred_choices'][('pooler',1)]
  if kind=='sel_owner':
   assert record['choice']=='predxor'and state['sel'][('pooler',1)][:7]==p
   assert len(state['sel'][('pooler',1)])==8 and state['gv'][('pooler',1)]is p
  else:
   assert record['choice']=='original'and state['gv'][('pooler',1)]is p
  if kind=='budget_mid_pairs':assert len(record['pairs_us'])==1
  if kind=='budget_after_pairs':assert len(record['pairs_us'])==4
  if kind=='budget_after_validation':assert not calls and validation[0]==1
  if kind=='validation_decline':assert record['status']=='NEW_VALIDATION_DECLINED_ORIGINAL_RETAINED'
 if calls:
  assert len(set(input_ptrs))==1
  if len(calls)>1:assert len(set(output_ptrs))==2
 return {'case':kind,'launch_callbacks':len(calls),'validation_callbacks':validation[0],'numeric_or_input_assertion_expected':raised,'CUDA_initialized':torch.cuda.is_initialized()}
results=[exercise(k)for k in ('sel_owner','RQ_precedence','budget_before','budget_after_validation','budget_mid_pairs','budget_after_pairs','validation_decline','invalid_nan','changed_X','bad_own_output')]
assert not torch.cuda.is_initialized()
print(json.dumps({'state':'PASS_ACTUAL_HELPER_EDGE_FIXTURES','cases':results,'scope':'CPU Tensor transport only; no GPU numeric/kernel validation'},indent=2))
