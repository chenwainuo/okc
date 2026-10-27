"""Independent CPU extraction of QKV actual terminal/workspace gates; no GPU lab run."""
import ast,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[3]
LAB=ROOT/'proj_rsi/tools/group8_qkv_runtime_lab.py'

def check():
 text=LAB.read_text();tree=ast.parse(text)
 funcs={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
 nodes=[funcs[name] for name in ('terminal_record_contract','oracle','workspace_oracle','terminal_gate')]
 ns={'torch':torch,'LABELS':('pw36','rs32','new'),'CONFIG':{'layer':'qkv_gate'},
     'reference':SimpleNamespace(check_implementation=lambda data,out,weights:(torch.equal(out,data['x'].float()),'CPU_SYNTHETIC_NOT_HARDWARE'))}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LAB),'exec'),ns)
 labels=ns['LABELS']
 def case(corrupt=None):
  seeds=[720104,720083,720062];fixtures={seed:torch.arange(8,dtype=torch.int32).reshape(2,4)+20*i for i,seed in enumerate(seeds)}
  last=dict(zip(labels,seeds));x=fixtures[seeds[-1]].clone();outputs={l:fixtures[last[l]].float().clone() for l in labels}
  raw=[dict(label=l,input_seed=last[l],mode='F1') for l in labels]
  pipe=SimpleNamespace(xp=fixtures[last['new']].reshape(-1).clone(),partial=torch.stack([fixtures[last['new']].float(),*[torch.zeros((2,4)) for _ in range(3)]]))
  w=torch.tensor([1.]);packed=torch.tensor([2]);escape=torch.tensor([3]);immutable=[(a,a.clone()) for a in (w,packed,escape)];counter=torch.tensor([0])
  def invariants(context):
   assert all(torch.equal(a,b) for a,b in immutable),'CPU immutable mismatch'
   assert not int(counter.abs().sum()),'CPU counter mismatch'
  ns.update(fixtures=fixtures,last=last,x=x,outputs=outputs,pipe=pipe,w=w,pack_index=torch.arange(8),invariants=invariants,persist=lambda:None,result={'raw_calls':raw})
  if corrupt=='output':outputs['pw36'][0,0]+=1
  if corrupt=='NaN_output':outputs['rs32'][0,0]=float('nan')
  if corrupt=='XP':pipe.xp[0]+=1
  if corrupt=='partial_NaN':pipe.partial[1,0,0]=float('nan')
  if corrupt=='partial_ordered_reduction':pipe.partial[2,0,0]+=1
  if corrupt=='current_X':x[0,0]+=1
  if corrupt=='W':w[0]+=1
  if corrupt=='packed':packed[0]+=1
  if corrupt=='escape':escape[0]+=1
  if corrupt=='counter':counter[0]=1
  if corrupt=='wrong_last_fixture':last['pw36']=last['new']
  if corrupt=='missing_policy':last.pop('rs32')
  ns['terminal_gate']('CPU_SYNTHETIC')
 case();rejected={}
 for corrupt in ('output','NaN_output','XP','partial_NaN','partial_ordered_reduction','current_X','W','packed','escape','counter','wrong_last_fixture','missing_policy'):
  try:case(corrupt)
  except AssertionError as e:rejected[corrupt]=str(e)
  else:raise AssertionError('actual terminal gate admitted '+corrupt)
 terminal=funcs['terminal_gate']
 calls={n.func.attr for n in ast.walk(terminal) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
 assert not calls&{'replay','copy_','fill_','zero_','reset'}
 assert text.index("if index==120:terminal_gate('F1')")<text.index("x.copy_(fixtures[record['input_seed']])")
 assert text.index("terminal_gate('paired')")<text.index("phase('profiles_only_after_both_strict_terminal_gates')")
 return {'status':'CPU_INDEPENDENT_TERMINAL_NEGATIVES_PASS','GPU_calls':0,'fixture':'CPU_SYNTHETIC_NOT_HARDWARE','driver_sha256':hashlib.sha256(text.encode()).hexdigest(),'negative_controls':rejected,'strict_boundary_order':'PASS_F1_BEFORE_NEXT_X_COPY_PAIRED_BEFORE_PROFILE'}
if __name__=='__main__':
 result=check();Path(__file__).with_name('negative-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
