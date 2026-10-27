"""CPU-only adversarial terminal/schedule/source admission; never imports a candidate."""
import ast
import collections
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def proof():
 S=load(HERE/'support.py','matched_support_cpu')
 if (HERE/'manifest.json').is_file():S.verify()
 M=S.driver();original=M.cpu_check()
 import torch
 torch.set_num_threads(2)
 def forbidden(*args,**kwargs):raise AssertionError('CPU proof attempted CUDA')
 for name in ('_lazy_init','init','is_available','device_count','synchronize','current_device'):
  setattr(torch.cuda,name,forbidden)
 tree=ast.parse((ROOT/'proj_rsi/tools/smallm_matched_geometry_lab.py').read_text())
 terminal=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='terminal_gate')
 bad=[node.attr for node in ast.walk(terminal) if isinstance(node,ast.Attribute) and node.attr in ('replay','copy_','fill_','zero_')]
 assert not bad,('terminal gate mutates/replays',bad)
 class Oracle:
  @staticmethod
  def check_implementation(data,out,weights):
   want=data['x']@weights[data['config']['layer']].t()
   return (bool(torch.equal(out,want)),'CPU exact oracle')
 ns={'torch':torch,'reference':Oracle,'LABELS':S.LABELS}
 exec(compile(ast.Module(body=[terminal],type_ignores=[]),'<actual terminal_gate>','exec'),ns)
 gate=ns['terminal_gate'];schedule=S.schedule();calls=[row for row in schedule if row[0]!='terminal']
 assert len(calls)==462 and len(schedule)==464
 assert collections.Counter(row[0] for row in calls)=={'warmup':35,'F1':231,'paired':196}
 markers=[(i,row[1]) for i,row in enumerate(schedule) if row[0]=='terminal']
 assert markers==[(266,'F1'),(463,'paired')],markers
 f1=[row for row in calls if row[0]=='F1'];last={}
 for mode,label,seed,key in f1:last[label]=seed
 assert set(last)==set(S.LABELS) and len(set(last.values()))==7
 pairs=[row for row in calls if row[0]=='paired']
 orders=collections.Counter(tuple(row[1] for row in pairs[i:i+7]) for i in range(0,196,7))
 assert len(orders)==14 and set(orders.values())=={2}
 for label in S.LABELS:
  positions=collections.Counter(i%7 for i,row in enumerate(pairs) if row[1]==label)
  assert set(positions.values())=={4}
 assert len({seed for _,_,seed,_ in pairs})==28
 def fixture(seed):return (torch.arange(4,dtype=torch.float32).reshape(2,2)+(seed%101))/32
 def build():
  fixtures={seed:fixture(seed) for _,_,seed,_ in calls}
  fixtures[max(last.values())+1]=fixture(max(last.values())+1)
  w=torch.tensor([[1.,2.],[3.,5.]])
  final={'label':f1[-1][1],'input_seed':f1[-1][2]}
  x=fixtures[final['input_seed']].clone()
  outputs={label:fixtures[seed]@w.t() for label,seed in last.items()}
  pack=torch.tensor([2,3],dtype=torch.int32);esc=torch.tensor([7,9],dtype=torch.int32);counter=torch.zeros(4,dtype=torch.int32)
  immutable=[('W',w,w.clone()),('packed',pack,pack.clone()),('escape',esc,esc.clone())]
  def invariants():assert not int(torch.count_nonzero(counter)),'selected counter failed reset'
  return dict(outputs=outputs,x=x,config={'layer':'q'},w=w,fixtures=fixtures,last=dict(last),final=final,immutable=immutable,invariants=invariants),counter
 positive,counter=build();assert len(gate(**positive))==7
 rejected=[]
 for name in ('nonfinite_output','corrupt_output','changed_X','changed_W','changed_packed','changed_escape','counter_nonzero','wrong_recorded_seed','output_alias'):
  case,counter=build()
  if name=='nonfinite_output':case['outputs']['cursor5'].fill_(float('nan'))
  elif name=='corrupt_output':case['outputs']['cursor6'][0,0]+=1
  elif name=='changed_X':case['x'][0,0]+=1
  elif name=='changed_W':case['w'][0,0]+=1
  elif name=='changed_packed':case['immutable'][1][1][0]+=1
  elif name=='changed_escape':case['immutable'][2][1][0]+=1
  elif name=='counter_nonzero':counter[0]=1
  elif name=='wrong_recorded_seed':case['last']['actual_pw36']+=1
  elif name=='output_alias':case['outputs']['cursor5']=case['outputs']['parent5']
  try:gate(**case)
  except AssertionError:rejected.append(name)
  else:raise AssertionError(('negative accepted',name))
 # Actual captured-policy arms must not add the basis-only copy_ wrapper.
 source=(ROOT/'proj_rsi/tools/smallm_matched_geometry_lab.py').read_text()
 assert "with torch.cuda.graph(g):out=modules[role].kernel_fn" in source
 assert source.index("entry['timing']=measure")<source.index("entry['mutations']=[]")<source.index('from torch.profiler import')
 assert "M.jit_gate(mod,role" in source and "shutil.copyfile(binary,target)" in source
 assert "M.select_counter_contract" in source and "M.counter_storage_contract" in source
 return {'status':'CPU_PASS_GPU_UNRUN_INDEPENDENT_REVIEW_REQUIRED','CUDA_calls':0,'candidate_imports':0,'raw_calls_per_shape':462,'raw_calls_total':1386,
         'shape_count':3,'labels':list(S.LABELS),'schedule_counts':dict(collections.Counter(row[0] for row in calls)),
         'balanced_orders':14,'repeats_per_order':2,'position_count_per_label':4,'paired_rounds':28,
         'strict_terminal_oracles_per_shape':14,'strict_terminal_oracles_total':42,'distinct_F1_last_fixtures':7,
         'read_only_actual_terminal_gate_positive':'PASS','terminal_negative_rejections':rejected,
         'actual_graph_direct_kernel_fn_no_basis_wrapper_copy':'PASS','mutation_and_profiling_after_timed_oracles':'PASS',
         'source_binding_alias_counter_positive_negative_proof':original['counter_contract_proof'],
         'cursor_events':original['cursor_events'],'selected_body_math_barrier_LDS_counts_unchanged':original['all_selected_body_barrier_math_LDS_counts_unchanged'],
         'physical_register_budget':original['physical_register_budget'],'rqrun_query_routing':original['rqrun_argument_routing'],
         'runtime_unvalidated':'No CUDA context, extension load, prepare, math, replay or timing performed. Actual JIT/source/SO/resources and full correctness remain mandatory.'}

if __name__=='__main__':print(json.dumps(proof(),indent=2))
