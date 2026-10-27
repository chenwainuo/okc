"""CPU-only trace/replay/terminal/counter proof; no production or CUDA import."""
import ast,copy,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def proof():
 S=load(HERE/'support.py','hotpath_support_CPU');stack=S.stack()
 source=(ROOT/'proj_rsi/tools/pw38_hotpath_trace.py').read_text();tree=ast.parse(source)
 assert len(S.case_scope('all30'))==30 and len(S.case_scope('top4'))==4
 assert (HERE/'graph_metadata.py').read_bytes()==(ROOT/'proj_rsi/candidates/pw38_selected_profile_trial/graph_metadata.py').read_bytes()
 # Duplicate-symbol multinode metadata is legal and must produce two calls per replay.
 nodes=[{'node_type':0,'function_mangled':'_Z1fv','demangled':'f()'},{'node_type':0,'function_mangled':'_Z1fv','demangled':'f()'},{'node_type':1}]
 events=[{'ph':'X','cat':'user_annotation','name':'PW38_TRACE_SETUP','ts':0,'dur':80},
         {'ph':'X','cat':'kernel','name':'setup_fill','ts':10,'dur':5}]
 for i in range(3):
  t=100+i*100;events+=[{'ph':'X','cat':'user_annotation','name':'PW38_TRACE_REPLAY/'+str(i),'ts':t,'dur':80},
   {'ph':'X','cat':'kernel','name':'f()','ts':t+10,'dur':20},
   {'ph':'X','cat':'kernel','name':'f()','ts':t+20,'dur':20},
   {'ph':'X','cat':'gpu_memcpy','name':'Memcpy DtoD','ts':t+45,'dur':2},
   {'ph':'X','cat':'gpu_memset','name':'Memset','ts':t+50,'dur':2},
   {'ph':'X','cat':'cuda_runtime','name':'cudaDeviceSynchronize','ts':t+5,'dur':60}]
 trace={'traceEvents':events};good=S.parse_trace(trace,nodes,3)
 assert good['kernels'][0]['calls']==6 and good['kernels'][0]['calls_per_replay']==2
 assert all(r['kernel_duration_sum_us']==40 and r['kernel_first_to_last_span_us']==30 and r['host_replay_plus_sync_inclusive_us']==80 for r in good['per_replay'])
 assert len(good['setup_GPU_events'])==1 and len(good['host_runtime_events'])==3
 negatives=[]
 for kind in ('missingKernel','extraKernel','duplicateReplay','zeroDuration','NaNDuration','infiniteStart','wrongSymbol','wrongMetadata','outsideScope','missingReplay','badSetup','reorderedScopes'):
  bad=copy.deepcopy(trace);ns=copy.deepcopy(nodes);ev=bad['traceEvents']
  first=next(e for e in ev if e.get('cat')=='kernel' and e.get('name')=='f()')
  if kind=='missingKernel':ev.remove(first)
  elif kind=='extraKernel':ev.append(copy.deepcopy(first))
  elif kind=='duplicateReplay':ev.append(copy.deepcopy(next(e for e in ev if e['name']=='PW38_TRACE_REPLAY/0')))
  elif kind=='zeroDuration':first['dur']=0
  elif kind=='NaNDuration':first['dur']=float('nan')
  elif kind=='infiniteStart':first['ts']=float('inf')
  elif kind=='wrongSymbol':first['name']='other()'
  elif kind=='wrongMetadata':ns[0]['demangled']='other()'
  elif kind=='outsideScope':first['ts']=9999
  elif kind=='missingReplay':ev.remove(next(e for e in ev if e['name']=='PW38_TRACE_REPLAY/2'))
  elif kind=='badSetup':ev[0]['dur']=0
  elif kind=='reorderedScopes':next(e for e in ev if e['name']=='PW38_TRACE_REPLAY/1')['ts']=50
  try:S.parse_trace(bad,ns,3)
  except (AssertionError,KeyError):negatives.append(kind)
  else:raise AssertionError('trace negative admitted '+kind)
 # Source-bound reset prefix and dormant baseline, zero-call plans stay inactive.
 plans={'a':{'arena':{'mode':'reset0','offset':1,'reset_indices':[0],'changes':{}}},'b':{'arena':{'mode':'reset0','offset':0,'reset_indices':[2],'changes':{}}}}
 plan=S.weighted_plan(plans,{'a':2,'b':0});assert plan['arena']['reset_indices']==[1]
 before={'arena':{'name':'arena','values':[19,7,23]}}
 stack.counter_check(before,{'arena':{'name':'arena','values':[19,0,23]}},plan,1)
 assert S.weighted_plan(plans,{'a':0,'b':0})=={}
 stack.counter_check(before,before,{},1)
 for bad in ([19,7,23],[0,0,23],[19,0,0]):
  try:stack.counter_check(before,{'arena':{'name':'arena','values':bad}},plan,1)
  except AssertionError:pass
  else:raise AssertionError('reset/dormant corruption admitted')
 merged=S.weighted_plan(plans,{'a':1,'b':1});assert merged['arena']['reset_indices']==[1,2]
 progressive={'a':{'arena':{'mode':'progress','offset':1,'changes':{0:3,1:1}}},'b':{'arena':{'mode':'progress','offset':0,'changes':{1:5}}}}
 assert S.weighted_plan(progressive,{'a':2,'b':3})['arena']['changes']=={1:21,2:2}
 # Actual current-case and final-scope oracles, CPU tensors only.
 import torch
 torch.set_num_threads(2)
 def forbidden(*a,**kw):raise AssertionError('CPU proof attempted CUDA')
 for key in ('init','_lazy_init','is_available','device_count','current_device','synchronize'):setattr(torch.cuda,key,forbidden)
 class Oracle:
  @staticmethod
  def check_implementation(data,out,weights):return bool(torch.equal(out,data['x']@weights[data['config']['layer']].T)),'CPU exact oracle'
 names=('same_bytes','final_oracle');env={'torch':torch,'reference':Oracle,'S':S}
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),'<actual terminal>','exec'),env)
 run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');terminal=next(n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)and n.name=='terminal')
 assert not any(isinstance(n,ast.Attribute)and n.attr in('replay','copy_','fill_','zero_') for n in ast.walk(terminal))
 exec(compile(ast.Module(body=[terminal],type_ignores=[]),'<actual own terminal>','exec'),env)
 def fixtures():
  weights={l:torch.tensor([[1.,2.],[3.,5.]])for _,l,_ in S.CASES}
  inputs={name:torch.arange(m*2,dtype=torch.float32).reshape(m,2)/64 for name,_,m in S.CASES}
  cfg={name:{'layer':l}for name,l,m in S.CASES};fresh={n:x.clone()for n,x in inputs.items()};outputs={n:x@weights[cfg[n]['layer']].T for n,x in inputs.items()}
  immutable=[('W/'+l,t,t.clone())for l,t in weights.items()]+[('pack',torch.tensor([1,2]),torch.tensor([1,2]))]
  counter=[0]
  def check():assert counter[0]==0,'counter failure'
  env.update(inputs=inputs,outputs=outputs,fixtures=fresh,configs=cfg,weights=weights,immutable=immutable,check_counters=check)
  return counter
 terminal_negatives=[]
 for scope in ('top4','all30'):
  S.CASES=S.case_scope(scope);fixtures();name=S.CASES[0][0]
  assert env['terminal'](name)['actual_retained_output_before_any_replay']=='PASS'
  assert len(env['final_oracle'](env['outputs'],env['inputs'],env['fixtures'],env['configs'],env['weights'],env['immutable'],env['check_counters']))==len(S.CASES)
 for kind in ('corruptOutput','NaNOutput','staleX','wrongFixture','WChanged','packChanged','counterChanged','missingOutput'):
  counter=fixtures();name=S.CASES[0][0]
  if kind=='corruptOutput':env['outputs'][name][0,0]+=1
  elif kind=='NaNOutput':env['outputs'][name].fill_(float('nan'))
  elif kind=='staleX':env['inputs'][name][0,0]+=1
  elif kind=='wrongFixture':env['fixtures'][name][0,0]+=1
  elif kind=='WChanged':env['weights'][env['configs'][name]['layer']][0,0]+=1
  elif kind=='packChanged':env['immutable'][-1][1][0]+=1
  elif kind=='counterChanged':counter[0]=1
  elif kind=='missingOutput':env['outputs'].pop(name)
  try:env['terminal'](name)
  except (AssertionError,KeyError):terminal_negatives.append(kind)
  else:raise AssertionError('own output corruption admitted '+kind)
 # Ordering and source freeze guards, no hidden probe replay in terminal.
 assert source.index("entry['profile_terminal']=terminal(name)")<source.index('prof.export_chrome_trace')
 assert source.index("result['final_all_retained_output_oracles']=final_oracle")<source.index('    except BaseException:',source.index("result['final_all_retained_output_oracles']=final_oracle"))
 assert 'len(actual_args)==19' in source and "stack.actual_rq_gate" in source
 assert 'from torch.profiler import' in source and not hasattr(S,'NCU')
 assert subprocess.check_output(['c++filt','_Z1fv'],text=True).strip()=='f()'
 return {'status':'CPU_PASS_TRACE_GPU_UNRUN_REVIEW_REQUIRED','cases':{'top4':S.case_scope('top4'),'all30':S.case_scope('all30')},'source_PW38':S.PW38_SHA,
  'CUDA_contexts_or_API_calls':0,'production_imports':0,'NCU_calls':0,'multinode_duplicate_symbol_positive':True,'trace_negative_rejections':negatives,'own_terminal_negative_rejections':terminal_negatives,
  'active_reset_prefix_dormant_preservation_and_zero_calls':'PASS','weighted_progress_alias_offsets':'PASS','unprofiled_benchmark_replays_percase':11,'profile_replays_percase':3,'mutations_percase':4,'own_current_case_terminals_before_export':'PASS','one_final_all_retained_scope_terminal':'PASS','strict_units_overlap_and_host_sync_boundaries':'PASS',
  'validity_reuse':'Unchanged PW38 source prior complete fullpolicy validation reused; newnormal/outlier owncase oracles rerun, no redundant allK campaign.'}
if __name__=='__main__':print(json.dumps(proof(),indent=2))
