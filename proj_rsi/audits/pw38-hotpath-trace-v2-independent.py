"""CPU-only actual-Chrome regression and immutable hotpath02 review."""
import ast, copy, hashlib, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TRIAL=ROOT/'proj_rsi/candidates/pw38_hotpath_trace_v2_trial'
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.01/collection-1793114585674218450'
RUN=COLLECTION/'logs/scheduler/pw38-hotpath-trace-01/run'
PINS={'driver':'e71a6c1350432eb122acacffeccd2f6a6b49d0d006e4cbf8e17609c4037d57ef',
      'manifest':'dd02e9e58bb59e9b37d0241d11864c99b714ae94c7e32359baf5070a6027d20a',
      'support':'193666639b9fa79f81cfdb50a3758bd6be32e0940dd49fcb2b66f84c91b1e986',
      'actual_capture':'1964b8894e2a88e73c05cfa12cbae7e87ed210cca0f328a2abc33a15e3844791',
      'actual_chrome':'628748fdaa16a3008a6916c868a465f30b166fdedd1f122c70f41a1104019652'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def review():
 assert sha(ROOT/'proj_rsi/tools/pw38_hotpath_trace_v2.py')==PINS['driver']
 assert sha(TRIAL/'manifest.json')==PINS['manifest'] and sha(TRIAL/'support.py')==PINS['support']
 manifest=json.loads((TRIAL/'manifest.json').read_text())
 for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
 inventory=json.loads((COLLECTION/'inventory.json').read_text())
 for rel,h in inventory['files'].items():assert sha(COLLECTION/rel)==h,rel
 assert sha(RUN/'capture.json')==PINS['actual_capture'] and sha(RUN/'decode1-qkv_gate.chrome.json')==PINS['actual_chrome']
 old=ROOT/'proj_rsi/tools/pw38_hotpath_trace.py';new=ROOT/'proj_rsi/tools/pw38_hotpath_trace_v2.py'
 assert new.read_text().replace('pw38_hotpath_trace_v2_trial','pw38_hotpath_trace_trial')==old.read_text()
 capture=json.loads((RUN/'capture.json').read_text());trace=json.loads((RUN/'decode1-qkv_gate.chrome.json').read_text())
 nodes=capture['cases']['decode1-qkv_gate']['actual_graph_nodes']
 S=load(TRIAL/'support.py','independent_hotpath_v2_support')
 oldsource=(ROOT/'proj_rsi/candidates/pw38_hotpath_trace_trial/support.py').read_text()
 fn=next(n for n in ast.parse(oldsource).body if isinstance(n,ast.FunctionDef)and n.name=='parse_trace')
 env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen01-parser>', 'exec'),env)
 try:env['parse_trace'](trace,nodes,3)
 except AssertionError as e:oldfailure=str(e)
 else:raise AssertionError('old actual-format defect not reproduced')
 parsed=S.parse_trace(trace,nodes,3)
 assert [r['kernel_calls']for r in parsed['per_replay']]==[1,1,1]
 assert [r['kernel_duration_sum_us']for r in parsed['per_replay']]==[13.024,12.0,11.776]
 assert len(parsed['gpu_correlated_annotations'])==4
 negatives=[]
 for kind in ('duplicate_CPU_replay','GPU_only_replay','GPU_only_setup','extra_actual_kernel','missing_actual_kernel','wrong_actual_symbol'):
  bad=copy.deepcopy(trace);events=bad['traceEvents']
  cpu=next(e for e in events if e.get('name')=='PW38_TRACE_REPLAY/0'and e.get('cat')=='user_annotation')
  kernel=next(e for e in events if e.get('cat')=='kernel'and e.get('name')==nodes[0]['demangled'])
  if kind=='duplicate_CPU_replay':events.append(copy.deepcopy(cpu))
  elif kind=='GPU_only_replay':events.remove(cpu)
  elif kind=='GPU_only_setup':events.remove(next(e for e in events if e.get('name')=='PW38_TRACE_SETUP'and e.get('cat')=='user_annotation'))
  elif kind=='extra_actual_kernel':events.append(copy.deepcopy(kernel))
  elif kind=='missing_actual_kernel':events.remove(kernel)
  elif kind=='wrong_actual_symbol':kernel['name']='unrelated()'
  try:S.parse_trace(bad,nodes,3)
  except AssertionError:negatives.append(kind)
  else:raise AssertionError(kind)
 proof=load(TRIAL/'cpu_proof.py','independent_hotpath_v2_proof').proof()
 assert len(proof['trace_negative_rejections'])==14 and len(proof['own_terminal_negative_rejections'])==8
 assert capture['status']=='INVALID_SELECTED_PROFILE_NO_COUNTER_ADMISSION'
 assert capture['graph_cleanup']==['DRAINED_RESET']*30 and inventory['terminal_exit']==1
 cleanup=inventory['cleanup_observation'];assert not cleanup['cleanup_failed']and cleanup['gpu_memory_mib']==0 and not cleanup['container_exists']and not cleanup['compute_pids']and not cleanup['guard_pids']
 return {'decision':'GO_BOUNDED_ATTEMPT02_GPU_UNRUN','pins':PINS,'manifest_files_verified':len(manifest['files']),
         'failed01_artifacts_verified':len(inventory['files']),'old_parser_actual_failure':oldfailure,
         'failed01_state':capture['status'],'failed01_cleanup':'30_DRAINED_RESET_AND_PHYSICAL_ZERO',
         'actual_trace_reparse':parsed,'actual_trace_negative_rejections':negatives,'author_cpu_proof':proof,
         'normal_inner_seconds':600,'normal_outer_seconds':720,'GPU_NCU_API_queue_calls_by_reviewer':0}
if __name__=='__main__':print(json.dumps(review(),indent=2))
