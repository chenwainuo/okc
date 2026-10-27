"""CPU-only collection ingestion. Profiler duration is never gain evidence."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D=Path(__file__).resolve().parent;CAND=D.parent;ROOT=CAND.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=a.collection/'logs/scheduler/rq-pooler32-exactso-counters-01/run'
def read(name):
 p=P/name;return json.loads(p.read_text())if p.exists()else None
inv=json.loads((a.collection/'inventory.json').read_text());session=read('session.json');capture=read('result.json');analysis=read('analysis.json')
pins=inv.get('dispatch',{}).get('source_pins',{})
for f,h in pins.items():assert sha(ROOT/f)==h,('actualdispatch source drift',f)
numeric_valid=bool(capture and capture.get('state')=='VALID_EXACT_CURRENT_POOLER32_RQ_PROFILE_OUTPUTS')
counter_complete=bool(session and session.get('state')=='COMPLETE_EXACT_CURRENT_COUNTER_DIAGNOSTIC_ROOT_EXTERNAL_CLEANUP_REQUIRED'and analysis)
metrics=None
if counter_complete:
 s=importlib.util.spec_from_file_location('frozen_profile_recipe',CAND/'profile.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 parsed=m.parse((P/'raw.csv').read_text(),capture);assert parsed['metrics']==analysis['metrics']and parsed['replay_passes']==analysis['replay_passes']
 assert analysis['capture_SHA256']==sha(P/'result.json')and session['capture_process_group_drained']and session['capture_returncode']==0
 metrics=analysis['metrics']
if numeric_valid:
 assert capture['build']['SO_sha256']=='ab6c3ad71701098713f7c3435439f905c4d482ae66a995948bfcda254494a32f'and capture['build']['native_arm']==0 and capture['JIT_calls']==0
 assert capture['own_terminal']['seed']==capture['fixture']['X_seed']==91100
 assert capture['own_terminal']['before_any_later_replay_or_mutation']and capture['own_terminal']['oracle']['current_X']=='EXACT'
 assert len(capture['validation'])==4 and len(capture['mutations'])==5 and len(capture['raw'])==6
 assert all(x['oracle']['current_X']=='EXACT'and x['oracle']['RQ_counters_reset0']for x in capture['validation']+capture['mutations'])
 assert capture['post_scope_cleanup']['allocated']==0 and capture['post_scope_cleanup']['blas_after'][2]==0 and not capture['graph_cleanup']
cleanup=inv.get('cleanup_observation',{})
clean=cleanup.get('terminal_exit')==0 and not cleanup.get('cleanup_failed')and cleanup.get('gpu_memory_mib')==0 and not cleanup.get('compute_pids')and not cleanup.get('container_exists')
state='VALID_COUNTER_DIAGNOSTIC'if numeric_valid and counter_complete and clean else'INCOMPLETE_DIAGNOSTIC_NO_COUNTER_OR_GAIN_ADMISSION'
o={'state':state,'collection':str(a.collection),'capture_numeric_valid':numeric_valid,'counter_complete':counter_complete,'external_clean':clean,'source_dispatch_pins_verified':len(pins),'metrics':metrics,'replay_passes':analysis.get('replay_passes')if analysis else None,'capture_state':capture.get('state')if capture else None,'session_state':session.get('state')if session else None,'capture_cost':capture.get('cost')if capture else None,'diagnostic_wall_seconds':session.get('elapsed_seconds')if session else None,'actual_graph_nodes':capture.get('actual_graph_nodes')if capture else None,'fixture':capture.get('fixture')if capture else None,'post_scope_cleanup':capture.get('post_scope_cleanup')if capture else None,'external_cleanup':cleanup,'performance_gain_us':None,'cause_limits':['Instrumented duration is a diagnostic cost/counter, not clean kernel benchmark or speed improvement.','Aggregate metrics cannot assign producer versus consumer roles or removable time.','Numerical validity is independent of profiler capability/export success.','Current original SO and source reuse do not eliminate profiler/cache/replay context differences.'],'files':{str(p):sha(p)for p in [a.collection/'inventory.json',*[p for p in P.iterdir()if p.is_file()]]},'contract':'proj_rsi/ITERATION.md'}
s=json.dumps(o,indent=2)+'\n';assert not a.output.exists()or a.output.read_text()==s;a.output.write_text(s);print(s)
