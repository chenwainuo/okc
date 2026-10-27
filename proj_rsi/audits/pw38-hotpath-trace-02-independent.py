"""Read-only CPU validation of collected PW38 trace02. No CUDA loading."""
import collections, hashlib, importlib.util, json, math, re, statistics, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COL=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.02/collection-1793115060902266190'
RUN=COL/'logs/scheduler/pw38-hotpath-trace-02/run'
TRIAL=ROOT/'proj_rsi/candidates/pw38_hotpath_trace_v2_trial'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def tables(s):
 out={}
 for block in s.split('Function : ')[1:]:
  symbol=block.splitlines()[0].strip()
  out[symbol]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',block)]
 return out
def analyze():
 inv=json.loads((COL/'inventory.json').read_text());r=json.loads((RUN/'capture.json').read_text())
 for p,h in inv['files'].items():assert sha(COL/p)==h,p
 for p,h in inv['dispatch']['source_pins'].items():assert sha(ROOT/p)==h,p
 S=load(TRIAL/'support.py','independent_trace02_support');S.verify();B=S.stack()
 assert inv['terminal_exit']==0 and not inv['missing'] and not r['jit_errors']
 assert r['source_sha256']==S.PW38_SHA and r['scope']=='all30'
 assert r['status']=='VALID_PW38_HOTPATH_TRACE_ROOT_REVIEW_REQUIRED' and r['phase']=='ALL_ACTUAL_PROFILE_TERMINALS_PASSED'
 assert r['review_sha256']==sha(ROOT/'proj_rsi/audits/pw38-hotpath-trace-v2-independent.md')
 names={n for n,l,m in S.case_scope('all30')}
 assert set(r['cases'])==set(r['trace_cases'])==names and len(names)==30
 assert r['profile_order']==[n for n,l,m in S.case_scope('all30')]
 assert r['hardware']['compute_capability']==[9,0] and r['hardware']['SM_count']>=128
 assert r['prepare']['clock_reset']is False and r['prepare']['original_import_clock_elapsed']<300
 assert r['graph_cleanup']==['DRAINED_RESET']*30
 c=inv['cleanup_observation'];assert not c['cleanup_failed']and not c['container_exists']and not c['compute_pids']and not c['guard_pids']and c['gpu_memory_mib']==0
 # Redump actual archived SOs with CPU cuobjdump, then rerun source-bound gates.
 allargs=set();jit_receipts=[]
 for j in r['JIT']:
  prefix=j['prefix'];cu=(RUN/(prefix+'.cu')).read_text();sass=(RUN/(prefix+'.sass')).read_text();res=(RUN/(prefix+'.resources')).read_text();so=RUN/j['archived_SO']
  assert sha(so)==j['SO_sha256'] and sha(RUN/(prefix+'.cu'))==j['CUDA_source_sha256']
  assert sha(RUN/(prefix+'.sass'))==j['SASS_sha256']and sha(RUN/(prefix+'.resources'))==j['resource_sha256']
  assert subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(so)],text=True,timeout=60)==sass
  assert subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(so)],text=True,timeout=60)==res
  assert j['before_return_to_prepare_math']is True
  if 'struct RqP'in cu:
   gate=S.P45.canonical_gate(cu,sass,res);allargs.update(map(tuple,gate['args']))
   eh=B.actual_rq_gate(sass,res,bool(B.rq_targets(sass)))
   assert eh==j['RQ_gate']['EH']
  jit_receipts.append({'prefix':prefix,'SO_SHA256':sha(so),'actual_SO_redump':'EXACT_SASS_AND_RESOURCES'})
 assert len(allargs)==r['actual_RQ_functions']==19
 # Verify exact full physical counter arithmetic, including dormant nonzero lanes.
 def arenas(rows):
  result={}
  for x in rows:
   assert hashlib.sha256(json.dumps(x['values'],separators=(',',':')).encode()).hexdigest()==x['digest']
   result[tuple(x['storage'])]={'name':x['name'],'values':x['values']}
  return result
 before=arenas(r['pre_trace_counters']);after=arenas(r['post_trace_counters'])
 plans={n:{tuple(v['storage']):{**{k:w for k,w in v.items()if k!='storage'},'changes':{int(k):w for k,w in v.get('changes',{}).items()}}for v in e['counter_plan']}for n,e in r['cases'].items()}
 counts={n:r['trace_cases'][n]['counter_calls_since_snapshot']for n in names};assert set(counts.values())=={18}
 B.counter_check(before,after,S.weighted_plan(plans,counts),1)
 def terminal(t,n):
  assert t['case']==n and t['current_X']=='BYTE_IMMUTABLE'and t['actual_retained_output_before_any_replay']=='PASS'and t['oracle']
 rankings=[];kernel_events=0;node_counts=collections.Counter();raw=mut=term=0
 for n,l,m in S.case_scope('all30'):
  setup=r['cases'][n];entry=r['trace_cases'][n]
  assert entry['pick']==setup['actual_selected_policy']and setup['layer']==l and setup['m']==m
  assert len(setup['numerics']['task'])==2 and {v['dist']for v in setup['numerics']['task']}=={'normal','outliers'}
  assert setup['numerics']['source_bound_counter_contract']=='PASS'
  benchmarks=entry['benchmarks'];assert len(benchmarks)==11
  for i,v in enumerate(benchmarks):assert v['replay']==i and math.isfinite(v['cold_L2_event_us'])and v['cold_L2_event_us']>0
  assert [v['kind']for v in entry['mutations']]==['zero','outliers','all_probes','fresh_normal']
  terminal(entry['benchmark_terminal'],n);terminal(entry['profile_terminal'],n)
  for v in entry['mutations']:terminal(v['terminal'],n)
  raw+=11;mut+=4;term+=6
  tracepath=RUN/entry['trace_path'];assert sha(tracepath)==entry['trace_sha256']
  parsed=S.parse_trace(json.loads(tracepath.read_text()),setup['actual_graph_nodes'],3)
  assert parsed==entry['trace']and entry['profile_replays']==3
  nodes=[v for v in setup['actual_graph_nodes']if v['node_type']==0];node_counts[len(nodes)]+=1
  for node in nodes:
   assert subprocess.check_output(['c++filt',node['function_mangled']],text=True).strip()==node['demangled']
   assert node['attributes']['local_bytes']==0 and node['theoretical_CTA_occupancy_per_SM']>0
   for binding in node['source_binding']['custom_JIT_records']:
    j=next(j for j in r['JIT']if j['prefix']==binding['prefix'])
    assert node['function_mangled']in j['functions']and binding['SO_sha256']==j['SO_sha256']
  kernel_events+=sum(v['kernel_calls']for v in parsed['per_replay'])
  rankings.append({'case':n,'layer':l,'m':m,'pick':entry['pick'],'kernels':[v['symbol']for v in parsed['kernels']],
    'profile_CUDA_mean_kernel_sum_us':statistics.mean(v['kernel_duration_sum_us']for v in parsed['per_replay']),
    'profile_CUDA_raw_kernel_sums_us':[v['kernel_duration_sum_us']for v in parsed['per_replay']],
    'profile_host_replay_plus_sync_raw_us':[v['host_replay_plus_sync_inclusive_us']for v in parsed['per_replay']],
    'cold_event_raw_us':[v['cold_L2_event_us']for v in benchmarks],
    'cold_event_median_us':statistics.median(v['cold_L2_event_us']for v in benchmarks)})
 assert (raw,mut,term,kernel_events)==(330,120,180,90)
 assert len(r['final_all_retained_output_oracles'])==30
 for t in r['final_all_retained_output_oracles']:terminal(t,t['case'])
 assert {t['case']for t in r['final_all_retained_output_oracles']}==names
 # Four high-score-sensitivity GV symbols also exist with exact PC streams in PW45.
 pw45=ROOT/'results/proj-rsi/scheduler/artifacts/research.pw45.gpu.01/collection-1793111000996134576/logs/scheduler/pw45-stack-01/run/pw45.gv20_7bb50cec.sass'
 actualgv=next(RUN.glob('JIT*.gv20_*.sass'));left=tables(actualgv.read_text());right=tables(pw45.read_text())
 same=[]
 for n in ['decode1-pooler','verify1-pooler','decode16-pooler','decode16-unpooler']:
  symbol=r['cases'][n]['actual_graph_nodes'][0]['function_mangled'];assert left[symbol]==right[symbol];same.append({'case':n,'symbol':symbol,'PC_predicate_instructions':'EXACT_PW45'})
 rankings.sort(key=lambda x:-x['profile_CUDA_mean_kernel_sum_us'])
 return {'decision':'VALID_DIAGNOSTIC_TRACE_NO_GAIN_CLAIM','artifact_files_verified':len(inv['files']),
 'dispatch_source_pins_verified':len(inv['dispatch']['source_pins']),'capture_SHA256':sha(RUN/'capture.json'),
 'collection':str(COL.relative_to(ROOT)),'prepare':r['prepare'],'hardware':r['hardware'],
 'counts':{'cases':30,'reparsed_Chrome_files':30,'profile_graph_replays':90,'actual_kernel_events':90,'cold_event_samples':raw,'mutations':mut,'own_current_case_terminals':term,'final_retained_outputs':30,'physical_counter_arenas':len(before)},
 'kernel_nodes_per_graph':dict(node_counts),'JIT':jit_receipts,'PW45_four_hot_GV_exact_PC_identity':same,
 'ranking_by_instrumented_CUDA_mean':rankings,'cleanup':'ALL_30_DRAINED_RESET_PHYSICAL_ZERO',
 'limits':['No archived output/X/W/counter tensor bytes for offline numerical regeneration; source-bound executed assertions and counter snapshots are validated.',
 'Instrumented CUDA durations are not cold event or official scored latencies; host spans include replay and synchronization.',
 'Three profile samples per case; uniform harness case counts are not serving request frequencies or score weights.',
 'One kernel per actual graph gives launch boundaries, not internal loop/stall/instruction costs; closed-library source is UNKNOWN.'],
 'GPU_NCU_API_queue_calls_by_reviewer':0}
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
