"""Root-only bounded exact-SO NCU orchestration. CPU-check invokes no hardware/tool subprocess."""
import argparse,csv,hashlib,io,json,math,os,re,subprocess,sys,time,traceback,uuid
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
NCU='/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu'
METRICS=(
'gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
'dram__throughput.avg.pct_of_peak_sustained_elapsed',
'smsp__issue_active.avg.pct_of_peak_sustained_active',
'smsp__warps_eligible.avg.per_cycle_active',
'smsp__inst_executed.sum',
'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
'smsp__warp_issue_stalled_wait_per_warp_active.pct',
'smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct',
'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct')
LABELS=('control','quarter','sparse')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 assert __debug__
 m=json.loads((D/'runtime-manifest.json').read_text())
 for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
 return m

def metric_names(raw):
 names=set(re.findall(r'(?<![A-Za-z0-9_.])([a-z][a-z0-9_]*__[A-Za-z0-9_.]+)(?![A-Za-z0-9_.])',raw))
 assert set(METRICS)<=names,('required counters unavailable',sorted(set(METRICS)-names))
 return sorted(names)
def command(out,review):
 return [NCU,'--target-processes','application-only','--kill','no','--replay-mode','kernel','--graph-profiling','node','--cache-control','all','--clock-control','none','--nvtx','--nvtx-include','profile/','--launch-count','3','--metrics',','.join(METRICS),'--export',str(out/'selected'),'-f',sys.executable,str(D/'capture.py'),'--output',str(out),'--review',str(review)]
def parse(raw,capture):
 assert capture['state']=='THREE_ARM_EXACTSO_PROFILE_COMPLETE_ROOT_CLEANUP_REQUIRED' and capture['phase']=='complete' and not capture['cleanup']
 cl=capture['post_scope_cleanup'];assert cl['allocated']==cl['reserved']==0 and cl['clear_error']is None and cl['blas_map_entries_workspace_bytes_after'][2]==0
 cell=capture['fixed_cells']['8'];assert set(cell['terminals'])==set(LABELS)and len(cell['mutations'])==15
 assert len(cell['raw'])==18 and [r['arm']for r in cell['raw']if r['mode']=='profile']==list(LABELS)
 rows=list(csv.reader(io.StringIO(raw)));ix=[i for i,r in enumerate(rows)if 'Kernel Name'in r and'ID'in r];assert len(ix)==1
 h=rows[ix[0]];assert len(set(h))==len(h);units=dict(zip(h,rows[ix[0]+1]));assert set(METRICS)|{'profiler__replayer_passes','Block Size','Grid Size'}<=set(h)
 data=[]
 for row in rows[ix[0]+2:]:
  if not row or len(row)==1 and row[0].startswith('=='):continue
  assert len(row)==len(h);r=dict(zip(h,row));assert r['ID'].isdigit();data.append(r)
 assert len(data)==3 and len({r['ID']for r in data})==3
 result=[]
 for arm,row in zip(LABELS,data):
  nodes=capture['actual_graph_nodes'][arm];assert len(nodes)==1;node=nodes[0]
  assert node['node_type']==0 and node['cluster_query_returncode']==0 and node['cluster_dimension']in([0,0,0],[1,1,1])
  assert node['grid']==[1,128,1]and node['block']==[1024,1,1]and node['dynamic_shared_bytes']==17408
  assert len(node['SASS_sha256'])==64 and node['attributes']['local_bytes']==0
  assert capture['arm_bindings'][arm]['SO_sha256']in('2eb511c0af3c4ce5bd3990e45d376c64d12016b876a2109ac0f263eafc8c17a7','4b85c059c953e91989f6d613e8148269cfbb8e0480ab120585abcac492c3e44c')
  assert row['Kernel Name']==node['function_mangled']
  assert row['Block Size'].replace(' ','')=='(1024,1,1)' and row['Grid Size'].replace(' ','')=='(1,128,1)'
  terminal=cell['terminals'][arm];assert terminal['before_any_replay_or_mutation'] and terminal['oracle']['current_X']=='EXACT'
  values={k:float(row[k].replace(',',''))for k in METRICS};assert all(math.isfinite(v)and v>=0 for v in values.values())
  passes=float(row['profiler__replayer_passes'].replace(',',''));assert passes>=1 and passes.is_integer() and units['profiler__replayer_passes']=='pass'
  assert units[METRICS[0]]in('ns','us','ms','s')and values[METRICS[0]]>0
  assert all(units[k]=='%'for k in METRICS if k.endswith('.pct')or'.pct_'in k)
  result.append({'arm':arm,'kernel':row['Kernel Name'],'metrics':{k:{'value':v,'unit':units[k]}for k,v in values.items()},'replay_passes':int(passes),'SO_binding':capture['arm_bindings'][arm]})
 return {'state':'VALID_AGGREGATE_COUNTER_DIAGNOSTIC_ONLY','arms':result,'gain_us':None,'official_score':None,'repair_time_fraction':None,'clock_context':capture['clock_context'],'limits':'NCU kernel replay/cache flush instrumentation changes duration/state. Stall percentages are scheduler samples, not kernel time fractions; aggregates cannot assign stalls to repair PCs. Clock snapshots bracket flush/profile/oracles, not perpass continuous telemetry.'}
def cpu_check():
 verify();metric_names('\n'.join(METRICS))
 try:metric_names('\n'.join(METRICS[:-1]))
 except AssertionError:pass
 else:raise AssertionError('missing metric admitted')
 c=command(Path('/CPU_ONLY_NOT_CREATED'),Path('/CPU_ONLY_REVIEW'));assert c[c.index('--clock-control')+1]=='none'and c[c.index('--launch-count')+1]=='3'
 assert '--graph-profiling'in c and '--nvtx-include'in c
 return {'state':'CPU_PROFILE_RECIPE_PASS','GPU_NCU_subprocess_calls':0,'profile_launches_max':3,'metrics':METRICS,'runtime_capability':'PinnedNCUversion/exactmetricquery/liveclusterattrsrequired'}
def run(a):
 verify();assert a.review.is_file()and a.output.is_absolute()and a.output.resolve()==a.output and ROOT/'logs/scheduler'in a.output.parents and not a.output.exists()
 a.output.mkdir(parents=True);out=a.output;session={'state':'STARTED','capture_token':uuid.uuid4().hex,'review_sha256':sha(a.review),'profile_durations_are_gain_evidence':False,'NCU_child_timeout_seconds':240,'orchestrator_overhead_budget_seconds':60}
 def save():(out/'session.json').write_text(json.dumps(session,indent=2)+'\n')
 save()
 try:
  v=subprocess.run([NCU,'--version'],capture_output=True,text=True,timeout=5);assert v.returncode==0 and re.search(r'\b2025\.1\.1(?:\.0)?(?=\s|$)',v.stdout+v.stderr)
  session['ncu_version']=v.stdout+v.stderr;session['ncu_sha256']=sha(NCU)
  argv=[NCU,'--chips','gh100','--query-metrics-mode','suffix','--query-metrics-collection','profiling','--metrics',','.join(dict.fromkeys(m.split('.')[0]for m in METRICS))]
  q=subprocess.run(argv,capture_output=True,text=True,timeout=15);(out/'metric-query.stdout').write_text(q.stdout);(out/'metric-query.stderr').write_text(q.stderr);session['metric_query_argv']=argv;session['metric_query_returncode']=q.returncode;save();assert q.returncode==0
  session['available_metrics']=metric_names(q.stdout);session['capture_argv']=command(out,a.review);save()
  env=os.environ.copy();env['GV53_COUNTER_TOKEN']=session['capture_token']
  with(out/'capture.stdout').open('x')as log:r=subprocess.run(session['capture_argv'],stdout=log,stderr=subprocess.STDOUT,timeout=240,env=env)
  session['capture_returncode']=r.returncode;save();assert r.returncode==0
  report=out/'selected.ncu-rep';before=sha(report);export=[NCU,'--import',str(report),'--page','raw','--csv','--print-kernel-base','mangled','--print-units','base'];session['export_argv']=export;save()
  with(out/'raw.csv').open('x')as f,(out/'export.stderr').open('x')as err:e=subprocess.run(export,stdout=f,stderr=err,timeout=30)
  assert e.returncode==0 and sha(report)==before
  capture=json.loads((out/'result.json').read_text());analysis=parse((out/'raw.csv').read_text(),capture);(out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
  session['state']='COMPLETE_COUNTER_DIAGNOSTIC_ROOT_EXTERNAL_CLEANUP_REQUIRED';session['artifact_hashes']={p.name:sha(p)for p in out.iterdir()if p.is_file()and p.name!='session.json'};save()
 except BaseException:session['state']='INVALID_OR_INCOMPLETE_NO_COUNTER_ADMISSION';session['failure']=traceback.format_exc();save();raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2))
 else:run(a)
