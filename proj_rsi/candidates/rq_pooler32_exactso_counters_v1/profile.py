"""Bounded root-only NCU orchestration; CPU-check does not run NCU/CUDA."""
from pathlib import Path
import argparse,ast,csv,hashlib,io,json,math,os,re,signal,subprocess,sys,time,traceback,uuid
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
NCU='/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu'
RECIPE=ROOT/'proj_rsi/candidates/gv53_exactso_counters_v1/profile.py'
METRICS=next(ast.literal_eval(n.value)for n in ast.parse(RECIPE.read_text()).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='METRICS')
FUNCTION='_ZN10rq_control2rqILi32ELi6ELi1ELi0ELi0ELi0ELi35EEEvNS_3RqPENS_6RqTmapES2_'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def verify():
 m=json.loads((D/'runtime-manifest.json').read_text())
 for p,h in m['files'].items():assert sha(ROOT/p)==h,('pin drift',p)
 return m
def metric_names(raw):
 names=set(re.findall(r'(?<![A-Za-z0-9_.])([a-z][a-z0-9_]*__[A-Za-z0-9_.]+)(?![A-Za-z0-9_.])',raw));assert set(METRICS)<=names,sorted(set(METRICS)-names);return sorted(names)
def command(out,review):
 return [NCU,'--target-processes','application-only','--kill','no','--replay-mode','kernel','--graph-profiling','node','--cache-control','all','--clock-control','none','--nvtx','--nvtx-include','profile/','--launch-count','1','--metrics',','.join(METRICS),'--export',str(out/'selected'),'-f',sys.executable,str(D/'capture.py'),'--output',str(out),'--review',str(review)]
def drain_group(pid):
 try:os.killpg(pid,0)
 except ProcessLookupError:return True
 try:os.killpg(pid,signal.SIGKILL)
 except ProcessLookupError:return True
 deadline=time.monotonic()+3
 while time.monotonic()<deadline:
  try:os.killpg(pid,0)
  except ProcessLookupError:return True
  time.sleep(.05)
 return False
def parse(raw,c):
 assert c['state']=='VALID_EXACT_CURRENT_POOLER32_RQ_PROFILE_OUTPUTS'and c['phases'][-1]=='complete'
 assert len(c['raw'])==6 and sum(x['mode']=='profile'for x in c['raw'])==1
 assert len(c['validation'])==4 and len(c['mutations'])==5 and c['own_terminal']['before_any_later_replay_or_mutation']
 cleanup=c['post_scope_cleanup'];assert cleanup['allocated']==0 and cleanup['blas_after'][2]==0
 assert c['build']['SO_sha256']=='ab6c3ad71701098713f7c3435439f905c4d482ae66a995948bfcda254494a32f'and not c['build']['JIT']
 assert c['build']['native_arm']==0
 nodes=c['actual_graph_nodes'];assert len(nodes)==1;node=nodes[0]
 assert node['function_mangled']==FUNCTION and node['grid']==[32,4,1]and node['block']==[384,1,1]and node['dynamic_shared_bytes']==99488
 rows=list(csv.reader(io.StringIO(raw)));ix=[i for i,r in enumerate(rows)if 'ID'in r and'Kernel Name'in r];assert len(ix)==1
 h=rows[ix[0]];assert len(h)==len(set(h));units=dict(zip(h,rows[ix[0]+1]));assert set(METRICS)|{'profiler__replayer_passes','Block Size','Grid Size'}<=set(h)
 data=[]
 for row in rows[ix[0]+2:]:
  if not row or len(row)==1 and row[0].startswith('=='):continue
  assert len(row)==len(h);d=dict(zip(h,row));assert d['ID'].isdigit();data.append(d)
 assert len(data)==1;d=data[0]
 assert d['Kernel Name']==FUNCTION and d['Block Size'].replace(' ','')=='(384,1,1)'and d['Grid Size'].replace(' ','')=='(32,4,1)'
 values={k:float(d[k].replace(',',''))for k in METRICS};assert all(math.isfinite(v)and v>=0 for v in values.values())
 assert units[METRICS[0]]in('ns','us','ms','s')and values[METRICS[0]]>0
 for k in METRICS:
  if k.endswith('.pct')or'.pct_'in k:assert units[k]=='%',(k,units[k])
 raw_passes=float(d['profiler__replayer_passes'].replace(',',''));assert math.isfinite(raw_passes)and raw_passes>=1 and raw_passes.is_integer()and units['profiler__replayer_passes']=='pass'
 passes=int(raw_passes)
 return {'state':'VALID_EXACT_CURRENT_POOLER32_AGGREGATE_COUNTER_DIAGNOSTIC','kernel':FUNCTION,'metrics':{k:{'value':v,'unit':units[k]}for k,v in values.items()},'replay_passes':passes,'scope':'Aggregate counters only. No producer/consumer PC attribution, clean timing/speed gain or official score.','capture_SHA256':None}
def cpu_check():
 verify();assert len(METRICS)==12;metric_names('\n'.join(METRICS));c=command(Path('/tmp/probe'),Path('/tmp/review'));assert c[c.index('--launch-count')+1]=='1'
 return {'state':'CPU_ONE_KERNEL_PROFILE_RECIPE_PASS','metrics':list(METRICS),'NCU_GPU_subprocesses':0,'capture_timeout_max_seconds':60,'overall_inner_seconds':90,'outer_seconds':120,'process_group_termination':True}
def run(a):
 verify();assert a.review.is_file()and a.output.is_absolute()and not a.output.exists()and ROOT/'logs/scheduler'in a.output.parents
 a.output.mkdir(parents=True);out=a.output;begin=time.monotonic();s={'state':'STARTED','capture_token':uuid.uuid4().hex,'review_sha256':sha(a.review),'diagnostic_only':True,'bounds':{'inner':90,'outer':120,'NCU_child_max':60}}
 def save():dump(out/'session.json',s)
 def remaining(maximum):
  left=88-(time.monotonic()-begin);assert left>0,'overall88s processing bound';return min(maximum,left)
 save()
 try:
  q=subprocess.run([NCU,'--version'],capture_output=True,text=True,timeout=remaining(5));assert q.returncode==0 and re.search(r'\b2025\.1\.1(?:\.0)?(?=\s|$)',q.stdout+q.stderr)
  s['NCU_version']=q.stdout+q.stderr;s['NCU_sha256']=sha(NCU)
  argv=[NCU,'--chips','gh100','--query-metrics-mode','suffix','--query-metrics-collection','profiling','--metrics',','.join(dict.fromkeys(m.split('.')[0]for m in METRICS))]
  q=subprocess.run(argv,capture_output=True,text=True,timeout=remaining(12));(out/'metric-query.stdout').write_text(q.stdout);(out/'metric-query.stderr').write_text(q.stderr);s['metric_query_returncode']=q.returncode;s['metric_query_argv']=argv;save();assert q.returncode==0;s['available_metrics']=metric_names(q.stdout)
  s['capture_argv']=command(out,a.review);env=os.environ.copy();env['RQ_COUNTER_TOKEN']=s['capture_token'];save()
  with(out/'capture.stdout').open('x')as log:
   child=subprocess.Popen(s['capture_argv'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);s['capture_group_pid']=child.pid;save()
   try:s['capture_returncode']=child.wait(timeout=remaining(60))
   except BaseException:
    s['capture_timeout_or_orchestrator_error']=traceback.format_exc()
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     child.wait(timeout=3)
    s['capture_process_group_drained']=drain_group(child.pid);save();raise
  # The NCU leader has exited. Drain any surviving descendants in its unique group.
  s['capture_process_group_drained']=drain_group(child.pid)
  save();assert s['capture_returncode']==0 and s['capture_process_group_drained']
  report=out/'selected.ncu-rep';before=sha(report);argv=[NCU,'--import',str(report),'--page','raw','--csv','--print-kernel-base','mangled','--print-units','base'];s['export_argv']=argv;save()
  with(out/'raw.csv').open('x')as f,(out/'export.stderr').open('x')as err:q=subprocess.run(argv,stdout=f,stderr=err,timeout=remaining(12))
  assert q.returncode==0 and sha(report)==before
  c=json.loads((out/'result.json').read_text());analysis=parse((out/'raw.csv').read_text(),c);analysis['capture_SHA256']=sha(out/'result.json');dump(out/'analysis.json',analysis)
  s['state']='COMPLETE_EXACT_CURRENT_COUNTER_DIAGNOSTIC_ROOT_EXTERNAL_CLEANUP_REQUIRED'
 except BaseException:
  # Preserve capture numerical disposition separately: profiler/export failure is not a kernel correctness result.
  s['state']='INCOMPLETE_COUNTER_DIAGNOSTIC_NO_COUNTER_OR_GAIN_ADMISSION';s['error']=traceback.format_exc()
  if(out/'result.json').exists():s['capture_observed_state']=json.loads((out/'result.json').read_text()).get('state')
  save();raise
 finally:s['elapsed_seconds']=time.monotonic()-begin;s['artifacts']={p.name:sha(p)for p in out.iterdir()if p.is_file()and p.name!='session.json'};save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2))
 else:run(a)
