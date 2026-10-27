"""Independent CPU-only adversarial scheduling tests; temp state only."""
import copy,hashlib,json,multiprocessing as mp,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from proj_rsi.scheduling.model import load_policy,score
from proj_rsi.scheduling.store import QueueStore,CorruptWAL,QueueError

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def job(root,i,resource='gpu',value=10):
 return {'id':f'test.{i}','resource':resource,'kind':'optimization','state':'BACKLOG','author':'author','created_at':100,'depends_on':[],'write_paths':[],
 'estimate':{'direct':{'scenario_score_us':value,'p_gain':.5,'p_valid':.8},'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0,'unblocks':[],'decision':''},'basis':'subjective conditional prior, not measured','confidence':'low','evidence':['evidence.md']},
 'cost':{'gpu_seconds':30,'followup_gpu_seconds':0,'cpu_minutes':1},'files':{'source.py':digest(root/'source.py'),'proj_rsi/tools/h100_trial.sh':digest(root/'proj_rsi/tools/h100_trial.sh'),'proj_rsi/tools/h100_container.sh':digest(root/'proj_rsi/tools/h100_container.sh')},'review':{'actor':'critic','path':'review.md','sha256':digest(root/'review.md'),'decision':'GO'},
 'launch':{'host':'root@107.170.6.60','remote_repo':'/root/okc','container':f'okc-proj-test-{i}','inner_seconds':60,'outer_seconds':90,'profile':False,'env':{'MAX_JOBS':'1'},'argv':['python3','/okc/source.py'],'artifacts':['/root/okc/result.json']}}

def claim_worker(state,root,start,output):
 try:
  start.wait();r=QueueStore(state,repo_root=root).claim('gpu','worker');output.put(('ok',r['id'] if r else None))
 except Exception as e:output.put(('error',repr(e)))

class Adversarial(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.state=self.root/'state'
  for f in ('source.py','review.md','evidence.md'):(self.root/f).write_text('fixture\n')
  guard=self.root/'proj_rsi/tools/h100_trial.sh';guard.parent.mkdir(parents=True);guard.write_text('guard fixture\n');(guard.parent/'h100_container.sh').write_text('runner fixture\n')
  self.store=QueueStore(self.state,repo_root=self.root)
 def tearDown(self):self.tmp.cleanup()
 def test_real_concurrent_GPU_claims_one(self):
  for i in range(16):self.store.register(job(self.root,i))
  ctx=mp.get_context('fork');start=ctx.Event();output=ctx.Queue();procs=[ctx.Process(target=claim_worker,args=(str(self.state),str(self.root),start,output)) for _ in range(16)]
  for p in procs:p.start()
  start.set();answers=[output.get(timeout=30) for _ in procs]
  for p in procs:p.join(30);self.assertEqual(p.exitcode,0)
  self.assertTrue(all(kind=='ok' for kind,_ in answers),answers)
  self.assertEqual(sum(v is not None for _,v in answers),1,answers)
  self.assertEqual(sum(j['state']=='RUNNING' for j in QueueStore(self.state,repo_root=self.root).snapshot()['jobs'].values()),1)
 def test_CPU_cap_and_conflict(self):
  for i in range(13):self.store.register(job(self.root,i,'cpu'))
  self.assertEqual(sum(self.store.claim('cpu','worker') is not None for _ in range(13)),10)
 def test_CPU_path_overlap(self):
  a=job(self.root,0,'cpu');b=job(self.root,1,'cpu');a['write_paths']=['foo'];b['write_paths']=['foo/bar']
  self.store.register(a);self.store.register(b);self.assertIsNotNone(self.store.claim('cpu','one'));self.assertIsNone(self.store.claim('cpu','two'))
 def test_topK_does_not_preempt_RUNNING(self):
  policy=load_policy();policy['ready_capacity']['gpu']=1;s=QueueStore(self.state,repo_root=self.root,policy=policy)
  s.register(job(self.root,0,value=1));running=s.claim('gpu','worker');s.register(job(self.root,1,value=100));s.rebalance('gpu')
  self.assertEqual(s.get(running['id'])['state'],'RUNNING');self.assertIsNone(s.claim('gpu','other'))
 def test_source_changed_after_admission(self):
  self.store.register(job(self.root,0));(self.root/'source.py').write_text('drift')
  self.assertIsNone(self.store.claim('gpu','worker'));self.assertEqual(self.store.get('test.0')['state'],'HELD')
 def test_NCU_cannot_hide_shell_target_before_later_python(self):
  from proj_rsi.scheduling.model import validate_job
  p=self.root/'proj_rsi/tools/pinned.py';p.write_text('fixture');j=job(self.root,0);j['kind']='profile';j['estimate']['direct']['scenario_score_us']=0;j['launch']['profile']=True;j['files']['proj_rsi/tools/pinned.py']=digest(p);j['launch']['argv']=['ncu','--clock-control','none','bash','-c','arbitrary shell code','python3','/okc/proj_rsi/tools/pinned.py']
  self.assertTrue(validate_job(j,for_admission=True))
 def test_NCU_clock_changes_or_implicit_defaults_rejected(self):
  from proj_rsi.scheduling.model import validate_job
  p=self.root/'proj_rsi/tools/pinned.py';p.write_text('fixture');j=job(self.root,0);j['kind']='profile';j['estimate']['direct']['scenario_score_us']=0;j['launch']['profile']=True;j['files']['proj_rsi/tools/pinned.py']=digest(p)
  for flags in ([],['--clock-control','base']):
   j['launch']['argv']=['ncu',*flags,'python3','/okc/proj_rsi/tools/pinned.py'];self.assertTrue(validate_job(j,for_admission=True))
 def test_unknown_numeric_priors(self):
  for value in (None,float('nan'),float('inf'),-1,10**1000):
   with self.subTest(value=str(value)[:30]):
    j=job(self.root,0);j['estimate']['direct']['p_gain']=value;r=score(j);self.assertIsNone(r['priority']);self.assertFalse(r['rankable'])
 def test_diagnostic_cannot_evade_information_gate_via_direct_branch(self):
  for kind in ('diagnostic','profile'):
   j=job(self.root,0);j['kind']=kind;self.assertIsNone(score(j)['priority'])
 def test_information_needs_decision_and_followup_cost(self):
  j=job(self.root,0);j['estimate']['information']={'scenario_score_us':10,'p_resolve':.5,'p_followup_gain':.5,'unblocks':[],'decision':''}
  self.assertIsNone(score(j)['priority']);j['estimate']['information'].update(unblocks=['followup.1'],decision='choose next kernel');self.assertIsNone(score(j)['priority'])
  j['cost']['followup_gpu_seconds']=30;self.assertTrue(score(j)['rankable'])
 def test_torn_tail_preserved_restart_no_duplicate(self):
  self.store.register(job(self.root,0));running=self.store.claim('gpu','worker');tail=b'{"payload":'
  with self.store.wal_path.open('ab') as f:f.write(tail)
  restarted=QueueStore(self.state,repo_root=self.root);self.assertEqual(restarted.get(running['id'])['state'],'RUNNING');self.assertIsNone(restarted.claim('gpu','again'))
  self.assertTrue(any(p.read_bytes()==tail for p in self.state.glob('torn-final-*.bin')))
 def test_middle_corruption_fails_closed(self):
  self.store.register(job(self.root,0));self.store.register(job(self.root,1));raw=self.store.wal_path.read_bytes();self.store.wal_path.write_bytes(b'garbage\n'+raw)
  with self.assertRaises(CorruptWAL):self.store.claim('gpu','worker')
 def test_WAL_claim_survives_checkpoint_failure(self):
  from unittest.mock import patch
  self.store.register(job(self.root,0))
  with patch('proj_rsi.scheduling.store._atomic',side_effect=OSError('checkpoint failure')):
   with self.assertRaises(OSError):self.store.claim('gpu','worker')
  restarted=QueueStore(self.state,repo_root=self.root);self.assertEqual(restarted.get('test.0')['state'],'RUNNING');self.assertIsNone(restarted.claim('gpu','again'))
 def test_checkpoint_not_authoritative(self):
  self.store.register(job(self.root,0));self.store.claim('gpu','worker');self.store.checkpoint_path.write_text('{"jobs":{}}')
  self.assertIsNone(QueueStore(self.state,repo_root=self.root).claim('gpu','again'))
 def test_policy_cannot_raise_fixed_GPU_cap(self):
  policy=load_policy();policy['active_capacity']['gpu']=2
  try:
   s=QueueStore(self.state,repo_root=self.root,policy=policy);s.register(job(self.root,0));s.register(job(self.root,1))
  except (ValueError,QueueError):return
  self.assertIsNotNone(s.claim('gpu','one'));self.assertIsNone(s.claim('gpu','two'))
 def test_changed_launch_requires_fresh_review(self):
  self.store.register(job(self.root,0));launch=job(self.root,0)['launch'];launch['argv'].append('--different')
  updated=self.store.update('test.0',{'launch':launch},actor='author',reason='change execution')
  self.assertNotEqual(updated['state'],'READY');self.assertIsNone(self.store.claim('gpu','worker'))
 def test_active_execution_immutable(self):
  self.store.register(job(self.root,0));self.store.claim('gpu','worker')
  with self.assertRaises((ValueError,QueueError)):self.store.update('test.0',{'launch':job(self.root,0)['launch']},actor='root',reason='rewrite active')
 def test_invalid_dependency_never_ready(self):
  parent=job(self.root,0);parent.update(state='DONE',disposition='INVALID_CORRECTNESS');self.store.register(parent)
  child=job(self.root,1);child['depends_on']=['test.0'];self.assertEqual(self.store.register(child)['state'],'BACKLOG');self.assertIsNone(self.store.claim('gpu','worker'))
 def test_same_review_content_cannot_be_reused_by_changed_actor(self):
  self.store.register(job(self.root,0));j=job(self.root,0);j['launch']['argv'].append('--changed');j['review']['actor']='another-critic'
  self.store.update('test.0',{'launch':j['launch'],'review':j['review']},actor='author',reason='change execution')
  self.assertIsNone(self.store.claim('gpu','worker'))
 def test_GPU_finish_without_cleanup_rejected(self):
  self.store.register(job(self.root,0));self.store.claim('gpu','worker')
  with self.assertRaises((ValueError,QueueError)):self.store.finish('test.0','VALID_GAIN',evidence={'exit':0},actor='root')
 def test_author_cannot_self_validate_GPU(self):
  self.store.register(job(self.root,0));self.store.claim('gpu','worker');self.store.finish('test.0','VALID_GAIN',evidence={'exit':0,'cleanup_verified':True,'cleanup_observation':{'container_exists':False,'guard_pids':[],'compute_pids':[],'gpu_memory_mib':0,'containers':[],'cleanup_failed':False}},actor='worker')
  with self.assertRaises((ValueError,QueueError)):self.store.transition('test.0','DONE',actor='author',reason='self declares pass',evidence={'disposition':'VALID_GAIN','exit':0})


class FakeTransport:
 def __init__(self,root):self.root=root;self.launches=[];self.probes=[];self.launch_error=None;self.busy=False;self.terminal=None;self.mutate_upload=False
 def upload(self,host,local,remote,paths):
  if self.mutate_upload:(self.root/'source.py').write_text('changed during upload')
 def runner(self,host):return {'sha256':digest(self.root/'proj_rsi/tools/h100_container.sh'),'mode':0o755}
 def hashes(self,host,remote,paths):return {p:digest(self.root/p) for p in paths}
 def probe(self,host,handles):
  self.probes.append(copy.deepcopy(handles))
  if self.probes and getattr(self,'probe_error',None):raise self.probe_error
  return {'container_exists':self.busy,'guard_pids':[123] if self.busy else [],'compute_pids':[456] if self.busy else [],'gpu_memory_mib':10 if self.busy else 0,'containers':[handles['container']] if self.busy else [],'terminal_exit':self.terminal,'cleanup_failed':False}
 def launch(self,host,command):
  self.launches.append(command)
  if self.launch_error:raise self.launch_error
  return '123\n'
 def inventory(self,host,remote,paths):return {p:hashlib.sha256(b'raw fixture\n').hexdigest() for p in paths}
 def download(self,host,remote,paths,destination):
  for p in paths:
   f=Path(destination)/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'raw fixture\n')

class DispatcherAdversarial(Adversarial):
 # Inherits queue tests too, providing one repeat under the integrated suite.
 def setUp(self):
  super().setUp()
  from proj_rsi.scheduling.dispatch import GPUDispatcher
  self.transport=FakeTransport(self.root);self.dispatcher=GPUDispatcher(self.store,self.root,transport=self.transport)
 def test_ambiguous_launch_never_duplicates(self):
  self.store.register(job(self.root,0));self.transport.launch_error=TimeoutError('unknown SSH outcome')
  launched=self.dispatcher.launch_next();self.assertEqual(launched['state'],'RUNNING');self.assertEqual(len(self.transport.launches),1)
  self.transport.probe_error=TimeoutError('unknown probe')
  self.assertEqual(self.dispatcher.reconcile('test.0')['status'],'UNKNOWN');self.assertIsNone(self.dispatcher.launch_next());self.assertEqual(len(self.transport.launches),1)
 def test_quoting_no_extra_hardware_lock(self):
  import shlex
  j=job(self.root,0);literal='a; $(touch /tmp/should-never-exist) * \" quoted'
  j['launch']['argv'].append(literal);j['launch']['env']['TEST_LITERAL']='a; echo unsafe'
  handles={'host_log':'/root/okc/logs/scheduler/test.host.log'}
  outer=shlex.split(self.dispatcher._command(j['launch'],handles));self.assertEqual(outer[:2],['bash','-c']);lines=outer[2].splitlines();tokens=shlex.split(next(line for line in lines if line.startswith('nohup ')))
  self.assertIn(literal,tokens);self.assertIn('TEST_LITERAL=a; echo unsafe',tokens);self.assertNotIn('flock',tokens);self.assertEqual(sum(line.startswith('nohup ') for line in lines),1)
 def test_restarted_dispatcher_preserves_unknown_handles(self):
  from proj_rsi.scheduling.dispatch import GPUDispatcher
  self.store.register(job(self.root,0));self.transport.launch_error=TimeoutError('ambiguous launch');self.dispatcher.launch_next()
  restarted=GPUDispatcher(QueueStore(self.state,repo_root=self.root),self.root,transport=self.transport);self.transport.probe_error=TimeoutError('unknown')
  self.assertEqual(restarted.reconcile('test.0')['status'],'UNKNOWN');self.assertIsNone(restarted.launch_next());self.assertEqual(len(self.transport.launches),1)
 def test_unknown_launch_cannot_be_released_as_unlaunched(self):
  self.store.register(job(self.root,0));self.transport.launch_error=TimeoutError('unknown');self.dispatcher.launch_next()
  with self.assertRaises((ValueError,QueueError)):self.store.release_unlaunched('test.0',actor='root',reason='attempt unsafe lease release')
  self.assertEqual(self.store.get('test.0')['state'],'RUNNING')
 def test_installed_runner_mismatch_blocks_launch(self):
  self.store.register(job(self.root,0));self.transport.runner=lambda host:{'sha256':'0'*64,'mode':0o755}
  self.assertEqual(self.dispatcher.launch_next()['state'],'HELD');self.assertEqual(self.transport.launches,[])
 def test_busy_no_launch_clean_block(self):
  self.store.register(job(self.root,0));self.transport.busy=True;result=self.dispatcher.launch_next();self.assertIn(result['state'],('HELD','VALIDATING'));self.assertEqual(self.transport.launches,[])
 def test_drift_during_upload_never_launches(self):
  self.store.register(job(self.root,0));self.transport.mutate_upload=True;result=self.dispatcher.launch_next();self.assertIn(result['state'],('HELD','VALIDATING'));self.assertEqual(self.transport.launches,[])
 def test_exit0_requires_root_validation(self):
  self.store.register(job(self.root,0));self.dispatcher.launch_next();self.transport.terminal=0;result=self.dispatcher.collect('test.0')
  self.assertEqual(result['status'],'AWAITING_ROOT_VALIDATION');self.assertEqual(self.store.get('test.0')['state'],'VALIDATING');self.assertNotIn('disposition',self.store.get('test.0'))
 def test_live_process_blocks_collection_despite_exitmarker(self):
  self.store.register(job(self.root,0));self.dispatcher.launch_next();self.transport.terminal=0;self.transport.busy=True
  self.assertEqual(self.dispatcher.collect('test.0')['status'],'AWAITING_TERMINAL_CLEANUP');self.assertEqual(self.store.get('test.0')['state'],'RUNNING')

if __name__=='__main__':unittest.main(verbosity=2)
