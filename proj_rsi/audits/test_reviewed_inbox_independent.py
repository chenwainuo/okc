"""Independent inbox negatives; temporary fixtures/FakeStore only, no QueueStore."""
import copy,hashlib,json,tempfile,threading,unittest
from pathlib import Path
from proj_rsi.scheduling import _reviewed_inbox_proposal as inbox
from proj_rsi.scheduling.model import load_policy

class FakeStore:
 def __init__(self,jobs=None):self.policy=load_policy();self.jobs=jobs or {};self.sequence=0;self.calls=[];self.lock=threading.Lock()
 def snapshot(self):
  with self.lock:return {'jobs':copy.deepcopy(self.jobs),'sequence':self.sequence}
 def register(self,job):
  with self.lock:
   if job['id'] in self.jobs:raise ValueError('existing id')
   self.calls.append(copy.deepcopy(job));self.jobs[job['id']]=copy.deepcopy(job);self.sequence+=1;return copy.deepcopy(job)

class IndependentInbox(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.rel='proj_rsi/scheduling/jobs/independent.json';self.path=self.root/self.rel;self.path.parent.mkdir(parents=True)
  def pin(rel,text):
   path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return hashlib.sha256(text.encode()).hexdigest()
  files={p:pin(p,'fixture '+p) for p in ('proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh','proj_rsi/tools/independent.py')}
  review='proj_rsi/scheduling/reviews/independent.md';rh=pin(review,'independent GO fixture')
  self.job={'id':'independent.attempt01','author':'author','resource':'gpu','kind':'optimization','state':'BACKLOG','created_at':0,'depends_on':[],'write_paths':[],'estimate':{'basis':'subjective hypothetical scenario','confidence':'low','evidence':[review],'direct':{'scenario_score_us':1,'p_gain':.1,'p_valid':.8},'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0}},'cost':{'gpu_seconds':300,'followup_gpu_seconds':0,'cpu_minutes':10},'files':files,'review':{'actor':'critic','path':review,'sha256':rh,'decision':'GO'},'launch':{'host':'root@107.170.6.60','remote_repo':'/root/okc','container':'okc-proj-independent-attempt01','inner_seconds':240,'outer_seconds':300,'profile':False,'env':{},'argv':['python3','/okc/proj_rsi/tools/independent.py'],'artifacts':['/root/okc/logs/independent.json']}}
  self.policy={'enabled':True,'controller_id':'root-worker','allowed_ids':[self.job['id']],'allowed_prefixes':[],'bounds':{'normal':{'inner_seconds':240,'outer_seconds':300},'profile':{'inner_seconds':240,'outer_seconds':300}},'admission_lock':'results/proj-rsi/scheduler/reviewed-inbox.proposal.lock'}
  self.store=FakeStore();self.write()
 def write(self):self.path.write_text(json.dumps(self.job))
 def inspect(self,**kwargs):return inbox.inspect_descriptor(self.root,self.rel,kwargs.get('snapshot',self.store.snapshot()),load_policy(),kwargs.get('policy',self.policy))
 def test_baseline(self):
  p=self.inspect();self.assertEqual(p['code'],'APPROVED');self.assertEqual(p['registration']['state'],'BACKLOG');self.assertAlmostEqual(p['roi_summary']['cost_usd'],300*5/3600)
 def test_existing_all_states(self):
  for state in ('BACKLOG','READY','HELD','RUNNING','VALIDATING','DONE','CANCELLED'):
   self.store.jobs={self.job['id']:{'state':state}};self.assertEqual(self.inspect()['code'],'EXISTING_ID_NEVER_MUTATE')
 def test_malformed_inputs_skip(self):
  for p in ({'enabled':False,'bad':float('nan')},[],None,{**self.policy,'bounds':[]},{**self.policy,'source_review_bindings':[]}):
   with self.subTest(policy=p):self.assertEqual(self.inspect(policy=p)['action'],'SKIP')
  self.assertEqual(self.inspect(snapshot={'jobs':{self.job['id']:None}})['action'],'SKIP')
 def test_partial_duplicate_nonfinite(self):
  for raw in ('{','{"id":"x","id":"y"}','{"id":NaN}'):
   self.path.write_text(raw);self.assertEqual(self.inspect()['action'],'SKIP')
 def test_pin_review_roi_and_state_negatives(self):
  original=copy.deepcopy(self.job)
  edits=[lambda j:j['review'].update(actor='author'),lambda j:j['review'].update(decision='HOLD'),lambda j:j.update(state='READY'),lambda j:j['cost'].update(gpu_seconds=None),lambda j:j['estimate']['direct'].update(p_gain=float('nan')),lambda j:j['files'].pop('proj_rsi/tools/independent.py'),lambda j:j.update(annotations={'source_review':None}),lambda j:j['launch'].update(outer_seconds=301)]
  for edit in edits:
   self.job=copy.deepcopy(original);edit(self.job);self.write();self.assertEqual(self.inspect()['action'],'SKIP')
 def test_dependencies_no_invalid_acceptance(self):
  self.job['depends_on']=[{'id':'dep','accepted_dispositions':['INVALID_SETUP']}];self.write();self.store.jobs={'dep':{'state':'DONE','disposition':'INVALID_SETUP'}};self.assertEqual(self.inspect()['action'],'SKIP')
  self.job['depends_on']=['dep'];self.write()
  for state,dispo in [('RUNNING','CPU_PASS'),('DONE',None),('DONE','INVALID_SETUP')]:
   self.store.jobs={'dep':{'state':state,'disposition':dispo}};self.assertEqual(self.inspect()['action'],'SKIP')
  self.store.jobs={'dep':{'id':'dep','state':'DONE','disposition':'CPU_PASS'}};self.assertEqual(self.inspect()['code'],'APPROVED')
 def test_apply_optin_only_and_races(self):
  r=inbox.sync_reviewed_new(self.store,self.root,[self.rel],self.policy);self.assertFalse(self.store.calls)
  def steal(p):self.store.register(p['registration'])
  r=inbox.sync_reviewed_new(self.store,self.root,[self.rel],self.policy,root_opt_in=True,before_register=steal);self.assertEqual(len(self.store.calls),1);self.assertEqual(r[0]['action'],'SKIP')
 def test_six_real_process_controllers_register_once(self):
  import multiprocessing as mp
  ctx=mp.get_context('fork')
  with ctx.Manager() as manager:
   jobs=manager.dict();calls=manager.list();gate=ctx.Lock();outputs=ctx.Queue()
   class SharedFake:
    policy=load_policy()
    def snapshot(self):
     with gate:return {'jobs':dict(jobs),'sequence':len(jobs)}
    def register(self,job):
     with gate:
      if job['id'] in jobs:raise ValueError('existing')
      jobs[job['id']]=job;calls.append(job['id']);return job
   def worker():
    result=inbox.sync_reviewed_new(SharedFake(),self.root,[self.rel],self.policy,root_opt_in=True)
    outputs.put(result[0]['action'])
   procs=[ctx.Process(target=worker) for _ in range(6)]
   for proc in procs:proc.start()
   for proc in procs:proc.join(10);self.assertEqual(proc.exitcode,0)
   actions=[outputs.get(timeout=2) for _ in procs]
   self.assertEqual(actions.count('REGISTERED_NEW'),1);self.assertEqual(len(calls),1)
 def test_drift_between_inspection_and_registration(self):
  def drift(p):self.job['cost']['gpu_seconds']=301;self.write()
  r=inbox.sync_reviewed_new(self.store,self.root,[self.rel],self.policy,root_opt_in=True,before_register=drift);self.assertFalse(self.store.calls);self.assertEqual(r[0]['action'],'SKIP')

if __name__=='__main__':unittest.main()
