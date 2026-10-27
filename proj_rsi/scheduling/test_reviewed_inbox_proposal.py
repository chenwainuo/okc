"""Isolated FakeStore-only tests. No QueueStore, SSH, CLI, API or GPU instance."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import tempfile,threading,unittest
from unittest.mock import patch

from ._reviewed_inbox_proposal import inspect_descriptor,sync_reviewed_new
from .model import load_policy

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class FakeStore:
 def __init__(self):self.policy=load_policy();self.jobs={};self.sequence=0;self.calls=[];self.mutex=threading.Lock();self.on_register=None
 def snapshot(self):return {'version':1,'sequence':self.sequence,'jobs':deepcopy(self.jobs)}
 def register(self,job):
  if self.on_register:self.on_register(job)
  with self.mutex:
   if job['id'] in self.jobs:raise ValueError('atomic duplicate')
   self.calls.append(deepcopy(job));self.jobs[job['id']]=deepcopy(job);self.sequence+=1;return deepcopy(job)

class InboxTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.store=FakeStore()
  for rel,content in [('proj_rsi/tools/lab.py','print("local GPU research")'),('proj_rsi/tools/h100_trial.sh','guard'),('proj_rsi/tools/h100_container.sh','container'),('reviews/driver.md','Independent GO'),('reviews/source.md','Independent source GO')]:
   p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
  self.files={rel:sha(self.root/rel) for rel in ('proj_rsi/tools/lab.py','proj_rsi/tools/h100_trial.sh','proj_rsi/tools/h100_container.sh','reviews/source.md')}
  self.job={'id':'local.v1.attempt01','resource':'gpu','kind':'optimization','state':'BACKLOG','author':'implementer','created_at':1,'depends_on':[],'write_paths':[],
   'estimate':{'direct':{'scenario_score_us':30,'p_gain':.2,'p_valid':.8},'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0},'basis':'subjective conditional prior; GPU seconds include full transfer/compile/prepare/validation/timing reservation','confidence':'low','evidence':['reviews/source.md']},
   'cost':{'gpu_seconds':120,'followup_gpu_seconds':0,'cpu_minutes':10},'files':self.files,
   'review':{'actor':'critic-driver','decision':'GO','path':'reviews/driver.md','sha256':sha(self.root/'reviews/driver.md')},
   'launch':{'host':'root@107.170.6.60','remote_repo':'/root/okc','container':'okc-proj-local-v1','inner_seconds':240,'outer_seconds':300,'profile':False,'env':{'MAX_JOBS':'1'},'argv':['python3','/okc/proj_rsi/tools/lab.py','--output','/okc/logs/child'],'artifacts':['/root/okc/logs/child']}}
  self.policy={'enabled':True,'controller_id':'root-worker','allowed_ids':[self.job['id']],'allowed_prefixes':[],
   'bounds':{'normal':{'inner_seconds':3300,'outer_seconds':3420},'profile':{'inner_seconds':240,'outer_seconds':300}},
   'source_review_bindings':{},'admission_lock':'results/proj-rsi/scheduler/reviewed-inbox.proposal.lock'}
  self.relative='proj_rsi/scheduling/jobs/local-v1.json';self.write()
 def write(self,job=None):
  p=self.root/self.relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(self.job if job is None else job));return p
 def inspect(self,**kw):return inspect_descriptor(self.root,self.relative,self.store.snapshot(),self.store.policy,self.policy,**kw)
 def apply(self,**kw):return sync_reviewed_new(self.store,self.root,[self.relative],self.policy,root_opt_in=True,**kw)
 def test_default_off_has_no_store_or_filesystem_effect(self):
  class Forbidden:
   def __getattr__(self,key):raise AssertionError('store accessed while disabled')
  before=set(self.root.rglob('*'))
  result=sync_reviewed_new(Forbidden(),self.root,[self.relative],self.policy)
  self.assertEqual(result[0]['action'],'SKIP');self.assertEqual(before,set(self.root.rglob('*')))
 def test_pure_inspection_proposes_without_any_registration(self):
  before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
  decision=self.inspect();self.assertEqual(decision['code'],'APPROVED')
  self.assertEqual(decision['registration']['state'],'BACKLOG');self.assertEqual(self.store.calls,[])
  self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
  self.assertAlmostEqual(decision['roi_summary']['cost_usd'],120*5/3600)
 def test_apply_registers_only_new_backlog_and_never_claims_or_launches(self):
  result=self.apply();self.assertEqual(result[0]['action'],'REGISTERED_NEW')
  self.assertEqual(len(self.store.calls),1);self.assertEqual(self.store.calls[0]['state'],'BACKLOG')
  self.assertEqual(self.apply()[0]['code'],'EXISTING_ID_NEVER_MUTATE');self.assertEqual(len(self.store.calls),1)
 def test_all_existing_states_including_held_are_immutable(self):
  for state in ('BACKLOG','READY','RUNNING','VALIDATING','HELD','DONE','CANCELLED'):
   self.store.jobs={self.job['id']:{'id':self.job['id'],'state':state,'sentinel':123}}
   before=deepcopy(self.store.jobs);self.assertEqual(self.apply()[0]['code'],'EXISTING_ID_NEVER_MUTATE')
   self.assertEqual(self.store.jobs,before);self.assertEqual(self.store.calls,[])
 def test_partial_duplicate_key_and_nonfinite_json_skip_without_registration(self):
  for text in ('{"id":','{"id":"x","id":"y"}','{"cost":NaN}'):
   (self.root/self.relative).write_text(text);self.assertEqual(self.apply()[0]['action'],'SKIP');self.assertEqual(self.store.calls,[])
 def test_missing_file_hash_drift_hold_or_no_independence_skip(self):
  for kind in ('missing','drift','HOLD','same-author'):
   job=deepcopy(self.job)
   if kind=='missing':job['files']['missing.py']='a'*64
   if kind=='drift':job['files']['proj_rsi/tools/lab.py']='a'*64
   if kind=='HOLD':job['review']['decision']='HOLD'
   if kind=='same-author':job['review']['actor']=job['author']
   self.write(job);self.assertEqual(self.apply()[0]['action'],'SKIP',kind)
  self.assertEqual(self.store.calls,[])
 def test_only_root_allowlisted_local_gpu_backlog(self):
  for kind in ('not-allowed','cpu','DONE','root-impostor','official-module'):
   job=deepcopy(self.job);kw={}
   if kind=='not-allowed':job['id']='other.v1'
   if kind=='cpu':job['resource']='cpu'
   if kind=='DONE':job['state']='DONE'
   if kind=='root-impostor':kw['controller']='other'
   if kind=='official-module':job['launch']['argv']=['python3','-m','proj_rsi','check','x']
   self.write(job);self.assertEqual(self.apply(**kw)[0]['action'],'SKIP',kind)
  self.assertEqual(self.store.calls,[])
 def test_root_bounds_profile_kind_and_five_dollars_cannot_be_overridden(self):
  job=deepcopy(self.job);job['launch'].update(inner_seconds=3400,outer_seconds=3520);self.write(job)
  self.assertEqual(self.apply()[0]['code'],'ROOT_WORKER_BOUND_EXCEEDED')
  self.write();self.store.policy['gpu_usd_per_hour']=1
  self.assertEqual(self.apply()[0]['action'],'SKIP')
  self.store.policy=load_policy();job=deepcopy(self.job);job['launch']['profile']=True;self.write(job)
  self.assertEqual(self.apply()[0]['action'],'SKIP')
 def test_dependencies_done_valid_only_and_no_invalid_override(self):
  for state,disposition,accepted,want in [('RUNNING','CPU_PASS',None,False),('VALIDATING','VALID_GAIN',None,False),('DONE','INVALID_CORRECTNESS',['INVALID_CORRECTNESS'],False),('DONE','CPU_PASS',['CPU_PASS'],True),('DONE','VALID_REGRESSION',None,True)]:
   job=deepcopy(self.job);job['depends_on']=[{'id':'parent.v1',**({'accepted_dispositions':accepted} if accepted is not None else {})}]
   self.store.jobs={'parent.v1':{'id':'parent.v1','state':state,'disposition':disposition}}
   self.write(job);self.assertEqual(self.inspect()['code']=='APPROVED',want,(state,disposition))
 def test_source_review_annotation_requires_separate_full_pinned_receipt(self):
  job=deepcopy(self.job);job['annotations']={'source_review':'source-review GO shortsha'};self.write(job)
  self.assertEqual(self.apply()[0]['code'],'SOURCE_REVIEW_UNRESOLVED_OR_INVALID')
  receipt={'actor':'critic-source','decision':'GO','path':'reviews/source.md','sha256':sha(self.root/'reviews/source.md')}
  self.policy['source_review_bindings'][job['id']]=receipt
  self.assertEqual(self.inspect()['code'],'APPROVED')
  for key,value in [('actor','implementer'),('decision','HOLD'),('sha256','0'*64),('path','reviews/driver.md')]:
   wrong=deepcopy(receipt);wrong[key]=value;self.policy['source_review_bindings'][job['id']]=wrong
   self.assertEqual(self.inspect()['action'],'SKIP',key)
 def test_source_review_null_or_bad_annotations_do_not_skip_dual_review_requirement(self):
  for annotations in ({'source_review':None},'not-an-object'):
   job=deepcopy(self.job);job['annotations']=annotations;self.write(job)
   self.assertEqual(self.inspect()['action'],'SKIP')
 def test_fresh_revalidation_rejects_descriptor_or_source_change(self):
  def drift(proposal):(self.root/'proj_rsi/tools/lab.py').write_text('changed after approval')
  self.assertEqual(self.apply(before_register=drift)[0]['action'],'SKIP');self.assertEqual(self.store.calls,[])
 def test_atomic_duplicate_register_race_does_not_update_or_retry(self):
  def race(job):self.store.jobs[job['id']]={'id':job['id'],'state':'RUNNING','owner':'other-controller'}
  self.store.on_register=race;result=self.apply()
  self.assertEqual(result[0]['code'],'REGISTRATION_RACE_EXISTING_ID');self.assertEqual(self.store.jobs[self.job['id']]['owner'],'other-controller')
  self.assertEqual(self.store.calls,[])
 def test_existing_container_conflict_and_dedicated_lock(self):
  self.store.jobs={'other.v1':{'id':'other.v1','state':'DONE','launch':{'container':'okc-proj-local-v1'}}}
  self.assertEqual(self.apply()[0]['action'],'SKIP')
  self.store.jobs={};self.policy['admission_lock']='results/proj-rsi/scheduler/.lock'
  self.assertEqual(self.apply()[0]['action'],'SKIP');self.assertEqual(self.store.calls,[])
 def test_malformed_policy_and_snapshot_fail_closed_not_exception(self):
  variants=[None,{'enabled':False,'bad':float('nan')},dict(self.policy,bounds=[]),dict(self.policy,source_review_bindings=[]),dict(self.policy,allowed_prefixes=[''])]
  for policy in variants:
   decision=inspect_descriptor(self.root,self.relative,self.store.snapshot(),self.store.policy,policy)
   self.assertEqual(decision['action'],'SKIP');self.assertTrue(decision['reason_sha256'])
  for jobs in ({self.job['id']:None},{self.job['id']:{'state':{'bad':float('nan')}}}):
   self.assertEqual(inspect_descriptor(self.root,self.relative,{'jobs':jobs,'sequence':float('nan')},self.store.policy,self.policy)['action'],'SKIP')
 def test_allowed_prefix_and_explicit_profile_keep_chain_cost(self):
  self.policy['allowed_ids']=[];self.policy['allowed_prefixes']=['local.']
  job=deepcopy(self.job);job['kind']='diagnostic';job['launch']['profile']=True
  job['launch']['argv']=['ncu','--clock-control','none','python3','/okc/proj_rsi/tools/lab.py']
  job['estimate']['direct']['scenario_score_us']=0
  job['estimate']['information']={'scenario_score_us':30,'p_resolve':.5,'p_followup_gain':.2,'unblocks':['followup.v1'],'decision':'choose one next version'}
  job['cost']['followup_gpu_seconds']=120;self.write(job)
  decision=self.inspect();self.assertEqual(decision['code'],'APPROVED')
  self.assertAlmostEqual(decision['roi_summary']['cost_usd'],240*5/3600)
  self.assertGreater(decision['roi_summary']['information_expected_us'],0)
 def test_pinned_unknown_cost_and_official_entrypoint_are_not_admitted(self):
  job=deepcopy(self.job);job['cost']['gpu_seconds']=None;self.write(job)
  self.assertEqual(self.inspect()['action'],'SKIP')
  job=deepcopy(self.job);job['launch']['argv']=['python3','/okc/proj_rsi/__main__.py','submit','candidate'];self.write(job)
  self.assertEqual(self.inspect()['code'],'NOT_PINNED_LOCAL_PYTHON_DRIVER')
  self.assertEqual(self.store.calls,[])
 def test_actual_pw43_example_with_synthetic_empty_snapshot_not_live_registration(self):
  root=Path(__file__).resolve().parents[2]
  relative='proj_rsi/scheduling/jobs/pw43-stack-01.json'
  if not (root/relative).exists():self.skipTest('optional repository example absent')
  descriptor=json.loads((root/relative).read_text());policy=deepcopy(self.policy)
  policy['allowed_ids']=[descriptor['id']]
  source_path='proj_rsi/scheduling/reviews/pw43-source-01.md'
  policy['source_review_bindings']={descriptor['id']:{'actor':'sol08','decision':'GO','path':source_path,'sha256':descriptor['files'][source_path]}}
  decision=inspect_descriptor(root,relative,{'version':1,'sequence':0,'jobs':{}},self.store.policy,policy)
  self.assertEqual(decision['code'],'APPROVED')
  self.assertEqual(decision['registration']['state'],'BACKLOG')
  self.assertAlmostEqual(decision['roi_summary']['cost_usd'],600*5/3600)
  self.assertEqual(self.store.calls,[])
 def test_skip_reason_is_hash_bound_and_contains_no_exception_payload(self):
  self.job['review']['sha256']='a'*64;self.write()
  result=self.inspect();self.assertEqual(result['action'],'SKIP')
  self.assertEqual(result['evidence']['descriptor_sha256'],sha(self.root/self.relative))
  self.assertEqual(result['reason_sha256'],hashlib.sha256(json.dumps(result['evidence'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())

if __name__=='__main__':unittest.main()
