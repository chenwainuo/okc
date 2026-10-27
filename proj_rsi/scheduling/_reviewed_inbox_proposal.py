"""UNACTIVATED root-worker reviewed GPU inbox proposal; no CLI or polling hook.

Import and inspect_descriptor are read-only. sync_reviewed_new is disabled unless
both caller and root policy opt in; tests use FakeStore and temporary files only.
It registers new BACKLOG descriptors, never claims/launches/updates/retries jobs.
"""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os,stat
from pathlib import Path
import re

from .dispatch import GPUDispatcher
from .model import score,validate_job,ncu_target

ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
SHA=re.compile(r'[a-f0-9]{64}\Z')
VALID_DONE={'CPU_PASS','VALID_GAIN','VALID_NO_GAIN','VALID_REGRESSION'}


def _hash(value):return hashlib.sha256(value).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _path(root,relative):
 if not isinstance(relative,str) or not relative or relative.startswith('-') or '\\' in relative or any(ord(c)<32 for c in relative):raise ValueError('unsafe path')
 rel=Path(relative)
 if rel.is_absolute() or any(p in ('','.','..') for p in relative.split('/')):raise ValueError('unsafe path')
 path=root/rel
 if path.resolve()!=path or not path.is_relative_to(root):raise ValueError('symlink or escape')
 return path

def _pairs(pairs):
 result={}
 for key,value in pairs:
  if key in result:raise ValueError('duplicate JSON key')
  result[key]=value
 return result

def _parse(raw):
 if len(raw)>1_000_000:raise ValueError('oversized descriptor')
 return json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

class _FrozenView:
 def __init__(self,snapshot,policy):self.data=deepcopy(snapshot);self.policy=deepcopy(policy)
 def snapshot(self):return deepcopy(self.data)

class _NoTransport:
 def __getattr__(self,name):raise AssertionError('proposal must never call transport: '+name)


def _decision(code,path,digest,job_id=None,*,policy=None,sequence=None,pins=None,state=None):
 try:policy_hash=_hash(_canonical(policy)) if policy is not None else None
 except (TypeError,ValueError,OverflowError):policy_hash=None
 sequence=sequence if type(sequence) is int else None
 state=state if isinstance(state,str) or state is None else 'UNKNOWN_MALFORMED'
 path=path if isinstance(path,str) or path is None else None
 evidence={'code':code,'descriptor_path':path,'descriptor_sha256':digest,'id':job_id,'policy_sha256':policy_hash,'snapshot_sequence':sequence,'pins':pins or {},'existing_state':state}
 return {'action':'APPROVED_PROPOSAL' if code=='APPROVED' else 'SKIP','code':code,'evidence':evidence,'reason_sha256':_hash(_canonical(evidence))}


def _policy_ok(policy,scheduler_policy,controller):
 if not isinstance(policy,dict) or not isinstance(scheduler_policy,dict) or policy.get('enabled') is not True:return False
 try:_canonical(policy)
 except (TypeError,ValueError,OverflowError):return False
 if policy.get('controller_id')!=controller or controller!='root-worker':return False
 if scheduler_policy.get('gpu_usd_per_hour')!=5:return False
 ids=policy.get('allowed_ids',[]);prefixes=policy.get('allowed_prefixes',[])
 if not isinstance(ids,list) or not isinstance(prefixes,list) or not ids and not prefixes:return False
 if any(not isinstance(x,str) or not ID.fullmatch(x) for x in ids+prefixes):return False
 if any(len(x)<3 for x in prefixes):return False
 if not isinstance(policy.get('bounds'),dict) or not isinstance(scheduler_policy.get('launch_limits',{}),dict):return False
 if not isinstance(policy.get('source_review_bindings',{}),dict):return False
 for mode in ('normal','profile'):
  bound=policy['bounds'].get(mode,{})
  if not isinstance(bound,dict):return False
  inner,outer=bound.get('inner_seconds'),bound.get('outer_seconds')
  if type(inner) is not int or type(outer) is not int or not 0<inner<outer-20:return False
  max_inner=scheduler_policy.get('launch_limits',{}).get('max_inner_seconds',3600);max_outer=scheduler_policy.get('launch_limits',{}).get('max_outer_seconds',3720)
  if type(max_inner) is not int or type(max_outer) is not int or inner>max_inner or outer>max_outer:return False
 return True


def _dependencies(job,snapshot):
 parents=snapshot['jobs']
 for dep in job.get('depends_on',[]):
  if isinstance(dep,str):name=dep;accepted=None
  elif isinstance(dep,dict):
   name=dep.get('id');accepted=dep.get('accepted_dispositions',dep.get('dispositions'))
   if 'accepted_dispositions' in dep and 'dispositions' in dep and dep['accepted_dispositions']!=dep['dispositions']:return False
   if accepted is not None and (not isinstance(accepted,list) or not accepted or any(x not in VALID_DONE for x in accepted)):return False
  else:return False
  parent=parents.get(name)
  if not isinstance(parent,dict) or parent.get('state')!='DONE' or parent.get('disposition') not in VALID_DONE:return False
  if accepted is not None and parent['disposition'] not in accepted:return False
 return True


def _source_review(job,policy,root,pins):
 annotations=job.get('annotations',{})
 if not isinstance(annotations,dict):return False
 if 'source_review' not in annotations:return True
 annotation=annotations['source_review']
 receipt=annotation if isinstance(annotation,dict) else policy.get('source_review_bindings',{}).get(job['id'])
 if not isinstance(receipt,dict) or receipt.get('decision')!='GO' or not isinstance(receipt.get('actor'),str) or not receipt['actor'].strip() or receipt['actor']==job.get('author'):return False
 path,digest=receipt.get('path'),receipt.get('sha256')
 if path==job['review']['path'] or not isinstance(digest,str) or not SHA.fullmatch(digest):return False
 if job['files'].get(path)!=digest:return False
 if _hash(_path(root,path).read_bytes())!=digest:return False
 pins[path]=digest
 return True


def inspect_descriptor(repo_root,descriptor_path,snapshot,scheduler_policy,policy,*,controller='root-worker'):
 """Pure eligibility proposal over a supplied snapshot; never reads real store.

 Legacy free-text source_review is not parsed/guessed: root must supply a full
 job-bound source_review_bindings receipt, or descriptor must carry one.
 """
 root=Path(repo_root).resolve();digest=None;job_id=None;sequence=snapshot.get('sequence') if isinstance(snapshot,dict) else None
 def verdict(code,**kw):return _decision(code,descriptor_path,digest,job_id,policy=policy,sequence=sequence,**kw)
 try:
  if not _policy_ok(policy,scheduler_policy,controller):return verdict('DISABLED_OR_POLICY_INVALID')
  if not isinstance(snapshot,dict) or not isinstance(snapshot.get('jobs'),dict):return verdict('INVALID_STORE_SNAPSHOT')
  path=_path(root,descriptor_path)
  if not descriptor_path.startswith('proj_rsi/scheduling/jobs/') or path.suffix!='.json':return verdict('OUTSIDE_DESCRIPTOR_INBOX')
  try:raw=path.read_bytes();digest=_hash(raw);job=_parse(raw)
  except (OSError,ValueError,UnicodeError):return verdict('PARTIAL_MISSING_OR_INVALID_JSON')
  if not isinstance(job,dict) or not isinstance(job.get('id'),str) or not ID.fullmatch(job['id']):return verdict('INVALID_DESCRIPTOR_ID')
  job_id=job['id']
  if job_id in snapshot['jobs']:
   existing=snapshot['jobs'][job_id]
   return verdict('EXISTING_ID_NEVER_MUTATE',state=existing.get('state') if isinstance(existing,dict) else 'UNKNOWN_MALFORMED')
  if job_id not in policy['allowed_ids'] and not any(job_id.startswith(p) for p in policy['allowed_prefixes']):return verdict('ID_NOT_ROOT_ALLOWLISTED')
  if job.get('resource')!='gpu' or job.get('kind') not in ('optimization','diagnostic','profile'):return verdict('NOT_LOCAL_GPU_WORK')
  if job.get('state')!='BACKLOG':return verdict('NEW_DESCRIPTOR_MUST_BE_BACKLOG')
  if validate_job(job,for_admission=True):return verdict('DESCRIPTOR_ADMISSION_INVALID')
  roi=score(job,policy=scheduler_policy)
  if roi.get('rankable') is not True or roi.get('priority') is None:return verdict('UNKNOWN_OR_INVALID_ROI')
  if not _dependencies(job,snapshot):return verdict('DEPENDENCY_NOT_DONE_VALID')
  launch=job['launch'];mode='profile' if launch.get('profile') else 'normal';bounds=policy['bounds'][mode]
  if launch['inner_seconds']>bounds['inner_seconds'] or launch['outer_seconds']>bounds['outer_seconds']:return verdict('ROOT_WORKER_BOUND_EXCEEDED')
  argv=launch['argv'];python=('python3','python','/usr/bin/python3')
  if argv[0] in python:
   if len(argv)<2 or not argv[1].startswith('/okc/proj_rsi/tools/') or not argv[1].endswith('.py'):return verdict('NOT_PINNED_LOCAL_PYTHON_DRIVER')
   if argv[1][len('/okc/'):] not in job['files']:return verdict('LOCAL_DRIVER_NOT_PINNED')
  elif argv[0] in ('ncu','/usr/local/cuda/bin/ncu'):
   target=ncu_target(argv)
   if target is None or target+1>=len(argv) or not argv[target+1].startswith('/okc/proj_rsi/tools/') or not argv[target+1].endswith('.py'):return verdict('NOT_PINNED_LOCAL_NCU_DRIVER')
  else:return verdict('NOT_LOCAL_GPU_WORK')
  # Validate using production pure _validate with a frozen in-memory view. Do not
  # construct a real store/SSH transport or call launch_next/claim/any remote API.
  checker=object.__new__(GPUDispatcher);checker.repo_root=root;checker.store=_FrozenView(snapshot,scheduler_policy);checker.transport=_NoTransport()
  _,pins=GPUDispatcher._validate(checker,job)
  if not _source_review(job,policy,root,pins):return verdict('SOURCE_REVIEW_UNRESOLVED_OR_INVALID',pins=pins)
  if _hash(path.read_bytes())!=digest:return verdict('DESCRIPTOR_CHANGED_DURING_VALIDATION',pins=pins)
  result=verdict('APPROVED',pins=pins);result['registration']=deepcopy(job);result['registration']['state']='BACKLOG'
  result['roi_summary']={k:roi.get(k) for k in ('priority','direct_expected_us','information_expected_us','effective_value_us','cost_usd')}
  return result
 except (OSError,ValueError,TypeError,KeyError,OverflowError,AttributeError):return verdict('PIN_REVIEW_OR_DESCRIPTOR_VALIDATION_FAILED')


@contextmanager
def _admission_lock(root,path):
 """Separate future inbox-controller lock, never hardware or store's own .lock.
 Only reached after explicit opt-in. Tests use isolated temporary directories.
 """
 if path!='results/proj-rsi/scheduler/reviewed-inbox.proposal.lock':raise ValueError('dedicated inbox admission lock required')
 lock=_path(root,path)
 lock.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('lock is not regular file')
 with os.fdopen(fd,'a+b') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(stream,fcntl.LOCK_UN)


def sync_reviewed_new(store,repo_root,descriptor_paths,policy,*,root_opt_in=False,controller='root-worker',before_register=None):
 """UNACTIVATED future integration hook; only FakeStore is exercised in tests.

 Register BACKLOG exactly once via the store's atomic unique-ID operation.
 Store alone performs ROI rebalance; helper never claims/launches/transitions.
 Existing HELD/RUNNING/VALIDATING/DONE/CANCELLED IDs always remain untouched.
 """
 if not root_opt_in or not _policy_ok(policy,store.policy,controller):return [_decision('DISABLED_OR_POLICY_INVALID',None,None,policy=policy)]
 root=Path(repo_root).resolve();results=[]
 try:
  with _admission_lock(root,policy['admission_lock']):
   for path in sorted(set(descriptor_paths)):
    snapshot=store.snapshot();proposal=inspect_descriptor(root,path,snapshot,store.policy,policy,controller=controller)
    if proposal['code']!='APPROVED':results.append(proposal);continue
    if before_register is not None:before_register(proposal)
    fresh=inspect_descriptor(root,path,store.snapshot(),store.policy,policy,controller=controller)
    if fresh['code']!='APPROVED':results.append(fresh);continue
    if fresh['evidence']['descriptor_sha256']!=proposal['evidence']['descriptor_sha256'] or fresh['evidence']['pins']!=proposal['evidence']['pins']:
     results.append(_decision('CHANGED_SINCE_PROPOSAL',path,fresh['evidence']['descriptor_sha256'],fresh['evidence']['id'],policy=policy));continue
    try:
     registered=store.register(fresh['registration'])
     fresh.pop('registration');fresh['action']='REGISTERED_NEW';fresh['code']='REGISTERED_NEW_BACKLOG_DESCRIPTOR';fresh['store_result_state']=registered.get('state');results.append(fresh)
    except Exception:
     # The atomic store may reject a duplicate ID registered by another control
     # path after our fresh check. Never update it or retry under another ID.
     latest=store.snapshot();code='REGISTRATION_RACE_EXISTING_ID' if fresh['evidence']['id'] in latest['jobs'] else 'ATOMIC_REGISTRATION_REJECTED'
     results.append(_decision(code,path,fresh['evidence']['descriptor_sha256'],fresh['evidence']['id'],policy=policy,sequence=latest.get('sequence')))
 except (OSError,ValueError,TypeError,KeyError,AttributeError):results.append(_decision('ADMISSION_LOCK_OR_POLICY_FAILED',None,None,policy=policy))
 return results
