"""UNACTIVATED admissions-only root monitor. No store/action without --run.

Root starts a NEW process only after independent integration review. Existing
GPU worker/claim/dispatch stays authoritative. This monitor never launches work.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import time
from . import _reviewed_inbox_proposal as inbox

ROOT=Path(__file__).resolve().parents[2]
POLICY='proj_rsi/scheduling/reviewed_inbox_policy.proposal.json'
MANIFEST='proj_rsi/scheduling/reviewed_inbox_monitor.proposal-manifest.json'
REQUIRED_FILES={POLICY,'proj_rsi/scheduling/_reviewed_inbox_monitor_proposal.py','proj_rsi/scheduling/_reviewed_inbox_proposal.py','proj_rsi/scheduling/model.py','proj_rsi/scheduling/policy.json','proj_rsi/scheduling/dispatch.py','proj_rsi/scheduling/store.py','proj_rsi/audits/reviewed-inbox-independent.md'}
PREFIXES=['pw41.','pw42.','pw44.','pw45.','pw38.selected.profile.']


def _sha(data):return hashlib.sha256(data).hexdigest()
def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def validate_policy(policy):
 if not isinstance(policy,dict) or policy.get('enabled') is not True or policy.get('controller_id')!='root-worker':raise ValueError('explicit root policy required')
 if policy.get('gpu_usd_per_hour')!=5 or policy.get('allowed_ids')!=[] or policy.get('allowed_prefixes')!=PREFIXES:raise ValueError('root allowlist/price changed')
 if policy.get('bounds')!={'normal':{'inner_seconds':3600,'outer_seconds':3720},'profile':{'inner_seconds':600,'outer_seconds':720}}:raise ValueError('root deadline caps changed')
 if policy.get('admission_lock')!='results/proj-rsi/scheduler/reviewed-inbox.proposal.lock':raise ValueError('dedicated admission lock required')
 if policy.get('max_wall_seconds')!=7200 or policy.get('max_poll_seconds')!=30:raise ValueError('monitor bounds changed')
 if type(policy.get('default_poll_seconds')) is not int or not 1<=policy['default_poll_seconds']<=30:raise ValueError('invalid poll interval')
 if not isinstance(policy.get('source_review_bindings'),dict):raise ValueError('source receipt map required')
 _json(policy)
 return policy


def verify_bundle(root=ROOT,expected_sha256=None):
 root=Path(root).resolve();raw=(root/MANIFEST).read_bytes()
 if not isinstance(expected_sha256,str) or not inbox.SHA.fullmatch(expected_sha256) or _sha(raw)!=expected_sha256:raise ValueError('root-pinned manifest SHA required')
 manifest=inbox._parse(raw)
 if not isinstance(manifest,dict) or not isinstance(manifest.get('files'),dict) or set(manifest['files'])!=REQUIRED_FILES:raise ValueError('incomplete reviewed bundle')
 for relative,digest in manifest['files'].items():
  if not isinstance(digest,str) or not inbox.SHA.fullmatch(digest):raise ValueError('invalid bundle SHA')
  path=inbox._path(root,relative)
  if _sha(path.read_bytes())!=digest:raise ValueError('reviewed monitor bundle drift')
 return validate_policy(inbox._parse((root/POLICY).read_bytes())),manifest


def compact(receipt,now):
 evidence=receipt.get('evidence',{})
 return {'observed_at':now,'id':evidence.get('id'),'action':receipt.get('action'),'code':receipt.get('code'),
  'descriptor_sha256':evidence.get('descriptor_sha256'),'policy_sha256':evidence.get('policy_sha256'),
  'reason_sha256':receipt.get('reason_sha256'),'closure_sha256':_sha(_json(evidence.get('pins',{})).encode()),
  'existing_state':evidence.get('existing_state'),'store_result_state':receipt.get('store_result_state')}


def _signature(row):
 # Sequence-only WAL heartbeat changes do not spam identical skip decisions.
 return _json({k:row.get(k) for k in ('id','action','code','descriptor_sha256','policy_sha256','closure_sha256','existing_state','store_result_state')})


def run_monitor(store_factory,root,policy,*,enabled=False,poll_seconds=15,max_wall_seconds=7200,once=False,clock=time.monotonic,sleep=time.sleep,wall=time.time,emit=lambda row:None):
 """Inject FakeStore/clock for tests; real store factory is lazy and root-only.
 Exactly one discovery snapshot per cycle; existing IDs never call sync/get/rank.
 """
 if enabled is not True:return {'status':'DISABLED_NO_STORE_CREATED','cycles':0,'registrations':0}
 validate_policy(policy)
 if type(poll_seconds) is not int or not 1<=poll_seconds<=30:raise ValueError('poll must be1..30seconds')
 if type(max_wall_seconds) is not int or not 1<=max_wall_seconds<=7200:raise ValueError('wall bound must be1..7200seconds')
 root=Path(root).resolve();last=None
 def tick():
  nonlocal last
  value=clock()
  if type(value) not in (int,float) or not math.isfinite(value) or (last is not None and value<last):raise ValueError('invalid monotonic clock')
  last=value;return value
 start=tick();deadline=start+max_wall_seconds
 store=store_factory();seen={};cycles=registrations=0;directory=root/'proj_rsi/scheduling/jobs'
 def report(key,receipt):
  row=compact(receipt,wall());signature=_signature(row)
  if seen.get(key)!=signature:emit(row);seen[key]=signature
 while tick()<deadline:
  begun=tick();snapshot=store.snapshot();cycles+=1
  jobs=snapshot.get('jobs') if isinstance(snapshot,dict) else None
  if not isinstance(jobs,dict):raise ValueError('malformed authoritative snapshot')
  for path in sorted(directory.glob('*.json')):
   if tick()>=deadline:break
   relative=str(path.relative_to(root));digest=None;job_id=None
   try:
    if path.resolve()!=path or not path.is_file():raise ValueError('unsafe descriptor path')
    if path.stat().st_size>1_000_000:raise ValueError('oversized descriptor')
    raw=path.read_bytes();digest=_sha(raw);job=inbox._parse(raw)
    if not isinstance(job,dict) or not isinstance(job.get('id'),str) or not inbox.ID.fullmatch(job['id']):raise ValueError('invalid descriptor id')
    job_id=job['id']
   except (OSError,ValueError,UnicodeError):
    report(relative,inbox._decision('SCAN_PARTIAL_MISSING_OR_INVALID',relative,digest,policy=policy));continue
   if job_id in jobs:
    other=jobs[job_id];state=other.get('state') if isinstance(other,dict) else 'UNKNOWN_MALFORMED'
    report(relative,inbox._decision('EXISTING_ID_NEVER_MUTATE',relative,digest,job_id,policy=policy,sequence=snapshot.get('sequence'),state=state));continue
   if not any(job_id.startswith(prefix) for prefix in PREFIXES):
    report(relative,inbox._decision('ID_NOT_ROOT_ALLOWLISTED',relative,digest,job_id,policy=policy));continue
   if job.get('resource')!='gpu':
    report(relative,inbox._decision('NOT_LOCAL_GPU_WORK',relative,digest,job_id,policy=policy));continue
   # New root-allowlisted candidates alone reach the reviewed helper, which
   # refreshes closure/dependencies/ID under its own dedicated admission lock.
   decisions=inbox.sync_reviewed_new(store,root,[relative],policy,root_opt_in=True,controller='root-worker')
   for decision in decisions:
    report(relative,decision)
    if decision.get('action')=='REGISTERED_NEW':
     registrations+=1;jobs[job_id]={'id':job_id,'state':decision.get('store_result_state')}
  if once or tick()>=deadline:break
  remaining=deadline-tick();delay=min(max(0,poll_seconds-(tick()-begun)),remaining)
  if delay>0:sleep(delay)
 return {'status':'BOUNDED_MONITOR_EXIT','cycles':cycles,'registrations':registrations,'elapsed_seconds':tick()-start,
  'mode':'admissions only; no claim/launch/retry/update','max_wall_seconds':max_wall_seconds}


def append_compact(path,row):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.is_symlink():raise ValueError('log symlink rejected')
 fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'a') as stream:stream.write(_json(row)+'\n');stream.flush();os.fsync(stream.fileno())


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--manifest-sha256');parser.add_argument('--once',action='store_true');parser.add_argument('--poll-seconds',type=int,default=15);parser.add_argument('--max-wall-seconds',type=int,default=7200);parser.add_argument('--store-directory',default='results/proj-rsi/scheduler');parser.add_argument('--decision-log',default='results/proj-rsi/inbox-monitor/decisions.jsonl');args=parser.parse_args(argv)
 if not args.run:
  print(_json({'status':'DISABLED_NO_STORE_CREATED','activation':'independent integration GO and root --run required'}));return
 policy,manifest=verify_bundle(ROOT,args.manifest_sha256)
 store_dir=inbox._path(ROOT,args.store_directory);log=inbox._path(ROOT,args.decision_log)
 def factory():
  # Fresh process import sees whichever existing store root has approved and
  # installed (including reviewed cache after root activation). No live patch.
  from .store import QueueStore
  return QueueStore(store_dir,repo_root=ROOT)
 result=run_monitor(factory,ROOT,policy,enabled=True,poll_seconds=args.poll_seconds,max_wall_seconds=args.max_wall_seconds,once=args.once,emit=lambda row:append_compact(log,row))
 print(_json(result))
if __name__=='__main__':main()
