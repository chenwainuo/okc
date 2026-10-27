"""Read-only, sanitized dashboard snapshots; never executes or repairs jobs.

python3 -m proj_rsi.dashboard.snapshot --output results/proj-rsi/dashboard/snapshot.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
STATES=('BACKLOG','READY','RUNNING','VALIDATING','DONE','HELD','CANCELLED')
STALE_SECONDS=120


def finite(value):
 try:return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
 except (TypeError,OverflowError):return False


def safe_text(value,limit=1800):
 if not isinstance(value,str):return None
 value=''.join(c if ord(c)>=32 else ' ' for c in value)[:limit]
 value=re.sub(r'https?://\S+|[\w.+-]+@(?:[\w.-]+)|\b(?:\d{1,3}\.){3}\d{1,3}\b','[redacted address]',value)
 value=re.sub(r'''(?i)\b(?:api[_-]?key|[a-z0-9_-]*token|secret|password|authorization)\b["']?\s*[:=]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|(?:Bearer\s+)?[^\s"',}\]]+)|\bBearer\s+[^\s"',}\]]+''','[redacted credential]',value)
 value=re.sub(r'(?i)(?<!\w)(?:ssh|curl|docker|bash|ncu)\s+\S+.*|(?<!\w)python3?\s+(?:[-/]|[\w./]+\.py\b).*','[redacted command]',value)
 value=re.sub(r'''(?<![\w:])/(?!/)[^\s<>"'`]+|\b[A-Za-z]:\\[^\s<>"'`]+''','[redacted path]',value)
 return value


def strings(values):return [v for x in values if (v:=safe_text(x)) is not None] if isinstance(values,list) else []

def timestamp(value):
 if finite(value):return float(value)
 if isinstance(value,str):
  try:return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
  except (ValueError,OverflowError):return None
 return None


def freshness(source,observed_at,now,status='ok'):
 observed=timestamp(observed_at)
 age=max(0.,now-observed) if observed is not None else None
 if observed is not None and observed > now+30:
  return {'source':source,'status':'future_observation','observed_at':observed,'age_seconds':None,'stale':None}
 return {'source':source,'status':status,'observed_at':observed,'age_seconds':age,'stale':None if age is None else age>STALE_SECONDS}


def rows(value):return value if isinstance(value,list) else []

def optional_json(path):
 try:return json.loads(path.read_text()),'ok'
 except FileNotFoundError:return None,'missing'
 except (OSError,ValueError):return None,'unavailable'


def read_scheduler(directory):
 """Replay checksummed authoritative WAL under existing lock, with zero writes.

 QueueStore.snapshot() creates/replaces checkpoints and repairs torn tails. This
 viewer uses the same envelope/checksum/sequence rules but refuses to repair.
 It does not construct QueueStore, create its lock, rebalance, or change leases.
 """
 directory=Path(directory);lock=directory/'.lock';wal=directory/'events.jsonl'
 if not lock.exists() or not wal.exists():return None,None,'missing'
 try:
  with lock.open('rb') as handle:
   try:fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
   except BlockingIOError:return None,None,'busy'
   try:raw=wal.read_bytes()
   finally:fcntl.flock(handle,fcntl.LOCK_UN)
 except OSError:return None,None,'unavailable'
 if raw and not raw.endswith(b'\n'):return None,None,'torn_wal_no_repair'
 state={'version':1,'sequence':0,'jobs':{}};last=None
 for line in raw.splitlines():
  try:
   entry=json.loads(line);payload=entry['payload']
   canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
   if hashlib.sha256(canonical).hexdigest()!=entry['sha256']:raise ValueError('checksum')
   if payload['sequence']!=state['sequence']+1:raise ValueError('sequence')
   candidate=payload['snapshot']
   if candidate['version']!=1 or candidate['sequence']!=payload['sequence'] or not isinstance(candidate['jobs'],dict):raise ValueError('envelope')
   state=candidate;last=timestamp(payload.get('timestamp'))
  except (ValueError,KeyError,TypeError,OverflowError):return None,None,'corrupt_wal_no_repair'
 return state,last,'ok'


def summarize_job(job,now):
 from proj_rsi.scheduling.model import score,validate_job
 if not isinstance(job,dict):return None
 resource=job.get('resource') if job.get('resource') in ('cpu','gpu') else 'unknown'
 state=job.get('state') if job.get('state') in STATES else 'UNKNOWN'
 try:
  result=score(job,now=now)
  errors=validate_job(job,for_admission=True)
 except (ValueError,KeyError,TypeError,OverflowError):result={};errors=['unavailable']
 review=job.get('review') if isinstance(job.get('review'),dict) else {}
 outcome=job.get('disposition')
 # Outcome is the recorded disposition, never inferred from subprocess exit.
 observations=job.get('observations',[]) if isinstance(job.get('observations'),list) else []
 recorded_status=None
 for observation in reversed(observations):
  if isinstance(observation,dict) and isinstance(observation.get('validation'),dict):
   recorded_status=safe_text(observation['validation'].get('disposition'));break
 roi={key:result.get(key) if finite(result.get(key)) else None for key in ('priority','direct_expected_us','information_expected_us','effective_value_us','cost_usd','cpu_minutes','age_multiplier')}
 roi['rankable']=result.get('rankable') is True
 return {'id':safe_text(job.get('id'),160),'resource':resource,'kind':safe_text(job.get('kind'),80),'state':state,'author':safe_text(job.get('author'),100),
  'created_at':timestamp(job.get('created_at')),'updated_at':timestamp(job.get('updated_at')),
  'review':{'decision':safe_text(review.get('decision'),40),'independent':bool(review.get('actor') and review.get('actor')!=job.get('author'))},
  'readiness':{'descriptor_valid':not errors,'reasons_count':len(errors),'note':'Descriptor checks only; file hashes and hardware admission are separate executor gates.'},
  'roi':roi,'disposition':safe_text(outcome,80),'outcome':{'status':recorded_status or safe_text(outcome,80)}}


def summarize_official(row):
 return {'job_id':safe_text(row.get('job_id'),80),'submitted_utc':safe_text(row.get('submitted_utc'),60),'candidate':safe_text(row.get('candidate'),100),
  'score_us':row.get('score_us') if finite(row.get('score_us')) else None,'passed':row.get('passed') if isinstance(row.get('passed'),bool) else None,
  'stack_description':safe_text(row.get('stack_description')),'compiled_features':strings(row.get('compiled_or_available_features')),'selected_features':strings(row.get('actual_selected_features')),
  'archive_status':safe_text(row.get('archive_status'),80)}


def build_snapshot(repo_root=ROOT,*,now=None,scheduler_dir=None,agents_path=None,hardware_path=None):
 root=Path(repo_root);now=time.time() if now is None else now
 if not finite(now):raise ValueError('now must be finite epoch seconds')
 sources={};history_path=root/'proj_rsi/history_catalog.json';history,hstatus=optional_json(history_path)
 if not isinstance(history,dict):
  hstatus='invalid_shape' if hstatus=='ok' else hstatus;history={}
 sources['history']=freshness('canonical history catalog',history_path.stat().st_mtime if history_path.exists() else None,now,hstatus)
 state,last,qstatus=read_scheduler(scheduler_dir or root/'results/proj-rsi/scheduler')
 sources['scheduler']=freshness('authoritative checksummed scheduler WAL',now if state is not None else None,now,qstatus)
 sources['scheduler']['last_event_at']=last
 jobs=[];counts={r:{s:0 for s in STATES} for r in ('cpu','gpu')}
 if state:
  for job in state['jobs'].values():
   row=summarize_job(job,now)
   if row is None:continue
   jobs.append(row)
   if row['resource'] in counts and row['state'] in STATES:counts[row['resource']][row['state']]+=1
 if state is None:
  counts={resource:{state:None for state in STATES} for resource in ('cpu','gpu')}
 jobs.sort(key=lambda r:(r['resource'],r['roi']['priority'] is None,-(r['roi']['priority'] or 0),r['created_at'] or 0,r['id'] or ''))
 apath=Path(agents_path) if agents_path else root/'results/proj-rsi/dashboard/agents.json'
 agent_data,astatus=optional_json(apath)
 if not isinstance(agent_data,dict):
  astatus='invalid_shape' if astatus=='ok' else astatus;agent_data={}
 agents_observed=agent_data.get('observed_at');sources['agents']=freshness('root collaboration agent observation',agents_observed,now,astatus)
 agents=[]
 for item in agent_data.get('agents',[]) if isinstance(agent_data.get('agents'),list) else []:
  if not isinstance(item,dict):continue
  status=item.get('agent_status',item.get('status'))
  if isinstance(status,dict):status='completed' if 'completed' in status else ('errored' if 'errored' in status else 'unknown')
  if status not in ('running','idle','completed','errored','interrupted','pending','shutdown'):status='unknown'
  observed=item.get('observed_at',agents_observed);fresh=freshness('root collaboration observation',observed,now)
  raw_name=item.get('agent_name',item.get('name'))
  if isinstance(raw_name,str) and raw_name.startswith('/root'):raw_name=raw_name.lstrip('/')
  agents.append({'name':safe_text(raw_name,180),'status':status,'observed_at':fresh['observed_at'],'age_seconds':fresh['age_seconds'],'stale':fresh['stale']})
 hpath=Path(hardware_path) if hardware_path else root/'results/proj-rsi/dashboard/hardware.json'
 hardware,hstat=optional_json(hpath)
 if not isinstance(hardware,dict):
  hstat='invalid_shape' if hstat=='ok' else hstat;hardware={}
 sources['hardware']=freshness('optional root hardware observation',hardware.get('observed_at'),now,hstat)
 gpu_status=hardware.get('status') if hardware.get('status') in ('idle','busy','unknown','unavailable') else 'unknown'
 authority=history.get('authority',{}) if isinstance(history.get('authority'),dict) else {}
 findings=[]
 for row in rows(history.get('improvements')):
  if not isinstance(row,dict):continue
  findings.append({'id':safe_text(row.get('id')),'classification':safe_text(row.get('classification')),'current_inclusion':safe_text(row.get('currently_included',row.get('in_rs32'))),
   'scope':safe_text(row.get('scope')),'finding':safe_text(row.get('finding')),'next_action':safe_text(row.get('next_action')),'stacked_versions':strings(row.get('stacked_versions')),'score_tested_jobs':strings(row.get('score_tested_jobs'))})
 candidates=[]
 for row in rows(history.get('candidates')):
  if not isinstance(row,dict):continue
  candidates.append({'id':safe_text(row.get('candidate')),'parent':safe_text(row.get('parent')),'status':safe_text(row.get('status')) or 'No runtime status recorded in catalog',
   'description':safe_text(row.get('description')),'compiled_features':strings(row.get('compiled_or_available_features')),'leaderboard_submitted':row.get('leaderboard_submitted') is True,'official_jobs':strings(row.get('official_jobs'))})
 submissions=[summarize_official(row) for row in rows(history.get('submissions')) if isinstance(row,dict)]
 checks=[summarize_official(row) for row in rows(history.get('checks')) if isinstance(row,dict)]
 return {'schema_version':1,'generated_at':float(now),'sources':sources,
  'summary':{'best_score_us':authority.get('score_us') if finite(authority.get('score_us')) else None,'best_job_id':safe_text(authority.get('best_job'),80),
   'submissions':len(submissions) if hstatus=='ok' else None,'checks':len(checks) if hstatus=='ok' else None,'findings':len(findings) if hstatus=='ok' else None,'candidates':len(candidates) if hstatus=='ok' else None,'cpu_capacity':10,'gpu_capacity':1},
  'agents':{'items':agents,'capacity':10,'observed_count':len(agents) if sources['agents']['observed_at'] is not None else None},
  'hardware':{'status':gpu_status,'gpu_name':safe_text(hardware.get('gpu_name'),100),'memory_used_mb':hardware.get('memory_used_mb') if finite(hardware.get('memory_used_mb')) else None,'process_count':hardware.get('process_count') if isinstance(hardware.get('process_count'),int) and not isinstance(hardware.get('process_count'),bool) else None},
  'queue':{'counts':counts,'jobs':jobs,'sequence':state.get('sequence') if state else None,'state_source':'Recorded scheduler states; not inferred process liveness.'},
  'submissions':submissions,'checks':checks,'findings':findings,'candidates':candidates}


def write_snapshot(output,snapshot):
 path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
 encoded=json.dumps(snapshot,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
 fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as handle:handle.write(encoded);handle.flush();os.fsync(handle.fileno())
  os.replace(temporary,path)
 finally:
  if os.path.exists(temporary):os.unlink(temporary)


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');args=parser.parse_args(argv)
 snapshot=build_snapshot()
 if args.output:write_snapshot(args.output,snapshot)
 else:print(json.dumps(snapshot,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
