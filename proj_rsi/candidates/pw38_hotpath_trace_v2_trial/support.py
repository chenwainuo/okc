"""Pure source/counter/command bindings. No production import or CUDA at import."""
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PW38_SHA='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
TOP4=[('decode1-pooler','pooler',1),('verify1-pooler','pooler',8),('decode16-pooler','pooler',16),('decode16-unpooler','unpooler',16)]
LOADS={'decode1':1,'decode16':16,'decode32':32,'verify1':8,'verify16':128,'verify32':256}
LAYERS=('qkv_gate','o','pooler','unpooler','router')
def case_scope(scope):
 assert scope in ('top4','all30')
 return list(TOP4) if scope=='top4' else [(load+'-'+layer,layer,m) for load,m in LOADS.items() for layer in LAYERS]
CASES=case_scope('all30')
BRIEFS=['pw38.next.p1-late-ticket','pw38.next.gv-empty-escape-specialization','pw38.next.lt256-heldout-margin','pw38.next.eh-private-output-slabs','pw38.next.o32-half-fullk-early']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 return h.hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stack():return module(ROOT/'proj_rsi/tools/pw45_stack_lab.py','pw38_profile_counter_contract')
def verify():
 m=json.loads((HERE/'manifest.json').read_text())
 for rel,pin in m['files'].items():assert sha(ROOT/rel)==pin,('source/dependency drift',rel)
 assert sha(ROOT/'proj_rsi/candidates/pw38/submission.py')==PW38_SHA
 return m

def merge_plans(plans):
 merged={}
 for plan in plans:
  for key,rule in plan.items():
   mode=rule['mode'];assert mode in ('progress','reset0')
   if key not in merged:merged[key]={'mode':mode,'offset':0,'changes':{},'reset_indices':[]}
   assert merged[key]['mode']==mode,'conflicting active protocols share physical storage'
   if mode=='reset0':merged[key]['reset_indices']=sorted(set(merged[key]['reset_indices'])|{rule['offset']+i for i in (rule.get('reset_indices') or [])})
   if mode=='progress':
    for index,delta in rule['changes'].items():
     absolute=rule['offset']+index;merged[key]['changes'][absolute]=merged[key]['changes'].get(absolute,0)+delta
 return merged

def serial_counters(arenas):
 return [{'storage':list(key),'name':v['name'],'values':v['values'],'digest':hashlib.sha256(json.dumps(v['values'],separators=(',',':')).encode()).hexdigest()}for key,v in arenas.items()]
def serial_plan(plan):return [{'storage':list(key),**rule}for key,rule in plan.items()]


P45=module(ROOT/'proj_rsi/candidates/pw45_trial/support.py','frozen_pw45_trace_support')

def weighted_plan(plans,counts):
 scaled=[]
 assert set(plans)==set(counts)
 for name,plan in plans.items():
  calls=counts[name];assert type(calls)is int and calls>=0
  if calls==0:continue
  scaled.append({key:{**rule,'changes':{i:delta*calls for i,delta in rule.get('changes',{}).items()}} for key,rule in plan.items()})
 return merge_plans(scaled)

def parse_trace(trace,nodes,replays):
 import math,collections,re
 assert replays==3 and isinstance(trace,dict) and isinstance(trace.get('traceEvents'),list)
 events=trace['traceEvents']
 def number(v):return type(v)in(int,float) and math.isfinite(v)
 def complete(e):return e.get('ph')=='X' and number(e.get('ts')) and number(e.get('dur')) and e['dur']>0
 scopes=[]
 for i in range(replays):
  found=[e for e in events if e.get('name')=='PW38_TRACE_REPLAY/'+str(i) and e.get('ph')=='X' and e.get('cat')=='user_annotation']
  assert len(found)==1 and complete(found[0]),('missing/duplicate/invalid replay scope',i)
  scopes.append(found[0])
 assert all(scopes[i]['ts']+scopes[i]['dur']<=scopes[i+1]['ts'] for i in range(replays-1)),'overlapping or reordered replay scopes'
 setup=[e for e in events if e.get('name')=='PW38_TRACE_SETUP' and e.get('ph')=='X' and e.get('cat')=='user_annotation'];assert len(setup)==1 and complete(setup[0])
 normalize=lambda s:re.sub(r'\s+','',s)
 kernel_nodes=[n for n in nodes if n['node_type']==0];assert kernel_nodes
 expected=collections.Counter(normalize(n['demangled']) for n in kernel_nodes)
 def within(e,scope):return e['ts']>=scope['ts'] and e['ts']+e['dur']<=scope['ts']+scope['dur']
 gpu=[];runtime=[]
 for e in events:
  category=e.get('cat','')
  if category in ('kernel','gpu_memcpy','gpu_memset'):
   assert complete(e) and isinstance(e.get('name'),str) and e['name'],('invalid GPU event',e)
   places=[i for i,s in enumerate(scopes) if within(e,s)]
   if not places:
    assert within(e,setup[0]),('unattributed GPU work outside setup/replays',e)
    where='setup'
   else:assert len(places)==1;where=places[0]
   gpu.append({'name':e['name'],'kind':category,'start_us':e['ts'],'duration_us':e['dur'],'replay':where,'args':e.get('args',{})})
  elif category=='cuda_runtime' and complete(e):runtime.append({'name':e.get('name'),'start_us':e['ts'],'duration_us':e['dur'],'kind':'host_cuda_runtime_inclusive'})
 per=[]
 for i,scope in enumerate(scopes):
  rows=[e for e in gpu if e['replay']==i];kernels=[e for e in rows if e['kind']=='kernel']
  assert collections.Counter(normalize(e['name']) for e in kernels)==expected,('actual replay kernel identities/count differ from CUDA graph metadata',i)
  per.append({'replay':i,'host_replay_plus_sync_start_us':scope['ts'],'host_replay_plus_sync_inclusive_us':scope['dur'],
   'kernel_calls':len(kernels),'kernel_duration_sum_us':sum(e['duration_us'] for e in kernels),
   'kernel_first_to_last_span_us':max(e['start_us']+e['duration_us'] for e in kernels)-min(e['start_us'] for e in kernels),
   'GPU_events':rows})
 totals=[]
 for symbol in sorted(expected):
  rows=[e for e in gpu if type(e['replay'])is int and e['kind']=='kernel' and normalize(e['name'])==symbol]
  totals.append({'symbol':rows[0]['name'],'calls':len(rows),'calls_per_replay':len(rows)/replays,'CUDA_duration_sum_us':sum(e['duration_us'] for e in rows),'CUDA_duration_mean_us':sum(e['duration_us'] for e in rows)/len(rows)})
 return {'profile_replays':replays,'graph_nodes':nodes,'per_replay':per,'kernels':totals,'setup_GPU_events':[e for e in gpu if e['replay']=='setup'],'host_runtime_events':runtime,'gpu_correlated_annotations':[e for e in events if e.get('cat')=='gpu_user_annotation'],
  'units':'Chrome ts/dur microseconds; CPU replay scopes include synchronization/wait; CUDA duration sums can overlap and are not graph elapsed or benchmark/scored latency'}
