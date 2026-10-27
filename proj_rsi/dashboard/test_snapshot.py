"""CPU-only isolation, redaction, provenance and freshness checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .snapshot import build_snapshot,read_scheduler,safe_text,write_snapshot

def wal(directory,jobs,*,torn=False,checksum=True):
 directory.mkdir(parents=True,exist_ok=True);(directory/'.lock').touch()
 state={'version':1,'sequence':1,'jobs':jobs}
 payload={'sequence':1,'timestamp':900.,'action':'registration','actor':'root','reason':'PRIVATE LOG','data':{'secret':'DO_NOT_EXPORT'},'snapshot':state}
 canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 entry={'payload':payload,'sha256':hashlib.sha256(canonical).hexdigest() if checksum else '0'*64}
 (directory/'events.jsonl').write_text(json.dumps(entry)+'\n'+('TORN_SECRET' if torn else ''))
 return state

def job():
 return {'id':'profile.v1','resource':'gpu','kind':'diagnostic','state':'RUNNING','author':'worker','created_at':900.,'depends_on':[],'write_paths':[],
  'estimate':{'direct':{'scenario_score_us':0,'p_gain':0,'p_valid':1},'information':{'scenario_score_us':30,'p_resolve':.5,'p_followup_gain':.2,'unblocks':['future.v1'],'decision':'choose next valid source'},'basis':'PRIVATE_BASIS','confidence':'low','evidence':['audit.md']},
  'cost':{'gpu_seconds':120,'followup_gpu_seconds':120,'cpu_minutes':10},
  'launch':{'host':'root@107.170.6.60','env':{'SECRET_TOKEN':'DO_NOT_EXPORT'},'argv':['bash','-c','PRIVATE_COMMAND']},
  'files':{'PRIVATE_PATH':'PRIVATE_HASH'},'owner':'PRIVATE_OWNER','lease':{'PID':999,'token':'DO_NOT_EXPORT'},
  'observations':[{'raw_log':'PRIVATE_LOG','host':'PRIVATE_HOST','validation':{'disposition':'VALID_REGRESSION'}}]}

class SnapshotTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  path=self.root/'proj_rsi/history_catalog.json';path.parent.mkdir(parents=True)
  path.write_text(json.dumps({'authority':{'best_job':'best.id','score_us':3200.615},'submissions':[{'job_id':'best.id','candidate':'rs32','score_us':3200.615,'passed':True,'stack_description':'lossless baseline','source':{'path':'PRIVATE_PATH'}}],
   'checks':[{'job_id':'failed.id','passed':False,'score_us':None}],
   'improvements':[{'id':'EH','classification':'OMITTED','currently_included':'No','finding':'historical positive; source only','next_action':'review source','secret':'DO_NOT_EXPORT'}],
   'candidates':[{'candidate':'pw38','parent':'pw36','status':'CPU_ONLY_NOT_SCORED','description':'optional EH','official_jobs':[],'leaderboard_submitted':False,'launch':{'token':'DO_NOT_EXPORT'}}]}))
  self.runtime=self.root/'results/proj-rsi/scheduler'
 def test_authoritative_wal_read_does_not_mutate_any_scheduler_file(self):
  wal(self.runtime,{'profile.v1':job()});checkpoint=self.runtime/'checkpoint.json';checkpoint.write_text('STALE_CHECKPOINT')
  before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.runtime.iterdir()}
  snap=build_snapshot(self.root,now=1000)
  after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.runtime.iterdir()}
  self.assertEqual(before,after)
  self.assertEqual(snap['queue']['sequence'],1)
  self.assertEqual(snap['queue']['counts']['gpu']['RUNNING'],1)
  self.assertEqual(snap['queue']['jobs'][0]['outcome']['status'],'VALID_REGRESSION')
 def test_redaction_allowlist_never_exports_execution_payloads(self):
  wal(self.runtime,{'profile.v1':job()});snap=build_snapshot(self.root,now=1000);encoded=json.dumps(snap)
  for forbidden in ('DO_NOT_EXPORT','PRIVATE_BASIS','PRIVATE_COMMAND','PRIVATE_LOG','PRIVATE_HOST','PRIVATE_PATH','PRIVATE_OWNER','107.170.6.60','SECRET_TOKEN','argv','lease'):
   self.assertNotIn(forbidden,encoded)
  self.assertNotIn('admission_valid',snap['queue']['jobs'][0]['readiness'])
  self.assertFalse(snap['queue']['jobs'][0]['readiness']['descriptor_valid'])
 def test_unknown_missing_sources_not_inferred_from_running_job(self):
  wal(self.runtime,{'profile.v1':job()});snap=build_snapshot(self.root,now=1000)
  self.assertEqual(snap['agents']['items'],[])
  self.assertEqual(snap['hardware']['status'],'unknown')
  self.assertIsNone(snap['sources']['agents']['observed_at'])
  self.assertIsNone(snap['agents']['observed_count'])
  self.assertIsNone(snap['sources']['hardware']['stale'])
  self.assertEqual(snap['sources']['scheduler']['last_event_at'],900)
 def test_explicit_agent_observation_and_staleness_preserve_recorded_status(self):
  folder=self.root/'results/proj-rsi/dashboard';folder.mkdir(parents=True)
  (folder/'agents.json').write_text(json.dumps({'observed_at':500,'agents':[{'agent_name':'/root/a','agent_status':'running','token':'DO_NOT_EXPORT'},{'agent_name':'/root/b','agent_status':{'completed':'PRIVATE_COMMAND'}}]}))
  snap=build_snapshot(self.root,now=1000)
  self.assertEqual([a['status'] for a in snap['agents']['items']],['running','completed'])
  self.assertTrue(all(a['stale'] for a in snap['agents']['items']))
  self.assertEqual(snap['agents']['items'][0]['age_seconds'],500)
  self.assertNotIn('PRIVATE_COMMAND',json.dumps(snap))
 def test_torn_and_corrupt_wal_are_unavailable_and_never_repaired(self):
  for torn,checksum,status in [(True,True,'torn_wal_no_repair'),(False,False,'corrupt_wal_no_repair')]:
   wal(self.runtime,{'profile.v1':job()},torn=torn,checksum=checksum)
   before={p.name:p.read_bytes() for p in self.runtime.iterdir()}
   snap=build_snapshot(self.root,now=1000)
   self.assertEqual(snap['sources']['scheduler']['status'],status)
   self.assertEqual(snap['queue']['jobs'],[])
   self.assertEqual(before,{p.name:p.read_bytes() for p in self.runtime.iterdir()})
 def test_missing_scheduler_does_not_create_directory_or_lock(self):
  build_snapshot(self.root,now=1000)
  self.assertFalse(self.runtime.exists())
  snap=build_snapshot(self.root,now=1000)
  self.assertIsNone(snap['queue']['counts']['gpu']['RUNNING'])
 def test_optional_hardware_is_timestamped_without_runtime_inference(self):
  folder=self.root/'results/proj-rsi/dashboard';folder.mkdir(parents=True)
  (folder/'hardware.json').write_text(json.dumps({'observed_at':999,'status':'idle','gpu_name':'H100','memory_used_mb':0,'process_count':0,'host':'PRIVATE_HOST','command':'PRIVATE_COMMAND'}))
  snap=build_snapshot(self.root,now=1000)
  self.assertEqual(snap['hardware']['status'],'idle')
  self.assertFalse(snap['sources']['hardware']['stale'])
  self.assertNotIn('PRIVATE_HOST',json.dumps(snap))
 def test_history_summaries_keep_failure_unknown_and_pending_source(self):
  snap=build_snapshot(self.root,now=1000)
  self.assertEqual(snap['summary']['submissions'],1)
  self.assertIsNone(snap['checks'][0]['score_us'])
  self.assertFalse(snap['checks'][0]['passed'])
  self.assertEqual(snap['candidates'][0]['status'],'CPU_ONLY_NOT_SCORED')
  self.assertFalse(snap['candidates'][0]['leaderboard_submitted'])
 def test_atomic_json_export_and_strict_serialization(self):
  snapshot=build_snapshot(self.root,now=1000);output=self.root/'public/snapshot.json'
  write_snapshot(output,snapshot)
  self.assertEqual(json.loads(output.read_text()),snapshot)
  self.assertEqual([p.name for p in output.parent.iterdir()],['snapshot.json'])
  with self.assertRaises(ValueError):write_snapshot(output,{'bad':float('nan')})
 def test_schema_allowlist_matches_exported_snapshot(self):
  schema=json.loads(Path(__file__).with_name('schema.json').read_text())
  snapshot=build_snapshot(self.root,now=1000)
  def check(value,rule):
   if '$ref' in rule:return check(value,schema['$defs'][rule['$ref'].split('/')[-1]])
   if 'const' in rule:self.assertEqual(value,rule['const'])
   if 'enum' in rule:self.assertIn(value,rule['enum'])
   if isinstance(value,dict):
    self.assertTrue(set(rule.get('required',[]))<=set(value))
    if rule.get('additionalProperties') is False:self.assertTrue(set(value)<=set(rule['properties']))
    for key,item in value.items():check(item,rule['properties'][key])
   if isinstance(value,list):
    for item in value:check(item,rule['items'])
  check(snapshot,schema)
 def test_future_agent_observation_is_not_labeled_fresh(self):
  folder=self.root/'results/proj-rsi/dashboard';folder.mkdir(parents=True)
  (folder/'agents.json').write_text(json.dumps({'observed_at':2000,'agents':[{'agent_name':'/root/a','agent_status':'running'}]}))
  snap=build_snapshot(self.root,now=1000)
  self.assertEqual(snap['sources']['agents']['status'],'future_observation')
  self.assertIsNone(snap['agents']['items'][0]['stale'])
 def test_quoted_credentials_bearer_and_internal_paths_redacted(self):
  for text in ('authorization: Bearer INDEPENDENT_SENTINEL', 'debug {"token": "INDEPENDENT_SENTINEL"}', 'debug {"password": "first INDEPENDENT_SENTINEL"}', 'service_token=INDEPENDENT_SENTINEL', r'C:\Users\INDEPENDENT_SENTINEL\secret', "api_key='INDEPENDENT_SENTINEL'", 'review /home/ubuntu/okc/private.env', '/tmp/private-token'):
   output=safe_text(text)
   for forbidden in ('INDEPENDENT_SENTINEL','/home/ubuntu/okc','/tmp/private-token'):
    self.assertNotIn(forbidden,output)
 def test_malformed_observation_shape_is_not_ok(self):
  folder=self.root/'results/proj-rsi/dashboard';folder.mkdir(parents=True)
  for name in ('agents','hardware'):(folder/(name+'.json')).write_text('[]')
  snap=build_snapshot(self.root,now=1000)
  self.assertEqual(snap['sources']['agents']['status'],'invalid_shape')
  self.assertEqual(snap['sources']['hardware']['status'],'invalid_shape')
 def test_text_sanitizer_redacts_embedded_credentials_addresses_commands(self):
  value=safe_text('token=SECRETVALUE root@107.170.6.60 https://host/private python3 /okc/driver.py --secret X')
  for forbidden in ('SECRETVALUE','107.170.6.60','host/private','driver.py'):
   self.assertNotIn(forbidden,value)

if __name__=='__main__':unittest.main()
