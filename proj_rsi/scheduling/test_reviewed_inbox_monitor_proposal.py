"""Monitor proposal tests: only temporary directories and FakeStore."""
from copy import deepcopy
import contextlib,io,json,unittest
from unittest.mock import patch
from . import _reviewed_inbox_monitor_proposal as monitor
from . import test_reviewed_inbox_proposal as fixtures

class Clock:
 def __init__(self):self.now=0;self.sleeps=[]
 def __call__(self):return self.now
 def sleep(self,n):self.sleeps.append(n);self.now+=n

class MonitorTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.InboxTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  f=self.fixture;self.root=f.root;self.store=f.store
  self.policy=json.loads((monitor.ROOT/monitor.POLICY).read_text())
  f.job['id']='pw44.sample.v1';f.write();self.clock=Clock();self.rows=[]
 def run_once(self,**kw):
  return monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,once=True,clock=self.clock,sleep=self.clock.sleep,emit=self.rows.append,**kw)
 def test_default_no_factory_or_main_store(self):
  def forbidden():raise AssertionError('factory touched')
  self.assertEqual(monitor.run_monitor(forbidden,self.root,None)['cycles'],0)
  with patch.object(monitor,'verify_bundle',side_effect=AssertionError('bundle touched')),contextlib.redirect_stdout(io.StringIO()) as out:monitor.main([])
  self.assertIn('DISABLED',out.getvalue())
 def test_exact_policy_and_bounds(self):
  monitor.validate_policy(self.policy)
  for key,value in [('gpu_usd_per_hour',1),('allowed_prefixes',['pw43.']),('allowed_ids',['pw44.x']),('max_wall_seconds',7201),('max_poll_seconds',31)]:
   bad=deepcopy(self.policy);bad[key]=value
   with self.assertRaises(ValueError):monitor.validate_policy(bad)
  for kw in ({'poll_seconds':31},{'poll_seconds':0},{'max_wall_seconds':7201},{'max_wall_seconds':0}):
   with self.assertRaises(ValueError):self.run_once(**kw)
 def test_new_registered_once_backlog_restart_immutable(self):
  self.assertEqual(self.run_once()['registrations'],1)
  self.assertEqual(self.store.calls[0]['state'],'BACKLOG')
  self.store.jobs['pw44.sample.v1']['state']='HELD'
  self.assertEqual(self.run_once()['registrations'],0);self.assertEqual(len(self.store.calls),1)
 def test_batch_existing_prefilter_one_snapshot_no_sync(self):
  self.store.jobs['pw44.sample.v1']={'state':'RUNNING'}
  original=self.store.snapshot
  with patch.object(self.store,'snapshot',wraps=original) as snap,patch.object(monitor.inbox,'sync_reviewed_new',side_effect=AssertionError('existing sync')):
   self.run_once();self.assertEqual(snap.call_count,1)
 def test_partial_then_complete_reconsidered_next_poll(self):
  f=self.fixture;(self.root/f.relative).write_text('{')
  self.run_once();self.assertEqual(len(self.store.calls),0)
  f.write();self.assertEqual(self.run_once()['registrations'],1)
 def test_outside_prefix_and_cpu_never_sync(self):
  for identifier,resource in [('pw43.sample','gpu'),('pw44.sample','cpu')]:
   self.fixture.job.update(id=identifier,resource=resource);self.fixture.write()
   with patch.object(monitor.inbox,'sync_reviewed_new',side_effect=AssertionError('forbidden sync')):self.run_once()
 def test_monotonic_wall_bound_and_onchange_logging(self):
  self.store.jobs['pw44.sample.v1']={'state':'RUNNING'}
  result=monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,poll_seconds=15,max_wall_seconds=31,clock=self.clock,sleep=self.clock.sleep,emit=self.rows.append)
  self.assertEqual(result['cycles'],3);self.assertEqual(self.clock.now,31);self.assertEqual(len(self.rows),1)
  self.assertTrue(all(n<=30 for n in self.clock.sleeps))
 def test_invalid_clock_before_factory_and_backward(self):
  for value in (float('nan'),float('inf'),True):
   with self.assertRaises(ValueError):monitor.run_monitor(lambda:self.fail('factory'),self.root,self.policy,enabled=True,clock=lambda:value)
  values=iter([2,1])
  with self.assertRaises(ValueError):monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,clock=lambda:next(values))
 def test_compact_allowlist_no_raw_content(self):
  row=monitor.compact({'action':'SKIP','code':'X','argv':['SECRET'],'evidence':{'id':'pw44.x','env':'SECRET','host':'SECRET','pins':{'private/path':'a'*64}}},0)
  serialized=json.dumps(row);self.assertNotIn('SECRET',serialized);self.assertNotIn('private/path',serialized)
 def test_manifest_requires_pin_complete_closure_and_drift(self):
  path=self.root/monitor.MANIFEST;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"files":{}}')
  with self.assertRaises(ValueError):monitor.verify_bundle(self.root)
  with self.assertRaises(ValueError):monitor.verify_bundle(self.root,monitor._sha(path.read_bytes()))
  for relative in monitor.REQUIRED_FILES:
   p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True)
   if relative==monitor.POLICY:p.write_text(json.dumps(self.policy))
   elif not p.exists():p.write_text('pinned')
  manifest={'files':{r:monitor._sha((self.root/r).read_bytes()) for r in monitor.REQUIRED_FILES}}
  path.write_text(json.dumps(manifest));pin=monitor._sha(path.read_bytes());monitor.verify_bundle(self.root,pin)
  (self.root/'proj_rsi/scheduling/store.py').write_text('drift')
  with self.assertRaises(ValueError):monitor.verify_bundle(self.root,pin)

if __name__=='__main__':unittest.main()
