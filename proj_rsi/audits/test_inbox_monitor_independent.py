"""Independent monitor integration tests: FakeStore, clocks and temp files only."""
import contextlib,copy,io,json,unittest
from pathlib import Path
from proj_rsi.scheduling import _reviewed_inbox_monitor_proposal as monitor
from proj_rsi.audits import test_reviewed_inbox_independent as fixtures

class Clock:
 def __init__(self):self.t=0
 def __call__(self):self.t+=.0001;return self.t
 def sleep(self,n):self.t+=n
class Tests(unittest.TestCase):
 def write(self):self.path.write_text(json.dumps(self.job))
 def setUp(self):
  fixtures.IndependentInbox.setUp(self)
  self.job['id']='pw45.independent.attempt01';self.write= lambda:self.path.write_text(json.dumps(self.job));self.write()
  self.policy=json.loads((Path(monitor.ROOT)/monitor.POLICY).read_text())
 def test_default_main_has_no_store(self):
  with contextlib.redirect_stdout(io.StringIO()) as out:monitor.main([])
  self.assertEqual(json.loads(out.getvalue())['status'],'DISABLED_NO_STORE_CREATED')
  def forbidden():raise AssertionError('created store')
  self.assertEqual(monitor.run_monitor(forbidden,self.root,self.policy)['status'],'DISABLED_NO_STORE_CREATED')
 def test_once_registers_new_only(self):
  rows=[];c=Clock();r=monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,once=True,clock=c,sleep=c.sleep,wall=c,emit=rows.append)
  self.assertEqual(r['registrations'],1);self.assertEqual(len(self.store.calls),1);self.assertEqual(self.store.calls[0]['state'],'BACKLOG')
  self.assertNotIn('registration',rows[0]);self.assertNotIn('files',rows[0]);self.assertEqual(rows[0]['action'],'REGISTERED_NEW')
 def test_existing_batch_does_not_sync_snapshot_per_row(self):
  for i in range(40):
   job=copy.deepcopy(self.job);job['id']='pw45.existing.'+str(i);(self.path.parent/(str(i)+'.json')).write_text(json.dumps(job));self.store.jobs[job['id']]={'id':job['id'],'state':'HELD'}
  self.store.jobs[self.job['id']]={'id':self.job['id'],'state':'RUNNING'}
  count=[0];native=self.store.snapshot
  def snap():count[0]+=1;return native()
  self.store.snapshot=snap;c=Clock();r=monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,once=True,clock=c,sleep=c.sleep,wall=c)
  self.assertEqual(count[0],1);self.assertFalse(self.store.calls);self.assertEqual(r['registrations'],0)
 def test_bounded_scan_deduplicates_and_no_retries(self):
  self.store.jobs[self.job['id']]={'id':self.job['id'],'state':'DONE'};rows=[];c=Clock();r=monitor.run_monitor(lambda:self.store,self.root,self.policy,enabled=True,poll_seconds=1,max_wall_seconds=3,clock=c,sleep=c.sleep,wall=c,emit=rows.append)
  self.assertGreaterEqual(r['cycles'],2);self.assertLessEqual(r['cycles'],4);self.assertEqual(len(rows),1);self.assertFalse(self.store.calls)
 def test_nonfinite_or_backward_clock_no_registration(self):
  created=[]
  def factory():created.append(True);return self.store
  for value in (float('nan'),float('inf'),True):
   with self.assertRaises(ValueError):monitor.run_monitor(factory,self.root,self.policy,enabled=True,clock=lambda:value)
  self.assertEqual(created,[])
  values=iter([0,1,.5])
  with self.assertRaises(ValueError):monitor.run_monitor(factory,self.root,self.policy,enabled=True,clock=lambda:next(values))
  self.assertFalse(self.store.calls)
 def test_manifest_requires_root_hash_and_full_closure(self):
  import hashlib
  path=self.root/monitor.MANIFEST;path.parent.mkdir(parents=True,exist_ok=True)
  def install(files):
   raw=json.dumps({'files':files}).encode();path.write_bytes(raw);return hashlib.sha256(raw).hexdigest()
  pin=install({})
  for expected in (None,pin,'0'*64):
   with self.assertRaises(ValueError):monitor.verify_bundle(self.root,expected)
  files={}
  for rel in monitor.REQUIRED_FILES:
   p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(self.policy) if rel==monitor.POLICY else 'reviewed fixture '+rel);files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
  pin=install(files);policy,manifest=monitor.verify_bundle(self.root,pin);self.assertEqual(policy,self.policy)
  (self.root/'proj_rsi/scheduling/_reviewed_inbox_proposal.py').write_text('drift')
  with self.assertRaises(ValueError):monitor.verify_bundle(self.root,pin)
 def test_exact_policy_cannot_expand(self):
  edits=[lambda p:p.update(allowed_prefixes=['pw']),lambda p:p.update(gpu_usd_per_hour=0),lambda p:p['bounds']['profile'].update(outer_seconds=3720),lambda p:p.update(max_wall_seconds=7201)]
  for edit in edits:
   p=copy.deepcopy(self.policy);edit(p)
   with self.assertRaises(ValueError):monitor.run_monitor(lambda:(_ for _ in()).throw(AssertionError('store created')),self.root,p,enabled=True)

if __name__=='__main__':unittest.main()
