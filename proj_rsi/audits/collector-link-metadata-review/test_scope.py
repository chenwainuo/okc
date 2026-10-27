"""Temporary fake-transport collector checks; no SSH/GPU/live queue."""
import json
from pathlib import Path
import unittest
from proj_rsi.scheduling import test_dispatch as fixtures

class LinkScope(unittest.TestCase):
 def fixture(self,metadata,tamper=False):
  fixture=fixtures.DispatcherTests('test_exit_zero_waits_for_root_validation');fixture.setUp();self.addCleanup(fixture.doCleanups)
  fixture.job['launch']['artifacts']=['/root/okc/logs/bundle'];fixture.launch();fixture.transport.terminal()
  fixture.transport.remote['logs/bundle/result.json']=b'{"status":"PASS"}\n'
  inventory=fixture.transport.inventory
  fixture.transport.inventory=lambda *args:{**inventory(*args),'symlinks':metadata}
  downloaded=[];download=fixture.transport.download
  def transfer(host,remote,paths,dest):
   downloaded.extend(paths)
   if tamper:fixture.transport.remote['logs/bundle/result.json']=b'tampered'
   return download(host,remote,paths,dest)
  fixture.transport.download=transfer
  return fixture,downloaded
 def test_metadata_only_target_never_transferred(self):
  metadata={'logs/bundle/repo/.claude/skills':'../.agents/skills','logs/bundle/absolute':'/etc/passwd'}
  f,downloaded=self.fixture(metadata);result=f.dispatch.collect(f.job['id'])
  self.assertEqual(result['status'],'AWAITING_ROOT_VALIDATION')
  inventory=f.root/result['inventory'];saved=json.loads(inventory.read_text())
  self.assertEqual(saved['symlinks_not_followed_or_materialized'],metadata)
  self.assertNotIn('logs/bundle/repo/.claude/skills',downloaded)
  self.assertNotIn('../.agents/skills',downloaded);self.assertNotIn('/etc/passwd',downloaded)
  for rel in metadata:self.assertFalse((inventory.parent/rel).exists());self.assertFalse((inventory.parent/rel).is_symlink())
  self.assertTrue((inventory.parent/'logs/bundle/result.json').is_file())
 def test_bad_scope_and_shape_rejected_before_transfer(self):
  for metadata in ({'../escape':'x'},{'logs/bundle2/link':'x'},{'other/link':'x'},{'logs/bundle/link':123},{'logs/bundle/link\n':'x'},{'logs/bundle/result.json':'x'}):
   with self.subTest(metadata=metadata):
    f,downloaded=self.fixture(metadata);result=f.dispatch.collect(f.job['id'])
    self.assertEqual(result['status'],'COLLECTION_FAILED');self.assertEqual(downloaded,[])
 def test_regular_file_hash_verification_unchanged(self):
  f,downloaded=self.fixture({'logs/bundle/link':'../outside'},tamper=True)
  result=f.dispatch.collect(f.job['id']);self.assertEqual(result['status'],'COLLECTION_FAILED');self.assertIn('hash mismatch',result['error'])
if __name__=='__main__':unittest.main(verbosity=2)
