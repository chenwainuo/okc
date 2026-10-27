"""Independent CPU-only proposal integrity and fault probes; temporary stores only."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
from unittest import mock
from proj_rsi.scheduling import _store_cached_proposal as C
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=ROOT/'proj_rsi/audits/scheduler-wal-cache/proposal-manifest.json';m=json.loads(manifest.read_text())
 for n,h in m['files'].items():assert sha(ROOT/n)==h,n
 assert sha(ROOT/'proj_rsi/scheduling/store.py')==m['baseline_store_sha256']
 assert sha(ROOT/'proj_rsi/scheduling/test_store.py')==m['baseline_test_store_sha256']
 old=ast.parse((ROOT/'proj_rsi/scheduling/store.py').read_text());new=ast.parse((ROOT/'proj_rsi/scheduling/_store_cached_proposal.py').read_text())
 def methods(tree):return {n.name:n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='QueueStore').body if isinstance(n,ast.FunctionDef)}
 a,b=methods(old),methods(new);changed=[k for k in a if ast.dump(a[k],include_attributes=False)!=ast.dump(b[k],include_attributes=False)]
 assert changed==['__init__','_read','_commit']
 probes={}
 with tempfile.TemporaryDirectory(prefix='independent-wal-cache-') as tmp:
  q=C.QueueStore(Path(tmp)/'queue',repo_root=ROOT);inode=q.lock_path.stat().st_ino
  # Only temporary synthetic state; no live store construction or WAL reads.
  with q._locked():
   s=q._read()
   for i in range(4):s['jobs']['fake']={'i':i};q._commit(s,'independent-synthetic')
   expected=q._read();wal=q.wal_path.read_bytes()
  assert q.cache_key_path.stat().st_mode&0o777==0o600
  probes['new_key_mode']='0600'
  # Permission limitation is documented, not silently assumed enforced.
  os.chmod(q.cache_key_path,0o644)
  with q._locked():assert q._read()==expected
  probes['existing_permissive_key']='ACCEPTED_UNDER_TRUSTED_DIRECTORY_MODEL; mode is not checked'
  os.chmod(q.cache_key_path,0o600)
  original=C._atomic
  def fail_optional(path,content,**kw):
   if path in (q.cache_key_path,q.cache_path):raise OSError('independent optional metadata failure')
   return original(path,content,**kw)
  q.cache_key_path.unlink();q.cache_path.unlink()
  with mock.patch.object(C,'_atomic',side_effect=fail_optional),q._locked():
   assert q._read()==expected;s=q._read();s['jobs']['fake']['i']=4;q._commit(s,'key-and-cache-unwritable')
  with q._locked():expected=q._read()
  assert expected['sequence']==5 and expected['jobs']['fake']['i']==4
  probes['both_key_and_cache_write_failure']='WAL_COMMIT_AND_FULL_REPLAY_PRESERVED'
  # A same-size valid-checksum edit to history has baseline semantics and invalidates hash cache.
  lines=q.wal_path.read_bytes().splitlines(keepends=True);entry=json.loads(lines[0]);entry['payload']['snapshot']['version']=2;entry['sha256']=hashlib.sha256(C._json(entry['payload'])).hexdigest();lines[0]=C._json(entry)+b'\n';q.wal_path.write_bytes(b''.join(lines))
  try:
   with q._locked():q._read()
  except C.CorruptWAL:pass
  else:raise AssertionError('historical envelope corruption hidden')
  probes['historical_checksum_valid_bad_envelope']='REFUSED_EVEN_WITH_OLD_CACHE'
  assert q.lock_path.stat().st_ino==inode;probes['stable_lock_inode']='PRESERVED'
 result={'verdict':'GO_EXACT_PROPOSAL_TRUSTED_LOCAL_DEPLOYMENT_ONLY','proposal_sha256':sha(ROOT/'proj_rsi/scheduling/_store_cached_proposal.py'),'manifest_sha256':sha(manifest),'manifest_files':len(m['files']),'baseline_store_sha256':m['baseline_store_sha256'],'baseline_test_sha256':m['baseline_test_store_sha256'],'live_baselines_untouched':True,'changed_existing_methods':changed,'independent_temporary_probes':probes,'live_queue_operations':0,'live_WAL_reads':0,'GPU_API_operations':0}
 (ROOT/'proj_rsi/audits/scheduler-wal-cache-independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
