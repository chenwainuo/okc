"""Meaningful CPU-only persistence/admission/concurrency tests; no transport imports."""
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from .model import load_policy
from .store import QueueStore, QueueError, CorruptWAL


def job(name, resource='cpu', value=10):
    return {'id':name,'resource':resource,'kind':'implementation' if resource=='cpu' else 'optimization',
            'state':'BACKLOG','author':'worker','created_at':1,'depends_on':[],'write_paths':[],
            'estimate':{'direct':{'scenario_score_us':value,'p_gain':.5,'p_valid':1},
                        'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0},
                        'basis':'subjective conditional prior','confidence':'low','evidence':['evidence.md']},
            'cost':{'gpu_seconds':120 if resource=='gpu' else 0,'followup_gpu_seconds':0,'cpu_minutes':1}}


def _claim_process(directory, repo, resource, result):
    try:
        claimed = QueueStore(directory, repo_root=repo).claim(resource, 'process')
        result.put(claimed['id'] if claimed else None)
    except BaseException as error:
        result.put('ERROR:' + repr(error))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'; self.repo.mkdir()
        (self.repo/'source.py').write_text('print(1)\n')
        (self.repo/'proj_rsi/tools').mkdir(parents=True)
        for name in ('h100_trial.sh', 'h100_container.sh'):
            (self.repo/'proj_rsi/tools'/name).write_text('# frozen test hardware guard\n')
        (self.repo/'review.md').write_text('independent GO for exact source\n')
        self.directory = self.root/'runtime'
        self.store = QueueStore(self.directory, repo_root=self.repo)

    def gpu(self, name, value=10):
        descriptor = job(name,'gpu',value)
        descriptor['files'] = {'source.py':hashlib.sha256((self.repo/'source.py').read_bytes()).hexdigest()}
        for tool_name in ('h100_trial.sh', 'h100_container.sh'):
            path = 'proj_rsi/tools/' + tool_name
            descriptor['files'][path] = hashlib.sha256((self.repo/path).read_bytes()).hexdigest()
        descriptor['review'] = {'actor':'critic','path':'review.md','sha256':hashlib.sha256((self.repo/'review.md').read_bytes()).hexdigest(),'decision':'GO'}
        descriptor['launch'] = {'host':'root@107.170.6.60','remote_repo':'/root/okc','container':'okc-proj-'+name,
                                'inner_seconds':240,'outer_seconds':300,'profile':False,'env':{},
                                'argv':['python3','/okc/source.py'],'artifacts':['/root/okc/logs/'+name+'.json']}
        return descriptor

    def test_ready_full_rebalances_and_preserves_running(self):
        for i in range(22): self.store.register(job('j%02d'%i,value=i+1))
        snapshot = self.store.snapshot()['jobs']
        self.assertEqual(sum(j['state']=='READY' for j in snapshot.values()),20)
        self.assertEqual(sum(j['state']=='BACKLOG' for j in snapshot.values()),2)
        running=self.store.claim('cpu','root')
        self.store.register(job('urgent',value=1000))
        snapshot=self.store.snapshot()['jobs']
        self.assertEqual(snapshot['urgent']['state'],'READY')
        self.assertEqual(snapshot[running['id']]['state'],'RUNNING')
        self.assertEqual(snapshot['j01']['state'],'BACKLOG')

    def test_process_race_gpu_exactly_one_claim_and_restart_handles(self):
        for i in range(12): self.store.register(self.gpu('g%02d'%i,i+1))
        context=multiprocessing.get_context('fork'); result=context.Queue()
        processes=[context.Process(target=_claim_process,args=(str(self.directory),str(self.repo),'gpu',result)) for _ in range(16)]
        for process in processes: process.start()
        for process in processes:
            process.join(20); self.assertEqual(process.exitcode,0)
        outputs=[result.get(timeout=3) for _ in processes]
        self.assertFalse(any(isinstance(v,str) and v.startswith('ERROR:') for v in outputs),outputs)
        claimed=[v for v in outputs if v is not None]
        self.assertEqual(claimed,['g11'])
        self.store.attach('g11',{'dispatch':{'pid':123,'container':'okc-proj-g11'}})
        restarted=QueueStore(self.directory,repo_root=self.repo)
        self.assertEqual(restarted.get('g11')['observations'][-1]['dispatch']['pid'],123)
        self.assertIsNone(restarted.claim('gpu','other'))

    def test_cpu_cap_and_overlapping_write_paths(self):
        for i in range(12): self.store.register(job('c%02d'%i))
        claims=[self.store.claim('cpu','worker') for _ in range(12)]
        self.assertEqual(sum(v is not None for v in claims),10)
        other=QueueStore(self.root/'other',repo_root=self.repo)
        a=job('a',value=30); a['write_paths']=['tree']
        b=job('b',value=20); b['write_paths']=['tree/sub/file.py']
        c=job('c',value=10); c['write_paths']=['different/file.py']
        for descriptor in (a,b,c): other.register(descriptor)
        self.assertEqual(other.claim('cpu','one')['id'],'a')
        self.assertEqual(other.claim('cpu','two')['id'],'c')
        self.assertIsNone(other.claim('cpu','three'))

    def test_wal_restart_checkpoint_ignored_and_torn_evidence(self):
        self.store.register(job('one'))
        stable=self.store.snapshot()
        self.store.checkpoint_path.write_text('{incorrect checkpoint')
        self.assertEqual(QueueStore(self.directory,repo_root=self.repo).snapshot(),stable)
        with self.store.wal_path.open('ab') as handle: handle.write(b'{"torn":')
        self.assertEqual(QueueStore(self.directory,repo_root=self.repo).snapshot(),stable)
        evidence=list(self.directory.glob('torn-final-*.bin'))
        self.assertEqual(len(evidence),1); self.assertEqual(evidence[0].read_bytes(),b'{"torn":')
        self.store.register(job('two'))
        self.assertEqual(set(QueueStore(self.directory,repo_root=self.repo).snapshot()['jobs']),{'one','two'})

    def test_middle_corruption_fail_closed(self):
        self.store.register(job('one')); self.store.register(job('two'))
        raw=self.store.wal_path.read_bytes(); self.store.wal_path.write_bytes(raw.replace(b'"register"',b'"tampered"',1))
        with self.assertRaises(CorruptWAL): self.store.claim('cpu','worker')

    def test_unknown_estimate_held_and_dependency_requires_valid_done(self):
        unknown=job('unknown'); unknown['estimate']['direct']['p_gain']=None
        self.assertEqual(self.store.register(unknown)['state'],'HELD')
        self.store.register(job('dependency'))
        dependent=job('dependent'); dependent['depends_on']=['dependency']
        self.assertEqual(self.store.register(dependent)['state'],'BACKLOG')
        self.store.claim('cpu','worker')
        self.store.finish('dependency','INVALID_SETUP',evidence={'exit':1},actor='worker')
        self.assertEqual(self.store.get('dependency')['state'],'VALIDATING')
        self.store.transition('dependency','DONE',actor='root',reason='invalid attempt retained',evidence={'disposition':'INVALID_SETUP','artifact':'raw.json'})
        self.assertEqual(self.store.get('dependent')['state'],'BACKLOG')
        accepted=job('explicit'); accepted['depends_on']=[{'id':'dependency','accepted_dispositions':['INVALID_SETUP']}]
        self.assertEqual(self.store.register(accepted)['state'],'READY')

    def test_exit_zero_never_promotion_root_validation_and_active_freeze(self):
        self.store.register(self.gpu('g'))
        self.store.claim('gpu','root')
        with self.assertRaises(QueueError): self.store.update('g',{'launch':{}},actor='root',reason='illegal')
        with self.assertRaises(QueueError): self.store.finish('g','VALID_GAIN',evidence={'exit':0},actor='runner')
        cleanup = {'cleanup_verified':True,'cleanup_observation':{'container_exists':False,'guard_pids':[], 'compute_pids':[], 'gpu_memory_mib':0,'containers':[], 'cleanup_failed':False}}
        self.store.finish('g','VALID_GAIN',evidence={'exit':0, **cleanup},actor='runner')
        self.assertEqual(self.store.get('g')['state'],'VALIDATING')
        with self.assertRaises(QueueError): self.store.transition('g','DONE',actor='worker',reason='self validate',evidence={'disposition':'VALID_GAIN'})
        with self.assertRaises(QueueError): self.store.transition('g','DONE',actor='root',reason='exit zero alone',evidence={'exit':0})
        self.store.transition('g','DONE',actor='root',reason='independent correctness/rules/task gates reviewed',evidence={'disposition':'VALID_NO_GAIN','review':'validation.md'})
        with self.assertRaises(QueueError): self.store.update('g',{'annotations':{}},actor='root',reason='terminal')

    def test_stale_lease_not_auto_restart_kill_or_cancel(self):
        self.store.register(self.gpu('g')); claimed=self.store.claim('gpu','root')
        self.store.heartbeat('g',{'remote_status':'UNKNOWN','lease_expired':True})
        self.store.rebalance('gpu')
        self.assertEqual(self.store.get('g')['state'],'RUNNING')
        self.assertEqual(self.store.get('g')['lease']['id'],claimed['lease']['id'])
        self.assertIsNone(self.store.claim('gpu','other'))
        with self.assertRaises(QueueError): self.store.transition('g','CANCELLED',actor='root',reason='stale lease')

    def test_gpu_source_review_drift_and_changed_launch_invalidate_go(self):
        self.store.register(self.gpu('g'))
        (self.repo/'source.py').write_text('print(2)\n')
        self.assertIsNone(self.store.claim('gpu','root'))
        self.assertEqual(self.store.get('g')['state'],'HELD')
        other=QueueStore(self.root/'other',repo_root=self.repo); descriptor=self.gpu('other'); other.register(descriptor)
        changed=deepcopy(descriptor['launch']); changed['argv'].append('--changed')
        revised=other.update('other',{'launch':changed},actor='root',reason='new executable needs review')
        self.assertEqual(revised['state'],'HELD'); self.assertNotIn('review',revised)


    def test_gpu_validating_cleanup_releases_slot_but_unknown_does_not(self):
        self.store.register(self.gpu('first',100)); self.store.register(self.gpu('second',10))
        self.store.claim('gpu','root')
        with self.assertRaises(QueueError): self.store.transition('first','VALIDATING',actor='root',reason='mere exit0',evidence={'exit':0})
        self.assertIsNone(self.store.claim('gpu','root'))
        cleanup={'cleanup_verified':True,'cleanup_observation':{'container_exists':False,'guard_pids':[], 'compute_pids':[], 'gpu_memory_mib':0,'containers':[], 'cleanup_failed':False}}
        self.store.transition('first','VALIDATING',actor='root',reason='physical cleanup verified',evidence=cleanup)
        self.assertEqual(self.store.claim('gpu','root')['id'],'second')
        self.assertEqual(self.store.get('first')['state'],'VALIDATING')

    def test_prelaunch_hold_only_before_launch_request(self):
        self.store.register(self.gpu('g')); self.store.claim('gpu','root')
        self.store.attach('g',{'dispatch':{'phase':'CLAIMED'}})
        self.assertEqual(self.store.release_unlaunched('g',actor='root',reason='remote busy before launch')['state'],'HELD')
        self.store.register(self.gpu('other')); self.store.claim('gpu','root')
        self.store.attach('other',{'dispatch':{'phase':'LAUNCH_REQUESTED'}})
        with self.assertRaises(QueueError): self.store.release_unlaunched('other',actor='root',reason='SSH timeout')
        self.assertEqual(self.store.get('other')['state'],'RUNNING')

    def test_same_review_content_cannot_reapprove_changed_execution(self):
        descriptor=self.gpu('g'); self.store.register(descriptor)
        review={**descriptor['review'],'actor':'other-critic'}
        launch={**descriptor['launch'],'argv':['python3','/okc/changed.py']}
        revised=self.store.update('g',{'review':review,'launch':launch},actor='root',reason='changed execution')
        self.assertEqual(revised['state'],'HELD'); self.assertNotIn('review',revised)

    def test_runtime_cap_policy_rejected(self):
        for resource, cap in [('gpu',2),('cpu',11)]:
            policy=load_policy(); policy['active_capacity'][resource]=cap
            with self.assertRaises(QueueError): QueueStore(self.root/resource,repo_root=self.repo,policy=policy)


if __name__=='__main__': unittest.main()
