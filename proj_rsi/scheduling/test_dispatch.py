"""CPU-only dispatcher checks. No SSH, CUDA or real guard is invoked."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock
from .dispatch import GPUDispatcher, HOST, REMOTE_REPO, _relative
from .store import QueueStore


def digest(data):return hashlib.sha256(data).hexdigest()
def idle():return dict(container_exists=False,containers=[],guard_pids=[],compute_pids=[],gpu_memory_mib=0,terminal_exit=None,cleanup_failed=False)

class FakeClock:
    def __init__(self,on_sleep=None):self.elapsed=0.;self.sleeps=[];self.on_sleep=on_sleep
    def monotonic(self):return self.elapsed
    def sleep(self,seconds):
        if not 0<seconds<=60:raise AssertionError('sleep must be positive and bounded')
        self.sleeps.append(seconds);self.elapsed+=seconds
        if self.on_sleep:self.on_sleep(self.elapsed)

class FakeTransport:
    def __init__(self):
        self.remote={};self.observation=idle();self.commands=[];self.uploaded=[]
        self.probe_error=False;self.launch_error=False;self.tamper=False;self.on_upload=None
        self.complete_on_launch=False;self.runner_mode=0o755;self.runner_bad=False
    def upload(self,host,local_root,remote_root,paths):
        self.uploaded=list(paths);self.remote.update({p:(Path(local_root)/p).read_bytes() for p in paths})
        if self.on_upload:self.on_upload()
    def hashes(self,host,remote_root,paths):
        result={p:digest(self.remote[p]) for p in paths}
        if self.tamper:result[paths[0]]='0'*64
        return result
    def runner(self,host):
        return {'sha256':'0'*64 if self.runner_bad else digest(self.remote['proj_rsi/tools/h100_container.sh']),'mode':self.runner_mode}
    def probe(self,host,handles):
        if self.probe_error:raise TimeoutError('simulated SSH outage')
        return copy.deepcopy(self.observation)
    def launch(self,host,command):
        self.commands.append(command)
        self.observation.update(container_exists=True,containers=['okc-proj-one'],guard_pids=[4242],compute_pids=[4243],gpu_memory_mib=19)
        if self.complete_on_launch:self.terminal()
        if self.launch_error:raise TimeoutError('reply lost after guard started')
        return '4242\n'
    def terminal(self,code=0,cleanup=True):
        self.remote['logs/result.json']=json.dumps({'status':'PASS','exit':code}).encode()
        self.remote['logs/scheduler/family.v1.attempt01.host.log']=('TRIAL_EXIT okc-proj-one %d\n'%code).encode()
        if cleanup:self.observation=idle()
        self.observation['terminal_exit']=code
    def inventory(self,host,remote_root,paths):
        files={p:digest(b) for p,b in self.remote.items() if any(p==rel or p.startswith(rel+'/') for rel in paths)}
        return {'files':files,'missing':[p for p in paths if p not in files and not any(f.startswith(p+'/') for f in files)]}
    def download(self,host,remote_root,paths,local_root):
        for p in paths:
            f=Path(local_root)/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(self.remote[p])

class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        data={'lab.py':b'print(1)\n','review.md':b'independent GO\n','proj_rsi/tools/h100_trial.sh':b'# fake guard\n','proj_rsi/tools/h100_container.sh':b'# fake runner\n'}
        for rel,content in data.items():
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        self.store=QueueStore(self.root/'queue',repo_root=self.root)
        self.transport=FakeTransport();self.dispatch=GPUDispatcher(self.store,self.root,transport=self.transport)
        self.job={'id':'family.v1.attempt01','resource':'gpu','kind':'optimization','state':'BACKLOG','author':'author','created_at':1,
            'depends_on':[],'write_paths':[],
            'estimate':{'direct':{'scenario_score_us':1,'p_gain':.5,'p_valid':1},'information':{'scenario_score_us':0,'p_resolve':0,'p_followup_gain':0,'unblocks':[],'decision':'none'},'basis':'subjective conditional prior','confidence':'low','evidence':['review.md']},
            'cost':{'gpu_seconds':60,'followup_gpu_seconds':0,'cpu_minutes':1},'files':{p:digest(b) for p,b in data.items() if p!='review.md'},
            'review':{'actor':'critic','path':'review.md','sha256':digest(data['review.md']),'decision':'GO'},
            'launch':{'host':HOST,'remote_repo':REMOTE_REPO,'container':'okc-proj-one','inner_seconds':30,'outer_seconds':60,'profile':False,'env':{'MAX_JOBS':'1'},'argv':['python3','/okc/lab.py'],'artifacts':['/root/okc/logs/result.json']}}
    def register(self,job=None):return self.store.register(job or self.job)
    def launch(self):self.register();return self.dispatch.launch_next()
    def test_exact_transfer_and_authoritative_pid_no_duplicate(self):
        job=self.launch();self.assertEqual(job['state'],'RUNNING');self.assertEqual(len(self.transport.commands),1)
        self.assertEqual(set(self.transport.uploaded),set(self.job['files'])|{'review.md'})
        self.assertEqual(self.dispatch._handles(job)['guard_pid'],4242)
        self.assertIsNone(self.dispatch.launch_next());self.assertEqual(len(self.transport.commands),1)
    def test_launch_reply_lost_remains_running_no_retry(self):
        self.transport.launch_error=True;job=self.launch();self.assertEqual(job['state'],'RUNNING')
        self.assertEqual(self.dispatch._handles(job)['phase'],'LAUNCH_UNKNOWN');self.transport.probe_error=True
        self.assertEqual(self.dispatch.reconcile(job['id'])['status'],'UNKNOWN');self.assertIsNone(self.dispatch.launch_next())
        self.assertEqual(len(self.transport.commands),1)
    def test_failed_prelaunch_probe_holds(self):
        self.transport.probe_error=True;job=self.launch();self.assertEqual(job['state'],'HELD');self.assertFalse(self.transport.commands)
    def test_external_busy_holds_without_kill(self):
        self.transport.observation.update(compute_pids=[99],gpu_memory_mib=100)
        job=self.launch();self.assertEqual(job['state'],'HELD');self.assertFalse(self.transport.commands)
        self.assertEqual(self.transport.observation['compute_pids'],[99])
    def test_source_change_during_transfer_blocks_launch(self):
        self.transport.on_upload=lambda:(self.root/'lab.py').write_text('changed')
        job=self.launch();self.assertEqual(job['state'],'HELD');self.assertFalse(self.transport.commands)
    def test_remote_hash_mismatch_blocks_launch(self):
        self.transport.tamper=True;job=self.launch();self.assertEqual(job['state'],'HELD');self.assertFalse(self.transport.commands)
    def test_existing_host_log_blocks_new_launch(self):
        self.transport.observation['host_log_exists']=True
        self.assertEqual(self.launch()['state'],'HELD');self.assertFalse(self.transport.commands)
    def test_guard_pin_required(self):
        del self.job['files']['proj_rsi/tools/h100_trial.sh'];self.register();job=self.dispatch.launch_next()
        self.assertTrue(job is None or job['state']=='HELD');self.assertFalse(self.transport.commands)
    def test_installed_runner_hash_and_mode_required(self):
        for mode,bad in ((0o755,True),(0o644,False)):
            self.transport.runner_mode=mode;self.transport.runner_bad=bad
            self.register();job=self.dispatch.launch_next();self.assertEqual(job['state'],'HELD');self.assertFalse(self.transport.commands)
            break
        # Separate descriptor after held failure still cannot bypass wrong mode.
        self.job['id']='family.v2.attempt01';self.job['launch']['container']='okc-proj-two'
        self.transport.runner_bad=False;self.transport.runner_mode=0o644;self.register()
        self.assertEqual(self.dispatch.launch_next()['state'],'HELD')
    def test_container_collision_blocks_launch(self):
        self.register();other=copy.deepcopy(self.job);other['id']='family.v2.attempt01';other['created_at']=2;self.register(other)
        self.assertEqual(self.dispatch.launch_next()['state'],'HELD');self.assertFalse(self.transport.commands)
    def test_profile_explicit_and_shell_metacharacters_literal(self):
        literal='$(printf bad) `printf bad` space " quote';self.job['launch']['argv'].append(literal);self.launch()
        script=shlex.split(self.transport.commands[0])[2];self.assertNotIn('&&',script)
        tokens=shlex.split(next(line for line in script.splitlines() if line.startswith('nohup ')))
        self.assertIn(literal,tokens);self.assertIn('PROJ_GPU_PROFILE=0',tokens)
        out=subprocess.check_output(['bash','-c',"printf '%s' "+shlex.quote(literal)],text=True);self.assertEqual(out,literal)
        launch=copy.deepcopy(self.job['launch']);launch['profile']=True
        command=self.dispatch._command(launch,self.dispatch._handles(self.store.get(self.job['id'])))
        self.assertIn('PROJ_GPU_PROFILE=1',shlex.split(shlex.split(command)[2].splitlines()[2]))
    def test_environment_cannot_escalate_normal_job(self):
        self.job['launch']['env']['PROJ_GPU_PROFILE']='1';self.register();job=self.dispatch.launch_next()
        self.assertTrue(job is None or job['state']=='HELD');self.assertFalse(self.transport.commands)
    def test_stale_lease_live_container_remains_running(self):
        job=self.launch();self.assertEqual(self.dispatch.reconcile(job['id'])['probe']['guard_pids'],[4242])
        self.assertEqual(self.store.get(job['id'])['state'],'RUNNING')
        self.assertEqual(self.dispatch.collect(job['id'])['status'],'AWAITING_TERMINAL_CLEANUP')
    def test_exit_zero_waits_for_root_validation(self):
        job=self.launch();self.transport.terminal();result=self.dispatch.collect(job['id'])
        self.assertEqual(result['status'],'AWAITING_ROOT_VALIDATION');job=self.store.get(job['id'])
        self.assertEqual(job['state'],'VALIDATING');self.assertNotIn('disposition',job)
        manifest=json.loads((self.root/result['inventory']).read_text())
        self.assertEqual(manifest['files']['logs/result.json'],digest(self.transport.remote['logs/result.json']))
    def test_failed_timeout_cleanup_never_releases(self):
        job=self.launch();self.transport.terminal(124,cleanup=False);self.transport.observation['cleanup_failed']=True
        self.assertEqual(self.dispatch.collect(job['id'])['status'],'AWAITING_TERMINAL_CLEANUP');self.assertEqual(self.store.get(job['id'])['state'],'RUNNING')
    def test_clean_timeout_missing_result_collected_for_validator(self):
        job=self.launch();self.transport.terminal(124);del self.transport.remote['logs/result.json']
        result=self.dispatch.collect(job['id']);self.assertEqual(result['status'],'AWAITING_ROOT_VALIDATION')
        manifest=json.loads((self.root/result['inventory']).read_text());self.assertEqual(manifest['missing'],['logs/result.json'])
        self.assertNotIn('disposition',self.store.get(job['id']))
    def test_recollection_preserves_previous_raw_inventory(self):
        job=self.launch();self.transport.terminal();a=self.dispatch.collect(job['id']);b=self.dispatch.collect(job['id'])
        self.assertNotEqual(a['inventory'],b['inventory']);self.assertTrue((self.root/a['inventory']).is_file())
    def test_work_budget_does_not_claim(self):
        self.register();self.assertEqual(self.dispatch.work(max_wall_seconds=59)['status'],'INSUFFICIENT_RESERVATION_BUDGET')
        self.assertEqual(self.store.get(self.job['id'])['state'],'READY');self.assertFalse(self.transport.commands)
    def test_work_collects_without_auto_validation(self):
        self.register();self.transport.complete_on_launch=True
        result=self.dispatch.work(max_jobs=1,max_wall_seconds=120);self.assertEqual(result['status'],'MAX_JOBS')
        self.assertEqual(self.store.get(self.job['id'])['state'],'VALIDATING')
    def test_work_unknown_preserves_handles(self):
        self.launch();self.transport.probe_error=True
        self.assertEqual(self.dispatch.work(max_wall_seconds=120)['status'],'UNKNOWN');self.assertEqual(len(self.transport.commands),1)
    def test_work_idle_wait_default_preserves_immediate_exit(self):
        clock=FakeClock()
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_wall_seconds=7200)
        self.assertEqual(result['status'],'NO_READY_JOB');self.assertEqual(clock.sleeps,[])
        self.assertFalse(self.transport.commands)
    def test_work_idle_wait_launches_late_highest_roi_job(self):
        high=copy.deepcopy(self.job);high['id']='family.v2.attempt01';high['launch']['container']='okc-proj-two'
        high['estimate']['direct']['scenario_score_us']=100
        def arrive(elapsed):
            if not self.store.snapshot()['jobs']:
                self.register();self.register(high)
        clock=FakeClock(arrive);self.transport.complete_on_launch=True
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_jobs=1,max_wall_seconds=120,poll_seconds=2,idle_wait_seconds=10)
        self.assertEqual(result['status'],'MAX_JOBS');self.assertEqual(result['collected'][0]['job_id'],high['id'])
        self.assertEqual(clock.sleeps,[2]);self.assertEqual(len(self.transport.commands),1)
        self.assertEqual(self.store.get(self.job['id'])['state'],'READY')
        self.assertEqual(self.store.get(high['id'])['state'],'VALIDATING')
    def test_work_idle_wait_timeout_sleeps_once_without_busy_spin(self):
        clock=FakeClock()
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_wall_seconds=120,poll_seconds=60,idle_wait_seconds=5)
        self.assertEqual(result['status'],'NO_READY_JOB');self.assertEqual(clock.sleeps,[5])
        self.assertEqual(clock.elapsed,5);self.assertFalse(self.transport.commands)
    def test_work_idle_wait_never_oversleeps_wall_deadline(self):
        clock=FakeClock()
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_wall_seconds=3,poll_seconds=60,idle_wait_seconds=10)
        self.assertEqual(result['status'],'WALL_BUDGET');self.assertEqual(clock.sleeps,[3])
        self.assertFalse(self.transport.commands)
    def test_work_idle_wait_does_not_launch_after_expired_idle_deadline(self):
        def arrive_after_deadline(elapsed):
            clock.elapsed+=2;self.register()
        clock=FakeClock(arrive_after_deadline)
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_wall_seconds=120,poll_seconds=60,idle_wait_seconds=5)
        self.assertEqual(result['status'],'NO_READY_JOB');self.assertEqual(clock.sleeps,[5])
        self.assertEqual(self.store.get(self.job['id'])['state'],'READY');self.assertFalse(self.transport.commands)
    def test_work_idle_window_resets_after_actual_job_progress(self):
        second=copy.deepcopy(self.job);second['id']='family.v2.attempt01';second['launch']['container']='okc-proj-two'
        arrived=[]
        def arrive(elapsed):
            if elapsed>=3 and not arrived:
                self.register();arrived.append(self.job['id'])
            elif elapsed>=6 and len(arrived)==1:
                self.transport.observation=idle();self.register(second);arrived.append(second['id'])
        clock=FakeClock(arrive);self.transport.complete_on_launch=True
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_jobs=2,max_wall_seconds=120,poll_seconds=1,idle_wait_seconds=4)
        self.assertEqual(result['status'],'MAX_JOBS');self.assertEqual([j['job_id'] for j in result['collected']],arrived)
        self.assertEqual(len(self.transport.commands),2);self.assertEqual(clock.elapsed,6)
    def test_work_idle_wait_does_not_duplicate_other_controller_guard(self):
        def arrive(elapsed):
            if not self.transport.commands:
                self.register();self.dispatch.launch_next(owner='other-controller')
        clock=FakeClock(arrive)
        with mock.patch('proj_rsi.scheduling.dispatch.time.monotonic',side_effect=clock.monotonic),mock.patch('proj_rsi.scheduling.dispatch.time.sleep',side_effect=clock.sleep):
            result=self.dispatch.work(max_wall_seconds=120,poll_seconds=2,idle_wait_seconds=10)
        self.assertEqual(result['status'],'OTHER_CONTROLLER');self.assertEqual(len(self.transport.commands),1)
        self.assertEqual(self.store.get(self.job['id'])['state'],'RUNNING')
        self.assertEqual(self.transport.observation['compute_pids'],[4243])
    def test_work_idle_wait_bounds(self):
        for bound in (-1,float('nan'),float('inf'),True,86401,'10'):
            with self.subTest(bound=bound),self.assertRaisesRegex(ValueError,'idle wait'):
                self.dispatch.work(idle_wait_seconds=bound)
        self.assertFalse(self.transport.commands)
    def test_cli_idle_wait_forwarding_and_negative_rejection(self):
        from . import cli
        import io
        with mock.patch.object(cli,'QueueStore'),mock.patch('proj_rsi.scheduling.dispatch.GPUDispatcher') as constructor,mock.patch('sys.stdout',io.StringIO()):
            constructor.return_value.work.return_value={'status':'NO_READY_JOB'}
            cli.main(['work','--idle-wait-seconds','600','--max-wall-seconds','7200'])
            self.assertEqual(constructor.return_value.work.call_args.kwargs['idle_wait_seconds'],600)
            cli.main(['work'])
            self.assertEqual(constructor.return_value.work.call_args.kwargs['idle_wait_seconds'],0)
            before=constructor.return_value.work.call_count
            with mock.patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
                cli.main(['work','--idle-wait-seconds','-1'])
            self.assertEqual(constructor.return_value.work.call_count,before)
    def ncu_job(self):
        p=self.root/'proj_rsi/tools/profile.py';p.write_text('print(1)')
        self.job['files']['proj_rsi/tools/profile.py']=digest(p.read_bytes())
        self.job['kind']='profile';self.job['launch']['profile']=True
        self.job['estimate']['direct'].update(scenario_score_us=0,p_gain=0)
        self.job['launch']['argv']=['ncu','--clock-control','none','--launch-count','3','python3','/okc/proj_rsi/tools/profile.py']
    def test_ncu_literal_profile_allowed(self):
        self.ncu_job();self.assertEqual(self.launch()['state'],'RUNNING')
        self.assertIn('ncu',self.transport.commands[0]);self.assertIn('PROJ_GPU_PROFILE=1',self.transport.commands[0])
    def test_ncu_requires_flag_kind_python_and_pinned_driver(self):
        self.ncu_job()
        for change in ('flag','kind','python','driver','shell','clocks'):
            job=copy.deepcopy(self.job)
            if change=='flag':job['launch']['profile']=False
            if change=='kind':job['kind']='optimization'
            if change=='python':job['launch']['argv']=['ncu','/okc/proj_rsi/tools/profile.py']
            if change=='driver':job['launch']['argv'][-1]='/okc/proj_rsi/tools/unreviewed.py'
            if change=='shell':job['launch']['argv']=['ncu','--clock-control','none','bash','-c','code','python3','/okc/proj_rsi/tools/profile.py']
            if change=='clocks':job['launch']['argv'][2]='base'
            with self.assertRaises(ValueError):self.dispatch._validate(job)
    def test_unsafe_paths_rejected(self):
        for p in ('../x','/x','a/../x','a//x','a/./x','-x','a\\x'):
            with self.assertRaises(ValueError):_relative(p)

if __name__=='__main__':unittest.main()
