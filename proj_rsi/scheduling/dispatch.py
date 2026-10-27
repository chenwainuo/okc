"""Root-controller GPU dispatch; construction and tests perform no remote actions.

Injected transports implement upload(host, local_root, remote_root, paths),
hashes(host, remote_root, paths), probe(host, handles), launch(host, command),
inventory(host, remote_root, paths), download(host, remote_root, paths, local_root).
runner(host) verifies the installed host runner's sha256 and mode.
All paths passed to transfer/hash methods are repository-relative. Transport errors
are observations, never evidence that an already requested launch has stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time
from .model import ncu_target, launch_tools

HOST = 'root@107.170.6.60'
REMOTE_REPO = '/root/okc'


def _relative(value):
    if not isinstance(value, str) or not value or any(ord(c)<32 or ord(c)==127 for c in value):
        raise ValueError('invalid repository-relative path')
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('unsafe repository-relative path: ' + value)
    if value.startswith('-') or '\\' in value:
        raise ValueError('unsafe transfer path: ' + value)
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# These scripts are data for an SSH python process. No shell interpolates job data.
_INVENTORY = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); result={}; missing=[]; allow_missing=len(sys.argv)>3
for rel in json.loads(sys.argv[2]):
 p=root/rel
 if not p.exists() and allow_missing: missing.append(rel); continue
 if p.resolve()!=p or not p.is_relative_to(root): raise RuntimeError('unsafe remote path')
 paths=sorted(p.rglob('*')) if p.is_dir() else [p]
 for f in paths:
  if f.is_symlink(): raise RuntimeError('remote symlink')
  if not f.is_file():
   if f.is_dir(): continue
   raise RuntimeError('missing remote artifact: '+str(f))
  if f.resolve()!=f or not f.is_relative_to(root): raise RuntimeError('unsafe artifact')
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''): h.update(block)
  result[str(f.relative_to(root))]=h.hexdigest()
print(json.dumps({'files':result,'missing':missing} if allow_missing else result,sort_keys=True))
'''

_PROBE = r'''
import json,os,pathlib,re,subprocess,sys
h=json.loads(sys.argv[1]); name=h['container']
def run(argv):
 p=subprocess.run(argv,text=True,capture_output=True,timeout=15)
 if p.returncode: raise RuntimeError('probe command failed: '+argv[0]+': '+p.stderr[-1000:])
 return p.stdout
containers=run(['docker','ps','--format','{{.Names}}']).splitlines()
p=subprocess.run(['docker','inspect',name],text=True,capture_output=True,timeout=15)
if p.returncode not in (0,1): raise RuntimeError('docker inspect unavailable')
# An inspect failure is absent only if the complete container inventory agrees.
all_names=run(['docker','ps','-a','--format','{{.Names}}']).splitlines()
exists=name in all_names
if bool(p.returncode==0)!=exists: raise RuntimeError('inconsistent container inventory')
compute=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
mem=run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'])
guards=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try: args=(proc/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 if any(a.endswith((b'/h100_trial.sh',b'/h100_sglang_trial.sh')) for a in args) and name.encode() in args: guards.append(int(proc.name))
log=pathlib.Path(h['host_log'])
text=log.read_text(errors='replace')[-2000000:] if log.is_file() else ''
matches=re.findall(r'^TRIAL_EXIT '+re.escape(name)+r' (\d+)\s*$',text,re.M)
print(json.dumps({'container_exists':exists,'containers':containers,'guard_pids':guards,
 'compute_pids':[int(x.strip()) for x in compute.splitlines() if x.strip()],
 'gpu_memory_mib':sum(int(x.strip()) for x in mem.splitlines() if x.strip()),
 'host_log_exists':log.exists(),'terminal_exit':int(matches[-1]) if matches else None,
 'cleanup_failed':('TRIAL_CLEANUP_FAILED '+name) in text,'host_log_tail':text[-4000:]}))
'''


class SSHTransport:
    """Bounded SSH/rsync adapter. Never called implicitly on import."""
    def _run(self, argv):
        p = subprocess.run(argv, text=True, capture_output=True, timeout=60)
        if p.returncode:
            raise RuntimeError('transport failed (%s): %s' % (p.returncode, p.stderr[-2000:]))
        return p.stdout

    def _ssh(self, host, command):
        return self._run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command])

    def _python(self, host, script, *args):
        return json.loads(self._ssh(host, shlex.join(['python3', '-c', script, *args])))

    def _transfer(self, host, local_root, remote_root, paths, upload):
        # --files-from selects exactly the pinned/inventoried names; no repo sweep.
        with tempfile.NamedTemporaryFile('w', encoding='utf-8') as manifest:
            manifest.write(''.join(_relative(p) + '\n' for p in paths)); manifest.flush()
            local = str(Path(local_root).resolve()) + '/'
            remote = host + ':' + remote_root.rstrip('/') + '/'
            argv = ['rsync', '-a', '--protect-args', '--relative', '--files-from=' + manifest.name,
                    '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10']
            self._run(argv + ([local, remote] if upload else [remote, local]))

    def upload(self, host, local_root, remote_root, paths):
        self._transfer(host, local_root, remote_root, paths, True)

    def download(self, host, remote_root, paths, local_root):
        self._transfer(host, local_root, remote_root, paths, False)

    def hashes(self, host, remote_root, paths):
        return self._python(host, _INVENTORY, remote_root, json.dumps(paths))

    def inventory(self, host, remote_root, paths):
        return self._python(host, _INVENTORY, remote_root, json.dumps(paths), 'allow_missing')

    def probe(self, host, handles):
        return self._python(host, _PROBE, json.dumps(handles))

    def runner(self, host, path='/root/okc_run.sh'):
        if path not in ('/root/okc_run.sh','/root/okc_run_sglang.sh'): raise ValueError('unapproved installed GPU runner path')
        script="import hashlib,json,os,stat,sys; p=sys.argv[1]; print(json.dumps({'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'mode':stat.S_IMODE(os.stat(p).st_mode),'symlink':os.path.islink(p)}))"
        return self._python(host,script,path)

    def launch(self, host, command):
        return self._ssh(host, command)


class GPUDispatcher:
    def __init__(self, store, repo_root, *, transport=None):
        self.store = store
        self.repo_root = Path(repo_root).resolve()
        self.transport = transport if transport is not None else SSHTransport()

    def _handles(self, job):
        for observation in reversed(job.get('observations', [])):
            observation = observation.get('observation', observation)
            if 'dispatch' in observation:
                return dict(observation['dispatch'])
        raise ValueError('no persisted dispatch handles for ' + job['id'])

    def _save(self, job_id, handles, **extra):
        observation = {'dispatch': dict(handles), **extra}
        self.store.attach(job_id, observation)
        return observation

    def _validate(self, job):
        job_id = job['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', job_id):
            raise ValueError('invalid job id')
        launch = job['launch']
        if launch.get('host') != HOST or launch.get('remote_repo') != REMOTE_REPO:
            raise ValueError('unsupported GPU host or remote repository')
        name = launch.get('container', '')
        if not re.fullmatch(r'okc-proj-[a-z0-9-]+', name) or len(name) > 128:
            raise ValueError('invalid named container')
        jobs = self.store.snapshot()['jobs']
        if isinstance(jobs, dict): jobs = jobs.values()
        if any(other['id'] != job_id and other.get('launch', {}).get('container') == name for other in jobs):
            raise ValueError('container name already belongs to another attempt')
        inner, outer = launch.get('inner_seconds'), launch.get('outer_seconds')
        if type(inner) is not int or type(outer) is not int or not (0 < inner <= self.store.policy.get('launch_limits', {}).get('max_inner_seconds', 3600) and inner + 20 < outer <= self.store.policy.get('launch_limits', {}).get('max_outer_seconds', 3720)):
            raise ValueError('invalid bounded guard deadlines')
        argv = launch.get('argv')
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or '\x00' in a or '\n' in a for a in argv) or not argv[0]:
            raise ValueError('invalid literal argument vector')
        python_names=('python3','python','/usr/bin/python3')
        if argv[0] in ('ncu','/usr/local/cuda/bin/ncu'):
            if launch.get('profile') is not True or job.get('kind') not in ('profile','diagnostic'):
                raise ValueError('NCU requires explicit profile diagnostic')
            i=ncu_target(argv)
            if i is None:raise ValueError('NCU options must identify actual Python target and preserve clocks')
            if i+1>=len(argv) or not argv[i+1].startswith('/okc/proj_rsi/'):
                raise ValueError('NCU driver must be under /okc/proj_rsi')
            driver=_relative(argv[i+1][len('/okc/'):])
            if driver not in job.get('files',{}):raise ValueError('NCU driver must be source-pinned')
        elif argv[0] not in python_names:raise ValueError('only literal Python or explicit NCU launch permitted')
        if type(launch.get('profile', False)) is not bool:
            raise ValueError('profile must be an explicit boolean')
        if launch.get('profile') is True and job.get('kind') not in ('profile','diagnostic'):
            raise ValueError('profiling capability only permitted for profile/diagnostic kind')
        env = launch.get('env', {})
        if not isinstance(env, dict): raise ValueError('invalid environment')
        for key, value in env.items():
            if not isinstance(key,str) or not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key) or key in ('LD_PRELOAD','BASH_ENV','ENV','PROJ_GPU_CONTAINER'):
                raise ValueError('reserved or invalid environment key')
            if not isinstance(value, str) or '\x00' in value or '\n' in value:
                raise ValueError('invalid environment value')
            if key=='PROJ_GPU_PROFILE' and not (launch.get('profile') is True and value=='1'):
                raise ValueError('profile capability requires explicit profile=true')
        pins = dict(job.get('files', {}))
        review = job.get('review', {})
        if review.get('decision') != 'GO' or not review.get('actor') or review.get('actor') == job.get('author'):
            raise ValueError('independent GO review required')
        guard_path,runner_pin,installed_runner=launch_tools(launch)
        if guard_path not in pins or runner_pin not in pins:raise ValueError('hardware guard and container runner must be source-pinned')
        review_path, review_hash = review.get('path'), review.get('sha256')
        if review_path in pins and pins[review_path] != review_hash:
            raise ValueError('conflicting review hash')
        pins[review_path] = review_hash
        if not job.get('files'): raise ValueError('source pins required')
        for rel, digest in pins.items():
            _relative(rel)
            if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('invalid source hash')
            p = self.repo_root / rel
            if p.resolve() != p or not p.is_file() or _sha(p) != digest:
                raise ValueError('local source/review hash mismatch: ' + rel)
        artifacts = launch.get('artifacts')
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError('exact artifact paths required')
        for value in artifacts:
            if not isinstance(value, str) or not value.startswith(REMOTE_REPO + '/'):
                raise ValueError('artifact outside remote repository')
            _relative(value[len(REMOTE_REPO) + 1:])
        return launch, pins

    @staticmethod
    def _clean(observation):
        required = ('container_exists', 'guard_pids', 'compute_pids', 'gpu_memory_mib', 'containers')
        if any(key not in observation for key in required): return False
        return (observation['container_exists'] is False and observation['guard_pids'] == []
                and observation['compute_pids'] == [] and observation['gpu_memory_mib'] == 0
                and observation['containers'] == [] and observation.get('cleanup_failed') is False)

    @staticmethod
    def _command(launch, handles):
        argv = ['env'] + [key + '=' + value for key, value in sorted((k,v) for k,v in launch.get('env', {}).items() if k!='PROJ_GPU_PROFILE')] + launch['argv']
        guard_path,_,_=launch_tools(launch)
        guard = ['bash', REMOTE_REPO + '/' + guard_path, launch['container'],
                 str(launch['outer_seconds']), str(launch['inner_seconds']), *argv]
        guard = ['env', 'PROJ_GPU_PROFILE=' + ('1' if launch.get('profile') is True else '0'), *guard]
        # The background job is exactly nohup, never a validation AND-list.
        script = 'set -eu\nmkdir -p ' + shlex.quote(REMOTE_REPO + '/logs/scheduler') + '\n'
        script += 'nohup ' + shlex.join(guard) + ' >' + shlex.quote(handles['host_log']) + ' 2>&1 </dev/null &\n'
        script += 'guard_pid=$!\nprintf \'%s\\n\' "$guard_pid"\n'
        return shlex.join(['bash', '-c', script])

    def launch_next(self, owner='root', *, deadline_monotonic=None):
        job = self.store.claim('gpu', owner)
        if job is None: return None
        job_id = job['id']
        handles = None
        launch_requested = False
        try:
            launch, pins = self._validate(job)
            handles = {'job_id': job_id, 'controller_owner': owner, 'host': HOST,
                       'remote_repo': REMOTE_REPO, 'container': launch['container'], 'guard_pid': None,
                       'host_log': REMOTE_REPO + '/logs/scheduler/' + job_id + '.host.log',
                       'artifacts': list(launch['artifacts']), 'phase': 'CLAIMED', 'claimed_at': time.time(),
                       'source_pins': pins}
            self._save(job_id, handles)
            paths = sorted(pins)
            self.transport.upload(HOST, self.repo_root, REMOTE_REPO, paths)
            actual = self.transport.hashes(HOST, REMOTE_REPO, paths)
            if actual != pins: raise ValueError('remote source/review hashes differ')
            _,runner_pin,installed_runner=launch_tools(launch)
            runner=self.transport.runner(HOST) if installed_runner=='/root/okc_run.sh' else self.transport.runner(HOST,installed_runner)
            if runner.get('sha256')!=pins[runner_pin] or runner.get('mode')!=0o755 or runner.get('symlink',False):
                raise ValueError('installed '+installed_runner+' differs from pinned runner or mode755; root manual installation required')
            self.store.attach(job_id,{'dispatch':handles,'installed_runner':runner})
            # Check again after transfer; a local edit during upload invalidates dispatch.
            self._validate(self.store.get(job_id))
            observation = self.transport.probe(HOST, handles)
            self.store.heartbeat(job_id, {'dispatch': handles, 'probe': observation})
            if not self._clean(observation) or observation.get('terminal_exit') is not None or observation.get('host_log_exists',False):
                raise ValueError('GPU/container/guard busy or container/log identity collision')
            if deadline_monotonic is not None and launch['outer_seconds'] > deadline_monotonic-time.monotonic():
                raise ValueError('worker wall budget insufficient for complete guard reservation')
            handles['phase'] = 'LAUNCH_REQUESTED'; handles['launch_requested_at'] = time.time()
            self._save(job_id, handles)
            launch_requested = True
            output = self.transport.launch(HOST, self._command(launch, handles)).strip()
            if not re.fullmatch(r'[1-9][0-9]*', output):
                raise RuntimeError('launch response was not an authoritative PID')
            handles.update(guard_pid=int(output), phase='RUNNING')
            self.store.heartbeat(job_id, {'dispatch': handles, 'launch_pid': int(output)})
        except Exception as error:
            if launch_requested:
                # The guard may already exist. Retain RUNNING and the exact identity.
                handles['phase'] = 'LAUNCH_UNKNOWN'
                self._save(job_id, handles, error=str(error), status='UNKNOWN')
            else:
                self.store.attach(job_id, {'error': str(error), 'dispatch': handles} if handles else {'error': str(error)})
                self.store.release_unlaunched(job_id, actor=owner, reason='dispatch blocked: ' + str(error))
        return self.store.get(job_id)

    def poll(self, job_id):
        job = self.store.get(job_id)
        handles = self._handles(job)
        try:
            probe = self.transport.probe(handles['host'], handles)
            if handles.get('guard_pid') is None and len(probe.get('guard_pids',[]))==1:
                handles.update(guard_pid=probe['guard_pids'][0],phase='RUNNING')
            observation = {'status': 'OBSERVED', 'probe': probe, 'dispatch': handles}
        except Exception as error:
            observation = {'status': 'UNKNOWN', 'error': str(error), 'dispatch': handles}
        self.store.heartbeat(job_id, observation)
        return observation

    def reconcile(self, job_id):
        # No lease-based launch, kill, release, or state inference.
        return self.poll(job_id)

    def collect(self, job_id):
        observation = self.poll(job_id)
        probe = observation.get('probe', {})
        if observation['status'] != 'OBSERVED' or type(probe.get('terminal_exit')) is not int or not self._clean(probe):
            return {'status': 'AWAITING_TERMINAL_CLEANUP', 'observation': observation}
        job = self.store.get(job_id); handles = self._handles(job)
        paths = [p[len(REMOTE_REPO) + 1:] for p in handles['artifacts']]
        paths.append(handles['host_log'][len(REMOTE_REPO) + 1:])
        paths = sorted(set(_relative(p) for p in paths))
        destination = self.repo_root / 'results/proj-rsi/scheduler/artifacts' / job_id / ('collection-'+str(time.time_ns()))
        if destination.resolve()!=destination:raise ValueError('unsafe collection destination')
        destination.mkdir(parents=True, exist_ok=False)
        try:
            remote_inventory = self.transport.inventory(HOST, REMOTE_REPO, paths)
            expected=remote_inventory.get('files',remote_inventory)
            missing=remote_inventory.get('missing',[]) if 'files' in remote_inventory else []
            if not isinstance(missing,list) or any(p not in paths for p in missing):raise ValueError('invalid missing-artifact inventory')
            if not expected: raise ValueError('empty artifact inventory')
            for rel, digest in expected.items():
                _relative(rel)
                if not any(rel == p or rel.startswith(p + '/') for p in paths):
                    raise ValueError('unexpected remote artifact')
                if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                    raise ValueError('invalid artifact digest')
            self.transport.download(HOST, REMOTE_REPO, sorted(expected), destination)
            for rel, digest in expected.items():
                p = destination / rel
                if p.resolve() != p or not p.is_file() or _sha(p) != digest:
                    raise ValueError('collected artifact hash mismatch: ' + rel)
            inventory = {'job_id': job_id, 'terminal_exit': probe['terminal_exit'], 'files': expected, 'missing':missing,
                         'dispatch': handles, 'cleanup_observation': probe, 'collected_at': time.time(),
                         'validation': 'ROOT_VALIDATION_REQUIRED_NO_PERFORMANCE_ADMISSION'}
            inventory_path = destination / 'inventory.json'
            inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
            evidence = {'inventory': str(inventory_path.relative_to(self.repo_root)), 'sha256': _sha(inventory_path),
                        'terminal_exit': probe['terminal_exit'], 'cleanup_verified': True, 'cleanup_observation':probe}
            self.store.attach(job_id, {'collection': evidence, 'dispatch': handles})
            # A second observation prevents admitting stale cleanup before collection.
            after = self.poll(job_id)
            if after['status'] != 'OBSERVED' or not self._clean(after['probe']) or after['probe'].get('terminal_exit') != probe['terminal_exit']:
                return {'status': 'AWAITING_TERMINAL_CLEANUP', 'observation': after}
            if self.store.get(job_id)['state'] == 'RUNNING':
                self.store.transition(job_id, 'VALIDATING', actor=handles['controller_owner'],
                                      reason='raw artifacts collected and actual terminal cleanup verified', evidence=evidence)
            return {'status': 'AWAITING_ROOT_VALIDATION', **evidence}
        except Exception as error:
            self.store.attach(job_id, {'collection_error': str(error), 'dispatch': handles})
            return {'status': 'COLLECTION_FAILED', 'error': str(error)}

    def work(self, owner='root', max_jobs=3, max_wall_seconds=3600, poll_seconds=10, idle_wait_seconds=0):
        """Bounded cooperative worker; never stops a guard to satisfy its own budget."""
        import math
        if type(max_jobs) is not int or not 0<max_jobs<=1000:
            raise ValueError('max_jobs must be integer 1..1000')
        if isinstance(max_wall_seconds,bool) or not isinstance(max_wall_seconds,(int,float)) or not math.isfinite(max_wall_seconds) or not 0<max_wall_seconds<=86400:
            raise ValueError('worker wall budget must be finite 0..86400 seconds')
        if isinstance(poll_seconds,bool) or not isinstance(poll_seconds,(int,float)) or not math.isfinite(poll_seconds) or not 0<poll_seconds<=60:
            raise ValueError('poll interval must be finite 0..60 seconds')
        if isinstance(idle_wait_seconds,bool) or not isinstance(idle_wait_seconds,(int,float)) or not math.isfinite(idle_wait_seconds) or not 0<=idle_wait_seconds<=86400:
            raise ValueError('idle wait must be finite 0..86400 seconds')
        deadline=time.monotonic()+max_wall_seconds
        idle_deadline=None
        completed=[]
        while len(completed)<max_jobs:
            now=time.monotonic()
            remaining=deadline-now
            if remaining<=0:return {'status':'WALL_BUDGET','collected':completed}
            if idle_deadline is not None and now>=idle_deadline:
                return {'status':'NO_READY_JOB','collected':completed}
            jobs=self.store.snapshot()['jobs']
            jobs=list(jobs.values()) if isinstance(jobs,dict) else jobs
            running=[j for j in jobs if j['resource']=='gpu' and j['state']=='RUNNING']
            if len(running)>1:return {'status':'ACTIVE_CAP_INVARIANT','collected':completed}
            if running:
                job=running[0]
                try:handles=self._handles(job)
                except ValueError:return {'status':'UNKNOWN_EXISTING_EXECUTION','job_id':job['id'],'collected':completed}
                if handles['controller_owner']!=owner:return {'status':'OTHER_CONTROLLER','job_id':job['id'],'collected':completed}
                idle_deadline=None
            else:
                ranked=self.store.rank('gpu')
                # rank() returns full job copies from one locked WAL read.
                # Use that coherent view as a hint; claim() rechecks atomically.
                ready=[row for row in ranked if row['state']=='READY']
                if not ready:
                    if idle_wait_seconds==0:return {'status':'NO_READY_JOB','collected':completed}
                    now=time.monotonic()
                    if idle_deadline is None:idle_deadline=min(deadline,now+idle_wait_seconds)
                    wait=min(poll_seconds,idle_deadline-now,deadline-now)
                    if wait<=0:
                        return {'status':'WALL_BUDGET' if now>=deadline else 'NO_READY_JOB','collected':completed}
                    # No reservation/guard is held while idle. Reread both
                    # active ownership and current ROI ranking after every wake.
                    time.sleep(wait)
                    continue
                candidate=ready[0]
                if candidate['launch']['outer_seconds']>remaining:
                    return {'status':'INSUFFICIENT_RESERVATION_BUDGET','collected':completed}
                job=self.launch_next(owner,deadline_monotonic=deadline)
                if job is None:return {'status':'NO_CLAIM','collected':completed}
                if job['state']!='RUNNING':return {'status':'HELD','job_id':job['id'],'collected':completed}
                idle_deadline=None
            while True:
                if time.monotonic()>=deadline:return {'status':'WALL_BUDGET','running':job['id'],'collected':completed}
                observation=self.poll(job['id'])
                if observation['status']=='UNKNOWN':return {'status':'UNKNOWN','job_id':job['id'],'collected':completed}
                probe=observation['probe']
                if probe.get('terminal_exit') is not None:
                    collection=self.collect(job['id'])
                    if collection['status']!='AWAITING_ROOT_VALIDATION':
                        return {'status':collection['status'],'job_id':job['id'],'collected':completed}
                    completed.append({'job_id':job['id'],**collection})
                    idle_deadline=None
                    break
                if not probe.get('guard_pids') and not probe.get('container_exists'):
                    return {'status':'UNKNOWN_EXECUTION','job_id':job['id'],'collected':completed}
                time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))
        return {'status':'MAX_JOBS','collected':completed}
