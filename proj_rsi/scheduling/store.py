"""Crash-recoverable local priority queue. WAL is authoritative, checkpoint is a cache.

snapshot: {version: 1, sequence: int, jobs: {job_id: job}}. Every fsynced WAL
line contains a checksummed complete next snapshot plus the audit event. A stable
.lock inode serializes processes. No lease expiry implies process death.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .model import load_policy, score, validate_job

STATES = {'BACKLOG', 'READY', 'RUNNING', 'VALIDATING', 'DONE', 'HELD', 'CANCELLED'}
DISPOSITIONS = {'CPU_PASS', 'VALID_GAIN', 'VALID_NO_GAIN', 'VALID_REGRESSION',
                'INVALID_SETUP', 'INVALID_CORRECTNESS', 'INVALID_TIMEOUT', 'INVALID_INTERRUPTED'}
ACTIVE = {'RUNNING', 'VALIDATING'}
TERMINAL = {'DONE', 'CANCELLED'}
TRANSITIONS = {
    'BACKLOG': {'READY', 'HELD', 'CANCELLED'}, 'READY': {'BACKLOG', 'RUNNING', 'HELD', 'CANCELLED'},
    'RUNNING': {'VALIDATING'}, 'VALIDATING': {'DONE', 'HELD'},
    'HELD': {'BACKLOG', 'CANCELLED'}, 'DONE': set(), 'CANCELLED': set()}
PROTECTED = {'files', 'review', 'launch', 'resource', 'write_paths', 'depends_on', 'id', 'created_at'}


class QueueError(ValueError):
    pass


class CorruptWAL(QueueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, content):
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with open(temporary, 'xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class QueueStore:
    def __init__(self, directory, *, repo_root=None, policy=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.policy = deepcopy(policy if policy is not None else load_policy())
        for key, hard in (('ready_capacity', {'cpu':20,'gpu':10}), ('active_capacity', {'cpu':10,'gpu':1})):
            for resource, ceiling in hard.items():
                value = self.policy.get(key, {}).get(resource, ceiling)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                    raise QueueError('%s.%s must be in [1,%s]' % (key, resource, ceiling))
        try:
            score({}, policy=self.policy)  # Validates supplied policy without filesystem/GPU work.
        except ValueError as error:
            raise QueueError('invalid scheduling policy: ' + str(error)) from error
        self.lock_path = self.directory / '.lock'
        self.wal_path = self.directory / 'events.jsonl'
        self.checkpoint_path = self.directory / 'checkpoint.json'
        # Never unlink the lock inode, including during restart/recovery.
        self.lock_path.touch(exist_ok=True)

    @contextmanager
    def _locked(self):
        with open(self.lock_path, 'a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        state = {'version': 1, 'sequence': 0, 'jobs': {}}
        if not self.wal_path.exists():
            return state
        raw = self.wal_path.read_bytes()
        complete = raw.rfind(b'\n') + 1
        if complete != len(raw):
            tail = raw[complete:]
            digest = hashlib.sha256(tail).hexdigest()
            evidence = self.directory / ('torn-final-' + digest + '.bin')
            if not evidence.exists():
                _atomic(evidence, tail)
            # Preserve first, then truncate only the incomplete final record.
            with open(self.wal_path, 'r+b') as handle:
                handle.truncate(complete)
                handle.flush()
                os.fsync(handle.fileno())
            raw = raw[:complete]
        for number, line in enumerate(raw.splitlines(), 1):
            try:
                entry = json.loads(line)
                payload = entry['payload']
                if hashlib.sha256(_json(payload)).hexdigest() != entry['sha256']:
                    raise ValueError('checksum mismatch')
                if payload['sequence'] != state['sequence'] + 1:
                    raise ValueError('non-contiguous sequence')
                candidate = payload['snapshot']
                if candidate['version'] != 1 or candidate['sequence'] != payload['sequence'] or not isinstance(candidate['jobs'], dict):
                    raise ValueError('invalid snapshot envelope')
                state = candidate
            except (ValueError, TypeError, KeyError) as error:
                raise CorruptWAL('WAL record %d is corrupt; execution refused: %s' % (number, error)) from error
        return state

    def _commit(self, state, action, *, actor='controller', reason='', data=None):
        state['sequence'] += 1
        payload = {'sequence': state['sequence'], 'timestamp': time.time(), 'action': action,
                   'actor': actor, 'reason': reason, 'data': data, 'snapshot': state}
        record = _json({'payload': payload, 'sha256': hashlib.sha256(_json(payload)).hexdigest()}) + b'\n'
        new_file = not self.wal_path.exists()
        with open(self.wal_path, 'ab') as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        if new_file:
            _sync_directory(self.directory)
        _atomic(self.checkpoint_path, _json(state) + b'\n')

    def snapshot(self):
        with self._locked():
            state = self._read()
            _atomic(self.checkpoint_path, _json(state) + b'\n')
            return deepcopy(state)

    def get(self, id):
        snapshot = self.snapshot()
        if id not in snapshot['jobs']:
            raise QueueError('unknown job: ' + id)
        return snapshot['jobs'][id]

    def _rank(self, state, resource):
        if resource not in ('cpu', 'gpu'):
            raise QueueError('resource must be cpu or gpu')
        now = time.time()
        ranked = []
        for job in state['jobs'].values():
            if job['resource'] != resource:
                continue
            estimate = score(job, resource=resource, now=now, policy=self.policy)
            ranked.append({**deepcopy(job), 'score': estimate})
        return sorted(ranked, key=lambda job: (-(job['score'].get('priority') or 0) if job['score'].get('priority') is not None else float('inf'), job['created_at'], job['id']))

    def rank(self, resource):
        with self._locked():
            return self._rank(self._read(), resource)

    def _file(self, name):
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise QueueError('unsafe repository file path')
        path = (self.repo_root / name).resolve()
        if not path.is_relative_to(self.repo_root) or not path.is_file():
            raise QueueError('missing/outside repository file: ' + name)
        return path

    def _eligibility(self, state, job):
        errors = list(validate_job(job, for_admission=job['resource'] == 'gpu'))
        result = score(job, resource=job['resource'], policy=self.policy)
        if result.get('priority') is None:
            errors += result.get('errors') or ['unknown estimate: explicit priors required']
        for dependency in job.get('depends_on', []) if isinstance(job.get('depends_on', []), list) else []:
            if not isinstance(dependency, (str, dict)):
                errors.append('malformed dependency'); continue
            name = dependency if isinstance(dependency, str) else dependency.get('id')
            parent = state['jobs'].get(name)
            accepted = dependency.get('accepted_dispositions') if isinstance(dependency, dict) else None
            if accepted is None and isinstance(dependency, dict):
                accepted = dependency.get('dispositions')
            if accepted is not None and (not isinstance(accepted, list) or not accepted or any(value not in DISPOSITIONS for value in accepted)):
                errors.append('dependency accepted dispositions must be a nonempty explicit disposition list'); continue
            if not parent or parent['state'] != 'DONE' or not parent.get('disposition') or (parent['disposition'] not in accepted if accepted is not None else parent['disposition'].startswith('INVALID')):
                errors.append('dependency unresolved/invalid: ' + str(name))
        if job['resource'] == 'gpu':
            try:
                if not job.get('files'):
                    raise QueueError('GPU job needs pinned files')
                for name, expected in job['files'].items():
                    if hashlib.sha256(self._file(name).read_bytes()).hexdigest() != expected:
                        raise QueueError('source hash drift: ' + name)
                review = job.get('review') or {}
                if review.get('decision') != 'GO' or not review.get('actor') or review.get('actor') == job.get('author'):
                    raise QueueError('GPU job needs independent review GO')
                if hashlib.sha256(self._file(review.get('path')).read_bytes()).hexdigest() != review.get('sha256'):
                    raise QueueError('review hash drift')
            except (OSError, QueueError, TypeError) as error:
                errors.append(str(error))
        return list(dict.fromkeys(errors))

    def _rebalance(self, state, resource):
        eligible = []
        changes = []
        for job in self._rank(state, resource):
            actual = state['jobs'][job['id']]
            if actual['state'] not in {'READY', 'BACKLOG'}:
                continue
            errors = self._eligibility(state, actual)
            if errors:
                # Dependency waits are backlog; invalid/unknown descriptors are held.
                if all(error.startswith('dependency unresolved/invalid:') for error in errors):
                    target = 'BACKLOG'
                else:
                    target = 'HELD'
                actual['eligibility_errors'] = errors
                if actual['state'] != target:
                    changes.append({'id': actual['id'], 'from': actual['state'], 'to': target, 'reasons': errors})
                    actual['state'] = target
            else:
                actual.pop('eligibility_errors', None)
                eligible.append(actual)
        limit = int(self.policy.get('ready_capacity', {}).get(resource, 20 if resource == 'cpu' else 10))
        for index, job in enumerate(eligible):
            target = 'READY' if index < limit else 'BACKLOG'
            if job['state'] != target:
                changes.append({'id': job['id'], 'from': job['state'], 'to': target, 'reason': 'ROI ready-capacity rebalance'})
                job['state'] = target
        return changes

    def rebalance(self, resource):
        with self._locked():
            state = self._read()
            changes = self._rebalance(state, resource)
            self._commit(state, 'rebalance', data=changes)
            return deepcopy(changes)

    def register(self, job):
        job = deepcopy(job)
        if not isinstance(job.get('id'), str) or not job['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in job['id']):
            raise QueueError('invalid job id')
        if job.get('resource') not in ('cpu', 'gpu') or job.get('state', 'BACKLOG') not in STATES:
            raise QueueError('invalid resource/state')
        job.setdefault('state', 'BACKLOG')
        if job['state'] in ACTIVE | {'READY'}:
            raise QueueError('register BACKLOG/HELD or historical terminal jobs only')
        job.setdefault('created_at', time.time())
        if not isinstance(job['created_at'], (int, float)):
            raise QueueError('created_at must be numeric')
        job.setdefault('depends_on', [])
        job.setdefault('write_paths', [])
        job.setdefault('observations', [])
        with self._locked():
            state = self._read()
            if job['id'] in state['jobs']:
                raise QueueError('job id already exists; use new attempt id')
            state['jobs'][job['id']] = job
            changes = self._rebalance(state, job['resource'])
            self._commit(state, 'register', actor=job.get('author', 'controller'), data={'id': job['id'], 'rebalance': changes})
            return deepcopy(state['jobs'][job['id']])

    def _write_conflict(self, left, right):
        for a in left.get('write_paths', []):
            pa = (self.repo_root / a).resolve()
            for b in right.get('write_paths', []):
                pb = (self.repo_root / b).resolve()
                if pa == pb or pa in pb.parents or pb in pa.parents:
                    return True
        return False

    def claim(self, resource, owner):
        if not isinstance(owner, str) or not owner:
            raise QueueError('owner is required')
        with self._locked():
            state = self._read()
            changes = self._rebalance(state, resource)
            active = [j for j in state['jobs'].values() if j['resource'] == resource and j['state'] in ({'RUNNING'} if resource == 'gpu' else ACTIVE)]
            cap = int(self.policy.get('active_capacity', {}).get(resource, 10 if resource == 'cpu' else 1))
            selected = None
            if len(active) < cap:
                for ranked in self._rank(state, resource):
                    job = state['jobs'][ranked['id']]
                    if job['state'] != 'READY':
                        continue
                    if resource == 'cpu' and any(self._write_conflict(job, current) for current in active):
                        continue
                    job['state'] = 'RUNNING'
                    job['owner'] = owner
                    job['lease'] = {'id': uuid.uuid4().hex, 'claimed_at': time.time(), 'heartbeat_at': time.time(), 'expiry_is_not_death': True}
                    job['selection'] = {'score': ranked['score'], 'sequence': state['sequence'] + 1}
                    selected = job
                    break
            changes += self._rebalance(state, resource)
            self._commit(state, 'claim', actor=owner, data={'id': selected['id'] if selected else None, 'rebalance': changes})
            return deepcopy(selected)

    def update(self, id, changes, *, actor, reason):
        if not isinstance(changes, dict) or {'state', 'id', 'observations', 'lease', 'owner', 'selection', 'disposition'} & changes.keys():
            raise QueueError('use guarded transitions/observations; id and execution state are immutable')
        with self._locked():
            state = self._read()
            job = state['jobs'][id]
            if job['state'] in TERMINAL:
                raise QueueError('terminal measurement immutable; create new attempt')
            if job['state'] in ACTIVE and PROTECTED & changes.keys():
                raise QueueError('cannot mutate active executable/review/ownership inputs')
            if job['resource'] == 'gpu' and {'files','launch'} & changes.keys():
                if not isinstance(changes.get('review'), dict) or changes['review'].get('sha256') == (job.get('review') or {}).get('sha256'):
                    # Old GO did not review this executable. Preserve it in WAL history.
                    job.pop('review', None)
                    changes = {key:value for key,value in changes.items() if key != 'review'}
            job.update(deepcopy(changes))
            rebalancing = self._rebalance(state, job['resource'])
            self._commit(state, 'update', actor=actor, reason=reason, data={'id': id, 'changes': changes, 'rebalance':rebalancing})
            return deepcopy(job)

    def attach(self, id, observation):
        if not isinstance(observation, dict):
            raise QueueError('observation must be an object')
        with self._locked():
            state = self._read()
            state['jobs'][id].setdefault('observations', []).append({'timestamp': time.time(), **deepcopy(observation)})
            self._commit(state, 'attach', data={'id': id, 'observation': observation})
            return deepcopy(state['jobs'][id])

    def heartbeat(self, id, observation):
        if not isinstance(observation, dict):
            raise QueueError('observation must be an object')
        with self._locked():
            state = self._read()
            job = state['jobs'][id]
            job.setdefault('observations', []).append({'timestamp': time.time(), **deepcopy(observation)})
            if 'lease' in job:
                job['lease']['heartbeat_at'] = time.time()
            self._commit(state, 'heartbeat', data={'id': id, 'observation': observation})
            return deepcopy(job)

    def transition(self, id, state, *, actor, reason, evidence=None):
        with self._locked():
            snapshot = self._read()
            job = snapshot['jobs'][id]
            prelaunch_hold = job['state'] == 'RUNNING' and state == 'HELD' and isinstance(evidence, dict) and evidence.get('unlaunched') is True
            if prelaunch_hold:
                if actor != job.get('owner'):
                    raise QueueError('only claiming controller can release unlaunched job')
                for observation in job.get('observations', []):
                    handles = observation.get('dispatch')
                    if isinstance(handles, dict) and handles.get('phase') not in (None, 'CLAIMED'):
                        raise QueueError('launch requested/unknown: cannot release running lease')
            if state not in TRANSITIONS[job['state']] and not prelaunch_hold:
                raise QueueError('forbidden transition %s -> %s' % (job['state'], state))
            if state == 'RUNNING':
                raise QueueError('RUNNING requires atomic claim')
            if state == 'READY':
                raise QueueError('READY requires ROI rebalance')
            if job['resource'] == 'gpu' and job['state'] == 'RUNNING' and state == 'VALIDATING':
                self._require_cleanup(evidence)
            if state == 'DONE':
                disposition = evidence.get('disposition') if isinstance(evidence, dict) else None
                if disposition not in DISPOSITIONS or not evidence or actor != 'root' or not reason:
                    raise QueueError('DONE needs root controller validator disposition/evidence/reason')
                job['disposition'] = disposition
                job['validation'] = {'actor': actor, 'reason': reason, 'evidence': deepcopy(evidence), 'timestamp': time.time()}
            old = job['state']
            job['state'] = state
            job.setdefault('observations', []).append({'timestamp': time.time(), 'transition': {'from': old, 'to': state, 'actor': actor, 'reason': reason, 'evidence': deepcopy(evidence)}})
            rebalancing = self._rebalance(snapshot, job['resource'])
            self._commit(snapshot, 'transition', actor=actor, reason=reason, data={'id': id, 'from': old, 'to': state, 'evidence': evidence, 'rebalance':rebalancing})
            return deepcopy(job)

    @staticmethod
    def _require_cleanup(evidence):
        probe = evidence.get('cleanup_observation') if isinstance(evidence, dict) else None
        if not isinstance(probe, dict) or evidence.get('cleanup_verified') is not True:
            raise QueueError('GPU VALIDATING requires verified physical cleanup observation')
        if not (probe.get('container_exists') is False and probe.get('guard_pids') == []
                and probe.get('compute_pids') == [] and probe.get('gpu_memory_mib') == 0
                and probe.get('containers') == [] and probe.get('cleanup_failed') is False):
            raise QueueError('GPU cleanup must prove no live guard/container/context/memory')

    def release_unlaunched(self, id, *, actor, reason):
        return self.transition(id, 'HELD', actor=actor, reason=reason, evidence={'unlaunched':True})

    def finish(self, id, disposition, *, evidence, actor):
        if disposition not in DISPOSITIONS or not evidence:
            raise QueueError('finish needs explicit disposition and evidence; exit0 is not validation')
        with self._locked():
            state = self._read()
            job = state['jobs'][id]
            if job['state'] != 'RUNNING':
                raise QueueError('finish requires RUNNING')
            if job['resource'] == 'gpu':
                self._require_cleanup(evidence)
            job['state'] = 'VALIDATING'
            job['execution_disposition'] = disposition
            job['execution_evidence'] = deepcopy(evidence)
            self._commit(state, 'finish', actor=actor, reason='execution ended; validator pending', data={'id': id, 'disposition': disposition, 'evidence': evidence})
            return deepcopy(job)
