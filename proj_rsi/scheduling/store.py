"""Crash-recoverable local priority queue. WAL is authoritative, checkpoint is a cache.

snapshot: {version: 1, sequence: int, jobs: {job_id: job}}. Every fsynced WAL
line contains a checksummed complete next snapshot plus the audit event. A stable
.lock inode serializes processes. No lease expiry implies process death.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import hmac
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


def _atomic(path, content, *, mode=None):
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with open(temporary, 'xb') as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
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
        self.cache_path = self.directory / 'replay-cache.json'
        self.cache_key_path = self.directory / '.replay-cache.key'
        self._verified_prefix = None
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

    def _complete_wal(self):
        """Preserve the original torn-tail recovery, before validating any record."""
        size = self.wal_path.stat().st_size
        if not size:
            return 0
        with self.wal_path.open('rb') as handle:
            handle.seek(size - 1)
            if handle.read(1) == b'\n':
                return size
            end = size
            complete = 0
            while end:
                start = max(0, end - (1 << 16))
                handle.seek(start)
                block = handle.read(end - start)
                index = block.rfind(b'\n')
                if index >= 0:
                    complete = start + index + 1
                    break
                end = start
            handle.seek(complete)
            tail = handle.read(size - complete)
        evidence = self.directory / ('torn-final-' + hashlib.sha256(tail).hexdigest() + '.bin')
        if not evidence.exists():
            _atomic(evidence, tail)
        with self.wal_path.open('r+b') as handle:
            handle.truncate(complete)
            handle.flush()
            os.fsync(handle.fileno())
        return complete

    @staticmethod
    def _record(line, sequence, number):
        """The original authoritative per-event checksum/sequence/envelope checks."""
        try:
            entry = json.loads(line)
            payload = entry['payload']
            if hashlib.sha256(_json(payload)).hexdigest() != entry['sha256']:
                raise ValueError('checksum mismatch')
            if payload['sequence'] != sequence + 1:
                raise ValueError('non-contiguous sequence')
            candidate = payload['snapshot']
            if candidate['version'] != 1 or candidate['sequence'] != payload['sequence'] or not isinstance(candidate['jobs'], dict):
                raise ValueError('invalid snapshot envelope')
            return candidate
        except (ValueError, TypeError, KeyError) as error:
            raise CorruptWAL('WAL record %d is corrupt; execution refused: %s' % (number, error)) from error

    def _cache_key(self, *, create=False):
        try:
            with self.cache_key_path.open('rb') as handle:
                key = handle.read(33)
            if len(key) == 32:
                return key
        except OSError:
            pass
        if not create:
            return None
        key = os.urandom(32)
        _atomic(self.cache_key_path, key, mode=0o600)
        return key

    def _cached_prefix(self, handle, size):
        """Authenticate prior validation, rehash every prefix byte, bind actual last event.

        The seal is a local cache attestation, not authentication against a malicious
        queue-directory owner (who can also rewrite valid WAL events). Neither the
        checkpoint nor a cache-supplied snapshot is ever used as authoritative state.
        """
        try:
            with self.cache_path.open('rb') as cached:
                raw = cached.read(4097)
            if len(raw) > 4096:
                return None
            envelope = json.loads(raw)
            if set(envelope) != {'payload', 'hmac_sha256'}:
                return None
            payload = envelope['payload']
            key = self._cache_key()
            if key is None or not hmac.compare_digest(hmac.new(key, _json(payload), hashlib.sha256).hexdigest(), envelope['hmac_sha256']):
                return None
            if set(payload) != {'version', 'bytes', 'sha256', 'sequence', 'last_start', 'last_sha256'} or payload['version'] != 1:
                return None
            offset, last, sequence = payload['bytes'], payload['last_start'], payload['sequence']
            if not all(type(v) is int for v in (offset, last, sequence)) or not (0 <= last < offset <= size and sequence >= 1):
                return None
            handle.seek(0)
            digest = hashlib.sha256()
            remaining = offset
            while remaining:
                block = handle.read(min(1 << 20, remaining))
                if not block:
                    return None
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != payload['sha256']:
                return None
            handle.seek(last)
            line = handle.read(offset - last)
            if not line.endswith(b'\n') or b'\n' in line[:-1] or hashlib.sha256(line).hexdigest() != payload['last_sha256']:
                return None
            state = self._record(line, sequence - 1, sequence)
            handle.seek(offset)
            return state, digest, offset, last, payload['last_sha256']
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return None  # An optional cache never suppresses authoritative replay.

    def _publish_prefix(self):
        prefix = self._verified_prefix
        if not prefix or not prefix['sequence']:
            return
        payload = {'version': 1, 'bytes': prefix['bytes'], 'sha256': prefix['digest'].hexdigest(),
                   'sequence': prefix['sequence'], 'last_start': prefix['last_start'], 'last_sha256': prefix['last_sha256']}
        try:
            key = self._cache_key(create=True)
            envelope = {'payload': payload, 'hmac_sha256': hmac.new(key, _json(payload), hashlib.sha256).hexdigest()}
            _atomic(self.cache_path, _json(envelope) + b'\n')
        except OSError:
            pass  # WAL/checkpoint durability and operation results do not depend on cache writes.

    def _read(self):
        state = {'version': 1, 'sequence': 0, 'jobs': {}}
        self._verified_prefix = None
        if not self.wal_path.exists():
            self._verified_prefix = {'bytes': 0, 'digest': hashlib.sha256(), 'sequence': 0,
                                     'last_start': 0, 'last_sha256': None}
            return state
        size = self._complete_wal()
        with self.wal_path.open('rb') as handle:
            cached = self._cached_prefix(handle, size)
            if cached is None:
                handle.seek(0)
                digest, offset, last, last_sha = hashlib.sha256(), 0, 0, None
            else:
                state, digest, offset, last, last_sha = cached
            starting_offset = offset
            while offset < size:
                line = handle.readline(size - offset)
                if not line.endswith(b'\n'):
                    raise CorruptWAL('WAL changed during locked replay; execution refused')
                state = self._record(line, state['sequence'], state['sequence'] + 1)
                last, last_sha = offset, hashlib.sha256(line).hexdigest()
                digest.update(line)
                offset += len(line)
            self._verified_prefix = {'bytes': offset, 'digest': digest, 'sequence': state['sequence'],
                                     'last_start': last, 'last_sha256': last_sha}
        if cached is None or offset != starting_offset:
            self._publish_prefix()
        return state

    def _commit(self, state, action, *, actor='controller', reason='', data=None):
        state['sequence'] += 1
        payload = {'sequence': state['sequence'], 'timestamp': time.time(), 'action': action,
                   'actor': actor, 'reason': reason, 'data': data, 'snapshot': state}
        record = _json({'payload': payload, 'sha256': hashlib.sha256(_json(payload)).hexdigest()}) + b'\n'
        new_file = not self.wal_path.exists()
        previous_size = self.wal_path.stat().st_size if not new_file else 0
        with open(self.wal_path, 'ab') as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        if new_file:
            _sync_directory(self.directory)
        _atomic(self.checkpoint_path, _json(state) + b'\n')
        prefix = self._verified_prefix
        if prefix and prefix['bytes'] == previous_size and prefix['sequence'] == state['sequence'] - 1:
            # Only strictly valid new events may extend an attested prefix, even
            # if a private caller supplied a malformed snapshot to _commit.
            try:
                self._record(record, prefix['sequence'], state['sequence'])
            except CorruptWAL:
                self._verified_prefix = None
                return
            digest = prefix['digest'].copy()
            digest.update(record)
            self._verified_prefix = {'bytes': previous_size + len(record), 'digest': digest,
                                     'sequence': state['sequence'], 'last_start': previous_size,
                                     'last_sha256': hashlib.sha256(record).hexdigest()}
            self._publish_prefix()
        else:
            self._verified_prefix = None

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
