import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from proj_rsi.dashboard import observe_hardware as observer

NOW = 1800000000.


def idle():
    return dict(container_exists=False, compute_pids=[], gpu_memory_mib=0,
                guard_pids=[], containers=[], cleanup_failed=False)


def job(probe=None, at=NOW - 10, **extra):
    return dict(resource='gpu', state='DONE', observations=[] if probe is None else [dict(timestamp=at, status='OBSERVED', probe=probe)], **extra)


def state(*jobs):
    return {'version': 1, 'sequence': 1, 'jobs': {str(i): value for i, value in enumerate(jobs)}}


class HardwareObserverTests(unittest.TestCase):
    def derive(self, *jobs, **kwargs):
        return observer.derive_hardware(state(*jobs), now=NOW, gpu_name='H100', **kwargs)

    def test_no_actual_observations_and_no_queue_inference(self):
        for recorded in ('RUNNING', 'READY', 'DONE', 'HELD'):
            value = job(); value['state'] = recorded
            result = self.derive(value)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['observed_at'])
            self.assertIsNone(result['memory_used_mb'])
            self.assertIsNone(result['process_count'])

    def test_busy_signals_and_complete_idle(self):
        for changes in (dict(compute_pids=[123]), dict(container_exists=True), dict(gpu_memory_mib=12)):
            probe = idle(); probe.update(changes)
            self.assertEqual(self.derive(job(probe))['status'], 'busy')
        result = self.derive(job(idle()))
        self.assertEqual(result['status'], 'idle')
        self.assertEqual(result['observed_at'], NOW - 10)
        self.assertEqual(result['process_count'], 0)
        self.assertEqual(result['memory_used_mb'], 0)
        self.assertEqual(result['gpu_name'], 'H100')
        # A known positive signal suffices without manufacturing missing counts.
        result = self.derive(job({'container_exists': True}))
        self.assertEqual(result['status'], 'busy')
        self.assertIsNone(result['memory_used_mb'])
        self.assertIsNone(result['process_count'])

    def test_missing_malformed_and_inconsistent_probes_are_unknown(self):
        for field in idle():
            probe = idle(); probe.pop(field)
            self.assertEqual(self.derive(job(probe))['status'], 'unknown', field)
        for changes in (dict(compute_pids=['12']), dict(compute_pids=[True]), dict(gpu_memory_mib=-1),
                        dict(gpu_memory_mib=float('nan')), dict(gpu_memory_mib=True),
                        dict(container_exists=1), dict(guard_pids=[123]), dict(containers=['other']),
                        dict(cleanup_failed=True)):
            probe = idle(); probe.update(changes)
            self.assertEqual(self.derive(job(probe))['status'], 'unknown', changes)

    def test_latest_actual_across_gpu_jobs_not_latest_queue_event(self):
        old = job(idle(), NOW - 40)
        recent = job({**idle(), 'compute_pids': [10], 'gpu_memory_mib': 20}, NOW - 10)
        unrelated = job(idle(), NOW - 1); unrelated['resource'] = 'cpu'
        recent['observations'].append({'timestamp': NOW, 'transition': {'to': 'DONE'}})
        result = self.derive(old, recent, unrelated)
        self.assertEqual(result['status'], 'busy')
        self.assertEqual(result['observed_at'], NOW - 10)

    def test_stale_and_future_timestamp_stay_original_and_status_unknown(self):
        for at, reason in ((NOW - 121, 'stale_observation'), (NOW + 31, 'future_observation')):
            result = self.derive(job(idle(), at))
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['reason'], reason)
            self.assertEqual(result['observed_at'], at)
        value = job(idle(), NOW - 100)
        before = self.derive(value)
        later = observer.derive_hardware(state(value), now=NOW + 100)
        self.assertEqual(before['observed_at'], later['observed_at'])
        self.assertEqual(later['status'], 'unknown')

    def test_new_claim_or_launch_cannot_reuse_prelaunch_idle(self):
        original = job(idle(), NOW - 10)
        new = job(lease={'claimed_at': NOW - 5}); new['state'] = 'RUNNING'
        self.assertEqual(self.derive(original, new)['reason'], 'claim_or_launch_after_observation')
        original['lease'] = {'claimed_at': NOW - 20}
        original['state'] = 'RUNNING'
        original['observations'].append({'timestamp': NOW - 5, 'dispatch': {'phase': 'RUNNING', 'launch_requested_at': NOW - 6}})
        self.assertEqual(self.derive(original)['status'], 'unknown')
        original['observations'].append({'timestamp': NOW - 1, 'status': 'OBSERVED', 'probe': idle()})
        self.assertEqual(self.derive(original)['status'], 'idle')
        # CPU leases are unrelated to GPU hardware evidence.
        cpu = job(lease={'claimed_at': NOW}); cpu['resource'] = 'cpu'; cpu['state'] = 'RUNNING'
        self.assertEqual(self.derive(original, cpu)['status'], 'idle')

    def test_later_failed_probe_does_not_refresh_or_keep_asserting_idle(self):
        value = job(idle())
        value['observations'].append({'timestamp': NOW - 1, 'status': 'UNKNOWN', 'error': 'private host'})
        result = self.derive(value)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['observed_at'], NOW - 10)
        self.assertEqual(result['reason'], 'later_probe_unavailable')
        self.assertNotIn('private', json.dumps(result))

    def test_collection_copy_retains_original_probe_time_not_attach_time(self):
        probe = {**idle(), 'host_log_tail': 'secret raw logs not exported'}
        value = job(probe, NOW - 200)
        value['observations'].append({'timestamp': NOW - 1, 'collection': {'cleanup_observation': copy.deepcopy(probe)}})
        result = self.derive(value)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['observed_at'], NOW - 200)
        self.assertNotIn('secret', json.dumps(result))
        standalone = job()
        standalone['observations'] = [{'timestamp': NOW - 1, 'collection': {'cleanup_observation': {**idle(), 'observed_at': NOW - 20}}}]
        result = self.derive(standalone)
        self.assertEqual(result['status'], 'idle')
        self.assertEqual(result['observed_at'], NOW - 20)
        standalone['observations'][0]['collection']['cleanup_observation'].pop('observed_at')
        self.assertIsNone(self.derive(standalone)['observed_at'])

    def test_conflicting_latest_actual_observations_unknown(self):
        result = self.derive(job(idle()), job({**idle(), 'gpu_memory_mib': 4}))
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'conflicting_latest_observations')

    def test_untimed_or_malformed_observation_cannot_leave_idle_assertion(self):
        for observation in ({'probe': idle()}, {'timestamp': NOW, 'probe': None}, {'status': 'UNKNOWN'}):
            value = job(idle()); value['observations'].append(observation)
            result = self.derive(value)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['observed_at'], NOW - 10)

    def test_unavailable_or_torn_wal_never_creates_repairs_or_mutates_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); scheduler = root / 'scheduler'; scheduler.mkdir()
            output = root / 'dashboard/hardware.json'
            result = observer.observe_once(scheduler_dir=scheduler, output=output, now=NOW)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(list(scheduler.iterdir()), [])
            (scheduler / '.lock').write_bytes(b'')
            (scheduler / 'events.jsonl').write_bytes(b'{torn')
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in scheduler.iterdir()}
            result = observer.observe_once(scheduler_dir=scheduler, output=output, now=NOW)
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in scheduler.iterdir()})

    def test_actual_checksummed_wal_readonly_no_subprocess_network_or_queue_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); scheduler = root / 'scheduler'; scheduler.mkdir()
            (scheduler / '.lock').write_bytes(b'')
            payload = {'sequence': 1, 'timestamp': NOW, 'snapshot': state(job(idle()))}
            canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            (scheduler / 'events.jsonl').write_text(json.dumps({'payload': payload, 'sha256': hashlib.sha256(canonical).hexdigest()}) + '\n')
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in scheduler.iterdir()}
            with mock.patch('subprocess.run', side_effect=AssertionError('no subprocess')), mock.patch('socket.create_connection', side_effect=AssertionError('no network')), mock.patch('proj_rsi.scheduling.store.QueueStore.snapshot', side_effect=AssertionError('no mutating QueueStore snapshot')):
                result = observer.observe_once(scheduler_dir=scheduler, output=root / 'hardware.json', now=NOW)
            self.assertEqual(result['status'], 'idle')
            self.assertEqual(json.loads((root / 'hardware.json').read_text()), result)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in scheduler.iterdir()})
            self.assertEqual(list(root.glob('.hardware.json.*')), [])


if __name__ == '__main__': unittest.main()
