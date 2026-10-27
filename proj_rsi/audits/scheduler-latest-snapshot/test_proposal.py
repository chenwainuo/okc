"""Temporary queues only. Strict default and explicit latest-snapshot contract."""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from proj_rsi.scheduling import test_store as existing

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('proj_rsi.scheduling._isolated_latest', HERE/'store_proposal.py')
candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
existing.QueueStore = candidate.QueueStore
existing.QueueError = candidate.QueueError
existing.CorruptWAL = candidate.CorruptWAL
StrictStoreTests = existing.StoreTests


class HotStore(candidate.QueueStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, latest_snapshot_reads=True, **kwargs)


def claim_process(directory, results):
    try:
        job = HotStore(directory).claim('cpu', 'cross-process')
        results.put(job['id'] if job else None)
    except BaseException as error:
        results.put('ERROR:' + repr(error))


class LatestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.store = HotStore(self.directory)
        self.store.register(existing.job('one'))
        self.store.register(existing.job('two'))

    def read(self):
        with self.store._locked():
            return self.store._read()

    def corrupt_last(self):
        raw = self.store.wal_path.read_bytes()
        before, after = raw.rsplit(b'"register"', 1)
        self.store.wal_path.write_bytes(before + b'"tampered"' + after)

    def test_latest_same_size_corruption_immediate_refused(self):
        self.corrupt_last()
        with self.assertRaises(candidate.CorruptWAL):
            self.store.claim('cpu', 'worker')

    def test_historic_corruption_deferred_but_explicit_and_startup_audit_refuse(self):
        raw = self.store.wal_path.read_bytes()
        self.store.wal_path.write_bytes(raw.replace(b'"register"', b'"tampered"', 1))
        self.assertEqual(self.read()['sequence'], 2)
        self.assertEqual(set(self.read()['jobs']), {'one', 'two'})
        with self.assertRaises(candidate.CorruptWAL):
            self.store.audit()
        with self.assertRaises(candidate.CorruptWAL):
            HotStore(self.directory)
        with self.assertRaises(candidate.CorruptWAL):
            candidate.QueueStore(self.directory).snapshot()

    def test_no_cache_trust_and_current_state_matches_strict(self):
        self.store.cache_path.write_text('{"garbage":true}')
        self.store.checkpoint_path.write_text('{"garbage":true}')
        expected = candidate.QueueStore(self.directory).snapshot()
        self.store.cache_path.write_text('{"garbage":true}')
        with mock.patch.object(self.store, '_cached_prefix', side_effect=AssertionError('cache trusted')):
            self.assertEqual(self.read(), expected)
            self.assertEqual(self.store.snapshot(), expected)

    def test_complete_truncation_rolls_back_and_refuses(self):
        first = self.store.wal_path.read_bytes().splitlines(keepends=True)[0]
        self.store.wal_path.write_bytes(first)
        with self.assertRaisesRegex(candidate.CorruptWAL, 'rolled back'):
            self.read()

    def test_missing_or_empty_wal_refused_after_state_seen(self):
        self.store.wal_path.write_bytes(b'')
        with self.assertRaisesRegex(candidate.CorruptWAL, 'rolled back'):
            self.read()
        self.store.wal_path.unlink()
        with self.assertRaisesRegex(candidate.CorruptWAL, 'rolled back'):
            self.read()

    def test_partial_truncation_preserves_evidence_then_refuses_rollback(self):
        raw = self.store.wal_path.read_bytes()
        self.store.wal_path.write_bytes(raw[:-20])
        with self.assertRaisesRegex(candidate.CorruptWAL, 'rolled back'):
            self.read()
        self.assertTrue(list(self.directory.glob('torn-final-*.bin')))

    def test_torn_append_recovers(self):
        expected = self.read()
        with self.store.wal_path.open('ab') as append:
            append.write(b'{"torn":')
        self.assertEqual(self.read(), expected)
        self.assertEqual(next(self.directory.glob('torn-final-*.bin')).read_bytes(), b'{"torn":')

    def test_invalid_latest_sequence_and_envelope(self):
        raw = self.store.wal_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        for change in ('sequence', 'envelope'):
            entry = json.loads(lines[-1]); payload = entry['payload']
            if change == 'sequence': payload['sequence'] = 0
            else: payload['snapshot']['sequence'] = 123
            entry['sha256'] = candidate.hashlib.sha256(candidate._json(payload)).hexdigest()
            self.store.wal_path.write_bytes(b''.join(lines[:-1]) + candidate._json(entry) + b'\n')
            with self.assertRaises(candidate.CorruptWAL): self.read()

    def test_crossprocess_atomic_claim_and_ownership(self):
        # One READY job at a time can own an overlapping write path.
        for id in ('one', 'two'):
            self.store.update(id, {'write_paths': ['shared']}, actor='root', reason='test')
        context = multiprocessing.get_context('fork'); results = context.Queue()
        processes = [context.Process(target=claim_process, args=(self.directory, results)) for _ in range(8)]
        for process in processes: process.start()
        for process in processes:
            process.join(10); self.assertEqual(process.exitcode, 0)
        claims = [results.get(timeout=2) for _ in processes]
        self.assertEqual(sum(value is not None for value in claims), 1, claims)
        snapshot = self.read()
        self.assertEqual(sum(job['state'] == 'RUNNING' for job in snapshot['jobs'].values()), 1)
        self.assertEqual(snapshot, candidate.QueueStore(self.directory).snapshot())

    def test_large_latest_record_across_reverse_blocks(self):
        with self.store._locked():
            state = self.store._read()
            state['jobs']['one']['large'] = 'x' * (3 * (1 << 16) + 11)
            self.store._commit(state, 'large')
        self.assertEqual(len(self.read()['jobs']['one']['large']), 3 * (1 << 16) + 11)

    def test_noncooperating_append_during_read_refused(self):
        original = self.store._record
        def changed(*args):
            state = original(*args)
            with self.store.wal_path.open('ab') as append:
                append.write(b'\n')
            return state
        with mock.patch.object(self.store, '_record', side_effect=changed):
            with self.assertRaises(candidate.CorruptWAL):
                self.read()

    def test_single_record_and_empty_queue(self):
        other = HotStore(self.directory/'other')
        self.assertEqual(other.snapshot()['sequence'], 0)
        other.register(existing.job('only'))
        self.assertEqual(other.snapshot()['sequence'], 1)

    def test_external_write_observed_no_memo(self):
        other = candidate.QueueStore(self.directory)
        other.register(existing.job('external'))
        self.assertIn('external', self.read()['jobs'])
        self.assertEqual(self.read(), other.snapshot())

    def test_latest_read_deepcopy_isolation(self):
        state = self.read(); state['jobs']['one']['state'] = 'DONE'
        self.assertEqual(self.read()['jobs']['one']['state'], 'READY')


if __name__ == '__main__': unittest.main(verbosity=2)
