"""CPU-only isolated proposal tests; no live queue access."""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
from proj_rsi.scheduling import test_store as existing

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('proj_rsi.scheduling._isolated_memo', HERE / 'store_proposal.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
QueueStore = candidate.QueueStore
# Execute all existing tests against the proposal, including forked claim races.
existing.QueueStore = QueueStore
existing.QueueError = candidate.QueueError
existing.CorruptWAL = candidate.CorruptWAL
StoreTests = existing.StoreTests


def writer(directory):
    QueueStore(directory).register(existing.job('external'))


class MemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.store = QueueStore(self.directory)
        self.store.register(existing.job('one'))

    def read(self):
        with self.store._locked():
            return self.store._read()

    def corrupt(self):
        raw = self.store.wal_path.read_bytes()
        self.store.wal_path.write_bytes(raw.replace(b'"register"', b'"tampered"', 1))

    def test_unchanged_and_own_commit_skip_verified_read(self):
        with mock.patch.object(self.store, '_read_verified', side_effect=AssertionError('rehash')):
            self.assertEqual(self.read()['sequence'], 1)
            self.store.register(existing.job('two'))
            self.assertEqual(self.read()['sequence'], 2)

    def test_cold_corruption_fails(self):
        self.corrupt()
        with self.assertRaises(candidate.CorruptWAL):
            QueueStore(self.directory).snapshot()

    def test_warm_corruption_restored_mtime_ctime_detects(self):
        old = self.store.wal_path.stat()
        time.sleep(.002)
        self.corrupt()
        os.utime(self.store.wal_path, ns=(old.st_atime_ns, old.st_mtime_ns))
        current = self.store.wal_path.stat()
        self.assertEqual(current.st_size, old.st_size)
        self.assertEqual(current.st_mtime_ns, old.st_mtime_ns)
        self.assertNotEqual(current.st_ctime_ns, old.st_ctime_ns)
        with self.assertRaises(candidate.CorruptWAL):
            self.read()

    def test_ctime_is_independent_fingerprint_component(self):
        before = self.store._verified_state['fingerprint']
        with mock.patch.object(self.store, '_wal_fingerprint', return_value=(*before[:4], before[4] + 1)):
            with mock.patch.object(self.store, '_read_verified', side_effect=RuntimeError('fallback')):
                with self.assertRaisesRegex(RuntimeError, 'fallback'):
                    self.read()

    def test_external_process_commit_invalidates_warm_memo(self):
        context = multiprocessing.get_context('fork')
        process = context.Process(target=writer, args=(self.directory,))
        process.start(); process.join(10)
        self.assertEqual(process.exitcode, 0)
        with mock.patch.object(self.store, '_read_verified', wraps=self.store._read_verified) as replay:
            self.assertIn('external', self.read()['jobs'])
            self.assertEqual(replay.call_count, 1)

    def test_deepcopy_and_uncommitted_rollback(self):
        state = self.read()
        state['jobs']['one']['estimate']['direct']['p_gain'] = 99
        state['jobs']['fake'] = {'bad': True}
        actual = self.read()
        self.assertEqual(actual['jobs']['one']['estimate']['direct']['p_gain'], .5)
        self.assertNotIn('fake', actual['jobs'])
        self.store.snapshot()['jobs']['one']['estimate']['direct']['p_gain'] = 100
        self.assertEqual(self.read()['jobs']['one']['estimate']['direct']['p_gain'], .5)

    def test_failed_commit_forces_replay(self):
        with self.store._locked():
            state = self.store._read()
            with mock.patch.object(candidate, '_atomic', side_effect=OSError('checkpoint fail')):
                with self.assertRaises(OSError):
                    self.store._commit(state, 'test')
            self.assertIsNone(self.store._verified_state)
            self.assertEqual(self.store._read()['sequence'], 2)

    def test_same_size_replacement_and_truncation_invalidate(self):
        raw = self.store.wal_path.read_bytes()
        replacement = self.directory / 'replacement'
        replacement.write_bytes(raw)
        os.replace(replacement, self.store.wal_path)
        with mock.patch.object(self.store, '_read_verified', wraps=self.store._read_verified) as replay:
            self.assertEqual(self.read()['sequence'], 1)
            self.assertEqual(replay.call_count, 1)
        self.store.wal_path.write_bytes(b'')
        self.assertEqual(self.read()['sequence'], 0)

    def test_warm_torn_tail_recovers_with_evidence(self):
        with self.store.wal_path.open('ab') as handle:
            handle.write(b'{"torn":')
        self.assertEqual(self.read()['sequence'], 1)
        self.assertEqual(next(self.directory.glob('torn-final-*.bin')).read_bytes(), b'{"torn":')

    def test_own_commit_memo_uses_json_roundtrip(self):
        with self.store._locked():
            state = self.store._read()
            state['jobs']['one']['extra'] = ('tuple', 1)
            self.store._commit(state, 'test')
        self.assertEqual(self.read()['jobs']['one']['extra'], ['tuple', 1])

    def test_replay_concurrent_change_refused(self):
        self.store._verified_state = None
        original = self.store._cached_prefix
        def changed(handle, size):
            result = original(handle, size)
            with self.store.wal_path.open('ab') as append:
                append.write(b'\n')
            return result
        with mock.patch.object(self.store, '_cached_prefix', side_effect=changed):
            with self.assertRaises(candidate.CorruptWAL):
                self.read()
        self.assertIsNone(self.store._verified_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
