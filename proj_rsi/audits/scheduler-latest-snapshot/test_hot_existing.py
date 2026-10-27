"""Existing store behavior under opt-in reads; historical-corruption timing explicitly changed."""
import importlib
import unittest
from proj_rsi.scheduling import test_store as existing
proposal = importlib.import_module('proj_rsi.audits.scheduler-latest-snapshot.test_proposal')
existing.QueueStore = proposal.HotStore
existing.QueueError = proposal.candidate.QueueError
existing.CorruptWAL = proposal.candidate.CorruptWAL


class HotExistingTests(existing.StoreTests):
    def test_middle_corruption_fail_closed(self):
        # This contract intentionally differs: startup/explicit audit detects old
        # corruption, while hot operation validates actual latest full snapshot.
        self.store.register(existing.job('one')); self.store.register(existing.job('two'))
        raw = self.store.wal_path.read_bytes()
        self.store.wal_path.write_bytes(raw.replace(b'"register"', b'"tampered"', 1))
        self.assertIsNotNone(self.store.claim('cpu', 'worker'))
        with self.assertRaises(proposal.candidate.CorruptWAL): self.store.audit()
        with self.assertRaises(proposal.candidate.CorruptWAL): proposal.HotStore(self.directory, repo_root=self.repo)


if __name__ == '__main__': unittest.main(verbosity=2)
