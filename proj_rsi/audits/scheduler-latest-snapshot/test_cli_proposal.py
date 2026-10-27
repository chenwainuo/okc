"""CPU-only CLI wiring tests using temporary queues and fake dispatcher module."""
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
from proj_rsi.scheduling.test_store import job

HERE = Path(__file__).resolve().parent
proposal = importlib.import_module('proj_rsi.audits.scheduler-latest-snapshot.test_proposal')
spec = importlib.util.spec_from_file_location('proj_rsi.scheduling._latest_cli_proposal', HERE/'cli_proposal.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
cli.QueueStore = proposal.candidate.QueueStore
cli.QueueError = proposal.candidate.QueueError


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)/'queue'
        self.dispatcher = mock.Mock()
        self.dispatcher.work.return_value = {'cpu_test': 'work'}
        self.dispatcher.launch_next.return_value = {'cpu_test': 'run-next'}
        self.constructor = mock.Mock(return_value=self.dispatcher)
        fake = types.ModuleType('proj_rsi.scheduling.dispatch')
        fake.GPUDispatcher = self.constructor
        self.modules = mock.patch.dict(sys.modules, {'proj_rsi.scheduling.dispatch': fake})
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.stdout = io.StringIO(); self.stderr = io.StringIO()

    def invoke(self, *args):
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            cli.main(['--directory', str(self.directory), *args])

    def store(self):
        return self.constructor.call_args.args[0]

    def test_work_default_remains_strict(self):
        self.invoke('work', '--max-jobs', '1')
        self.assertFalse(self.store()._latest_snapshot_reads)
        self.dispatcher.work.assert_called_once()

    def test_work_optin_enables_latest_after_strict_startup(self):
        with mock.patch.object(proposal.candidate.QueueStore, '_read_strict', autospec=True,
                               return_value={'version': 1, 'sequence': 17, 'jobs': {}}) as strict:
            self.invoke('work', '--latest-snapshot', '--max-jobs', '1')
        self.assertEqual(strict.call_count, 1)
        self.assertTrue(self.store()._latest_snapshot_reads)
        self.assertEqual(self.store()._sequence_floor, 17)
        self.dispatcher.work.assert_called_once()

    def test_optin_corrupt_history_refused_before_dispatch(self):
        seed = proposal.candidate.QueueStore(self.directory)
        seed.register(job('one')); seed.register(job('two'))
        raw = seed.wal_path.read_bytes()
        seed.wal_path.write_bytes(raw.replace(b'"register"', b'"tampered"', 1))
        with self.assertRaises(SystemExit) as error:
            self.invoke('work', '--latest-snapshot')
        self.assertEqual(error.exception.code, 2)
        self.assertIn('corrupt', self.stderr.getvalue())
        self.constructor.assert_not_called()

    def test_strict_other_commands_and_one_shot_run_next(self):
        self.invoke('run-next')
        self.assertFalse(self.store()._latest_snapshot_reads)
        self.dispatcher.launch_next.assert_called_once()
        real = proposal.candidate.QueueStore
        made = []
        def constructor(*args, **kwargs):
            store = real(*args, **kwargs); made.append(store); return store
        descriptor = Path(self.temp.name)/'job.json'; descriptor.write_text(json.dumps(job('one')))
        with mock.patch.object(cli, 'QueueStore', side_effect=constructor):
            self.invoke('register', str(descriptor))
            self.invoke('status')
        self.assertEqual(len(made), 2)
        self.assertTrue(all(not store._latest_snapshot_reads for store in made))

    def test_flag_is_only_accepted_for_work(self):
        for args in [('run-next', '--latest-snapshot'), ('status', '--latest-snapshot'),
                     ('register', '-', '--latest-snapshot')]:
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.invoke(*args)
            self.assertEqual(error.exception.code, 2)
        self.constructor.assert_not_called()

    def test_work_help_discloses_history_contract(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke('work', '--help')
        self.assertEqual(error.exception.code, 0)
        self.assertIn('historical corruption detection', ' '.join(self.stdout.getvalue().split()))
        self.assertIn('strict startup history audit', ' '.join(self.stdout.getvalue().split()))


if __name__ == '__main__': unittest.main(verbosity=2)
