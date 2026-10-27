"""Submission failure recovery with fake API responses; no okc/GPU calls."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from proj_rsi import SCHEMA_VERSION, runner
from proj_rsi.ledger import Ledger


class SubmissionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'candidate/submission.py'
        self.source.parent.mkdir()
        self.source.write_text('# okc: problem=muse_bf16_proj\n# frozen source\n')
        self.jobs = self.root / 'results/jobs'
        self.ledger = Ledger(self.root / 'ledger.json')
        self.ledger.save({'schema_version': SCHEMA_VERSION, 'experiments': {
            'registered': {'jobs': [], 'state': 'reviewed'}}, 'jobs': {}})
        self.limits = {'checks': {'left': 7}, 'scored': {'left': 3}}
        self.response = {'job_id': 'accepted123', 'status': 'queued', 'server_metadata': {'keep': ['all', 'response', 'fields']}}
        self.api_calls = []
        self.patches = [mock.patch.object(runner, 'ROOT', self.root),
                        mock.patch.object(runner, 'JOBS', self.jobs),
                        mock.patch.object(runner, 'Ledger', return_value=self.ledger),
                        mock.patch.object(runner, 'check_file', return_value={'ok': True}),
                        mock.patch.object(runner, 'okc', side_effect=self.api)]
        for patch in self.patches:
            patch.start(); self.addCleanup(patch.stop)

    def api(self, *args, **kwargs):
        self.api_calls.append(args)
        if args == ('problems',):
            return {'problems': [{'name': runner.PROBLEM, 'limits': self.limits}]}
        return self.response

    def receipts(self):
        return list(self.jobs.glob('attempt-*.response.json'))

    def test_unknown_experiment_is_rejected_before_any_api_or_quota_call(self):
        with self.assertRaisesRegex(ValueError, 'unknown experiment missing'):
            runner.submit(self.source, experiment='missing')
        self.assertEqual(self.api_calls, [])
        self.assertFalse(self.jobs.exists())

    def test_accepted_id_raw_response_hash_and_quota_durable_before_ledger_failure(self):
        fsynced = []
        original_fsync = os.fsync
        def fsync(fd):
            fsynced.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
            original_fsync(fd)
        def failed_attach(experiment, job, kind):
            receipts = self.receipts()
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt['job'], 'accepted123')
            self.assertEqual(receipt['submit_response'], self.response)
            self.assertEqual(receipt['quota_before'], self.limits)
            self.assertEqual(receipt['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
            self.assertEqual(receipt['kind'], 'check')
            self.assertEqual(receipt['experiment'], 'registered')
            self.assertTrue(receipt['submitted_at'].endswith('Z'))
            self.assertTrue((self.jobs / 'accepted123.submitted.json').is_file())
            self.assertEqual(fsynced, ['file', 'directory'] * 3)
            raise OSError('simulated ledger write failure')
        with mock.patch.object(self.ledger, 'attach_job', side_effect=failed_attach), mock.patch.object(runner.os, 'fsync', side_effect=fsync):
            with self.assertRaises(runner.SubmissionRecoveryError) as caught:
                runner.submit(self.source, experiment='registered')
        message = str(caught.exception)
        self.assertIn('Accepted job accepted123', message)
        self.assertIn(str(self.receipts()[0]), message)
        self.assertIn('Never retry blindly', message)
        self.assertEqual([call[0] for call in self.api_calls], ['problems', 'check'])

    def test_normal_flow_legacy_receipt_fields_and_registered_ledger_attach(self):
        self.assertEqual(runner.submit(self.source, experiment='registered'), 'accepted123')
        receipt = json.loads((self.jobs / 'accepted123.submitted.json').read_text())
        self.assertEqual(receipt['candidate'], 'candidate/submission.py')
        self.assertEqual(receipt['verb'], 'check')
        self.assertEqual(receipt['quota_bucket'], 'checks')
        self.assertEqual(receipt['status'], 'accepted')
        self.assertEqual(json.loads(self.receipts()[0].read_text()), receipt)
        attached = self.ledger.load()['experiments']['registered']['jobs']
        self.assertEqual(attached[0]['job'], 'accepted123')
        self.assertEqual(attached[0]['kind'], 'check')
        self.assertEqual(list(self.jobs.glob('.*.json.*')), [])

    def test_scored_and_legacy_kind_and_legacy_job_identifier_fields(self):
        for kind, key in (('scored', 'job'), ('submit', 'id'), ('legacy-score-kind', 'job_id')):
            with self.subTest(kind=kind):
                self.response = {key: 'accepted-' + kind}
                self.api_calls.clear()
                job = runner.submit(self.source, kind=kind)
                receipt = json.loads((self.jobs / (job + '.submitted.json')).read_text())
                self.assertEqual(receipt['kind'], kind)
                self.assertEqual(receipt['verb'], 'submit')
                self.assertEqual(receipt['quota_bucket'], 'scored')
                self.assertEqual([call[0] for call in self.api_calls], ['problems', 'submit'])

    def test_malformed_responses_are_retained_and_never_retried_or_attached(self):
        for response in ({'status': 'queued'}, ['accepted123'], None, 'unexpected', {'job_id': '../escaped'}, {'job_id': {'id': 'nested'}}, {'job_id': 123}):
            with self.subTest(response=response):
                self.response = response; self.api_calls.clear()
                before = set(self.receipts())
                with mock.patch.object(self.ledger, 'attach_job') as attach:
                    with self.assertRaises(runner.SubmissionRecoveryError) as caught:
                        runner.submit(self.source, experiment='registered')
                    attach.assert_not_called()
                self.assertIn('Never retry blindly', str(caught.exception))
                new = set(self.receipts()) - before
                self.assertEqual(len(new), 1)
                receipt_path = new.pop()
                self.assertEqual(json.loads(receipt_path.read_text())['submit_response'], response)
                self.assertIn(str(receipt_path), str(caught.exception))
                self.assertEqual([call[0] for call in self.api_calls], ['problems', 'check'])
        self.assertFalse((self.root / 'escaped.submitted.json').exists())

    def test_malformed_cli_output_retains_untruncated_raw_response(self):
        self.patches[-1].stop()
        raw = 'accepted but malformed ' + 'x' * 1000
        processes = [subprocess.CompletedProcess([], 0, json.dumps({'problems': [{'name': runner.PROBLEM, 'limits': self.limits}]}), ''),
                     subprocess.CompletedProcess([], 0, raw, 'server stderr')]
        with mock.patch.object(runner.subprocess, 'run', side_effect=processes) as process:
            with self.assertRaises(runner.SubmissionRecoveryError) as caught:
                runner.submit(self.source)
        self.assertEqual(process.call_count, 2)
        receipt = json.loads(self.receipts()[0].read_text())
        self.assertEqual(receipt['submit_response'], {'stdout': raw, 'stderr': 'server stderr', 'returncode': 0})
        self.assertIn('outcome unknown', str(caught.exception))
        self.assertIn('Never retry blindly', str(caught.exception))

    def test_preflight_receipt_failure_prevents_submission(self):
        with mock.patch.object(runner, '_atomic_receipt', side_effect=OSError('read only filesystem')):
            with self.assertRaises(OSError):
                runner.submit(self.source)
        self.assertEqual([call[0] for call in self.api_calls], ['problems'])

    def test_legacy_receipt_failure_keeps_primary_recovery_receipt_and_skips_ledger(self):
        original = runner._atomic_receipt
        def write(path, data):
            if path.name.endswith('.submitted.json'):
                raise OSError('legacy receipt write failed')
            original(path, data)
        with mock.patch.object(runner, '_atomic_receipt', side_effect=write), mock.patch.object(self.ledger, 'attach_job') as attach:
            with self.assertRaises(runner.SubmissionRecoveryError) as caught:
                runner.submit(self.source, experiment='registered')
            attach.assert_not_called()
        self.assertEqual(json.loads(self.receipts()[0].read_text())['job'], 'accepted123')
        self.assertIn('accepted123', str(caught.exception))
        self.assertIn(str(self.receipts()[0]), str(caught.exception))

    def test_lint_quota_and_external_candidate_failures_do_not_submit(self):
        with mock.patch.object(runner, 'check_file', return_value={'ok': False, 'findings': ['bad']}):
            with self.assertRaisesRegex(RuntimeError, 'lint failed'):
                runner.submit(self.source)
        self.assertEqual(self.api_calls, [])
        with self.assertRaisesRegex(RuntimeError, 'quota'):
            runner.submit(self.source, reserve=7)
        self.assertEqual([call[0] for call in self.api_calls], ['problems'])
        self.api_calls.clear()
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / 'submission.py'; path.write_text('external frozen file')
            self.assertEqual(runner.submit(path), 'accepted123')
            receipt = json.loads(self.receipts()[0].read_text())
            self.assertEqual(receipt['candidate'], str(path))


if __name__ == '__main__': unittest.main()
