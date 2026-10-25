import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_pair import run_pair, WorkerFailure


FAKE_WORKER = r'''
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--role');a=p.parse_args()
root=pathlib.Path(a.root); imports=dict(source_root=a.root,archive_sha256='a'*64,modules={name:dict(sha256='b'*64) for name in ('reference','baseline','nvfp4')}); mode=os.environ.get('RSI_FAKE_MODE','ok'); batch=None
counts=dict(reference_cases=5,eager_reference_cases=5,captured_valid_cases=0,timed_reference_cases=0,empty_local_eager_cases=1,empty_local_graph_cases=0)
(root/f'{a.role}.pid').write_text(str(os.getpid()))
for line in sys.stdin:
 r=json.loads(line);op=r['op']
 with open(os.environ['RSI_ORDER_LOG'],'a') as out:out.write(json.dumps(dict(role=a.role,op=op,seed=r.get('input_seed')))+'\n')
 if op=='init':
  result=dict(counts=counts,weight_seed=r['seed'],imports=imports,runtime=dict(torch='fake',cuda='fake'))
 elif op=='configure':
  batch=r['batch'];counts['captured_valid_cases']+=1;counts['reference_cases']+=1;counts['empty_local_graph_cases']+=1
  result=dict(batch=batch,input_sha256=hashlib.sha256(str(batch).encode()).hexdigest(),counts=counts)
 elif op=='measure':
  if mode=='timeout' and a.role=='baseline':
   child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
   (root/'grandchild.pid').write_text(str(child.pid))
   signal.signal(signal.SIGTERM,signal.SIG_IGN)
   sys.stdout.write('{"id":');sys.stdout.flush();time.sleep(60)
  duration=100. if a.role=='baseline' else 80.
  if mode=='zero':duration=0.
  if mode=='nan':duration=float('nan')
  if mode=='negative':duration=-1.
  counts['reference_cases']+=1;counts['timed_reference_cases']+=1
  fingerprint=hashlib.sha256(f"{batch}:{r['input_seed']}".encode()).hexdigest()
  if mode=='mismatch' and a.role=='candidate':fingerprint='f'*64
  result=dict(batch=batch,input_seed=r['input_seed'],input_sha256=fingerprint,elapsed_us=duration,reference_pass=True,input_immutable_pass=mode!='mutated',counts=counts)
 elif op=='close':
  result=dict(counts=counts,imports=imports,reference_pass=True,empty_local_eager_pass=True,empty_local_graph_pass=True,input_immutable_pass=True)
  if mode=='bad_terminal':result['empty_local_graph_pass']=False
 print(json.dumps(dict(id=r['id'],ok=True,result=result)),flush=True)
 if op=='close':sys.exit(0)
'''


class RSIPairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.baseline=self.root/'baseline';self.candidate=self.root/'candidate'
        for root in (self.baseline,self.candidate):
            root.mkdir();(root/'.rsi-bundle.json').write_text(json.dumps(dict(archive_sha256='a'*64,files={})))
        self.fake=self.root/'fake_worker.py';self.fake.write_text(FAKE_WORKER)
        self.order=self.root/'order.log';self.output=self.root/'result.json'

    def execute(self, mode='ok', timeout=10):
        with patch.dict(os.environ,{'RSI_FAKE_MODE':mode,'RSI_ORDER_LOG':str(self.order)}):
            return run_pair(self.baseline,self.candidate,self.output,123,4,[8,32],timeout,worker_script=self.fake)

    def assert_no_processes(self):
        for root in (self.baseline,self.candidate):
            for file in root.glob('*.pid'):
                pid=int(file.read_text())
                self.assertFalse(Path(f'/proc/{pid}').exists(),f'Worker/descendant remains: {pid}')

    def test_alternating_serial_order_input_equality_and_clean_shutdown(self):
        result=self.execute()
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['process_groups_terminated'])
        self.assertEqual(result['input_fingerprint_checks'],10)
        self.assertEqual(result['results'][0]['orders'],['AB','BA','AB','BA'])
        self.assertEqual(result['results'][0]['balanced_speedup'],1.25)
        actions=[json.loads(line) for line in self.order.read_text().splitlines()]
        measurements=[entry['role'] for entry in actions if entry['op']=='measure']
        self.assertEqual(measurements,['baseline','candidate','candidate','baseline']*4)
        for worker in result['workers'].values():
            self.assertEqual(worker['counts']['timed_reference_cases'],8)
            self.assertTrue(worker['clean_exit'])
        self.assertNotEqual(result['workers']['baseline']['cache_root'],result['workers']['candidate']['cache_root'])
        self.assert_no_processes()

    def test_input_mismatch_fails_closed_and_reaps_workers(self):
        with self.assertRaisesRegex(WorkerFailure,'fingerprints differ'):
            self.execute('mismatch')
        self.assertEqual(json.loads(self.output.read_text())['status'],'failed')
        self.assert_no_processes()

    def test_nonpositive_or_nonfinite_duration_rejected(self):
        for mode in ('zero','negative','nan'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(WorkerFailure,'duration'):
                    self.execute(mode)
                self.assert_no_processes()
                self.assertEqual(json.loads(self.output.read_text())['status'],'failed')

    def test_partial_line_timeout_kills_and_waits_compiler_descendant(self):
        with self.assertRaises(TimeoutError):
            self.execute('timeout',timeout=1)
        self.assertTrue((self.baseline/'grandchild.pid').exists())
        report=json.loads(self.output.read_text())
        self.assertEqual(report['status'],'failed')
        self.assertTrue(report['process_groups_terminated'])
        self.assert_no_processes()

    def test_mutated_input_claim_fails_closed(self):
        with self.assertRaisesRegex(WorkerFailure,'correctness mismatch'):
            self.execute('mutated')
        self.assert_no_processes()
        self.assertEqual(json.loads(self.output.read_text())['status'],'failed')

    def test_missing_terminal_correctness_cannot_complete(self):
        with self.assertRaisesRegex(WorkerFailure,'terminal correctness'):
            self.execute('bad_terminal')
        self.assert_no_processes()

    def test_sigterm_coordinator_reaps_workers_and_compiler(self):
        environment=os.environ.copy()
        environment.update(RSI_FAKE_MODE='timeout',RSI_ORDER_LOG=str(self.order),
                           PYTHONPATH=str(Path(__file__).resolve().parent))
        code=('from rsi_pair import run_pair; '
              f'run_pair({str(self.baseline)!r},{str(self.candidate)!r},{str(self.output)!r},'
              f'123,4,[8,32],60,worker_script={str(self.fake)!r})')
        with (self.root/'coordinator.stderr').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-c',code],env=environment,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+5
                while not (self.baseline/'grandchild.pid').exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue((self.baseline/'grandchild.pid').exists())
                process.send_signal(signal.SIGTERM)
                self.assertNotEqual(process.wait(timeout=8),0)
            finally:
                if process.poll() is None:
                    process.kill();process.wait()
        self.assert_no_processes()
        report=json.loads(self.output.read_text())
        self.assertEqual(report['status'],'failed')
        self.assertTrue(report['process_groups_terminated'])

    def test_duplicate_batches_rejected_before_workers(self):
        with self.assertRaises(ValueError):
            run_pair(self.baseline,self.candidate,self.output,123,4,[8,8],10,worker_script=self.fake)
        self.assertFalse(self.order.exists())


if __name__=='__main__':unittest.main()
