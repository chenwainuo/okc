"""Independent CPU-only integration and adversarial redaction checks."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from .snapshot import build_snapshot, safe_text


class IndependentDashboardTests(unittest.TestCase):
    def test_sensitive_allowed_text_fields_are_redacted(self):
        samples = (
            'authorization: Bearer INDEPENDENT_SENTINEL',
            'Authorization = "Bearer INDEPENDENT_SENTINEL"',
            'debug {"token": "INDEPENDENT_SENTINEL"}',
            "debug {'api_key': 'INDEPENDENT_SENTINEL'}",
            'source /home/ubuntu/private/INDEPENDENT_SENTINEL.env',
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertNotIn('INDEPENDENT_SENTINEL', safe_text(sample))

    def test_actual_python_snapshot_roundtrips_strict_worker_schema(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            snapshot = build_snapshot(Path(directory), now=1800000000)
        self.assertIsNone(snapshot['queue']['counts']['gpu']['RUNNING'])
        self.assertIsNone(snapshot['agents']['observed_count'])
        self.assertEqual(snapshot['hardware']['status'], 'unknown')
        script = r'''
import { readFile } from 'node:fs/promises';
import { createWorker, sanitizeSnapshot } from './proj_rsi/dashboard/site/worker/backend.mjs';
import assert from 'node:assert/strict';
import vm from 'node:vm';
const schema = JSON.parse(await readFile('./proj_rsi/dashboard/schema.json', 'utf8'));
let input = ''; for await (const chunk of process.stdin) input += chunk;
const data = JSON.parse(input), token = 'independent-test-secret-'.repeat(3);
assert.deepEqual(sanitizeSnapshot(data, schema), data);
const page = await readFile('./proj_rsi/dashboard/ui.html', 'utf8');
const client = page.split('<script>')[1].split('function match(')[0];
const context = vm.createContext({document:{},console});
vm.runInContext(client + '\nglobalThis.dashboardModel=model;', context);
const recordedRunning = structuredClone(data);
recordedRunning.queue.jobs=[{id:'recorded',resource:'gpu',state:'RUNNING'}];
assert.equal(context.dashboardModel(recordedRunning).gpu.status,'unknown');
assert.ok(page.includes("?'Queue unknown'"));
assert.ok(page.includes("?'Scheduler unavailable'"));
assert.ok(!/\b(?:innerHTML|outerHTML)\s*=/.test(page));
let stored = null;
const bucket = { get: async () => stored && {etag:'1',text:async()=>stored},
 put: async (key, value) => {stored=value;return {etag:'1'};} };
const worker = createWorker({page:'<!doctype html>',schema,now:()=>1800000000});
const env = {BUCKET:bucket,INGEST_TOKEN:token};
const post = value => new Request('https://dashboard.test/_ingest', {
 method:'POST',headers:{authorization:'Bearer '+token,'content-type':'application/json'},
 body:JSON.stringify(value)});
assert.equal((await worker.fetch(post(data),env)).status,201);
const get = await worker.fetch(new Request('https://dashboard.test/api/snapshot'),env);
assert.equal(get.status,200); assert.deepEqual(await get.json(),data);
assert.equal((await worker.fetch(post(data),env)).status,409);
assert.equal((await worker.fetch(post({...data,command:'launch'}),env)).status,422);
for (const sample of ['authorization: Bearer INDEPENDENT_SENTINEL','debug {"token": "INDEPENDENT_SENTINEL"}']) {
 const malicious = structuredClone(data); malicious.queue.state_source=sample;
 assert.ok(!JSON.stringify(sanitizeSnapshot(malicious,schema)).includes('INDEPENDENT_SENTINEL'));
}
'''
        result = subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(snapshot),
                                text=True, capture_output=True, cwd=root, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
