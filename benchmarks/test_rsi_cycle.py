"""CPU tests of cycle contracts; no GPU, agents, or ledger mutations."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from rsi import cycle


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root/'input.txt'; self.input.write_text('baseline')
        self.ledger = self.root/'ledger.json'
        self.queue = self.root/'queue.json'
        self.gpu = self.root/'gpu.json'; self.gpu.write_text('null')
        protected = cycle.artifact(self.input)
        self.state = {'experiments': {}, 'anchor': protected, 'benchmark_contract': protected,
                      'measurement': protected, 'gpu_lease': None, 'budget': {'spent_usd': 1, 'usd': 5},
                      'local_incumbent': None, 'full_serving_goal_achieved': False}
        self.ledger.write_text(json.dumps(self.state))
        self.queue.write_text(json.dumps({'items': [{'id': 'old', 'rank': 0, 'status': 'rejected'}],
            'active_priority': [{'id': 'new', 'rank': 2, 'status': 'needs_harness'}]}))
        self.spec = {'id': 'cycle-1', 'kind': 'kernel', 'evidence_scope': 'development', 'tasks': []}
        for task_id, role, actor, filename in [('impl', 'implementer', '/root/impl', 'candidate.py'),
                                               ('review', 'critic', '/root/critic', 'review.json')]:
            self.spec['tasks'].append({'id': task_id, 'role': role, 'actor': actor, 'inputs': ['input.txt'],
                'owned_files': [filename], 'outputs': [filename], 'forbidden_actions': ['gpu_execution'],
                'completion_evidence': ['hashed report'], 'handoff_recipient': '/root', 'review_question': 'Does it preserve correctness?'})
        self.spec_path = self.root/'spec.json'; self.manifest = self.root/'cycle.json'

    def init(self):
        self.spec_path.write_text(json.dumps(self.spec))
        return cycle.initialize(self.spec_path, self.manifest, self.ledger, self.queue, self.gpu, self.root)

    def test_init_paths_protected_code_and_immutable_publish(self):
        first = self.init(); value = cycle.read(self.manifest)
        protected = {p['path'] for p in value['protected_inputs']}
        self.assertIn(str(cycle.ROOT/'rsi/ledger.py'), protected)
        self.assertIn(str(cycle.ROOT/'rsi/cycle.py'), protected)
        self.assertEqual(value['tasks'][0]['inputs'][0]['path'], str(self.input))
        self.assertEqual(cycle.check(self.manifest)['valid'], True)
        old = self.manifest.read_bytes()
        with self.assertRaises(FileExistsError): self.init()
        self.assertEqual(self.manifest.read_bytes(), old)
        self.assertEqual(first['manifest_sha256'], value['manifest_sha256'])

    def test_stale_blocks_ready_handoff(self):
        self.init(); self.input.write_text('mutated')
        result = cycle.status(self.manifest)
        self.assertFalse(result['check']['valid'])
        self.assertFalse(result['next_handoff']['ready'])
        self.assertTrue(result['check']['stale_inputs'])

    def test_role_separation(self):
        self.spec['tasks'][1]['actor'] = '/root/impl'
        with self.assertRaisesRegex(ValueError, 'must differ'): self.init()
        self.assertFalse(self.manifest.exists())

    def test_validator_separation(self):
        extra = copy.deepcopy(self.spec['tasks'][1]); extra.update(id='validate', role='validator', actor='/root/impl', owned_files=[], outputs=[])
        self.spec['tasks'].append(extra)
        with self.assertRaisesRegex(ValueError, 'Validator'): self.init()

    def test_parent_child_write_overlap(self):
        self.spec['tasks'][0].update(owned_files=['out'], outputs=['out/a.py'])
        self.spec['tasks'][1].update(owned_files=['out/review.json'], outputs=['out/review.json'])
        with self.assertRaisesRegex(ValueError, 'Overlapping'): self.init()

    def test_symlink_write_alias(self):
        (self.root/'real').mkdir(); (self.root/'alias').symlink_to(self.root/'real', target_is_directory=True)
        self.spec['tasks'][0].update(owned_files=['real/a'], outputs=['real/a'])
        self.spec['tasks'][1].update(owned_files=['alias/a'], outputs=['alias/a'])
        with self.assertRaisesRegex(ValueError, 'Overlapping'): self.init()

    def test_hardlink_write_alias(self):
        a=self.root/'a';a.write_text('same inode');os.link(a,self.root/'b')
        self.spec['tasks'][0].update(owned_files=['a'],outputs=['a'])
        self.spec['tasks'][1].update(owned_files=['b'],outputs=['b'])
        with self.assertRaisesRegex(ValueError,'Overlapping'): self.init()

    def test_manifest_or_spec_ownership_rejected(self):
        for path in ('cycle.json','spec.json'):
            self.spec['tasks'][0].update(owned_files=[path],outputs=[path])
            with self.assertRaisesRegex(ValueError,'manifest or spec'): self.init()

    def test_task_binding_and_same_role_ambiguity(self):
        self.spec['tasks'][0]['queue_item_id']='other'
        with self.assertRaisesRegex(ValueError,'binding'): self.init()
        del self.spec['tasks'][0]['queue_item_id']
        duplicate=copy.deepcopy(self.spec['tasks'][0]);duplicate.update(id='another',owned_files=[],outputs=[])
        self.spec['tasks'].append(duplicate)
        with self.assertRaisesRegex(ValueError,'One task per role'): self.init()

    def test_completion_contract_and_protected_ownership(self):
        del self.spec['tasks'][0]['completion_evidence']
        with self.assertRaisesRegex(ValueError, 'completion_evidence'): self.init()
        self.spec['tasks'][0]['completion_evidence'] = ['hashes']
        self.spec['tasks'][0].update(owned_files=['input.txt'], outputs=['input.txt'])
        with self.assertRaisesRegex(ValueError, 'protected'): self.init()

    def test_tamper_detected(self):
        self.init(); value = cycle.read(self.manifest); value['tasks'][1]['actor']='/root/impl'
        self.manifest.chmod(0o644); self.manifest.write_text(json.dumps(value))
        self.assertFalse(cycle.check(self.manifest)['valid'])
        self.assertIn('self-integrity', cycle.check(self.manifest)['errors'][0])

    def test_active_queue_not_legacy_or_obsolete_ledger_order(self):
        self.state['experiments']['obsolete']={'state':'hypothesis'}
        self.ledger.write_text(json.dumps(self.state)); self.init()
        result = cycle.status(self.manifest)
        self.assertEqual(result['queue_item']['id'], 'new')
        self.assertEqual(result['next_handoff']['task'], 'impl')
        self.assertTrue(result['next_handoff']['ready'])

    def test_live_context_changes_and_outputs_do_not_complete_task(self):
        self.init(); (self.root/'candidate.py').write_text('present')
        self.state['budget']['spent_usd']=2; self.ledger.write_text(json.dumps(self.state))
        result=cycle.status(self.manifest)
        self.assertIn('ledger',result['context_changes'])
        self.assertTrue(result['outputs'][0]['present'])
        self.assertFalse(result['outputs'][0]['completion_inferred'])
        self.assertEqual(result['next_handoff']['task'],'impl')

    def test_expired_lease_never_released_and_token_hidden(self):
        self.init(); self.gpu.write_text(json.dumps({'experiment':'other','deadline_unix':1,'token':'secret'}))
        result=cycle.status(self.manifest)
        self.assertFalse(result['next_handoff']['ready'])
        self.assertEqual(result['next_handoff']['reason'],'lease_requires_process_inspection')
        self.assertNotIn('secret',json.dumps(result))
        self.assertEqual(cycle.read(self.gpu)['token'],'secret')

    def test_development_and_method_cannot_promote(self):
        self.spec['experiment_id']='trial';self.state['experiments']['trial']={'state':'measured'}
        self.ledger.write_text(json.dumps(self.state)); self.init()
        self.assertEqual(cycle.status(self.manifest)['next_handoff']['reason'],'development_not_promotion_eligible')
        self.manifest=self.root/'method.json';self.spec['kind']='optimizer_method';self.init()
        self.assertEqual(cycle.status(self.manifest)['next_handoff']['reason'],'method_evaluator_unimplemented')

    def test_status_independent_of_current_directory(self):
        self.init(); before=os.getcwd()
        try:
            os.chdir('/')
            self.assertTrue(cycle.status(self.manifest)['check']['valid'])
        finally: os.chdir(before)


if __name__=='__main__': unittest.main()
