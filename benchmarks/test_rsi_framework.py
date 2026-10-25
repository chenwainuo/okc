import copy
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from rsi.ledger import Ledger, digest


def observation_writer(ledger_path, gpu_path, source, number):
    Ledger(ledger_path, gpu_path).observe(f'obs-{number}', f'agent-{number}', [source], 'Research only')


class RSIFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor_source = self.write('anchor_source.py', 'original frozen serving source')
        self.anchor = self.write('anchor.json', {'objective':'3x the current benchmark','target_speedup':3,
                                               'local_primary_batches':[8],'regression_batches':[32], 'anchor_source_sha256':digest(self.anchor_source)})
        self.candidate = self.write('candidate.py', 'candidate')
        self.baseline = self.write('baseline.py', 'baseline')
        self.dependency = self.write('dependencies.bundle', 'frozen transitive sources')
        self.ledger = Ledger(self.root/'ledger.json', self.root/'gpu.json')
        self.ledger.initialize(self.anchor, 25)
        self.plan = {'kind':'kernel','hypothesis':'Reduce register spills for actual fixed dimensions',
                     'suite':[8,32], 'screen_seeds':[100], 'holdout_seeds':[200,300],
                     'min_pairs':20,'max_gpu_minutes':45,
                     'roi_prior':{'basis':'engineering_prior','speedup_range':[1,1.2]},
                     'baseline_isolation':'immutable_tree','candidate':str(self.candidate),
                     'baseline':str(self.baseline),'baseline_anchor_source':str(self.anchor_source),
                     'dependencies':[str(self.dependency),str(self.anchor_source)]}
        self.checklist = {k:True for k in ('baseline_bundle_complete','candidate_bundle_complete','baseline_isolated',
                                           'hypothesis_falsifiable','suite_and_holdouts_frozen','precision_preserved')}

    def write(self, name, value):
        p = self.root/name
        p.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return p

    def proposed(self):
        return self.ledger.propose('e1','author',self.plan)

    def started(self):
        exp = self.proposed()
        self.ledger.review('e1','reviewer',self.checklist)
        lease = self.ledger.start('e1','executor')
        return exp,lease

    def evidence(self, exp, mutate=None):
        manifest = {'process_terminal':True,'process_handle':'docker:id exited:0','runs':[]}
        for seed in self.plan['holdout_seeds']:
            report = {'status':'complete','candidate_sha256':digest(self.candidate),'baseline_sha256':digest(self.baseline),
                      'rsi_plan_sha256':exp['plan_sha256'],
                      'dependency_hashes':{str(self.dependency):digest(self.dependency),str(self.anchor_source):digest(self.anchor_source)},
                      'seed':seed,'session_id':f'session-{seed}', 'profiling':False,
                      'environment':dict(gpu_uuid='GPU-1',driver='595',torch='2.14',cuda='13',container_digest=self.ledger.status()['expected_container_digest'],exclusive_gpu=True),
                      'results':[{'bs':bs,'baseline_samples_us':[100.]*20,'candidate_samples_us':[80.]*20,
                                  'balanced_speedup':999,'bootstrap_95':[999,999]} for bs in self.plan['suite']]}
            if mutate:
                mutate(report)
            path = self.write(f'report-{seed}.json',report)
            correctness = self.write(f'correctness-{seed}.json',dict(report_sha256=digest(path),reference_pass=True,
                empty_local_eager_pass=True,empty_local_graph_pass=True,precision_preserved=True,reference_cases=5))
            manifest['runs'].append(dict(report=str(path),correctness=str(correctness)))
        return manifest

    def finish(self, mutate=None):
        exp,lease = self.started()
        manifest = self.evidence(exp,mutate)
        self.ledger.finish('e1','executor',lease['token'],manifest)
        return manifest

    def test_local_promotion_recomputes_claims_and_never_completes_goal(self):
        self.finish()
        result = self.ledger.validate('e1','independent')
        self.assertEqual(result['recomputed_results'][0]['rows'][0]['balanced_speedup'],1.25)
        self.ledger.promote('e1','independent')
        state = self.ledger.status()
        self.assertEqual(state['local_incumbent'],'e1')
        self.assertFalse(state['full_serving_goal_achieved'])
        self.assertEqual(state['objective']['target_speedup'],3)

    def test_author_and_executor_cannot_self_validate(self):
        self.proposed()
        with self.assertRaisesRegex(ValueError,'Author cannot'):
            self.ledger.review('e1','author',self.checklist)
        self.ledger.review('e1','reviewer',self.checklist)
        lease = self.ledger.start('e1','executor')
        exp = self.ledger.status()['experiments']['e1']
        self.ledger.finish('e1','executor',lease['token'],self.evidence(exp))
        for actor in ('author','executor'):
            with self.assertRaises(ValueError):
                self.ledger.validate('e1',actor)

    def test_anchor_gain_cannot_replace_existing_incumbent(self):
        self.finish()
        self.ledger.validate('e1', 'independent')
        self.ledger.promote('e1', 'independent')
        self.candidate = self.write('second_candidate.py', 'different candidate')
        self.plan = {**self.plan, 'candidate': str(self.candidate), 'holdout_seeds': [400, 500]}
        exp = self.ledger.propose('e2', 'author', self.plan)
        self.ledger.review('e2', 'reviewer', self.checklist)
        lease = self.ledger.start('e2', 'executor')
        def weaker_anchor_gain(report):
            for row in report['results']:
                row['candidate_samples_us'] = [90.] * 20
        self.ledger.finish('e2', 'executor', lease['token'], self.evidence(exp, weaker_anchor_gain))
        self.ledger.validate('e2', 'independent')
        before = self.ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'current-incumbent comparison'):
            self.ledger.promote('e2', 'independent')
        self.assertEqual(before, self.ledger.path.read_bytes())
        self.assertEqual(self.ledger.status()['local_incumbent'], 'e1')
        # Existing lifecycle semantics remain: promotion is not an idempotent command.
        with self.assertRaisesRegex(ValueError, 'Independent validated review'):
            self.ledger.promote('e1', 'independent')

    def test_promotion_cannot_skip_states(self):
        self.proposed()
        with self.assertRaises(ValueError):self.ledger.promote('e1','reviewer')
        with self.assertRaises(ValueError):self.ledger.start('e1','executor')

    def test_candidate_revision_invalidates_review_atomically(self):
        self.proposed();self.ledger.review('e1','reviewer',self.checklist)
        prior = self.ledger.path.read_bytes()
        self.candidate.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Artifact changed'):
            self.ledger.start('e1','executor')
        self.assertEqual(prior,self.ledger.path.read_bytes())
        self.assertFalse((self.root/'gpu.json').exists())

    def test_evidence_changes_after_validation_block_promotion(self):
        manifest = self.finish();self.ledger.validate('e1','independent')
        Path(manifest['runs'][0]['report']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Artifact changed'):
            self.ledger.promote('e1','independent')

    def test_nonfinite_samples_rejected_even_with_claimed_gain(self):
        self.finish(lambda r:r['results'][0]['candidate_samples_us'].__setitem__(0,float('nan')))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            self.ledger.validate('e1','independent')

    def test_duplicate_shape_rows_rejected(self):
        self.finish(lambda r:r['results'].__setitem__(1,copy.deepcopy(r['results'][0])))
        with self.assertRaisesRegex(ValueError,'suite rows'):
            self.ledger.validate('e1','independent')

    def test_regressions_rejected_despite_summary_claims(self):
        self.finish(lambda r:r['results'][1].update(candidate_samples_us=[110.]*20))
        with self.assertRaisesRegex(ValueError,'Noninferiority'):
            self.ledger.validate('e1','independent')

    def test_missing_correctness_rejected(self):
        exp,lease = self.started();manifest=self.evidence(exp)
        correctness=Path(manifest['runs'][0]['correctness']);x=json.loads(correctness.read_text());x.pop('empty_local_graph_pass');correctness.write_text(json.dumps(x))
        self.ledger.finish('e1','executor',lease['token'],manifest)
        with self.assertRaisesRegex(ValueError,'Correctness gate'):
            self.ledger.validate('e1','independent')

    def test_global_gpu_ownership_across_ledgers(self):
        exp,lease=self.started()
        other=Ledger(self.root/'other.json',self.root/'gpu.json');other.initialize(self.anchor,25)
        other.propose('e2','other-author',self.plan);other.review('e2','other-reviewer',self.checklist)
        with self.assertRaisesRegex(ValueError,'another ledger'):other.start('e2','other-executor')
        with self.assertRaisesRegex(ValueError,'ownership mismatch'):
            self.ledger.finish('e1','intruder',lease['token'],self.evidence(exp))
        self.assertEqual(self.ledger.status()['gpu_lease']['executor'],'executor')
        self.ledger.finish('e1','executor',lease['token'],self.evidence(exp))
        self.assertEqual(other.start('e2','other-executor')['experiment'],'e2')

    def test_consumed_holdout_cannot_be_reused(self):
        self.finish()
        with self.assertRaisesRegex(ValueError,'already consumed'):
            self.ledger.propose('e2','author',self.plan)

    def test_budget_admission_failure_does_not_acquire_gpu(self):
        small=Ledger(self.root/'small.json',self.root/'gpu.json');small.initialize(self.anchor,1)
        small.propose('e1','author',self.plan);small.review('e1','reviewer',self.checklist)
        with self.assertRaisesRegex(ValueError,'remaining run budget'):small.start('e1','executor')
        self.assertIsNone(small.status()['gpu_lease'])

    def test_objective_cannot_be_reinitialized_or_mutated(self):
        with self.assertRaises(ValueError):self.ledger.initialize(self.anchor,25)
        self.anchor.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Artifact changed'):self.ledger.status()

    def test_legacy_observation_cannot_enter_promotion_path(self):
        self.ledger.observe('legacy','root',[self.candidate],'Not preregistered')
        with self.assertRaises(ValueError):self.ledger.review('legacy','critic',self.checklist)
        with self.assertRaises(ValueError):self.ledger.promote('legacy','critic')

    def test_wrong_container_image_rejected(self):
        self.finish(lambda r:r['environment'].update(container_digest='sha256:'+'0'*64))
        with self.assertRaisesRegex(ValueError,'Container digest differs'):
            self.ledger.validate('e1','independent')

    def test_changed_gpu_or_runtime_between_holdouts_rejected(self):
        for key in ('gpu_uuid','driver','torch','cuda'):
            with self.subTest(key=key):
                # Reuse only isolated test fixtures; every case owns a fresh ledger.
                ledger_path=self.root/f'changed-{key}.json'
                self.ledger=Ledger(ledger_path,self.root/'gpu.json')
                self.ledger.initialize(self.anchor,25)
                self.finish(lambda r:r['environment'].update({key:'changed'}) if r['seed']==300 else None)
                with self.assertRaisesRegex(ValueError,'fingerprint changed'):
                    self.ledger.validate('e1','independent')

    def test_v1_remains_readable_but_cannot_execute_or_retroactively_promote(self):
        self.proposed();self.ledger.review('e1','reviewer',self.checklist)
        state=json.loads(self.ledger.path.read_text());state['schema_version']=1
        for key in ('benchmark_contract','expected_container_image','expected_container_digest'):
            state.pop(key)
        self.ledger.path.write_text(json.dumps(state))
        self.assertEqual(self.ledger.status()['schema_version'],1)
        with self.assertRaisesRegex(ValueError,'initialize a v2 ledger'):
            self.ledger.start('e1','executor')
        with self.assertRaisesRegex(ValueError,'no frozen image pin'):
            self.ledger.validate_evidence(state,state['experiments']['e1'])
        self.assertIsNone(self.ledger.status()['gpu_lease'])

    def test_contract_is_frozen_by_hash(self):
        state=self.ledger.status()
        self.assertEqual(state['benchmark_contract']['sha256'],digest(state['benchmark_contract']['path']))
        self.assertEqual(state['expected_container_digest'],state['expected_container_image'].split('@')[1])

    def test_one_shape_gain_does_not_hide_primary_aggregate_regression(self):
        target=json.loads(self.anchor.read_text());target.update(local_primary_batches=[8,16,32,64],regression_batches=[128])
        other_anchor=self.write('aggregate-anchor.json',target)
        self.ledger=Ledger(self.root/'aggregate-ledger.json',self.root/'gpu.json')
        self.ledger.initialize(other_anchor,25)
        self.plan['suite']=[8,16,32,64,128]
        def mutate(report):
            for row in report['results']:
                row['candidate_samples_us']=[100/(1.03 if row['bs']==8 else .982)]*20
        self.finish(mutate)
        with self.assertRaisesRegex(ValueError,'Primary aggregate'):
            self.ledger.validate('e1','independent')

    def test_changed_artifact_can_abort_without_promotion_or_lease_leak(self):
        exp,lease=self.started();self.candidate.write_text('unexpected mutation')
        proof={'process_terminal':True,'process_handle':'container stopped exit1'}
        with self.assertRaises(ValueError):self.ledger.abort('e1','intruder',lease['token'],proof,'failure')
        with self.assertRaises(ValueError):self.ledger.abort('e1','executor',lease['token'],{},'failure')
        self.ledger.abort('e1','executor',lease['token'],proof,'candidate changed during failed run')
        state=self.ledger.status()
        self.assertIsNone(state['gpu_lease'])
        self.assertIsNone(json.loads((self.root/'gpu.json').read_text()))
        self.assertEqual(state['experiments']['e1']['state'],'rejected')
        with self.assertRaises(ValueError):self.ledger.promote('e1','independent')

    def test_concurrent_writers_do_not_lose_events(self):
        workers=[multiprocessing.Process(target=observation_writer,args=(self.ledger.path,self.root/'gpu.json',self.candidate,i)) for i in range(4)]
        for worker in workers:worker.start()
        for worker in workers:worker.join(10);self.assertEqual(worker.exitcode,0)
        state=self.ledger.status()
        self.assertEqual(len(state['experiments']),4)
        self.assertEqual([e['sequence'] for e in state['events']],list(range(5)))


if __name__ == '__main__':
    unittest.main()
