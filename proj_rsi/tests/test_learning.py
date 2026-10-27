import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from proj_rsi import learning as L

class LearningTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads(L.DEFAULT.read_text())

    def seal(self, record):
        record['record_sha256']=L.digest(record)

    def test_real_registry_evidence_and_all_six_topics(self):
        result=L.validate(self.data)
        self.assertGreaterEqual(result['latest_lessons'],6)
        self.assertGreaterEqual(result['idea_versions'],6)
        self.assertEqual(result['GPU_calls'],0)
        self.assertTrue({'native-X','native-X-composition','EH','measurement-counter-contract','lossless-weight-codec','prepare-selection'} <= {x['context']['family'] for x in self.data['lessons']})

    def test_changed_lesson_without_new_version_fails_digest(self):
        self.data['lessons'][0]['title']='rewritten history'
        with self.assertRaisesRegex(L.ValidationError,'digest mismatch'):L.validate(self.data)

    def test_stale_evidence_bytes_fail_even_with_original_record_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=copy.deepcopy(self.data)
            for lesson in data['lessons']:
                for entry in lesson['evidence']+lesson['context']['sources']:
                    path=root/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes((L.ROOT/entry['path']).read_bytes())
            path=root/data['lessons'][0]['evidence'][0]['path'];path.write_bytes(path.read_bytes()+b'\nSTALE')
            with self.assertRaisesRegex(L.ValidationError,'stale evidence hash'):L.validate(data,root)

    def test_no_empty_outcomes_even_after_resealing(self):
        record=self.data['lessons'][0];record['observations']=[];self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'empty outcomes'):L.validate(self.data)

    def test_cannot_erase_negative_evidence_from_regression(self):
        record=self.data['lessons'][0]
        record['observations']=[o for o in record['observations'] if o['outcome']!='negative']
        self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'negative evidence cannot be erased'):L.validate(self.data)

    def test_negative_array_indices_and_malformed_pointer_escapes_rejected(self):
        for pointer in ('/rows/-1','/rows/+1','/rows/01','/rows/-','/bad~2key','/bad~'):
            with self.subTest(pointer=pointer):
                with self.assertRaises(L.ValidationError):L.pointer({'rows':[1,2],'bad~2key':3,'bad~':4},pointer)
        self.assertEqual(L.pointer({'a/b':{'~':[5]}},'/a~1b/~0/0'),5)

    def test_gain_disposition_requires_positive_evidence(self):
        record=self.data['lessons'][0];record['context']['disposition']='VALID_GAIN'
        record['observations']=[o for o in record['observations'] if o['outcome']=='negative'];self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'requires positive evidence'):L.validate(self.data)

    def test_nested_json_boolean_cannot_equal_numeric_fact(self):
        self.assertFalse(L.exact_json_equal({'x':[True]}, {'x':[1]}))
        self.assertFalse(L.exact_json_equal({'x':1.0}, {'x':1}))
        self.assertTrue(L.exact_json_equal({'x':[True,1,None]}, {'x':[True,1,None]}))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=copy.deepcopy(self.data)
            for lesson in data['lessons']:
                for entry in lesson['evidence']+lesson['context']['sources']:
                    path=root/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes((L.ROOT/entry['path']).read_bytes())
            record=data['lessons'][0];entry=record['evidence'][0];path=root/entry['path']
            payload=json.loads(path.read_text());payload['nested']={'x':True};path.write_text(json.dumps(payload))
            entry['sha256']=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
            record['observations'][0]['selector']={'json_pointer':'/nested','equals':{'x':1}};self.seal(record)
            with self.assertRaisesRegex(L.ValidationError,'does not match evidence'):L.validate(data,root)

    def test_observation_must_match_exact_bound_value(self):
        record=self.data['lessons'][0];record['observations'][0]['selector']['equals']=999999.0;self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'does not match evidence'):L.validate(self.data)

    def test_unsupported_cpu_to_gpu_transfer_promotion_rejected(self):
        idea=next(x for x in self.data['ideas'] if 'rq11' in x['id'])
        idea['transfer_status']='validated_on_H200';self.seal(idea)
        with self.assertRaisesRegex(L.ValidationError,'unsupported transfer'):L.validate(self.data)

    def test_cpu_observation_context_cannot_be_relabelled_gpu(self):
        record=next(x for x in self.data['lessons'] if x['id']=='rq11-full-residency-ledger')
        record['context']['hardware']={'kind':'GPU','model':'H200'};self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'CPU evidence cannot be promoted'):L.validate(self.data)

    def test_counterfactual_cannot_be_relabelled_measured_gain(self):
        idea=self.data['ideas'][0];idea['conditional_benefit']['kind']='measured_gain';idea['conditional_benefit']['value']=8.0;self.seal(idea)
        with self.assertRaisesRegex(L.ValidationError,'cannot be a measured'):L.validate(self.data)

    def test_unknown_benefit_and_cost_are_null_not_zero_claims(self):
        L.validate(self.data)
        self.assertIsNone(self.data['ideas'][0]['conditional_benefit']['value'])
        record=self.data['lessons'][0];record['cost']['gpu_seconds']=float('inf')
        with self.assertRaisesRegex(L.ValidationError,'invalid cost'):L.validate(self.data)

    def test_append_version_preserves_old_lesson_and_links_digest(self):
        old=self.data['lessons'][0];snapshot=copy.deepcopy(old);new=copy.deepcopy(old)
        new.update(version=2,supersedes=old['record_sha256'],title='A separately reviewed refined interpretation')
        self.seal(new);self.data['lessons'].append(new);self.data['registry_version']+=1
        result=L.validate(self.data)
        self.assertEqual(self.data['lessons'][0],snapshot)
        self.assertEqual(result['lesson_versions'],len(self.data['lessons']))
        self.assertEqual(result['latest_lessons'],len({x['id'] for x in self.data['lessons']}))

    def test_latest_context_excludes_unexecuted_pw39_and_release_controls(self):
        records={x['id']:x for x in L.latest(self.data['lessons'])}
        self.assertEqual(records['prepare-selection-is-experiment']['version'],2)
        self.assertFalse(any('PW39' in x for x in records['prepare-selection-is-experiment']['context']['comparators']))
        self.assertTrue(any('PW39' in x for x in records['prepare-selection-is-experiment']['next_test']['controls']))
        self.assertFalse(any('safely released' in x for x in records['rq11-full-residency-ledger']['context']['comparators']))

    def test_broken_version_parent_rejected(self):
        old=self.data['lessons'][0];new=copy.deepcopy(old);new.update(version=2,supersedes='0'*64);self.seal(new);self.data['lessons'].append(new)
        with self.assertRaisesRegex(L.ValidationError,'broken supersedes'):L.validate(self.data)

    def test_strict_unknown_field_rejected(self):
        self.data['lessons'][0]['measured_H200_gain_us']=10
        with self.assertRaisesRegex(L.ValidationError,'exact keys'):L.validate(self.data)

    def test_malformed_nested_selector_fails_as_validation_error(self):
        record=self.data['lessons'][0];record['observations'][0]['selector']=None;self.seal(record)
        with self.assertRaisesRegex(L.ValidationError,'selector must be an object'):L.validate(self.data)

    def test_query_links_specific_test_and_does_not_transfer_cpu_observation(self):
        results=L.query(self.data,'rq11 residency',hardware='H100')
        self.assertTrue(any(x['id']=='idea.rq11-live-residency-bound' for x in results))
        self.assertFalse(any(x['kind']=='lessons' and x['id']=='rq11-full-residency-ledger' for x in results))
        self.assertTrue(all('unproven' in x['transfer'] for x in results))
        self.assertTrue(all(x['next_test'] for x in results))

    def test_cli_is_read_only_and_reports_invalid_registry(self):
        before=L.DEFAULT.read_bytes()
        result=subprocess.run([sys.executable,'-m','proj_rsi.learning','validate'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'VALID')
        self.assertEqual(L.DEFAULT.read_bytes(),before)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.json';p.write_text('{}')
            result=subprocess.run([sys.executable,'-m','proj_rsi.learning','--registry',str(p),'validate'],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertEqual(json.loads(result.stderr)['status'],'INVALID')

if __name__=='__main__':unittest.main()
