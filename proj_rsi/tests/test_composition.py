"""Meaningful CPU contract tests; no candidate, torch or CUDA imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from proj_rsi.composition import core_digest, digest, file_sha, validate_stack


def change(name, axes=None, requires=(), scope=None, regions=None, admission='CPU_ONLY'):
    c = dict(id='change:'+name+':v1',kind='OPTIMIZATION_DELTA',base_sha256='a'*64,
             shape_scope=scope or [{'m':[128]}],semantic_axes=axes or {},
             requires=list(requires),conflicts=[],write_regions=regions or [],artifact_roots=[],artifact_bindings={},
             evidence=[],admission=dict(status=admission,retained=True,minimum_gain_us=None))
    c['core_sha256']=core_digest(c)
    return c


def registry(*changes):
    return dict(changes=list(changes),artifact_roots={},merge_recipes=[],full_stack_validations=[])


class CompositionTests(unittest.TestCase):
    def test_incompatible_axis_keeps_alternatives_selectable(self):
        a,b=change('packed',{'x.delivery':'packed'}),change('native',{'x.delivery':'tma'})
        r=registry(a,b)
        self.assertTrue(validate_stack([a['id']],r)['compatible'])
        result=validate_stack([a['id'],b['id']],r)
        self.assertFalse(result['compatible'])
        self.assertTrue(any(e['type']=='incompatible_axis' for e in result['errors']))

    def test_cycle_and_missing_dependency(self):
        a,b=change('a',requires=['change:b:v1']),change('b',requires=['change:a:v1'])
        result=validate_stack([a['id']],registry(a,b))
        self.assertEqual({'dependency_cycle','missing_dependency'}, {e['type'] for e in result['errors']})

    def test_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'source.cu';path.write_text('frozen')
            c=change('a');c['artifact_roots']=['kernel'];c['core_sha256']=core_digest(c)
            r=registry(c);r['artifact_roots']['kernel']={'files':{'source.cu':file_sha(path)}}
            c['artifact_bindings']={'kernel':digest(r['artifact_roots']['kernel'])};c['core_sha256']=core_digest(c)
            self.assertTrue(validate_stack([c['id']],r,root)['compatible'])
            path.write_text('changed')
            self.assertFalse(validate_stack([c['id']],r,root)['compatible'])

    def test_core_binding_and_typed_unique_ids(self):
        a=change('a');a['shape_scope']=[{'m':[32]}]
        result=validate_stack([a['id']],registry(a))
        self.assertIn('core_metadata_drift',{e['type'] for e in result['errors']})
        a=change('a');result=validate_stack([a['id']],registry(a,copy.deepcopy(a)))
        self.assertIn('invalid_or_duplicate_change_id',{e['type'] for e in result['errors']})

    def test_repeatable_tiny_gain_not_dropped(self):
        c=change('tiny',admission='MEASURED_PARENT_WIN')
        c['evidence']=[dict(paired_gain_us=0.001,repeatable=True)]
        result=validate_stack([c['id']],registry(c))
        self.assertTrue(result['compatible'])
        self.assertTrue(result['admissions'][c['id']]['retained'])
        self.assertIsNone(result['performance']['retention_minimum_gain_us'])
        self.assertFalse(result['production_eligible'])

    def test_individual_wins_are_not_additive_or_full_stack_validation(self):
        a,b=change('a',admission='MEASURED_INCUMBENT_WIN'),change('b',admission='MEASURED_INCUMBENT_WIN')
        result=validate_stack([a['id'],b['id']],registry(a,b))
        self.assertTrue(result['compatible'])
        self.assertFalse(result['production_eligible'])
        self.assertFalse(result['performance']['additive_gain_assumed'])
        self.assertIsNone(result['performance']['gain_estimate_us'])
        self.assertIn('prepare_budget_validation',result['required_runtime_controls'])

    def test_disjoint_shapes_can_coexist_but_need_full_scope_validation(self):
        a=change('small',{'delivery':'packed'},scope=[{'m':[16,32]}],regions=['same-source'])
        b=change('large',{'delivery':'tma'},scope=[{'m':[128]}],regions=['same-source'])
        result=validate_stack([a['id'],b['id']],registry(a,b))
        self.assertTrue(result['compatible']);self.assertEqual([],result['pending_rebases'])
        self.assertFalse(result['production_eligible'])
        self.assertIn('full_shape_scope_validation',result['required_runtime_controls'])

    def test_overlap_needs_explicit_merge_and_unverified_is_not_conflict(self):
        a,b=change('async',regions=['refill']),change('depth4',regions=['refill'])
        r=registry(a,b);result=validate_stack([a['id'],b['id']],r)
        self.assertTrue(result['compatible']);self.assertTrue(result['pending_rebases'])
        r['merge_recipes']=[dict(id='recipe:adapt:v1',changes=[a['id'],b['id']],explicit_merge=True,status='UNVERIFIED')]
        r['merge_recipes'][0]['core_sha256']=digest(r['merge_recipes'][0])
        result=validate_stack([a['id'],b['id']],r)
        self.assertTrue(result['compatible'])
        self.assertEqual(['recipe_adaptation_unverified'],[e['type'] for e in result['pending_rebases']])

    def test_invalid_version_retained_as_repair_not_family_ban(self):
        c=change('bad',admission='INVALID');r=validate_stack([c['id']],registry(c))
        self.assertTrue(r['compatible']);self.assertFalse(r['production_eligible'])
        self.assertEqual('invalid_version_requires_repair',r['pending_rebases'][0]['type'])

    def test_real_registry_native_x4_stores_and_async(self):
        r=json.loads((Path(__file__).parents[1]/'changes.json').read_text())
        base=['change:native-x:v1','change:x-depth4:v1','change:stmatrix-stores:v1']
        result=validate_stack(base,r)
        self.assertTrue(result['compatible'],result['errors'])
        self.assertFalse(result['production_eligible'])
        result=validate_stack(base[:2]+['change:x-async-refill:v1'],r)
        self.assertTrue(result['compatible'],result['errors'])
        self.assertIn('recipe_adaptation_unverified',{p['type'] for p in result['pending_rebases']})

if __name__=='__main__':unittest.main()
