"""CPU-only, evidence-aware change composition. Never imports CUDA or candidates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = Path(__file__).with_name('changes.json')
ID = re.compile(r'^change:[a-z][a-z0-9-]*:v[1-9][0-9]*$')
ADMISSIONS = {'CPU_ONLY', 'MEASURED_PARENT_WIN', 'MEASURED_INCUMBENT_WIN',
              'VALID_NO_GAIN', 'VALID_REGRESSION', 'UNCERTAIN', 'INVALID'}
CONTROLS = ('source_and_actual_JIT_identity', 'resources_and_lifetimes',
            'full_basis_FP64_task_numerics', 'actual_terminal_timed_outputs',
            'fresh_zero_outlier_mutations', 'immutability_counters_cleanup',
            'balanced_actual_incumbent_parent_stack_comparison',
            'full_shape_scope_validation', 'prepare_budget_validation',
            'independent_full_stack_review', 'official_H200_validation')
CORE_FIELDS = ('id', 'kind', 'base_sha256', 'shape_scope', 'semantic_axes',
               'requires', 'conflicts', 'write_regions', 'artifact_roots', 'artifact_bindings')


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def core_digest(change: dict) -> str:
    return digest({k: change[k] for k in CORE_FIELDS})


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scopes_overlap(left: list[dict], right: list[dict]) -> bool:
    for a in left:
        for b in right:
            if all(not (set(a[k]) & set(b[k]) == set()) for k in a.keys() & b.keys()
                   if '*' not in a[k] and '*' not in b[k]):
                return True
    return False


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes registry root: ' + relative)
    return path


def validate_stack(ids: list[str], registry: dict, root: Path = ROOT) -> dict:
    errors, pending, recipes_used = [], [], []
    changes = registry.get('changes', [])
    by_id = {}
    for change in changes:
        cid = change.get('id', '')
        if not ID.fullmatch(cid) or cid in by_id:
            errors.append({'type': 'invalid_or_duplicate_change_id', 'id': cid})
        by_id[cid] = change
        try:
            if core_digest(change) != change['core_sha256']:
                errors.append({'type': 'core_metadata_drift', 'id': cid})
            if change['admission']['status'] not in ADMISSIONS:
                errors.append({'type': 'unknown_admission', 'id': cid})
        except (KeyError, TypeError):
            errors.append({'type': 'invalid_change_schema', 'id': cid})
    if len(ids) != len(set(ids)):
        errors.append({'type': 'duplicate_selection'})
    selected = set(ids)
    for cid in sorted(selected):
        if cid not in by_id:
            errors.append({'type': 'unknown_change', 'id': cid})
    # Validate the complete dependency graph, including omitted dependencies.
    visiting, visited, order = set(), set(), []
    def visit(cid):
        if cid in visiting:
            errors.append({'type': 'dependency_cycle', 'id': cid}); return
        if cid in visited or cid not in by_id:
            return
        visiting.add(cid)
        for dep in by_id[cid].get('requires', []):
            if dep not in by_id:
                errors.append({'type': 'unknown_dependency', 'id': cid, 'dependency': dep})
            visit(dep)
        visiting.remove(cid); visited.add(cid); order.append(cid)
    for cid in by_id:
        visit(cid)
    artifact_keys = set()
    for cid in sorted(selected & by_id.keys()):
        change = by_id[cid]
        for dep in change.get('requires', []):
            if dep not in selected:
                errors.append({'type': 'missing_dependency', 'id': cid, 'dependency': dep})
        artifact_keys.update(change.get('artifact_roots', []))
        for key, expected in change.get('artifact_bindings', {}).items():
            if digest(registry.get('artifact_roots', {}).get(key)) != expected:
                errors.append({'type': 'artifact_binding_metadata_drift', 'id': cid, 'artifact': key})
        if change.get('kind') == 'UNCLASSIFIED_ARTIFACT':
            pending.append({'type': 'artifact_semantics_need_classification', 'id': cid})
        if change.get('admission', {}).get('status') == 'INVALID':
            pending.append({'type': 'invalid_version_requires_repair', 'id': cid})
    for recipe in registry.get('merge_recipes', []):
        if set(recipe['changes']) <= selected:
            if recipe.get('core_sha256') != digest({k:v for k,v in recipe.items() if k != 'core_sha256'}):
                errors.append({'type': 'merge_recipe_metadata_drift', 'recipe': recipe['id']})
            if recipe.get('artifact_root'):
                artifact_keys.add(recipe['artifact_root'])
                if recipe.get('artifact_binding_sha256') != digest(registry.get('artifact_roots', {}).get(recipe['artifact_root'])):
                    errors.append({'type':'recipe_artifact_binding_drift','recipe':recipe['id']})
    # Pin source, proof, binary and manifest bytes. No candidate code is executed.
    for key in sorted(artifact_keys):
        artifact = registry.get('artifact_roots', {}).get(key)
        if artifact is None:
            errors.append({'type': 'missing_artifact_binding', 'artifact': key}); continue
        for name, expected in artifact.get('files', {}).items():
            try:
                path = _path(root, name)
                if not path.is_file() or file_sha(path) != expected:
                    errors.append({'type': 'source_or_file_drift', 'artifact': key, 'path': name})
            except ValueError as error:
                errors.append({'type': 'invalid_artifact_path', 'error': str(error)})
    recipes = registry.get('merge_recipes', [])
    for pos, cid in enumerate(sorted(selected & by_id.keys())):
        a = by_id[cid]
        for did in sorted(selected & by_id.keys())[pos + 1:]:
            b = by_id[did]
            if not scopes_overlap(a['shape_scope'], b['shape_scope']):
                continue
            if did in a['conflicts'] or cid in b['conflicts']:
                errors.append({'type': 'explicit_conflict', 'changes': [cid, did]})
            for axis in a['semantic_axes'].keys() & b['semantic_axes'].keys():
                if a['semantic_axes'][axis] != b['semantic_axes'][axis]:
                    errors.append({'type': 'incompatible_axis', 'axis': axis, 'changes': [cid, did]})
            overlap = sorted(set(a['write_regions']) & set(b['write_regions']))
            pair = {cid, did}
            recipe = next((r for r in recipes if set(r['changes']) == pair), None)
            if overlap and not (recipe and recipe.get('explicit_merge')):
                pending.append({'type': 'overlapping_edits_need_explicit_merge_recipe',
                                'changes': [cid, did], 'write_regions': overlap})
            if recipe:
                recipes_used.append(recipe['id'])
                if recipe['status'] != 'IMPLEMENTED_VERIFIED':
                    pending.append({'type': 'recipe_adaptation_unverified', 'recipe': recipe['id'],
                                    'changes': [cid, did]})
            # Dependency chains are intentional rebases; independent bases still
            # need a concrete combined-source recipe even when edits commute.
            related = cid in b['requires'] or did in a['requires']
            if a['base_sha256'] != b['base_sha256'] and not related and not recipe:
                pending.append({'type': 'different_bases_need_rebase', 'changes': [cid, did]})
    full = next((v for v in registry.get('full_stack_validations', [])
                 if set(v.get('changes', [])) == selected and selected
                 and v.get('status') == 'FULL_STACK_VALIDATED'
                 and set(CONTROLS) <= set(v.get('passed_controls', []))
                 and v.get('source_sha256') and v.get('evidence_bindings')
                 and v.get('source_path') in v['evidence_bindings']
                 and v['evidence_bindings'][v['source_path']] == v['source_sha256']
                 and v.get('actual_incumbent_win') is True), None)
    # Validation evidence must itself be hash-bound, never a Boolean assertion.
    if full:
        for name, expected in full['evidence_bindings'].items():
            try:
                path = _path(root, name)
                if not path.is_file() or file_sha(path) != expected:
                    errors.append({'type': 'full_stack_evidence_drift', 'path': name})
            except ValueError as error:
                errors.append({'type': 'invalid_full_stack_evidence_path', 'error': str(error)})
    return {'compatible': not errors, 'selected': [c for c in order if c in selected],
            'errors': errors, 'pending_rebases': pending,
            'required_runtime_controls': [] if full and not pending and not errors else list(CONTROLS),
            'production_eligible': bool(full and not pending and not errors),
            'merge_recipes': sorted(set(recipes_used)),
            'performance': {'additive_gain_assumed': False, 'gain_estimate_us': None,
                            'retention_minimum_gain_us': None,
                            'policy': 'Retain repeatable tiny gains; uncertainty requests measurement/stack tests; regression is version evidence, not a family ban.'},
            'admissions': {cid: by_id[cid]['admission'] for cid in sorted(selected & by_id.keys())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate-stack'); validate.add_argument('ids', nargs='+')
    args = parser.parse_args(argv)
    result = validate_stack(args.ids, json.loads(args.registry.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['compatible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
