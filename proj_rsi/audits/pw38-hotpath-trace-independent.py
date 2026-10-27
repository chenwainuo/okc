"""Read-only CPU review of the exact bounded hotpath trace package."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'proj_rsi/candidates/pw38_hotpath_trace_trial'
DRIVER = ROOT / 'proj_rsi/tools/pw38_hotpath_trace.py'
PINS = {
    'driver': '20e8b62b56aa18fa4aa8ac16867275ce1213a690c2393f52afafdffc99e3999e',
    'manifest': '39d98c081399f5370687e9b144a1c9d10abb96a764c8472eebc45e73620716ce',
    'PW38': '29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def review():
    assert sha(DRIVER) == PINS['driver']
    assert sha(TRIAL / 'manifest.json') == PINS['manifest']
    manifest = json.loads((TRIAL / 'manifest.json').read_text())
    for rel, digest in manifest['files'].items():
        assert sha(ROOT / rel) == digest, rel
    S = load(TRIAL / 'support.py', 'independent_hotpath_support')
    stack = S.stack()
    assert S.PW38_SHA == PINS['PW38']
    assert len(set(S.case_scope('all30'))) == 30
    assert (TRIAL / 'graph_metadata.py').read_bytes() == (
        ROOT / 'proj_rsi/candidates/pw38_selected_profile_trial/graph_metadata.py').read_bytes()
    # A distinct-symbol, parallel two-node graph: sum is deliberately not elapsed.
    nodes = [{'node_type': 0, 'demangled': 'foo(int)'},
             {'node_type': 0, 'demangled': 'bar()'}, {'node_type': 2}]
    events = [{'ph': 'X', 'name': 'PW38_TRACE_SETUP', 'cat': 'user_annotation', 'ts': 0, 'dur': 50}]
    for i in range(3):
        t = 100 + i * 100
        events.extend([
            {'ph': 'X', 'name': 'PW38_TRACE_REPLAY/' + str(i), 'cat': 'user_annotation', 'ts': t, 'dur': 80},
            {'ph': 'X', 'name': 'foo( int )', 'cat': 'kernel', 'ts': t + 10, 'dur': 30},
            {'ph': 'X', 'name': 'bar()', 'cat': 'kernel', 'ts': t + 20, 'dur': 30},
        ])
    trace = {'traceEvents': events}
    parsed = S.parse_trace(trace, nodes, 3)
    assert all(r['kernel_calls'] == 2 and r['kernel_duration_sum_us'] == 60
               and r['kernel_first_to_last_span_us'] == 40 for r in parsed['per_replay'])
    rejected = []
    for kind in ('wrong_multiplicity', 'kernel_crosses_scope', 'missing_final_replay_kernel', 'unattributed_memset'):
        bad = copy.deepcopy(trace)
        if kind == 'wrong_multiplicity':
            bad['traceEvents'][2]['name'] = 'bar()'
        elif kind == 'kernel_crosses_scope':
            bad['traceEvents'][2]['dur'] = 90
        elif kind == 'missing_final_replay_kernel':
            bad['traceEvents'].pop()
        else:
            bad['traceEvents'].append({'ph': 'X', 'name': 'memset', 'cat': 'gpu_memset', 'ts': 999, 'dur': 1})
        try:
            S.parse_trace(bad, nodes, 3)
        except AssertionError:
            rejected.append(kind)
        else:
            raise AssertionError(kind)
    # Partial aliases share physical arenas; only called reset lanes may change.
    plans = {'first': {'r': {'mode': 'reset0', 'offset': 2, 'reset_indices': [1], 'changes': {}}},
             'second': {'r': {'mode': 'reset0', 'offset': 0, 'reset_indices': [0], 'changes': {}}}}
    merged = S.weighted_plan(plans, {'first': 18, 'second': 0})
    assert merged['r']['reset_indices'] == [3]
    before = {'r': {'name': 'physical', 'values': [7, 11, 13, 17, 19]}}
    stack.counter_check(before, {'r': {'name': 'physical', 'values': [7, 11, 13, 0, 19]}}, merged, 1)
    try:
        stack.counter_check(before, {'r': {'name': 'physical', 'values': [0, 11, 13, 0, 19]}}, merged, 1)
    except AssertionError:
        rejected.append('zero_call_dormant_lane_corrupted')
    else:
        raise AssertionError('dormant corruption admitted')
    try:
        S.merge_plans([plans['first'], {'r': {'mode': 'progress', 'offset': 0, 'changes': {0: 1}}}])
    except AssertionError:
        rejected.append('conflicting_physical_protocols')
    else:
        raise AssertionError('protocol conflict admitted')
    source = DRIVER.read_text()
    ast.parse(source)
    assert "get('EH',{}).get('flags')" in source
    assert 'len(actual_args)==19' in source
    assert source.index("entry['profile_terminal']=terminal(name)") < source.index('prof.export_chrome_trace')
    assert 'range(11)' in source and 'range(3)' in source
    proof = load(TRIAL / 'cpu_proof.py', 'independent_run_author_cpu_proof').proof()
    assert len(proof['trace_negative_rejections']) == 12
    assert len(proof['own_terminal_negative_rejections']) == 8
    return {'decision': 'GO_BOUNDED_DIAGNOSTIC_GPU_UNRUN', 'pins': PINS,
            'manifest_files_verified': len(manifest['files']),
            'independent_negative_rejections': rejected,
            'author_cpu_proof': proof,
            'schedule_all30': {'cold_event_samples': 330, 'profile_replays': 90,
                               'mutations': 120, 'current_case_terminals': 180,
                               'final_retained_output_oracles': 30},
            'normal_inner_seconds': 600, 'normal_outer_seconds': 720,
            'GPU_or_NCU_or_production_prepare_calls_by_reviewer': 0}

if __name__ == '__main__':
    print(json.dumps(review(), indent=2))
