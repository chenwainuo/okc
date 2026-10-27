"""CPU-only independent five-arm retained-result review; no GPU module is loaded."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ('incumbent', 'native', 'x4', 'stmx', 'new')
CANDIDATES = LABELS[1:]
SOURCE_SHA = {
    'native': 'ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377',
    'x4': 'e8447cbf0c113618462f5cbddc61d7f1e438ada2efa43130531f96c9fde768d3',
    'stmx': 'f6a765f0d0cf3174435f4981958015c242a53b5359f00064aa388b958cb9d9b6',
    'new': '61aad6a1d26ee56492c2b0a87b5b1e6c12c434fc166a0aad6e958a555becbf15',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(message):
    if message == 'exact zero complete output':
        return
    match = re.fullmatch(r'relative L2 error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\); (\d+) probe rows exact to ([0-9.e+-]+)', message)
    assert match, message
    # Recorded results are comfortably within the pinned task's .01/.05/.001 bounds.
    assert float(match[1]) <= .01 and float(match[2]) <= .05 and float(match[4]) <= .001, message


def analyze_data(data, path, support):
    assert data['status'] == 'NATIVE_X4_STMX_PROGRESS_DIAGNOSE_NEXT_LIMIT'
    assert data['versions_improved_both_methods'] == ['native']
    t = data['timing']; calls = t['all_timed_and_warmup_calls']
    assert len(calls) == 414 and sum(c['warmup'] for c in calls) == 25
    assert all(isinstance(c['us'], (float, int)) and not isinstance(c['us'], bool) and math.isfinite(c['us']) and c['us'] > 0 for c in calls)
    assert all(c['mode'] == 'warmup' and c['warmup'] is True and c['input_seed'] == 239999 for c in calls[:25])
    assert [c['label'] for c in calls[:25]] == [label for label in LABELS for _ in range(5)]
    f1calls = calls[25:214]; assert len(f1calls) == 189
    sequence = list(LABELS[:-1]) + ['new'] + list(reversed(LABELS[:-1]))
    counts = {}; raw = {}; last = {}
    for block, label in enumerate(sequence):
        rows = f1calls[21*block:21*(block+1)]
        assert all(c['label'] == label and c['mode'] == 'F1' and c['warmup'] is False for c in rows)
        assert [c['input_seed'] for c in rows] == list(range(240000+21*block, 240000+21*(block+1)))
        key = 'new' if label == 'new' else label + ('_before' if counts.get(label, 0) == 0 else '_after')
        counts[label] = counts.get(label, 0) + 1
        raw[key] = [c['us'] for c in rows]; last[label] = rows[-1]['input_seed']
    assert raw == t['F1_samples_us']
    med = {k: statistics.median(v) for k,v in raw.items()}; assert med == t['F1_medians_us']
    orders = []
    for seq in [(0,1,2,3,4), (4,3,2,1,0)]:
        orders.extend([seq[j:]+seq[:j] for j in range(5)])
    paired = []; pcalls = calls[214:]
    for rep in range(40):
        rows = pcalls[rep*5:rep*5+5]
        assert [c['label'] for c in rows] == [LABELS[i] for i in orders[rep%10]]
        assert all(c['mode'] == 'paired' and c['warmup'] is False and c['input_seed'] == 250000+rep for c in rows)
        paired.append({c['label']:c['us'] for c in rows})
    assert paired == t['paired_graph_us'] and len(pcalls) == 200
    assert t['terminal_last_timed_input_seed_per_policy'] == dict.fromkeys(LABELS,250039)
    for phase, expected, final in [('F1',last,240188),('paired',dict.fromkeys(LABELS,250039),250039)]:
        gate = t['strict_immediate_actual_timed_output_checks'][phase]
        assert len(gate['checks']) == 5 and {r['graph'] for r in gate['checks']} == set(LABELS)
        for row in gate['checks']:
            assert row['input_seed'] == expected[row['graph']]
            assert row['actual_timed_output_before_any_replay_or_mutation'] == 'PASS'
            assert row['XP_current_recorded_fixture_exact'] is False
            oracle(row['oracle'])
        assert gate['current_X_matches_last_actual_call_seed'] == final
        assert gate['W_packed_escape_immutable_counters_zero'] is True and gate['no_output_repair_replay'] is True
    recomputed = {
        'F1_new_gain_us':{l:min(med[l+'_before'],med[l+'_after'])-med['new'] for l in LABELS[:-1]},
        'F1_new_median_ratio':{l:min(med[l+'_before'],med[l+'_after'])/med['new'] for l in LABELS[:-1]},
        'paired_new_median_gain_us':{l:statistics.median(r[l]-r['new'] for r in paired) for l in LABELS[:-1]},
        'paired_new_median_ratio':{l:statistics.median(r[l]/r['new'] for r in paired) for l in LABELS[:-1]},
        'paired_new_wins':{l:sum(r[l]>r['new'] for r in paired) for l in LABELS[:-1]},
    }
    assert all(t[k] == v for k,v in recomputed.items())
    assert data['candidate_source_sha256'] == SOURCE_SHA
    assert data['parent_sha256'] == '2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
    assert data['incumbent_pick'] == {'kind':'rq','variant':'s4s5hz','KS':2}
    assert set(data['actual_JIT_SASS']) == set(CANDIDATES)
    for label, record in data['actual_JIT_SASS'].items():
        artifact = path.parent/Path(record['artifact']).name
        assert sha(artifact) == record['sha256']
        repeated = json.loads(json.dumps(support.actual_binary_gate(label,artifact.read_text())))
        assert all(record[k] == value for k,value in repeated.items())
        detail = record['X_layout_protocol_physical']; depth4 = label in ('x4','new')
        assert detail['layout']['all_coordinates_once'] and detail['layout']['stages'] == 64
        assert detail['protocol']['X_TMA_issues'] == 64 and detail['protocol']['X_phase_reuses'] == (60 if depth4 else 62)
        assert detail['protocol']['math_bar5_participants'] == list(range(256))
        assert detail['physical']['math_global_X_LDG'] == 0 and detail['physical']['static_LDSM'] == 32
        assert detail['physical']['static_X_TMA_steady_and_startup'] == 8 and len(detail['physical']['X_physical_blocks']) == 8
        assert record['producer_output_owner_proxy'] == 'EXACT_CANONICAL_MAIN_AND_FROZEN_SOURCE_PROOF'
        if label in ('stmx','new'):
            assert detail['candidate_actual_binary_gate']['status'] == 'PASS_ACTUAL_JIT_MAIN_AND_PHYSICAL_GATES'
    composition = data['actual_JIT_SASS']['new']['X_layout_protocol_physical']['composition']
    assert composition['status'] == 'PASS_EXACT_FULL_CUDA_COMMUTATIVITY_AND_BOTH_PARENT_INVERSES'
    assert composition['disjoint_source_edits'] and composition['inverse_targets'] == ['reverse_tma_x','reverse_tma_x4','reverse_tma_x_stmatrix']
    assert set(data['attrs']) == set(CANDIDATES)
    for label,r in data['attrs'].items():
        assert r['regs'] == 122 and r['local'] == 0 and r['threads'] == r['max_threads'] == 512
        assert r['rounded_register_pool'] == 65536 and r['occupancy_after_optin'] == 1
        assert r['smem'] == (170016 if label in ('x4','new') else 137232) and r['max_dynamic_after'] >= r['smem']
    assert data['packer_attrs'] == {} and data['GPU_pack_all65536_BF16_storage_bits'] == {}
    assert data['native_variants_storage_bits_gate']['status'] == 'NO_XP_BY_DESIGN_RAW_TMA_LDSM_BITS_UNPROBED'
    assert set(data['validation']) == set(LABELS)
    for checks in data['validation'].values():
        assert checks[0]['all4096_onehot_columns'] == 'EXACT' and checks[0]['random_FP64'] == 'PASS'
        assert {r['official_input_dist'] for r in checks if 'official_input_dist' in r} == {'normal','outliers'}
        for r in checks:
            if 'official_input_dist' in r:oracle(r['result'])
            if 'fixed_pointer_replay' in r:assert r['status'] == 'PASS'
    assert set(data['fresh_workspace_and_output']) == set(CANDIDATES)
    for r in data['fresh_workspace_and_output'].values():
        assert r['status'] == 'PASS' and r['workspace_bytes'] == 0 and r['XP'] == 'NONE'; oracle(r['oracle'])
    mutations = data['actual_timed_graph_checks']; assert len(mutations) == 25
    assert {(r['graph'],r['input']) for r in mutations} == {(l,n) for l in LABELS for n in ('original','zero','fresh_normal','fresh_outliers','all_probes')}
    for r in mutations:
        assert r['NaN_poison_overwritten'] is True and r['input_immutable'] is True and r['packed_poison_overwritten_byte_exact'] is False
        oracle(r['oracle'])
    assert set(data['profile']) == set(LABELS)
    for label,token in zip(LABELS,('rq<','rq_tma_x<','rq_tma_x4<','rq_tma_x_stmx<','rq_tma_x4_stmx<')):
        records=data['profile'][label]; assert len(records) == 1 and token in records[0]['name'] and 'rx_pack' not in records[0]['name']
    assert data['full_pipeline_in_timed_graph'] == dict.fromkeys(CANDIDATES,'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP')
    assert data['weights_inputs_counters'] == 'UNCHANGED_OR_RESET' and data['graph_cleanup'] == 'RESET_ALL_FIVE'
    distributions = {}
    for label in LABELS[:-1]:
        delta=[r[label]-r['new'] for r in paired]; midpoint=statistics.median(delta)
        distributions[label]={'median_control_minus_new_us':midpoint,'min_us':min(delta),'max_us':max(delta),
            'MAD_us':statistics.median(abs(v-midpoint) for v in delta),
            'win_tie_loss':[sum(v>0 for v in delta),sum(v==0 for v in delta),sum(v<0 for v in delta)]}
    return {'status':'INDEPENDENT_RECOMPUTATION_PASS','attempt_status':data['status'],'raw_calls':414,
        'actual_JIT_gates_reexecuted_from_saved_SASS':4,'current_pointer_descriptor_source_pin':'PASS',
        'strict_terminal_oracles':10,'mutation_oracles':25,'F1_medians_us':med,
        'paired_policy_medians_us':{l:statistics.median(r[l] for r in paired) for l in LABELS},
        **recomputed,'paired_distributions':distributions,'source_composition':'EXACT_COMMUTATIVITY_NOT_SPEED_ADDITIVITY',
        'raw_native_X_bits':'ALL_NATIVE_VARIANTS_UNPROBED_EXPLICIT_LIMIT','official_score':'NOT_MEASURED','result_sha256':sha(path)}


def evidence(path):
    sys.path.insert(0,str(ROOT/'proj_rsi/candidates/reverse_tma_x4_stmatrix_trial'))
    import native_x4_stmatrix_support as support
    support.verify_frozen();support.verify_lab()
    collection=path.parents[3]; inventory_path=collection/'inventory.json'; inventory=json.loads(inventory_path.read_text())
    assert inventory['job_id'] == 'native.x4.stmatrix.v1.01' and inventory['missing'] == [] and inventory['terminal_exit'] == 0
    for name,pin in inventory['files'].items():assert sha(collection/name) == pin, ('inventory',name)
    job=json.loads((ROOT/'proj_rsi/scheduling/jobs/native-x4-stmatrix-01.json').read_text())
    pins=inventory['dispatch']['source_pins']; assert pins == {**job['files'],job['review']['path']:job['review']['sha256']}
    assert len(pins) == 118
    for name,pin in pins.items():assert sha(ROOT/name) == pin, ('source',name)
    cleanup=inventory['cleanup_observation']
    assert cleanup['terminal_exit'] == 0 and cleanup['cleanup_failed'] is False and cleanup['container_exists'] is False
    assert cleanup['compute_pids'] == cleanup['guard_pids'] == cleanup['containers'] == [] and cleanup['gpu_memory_mib'] == 0 and cleanup['host_log_exists'] is True
    gate=runpy.run_path(str(ROOT/'proj_rsi/candidates/reverse_tma_x4_stmatrix_trial/timed_output_gate_proof.py'))['proof']()
    assert gate['GPU_calls'] == 0 and len(gate['negative_controls']) == 9
    data=json.loads(path.read_text());result=analyze_data(data,path,support);negative={}
    changes={
        'raw_seed':lambda v:v['timing']['all_timed_and_warmup_calls'][25].update(input_seed=1),
        'raw_timing':lambda v:v['timing']['all_timed_and_warmup_calls'][25].update(us=float('nan')),
        'paired_gain':lambda v:v['timing']['paired_new_median_gain_us'].update(x4=1.),
        'strict_terminal_fixture':lambda v:v['timing']['strict_immediate_actual_timed_output_checks']['F1']['checks'][0].update(input_seed=1),
        'mutation_poison':lambda v:v['actual_timed_graph_checks'][0].update(NaN_poison_overwritten=False),
        'resource_smem':lambda v:v['attrs']['new'].update(smem=137232),
    }
    for label,change in changes.items():
        altered=copy.deepcopy(data);change(altered)
        try:analyze_data(altered,path,support)
        except AssertionError:negative[label]='REJECTED'
        else:raise AssertionError(('negative admitted',label))
    sass=(path.parent/'result.new.jit.sass').read_text();assert 'LDSM.' in sass
    try:support.actual_binary_gate('new',sass.replace('LDSM.','LDS.',1))
    except AssertionError:negative['actual_JIT_instruction']='REJECTED'
    else:raise AssertionError('mutated SASS admitted')
    result.update(inventory_sha256=sha(inventory_path),inventory_files=len(inventory['files']),source_pins=len(pins),
        analysis_negative_controls=negative,terminal_gate_CPU_negative_controls=gate['negative_controls'],terminal_exit=0,
        cleanup_observation={k:v for k,v in cleanup.items() if k!='host_log_tail'})
    return result


if __name__ == '__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'result.json'
    result=evidence(path);(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
