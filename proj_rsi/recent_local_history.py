"""Append reviewed local attempts without changing official scoring authority.

All inputs are immutable analysis/result files. Counter diagnostics and invalid
cleanup timings never become admitted performance gains.
"""
import json


def append_recent(inputs, paths, root, findings, observations, reference):
    def evidence(key):
        return reference(root, paths[key])

    def finding(attempt, title, classification, text, scope, keys, extra=()):
        findings.append({
            'id': title, 'immutable_id': 'improvement.' + attempt,
            'classification': classification, 'currently_included': 'No; independent local research',
            'scope': scope, 'finding': text,
            'verification': 'Root-reviewed completed disposition; exact source/results and validity limits retained',
            'stacked_versions': [attempt], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
            'missing_reason': 'No projection official or whole-model gain follows from this local attempt',
            'next_action': 'Preserve source, signed controls and failure cause; test a new immutable causal follow-up, not a family-wide rejection',
            'evidence': [evidence(k) for k in keys] + list(extra),
        })

    def observation(attempt, version, shape, control, gain, pairs, wins, ties, loss,
                    key, ratio=None, before=None, after=None, scope='paired summary', valid=True):
        observations.append({
            'observation_id': 'paired.' + attempt + '.' + shape + '.' + control,
            'attempt_id': attempt, 'version': version, 'shape': shape, 'control': control,
            'gain_us': gain, 'F1_gain_us': (before + after) / 2 if before is not None and after is not None else None,
            'F1_before_gain_us': before, 'F1_after_gain_us': after,
            'paired_ratio': ratio, 'wins': wins, 'ties': ties, 'losses': loss, 'pairs': pairs,
            'observation_scope': scope, 'performance_admitted': valid,
            'validity': 'VALID_LOCAL_MEASUREMENT' if valid else 'INVALID_CLEANUP_RAW_DIAGNOSTIC_ONLY',
            'hardware': 'H100', 'shipping_gain': False, 'evidence': evidence(key),
        })

    for key, title in [('marlin_tiles_local', 'Original Marlin alternate tile local regression'),
                       ('marlin_four_local', 'Original Marlin four-block paired gain with F1 conflict')]:
        data = inputs[key]; attempt = data['job_id']
        raw_ref = next(r for r in data['evidence'] if r['path'].endswith('/result.json'))
        raw = json.loads((root / raw_ref['path']).read_text())
        four = key == 'marlin_four_local'
        text = ('Primary paired original minus four-block +0.591997us (10/12), shim minus four-block +0.463996us (8/12); '
                'both original F1 endpoints -1.535997/-2.560005us and shim endpoints -1.279995/-2.591997us. '
                'VALID_GAIN preserves the positive primary point and opposite F1; no significance cutoff, aggregate all-positive claim or official gain.'
                if four else
                'Alternate tile loses paired original -6.784000us and default shim -7.055998us, each0/12; both F1 brackets lose. '
                'Anchor/default-shim +0.175998us control difference remains separate; this is a tile-version regression.')
        finding(attempt, title, 'VALID_LOCAL_PAIRED_GAIN_F1_CONFLICT_NOT_SHIPPED' if four else 'VALID_LOCAL_TILE_REGRESSION_NOT_SHIPPED',
                text, 'M8 normal EP8 actual original/default shim/contrast; no full14/model/official/3x claim', [key], data['evidence'])
        for name, point in data['paired'].items():
            control, variant = name.split('_minus_', 1)
            endpoints = data['F1_control_minus_contrast'].get(name, {})
            wins, ties, losses = point['win_tie_loss']
            observation(attempt, variant, name + '.summary', control, point['median_us'], sum([wins, ties, losses]), wins, ties, losses,
                        key, point['median_ratio'], endpoints.get('before_us'), endpoints.get('after_us'))
            for index, pair in enumerate(raw['paired']):
                gain = pair[control] - pair[variant]
                observation(attempt, variant, name + '.round' + str(index), control, gain, 1, int(gain > 0), int(gain == 0), int(gain < 0),
                            key, pair[control] / pair[variant], scope='single matched round; not an independent aggregate gain')

    key = 'scatter_local'; data = inputs[key]; review = inputs['scatter_review']; attempt = review['job_id']
    finding(attempt, 'Two-value scatter loses best LUT and partly repairs failed rolled version',
            'VALID_LOCAL_ACTUAL_BEST_REGRESSION_WITH_FAILED_PARENT_PROGRESS_NOT_SHIPPED',
            'Actual best v1 minus scatter -82.432002us,0/8; failed rolled-v2 minus scatter +24.464004us,8/8. '
            'Fixed exact-workload F1 versus v1 is -4.767999/+0.192001us; versus failed v2 +17.056003/+18.784001us. '
            'The partial repair does not become a best-policy gain; all three arms and signs remain separate.',
            review['scope'], [key, 'scatter_review'], [reference(root, data['result_path'])])
    for name, point in data['comparisons'].items():
        control, variant = name.split('_versus_', 1)
        before = after = None
        if variant == 'two_value_scatter':
            prefix = 'v1' if control == 'v1_full_lut' else 'v2'
            before = data['F1'][prefix + '_before_minus_new_us']; after = data['F1'][prefix + '_after_minus_new_us']
        observation(attempt, variant, name + '.summary', control, point['median_control_minus_variant_us'], point['complete_pairs'],
                    point['variant_wins'], 0, point['variant_losses'], key, point['median_control_over_variant'], before, after)
        for index, gain in enumerate(point['signed_control_minus_variant_us']):
            seed = data['per_seed'][index]['seed']
            observation(attempt, variant, name + '.seed' + str(seed), control, gain, 1, int(gain > 0), int(gain == 0), int(gain < 0), key,
                        scope='single matched routing seed; not an independent aggregate gain')

    key = 'c11_cleanup_local'; data = inputs[key]; attempt = 'research.gv.codec.frontier.two-cell.01'
    finding(attempt, 'C11 frontier v2 completed raw evidence with invalid internal cleanup',
            'INVALID_SETUP_INVALID_CLEANUP_RAW_DIAGNOSTIC_ONLY',
            '84 CPU reconstruction checks,182 timing rows,8 own terminals and20 mutations pass, but post-scope allocated33554432B '
            'and trial exit1 invalidate performance admission. External GPU/container cleanup is clean and does not repair internal lifetime validity. '
            'Raw pooler paired -5.408000us/unpooler -12.656000us each0/24 remain diagnostic only. '
            'Format/source-request byte reductions are not measured HBM traffic or a serving-policy gain.',
            'Frozen C11 v2 two-cell source versus native controls; INVALID_CLEANUP, no admitted performance', [key])
    for cell, point in data['cells'].items():
        observation(attempt, 'C11_frontier_v2', cell, 'native_control', point['unadmitted_paired_control_minus_new_us'], 24,
                    point['paired_new_wins'], 0, 24 - point['paired_new_wins'], key,
                    before=point['F1_medians_us'][0] - point['F1_medians_us'][1],
                    after=point['F1_medians_us'][2] - point['F1_medians_us'][1], scope='unadmitted raw diagnostic; internal cleanup failed', valid=False)

    for key, review_key, title in [('counter_v1_local', 'counter_v1_review', 'Exact-v1 up counters attempt1 invalid empty-fixture oracle'),
                                   ('counter_v2_local', 'counter_v2_review', 'Exact-v1 up counters repaired diagnostic without performance admission')]:
        data = inputs[key]; review = inputs[review_key]; attempt = review['job_id']; v2 = key == 'counter_v2_local'
        text = ('Optional exact-v1 up diagnostic collected exact2 nodes/all8 metric values and own terminals after empty exact-zero repair; '
                'same source/SO and fixture identities retained,10 replay passes per kernel. Valid diagnostic/VALID_NO_GAIN; '
                'counters do not uniquely attribute a producer bottleneck and earn no performance or whole-serving gain.' if v2 else
                'Empty fixture incorrectly used the original nonlocal-rejecting reference; failure occurred before profiler start. '
                'G6/G7 numerical and bitexact stage checks passed, but no profiled kernel/hardware metrics were collected. '
                'INVALID_SETUP attempt retained; v2 is a separately pinned repair, not an overwrite.')
        result = data.get('result_path', data.get('source_result'))
        finding(attempt, title, 'VALID_DIAGNOSTIC_NO_PERFORMANCE_ADMISSION' if v2 else 'INVALID_SETUP_NO_COUNTER_COLLECTION',
                text, 'Optional exact-v1 up hardware diagnostic; no speedup observation or admission', [key, review_key], [reference(root, result)])
