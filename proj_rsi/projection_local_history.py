"""Preserve completed projection measurements without promoting fixed-cell gains."""
import hashlib
import json


def append_projection_results(inputs, paths, root, findings, observations, reference):
    def ev(key):
        return reference(root, paths[key])

    def add_finding(attempt, title, classification, text, scope, keys):
        findings.append({'id': title, 'immutable_id': 'improvement.' + attempt,
                         'classification': classification, 'currently_included': 'No; fixed-cell source or policy diagnostic only',
                         'scope': scope, 'finding': text,
                         'verification': 'Root DONE; frozen source, own terminals, all signed controls and cleanup retained',
                         'stacked_versions': [attempt], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
                         'missing_reason': 'No achieved cumulative stack or official improvement from these local measurements',
                         'next_action': 'Price a separately reviewed scoped follow-up against current selected GV9916; preserve mixed signs and actual source/execution distinction',
                         'evidence': [ev(k) for k in keys]})

    def obs(attempt, version, shape, control, gain, key, wins, ties, losses, pairs,
            before=None, after=None, ratio=None, scope='fixed selected-geometry paired summary'):
        observations.append({'observation_id': 'paired.' + attempt + '.' + shape + '.' + control,
                             'attempt_id': attempt, 'version': version, 'shape': shape, 'control': control,
                             'gain_us': gain, 'F1_before_gain_us': before, 'F1_after_gain_us': after,
                             'F1_gain_us': (before + after) / 2 if before is not None and after is not None else None,
                             'paired_ratio': ratio, 'wins': wins, 'ties': ties, 'losses': losses, 'pairs': pairs,
                             'observation_scope': scope, 'performance_admitted': True,
                             'validity': 'VALID_LOCAL_MEASUREMENT', 'hardware': 'H100',
                             'shipping_gain': False, 'evidence': ev(key)})

    key = 'gv53_keyed_local'; data = inputs[key]; review = inputs['gv53_keyed_review']; attempt = data['job_id']
    add_finding(attempt, 'GV53 keyed-half fixed M1 mixed and M8 positive local results',
                'VALID_LOCAL_FIXED_GAIN_WITH_M1_ESTIMATOR_CONFLICT_OFFICIAL_CHECK_REGRESSION',
                'M1 paired +0.063999556us but F1average -0.864000060us and marginal-median difference -0.016000122us; '
                'M8 paired +0.223999843us/F1average +0.176001340us. Preserve all raw/order/drift signs with no minimum-gain veto. '
                'Official CHECK ac49231d41d441f6 passed9/9 at3225.893245547708us with actual new keyed kernel served; '
                'it loses whole-policy to3157.0384036526784us. All30 seeds differ and unrelated reported routes change: '
                'cross-job score/target changes are not isolated cycle attribution. This is replacement of existing targets, '
                'not the older splitCTA source-included-but-unselected mechanism.',
                review['scope'], [key, 'gv53_keyed_review', 'gv53_keyed_official_diagnosis'])
    for row in data['rows']:
        shape = 'poolerM' + str(row['m']); values = row['paired_differences_all_us']
        obs(attempt, 'gv53_keyed_half_v1', shape + '.summary', 'current_GV_predxor',
            row['paired_median_control_minus_new_us'], key, row['paired_new_wins'], row['paired_ties'], row['paired_control_wins'], len(values),
            row['F1_control_before_us'] - row['F1_new_us'], row['F1_control_after_us'] - row['F1_new_us'])
        observations[-1]['paired_AB_gain_us'] = row['paired_AB_median_control_minus_new_us']
        observations[-1]['paired_BA_gain_us'] = row['paired_BA_median_control_minus_new_us']
        observations[-1]['difference_of_marginal_medians_gain_us'] = row['difference_of_marginal_paired_medians_control_minus_new_us']
        for index, gain in enumerate(values):
            obs(attempt, 'gv53_keyed_half_v1', shape + '.round' + str(index), 'current_GV_predxor', gain, key,
                int(gain > 0), int(gain == 0), int(gain < 0), 1, scope='one matched round; not an independent aggregate gain')

    key = 'rq_offset_local'; data = inputs[key]; review = inputs['rq_offset_review']; attempt = review['job_id']
    add_finding(attempt, 'RQ descriptor offset scoped unpooler gain with pooler regression',
                'VALID_LOCAL_UNPOOLER_FIXED_GAIN_POOLER_REGRESSION_SCOPE_REQUIRED',
                'Scored-selected ST6/FL35 fixed geometry: unpoolerM32KS2 paired +0.0480003655us/F1average +0.0800006092us; '
                'poolerM32KS4 paired -0.2240007743us while F1average +0.0800006092us. VALID_GAIN applies to unpooler only. '
                'Neither the unscoped source nor the new unpooler-only source has an achieved selected-policy/official gain. '
                'GV53+scopedRQ composition is pending, not an achieved stack; no arithmetic addition of isolated gains.',
                data['scope'], [key, 'rq_offset_review', 'rq_offset_native_supplement', 'rq_offset_cause_supplement'])
    for cell, row in data['cells'].items():
        paired = row['paired']; f1 = row['F1']
        obs(attempt, 'RQ_descriptor_offset_fixed', cell + 'M32.summary', 'current_RQ32_ST6_FL35',
            paired['median_control_minus_offset_us'], key, paired['offset_wins'], paired['ties'], paired['offset_losses'], paired['complete_pairs'],
            f1['control_before_minus_offset_us'], f1['control_after_minus_offset_us'], paired['median_control_over_offset'])
        for pair in row['per_pair']:
            gain = pair['signed_control_minus_offset_us']
            obs(attempt, 'RQ_descriptor_offset_fixed', cell + 'M32.seed' + str(pair['seed']), 'current_RQ32_ST6_FL35', gain, key,
                int(gain > 0), int(gain == 0), int(gain < 0), 1, ratio=pair['control_over_offset'],
                scope='one matched fixed-geometry seed; not an independent aggregate gain')

    key = 'gv16_deltakey_local'; data = inputs[key]; attempt = data['job_id']; timing = data['timing']
    raw_ref = next(r for r in data['evidence'] if r['path'].endswith('/result.json'))
    pairs = json.loads((root / raw_ref['path']).read_text())['fixed_cells']['16']['paired']
    add_finding(attempt, 'GV16 unpooler delta-key paired regression and positive F1',
                'VALID_LOCAL_FIXED_PAIRED_REGRESSION_POSITIVE_F1_NOT_STACKED',
                'UnpoolerM16 only: paired -0.0799996778us (7wins/2ties/15losses), F1before +0.0320002437us and after '
                '+0.1600012183us (average +0.0960007310us). Actual registers110→108,0spill, queried occupancy4 for both. '
                'All correctness/source/cleanup pass; primary paired disposition is VALID_REGRESSION, not a measured-gain stack. '
                'Pooler codegen spill counterevidence remains preserved and excluded. Reuse-SO robustness is CPU preparation, not a rerun outcome.',
                'One fixed scored-selected unpoolerM16 geometry versus source501d; no fresh policy or official result', [key])
    wins, ties, losses = timing['win_tie_loss']
    obs(attempt, 'gv_unpool16_deltakey_v1', 'unpoolerM16.summary', 'current_GV_predxor', timing['paired_delta_us'], key,
        wins, ties, losses, len(pairs), timing['F1_before_minus_new_us'], timing['F1_after_minus_new_us'], timing['paired_control_over_new_ratio'])
    for index, pair in enumerate(pairs):
        gain = pair['control'] - pair['deltakey']
        obs(attempt, 'gv_unpool16_deltakey_v1', 'unpoolerM16.round' + str(index), 'current_GV_predxor', gain, key,
            int(gain > 0), int(gain == 0), int(gain < 0), 1, scope='one matched fixed-geometry round; not an independent aggregate gain')

    # The corrected failure audit is frozen alongside its original and required
    # supplements. Its capture is historical, not a claim to cover later jobs.
    index = inputs['projection_failure_audit_index']; folder = (root / paths['projection_failure_audit_index']).parent
    for name, expected in index['sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen projection failure audit changed: ' + name)
