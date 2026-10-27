"""Append-only evidence reconciliation for three newly completed local jobs."""
import hashlib


def append_latest(inputs, paths, root, findings, observations, reference):
    def ev(key):
        return reference(root, paths[key])

    def finding(attempt, title, classification, text, scope, keys, extra=()):
        findings.append({'id': title, 'immutable_id': 'improvement.' + attempt,
                         'classification': classification, 'currently_included': 'No; separately measured local research',
                         'scope': scope, 'finding': text,
                         'verification': 'Root-reviewed DONE with original delivered evidence and separately named supplements preserved',
                         'stacked_versions': [attempt], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
                         'missing_reason': 'No official projection, whole-model or cross-distribution gain follows',
                         'next_action': 'Preserve every sign and exact source/selection scope; implement a new immutable causal follow-up',
                         'evidence': [ev(key) for key in keys] + list(extra)})

    def obs(attempt, version, shape, control, gain, key, wins, ties, losses, pairs,
            ratio=None, before=None, after=None, scope='same-run paired summary'):
        observations.append({'observation_id': 'paired.' + attempt + '.' + shape + '.' + control,
                             'attempt_id': attempt, 'version': version, 'shape': shape, 'control': control,
                             'gain_us': gain, 'F1_before_gain_us': before, 'F1_after_gain_us': after,
                             'F1_gain_us': (before + after) / 2 if before is not None and after is not None else None,
                             'paired_ratio': ratio, 'wins': wins, 'ties': ties, 'losses': losses, 'pairs': pairs,
                             'observation_scope': scope, 'performance_admitted': True,
                             'validity': 'VALID_LOCAL_MEASUREMENT', 'hardware': 'H100',
                             'shipping_gain': False, 'evidence': ev(key)})

    # The original delivered DONE is an immutable input; enrichment is linked
    # through its separately named index, never substituted at the original path.
    index = inputs['m3012_supplement_index']
    for key in ('original_done', 'byte_exact_supplement'):
        pin = index[key]
        if hashlib.sha256((root / pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('frozen M3012 evidence changed: ' + pin['path'])
    for pin in index['supplement_dependency_content_locations']:
        if hashlib.sha256((root / pin['preserved_content_path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('M3012 versioned supplement dependency changed')
    key = 'm3012_local'; data = inputs[key]; review = inputs['m3012_review']; attempt = review['job_id']
    finding(attempt, 'Actual M3012 T64 LUT whole-serving regression', 'VALID_LOCAL_M3012_WHOLE_SERVING_REGRESSION_NOT_SHIPPED',
            'Exact anchor minus LUT paired -12867.600024us, ratio0.132555299,0/8 wins; both fixed-fixture F1 endpoints '
            '-11582.239628/-11589.055657us. All39 checks and own timed terminals pass. Only actualM3012 normalEP8/busiest '
            'whole serving versus exact anchor; no M8/full14/model/official/3x gain. Original delivered DONE remains frozen; '
            'enriched resource/causal evidence is a separately indexed supplement. Driver-rounded shared allocation and active '
            'carveout remain UNKNOWN; preference/query occupancy do not establish measured allocation.',
            data['scope'], [key, 'm3012_review', 'm3012_supplement_index', 'm3012_supplement', 'm3012_wording'],
            [reference(root, data['result'])])
    obs(attempt, 'LUT_T64_M3012', 'M3012.summary', 'exact_anchor', data['paired_median_saving_us'], key,
        data['paired_LUT_wins'], 0, 8 - data['paired_LUT_wins'], 8, data['paired_median_anchor_over_LUT'],
        data['F1_before_saving_us'], data['F1_after_saving_us'])
    for pair in data['pairs']:
        gain = pair['signed_anchor_minus_LUT_us']
        obs(attempt, 'LUT_T64_M3012', 'M3012.seed' + str(pair['seed']), 'exact_anchor', gain, key,
            int(gain > 0), int(gain == 0), int(gain < 0), 1, pair['anchor_over_LUT'],
            scope='single matched fixture; not an independent aggregate gain')

    key = 'c11_preload_local'; data = inputs[key]; review = inputs['c11_preload_review']; attempt = review['job_id']
    finding(attempt, 'C11 preloaded metadata partial rescue still loses current C12',
            'VALID_LOCAL_FIXED_CELL_CURRENT_INCUMBENT_REGRESSION_WITH_V2_PROGRESS_NOT_SHIPPED',
            'Same-run pooler/unpooler currentC12 minus preload -3.216000/-8.384001us, each0/24. Exactv2 minus preload '
            '+2.192000/+4.208001us, each24/24; preserve this partial rescue without calling it an incumbent gain. '
            '155 checks,232 timed calls,12 own terminal sites and30 mutations pass. After graph drain/scope release, '
            'one33554432B BLAS cache entry clears to zero entries/zero allocated+reserved; external cleanup is clean. '
            'This new valid workspace-clear proof does not retroactively admit the old INVALID_CLEANUP attempt. '
            'Source-word byte savings are not measured HBM traffic or a serving-policy gain.',
            review['scope'], [key, 'c11_preload_review'])
    for cell, point in data['cells'].items():
        f1 = point['F1_medians_us']
        comparisons = [('currentC12', 'frontier', 'paired_control_minus_new_us', 'wins_vs_current', 'control_minus_new'),
                       ('currentC12', 'v2', 'paired_control_minus_v2_us', None, 'control_minus_v2'),
                       ('exact_v2_same_run', 'frontier', 'paired_v2_minus_new_us', 'wins_vs_v2', 'v2_minus_new')]
        for control, version, summary_key, wins_key, raw_key in comparisons:
            raw = point['signed_triples']; wins = point[wins_key] if wins_key else sum(r[raw_key] > 0 for r in raw)
            ties = sum(r[raw_key] == 0 for r in raw); before = after = None
            if control == 'currentC12':
                contender = f1['new' if version == 'frontier' else 'v2']
                before, after = f1['control_before'] - contender, f1['control_after'] - contender
            obs(attempt, version, cell + '.' + raw_key + '.summary', control, point[summary_key], key,
                wins, ties, len(raw) - wins - ties, len(raw), before=before, after=after,
                scope='same-run fixed-cell contrast; V2/new F1 is one within-block contrast' if control != 'currentC12' else 'same-run fixed-cell paired summary')
            if control != 'currentC12':
                observations[-1]['F1_gain_us'] = f1['v2'] - f1['new']
                observations[-1]['F1_estimator'] = 'single within-block v2 minus new contrast; no bracketing endpoints'
            for pair in raw:
                gain = pair[raw_key]
                obs(attempt, version, cell + '.' + raw_key + '.seed' + str(pair['seed']), control, gain, key,
                    int(gain > 0), int(gain == 0), int(gain < 0), 1, scope='single matched fixed-cell round; not an independent aggregate gain')

    key = 'marlin_sharedbank_local'; data = inputs[key]; attempt = data['job_id']
    finding(attempt, 'Marlin shared-bank scoped normal gain and skewed regression',
            'VALID_LOCAL_NORMAL_PAIRED_GAIN_SKEWED_REGRESSION_NOT_SHIPPED',
            'Normal original minus contrast +0.911999us; skewed -0.703998us. Preserve every distribution, original/default '
            'module control and blocked/interleaved estimator separately.41 source pins,60 oracle/zero checks,782 clean '
            'timing samples and46 own terminal oracles validate. Only retained exact terminal outputs were numerically '
            'checked; not every timing output. Per-row route receipts are checked, GPU fixture bytes are not separately '
            'archived. No routing-distribution/seed selector, whole-suite or official gain claim.',
            'M8 normal and skewed EP8 shared stored fixture bank; source availability is distinct from actual selected route',
            [key], data['evidence'])
    for dist, group in data['distributions'].items():
        for comparison, point in group['interleaved'].items():
            control, version = comparison.split('_minus_', 1); wins, ties, losses = point['win_tie_loss']
            obs(attempt, version, dist + '.' + comparison + '.interleaved.summary', control,
                point['all102_triplets_median_us'], key, wins, ties, losses, 102, point['median_ratio'],
                scope='102 interleaved matched triplets; repeated fixtures are not independent seeds')
            for seed, gain in point['per_seed_balanced_median_us'].items():
                obs(attempt, version, dist + '.' + comparison + '.interleaved.seed' + seed, control,
                    gain, key, int(gain > 0), int(gain == 0), int(gain < 0), 1,
                    scope='one seed median of six balanced orders; pairs/win/loss count one seed-level observation')
        for endpoint, point in group['blocked_estimator_matched_same_seed_differences'].items():
            wins, ties, losses = point['win_tie_loss']
            obs(attempt, 'contrast', dist + '.' + endpoint + '.blocked.summary', endpoint,
                point['median_same_fixture_delta_us'], key, wins, ties, losses, 17,
                scope='blocked same-seed endpoint estimator; separate from interleaved paired estimator')
            for seed, gain in point['per_seed_us'].items():
                obs(attempt, 'contrast', dist + '.' + endpoint + '.blocked.seed' + seed, endpoint,
                    gain, key, int(gain > 0), int(gain == 0), int(gain < 0), 1,
                    scope='single matched fixture blocked endpoint contrast; not an independent aggregate gain')
