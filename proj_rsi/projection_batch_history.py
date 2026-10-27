"""Batch append completed keyed-choice CHECK evidence and GV16 robustness."""

def append_batch(inputs, paths, root, findings, observations, reference):
    def ev(key):
        return reference(root, paths[key])

    def obs(attempt, version, shape, key, gain, wins, ties, losses, pairs, scope, hardware='H100', ratio=None):
        observations.append({'observation_id': 'paired.' + attempt + '.' + shape,
                             'attempt_id': attempt, 'version': version, 'shape': shape,
                             'control': 'current_GV_predxor', 'gain_us': gain, 'F1_gain_us': None,
                             'paired_ratio': ratio, 'wins': wins, 'ties': ties, 'losses': losses, 'pairs': pairs,
                             'observation_scope': scope, 'hardware': hardware,
                             'performance_admitted': True, 'validity': 'VALID_LOCAL_MEASUREMENT',
                             'shipping_gain': False, 'evidence': ev(key)})

    key = 'keyed_choice_local'; data = inputs[key]; attempt = data['job_id']
    findings.append({'id': 'H200 keyed-choice same-prepare target positives and negatives',
                     'immutable_id': 'improvement.' + attempt + '.matched_prepare',
                     'classification': 'VALID_MATCHED_TUNING_INPUT_M1_POINT_GAIN_M8_DECLINE_WHOLE_CHECK_REGRESSION',
                     'currently_included': 'No; completed CHECK retained, not scored promotion',
                     'scope': 'Same H200 preparation exact rolled pooler template; tuning input, not paired official case',
                     'finding': 'M1 parent minus keyed +0.0480003655us chooses keyed; M8 -0.0160001218us chooses parent. '
                                'All four AB/BA samples and opposite order signs retained. Own official CHECK fe5c45197ca041bb '
                                'passes9/9 at3196.414179553723us, whole-policy above3157.0384036526784us; root REFINE. '
                                'Same-prepare tuning evidence is not an isolated official-case score gain or a selector forcing policy.',
                     'verification': 'Frozen result manifest and exact own archived profiles/choice receipts',
                     'stacked_versions': ['gv53_keyed_choice_v1'], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
                     'missing_reason': 'Target-matched evidence does not establish a whole-policy incumbent improvement',
                     'next_action': 'Retain optional choice source/negative M8; price a new causal follow-up with current actual controls',
                     'evidence': [ev(key), ev('keyed_choice_manifest')]})
    for row in data['choice_receipts']:
        values = row['signed_block_effects_us']; m = row['record']['m']
        obs(attempt, 'gv53_keyed_choice_v1', 'poolerM' + str(m) + '.prepare.summary', key,
            row['same_prepare_effect_us'], sum(v > 0 for v in values), sum(v == 0 for v in values), sum(v < 0 for v in values),
            len(values), 'same-H200-prepare tuning-input comparison; not an official-case pair', 'H200')
        observations[-1]['paired_AB_gain_us'] = row['AB_median_effect_us']
        observations[-1]['paired_BA_gain_us'] = row['BA_median_effect_us']
        observations[-1]['actual_prepare_choice'] = row['record']['choice']
        for i, gain in enumerate(values):
            obs(attempt, 'gv53_keyed_choice_v1', 'poolerM' + str(m) + '.prepare.round' + str(i), key,
                gain, int(gain > 0), int(gain == 0), int(gain < 0), 1,
                'one same-prepare tuning-input AB/BA sample; not independent official performance', 'H200')

    key = 'gv16_robust_local'; data = inputs[key]; attempt = data['job_id']
    findings.append({'id': 'GV16 reuse-SO three-weight robustness regression', 'immutable_id': 'improvement.' + attempt,
                     'classification': 'VALID_THREE_WEIGHT_ROBUSTNESS_REGRESSION_SIGNS_PRESERVED',
                     'currently_included': 'No; robust fixed-cell diagnostic, no gain stack',
                     'scope': 'Three fresh weights6203/6257/6287; shared frozen input bank, normal/outliers, exact measured SO reuse',
                     'finding': 'Equal-weight interleaved diagnostic median -0.063999556us across48 samples (22W1T25L), '
                                'not a production workload average. Preserve positive/negative per-weight, distribution, order and '
                                'blocked matched-seed estimators separately. No JIT/profiler;3.653058s measured run. '
                                'Actual110→108 registers still occupancy4 both. Prior matched F1 endpoints remain positive; '
                                'nonlinear estimator differences alone do not explain the earlier paired-negative/F1-positive result. '
                                'Performance root cause remains UNKNOWN with weight/address/time/cache/clock confounding; no detected bug '
                                'is not proof every implementation bug is absent.',
                     'verification': '270 journal rows/240 clean calls,78 FP64 checks,30 own terminals/30 mutations; post-scope/external cleanup pass',
                     'stacked_versions': ['gv_unpool16_deltakey_robust_v1'], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
                     'missing_reason': 'Regressing fixed-cell robustness diagnostic; no selected-policy/official increment',
                     'next_action': 'Use bounded schedule/weight falsifiers only if marginal value justifies; no automatic identical rerun or family rejection',
                     'evidence': [ev(key)] + data['evidence']})

    def walk(value, label):
        if isinstance(value, dict):
            if {'n', 'median_us', 'win_tie_loss'} <= value.keys():
                wins, ties, losses = value['win_tie_loss']
                obs(attempt, 'gv_unpool16_deltakey_robust_v1', label, key, value['median_us'], wins, ties, losses,
                    value['n'], ('supplementary reconstruction of prior original GV16 run; not a fresh robustness timing' if label.startswith('prior_estimator') else 'overlapping named diagnostic stratum; not independent shipping evidence'), ratio=value.get('median_ratio'))
                if label.startswith('prior_estimator'):
                    observations[-1]['measurement_source_attempt'] = 'research.gv.unpool16.deltakey.01'
            for name, child in value.items():
                walk(child, label + '.' + name)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, label + '.' + str(i))
    for name in ['per_weight', 'aggregate_equal_weight_diagnostic', 'aggregate_conditions_equal_weight_diagnostic', 'prior_estimator_supplement_newfile_only']:
        walk(data[name], name)
    for weight, group in data['per_weight'].items():
        for seed, row in group['per_seed'].items():
            base = 'W' + weight + '.X' + seed + '.' + row['dist']
            for mode in ['interleaved', 'blocked']:
                gain = row[mode + '_control_minus_new_us']
                obs(attempt, 'gv_unpool16_deltakey_robust_v1', base + '.' + mode + '.seed_median', key,
                    gain, int(gain > 0), int(gain == 0), int(gain < 0), 1,
                    'one matched-seed estimator; separate from raw trial order counts')
            for i, gain in enumerate(row['interleaved_orders']):
                obs(attempt, 'gv_unpool16_deltakey_robust_v1', base + '.interleaved.order' + str(i), key,
                    gain, int(gain > 0), int(gain == 0), int(gain < 0), 1,
                    'one shared-bank matched trial order; repeated fixtures not independent weights')

    key = 'rq256_local'; data = inputs[key]; review = inputs['rq256_review']; attempt = 'research.rq.compact256.half.fixed.01'
    effect = review['signed_effects']
    findings.append({'id': 'RQ compact256 HALF capacity succeeds but timing regresses',
                     'immutable_id': 'improvement.' + attempt,
                     'classification': 'ROOT_VALID_REGRESSION_AUTHOR_VALID_NO_GAIN_UMBRELLA_RETAINED',
                     'currently_included': 'No; fixed unpooler32 compact diagnostic, no policy gain',
                     'scope': review['scope'],
                     'finding': 'Compact paired -0.0160001218us (5W1T6L), F1before +0.0959988683us/after -0.2239998430us '
                                '(average -0.0640004873us). Root closes VALID_REGRESSION; frozen author recommendation '
                                'VALID_NO_GAIN remains unchanged. Loaded compact256threads/128regs/62624B shared queries2CTAs, '
                                'capacity264 for256 required; incumbent384threads/168regs queries1CTA. This establishes capacity, '
                                'not a measured residency trace or speed gain. Old384-thread HALF queries capacity132 below256 '
                                'required and is skipped before launch: numerical/performance unavailable, never zero-imputed.',
                     'verification': 'Own current-input/FP64/zero/boundary/counter/guard/terminal/mutation/drain/cleanup evidence retained',
                     'stacked_versions': ['rq_compact256_half_v1'], 'stack_tested_jobs': [attempt], 'score_tested_jobs': [],
                     'missing_reason': 'Fixed geometry paired and F1-average regress despite viable queried capacity; no selected-policy/official gain',
                     'next_action': 'Retain proven capacity/unsafe-old-arm distinction; test a new mechanism only when marginal information/speed value justifies cost',
                     'evidence': [ev(key), ev('rq256_review'), ev('rq256_native_supplement')]})
    wins, losses, ties = effect['compact_wins_losses_ties']
    obs(attempt, 'rq_compact256_half_v1', 'unpoolerM32.summary', key,
        effect['paired_median_control_minus_compact_us'], wins, ties, losses, 12,
        'same fixed fixture paired control FULL35 versus compactHALF262195', ratio=effect['paired_control_over_compact_ratio'])
    observations[-1]['control'] = 'current_RQ32_ST6_FL35'
    observations[-1]['F1_before_gain_us'] = effect['F1_control_before_minus_compact_us']
    observations[-1]['F1_after_gain_us'] = effect['F1_control_after_minus_compact_us']
    observations[-1]['F1_gain_us'] = effect['F1_endpoint_average_saving_us']
    for i, gain in enumerate(effect['signed_pair_savings_us']):
        obs(attempt, 'rq_compact256_half_v1', 'unpoolerM32.round' + str(i), key,
            gain, int(gain > 0), int(gain == 0), int(gain < 0), 1,
            'single matched fixed-fixture pair; oldHALF arm unavailable, not a zero-time control')
        observations[-1]['control'] = 'current_RQ32_ST6_FL35'
