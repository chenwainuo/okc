"""Completed production/diagnostic/X-prefetch evidence; source inclusion != selection."""

def append_production_cut(inputs, paths, root, findings, observations, reference):
    for prior in findings:
        if prior.get('immutable_id')=='improvement.research.gv53.earlyload.fixed.01':
            prior['currently_included']='Own subsequent production40d measured: M1keyed/M8parent; incumbent unchanged'
            prior['next_action']='See production40d result and new optional unpoolerXprefetch/W-only preparation; initial fixed gain remains historical evidence'
            prior['followup_context']='Initial pending-production wording records its original cut. Subsequent source17d44d scored3175.899650724306us; M8earlyload not selected.'
    specs = [
        ('production_bodies', 'research.gv53.selected.earlyload.bodies.01', 'gv53_selected_earlyload_trial_v1',
         'VALID_NO_GAIN_FIXED_BODY_DIAGNOSTIC',
         'Fixed production bodies: M1 paired−.048000365us/F1−.960000791us; M8+.031999312us/F1+.176000409us, but M8mean−.033333354us and option-first−.191999599us. Exact3native bodies/current outputs/cleanup pass; prepare selection was not executed.'),
        ('rq_exactso', 'research.rq.pooler32.exactso.counters.01', 'rq_pooler32_exactso_counters_v1',
         'VALID_NO_GAIN_COUNTER_DIAGNOSTIC',
         'Exact current poolerM32 RQ:12metrics/4passes,51.026432MBread,67.29%DRAM,54.07%longscoreboard samples. No PC/role/walltime fraction or score claim; outputs/source identity/cleanup pass.'),
        ('unpool_xprefetch', 'research.gv.unpool16.xprefetch.fixed.01', 'gv_unpool16_xprefetch_v1',
         'VALID_FIXED_UNPOOLER_M16_GAIN_TRANSFER_UNKNOWN',
         'Paired+.576000661us,mean+.522666766us,21W3L; both orders positive. F1before−.127999112us/after+.544000417us,average+.208000652us,drift+.671999529us. Actual126vs110regs/0spill/4CTA both. Current outputs/native identity/cleanup pass; one fixed H100 cell, no policy/official gain.'),
        ('production_official', '40d856e085274f51', 'gv53_selected_earlyload_v1',
         'OFFICIAL_SCORED_VALID_WHOLE_LOSS_REFINE',
         'Official SCORED9/9,3175.899650724306us; best3157.0384036526784us unchanged. Actual M1keyed selected (prepare paired+.016000122us); M8parent selected (earlyload prepare−.016000122us). Source inclusion does not establish execution. Crossjob whole difference−18.861247072us is not a paired causal increment; root REFINE.')]
    for key, attempt, version, classification, finding in specs:
        data = inputs[key]; evidence = reference(root, paths[key])
        findings.append({'id':attempt,'immutable_id':'improvement.'+attempt,'classification':classification,
            'currently_included':'Historical measured result; official incumbent unchanged',
            'scope':'muse_bf16_proj only; actual fixed local/counter/official scopes remain distinct',
            'finding':finding,'verification':'Frozen analysis and own raw evidence; no frozen source/DONE rewrite',
            'stacked_versions':[version],'stack_tested_jobs':[attempt] if key!='production_official' else [],
            'score_tested_jobs':[attempt] if key=='production_official' else [],
            'missing_reason':'Policy transfer or mechanism beyond own scope remains unknown; no arithmetic stacking',
            'next_action':'Preserve all signs; optional Xprefetch integration/W-only/RQ-lookahead are separately reviewed preparation',
            'evidence':[evidence]})
        groups = data['cells'] if key=='production_bodies' else {'16':data['timing']} if key=='unpool_xprefetch' else {}
        for cell, timing in groups.items():
            body = key=='production_bodies'
            gain = timing['paired_median_control_minus_option_us' if body else 'paired_median_us']
            f1 = timing['F1_average_us' if body else 'F1_endpoint_average_us']
            base = {'attempt_id':attempt,'version':version,'control':'current_GV_predxor','performance_admitted':True,
                'validity':'VALID_FIXED_BODY' if body else 'VALID_FIXED_CELL','shipping_gain':False,'hardware':'H100','evidence':evidence}
            observations.append(dict(base,observation_id='paired.'+attempt+'.'+cell+'.summary',shape=('poolerM' if body else 'unpoolerM')+cell+'.summary',gain_us=gain,F1_gain_us=f1,wins=timing['win_tie_loss'][0],ties=timing['win_tie_loss'][1],losses=timing['win_tie_loss'][2],pairs=24,observation_scope='Fixed body summary; no selector/policy/official transfer',F1_before_gain_us=timing['F1_before_minus_option_us' if body else 'F1_before_minus_new_us'],F1_after_gain_us=timing['F1_after_minus_option_us' if body else 'F1_after_minus_new_us'],paired_mean_gain_us=timing['paired_mean_us'],paired_control_first_gain_us=timing['control_first_median_us' if body else 'control_first_paired_median_us'],paired_new_first_gain_us=timing['option_first_median_us' if body else 'new_first_paired_median_us']))
            for i, point in enumerate(timing['paired_each_delta_us']):
                observations.append(dict(base,observation_id='paired.'+attempt+'.'+cell+'.round'+str(i),shape=('poolerM' if body else 'unpoolerM')+cell+'.round'+str(i),gain_us=point,F1_gain_us=None,wins=int(point>0),ties=int(point==0),losses=int(point<0),pairs=1,observation_scope='One fixed matched round; not independent shipping evidence'))
