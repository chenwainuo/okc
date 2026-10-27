"""Append one completed projection cut; diagnostics never become kernel gains."""

def append_completed_cut(inputs, paths, root, findings, observations, reference):
    def ev(key):return reference(root,paths[key])
    def finding(key,title,classification,text,scope,extra=()):
        data=inputs[key];attempt=data['job_id']
        findings.append({'id':title,'immutable_id':'improvement.'+attempt,'classification':classification,
            'currently_included':'No new scored-incumbent improvement','scope':scope,'finding':text,
            'verification':'Root DONE; frozen source/result/own-output/cleanup evidence retained',
            'stacked_versions':[attempt],'stack_tested_jobs':[attempt],'score_tested_jobs':[],
            'missing_reason':'Fixed-cell or diagnostic outcome, not official/policy gain',
            'next_action':'Preserve all signs and PROVEN/LIKELY/UNKNOWN scope; use a newly reviewed causal follow-up only when ROI justifies',
            'evidence':[ev(key)]+list(extra)})
    def obs(key,version,shape,control,gain,wins,ties,losses,n,f1=None,before=None,after=None,scope='fixed-cell paired summary',admitted=True):
        observations.append({'observation_id':'paired.'+inputs[key]['job_id']+'.'+shape+'.'+control,
            'attempt_id':inputs[key]['job_id'],'version':version,'shape':shape,'control':control,
            'gain_us':gain,'F1_gain_us':f1,'F1_before_gain_us':before,'F1_after_gain_us':after,
            'wins':wins,'ties':ties,'losses':losses,'pairs':n,'hardware':'H100','shipping_gain':False,
            'performance_admitted':admitted,'validity':'VALID_FIXED_CELL' if admitted else 'VALID_INTERFACE_DIAGNOSTIC_NO_KERNEL_GAIN_ADMISSION',
            'observation_scope':scope,'evidence':ev(key)})
    def raw(key,version,prefix,control,values,admitted=True,scope='one matched round; not independent shipping evidence'):
        for i,gain in enumerate(values):obs(key,version,prefix+'.round'+str(i),control,gain,int(gain>0),int(gain==0),int(gain<0),1,scope=scope,admitted=admitted)

    key='cut_lane';data=inputs[key];timing=data['timing']
    finding(key,'Lane-base fixed unpooler16 tie with mixed F1','VALID_FIXED_PRIMARY_TIE_MIXED_F1_NO_GAIN',
        'Primary paired0us (10W4T10L); F1before -0.128000975us/after +1.088000834us, average +0.479999930us; '
        'bracket drift1.216001809us. Actual S2R5/108regs/occupancy4 is validated, not a gain. '
        'No unchanged rerun is active; passing finite checks does not identify a unique performance cause.',
        'One fixed unpoolerM16 source501d comparator; no policy/official gain',data['evidence'])
    obs(key,'gv_unpool16_lanebase_v1','unpoolerM16.summary','current_GV_predxor',timing['paired_delta_us'],*timing['win_tie_loss'],24,
        timing['F1_endpoint_average_minus_new_us'],timing['F1_before_minus_new_us'],timing['F1_after_minus_new_us'])
    raw(key,'gv_unpool16_lanebase_v1','unpoolerM16','current_GV_predxor',timing['paired_each_delta_us'])

    key='cut_quarter';data=inputs[key];row=data['rows'][0]
    finding(key,'Quarter-stream fixed M8 regression and blocked conflict','VALID_FIXED_PAIRED_REGRESSION_POSITIVE_BLOCKED_CONFLICT',
        'Quarter-parent paired -0.144000165us, F1average +0.256000087us/conservative -0.320000574us; '
        'control drift1.152001321us. Actual four PRMT before firstMMA,64regs/0spill retained. '
        'Source header/repair work and dynamic counter follow-up are separate from loss attribution.',
        'One fixed M8 pooler prospective Wseed5344; no prepare/policy/official claim',[ev('cut_quarter_done')])
    vals=row['paired_differences_all_us']
    obs(key,'gv53_quarter_stream_v1','poolerM8.summary','current_GV_predxor',row['paired_median_control_minus_new_us'],
        row['paired_new_wins'],row['paired_ties'],row['paired_control_wins'],len(vals),row['F1_endpoint_average_control_minus_new_us'],
        row['F1_control_before_us']-row['F1_new_us'],row['F1_control_after_us']-row['F1_new_us'])
    observations[-1]['F1_conservative_gain_us']=row['F1_conservative_min_endpoint_control_minus_new_us']
    raw(key,'gv53_quarter_stream_v1','poolerM8','current_GV_predxor',vals)

    key='cut_sparse';data=inputs[key]
    finding(key,'Sparse quarter mask regression against parent and quarter','VALID_THREE_ARM_FIXED_REGRESSION_NO_RESCUE',
        'Sparse-parent paired -0.223999843us/blocked -0.271999277us; sparse-quarter -0.063999556us/blocked -0.032000244us. '
        'Mask and unchanged packet realized; no rescue or incumbent gain. Parent-quarter contrast remains separately retained. '
        'All24 signed values/order/marginal estimators stay attached to their own comparator; no broad-scope credit.',
        'Same-bank three-arm fixed M8; no selector/official score',[ev('cut_sparse_done')])
    for cmp in data['comparisons']:
        first,second=cmp['first_arm'],cmp['second_arm'];vals=cmp['all24signed_effects_us'];prefix='poolerM8.'+first+'_minus_'+second
        obs(key,second,prefix+'.summary',first,cmp['paired_median_first_minus_second_us'],cmp['second_arm_wins'],cmp['ties'],cmp['first_arm_wins'],len(vals),
            cmp['blocked_endpoint_average_first_minus_second_us'],scope='three-arm paired summary; F1 field is separately labeled blocked endpoint-average estimator')
        observations[-1]['blocked_conservative_gain_us']=cmp['blocked_conservative_minfirst_minus_maxsecond_us']
        observations[-1]['first_before_second_gain_us']=cmp['first_before_second_median_us']
        observations[-1]['second_before_first_gain_us']=cmp['second_before_first_median_us']
        raw(key,second,prefix,first,vals)

    key='cut_cabi';data=inputs[key]
    finding(key,'Thin C ABI validated seam and measured build cost','VALID_INTERFACE_BUILD_DIAGNOSTIC_NO_KERNEL_SCORE_GAIN',
        'Same-native seam validated: thin compile/link1.658049936s/full driver5.985287921s, sourceSASS/attrs/streams/owners/currentoutputs/cleanup pass. '
        'Scoped seam reuse ready; paired build saving UNKNOWN and no next standalone GPU test requested. '
        'All signed interface replay variability retained as diagnostics, not native or official gains.',
        'Four same-native interface cells; no native/score gain',[ev('cut_cabi_done')])
    for row in data['all_signed_interface_timing']:
        prefix='arm'+str(row['native_arm'])+'.M'+str(row['M']);vals=row['all_signed_deltas_us'];w,t,l=row['pair_wins_ties_losses']
        obs(key,'thin_C_ABI',prefix+'.summary','same_kernel_PyBind',row['paired_pybind_minus_cabi_us'],w,t,l,len(vals),
            row['F1_endpoint_average_pybind_minus_cabi_us'],scope='same-kernel interface replay variability; not admitted native/score benefit',admitted=False)
        observations[-1]['F1_conservative_gain_us']=row['F1_conservative_pybind_minus_cabi_us']
        raw(key,'thin_C_ABI',prefix,'same_kernel_PyBind',vals,admitted=False,scope='one same-kernel interface replay pair; no native gain admission')

    key='cut_top8';data=inputs[key]
    finding(key,'Top8 exponent codebook statistics falsify unchanged record capacity','VALID_WEIGHT_ONLY_CAPACITY_FALSIFICATION_NO_PERFORMANCE_MEASUREMENT',
        'Both seeds/four layer cells: global top8 exponent sets are contiguous and equal best affine8, so arbitrarydictionary adds zero coverage. '
        'Every observed group exceeds64 as well as32 repair records; odd-base unpooler repair reduction still cannot fit original128B transport. '
        'Nominal11bit payload figures are UNREPRESENTABLE without overflow/index/padding work. '
        'This falsifies the unchanged encoding/transport, not the codec family; no native/kernel/output/performance measurement or automatic GPU follow-up.',
        data['scope'],[ev('cut_top8_done')])
    findings[-1]['direct_score_gain_us']=0;findings[-1]['measured_speed_gain_us']=None

    key='cut_profile';data=inputs[key]
    finding(key,'Exact-SO three-arm hardware counter diagnostic','VALID_EXACTSO_AGGREGATE_DIAGNOSTIC_NO_SPEED_CREDIT',
        'Three exact intended arms,12 metrics each/four replay passes: same approximately51MB reads; quarter +36.651811% and sparse +16.368792% executed warp instructions vs parent. '
        'Long-scoreboard samples79.48–82.56% are not wall-time fractions or repair-PC attribution. '
        'Six snapshots P0/1980SM/2619mem constrain endpoints only. Profile-duration ranking differs clean timing: no native/official gain or uniquely allocated clean-loss cause. '
        'No JIT; exact intended source/SO/node/own-terminal/cleanup bindings retained.',
        'One fixed M8 exact-SO counter capture; aggregate diagnostic, no performance admission')
    findings[-1]['direct_score_gain_us']=0;findings[-1]['measured_speed_gain_us']=None;findings[-1]['repair_time_fraction']=None
