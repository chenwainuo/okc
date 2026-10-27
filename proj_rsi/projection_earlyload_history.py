"""Preserve the measured fixed-M8 early-load gain, without claiming transfer."""

def append_earlyload(inputs,paths,root,findings,observations,reference):
    key='earlyload_done';data=inputs[key];timing=data['timing'];attempt=data['job_id'];evidence=reference(root,paths[key])
    findings.append({'id':'GV53 earlier-load fixed M8 valid gain','immutable_id':'improvement.'+attempt,
        'classification':'VALID_FIXED_M8_GAIN_POLICY_OFFICIAL_TRANSFER_UNKNOWN',
        'currently_included':'No official incumbent change; production source being built',
        'scope':'One fixed official501d poolerM8 tuple and prospective Wseed5344; same original C12/escape bank, no prepare/policy/official result',
        'finding':'Paired +0.192000531us (15W0T9L), control-first +0.128000043us/new-first +0.271999277us; '
            'F1before +0.191999599us/after +0.128000975us, average +0.160000287us; all signed negatives retained. '
            'Actual source/SO/native PC-predicate-encoding match; both64regs/local0/shared17408/queriedoccupancy1. '
            'Earlier W/X load issue is realized with384 vs360 native instructions; no unique bandwidth/stall-time fraction is inferred. '
            'A production M1keyed+M8earlyload source is pending: no combined measured effect, selected-route or official gain claimed.',
        'verification':'11 artifact hashes/18 source pins;91 journal rows/81 clean calls,4 own terminals/10 mutations; internal/external cleanup valid',
        'stacked_versions':['gv53_earlyload_v1'],'stack_tested_jobs':[attempt],'score_tested_jobs':[],
        'missing_reason':'Fixed local gain is retained; combined source, actual policy selection and official transfer remain separate unmeasured facts',
        'next_action':'Build/review explicit production composition then measure its own current-parent policy/official result; never add isolated M1/M8 gains',
        'evidence':[evidence]+[reference(root,p) for p in data['evidence']]})
    def add(shape,gain,wins,ties,losses,n,scope,f1=None):
        observations.append({'observation_id':'paired.'+attempt+'.'+shape,'attempt_id':attempt,'version':'gv53_earlyload_v1',
            'shape':shape,'control':'current_GV_predxor','gain_us':gain,'F1_gain_us':f1,'wins':wins,'ties':ties,'losses':losses,'pairs':n,
            'observation_scope':scope,'performance_admitted':True,'validity':'VALID_FIXED_CELL','shipping_gain':False,'hardware':'H100','evidence':evidence})
    add('poolerM8.summary',timing['paired_median_us'],*timing['win_tie_loss'],24,'fixed same-bank M8 paired summary; no policy/official transfer',timing['F1_endpoint_average_us'])
    observations[-1].update(F1_before_gain_us=timing['F1_before_minus_new_us'],F1_after_gain_us=timing['F1_after_minus_new_us'],paired_control_first_gain_us=timing['control_first_paired_median_us'],paired_new_first_gain_us=timing['new_first_paired_median_us'],paired_mean_gain_us=timing['paired_mean_us'])
    for i,gain in enumerate(timing['paired_each_delta_us']):add('poolerM8.round'+str(i),gain,int(gain>0),int(gain==0),int(gain<0),1,'one fixed same-bank matched round; not independent shipping evidence')
