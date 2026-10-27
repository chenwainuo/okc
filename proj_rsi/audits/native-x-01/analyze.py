"""CPU-only independent recomputation of native-X attempt01 retained evidence."""
import hashlib,json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LABELS=('incumbent','ds','dp','new')
def analyze(path):
    sys.path.insert(0,str(HERE.parents[1]/'candidates/reverse_tma_x'))
    sys.path.insert(0,str(HERE.parents[1]/'candidates/reverse_tma_x_trial'))
    import native_x_support as support
    support.verify_frozen();support.verify_lab()
    d=json.loads(path.read_text());t=d['timing'];calls=t['all_timed_and_warmup_calls']
    assert len(calls)==295 and sum(c['warmup'] for c in calls)==20
    raw={};paired=[];last={};strict=t['strict_immediate_actual_timed_output_checks']
    f1calls=[c for c in calls if c['mode']=='F1'];assert len(f1calls)==147
    sequence=['incumbent','ds','dp','new','dp','ds','incumbent']
    counts={};f1={}
    for block,label in enumerate(sequence):
        rows=f1calls[block*21:(block+1)*21];assert all(c['label']==label for c in rows)
        assert [c['input_seed'] for c in rows]==list(range(240000+21*block,240000+21*(block+1)))
        key='new' if label=='new' else label+('_before' if counts.get(label,0)==0 else '_after')
        counts[label]=counts.get(label,0)+1;raw[key]=[c['us'] for c in rows];last[label]=rows[-1]['input_seed']
    assert raw==t['F1_samples_us']
    med={k:statistics.median(v) for k,v in raw.items()};assert med==t['F1_medians_us']
    for phase,expected in [('F1',dict(last)),('paired',dict.fromkeys(LABELS,250031))]:
        gate=strict[phase];assert len(gate['checks'])==4
        for row in gate['checks']:
            assert row['input_seed']==expected[row['graph']]
            assert row['actual_timed_output_before_any_replay_or_mutation']=='PASS'
            assert row['XP_current_recorded_fixture_exact']==(row['graph'] in ('ds','dp'))
        assert gate['W_packed_escape_immutable_counters_zero'] and gate['no_output_repair_replay']
    pcalls=[c for c in calls if c['mode']=='paired'];assert len(pcalls)==128
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
    for rep in range(32):
        rows=pcalls[4*rep:4*rep+4]
        assert [c['label'] for c in rows]==[LABELS[i] for i in orders[rep%8]]
        assert all(c['input_seed']==250000+rep for c in rows)
        paired.append({c['label']:c['us'] for c in rows})
    assert paired==t['paired_graph_us']
    recomputed={
      'F1_new_gain_us':{l:min(med[l+'_before'],med[l+'_after'])-med['new'] for l in LABELS[:3]},
      'F1_new_median_ratio':{l:min(med[l+'_before'],med[l+'_after'])/med['new'] for l in LABELS[:3]},
      'paired_new_median_gain_us':{l:statistics.median(r[l]-r['new'] for r in paired) for l in LABELS[:3]},
      'paired_new_median_ratio':{l:statistics.median(r[l]/r['new'] for r in paired) for l in LABELS[:3]},
      'paired_new_wins':{l:sum(r[l]>r['new'] for r in paired) for l in LABELS[:3]}}
    assert all(t[k]==v for k,v in recomputed.items())
    assert set(d['GPU_pack_all65536_BF16_storage_bits'])=={'dp','ds'}
    assert d['native_X_storage_bits_gate']['status']=='NO_XP_BY_DESIGN_RAW_TMA_LDSM_BITS_UNPROBED'
    assert len(d['actual_timed_graph_checks'])==20 and len(d['fresh_workspace_and_output'])==3
    assert d['graph_cleanup']=='RESET_ALL_FOUR' and d['weights_inputs_counters']=='UNCHANGED_OR_RESET'
    assert set(d['validation'])==set(LABELS)
    for label,checks in d['validation'].items():
        assert checks[0]['all4096_onehot_columns']=='EXACT' and checks[0]['random_FP64']=='PASS'
        assert {r['official_input_dist'] for r in checks if 'official_input_dist' in r}=={'normal','outliers'}
    for label,record in d['actual_JIT_SASS'].items():
        assert record['canonical_main_semantic_equality']=='PASS_ALL_PC_PREDICATE_INSTRUCTIONS_EXACT'
        artifact=HERE/Path(record['artifact']).name
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==record['sha256']
        repeated=json.loads(json.dumps(support.actual_binary_gate(label,artifact.read_text())))
        assert all(record[k]==v for k,v in repeated.items())
    xp=d['actual_JIT_SASS']['new']['X_layout_protocol_physical']
    assert xp['physical']['math_global_X_LDG']==0 and xp['physical']['static_LDSM']==32
    assert xp['physical']['static_X_TMA_steady_and_startup']==8 and len(xp['physical']['X_physical_blocks'])==8
    assert xp['protocol']['math_bar5_participants']==list(range(256))
    assert xp['protocol']['X_TMA_issues']==64 and xp['protocol']['X_phase_reuses']==62
    assert xp['layout']['all_coordinates_once'] and xp['layout']['stages']==64
    assert d['actual_JIT_SASS']['new']['producer_output_owner_proxy']=='EXACT_CANONICAL_MAIN_AND_FROZEN_PRODUCER_SOURCE_PROOF'
    assert d['attrs']['new']['regs']==122 and d['attrs']['new']['smem']==137232
    assert all(r['local']==0 and r['occupancy_after_optin']>=1 for r in d['attrs'].values())
    assert len(d['profile']['new'])==1 and len(d['profile']['dp'])==len(d['profile']['ds'])==2
    return {'status':'INDEPENDENT_RECOMPUTATION_PASS','attempt_status':d['status'],'raw_calls':len(calls),
            'actual_JIT_gates_reexecuted_from_saved_SASS':'PASS', 'current_pointer_descriptor_source_pin':'PASS', 'strict_terminal_oracles':8,'mutation_oracles':20,'F1_medians_us':med,
            'paired_policy_medians_us':{l:statistics.median(r[l] for r in paired) for l in LABELS},
            **recomputed,'raw_native_X_bits':'UNPROBED_EXPLICIT_LIMIT','official_score':'NOT_MEASURED',
            'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if __name__=='__main__':
    result=analyze(HERE/'native-x-01.json');(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
