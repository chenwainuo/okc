"""CPU-only independent recomputation of native-X STMatrix scheduler attempt01 retained evidence."""
import hashlib,json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LABELS=('incumbent','ds','native','new')
def analyze(path):
    sys.path.insert(0,str(HERE.parents[1]/'candidates/reverse_tma_x_stmatrix'))
    sys.path.insert(0,str(HERE.parents[1]/'candidates/reverse_tma_x_stmatrix_trial'))
    import native_x_stmatrix_support as support
    support.verify_frozen();support.verify_lab()
    d=json.loads(path.read_text());t=d['timing'];calls=t['all_timed_and_warmup_calls']
    assert len(calls)==295 and sum(c['warmup'] for c in calls)==20
    assert all(c['us']>0 for c in calls)
    assert all(c['mode']=='warmup' and c['input_seed']==239999 for c in calls[:20])
    assert [c['label'] for c in calls[:20]]==[label for label in LABELS for _ in range(5)]
    raw={};paired=[];last={};strict=t['strict_immediate_actual_timed_output_checks']
    f1calls=[c for c in calls if c['mode']=='F1'];assert len(f1calls)==147
    sequence=['incumbent','ds','native','new','native','ds','incumbent']
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
            assert row['XP_current_recorded_fixture_exact']==(row['graph']=='ds')
        assert gate['current_X_matches_last_actual_call_seed']==(240146 if phase=='F1' else 250031)
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
    assert set(d['GPU_pack_all65536_BF16_storage_bits'])=={'ds'}
    assert d['native_variants_storage_bits_gate']['status']=='NO_XP_BY_DESIGN_RAW_TMA_LDSM_BITS_UNPROBED'
    assert len(d['actual_timed_graph_checks'])==20 and len(d['fresh_workspace_and_output'])==3
    mutations=d['actual_timed_graph_checks']
    expected={(label,name) for label in LABELS for name in ('original','zero','fresh_normal','fresh_outliers','all_probes')}
    assert {(r['graph'],r['input']) for r in mutations}==expected
    assert all(r['NaN_poison_overwritten'] and r['input_immutable'] for r in mutations)
    assert all(r['packed_poison_overwritten_byte_exact']==(r['graph']=='ds') for r in mutations)
    for label,record in d['fresh_workspace_and_output'].items():
        assert record['status']=='PASS' and record['workspace_bytes']==(1048576 if label=='ds' else 0)
        if label!='ds':assert record['XP']=='NONE'
    assert d['graph_cleanup']=='RESET_ALL_FOUR' and d['weights_inputs_counters']=='UNCHANGED_OR_RESET'
    assert set(d['validation'])==set(LABELS)
    for label,checks in d['validation'].items():
        assert checks[0]['all4096_onehot_columns']=='EXACT' and checks[0]['random_FP64']=='PASS'
        assert {r['official_input_dist'] for r in checks if 'official_input_dist' in r}=={'normal','outliers'}
    for label,record in d['actual_JIT_SASS'].items():
        assert record['canonical_main_semantic_equality']=='PASS_ALL_PC_PREDICATE_INSTRUCTIONS_EXACT'
        artifact=path.parent/Path(record['artifact']).name
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==record['sha256']
        repeated=json.loads(json.dumps(support.actual_binary_gate(label,artifact.read_text())))
        assert all(record[k]==v for k,v in repeated.items())
    parent=d['actual_JIT_SASS']['native']['X_layout_protocol_physical']
    assert parent['physical']['math_global_X_LDG']==0 and parent['physical']['static_LDSM']==32
    assert parent['physical']['static_X_TMA_steady_and_startup']==8 and len(parent['physical']['X_physical_blocks'])==8
    assert parent['protocol']['math_bar5_participants']==list(range(256))
    assert parent['protocol']['X_TMA_issues']==64 and parent['protocol']['X_phase_reuses']==62
    new=d['actual_JIT_SASS']['new']['X_layout_protocol_physical']
    assert new['physical']['math_global_X_LDG']==0 and new['physical']['static_LDSM']==32
    assert new['physical']['static_X_TMA_steady_and_startup']==8 and len(new['physical']['X_physical_blocks'])==8
    assert new['protocol']['X_TMA_issues']==64 and new['protocol']['X_phase_reuses']==62
    assert new['candidate_actual_binary_gate']['status']=='PASS_ACTUAL_JIT_MAIN_AND_PHYSICAL_GATES'
    for label,detail in [('native',parent),('new',new)]:
        assert detail['layout']['all_coordinates_once'] and detail['layout']['stages']==64
        assert d['actual_JIT_SASS'][label]['producer_output_owner_proxy']=='EXACT_CANONICAL_MAIN_AND_FROZEN_SOURCE_PROOF'
    assert d['attrs']['new']['regs']==122 and d['attrs']['new']['smem']==137232
    assert d['attrs']['native']['regs']==122 and d['attrs']['native']['smem']==137232
    assert d['attrs']['ds']['smem']==103600
    assert all(r['local']==0 and r['occupancy_after_optin']>=1 for r in d['attrs'].values())
    assert len(d['profile']['new'])==len(d['profile']['native'])==1 and len(d['profile']['ds'])==2
    for label,token in [('ds','rq_px_dp_stmx<'),('native','rq_tma_x<'),('new','rq_tma_x_stmx<')]:
        assert sum(token in r['name'] for r in d['profile'][label])==1
        assert sum('rx_pack' in r['name'] for r in d['profile'][label])==(1 if label=='ds' else 0)
    expected_source={'ds':'672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1',
                     'native':'ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377',
                     'new':'f6a765f0d0cf3174435f4981958015c242a53b5359f00064aa388b958cb9d9b6'}
    assert d['candidate_source_sha256']==expected_source
    assert d['parent_sha256']=='2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
    assert d['incumbent_pick']['kind']=='rq'
    assert d['full_pipeline_in_timed_graph']=={'ds':'FRESH_PACK_PLUS_MAIN_2_NODES','native':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP','new':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP'}
    return {'status':'INDEPENDENT_RECOMPUTATION_PASS','attempt_status':d['status'],'raw_calls':len(calls),
            'actual_JIT_gates_reexecuted_from_saved_SASS':'PASS', 'current_pointer_descriptor_source_pin':'PASS', 'strict_terminal_oracles':8,'mutation_oracles':20,'F1_medians_us':med,
            'paired_policy_medians_us':{l:statistics.median(r[l] for r in paired) for l in LABELS},
            **recomputed,'raw_native_X_bits':'BOTH_NATIVE_VARIANTS_UNPROBED_EXPLICIT_LIMIT','official_score':'NOT_MEASURED',
            'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if __name__=='__main__':
    source=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'result.json'
    result=analyze(source);(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
