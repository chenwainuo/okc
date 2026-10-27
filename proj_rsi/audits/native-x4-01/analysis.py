"""CPU-only independent recomputation of native-X4 scheduler attempt01 retained evidence."""
import copy,hashlib,json,math,re,runpy,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LABELS=('incumbent','ds','native','new')
def analyze(path):
    sys.path.insert(0,str(HERE.parents[1]/'candidates/reverse_tma_x4_trial'))
    import native_x4_support as support
    support.verify_frozen();support.verify_lab()
    d=json.loads(path.read_text());return analyze_data(d,path,support)

def analyze_data(d,path,support):
    t=d['timing'];calls=t['all_timed_and_warmup_calls']
    assert len(calls)==295 and sum(c['warmup'] for c in calls)==20
    assert all(isinstance(c['us'],(int,float)) and not isinstance(c['us'],bool) and math.isfinite(c['us']) and c['us']>0 for c in calls)
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
        assert {r['graph'] for r in gate['checks']}==set(LABELS)
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
    assert t['terminal_last_timed_input_seed_per_policy']==dict.fromkeys(LABELS,250031)
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
    assert new['protocol']['X_TMA_issues']==64 and new['protocol']['X_phase_reuses']==60
    assert new['source_restore']['status']=='PASS_FINITE_EXACT_RESTORE'
    for label,detail in [('native',parent),('new',new)]:
        assert detail['layout']['all_coordinates_once'] and detail['layout']['stages']==64
        assert d['actual_JIT_SASS'][label]['producer_output_owner_proxy']=='EXACT_CANONICAL_MAIN_AND_FROZEN_SOURCE_PROOF'
    assert d['attrs']['new']['regs']==122 and d['attrs']['new']['smem']==170016
    assert d['attrs']['native']['regs']==122 and d['attrs']['native']['smem']==137232
    assert d['attrs']['ds']['smem']==103600
    assert all(r['local']==0 and r['occupancy_after_optin']==1 and r['regs']==122 and
               r['threads']==r['max_threads']==512 and r['rounded_register_pool']==65536 and
               r['max_dynamic_after']>=r['smem'] for r in d['attrs'].values())
    pack=d['packer_attrs']['ds'];assert 0<pack['regs']<=64 and pack['local']==0 and pack['max_threads']>=256
    assert d['GPU_pack_all65536_BF16_storage_bits']['ds']=='PASS_1MiB_EXACT_INPUT_IMMUTABLE_STORAGE_ONLY'
    def numeric_oracle(text):
        if text=='exact zero complete output':return
        match=re.fullmatch(r'relative L2 error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\); (\d+) probe rows exact to ([0-9.e+-]+)',text)
        assert match, text
        assert float(match[1])<=.004 and float(match[2])<=.004 and int(match[3])>=0 and float(match[4])==0,text
    for records in d['validation'].values():
        for record in records:
            if 'official_input_dist' in record:numeric_oracle(record['result'])
            if 'fixed_pointer_replay' in record:assert record['status']=='PASS'
    for phase in strict.values():
        for record in phase['checks']:numeric_oracle(record['oracle'])
    for record in mutations:numeric_oracle(record['oracle'])
    for record in d['fresh_workspace_and_output'].values():numeric_oracle(record['oracle'])
    assert len(d['profile']['new'])==len(d['profile']['native'])==1 and len(d['profile']['ds'])==2
    for label,token in [('ds','rq_px_dp_stmx<'),('native','rq_tma_x<'),('new','rq_tma_x4<')]:
        assert sum(token in r['name'] for r in d['profile'][label])==1
        assert sum('rx_pack' in r['name'] for r in d['profile'][label])==(1 if label=='ds' else 0)
    expected_source={'ds':'672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1',
                     'native':'ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377',
                     'new':'e8447cbf0c113618462f5cbddc61d7f1e438ada2efa43130531f96c9fde768d3'}
    assert d['candidate_source_sha256']==expected_source
    assert d['parent_sha256']=='2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
    assert d['incumbent_pick']['kind']=='rq'
    assert d['full_pipeline_in_timed_graph']=={'ds':'FRESH_PACK_PLUS_MAIN_2_NODES','native':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP','new':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP'}
    return {'status':'INDEPENDENT_RECOMPUTATION_PASS','attempt_status':d['status'],'raw_calls':len(calls),
            'actual_JIT_gates_reexecuted_from_saved_SASS':'PASS', 'current_pointer_descriptor_source_pin':'PASS', 'strict_terminal_oracles':8,'mutation_oracles':20,'F1_medians_us':med,
            'paired_policy_medians_us':{l:statistics.median(r[l] for r in paired) for l in LABELS},
            **recomputed,'raw_native_X_bits':'BOTH_NATIVE_VARIANTS_UNPROBED_EXPLICIT_LIMIT','official_score':'NOT_MEASURED',
            'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def evidence(path):
    collection=path.parents[3];root=HERE.parents[2]
    inventory_path=collection/'inventory.json';inventory=json.loads(inventory_path.read_text())
    assert inventory['job_id']=='native.x4.v1.01' and inventory['missing']==[] and inventory['terminal_exit']==0
    for name,expected in inventory['files'].items():
        assert hashlib.sha256((collection/name).read_bytes()).hexdigest()==expected,('inventory hash',name)
    job=json.loads((root/'proj_rsi/scheduling/jobs/native-x4-01.json').read_text())
    pins=inventory['dispatch']['source_pins'];assert pins=={**job['files'],job['review']['path']:job['review']['sha256']} and len(pins)==62
    for name,expected in pins.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,('source hash',name)
    cleanup=inventory['cleanup_observation']
    assert cleanup['terminal_exit']==0 and cleanup['cleanup_failed'] is False
    assert cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[]
    assert cleanup['container_exists'] is False and cleanup['gpu_memory_mib']==0 and cleanup['host_log_exists'] is True
    terminal=runpy.run_path(str(root/'proj_rsi/candidates/reverse_tma_x4_trial/timed_output_gate_proof.py'))['proof']()
    assert len(terminal['negative_controls'])==9 and terminal['gpu_calls']==0
    result=analyze(path);d=json.loads(path.read_text())
    support=sys.modules['native_x4_support'];negative={}
    for label,mutate in {
        'raw_seed':lambda v:v['timing']['all_timed_and_warmup_calls'][20].update(input_seed=1),
        'raw_timing':lambda v:v['timing']['all_timed_and_warmup_calls'][20].update(us=float('nan')),
        'paired_gain':lambda v:v['timing']['paired_new_median_gain_us'].update(native=1.),
        'strict_terminal_fixture':lambda v:v['timing']['strict_immediate_actual_timed_output_checks']['F1']['checks'][0].update(input_seed=1),
        'mutation_poison':lambda v:v['actual_timed_graph_checks'][0].update(NaN_poison_overwritten=False),
        'resource_smem':lambda v:v['attrs']['new'].update(smem=137232),
    }.items():
        altered=copy.deepcopy(d);mutate(altered)
        try:analyze_data(altered,path,support)
        except AssertionError:negative[label]='REJECTED'
        else:raise AssertionError(('negative admitted',label))
    sass=(path.parent/'result.new.jit.sass').read_text()
    assert 'LDSM.' in sass
    try:support.actual_binary_gate('new',sass.replace('LDSM.','LDS.',1))
    except AssertionError:negative['actual_JIT_instruction']='REJECTED'
    else:raise AssertionError('mutated SASS admitted')
    paired=d['timing']['paired_graph_us'];distributions={}
    for label in LABELS[:3]:
        values=[r[label]-r['new'] for r in paired];mid=statistics.median(values)
        distributions[label]={'median_control_minus_new_us':mid,'min_us':min(values),'max_us':max(values),
            'MAD_us':statistics.median(abs(v-mid) for v in values),
            'win_tie_loss':[sum(v>0 for v in values),sum(v==0 for v in values),sum(v<0 for v in values)]}
    result.update(inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),inventory_files=len(inventory['files']),
        source_pins=len(pins),paired_distributions=distributions,analysis_negative_controls=negative,
        terminal_gate_CPU_negative_controls=terminal['negative_controls'],cleanup_observation={k:v for k,v in cleanup.items() if k!='host_log_tail'},terminal_exit=0)
    return result

if __name__=='__main__':
    source=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'result.json'
    result=evidence(source);(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
