"""Independent local CPU audit of authoritative PW40 full-stack artifacts."""
import collections,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/pw40.stack.fullscope.01/collection-1793081134233512859'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def analyze(collection=COLLECTION, mutate=None):
    inventory=json.loads((collection/'inventory.json').read_text());files=inventory['files']
    for rel,pin in files.items():assert sha(collection/rel)==pin,rel
    for rel,pin in inventory['dispatch']['source_pins'].items():assert sha(ROOT/rel)==pin,rel
    assert len(files)==63 and inventory['missing']==[] and inventory['terminal_exit']==0
    cleanup=inventory['cleanup_observation']
    assert cleanup['terminal_exit']==0 and not cleanup['cleanup_failed'] and cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[] and not cleanup['container_exists'] and cleanup['gpu_memory_mib']==0
    folder=collection/'logs/scheduler/pw40-stack-01';d=json.loads((folder/'result.json').read_text());source=ROOT/'proj_rsi/tools/pw40_stack_lab.py'
    if mutate:mutate(d)
    assert sha(source)=='9559a9ded1c4b3f60db4ef68decddba857cb13cd20dac048be1640a2dd53c1dc'
    spec=importlib.util.spec_from_file_location('pw40_frozen_review',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.verify()
    assert d['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED' and d['phase']=='complete' and not d['jit_errors']
    assert d['sources']==mod.PINS and d['review_sha256']==sha(ROOT/'proj_rsi/audits/pw40-stack-lab-independent.md') and not d['official_score_claim']
    assert d['graph_cleanup']==[]
    for label,prepare in d['prepare'].items():
        assert label in mod.LABELS and prepare['import_clock_elapsed']<300 and not prepare['budget_clock_reset'] and not prepare['serving_CPU_fallback'] and prepare['build_threads_joined']==2
    assert set(d['prepare'])==set(mod.LABELS)
    assert len(d['jit'])==15
    jit_counts=collections.Counter();rq_gates={};actual_sets=collections.defaultdict(set);wr_runtime=[]
    for rec in d['jit']:
        label,name=rec['label'],rec['name'];prefix=folder/(label+'.'+name)
        cu=Path(str(prefix)+'.cu');sass=Path(str(prefix)+'.sass');resources=Path(str(prefix)+'.resources')
        assert rec['source_sha256']==sha(cu) and rec['SASS_sha256']==sha(sass) and rec['before_return_to_prepare_math']
        production=Path(str(prefix)+'.production.cu')
        assert rec['production_source_sha256']==sha(production)
        appended=label=='pw40' and 'rq_go<128,6,1,0,0,0,1027>' in ''.join(production.read_text().split())
        assert rec['host_query_appended']==appended
        assert cu.read_text()==production.read_text()+(mod.WRINFO if appended else '')
        assert len(rec['so_sha256'])==64
        int(rec['so_sha256'],16)
        if rec['RQ_gate'] is not None:
            expected=set(rec['RQ_gate']['flags'])=={2,3,65538,65539}
            repeated=mod.actual_rq_gate(sass.read_text(),resources.read_text(),expected)
            repeated['full_RQ']=mod.full_rq_gate(production.read_text(),sass.read_text(),resources.read_text())
            actual_sets[label].update(tuple(a) for a in repeated['full_RQ']['args'])
            if appended:
                recorded=rec['RQ_gate']['Wring_runtime']
                repeated['Wring_runtime']=mod.wring_attributes(recorded['attrs'],d['PW40_device']['SM_count'])
                wr_runtime.append(repeated['Wring_runtime'])
            # JSON serializes tuples as arrays.
            assert json.loads(json.dumps(repeated))==rec['RQ_gate']
            rq_gates[label,name]=rec['RQ_gate']['flags']
        jit_counts[label]+=1
    assert set(jit_counts.values())=={5} and set(jit_counts)==set(mod.LABELS)
    assert any(label=='pw40' and flags==[2,3,65538,65539] for (label,name),flags in rq_gates.items())
    canonical=set(mod.rq_streams((ROOT/'proj_rsi/candidates/pw40/sass.txt').read_text()))
    for label in mod.LABELS:
        expected=set(canonical)
        if label!='pw40':expected.discard((128,6,1,0,0,0,1027))
        if label=='rs32':expected-={(128,5,1,0,0,0,65538),(128,5,1,0,0,0,65539)}
        assert actual_sets[label]==expected
    assert {k:len(v) for k,v in actual_sets.items()}==d['actual_RQ_functions']=={'rs32':17,'pw38':19,'pw40':20}
    assert len(wr_runtime)==1
    device=d['PW40_device'];assert device['before_candidate_import_and_prepare'] and device['source']==mod.PINS['pw40']
    assert device['compute_capability']==[9,0] and device['SM_count']>=126 and any(n in device['name'] for n in ['H100','H200'])
    assert device['device_index']==0 and device['UUID']
    attempts=d['Wring_prepare_calls'];assert attempts['rs32']==attempts['pw38']==[] and len(attempts['pw40'])==70
    assert all(a=={'name':'wring_x6w4hz','m':128,'n':2304,'KS':7,'code':0} for a in attempts['pw40'])
    assert len(d['checks'])==222
    for label in mod.LABELS:
        checks=[c for c in d['checks'] if c['label']==label]
        assert len(checks)==74
        assert [c['case'] for c in checks[:65]]==[{'m':m,'layer':layer,'dist':'normal'} for layer in mod.LAYERS for m in mod.MS]
        assert [c['case'] for c in checks[65:]]==mod.OFFICIAL
        assert all(c['official']==(i>=65) and c['counter_contract']=='PASS' and c['oracle'] for i,c in enumerate(checks))
    journal=[json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
    assert journal==d['raw'] and len(journal)==4350
    assert all(math.isfinite(c['us']) and c['us']>0 for c in journal)
    assert list(d['shapes'])==[load+'/'+layer for load in mod.LOADS for layer in mod.LAYERS]
    comparison={};medians={};terminal_count=mutation_count=0;eh=[]
    orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
    for shape,entry in d['shapes'].items():
        calls=[row for row in journal if row['shape']==shape];assert len(calls)==145
        assert set(entry['picks'])==set(mod.LABELS) and entry['graphs']=='RESET_AFTER_DRAIN'
        position=0;blocks={}
        for label,block in [('rs32','rs32_before'),('pw38','pw38_before'),('pw40','pw40'),('pw38','pw38_after'),('rs32','rs32_after')]:
            rows=calls[position:position+11];position+=11
            assert [(r['label'],r['seed'],r['mode']) for r in rows]==[(label,9000+i,'F1') for i in range(11)]
            blocks[block]=[r['us'] for r in rows]
        f1={key:statistics.median(values) for key,values in blocks.items()};assert f1==entry['F1_medians_us']
        pairs=[]
        for i in range(30):
            rows=calls[55+3*i:55+3*i+3]
            assert [(r['label'],r['seed'],r['mode']) for r in rows]==[(mod.LABELS[p],9020+i,'paired') for p in orders[i%6]]
            pairs.append({r['label']:r['us'] for r in rows})
        assert pairs==entry['paired_all_samples']
        delta={label:statistics.median(r[label]-r['pw40'] for r in pairs) for label in mod.LABELS[:-1]}
        ratio={label:statistics.median(r[label]/r['pw40'] for r in pairs) for label in mod.LABELS[:-1]}
        assert delta==entry['paired_control_minus_pw40_us'] and ratio==entry['paired_control_over_pw40_ratio']
        terminals=entry['terminal_oracles'];assert len(terminals)==6
        assert {(r['label'],r['phase'],r['seed']) for r in terminals}=={(label,phase,seed) for label in mod.LABELS for phase,seed in [('F1',9010),('paired',9049)]}
        assert all(r['oracle'] for r in terminals);terminal_count+=len(terminals)
        mutations=entry['mutations'];assert len(mutations)==12
        assert {(r['label'],r['kind']) for r in mutations}=={(label,kind) for label in mod.LABELS for kind in ['zero','outliers','all_probes','fresh_normal']}
        assert all(r['oracle'] and r['output_poison_overwritten'] and set(r['float_scratch_poisoned'])=={'/ws','/gvws','/rs_ws'} for r in mutations);mutation_count+=len(mutations)
        for label,pick in entry['picks'].items():
            assert not(pick['family']=='router' and pick['id'] in(8,11))
            if pick['family']=='RQ':
                regs,local,shared=pick['actual_regs_local_shared'];assert 0<regs<=168 and local==0
                v=pick['variant'];nt,st,fl=v[1],v[2],v[7]
                assert shared==1024+st*(nt*128+(6144 if fl&16 else 12288))+24*st+16
        if entry['picks']['pw40'].get('name','').endswith('EH'):eh.append(shape)
        medians[shape]={label:statistics.median(r[label] for r in pairs) for label in mod.LABELS}
        comparison[shape]={'paired_control_minus_pw40_us':delta,'paired_ratio':ratio,'F1_conservative_control_minus_pw40_us':{label:min(f1[label+'_before'],f1[label+'_after'])-f1['pw40'] for label in mod.LABELS[:-1]},'paired_win_tie_loss':{label:[sum(r[label]>r['pw40'] for r in pairs),sum(r[label]==r['pw40'] for r in pairs),sum(r[label]<r['pw40'] for r in pairs)] for label in mod.LABELS[:-1]},'picks':entry['picks']}
    assert terminal_count==180 and mutation_count==360 and eh==d['EH_selected']==['verify16/qkv_gate','verify16/pooler','verify16/unpooler']
    assert d['Wring_selected']==[shape for shape,entry in d['shapes'].items() if entry['picks']['pw40'].get('name')=='wring_x6w4hz']==[]
    loads={load:{label:sum(medians[load+'/'+layer][label] for layer in mod.LAYERS) for label in mod.LABELS} for load in mod.LOADS}
    aggregate={label:math.exp(sum(math.log(values[label]/values['pw40']) for values in loads.values())/6) for label in mod.LABELS[:-1]}
    f1loads={load:{label:sum((d['shapes'][load+'/'+layer]['F1_medians_us']['pw40'] if label=='pw40' else min(d['shapes'][load+'/'+layer]['F1_medians_us'][label+'_before'],d['shapes'][load+'/'+layer]['F1_medians_us'][label+'_after'])) for layer in mod.LAYERS) for label in mod.LABELS} for load in mod.LOADS}
    f1aggregate={label:math.exp(sum(math.log(values[label]/values['pw40']) for values in f1loads.values())/6) for label in mod.LABELS[:-1]}
    return {'status':'INDEPENDENT_FULL_SCOPE_VALIDITY_PASS','job_id':'pw40.stack.fullscope.01','inventory_sha256':sha(collection/'inventory.json'),'result_sha256':sha(folder/'result.json'),'inventory_files':len(files),'source_pins':len(inventory['dispatch']['source_pins']),'checks':222,'raw_calls':4350,'terminal_oracles':terminal_count,'mutations':mutation_count,'actual_JIT_records':len(d['jit']),'RQ_gates_reexecuted':{label+':'+name:flags for (label,name),flags in rq_gates.items()},'EH_selected':eh,'Wring_selected':d['Wring_selected'],'Wring_prepare_attempts':70,'actual_RQ_functions':d['actual_RQ_functions'],'Wring_runtime':wr_runtime,'PW40_device':device,'prepare':d['prepare'],'paired_policy_medians_us':medians,'comparison':comparison,'local_load_sums_of_shape_medians_us':loads,'local_geomean_control_over_pw40_load_ratio':aggregate,'local_F1_load_sums_conservative_shape_brackets_us':f1loads,'local_F1_geomean_control_over_pw40_load_ratio':f1aggregate,'aggregate_limit':'Local H100 full-policy screening using per-shape paired medians in separate windows; not official SGLang/H200 score, not isolated Wring contribution. Wring unselected; no isolated Wring gain claim.','cleanup':{k:v for k,v in cleanup.items() if k!='host_log_tail'},'official_score':'UNMEASURED','next_official_unscored_check_readiness':'GO_EXACT_SOURCE_3bc0df52d96f62479c3d5c19ab2717c7a9f4d18c1dd087fb786e2f9f080f0278'}
if __name__=='__main__':
    r=analyze();rejected=[]
    def perturb(kind):
        def change(d):
            if kind=='terminal_seed':d['shapes']['decode32/qkv_gate']['terminal_oracles'][0]['seed']+=1
            elif kind=='declared_gain':d['shapes']['decode32/qkv_gate']['paired_control_minus_pw40_us']['pw38']+=1
            elif kind=='Wring_status':d['Wring_prepare_calls']['pw40'][0]['code']=-6
            elif kind=='device_SM':d['PW40_device']['SM_count']=125
            elif kind=='function_set':d['actual_RQ_functions']['pw40']=19
            elif kind=='false_Wring_selection':d['Wring_selected']=['verify16/qkv_gate']
            elif kind=='Wring_occupancy':
                next(j for j in d['jit'] if (j.get('RQ_gate') or {}).get('Wring_runtime'))['RQ_gate']['Wring_runtime']['attrs']['occupancy']=0
            else:d['jit'][0]['production_source_sha256']='0'*64
        return change
    for kind in ['terminal_seed','declared_gain','Wring_status','device_SM','function_set','false_Wring_selection','Wring_occupancy','production_source']:
        try:analyze(mutate=perturb(kind))
        except AssertionError:rejected.append(kind)
        else:raise AssertionError('negative accepted '+kind)
    r['independent_negative_controls_rejected']=rejected
    r['evidence_limits']=['Fifteen extension-build records contain17/19/20 actual RQ functions acrossRS32/PW38/PW40, respectively;20 refers to PW40 RQ functions, not20 extension builds.', 'Collected SASS/resources/CUDA dumps are inventory-hash validated and CPU gates rerun; recordedSO hashes are structurally checked. Shared objects themselves are not in this collection and were not independently redumped.', 'GPU tensor outputs are not recollected/recomputed by this CPU review; strict terminal oracle evidence and exact reviewed hooks/order/source closure are checked.', 'Seventy Wring preparation calls are recorded, all rc0; the initial80-call summary was incorrect.', 'Wring unselected: timing is prepared full-policy behavior and does not isolate Wring contribution or selection robustness.']
    out=ROOT/'proj_rsi/audits/pw40-stack-independent.json';out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ['status','inventory_files','source_pins','checks','raw_calls','terminal_oracles','mutations','actual_RQ_functions','Wring_selected','local_load_sums_of_shape_medians_us','local_geomean_control_over_pw40_load_ratio','local_F1_geomean_control_over_pw40_load_ratio']},indent=2))
