"""Independent local CPU audit of authoritative PW38 full-stack artifacts."""
import collections,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.stack.fullscope.01/collection-1793076533579348342'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def analyze(collection=COLLECTION):
    inventory=json.loads((collection/'inventory.json').read_text());files=inventory['files']
    for rel,pin in files.items():assert sha(collection/rel)==pin,rel
    for rel,pin in inventory['dispatch']['source_pins'].items():assert sha(ROOT/rel)==pin,rel
    assert len(files)==48 and inventory['missing']==[] and inventory['terminal_exit']==0
    cleanup=inventory['cleanup_observation']
    assert cleanup['terminal_exit']==0 and not cleanup['cleanup_failed'] and cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[] and not cleanup['container_exists'] and cleanup['gpu_memory_mib']==0
    folder=collection/'logs/scheduler/pw38-stack-01';d=json.loads((folder/'result.json').read_text());source=ROOT/'proj_rsi/tools/pw38_stack_lab.py'
    assert sha(source)=='ddb48ef087790ceab114d651413999ca9a15b3c1bf5c24df99cfd2329ac9db15'
    spec=importlib.util.spec_from_file_location('pw38_frozen_review',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.verify()
    assert d['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED' and d['phase']=='complete' and not d['jit_errors']
    assert d['sources']==mod.PINS and d['review_sha256']==sha(ROOT/'proj_rsi/audits/pw38-stack-lab-independent.md') and not d['official_score_claim']
    assert d['graph_cleanup']==[]
    for label,prepare in d['prepare'].items():
        assert label in mod.LABELS and prepare['import_clock_elapsed']<300 and not prepare['budget_clock_reset'] and not prepare['serving_CPU_fallback'] and prepare['build_threads_joined']==2
    assert set(d['prepare'])==set(mod.LABELS)
    assert len(d['jit'])==15
    jit_counts=collections.Counter();rq_gates={}
    for rec in d['jit']:
        label,name=rec['label'],rec['name'];prefix=folder/(label+'.'+name)
        cu=Path(str(prefix)+'.cu');sass=Path(str(prefix)+'.sass');resources=Path(str(prefix)+'.resources')
        assert rec['source_sha256']==sha(cu) and rec['SASS_sha256']==sha(sass) and rec['before_return_to_prepare_math']
        assert len(rec['so_sha256'])==64
        if rec['RQ_gate'] is not None:
            expected=set(rec['RQ_gate']['flags'])=={2,3,65538,65539}
            repeated=mod.actual_rq_gate(sass.read_text(),resources.read_text(),expected)
            assert repeated==rec['RQ_gate']
            rq_gates[label,name]=rec['RQ_gate']['flags']
        jit_counts[label]+=1
    assert set(jit_counts.values())=={5} and set(jit_counts)==set(mod.LABELS)
    assert any(label=='pw38' and flags==[2,3,65538,65539] for (label,name),flags in rq_gates.items())
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
        for label,block in [('rs32','rs32_before'),('pw36','pw36_before'),('pw38','pw38'),('pw36','pw36_after'),('rs32','rs32_after')]:
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
        delta={label:statistics.median(r[label]-r['pw38'] for r in pairs) for label in mod.LABELS[:-1]}
        ratio={label:statistics.median(r[label]/r['pw38'] for r in pairs) for label in mod.LABELS[:-1]}
        assert delta==entry['paired_control_minus_pw38_us'] and ratio==entry['paired_control_over_pw38_ratio']
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
        if entry['picks']['pw38'].get('name','').endswith('EH'):eh.append(shape)
        medians[shape]={label:statistics.median(r[label] for r in pairs) for label in mod.LABELS}
        comparison[shape]={'paired_control_minus_pw38_us':delta,'paired_ratio':ratio,'F1_conservative_control_minus_pw38_us':{label:min(f1[label+'_before'],f1[label+'_after'])-f1['pw38'] for label in mod.LABELS[:-1]},'paired_win_tie_loss':{label:[sum(r[label]>r['pw38'] for r in pairs),sum(r[label]==r['pw38'] for r in pairs),sum(r[label]<r['pw38'] for r in pairs)] for label in mod.LABELS[:-1]},'picks':entry['picks']}
    assert terminal_count==180 and mutation_count==360 and eh==d['EH_selected']==['verify16/qkv_gate','verify16/pooler']
    loads={load:{label:sum(medians[load+'/'+layer][label] for layer in mod.LAYERS) for label in mod.LABELS} for load in mod.LOADS}
    aggregate={label:math.exp(sum(math.log(values[label]/values['pw38']) for values in loads.values())/6) for label in mod.LABELS[:-1]}
    f1loads={load:{label:sum((d['shapes'][load+'/'+layer]['F1_medians_us']['pw38'] if label=='pw38' else min(d['shapes'][load+'/'+layer]['F1_medians_us'][label+'_before'],d['shapes'][load+'/'+layer]['F1_medians_us'][label+'_after'])) for layer in mod.LAYERS) for label in mod.LABELS} for load in mod.LOADS}
    f1aggregate={label:math.exp(sum(math.log(values[label]/values['pw38']) for values in f1loads.values())/6) for label in mod.LABELS[:-1]}
    return {'status':'INDEPENDENT_FULL_SCOPE_VALIDITY_PASS','job_id':'pw38.stack.fullscope.01','inventory_sha256':sha(collection/'inventory.json'),'result_sha256':sha(folder/'result.json'),'inventory_files':len(files),'source_pins':len(inventory['dispatch']['source_pins']),'checks':222,'raw_calls':4350,'terminal_oracles':terminal_count,'mutations':mutation_count,'actual_JIT_records':len(d['jit']),'RQ_gates_reexecuted':{label+':'+name:flags for (label,name),flags in rq_gates.items()},'EH_selected':eh,'prepare':d['prepare'],'paired_policy_medians_us':medians,'comparison':comparison,'local_load_sums_of_shape_medians_us':loads,'local_geomean_control_over_pw38_load_ratio':aggregate,'local_F1_load_sums_conservative_shape_brackets_us':f1loads,'local_F1_geomean_control_over_pw38_load_ratio':f1aggregate,'aggregate_limit':'Local H100 full-policy screening using per-shape paired medians in separate windows; not official SGLang/H200 score, not isolated EH contribution.','cleanup':{k:v for k,v in cleanup.items() if k!='host_log_tail'},'official_score':'UNMEASURED','next_official_unscored_check_readiness':'GO_EXACT_SOURCE_WITH_ROOT_AUTHORIZATION'}
if __name__=='__main__':
    r=analyze();out=ROOT/'proj_rsi/audits/pw38-stack-independent.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','inventory_files','source_pins','checks','raw_calls','terminal_oracles','mutations','EH_selected','local_load_sums_of_shape_medians_us','local_geomean_control_over_pw38_load_ratio']},indent=2))
