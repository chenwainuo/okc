"""Author CPU reanalysis of frozen collected PW38 all-scope trial. No GPU imports."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.stack.fullscope.01/collection-1793076533579348342'
DATA=COLLECTION/'logs/scheduler/pw38-stack-01'
DRIVER=ROOT/'proj_rsi/tools/pw38_stack_lab.py'
DRIVER_PIN='ddb48ef087790ceab114d651413999ca9a15b3c1bf5c24df99cfd2329ac9db15'
LABELS=('rs32','pw36','pw38')
CONTROLS=LABELS[:-1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def interval(values):
    quart=statistics.quantiles(values,n=4,method='inclusive')
    return [quart[0],quart[2]]
def pick_name(p):
    return p.get('name') or ('%s%d'%(p['family'],p['id']) if 'id' in p else p['family']+':'+str(p.get('lt')))
def boot_median(values,seed):
    rng=random.Random(seed);medians=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
    return [medians[49],medians[1949]]
def main():
    assert sha(DRIVER)==DRIVER_PIN
    spec=importlib.util.spec_from_file_location('pw38_frozen_analysis_driver',DRIVER)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.cpu_check() # Only stdlib/AST, including source/physical/resource negatives.
    inventory=json.loads((COLLECTION/'inventory.json').read_text())
    assert inventory['job_id']=='pw38.stack.fullscope.01' and inventory['terminal_exit']==0 and inventory['missing']==[]
    for path,pin in inventory['files'].items():assert sha(COLLECTION/path)==pin,path
    for path,pin in inventory['dispatch']['source_pins'].items():assert sha(ROOT/path)==pin,path
    cleanup=inventory['cleanup_observation']
    assert cleanup['cleanup_failed'] is False and cleanup['container_exists'] is False
    assert cleanup['guard_pids']==cleanup['compute_pids']==cleanup['containers']==[] and cleanup['gpu_memory_mib']==0
    host=COLLECTION/'logs/scheduler/pw38.stack.fullscope.01.host.log';log=host.read_text()
    assert 'TRIAL_EXIT okc-proj-pw38-stack-01 0' in log and '\ncomplete\n' in log
    assert not any(token in log for token in ('skipped-budget','INVALID','ERROR','Traceback'))
    j=json.loads((DATA/'result.json').read_text())
    assert j['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED' and not j['jit_errors']
    assert j['sources']==driver.PINS and j['official_score_claim'] is False
    assert j['review_sha256']==sha(ROOT/'proj_rsi/audits/pw38-stack-lab-independent.md')
    assert j['graph_cleanup']==[] and all(e['graphs']=='RESET_AFTER_DRAIN' for e in j['shapes'].values())
    for label,prep in j['prepare'].items():
        assert label in LABELS and prep['budget_clock_reset'] is False and prep['serving_CPU_fallback'] is False
        assert prep['build_threads_joined']==2 and prep['import_clock_elapsed']<300
    assert len(j['jit'])==15
    jit=[]
    for record in j['jit']:
        prefix=record['label']+'.'+record['name'];sass=DATA/(prefix+'.sass');resource=DATA/(prefix+'.resources');cu=DATA/(prefix+'.cu')
        assert record['before_return_to_prepare_math'] is True
        assert sha(sass)==record['SASS_sha256'] and sha(cu)==record['source_sha256']
        if record.get('RQ_gate') is not None:
            actual=driver.actual_rq_gate(sass.read_text(),resource.read_text(),record['RQ_gate']['flags']==[2,3,65538,65539])
            assert actual==record['RQ_gate']
        jit.append({'label':record['label'],'name':record['name'],'SASS_sha256':sha(sass),'resource_sha256':sha(resource),'generated_source_sha256':sha(cu),'RQ_gate':record.get('RQ_gate')})
    assert sum(record['label']=='pw38' and record.get('RQ_gate',{}).get('flags')==[2,3,65538,65539] for record in j['jit'] if record.get('RQ_gate'))==1
    common_rq={}
    def all_streams(sass):
        return {table.splitlines()[0].strip():[(int(m[1],16),m[2].strip()) for m in __import__('re').finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)] for table in sass.split('Function : ')[1:]}
    for kind in ('rq32','rq128'):
        tables={label:all_streams(next(DATA.glob(label+'.'+kind+'*.sass')).read_text()) for label in LABELS}
        shared=tables['rs32'].keys()&tables['pw36'].keys()&tables['pw38'].keys()
        assert len(shared)==(11 if kind=='rq32' else 6)
        assert all(tables['rs32'][name]==tables['pw36'][name]==tables['pw38'][name] for name in shared)
        common_rq[kind]={'common_function_count':len(shared),'all_three_actual_PC_predicate_instruction_streams_equal':True,'symbols':sorted(shared)}
    expected_shapes=[load+'/' +layer for load in driver.LOADS for layer in driver.LAYERS]
    assert list(j['shapes'])==expected_shapes and len(j['checks'])==222
    for label in LABELS:
        checks=[c for c in j['checks'] if c['label']==label]
        assert len(checks)==74
        expected=[{'m':m,'layer':layer,'dist':'normal'} for layer in driver.LAYERS for m in driver.MS]+driver.OFFICIAL
        assert [c['case'] for c in checks]==expected
        assert [c['official'] for c in checks]==[False]*65+[True]*9
        assert all(c['counter_contract']=='PASS' and 'relative L2 error' in c['oracle'] for c in checks)
    raw=j['raw'];journal=[json.loads(line) for line in (DATA/'raw.jsonl').read_text().splitlines()]
    assert raw==journal and len(raw)==4350 and all(0<r['us']<float('inf') for r in raw)
    cursor=0;rows=[];load_sums={};same_picks=[];terminals=mutations=0
    orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
    for shape in expected_shapes:
        e=j['shapes'][shape];block=raw[cursor:cursor+145];cursor+=145
        expected=[]
        for label in CONTROLS:expected += [(label,9000+i,'F1') for i in range(11)]
        expected += [('pw38',9000+i,'F1') for i in range(11)]
        for label in reversed(CONTROLS):expected += [(label,9000+i,'F1') for i in range(11)]
        for i in range(30):expected += [(LABELS[index],9020+i,'paired') for index in orders[i%6]]
        assert [(r['label'],r['seed'],r['mode']) for r in block]==expected and all(r['shape']==shape for r in block)
        f1={}
        for k,(label,suffix) in enumerate((('rs32','before'),('pw36','before'),('pw38',''),('pw36','after'),('rs32','after'))):
            key=label+('_'+suffix if suffix else '');f1[key]=statistics.median(r['us'] for r in block[k*11:(k+1)*11])
        for key,value in f1.items():equal(value,e['F1_medians_us'][key])
        paired=[{r['label']:r['us'] for r in block[55+3*i:58+3*i]} for i in range(30)]
        assert paired==e['paired_all_samples']
        assert len(e['terminal_oracles'])==6
        assert {(r['label'],r['phase'],r['seed']) for r in e['terminal_oracles']}=={(label,phase,seed) for label in LABELS for phase,seed in [('F1',9010),('paired',9049)]}
        assert all('relative L2 error' in r['oracle'] for r in e['terminal_oracles']);terminals+=6
        assert len(e['mutations'])==12
        assert {(r['label'],r['kind']) for r in e['mutations']}=={(label,kind) for label in LABELS for kind in ('zero','outliers','all_probes','fresh_normal')}
        assert all(r['output_poison_overwritten'] and set(r['float_scratch_poisoned'])=={'/ws','/gvws','/rs_ws'} for r in e['mutations']);mutations+=12
        median={label:statistics.median(p[label] for p in paired) for label in LABELS}
        comparisons={}
        for control in CONTROLS:
            deltas=[p[control]-p['pw38'] for p in paired];ratios=[p[control]/p['pw38'] for p in paired]
            d=statistics.median(deltas);ratio=statistics.median(ratios)
            equal(d,e['paired_control_minus_pw38_us'][control]);equal(ratio,e['paired_control_over_pw38_ratio'][control])
            comparisons[control]={'paired_control_minus_pw38_us':d,'paired_median_control_over_pw38':ratio,
                'F1_control_minus_pw38_us':min(f1[control+'_before'],f1[control+'_after'])-f1['pw38'],
                'paired_delta_IQR_us':interval(deltas),'paired_median_bootstrap95_us':boot_median(deltas,len(rows)*2+CONTROLS.index(control)),
                'PW38_wins_ties_losses':[sum(v>0 for v in deltas),sum(v==0 for v in deltas),sum(v<0 for v in deltas)],'all_paired_deltas_us':deltas,
                'same_selected_plan_as_pw38':e['picks'][control]==e['picks']['pw38']}
        row={'shape':shape,'m':driver.LOADS[shape.split('/')[0]],'paired_marginal_medians_us':median,'F1_medians_us':f1,'comparisons':comparisons,'actual_picks':e['picks'],'picked_names':{label:pick_name(e['picks'][label]) for label in LABELS}}
        rows.append(row)
        load=shape.split('/')[0];summed=load_sums.setdefault(load,{'paired_marginal_median_sums_us':dict.fromkeys(LABELS,0.0),'F1_conservative_sum_us':dict.fromkeys(LABELS,0.0)})
        for label in LABELS:summed['paired_marginal_median_sums_us'][label]+=median[label]
        for control in CONTROLS:summed['F1_conservative_sum_us'][control]+=min(f1[control+'_before'],f1[control+'_after'])
        summed['F1_conservative_sum_us']['pw38']+=f1['pw38']
    assert cursor==4350 and terminals==180 and mutations==360
    eh=[r['shape'] for r in rows if r['picked_names']['pw38'].endswith('EH')]
    assert eh==j['EH_selected']==['verify16/qkv_gate','verify16/pooler']
    aggregate={}
    for control in CONTROLS:
        aggregate[control]={metric:math.exp(sum(math.log(s[metric][control]/s[metric]['pw38']) for s in load_sums.values())/6) for metric in ('paired_marginal_median_sums_us','F1_conservative_sum_us')}
        aggregate[control]['paired_shape_win_tie_loss']=[sum(r['comparisons'][control]['paired_control_minus_pw38_us']>0 for r in rows),sum(r['comparisons'][control]['paired_control_minus_pw38_us']==0 for r in rows),sum(r['comparisons'][control]['paired_control_minus_pw38_us']<0 for r in rows)]
    analysis={'state':'AUTHOR_CPU_REANALYSIS_PASS_VALID_LOCAL_EXECUTION_MIXED_SHAPES_MODEST_AGGREGATE_POINT_GAIN','source_sha256':j['sources'],'driver_sha256':DRIVER_PIN,
        'inventory_sha256':sha(COLLECTION/'inventory.json'),'result_sha256':sha(DATA/'result.json'),'raw_sha256':sha(DATA/'raw.jsonl'),'host_log_sha256':sha(host),
        'scope_verified':{'raw_calls':4350,'shapes':30,'checks':222,'terminal_oracles':180,'graph_mutations':360,'JIT_archives':15,'all_per_shape_graphs_reset':True,'final_graph_list_empty':True},
        'cleanup':{k:cleanup[k] for k in ('cleanup_failed','compute_pids','container_exists','containers','gpu_memory_mib','guard_pids')},
        'common_non_EH_actual_RQ_identity':common_rq,'source_pin_count':len(inventory['dispatch']['source_pins']),'artifact_pin_count':len(inventory['files']),'job_id':inventory['job_id'],'guard_pid':inventory['dispatch']['guard_pid'],'terminal_exit':inventory['terminal_exit'],'prepare':j['prepare'],'JIT':jit,'EH_selected':eh,
        'all_shapes':rows,'local_load_sums':load_sums,'local_descriptive_geomean_control_over_pw38':aggregate,
        'limitations':['Local H100 graph times include replay overhead; no official H200 score or transfer forecast.','Six-load sums use marginal per-shape medians from one full-policy run; ratios are descriptive, not an official calibrated score or a formal aggregate confidence interval.','Per-shape bootstrap intervals resample paired differences within one batch; they do not establish cross-run or H200 reproducibility.','Separate fresh prepares choose different plans even on source-identical paths. Credit isolated EH only when geometry/control comparison supports it; retain all selection and timing differences.','Counter/immutable/terminal runtime PASS is asserted by the frozen reviewed driver and supported by complete completion records, not reconstructed tensor bytes after process exit.']}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps({'state':analysis['state'],'aggregate':aggregate,'EH_selected':eh},indent=2))
if __name__=='__main__':main()
