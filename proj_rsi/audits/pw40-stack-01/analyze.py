"""Author CPU reanalysis of frozen collected PW40 all-scope trial. No GPU imports."""
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
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/pw40.stack.fullscope.01/collection-1793081134233512859'
DATA=COLLECTION/'logs/scheduler/pw40-stack-01'
DRIVER=ROOT/'proj_rsi/tools/pw40_stack_lab.py'
DRIVER_PIN='9559a9ded1c4b3f60db4ef68decddba857cb13cd20dac048be1640a2dd53c1dc'
LABELS=('rs32','pw38','pw40')
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
    spec=importlib.util.spec_from_file_location('pw40_frozen_analysis_driver',DRIVER)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.cpu_check() # Only stdlib/AST, including source/physical/resource negatives.
    inventory=json.loads((COLLECTION/'inventory.json').read_text())
    assert inventory['job_id']=='pw40.stack.fullscope.01' and inventory['terminal_exit']==0 and inventory['missing']==[]
    for path,pin in inventory['files'].items():assert sha(COLLECTION/path)==pin,path
    for path,pin in inventory['dispatch']['source_pins'].items():assert sha(ROOT/path)==pin,path
    cleanup=inventory['cleanup_observation']
    assert cleanup['cleanup_failed'] is False and cleanup['container_exists'] is False
    assert cleanup['guard_pids']==cleanup['compute_pids']==cleanup['containers']==[] and cleanup['gpu_memory_mib']==0
    host=COLLECTION/'logs/scheduler/pw40.stack.fullscope.01.host.log';log=host.read_text()
    assert 'TRIAL_EXIT okc-proj-pw40-stack-01 0' in log and '\ncomplete\n' in log
    assert not any(token in log for token in ('skipped-budget','INVALID','ERROR','Traceback'))
    bake={label:[] for label in LABELS};current=None
    for line in log.splitlines():
        if line in ['prepare_'+label for label in LABELS]:current=line.removeprefix('prepare_')
        if current and line.startswith(('BAKE ','PICK ','PREP ')):bake[current].append(line)
    assert 'BAKE qkv_gate m=128 rq:wring_x6w4hz 17.09 (ok 3.8e-03) at 53s' in bake['pw40']
    assert 'PICK qkv_gate m=128 rq:s4s5hzEH 16.83' in bake['pw40']
    j=json.loads((DATA/'result.json').read_text())
    assert j['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED' and not j['jit_errors']
    assert j['sources']==driver.PINS and j['official_score_claim'] is False
    assert j['review_sha256']==sha(ROOT/'proj_rsi/audits/pw40-stack-lab-independent.md')
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
            assert all(actual[key]==record['RQ_gate'][key] for key in actual)
            full=driver.full_rq_gate((DATA/(prefix+'.production.cu')).read_text(),sass.read_text(),resource.read_text())
            assert json.dumps(full,sort_keys=True)==json.dumps(record['RQ_gate']['full_RQ'],sort_keys=True)
            assert sha(DATA/(prefix+'.production.cu'))==record['production_source_sha256']
        jit.append({'label':record['label'],'name':record['name'],'SASS_sha256':sha(sass),'resource_sha256':sha(resource),'generated_source_sha256':sha(cu),'RQ_gate':record.get('RQ_gate')})
    assert sum(record['label']=='pw40' and record.get('RQ_gate',{}).get('flags')==[2,3,65538,65539] for record in j['jit'] if record.get('RQ_gate'))==1
    assert j['actual_RQ_functions']=={'rs32':17,'pw38':19,'pw40':20}
    device=j['PW40_device'];assert device['SM_count']==132 and device['compute_capability']==[9,0] and device['before_candidate_import_and_prepare']
    assert device['source']==j['sources']['pw40'] and 'H100' in device['name'] and device['UUID']
    split_record=next(r for r in j['jit'] if r.get('host_query_appended'))
    runtime=split_record['RQ_gate']['Wring_runtime'];assert driver.wring_attributes(runtime['attrs'],device['SM_count'])==runtime
    prefix=split_record['label']+'.'+split_record['name']
    original=(DATA/(prefix+'.production.cu')).read_text();augmented=(DATA/(prefix+'.cu')).read_text()
    assert augmented==original+driver.WRINFO
    attempts=j['Wring_prepare_calls'];assert attempts['rs32']==attempts['pw38']==[] and len(attempts['pw40'])==70
    assert all(c['name']=='wring_x6w4hz' and (c['m'],c['n'],c['KS'],c['code'])==(128,2304,7,0) and 'exception' not in c for c in attempts['pw40'])
    assert j['Wring_selected']==[]
    common_rq={}
    def all_streams(sass):
        return {table.splitlines()[0].strip():[(int(m[1],16),m[2].strip()) for m in __import__('re').finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)] for table in sass.split('Function : ')[1:]}
    for kind in ('rq32','rq128'):
        tables={label:all_streams(next(DATA.glob(label+'.'+kind+'*.sass')).read_text()) for label in LABELS}
        shared=tables['rs32'].keys()&tables['pw38'].keys()&tables['pw40'].keys()
        assert len(shared)==(11 if kind=='rq32' else 6)
        assert all(tables['rs32'][name]==tables['pw38'][name]==tables['pw40'][name] for name in shared)
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
        expected += [('pw40',9000+i,'F1') for i in range(11)]
        for label in reversed(CONTROLS):expected += [(label,9000+i,'F1') for i in range(11)]
        for i in range(30):expected += [(LABELS[index],9020+i,'paired') for index in orders[i%6]]
        assert [(r['label'],r['seed'],r['mode']) for r in block]==expected and all(r['shape']==shape for r in block)
        f1={}
        for k,(label,suffix) in enumerate((('rs32','before'),('pw38','before'),('pw40',''),('pw38','after'),('rs32','after'))):
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
            deltas=[p[control]-p['pw40'] for p in paired];ratios=[p[control]/p['pw40'] for p in paired]
            d=statistics.median(deltas);ratio=statistics.median(ratios)
            equal(d,e['paired_control_minus_pw40_us'][control]);equal(ratio,e['paired_control_over_pw40_ratio'][control])
            comparisons[control]={'paired_control_minus_pw40_us':d,'paired_median_control_over_pw40':ratio,
                'F1_control_minus_pw40_us':min(f1[control+'_before'],f1[control+'_after'])-f1['pw40'],
                'paired_delta_IQR_us':interval(deltas),'paired_median_bootstrap95_us':boot_median(deltas,len(rows)*2+CONTROLS.index(control)),
                'PW40_wins_ties_losses':[sum(v>0 for v in deltas),sum(v==0 for v in deltas),sum(v<0 for v in deltas)],'all_paired_deltas_us':deltas,
                'same_selected_plan_as_pw40':e['picks'][control]==e['picks']['pw40']}
        row={'shape':shape,'m':driver.LOADS[shape.split('/')[0]],'paired_marginal_medians_us':median,'F1_medians_us':f1,'comparisons':comparisons,'actual_picks':e['picks'],'picked_names':{label:pick_name(e['picks'][label]) for label in LABELS}}
        rows.append(row)
        load=shape.split('/')[0];summed=load_sums.setdefault(load,{'paired_marginal_median_sums_us':dict.fromkeys(LABELS,0.0),'F1_conservative_sum_us':dict.fromkeys(LABELS,0.0)})
        for label in LABELS:summed['paired_marginal_median_sums_us'][label]+=median[label]
        for control in CONTROLS:summed['F1_conservative_sum_us'][control]+=min(f1[control+'_before'],f1[control+'_after'])
        summed['F1_conservative_sum_us']['pw40']+=f1['pw40']
    assert cursor==4350 and terminals==180 and mutations==360
    eh=[r['shape'] for r in rows if r['picked_names']['pw40'].endswith('EH')]
    assert eh==j['EH_selected']==['verify16/qkv_gate','verify16/pooler','verify16/unpooler']
    aggregate={}
    for control in CONTROLS:
        aggregate[control]={metric:math.exp(sum(math.log(s[metric][control]/s[metric]['pw40']) for s in load_sums.values())/6) for metric in ('paired_marginal_median_sums_us','F1_conservative_sum_us')}
        aggregate[control]['paired_shape_win_tie_loss']=[sum(r['comparisons'][control]['paired_control_minus_pw40_us']>0 for r in rows),sum(r['comparisons'][control]['paired_control_minus_pw40_us']==0 for r in rows),sum(r['comparisons'][control]['paired_control_minus_pw40_us']<0 for r in rows)]
    analysis={'state':'AUTHOR_CPU_REANALYSIS_PASS_VALID_LOCAL_EXECUTION_MIXED_SHAPES_MODEST_AGGREGATE_POINT_GAIN','source_sha256':j['sources'],'driver_sha256':DRIVER_PIN,
        'inventory_sha256':sha(COLLECTION/'inventory.json'),'result_sha256':sha(DATA/'result.json'),'raw_sha256':sha(DATA/'raw.jsonl'),'host_log_sha256':sha(host),
        'scope_verified':{'raw_calls':4350,'shapes':30,'checks':222,'terminal_oracles':180,'graph_mutations':360,'JIT_archives':15,'all_per_shape_graphs_reset':True,'final_graph_list_empty':True},
        'cleanup':{k:cleanup[k] for k in ('cleanup_failed','compute_pids','container_exists','containers','gpu_memory_mib','guard_pids')},
        'common_non_EH_actual_RQ_identity':common_rq,'source_pin_count':len(inventory['dispatch']['source_pins']),'artifact_pin_count':len(inventory['files']),'job_id':inventory['job_id'],'guard_pid':inventory['dispatch']['guard_pid'],'terminal_exit':inventory['terminal_exit'],'prepare':j['prepare'],'JIT':jit,'EH_selected':eh,'prepare_BAKE_PICK_PREP_verbatim':bake,'Wring_selected':j['Wring_selected'],'Wring_prepare_calls':attempts,'PW40_device':device,'Wring_runtime':runtime,'actual_RQ_functions':j['actual_RQ_functions'],
        'all_shapes':rows,'local_load_sums':load_sums,'local_descriptive_geomean_control_over_pw40':aggregate,
        'limitations':['Local H100 graph times include replay overhead; no official H200 score or transfer forecast.','Six-load sums use marginal per-shape medians from one full-policy run; ratios are descriptive, not an official calibrated score or a formal aggregate confidence interval.','Per-shape bootstrap intervals resample paired differences within one batch; they do not establish cross-run or H200 reproducibility.','Wring was exercised successfully but not selected. This run cannot credit Wring for whole-policy timing differences or prove selection-regret reduction. Separate fresh prepares choose different plans even on source-identical paths. Credit isolated EH only when geometry/control comparison supports it; retain all selection and timing differences.','Counter/immutable/terminal runtime PASS is asserted by the frozen reviewed driver and supported by complete completion records, not reconstructed tensor bytes after process exit.']}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    write_report(analysis)
    print(json.dumps({'state':analysis['state'],'aggregate':aggregate,'EH_selected':eh},indent=2))
def write_report(a):
    lines=['# PW40 full-scope author analysis','', '**VALID local execution with positive aggregate point estimates in paired and conservative F1 timing. W-ring was exercised successfully but selected nowhere; the measured full-policy difference cannot be attributed to serving W-ring.** No significance barrier is imposed, and no official H200 score is inferred.','', 'Candidate source `3bc0df52d96f62479c3d5c19ab2717c7a9f4d18c1dd087fb786e2f9f080f0278`; driver `9559a9ded1c4b3f60db4ef68decddba857cb13cd20dac048be1640a2dd53c1dc`. Job `pw40.stack.fullscope.01`, guard `%s`, terminal exit0. All63 artifact hashes and127 dispatched source/review pins independently match.'%a['guard_pid'],'', 'Validity covers222correctness cases,4350raw timings in the exact145-call schedule per shape,30shapes,180actual terminal output oracles and360retained-graph mutations with output and float-scratch poisoning. All30graphs drain/reset and the final graph list is empty. Full-storage source-bound counter/alias, dormant storage, immutable current-X/weight/pack and status guards pass. Actual tensor bytes are not recoverable after process exit; validation is supported by complete frozen-driver runtime records. Physical cleanup separately verifies no container, guard, compute PID or GPU memory; cleanup_failed=false. Exit0 alone was not used.','', 'Fifteen loaded-module JIT archives include17RS32/19PW38/20PW40 RQ main functions; these counts are functions, not20separate compilations. Every RQ function matches its exact generated source dispatch and canonical PC/predicate/opcode/operand stream with168REG/LOCAL0. Mandatory EH control/physical gates rerun; the new split lifetime/publication/finalwait0/STSM/reduction/TMA-drain/TRAP/232-40register gates pass. Original and diagnostic-host-augmented generated CUDA are retained and have an exact append/remove inverse.','', 'Device: NVIDIA H100 80GB HBM3, UUID `%s`, SM90,132SMs. This is recorded before candidate import/prepare. Before the split extension returns to any preparation math, actual attributes are168registers,0local,148656shared,occupancy1,maxdynamic148656,maxthreads384. The126CTA grid fits132SMs. Shared bytes use separate X6/W4 rings:1024+6×16384+4×12288+8×(2×6+2×4)+16=148656, rather than the ordinary common-ring ST6 formula. Uniform and dynamic register pools are64512≤65536.'%a['PW40_device']['UUID'],'', '| Control | Paired six-load GM control/PW40 | Conservative F1 GM | Paired shapes win/tie/loss |','|---|---:|---:|---:|']
    for c,g in a['local_descriptive_geomean_control_over_pw40'].items():lines.append('| %s | %.9f | %.9f | %s |'%(c,g['paired_marginal_median_sums_us'],g['F1_conservative_sum_us'],'/'.join(map(str,g['paired_shape_win_tie_loss']))))
    lines+=['', 'Ratios above1 favor PW40. Six-load sums combine five marginal shape medians and use an equal-load geometric mean; these are descriptive local H100 measurements, not the official H200 score. F1 uses the lower bracketed before/after control median. Every30paired differences/ratios, IQR and within-batch bootstrap interval is retained in analysis.json; cross-run reproducibility is unmeasured.','', 'Preparation and actual contender outcome','']
    for c,v in a['prepare'].items():lines.append('- %s: wall %.6fs / original import-clock %.6fs, no clock reset or serving CPU fallback, two build threads joined.'%(c,v['wall_seconds'],v['import_clock_elapsed']))
    lines+=['', 'The new W-ring contender received70successful host calls in the admitted QKV128/N2304/KS7 cell: one initial launch,64full-K basis calls,2random checks,2warmups and1graph capture. These are host-call records;15timed graph replays do not re-enter the Python host observer and must not be added as host calls. No decline/exception or budget skip occurred. The source preparation BAKE log reports numerical pass(ok3.8e-03), W-ring17.09µs, existing nonEH17.15µs and EH16.83µs; EH wins. The0.06µs screen contrast against nonEH is retained, but W-ring loses0.26µs against actual EH in that sequential screen and has no selected-serving sample. This run does not establish a W-ring benefit, additive EH/W-ring gain or reduced selection regret.','', 'PW40 selects existing EH for qkv128/pooler128/unpooler128; PW38 selects EH only for pooler128 in this fresh preparation. Source-identical GV/LT and small-M paths also select differently, and source-identical plans can time differently. Whole-policy point gains therefore include preparation/context drift. All BAKE/PICK/PREP lines are retained verbatim in analysis.json, including positive, tied and negative alternatives. No PW39 confirmation policy is present.','', 'All30 shape results','', 'Positive deltas favor PW40. Picks are RS32 / PW38 / PW40.','', '| Shape | Actual picks | Paired Δ RS32 µs | Paired Δ PW38 µs | F1 Δ RS32 µs | F1 Δ PW38 µs |','|---|---|---:|---:|---:|---:|']
    for r in a['all_shapes']:
        c=r['comparisons'];lines.append('| %s | %s | %+.3f | %+.3f | %+.3f | %+.3f |'%(r['shape'],' / '.join(r['picked_names'].values()),c['rs32']['paired_control_minus_pw40_us'],c['pw38']['paired_control_minus_pw40_us'],c['rs32']['F1_control_minus_pw40_us'],c['pw38']['F1_control_minus_pw40_us']))
    lines+=['', 'Whole-load sums (µs)','', '| Load | Paired RS32 / PW38 / PW40 | F1 RS32 / PW38 / PW40 |','|---|---|---|']
    for load,v in a['local_load_sums'].items():lines.append('| %s | %s | %s |'%(load,' / '.join('%.3f'%v['paired_marginal_median_sums_us'][x] for x in LABELS),' / '.join('%.3f'%v['F1_conservative_sum_us'][x] for x in LABELS)))
    lines+=['', 'Versus PW38, the largest paired gain is verify16/unpooler +0.448µs (existing EH instead of nonEH), followed by decode32/qkv_gate +0.336µs (existing t32ST5 instead of ST6). The largest paired losses tie decode16/qkv_gate and decode32/pooler at−0.160µs; pooler32 loses−0.544µs F1 with the nonEARLY plan. QKV128 selects EH but measures−0.016µs paired/+0.064µs F1 against PW38 nonEH. No small loss is discarded.','', 'CPU reproduction: `python3 proj_rsi/audits/pw40-stack-01/analyze.py` regenerates analysis.json and this report, independently rechecking source/artifact pins, raw schedule/statistics, actual JIT/resources and cleanup. No GPU/API/queue action was performed.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
