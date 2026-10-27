"""Offline QKV runtime evidence validation and unfiltered arithmetic. No GPU/API."""
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/research.qkv.split4.harness.gpu.01/collection-1793077707233847694'
RUN=COLLECTION/'logs/scheduler/qkv-runtime-01/run'
LABELS=('pw36','rs32','new')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def equal(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def stats(v):
    q=statistics.quantiles(v,n=4,method='inclusive')
    return dict(n=len(v),median=statistics.median(v),p25=q[0],p75=q[2],min=min(v),max=max(v))

def generated_parent(text):
    nodes=ast.parse(text).body;values={}
    for n in nodes:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in ('RQCU','RQPROD','RQV'):
                    # Trusted hash-bound frozen arithmetic/list definitions only.
                    values[t.id]=eval(compile(ast.Expression(n.value),'<frozen source data>','eval'),{'__builtins__':{}})
    namespace={'RQCU':values['RQCU']}
    for n in nodes:
        if isinstance(n,ast.FunctionDef) and n.name in ('_rqwg','_rqss','_rqsrc'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<frozen CUDA source factory>','exec'),namespace)
    variants={v[0]:v for v in values['RQV']}
    return namespace['_rqsrc']([variants[name] for name in values['RQPROD']]),values['RQPROD'],variants

def analyze():
    inv=json.loads((COLLECTION/'inventory.json').read_text());data=json.loads((RUN/'result.json').read_text())
    # Exit/status is the first admission boundary; invalid attempts never yield gains.
    if inv.get('terminal_exit')!=0 or data.get('status')!='VALID_LOCAL_COMPARISON_REVIEW_REQUIRED_NO_OFFICIAL_GAIN':
        return dict(disposition='INVALID',terminal_exit=inv.get('terminal_exit'),attempt_status=data.get('status'),failure=data.get('failure'),repair='Retain exact attempt and diagnose first failing source/runtime gate before a new version or explicit setup repair.')
    assert inv['missing']==[]
    cleanup=inv['cleanup_observation']
    assert cleanup['terminal_exit']==0 and cleanup['cleanup_failed'] is False and cleanup['gpu_memory_mib']==0
    assert cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[] and cleanup['container_exists'] is False
    for path,pin in inv['files'].items():assert sha(COLLECTION/path)==pin,path
    for path,pin in inv['dispatch']['source_pins'].items():assert sha(ROOT/path)==pin,path
    review=ROOT/'proj_rsi/scheduling/reviews/qkv-runtime-01.md';assert sha(review)==data['review_sha256']
    assert data['scope']==dict(layer='qkv_gate',m=128,n=2304,k=8192)
    driver=load(ROOT/'proj_rsi/tools/group8_qkv_runtime_lab.py','qkv_analysis_driver');manifest=driver.verify()
    assert sha(ROOT/'proj_rsi/candidates/group8_qkv_runtime_trial/manifest.json')==data['manifest_sha256']
    phase=[p['phase'] for p in data['phases']]
    assert phase.index('candidate_JIT_before_any_candidate_math')<phase.index('actual_task_W_lossless_pack_CSR_and_pipeline_ownership')<phase.index('all8192_basis_and_FP64_normal_outliers_all_three_policies')<phase.index('full_pipeline_F1_and_balanced_pairs')<phase.index('profiles_only_after_both_strict_terminal_gates')
    assert data['graph_cleanup']=='ALL_CAPTURED_GRAPHS_DRAINED_RESET' and data['official_score']=='NOT_MEASURED'
    runtime=load(ROOT/'proj_rsi/candidates/group8_qkv_trial/runtime.py','qkv_analysis_runtime')
    assert hashlib.sha256(runtime.source().encode()).hexdigest()==data['JIT']['new']['source_sha256']
    so=RUN/'candidate.actual.so';assert sha(so)==data['JIT']['new']['so_sha256']
    actual=(RUN/'candidate.jit.sass').read_text();assert sha(RUN/'candidate.jit.sass')==data['JIT']['new']['sass_sha256']
    # Inspect the archived real shared object offline, never load it as a module.
    redump=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(so)],text=True)
    canonical=(ROOT/'proj_rsi/candidates/group8_qkv_trial/host.sass').read_text()
    physical=load(ROOT/'proj_rsi/candidates/operand_reverse_group8/liveness.py','qkv_analysis_liveness')
    kernels={}
    for token in ('rq_qkv_g8_sk4','rx_pack','qkv_reduce'):
        rows=driver.instructions(driver.sass_body(actual,token))
        assert rows==driver.instructions(driver.sass_body(canonical,token))==driver.instructions(driver.sass_body(redump,token))
        kernels[token]={'instructions':len(rows),'actual_SO_collected_SASS_and_canonical':'EXACT'}
        if token=='rq_qkv_g8_sk4':kernels[token]['physical_A_D_fences']=physical.check_one(driver.sass_body(actual,token))
    attrs=data['JIT']['new']['attrs']
    for name,regs,threads in [('pack',22,256),('main',122,512),('reduce',16,256)]:
        assert attrs[name]['0']==regs and attrs[name]['1']==0 and attrs[name]['3']>=threads
    assert data['JIT']['new']['occupancy_after_optin']==1
    assert ((122+7)//8)*8*512==65536
    for label in LABELS[:2]:
        src=ROOT/'proj_rsi/candidates'/label/'submission.py';expected,names,variants=generated_parent(src.read_text())
        captured=RUN/(label+'.selected.cu');assert captured.read_text()==expected
        assert sha(captured)==data['JIT'][label]['source_sha256']
        assert sha(RUN/(label+'.selected.sass'))==data['JIT'][label]['sass_sha256']
        selected=data['actual_prepared_picks'][label]
        assert selected['counter_contract']=='RQ' and selected['selected_rq']['KS']==7 and selected['selected_rq']['n']==2304
        picked=variants[names[selected['selected_rq']['i']]];assert picked==('s4s5hz',128,5,1,0,0,0,3)
        assert data['JIT'][label]['attrs']==[168,0,144520]
        assert len(data['actual_profile_kernels'][label])==1 and 'rq<128, 5, 1, 0, 0, 0, 3>' in data['actual_profile_kernels'][label][0]
    assert data['JIT']['pw36']['so_sha256']==data['JIT']['rs32']['so_sha256']
    profiles=data['actual_profile_kernels']['new'];assert len(profiles)==3
    assert all(sum(token in p for p in profiles)==1 for token in ('rx_pack','rq_qkv_g8_sk4','qkv_reduce'))
    assert profiles[0].startswith('rx_pack') and 'rq_qkv_g8_sk4' in profiles[1] and profiles[2].startswith('qkv_reduce')
    assert data['actual_W_codec']==dict(exact_BF16_bits='PASS',CSR_CAP=32,XP_bytes=2097152,partial_bytes=4718592,input_derived_cache=False)
    assert data['all65536_BF16_storage_pack']=='PASS_STORAGE_ONLY_NOT_NUMERICAL'
    assert data['fresh_pipeline_workspace_output']=='PASS'
    numeric=[r for r in data['validations'] if 'all8192_basis_exact' in r]
    assert {(r['label'],r['dist']) for r in numeric}==set(itertools.product(LABELS,('normal','outliers')))
    assert all(r['all8192_basis_exact'] is True and r['relative_L2']<=4e-3 and r['max_row']<=2e-2 for r in numeric)
    mutations=[r for r in data['validations'] if 'context' in r]
    assert len(mutations)==45
    assert {(r['context'],r['label'],r['mutation']) for r in mutations}==set(itertools.product(('pre_timing','after_storage_bits','post_timing'),LABELS,('original','zero','fresh_normal','fresh_outliers','allrows_probe')))
    assert all(r['NaN_poison']==r['current_input']=='PASS' for r in mutations)
    baselines={r['name']:r for r in data['counter_prepare_baselines']};assert len(baselines)==49
    active={label:tuple(baselines[label+"/rq/('qkv_gate', 128)/cnt"]['storage']) for label in LABELS[:2]}
    count_records=0;dormant_nonzero_names=set();contract_counts=Counter()
    for context in data['counter_checks']:
        assert {r['name'] for r in context['records']}==set(baselines)
        for r in context['records']:
            initial=baselines[r['name']];label=r['name'].split('/')[0]
            expected_contract='reset0' if tuple(initial['storage'])==active[label] else 'dormant_unchanged'
            assert r['contract']==expected_contract
            if expected_contract=='reset0':assert r['final_nonzero']==0
            else:
                assert r['final_digest']==initial['digest'] and r['final_nonzero']==initial['nonzero']
                if r['final_nonzero']:dormant_nonzero_names.add(r['name'])
            contract_counts[expected_contract]+=1;count_records+=1
    assert len(data['counter_checks'])==54 and count_records==2646
    raw=[json.loads(line) for line in (RUN/'raw.jsonl').read_text().splitlines()]
    assert raw==data['raw_calls'] and len(raw)==210
    plan=driver.measurement_plan();assert len(plan)==210
    for index,(r,p) in enumerate(zip(raw,plan)):
        assert r['index']==index and {k:r[k] for k in ('label','input_seed','mode')}==p
        assert math.isfinite(r['us']) and r['us']>0
    assert Counter(r['mode'] for r in raw)==dict(warmup=15,F1=105,paired=90)
    pairs=[{r['label']:r['us'] for r in raw[120+3*i:123+3*i]} for i in range(30)]
    orders=[tuple(r['label'] for r in raw[120+3*i:123+3*i]) for i in range(30)]
    assert Counter(orders)=={o:5 for o in itertools.permutations(LABELS)}
    for label in LABELS:assert Counter(o.index(label) for o in orders)=={p:10 for p in range(3)}
    assert pairs==data['timing']['paired_rows']
    terminal=0
    for phase,seeds in [('F1',dict(pw36=720104,rs32=720083,new=720062)),('paired',{l:730029 for l in LABELS})]:
        records=data['strict_terminal_checks'][phase];assert len(records)==3
        assert {r['label']:r['input_seed'] for r in records}==seeds
        assert all(r['actual_terminal_before_any_replay']=='PASS' for r in records);terminal+=3
    meds=[statistics.median(r['us'] for r in raw[15+21*i:36+21*i]) for i in range(5)]
    assert dict(zip(('pw36_before','rs32_before','new','rs32_after','pw36_after'),meds))==data['timing']['F1_block_medians_us']
    comparison={}
    for label,before,after in [('pw36',0,4),('rs32',1,3)]:
        gains=[r[label]-r['new'] for r in pairs]
        f1=min(meds[before],meds[after])-meds[2];equal(f1,data['timing']['F1_new_gain_us'][label])
        equal(statistics.median(gains),data['timing']['paired_new_gain_us'][label])
        ratio=statistics.median(r[label]/r['new'] for r in pairs);equal(ratio,data['timing']['paired_control_over_new_ratio'][label])
        earlier=[gains[i] for i,o in enumerate(orders) if o.index(label)<o.index('new')]
        later=[gains[i] for i,o in enumerate(orders) if o.index(label)>o.index('new')]
        assert len(earlier)==len(later)==15
        comparison[label]=dict(F1_gain_us=f1,paired_gain=stats(gains),paired_all_gains_us=gains,
            paired_control_over_new_ratio=ratio,paired_slowdown_new_over_control_median=statistics.median(r['new']/r[label] for r in pairs),
            wins=sum(g>0 for g in gains),ties=sum(g==0 for g in gains),losses=sum(g<0 for g in gains),
            control_before_new=stats(earlier),new_before_control=stats(later),
            all_six_order_strata={','.join(o):stats([gains[i] for i,p in enumerate(orders) if p==o]) for o in itertools.permutations(LABELS)})
    assert all(c['wins']==0 and c['losses']==30 for c in comparison.values())
    # Logical request accounting is a source model, never DRAM or elapsed time.
    csrc=(ROOT/'proj_rsi/candidates/group8_qkv/device_source.cu').read_text();host=(ROOT/'proj_rsi/candidates/group8_qkv_trial/host.cuh').read_text()
    assert 'constexpr int KC=32' in csrc and 'dim3(36,4),512' in host
    logical=dict(candidate_CTAs=36*4,selected_control_CTAs=18*7,candidate_X_main_bytes=36*4*32*128*64*2,
        control_X_TMA_bytes=18*128*128*64*2,candidate_pack_read_bytes=2097152,candidate_pack_write_bytes=2097152,
        candidate_FP32_partial_write_bytes=4718592,candidate_FP32_reduce_read_bytes=4718592,
        scope='Logical global payload/request model for frozen geometry, excluding metadata and cache response; not actual DRAM bytes or timing')
    phase_costs={entry['phase']:(data['phases'][i+1]['at'] if i+1<len(data['phases']) else data['ended_at'])-entry['at'] for i,entry in enumerate(data['phases'])}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [COLLECTION/'inventory.json',RUN/'result.json',RUN/'raw.jsonl',RUN/'candidate.actual.so',RUN/'candidate.jit.sass',HERE/'analysis.py']}
    return dict(disposition='VALID_REGRESSION',attempt_status=data['status'],scope=data['scope'],inputs=inputs,
        terminal_exit=0,driver_wall_seconds=data['ended_at']-data['started_at'],phase_elapsed_seconds=phase_costs,cleanup=cleanup,inventory_files_verified=len(inv['files']),source_pins_verified=len(inv['dispatch']['source_pins']),
        actual_SO_and_JIT=kernels,live_resources=data['JIT']['new'],actual_prepared_picks=data['actual_prepared_picks'],
        counts=dict(raw_calls=210,warmup=15,F1=105,paired_calls=90,paired_rounds=30,strict_terminal_oracles=terminal,
                    basis_FP64_task_policies=3,basis_columns_per_policy=8192,task_normal_outlier_cases=6,mutation_oracles=45,counter_contexts=54,counter_records=count_records),
        counters=dict(contract_counts=dict(contract_counts),dormant_nonzero_names=sorted(dormant_nonzero_names),active_alias_contract='All storage aliases of selected QKV RQ counter reset0; dormant values byte/digest unchanged; no blanket-zero rule'),
        numeric_records=numeric,comparison=comparison,F1_block_medians_us=data['timing']['F1_block_medians_us'],
        paired_policy_medians_us={l:statistics.median(p[l] for p in pairs) for l in LABELS},logical_request_model=logical,
        all_raw_calls=raw,official_score_claim=False,
        limits='Collected selected parent .so hashes are recorded, not archived; their exact captured CUDA source/SASS/resources and actual selected dispatch are verified. No new GPU/API/queue actions. No own-control-only or parent-relative promotion; no family ban.')

if __name__=='__main__':
    result=analyze();(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['disposition'])
    for label,c in result.get('comparison',{}).items():print(label,'F1',c['F1_gain_us'],'paired',c['paired_gain']['median'],'wins',c['wins'])
