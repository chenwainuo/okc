"""Reproduce EH fixed-geometry arithmetic and artifact validity, CPU/std-library only."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/research.eh.samegeometry.gpu.01/collection-1793076695219438309'
RESULT=COLLECTION/'logs/scheduler/eh-same-geometry-01/result.json'
FULL=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.stack.fullscope.01/collection-1793076533579348342/logs/scheduler/pw38-stack-01/result.json'
LABELS=('parent2','parent3','eh2','eh3')
SHAPES={'qkv_gate':(2304,8192,7),'pooler':(4096,8192,4),'unpooler':(8192,4096,2),'o':(8192,1024,2)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def median(values):return statistics.median(values)
def describe(values):
    q=statistics.quantiles(values,n=4,method='inclusive')
    return dict(n=len(values),median=median(values),p25=q[0],p75=q[2],min=min(values),max=max(values))
def loaded(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def analyze():
    data=json.loads(RESULT.read_text());inv=json.loads((COLLECTION/'inventory.json').read_text())
    assert data['status']=='VALID_EH_SAME_GEOMETRY_MECHANISM_ONLY'
    assert data['full_stack_or_actual_selected_incumbent_win_claim'] is False
    assert data['scope']=='EH_FIXED_GEOMETRY_MECHANISM_ONLY_NOT_SELECTED_SERVING_POLICY'
    assert data['all_candidate_JIT_and_resources_before_first_math']=='PASS'
    assert inv['missing']==[] and inv['terminal_exit']==0
    cleanup=inv['cleanup_observation']
    assert cleanup['terminal_exit']==0 and cleanup['cleanup_failed'] is False
    assert cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[]
    assert cleanup['container_exists'] is False and cleanup['gpu_memory_mib']==0
    for name,digest in inv['files'].items():assert sha(COLLECTION/name)==digest,name
    for name,digest in inv['dispatch']['source_pins'].items():assert sha(ROOT/name)==digest,name
    log=COLLECTION/'logs/scheduler/research.eh.samegeometry.gpu.01.host.log'
    text=log.read_text();assert 'TRIAL_EXIT okc-proj-eh-same-geometry-01 0' in text and 'Traceback' not in text
    # Recheck actual .so dumped instruction streams and physical lifetimes without
    # importing the lab, a submission, torch or any CUDA/extension module.
    trial=ROOT/'proj_rsi/candidates/eh_same_geometry_trial';support=loaded(trial/'support.py','eh_analysis_support')
    support.verify_frozen();support.verify_lab()
    assert data['source_pins']==support.PINS and data['source_roles']==support.SOURCES
    backport=ROOT/'proj_rsi/candidates/rs32_eh_backport';sys.path.insert(0,str(backport))
    preflight=loaded(backport/'preflight.py','eh_analysis_physical')
    sasses={};jits={}
    for role,record in data['JIT'].items():
        p=RESULT.with_suffix('.'+role+'.jit.sass');assert sha(p)==record['sha256']
        sasses[role]=p.read_text();gate=support.binary_gate(role,sasses[role]);assert gate==record['gate']
        jits[role]=dict(artifact=str(p.relative_to(ROOT)),sha256=sha(p),recorded_module_sha256=record['module_sha256'],canonical_gate=gate)
    physical=preflight.physical(sasses['parent']+'\n'+sasses['eh'])
    assert len(physical)==4 and all(p['WGMMA_drain_before_STSM']=='PASS' and p['both_TMA_reads_drain_before_exit']=='PASS' for p in physical)
    for label,attrs in data['attrs'].items():
        assert label in LABELS
        expected=dict(regs=168,local=0,shared=144520,occupancy=1,max_threads=384,threads=384,static_rounded_register_pool=64512,dynamic_register_pool=64512)
        assert attrs==expected,(label,attrs)
    analyses={};allcalls=[];terminalcount=mutationcount=0
    for layer,(n,k,ks) in SHAPES.items():
        r=data['shapes'][layer];t=r['timing']
        assert r['geometry']==dict(m=128,n=n,k=k,KS=ks,NT=128,ST=5,GS=1,PRB=0,WM=0,INS=0)
        assert r['status']=='VALID_FIXED_GEOMETRY_RESULT_NO_PRODUCTION_ADMISSION'
        assert r['graph_cleanup']=='DRAINED_RESET_ALL_FOUR'
        assert set(r['validation'])==set(LABELS)
        for label,checks in r['validation'].items():
            assert len(checks)==3 and checks[0]['all_K_basis']=='EXACT' and checks[0]['random_FP64']=='PASS'
            assert checks[0]['tolerance']=={7:0.008,4:0.006,2:0.005}[ks]
            assert [x['task_dist'] for x in checks[1:]]==['normal','outliers']
            assert len(r['graph_nodes'][label])==1
            fl=support.FLAGS[label];assert f'rq<128, 5, 1, 0, 0, 0, {fl}>' in r['graph_nodes'][label][0]
        mutations=r['mutations'];assert len(mutations)==20
        assert {(x['label'],x['input']) for x in mutations}=={(l,p) for l in LABELS for p in ['original','zero','fresh_normal','fresh_outliers','probes']}
        assert all(x['NaN_poison_overwritten'] is True for x in mutations);mutationcount+=len(mutations)
        calls=t['all_raw_calls'];assert len(calls)==274
        assert all(math.isfinite(c['us']) and c['us']>0 for c in calls)
        assert Counter(c['mode'] for c in calls)==dict(warmup=20,F1=126,paired=128)
        warm=calls[:20];assert [(c['label'],c['input_seed'],c['warmup']) for c in warm]==[(l,239999,True) for l in LABELS for _ in range(5)]
        blocks=[('parent2_before','parent2'),('eh2','eh2'),('parent2_after','parent2'),('parent3_before','parent3'),('eh3','eh3'),('parent3_after','parent3')]
        for block,(key,label) in enumerate(blocks):
            selected=calls[20+21*block:20+21*(block+1)]
            assert [(c['label'],c['input_seed'],c['warmup']) for c in selected]==[(label,240000+21*block+i,False) for i in range(21)]
            assert [c['us'] for c in selected]==t['F1_samples_us'][key]
            equal(median(c['us'] for c in selected),t['F1_medians_us'][key])
        paired_calls=calls[146:];paired=[];orders=[]
        for i in range(32):
            group=paired_calls[4*i:4*i+4];assert {c['label'] for c in group}==set(LABELS)
            assert {c['input_seed'] for c in group}=={250000+i}
            row={c['label']:c['us'] for c in group};assert row==t['paired_samples_us'][i]
            paired.append(row);orders.append(tuple(c['label'] for c in group))
        counts=Counter(orders);assert len(counts)==8 and set(counts.values())=={4}
        for label in LABELS:assert Counter(o.index(label) for o in orders)=={i:8 for i in range(4)}
        for phase,last in [('F1',{'parent2':240062,'parent3':240125,'eh2':240041,'eh3':240104}),('paired',{l:250031 for l in LABELS})]:
            records=t['strict_terminal_checks'][phase];assert len(records)==4
            assert {x['label']:x['input_seed'] for x in records}==last
            assert all(x['actual_terminal_output_before_any_replay']=='PASS' for x in records)
            terminalcount+=len(records)
        assert t['terminal_input_seed_per_policy']=={l:250031 for l in LABELS}
        comparisons={}
        for parent,new in [('parent2','eh2'),('parent3','eh3')]:
            med=t['F1_medians_us'];gains=[p[parent]-p[new] for p in paired]
            actual=t['comparisons'][new];f1=min(med[parent+'_before'],med[parent+'_after'])-med[new]
            equal(f1,actual['F1_gain_us']);equal(median(gains),actual['paired_median_gain_us'])
            assert gains==actual['paired_all_gains_us'];assert sum(x>0 for x in gains)==actual['paired_wins']
            equal(median(p[parent]/p[new] for p in paired),actual['paired_median_ratio'])
            assert actual['gain_threshold_us']==0 and actual['serving_incumbent_win_claim'] is False
            by_order={','.join(order):describe([gains[i] for i,o in enumerate(orders) if o==order]) for order in counts}
            before=[gains[i] for i,o in enumerate(orders) if o.index(parent)<o.index(new)]
            after=[gains[i] for i,o in enumerate(orders) if o.index(parent)>o.index(new)]
            assert len(before)==len(after)==16
            classification='BOTH_ESTIMATORS_POSITIVE_COMPONENT_OBSERVATION' if f1>0 and median(gains)>0 else 'BOTH_ESTIMATORS_NEGATIVE_THIS_VERSION' if f1<0 and median(gains)<0 else 'ESTIMATORS_DISAGREE_UNCERTAIN'
            comparisons[new]=dict(control=parent,FL_control=support.FLAGS[parent],FL_EH=support.FLAGS[new],F1_gain_us=f1,
                F1_control_before_us=med[parent+'_before'],F1_control_after_us=med[parent+'_after'],F1_EH_us=med[new],
                paired=describe(gains),paired_all_gains_us=gains,paired_ratio=actual['paired_median_ratio'],
                wins=sum(g>0 for g in gains),ties=sum(g==0 for g in gains),losses=sum(g<0 for g in gains),
                all_8_order_strata=by_order,control_before_EH=describe(before),EH_before_control=describe(after),
                first16=describe(gains[:16]),last16=describe(gains[16:]),classification=classification)
        analyses[layer]=dict(geometry=r['geometry'],comparisons=comparisons,raw_calls=274,validation_all4_basis_FP64_task='PASS',terminal_oracles=8,mutations=20,one_node_profiles=4,graph_cleanup=r['graph_cleanup'])
        allcalls.extend(dict(layer=layer,**c) for c in calls)
    assert len(allcalls)==1096 and terminalcount==32 and mutationcount==80
    full=json.loads(FULL.read_text());assert full['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED'
    full_comparison={}
    for layer in SHAPES:
        r=full['shapes']['verify16/'+layer];med=r['F1_medians_us'];rows=r['paired_all_samples']
        per_control={}
        for label in ['rs32','pw36']:
            g=[p[label]-p['pw38'] for p in rows];equal(median(g),r['paired_control_minus_pw38_us'][label])
            per_control[label]=dict(F1_gain_us=min(med[label+'_before'],med[label+'_after'])-med['pw38'],paired=describe(g),wins=sum(x>0 for x in g),ties=sum(x==0 for x in g),losses=sum(x<0 for x in g))
        full_comparison[layer]=dict(picks=r['picks'],comparisons=per_control,unchanged_control_PW36_minus_RS32=describe([p['pw36']-p['rs32'] for p in rows]))
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [RESULT,COLLECTION/'inventory.json',log,FULL,HERE/'analysis.py']}
    return dict(state='VALID_EH_FIXED_GEOMETRY_ALL_RAW_RECONSTRUCTED_NO_PRODUCTION_CLAIM',inputs=inputs,
                frozen_dispatch_pins_verified=len(inv['dispatch']['source_pins']),inventory_files_verified=len(inv['files']),
                sources=data['source_pins'],actual_JIT=jits,actual_physical_lifetimes=physical,live_resources=data['attrs'],
                counts=dict(raw_calls=1096,F1=504,paired_calls=512,warmup=80,paired_rounds=128,terminal_oracles=32,mutations=80),
                cleanup=cleanup,terminal_exit=0,shapes=analyses,full_PW38_comparison=full_comparison,
                full_PW38_result_sha256=sha(FULL),all_raw_calls=allcalls,
                interpretation='Retain every signed delta; no minimum-gain cutoff, no posthoc best-order selection, no additive transfer, no serving or official score claim. Order/temporal strata are descriptive sensitivity, not extra independent replicates.')

if __name__=='__main__':
    result=analyze();(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['state'])
    for layer,r in result['shapes'].items():
        for label,c in r['comparisons'].items():print(layer,label,'F1',round(c['F1_gain_us'],6),'paired',round(c['paired']['median'],6),'W/T/L',c['wins'],c['ties'],c['losses'],'orderbefore',round(c['control_before_EH']['median'],6),'orderafter',round(c['EH_before_control']['median'],6))
