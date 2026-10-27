"""Independent CPU recomputation of retained W-ring four-policy evidence."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAND=ROOT/'proj_rsi/candidates/pw36_wring_rebase'
LABELS=('incumbent','coupled','equal','split')
CONTRASTS=(('incumbent','split'),('coupled','equal'),('coupled','split'),('equal','split'),('incumbent','equal'),('incumbent','coupled'))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def oracle(message):
    if message=='exact complete zero':return
    match=re.fullmatch(r'relative L2 error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\); (\d+) probe rows exact to ([0-9.e+-]+)',message)
    assert match,message
    assert float(match[1])<=.01 and float(match[2])<=.05 and float(match[4])<=.001,message

def analyze_data(d,path,gates):
    assert d['status']=='VALID_RESEARCH_RESULT_NO_PROMOTION'
    assert d['source_sha256']==sha(CAND/'submission.py') and d['manifest_sha256']==sha(CAND/'manifest.json')
    actual=(path.parent/'result.jit.sass').read_text();incumbent=(path.parent/'result.incumbent.jit.sass').read_text()
    assert hashlib.sha256(actual.encode()).hexdigest()==d['actual_JIT']['sha256']
    canonical=gates.streams((CAND/'sass.txt').read_text())
    assert gates.streams(actual)=={a:canonical[a] for a in gates.ARMS.values()}
    physical=json.loads(json.dumps(gates.physical(actual,expected=3)))
    assert physical==d['actual_JIT']['physical'] and d['actual_JIT']['exact_all_main_streams'] is True
    selected=d['actual_incumbent'];assert selected['variant']=='s4s5hz' and selected['KS']==7 and selected['not_forced'] is True and selected['exact_selected_parent_JIT'] is True
    selected_args=(128,5,1,0,0,0,3)
    assert gates.streams(incumbent)[selected_args]==gates.streams((CAND/'control-sass.txt').read_text())[selected_args]
    assert set(d['attrs'])==set(LABELS[1:])
    for label,r in d['attrs'].items():
        assert r['regs']==168 and r['local']==0 and r['max_threads']>=384 and r['occupancy']>=1
        assert r['smem']=={'coupled':144520,'equal':144560,'split':148656}[label] and r['max_dynamic']>=r['smem']
        assert ((r['regs']+7)//8)*8*384==64512 and 232*256+40*128==64512
    assert set(d['validation'])==set(LABELS)
    for records in d['validation'].values():
        assert records[0]['all8192_basis']=='EXACT' and records[0]['FP64']=='PASS'
        assert {r['task_distribution'] for r in records if 'task_distribution' in r}=={'normal','outliers'}
        for r in records[1:]:assert r['fresh_current_X_and_output'] is True;oracle(r['result'])
    t=d['timing'];calls=t['all_calls'];assert len(calls)==295
    assert all(isinstance(r['us'],(int,float)) and not isinstance(r['us'],bool) and math.isfinite(r['us']) and r['us']>0 for r in calls)
    assert [r['label'] for r in calls[:20]]==[label for label in LABELS for _ in range(5)]
    assert all(r['seed']==600000 and r['mode']=='warmup' for r in calls[:20])
    raw={};seen={};last={};sequence=['incumbent','coupled','equal','split','equal','coupled','incumbent']
    for block,label in enumerate(sequence):
        rows=calls[20+21*block:20+21*(block+1)]
        assert all(r['label']==label and r['mode']=='F1' for r in rows)
        assert [r['seed'] for r in rows]==list(range(601000+21*block,601000+21*(block+1)))
        key='split' if label=='split' else label+('_before' if seen.get(label,0)==0 else '_after')
        seen[label]=seen.get(label,0)+1;raw[key]=[r['us'] for r in rows];last[label]=rows[-1]['seed']
    assert raw==t['F1_raw_us'];med={k:statistics.median(v) for k,v in raw.items()};assert med==t['F1_medians_us']
    paired=[];orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[i:]+seq[:i] for i in range(4)])
    for rep in range(32):
        rows=calls[167+4*rep:167+4*(rep+1)]
        assert [r['label'] for r in rows]==[LABELS[i] for i in orders[rep%8]]
        assert all(r['seed']==602000+rep and r['mode']=='paired' for r in rows)
        paired.append({r['label']:r['us'] for r in rows})
    assert paired==t['paired_raw_us']
    for phase,expected in [('F1',last),('paired',dict.fromkeys(LABELS,602031))]:
        gate=t['terminal_checks'][phase];assert gate['current_X_matches_last_call'] is True and gate['before_any_replay_or_mutation'] is True
        assert len(gate['checks'])==4 and {r['label'] for r in gate['checks']}==set(LABELS)
        for r in gate['checks']:assert r['input_seed']==expected[r['label']];oracle(r['result'])
    contrasts={};distributions={}
    for a,b in CONTRASTS:
        fa=med[a] if a=='split' else min(med[a+'_before'],med[a+'_after'])
        fb=med[b] if b=='split' else min(med[b+'_before'],med[b+'_after'])
        delta=[r[a]-r[b] for r in paired];mid=statistics.median(delta);key=a+'_minus_'+b
        contrasts[key]={'F1_gain_us':fa-fb,'paired_median_gain_us':mid,'paired_wins':sum(v>0 for v in delta)}
        distributions[key]={'median_us':mid,'min_us':min(delta),'max_us':max(delta),'MAD_us':statistics.median(abs(v-mid) for v in delta),
            'b_win_tie_loss':[sum(v>0 for v in delta),sum(v==0 for v in delta),sum(v<0 for v in delta)]}
    assert contrasts==t['comparisons']
    mutations=d['actual_graph_mutations'];assert len(mutations)==20
    assert {(r['label'],r['input']) for r in mutations}=={(l,k) for l in LABELS for k in ('original','zero','normal','outliers','all_rows')}
    for r in mutations:
        assert r['NaN_poison_overwritten'] is True;oracle(r['oracle'])
        if r['input']=='zero':assert r['oracle']=='exact complete zero'
        if r['input']=='all_rows':assert '128 probe rows exact to 0.0e+00' in r['oracle']
    expected={'incumbent':3,'coupled':3,'equal':1283,'split':1027}
    assert set(d['profile'])==set(LABELS)
    for label,flag in expected.items():
        st=6 if label=='split' else 5
        assert len(d['profile'][label])==1 and f'rq<128, {st}, 1, 0, 0, 0, {flag}>' in d['profile'][label][0]
    assert d['graph_cleanup']=='DRAINED_AND_RESET_ALL_CREATED_GRAPHS'
    return {'status':'INDEPENDENT_RECOMPUTATION_PASS','attempt_status':d['status'],'raw_calls':295,'actual_candidate_JIT_mains':3,
        'actual_selected_incumbent_JIT':'EXACT_PARENT_CONTROL','basis_FP64_policies':4,'terminal_oracles':8,'mutation_oracles':20,
        'F1_medians_us':med,'paired_policy_medians_us':{l:statistics.median(r[l] for r in paired) for l in LABELS},
        'comparisons':contrasts,'paired_distributions':distributions,'official_score':'NOT_MEASURED','result_sha256':sha(path)}

def evidence(path):
    sys.path.insert(0,str(CAND));gates=load(CAND/'preflight.py','independent_wring_actual_gates')
    cpu=runpy.run_path(str(ROOT/'proj_rsi/audits/wring-rebase-review/review_cpu.py'))['review']()
    assert cpu['GPU_calls']==0 and len(cpu['terminal']['negative_controls'])==7 and len(cpu['counters']['negative_controls'])==5
    collection=path.parents[3];ip=collection/'inventory.json';inventory=json.loads(ip.read_text())
    assert inventory['job_id']=='research.wring.rebase.gpu.01' and inventory['missing']==[] and inventory['terminal_exit']==0
    for name,pin in inventory['files'].items():assert sha(collection/name)==pin,('inventory',name)
    pins=inventory['dispatch']['source_pins'];assert len(pins)==33
    job=json.loads((ROOT/'proj_rsi/scheduling/jobs/backlog-separate-wring-rebase-gpu-held.json').read_text())
    assert pins==job['files']
    for name,pin in pins.items():assert sha(ROOT/name)==pin,('source',name)
    cleanup=inventory['cleanup_observation'];assert cleanup['terminal_exit']==0 and cleanup['cleanup_failed'] is False and cleanup['container_exists'] is False
    assert cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[] and cleanup['gpu_memory_mib']==0 and cleanup['host_log_exists'] is True
    d=json.loads(path.read_text());result=analyze_data(d,path,gates);negative={}
    changes={
        'raw_seed':lambda v:v['timing']['all_calls'][20].update(seed=1),
        'raw_nan':lambda v:v['timing']['all_calls'][20].update(us=float('nan')),
        'declared_gain':lambda v:v['timing']['comparisons']['incumbent_minus_split'].update(paired_median_gain_us=1.),
        'terminal_own_fixture':lambda v:v['timing']['terminal_checks']['F1']['checks'][1].update(input_seed=1),
        'mutation_poison':lambda v:v['actual_graph_mutations'][0].update(NaN_poison_overwritten=False),
        'resource_smem':lambda v:v['attrs']['split'].update(smem=144560),
        'forced_incumbent':lambda v:v['actual_incumbent'].update(not_forced=False),
    }
    for name,change in changes.items():
        altered=copy.deepcopy(d);change(altered)
        try:analyze_data(altered,path,gates)
        except AssertionError:negative[name]='REJECTED'
        else:raise AssertionError(('negative admitted',name))
    result.update(inventory_sha256=sha(ip),inventory_files=len(inventory['files']),source_pins=len(pins),analysis_negative_controls=negative,
        CPU_source_protocol_counter_lifetime_proof='INDEPENDENTLY_REEXECUTED_PASS',terminal_CPU_negative_controls=cpu['terminal']['negative_controls'],
        counter_CPU_negative_controls=cpu['counters']['negative_controls'],protocol_CPU_negative_controls=cpu['protocol']['negative_controls'],
        source_lifetime_limits=cpu['protocol']['limits'],terminal_exit=0,cleanup_observation={k:v for k,v in cleanup.items() if k!='host_log_tail'})
    (HERE/'independent-cpu-proof.json').write_text(json.dumps(cpu,indent=2)+'\n')
    return result

if __name__=='__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'result.json';result=evidence(path)
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
