"""Independent CPU-only audit of the collected all-five v3 attempt. No torch/SSH."""
import ast
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics as stats

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'results/proj-rsi/scheduler/artifacts/smallm.cursor.v3.all5.01/collection-1793074281123122605'
ARTIFACT=ARCHIVE/'logs/scheduler/smallm-cursor-v3-01'
LABELS=('pw36','rs32','increment')
SHAPES=('qkv32','qkv16','pooler32','unpooler32','o32')
ORDER=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def approx(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def ops(sass):
    return {part.splitlines()[0].strip():collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',part)) for part in sass.split('Function : ')[1:]}
def pure_rq_source(path):
    tree=ast.parse(path.read_text());ns={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('RQCU','RQV','RQSBAKE'):
            ns[node.targets[0].id]=ast.literal_eval(node.value)
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss','_rqsrc'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    names=list(dict.fromkeys(name for pairs in ns['RQSBAKE'].values() for name,_ in pairs))
    variants={v[0]:v for v in ns['RQV']}
    return ns['_rqsrc']([variants[n] for n in names])
def base_gv(path):
    tree=ast.parse(path.read_text());ns={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            names={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
            if names and names<= {'_0','_1','_01','GVC'}:
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        if isinstance(node,ast.FunctionDef) and node.name in ('_gvst','_gvinst'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['_gvinst']()
def summary(values):
    med=stats.median(values);q=stats.quantiles(values,n=4,method='inclusive')
    return {'n':len(values),'median_us':med,'min_us':min(values),'max_us':max(values),'q25_us':q[0],'q75_us':q[2],
            'MAD_us':stats.median(abs(x-med) for x in values)}

def main():
    inventory=json.loads((ARCHIVE/'inventory.json').read_text())
    assert inventory['job_id']=='smallm.cursor.v3.all5.01' and inventory['terminal_exit']==0
    assert not inventory.get('missing')
    for rel,digest in inventory['files'].items():assert sha(ARCHIVE/rel)==digest,rel
    cleanup=inventory['cleanup_observation']
    assert cleanup['container_exists'] is False and cleanup['guard_pids']==cleanup['compute_pids']==cleanup['containers']==[]
    assert cleanup['gpu_memory_mib']==0 and cleanup['cleanup_failed'] is False
    host=(ARCHIVE/'logs/scheduler/smallm.cursor.v3.all5.01.host.log').read_text()
    assert host.count('TRIAL_EXIT okc-proj-smallm-cursor-v3-01 0')==1
    result=json.loads((ARTIFACT/'result.json').read_text())
    assert result['status']=='VALID_FOCUSED_DEVELOPMENT_ONLY_FULL_SCOPE_REQUIRED'
    assert result['scope']==list(SHAPES) and 'failure' not in result and 'invariant_failure' not in result
    job=json.loads((ROOT/'proj_rsi/scheduling/jobs/smallm-cursor-v3-01.json').read_text())
    for rel,pin in job['files'].items():assert sha(ROOT/rel)==pin,rel
    assert sha(ROOT/job['review']['path'])==result['review_sha256']==job['review']['sha256']
    spec=importlib.util.spec_from_file_location('counter_v3_cpu',ROOT/'proj_rsi/tools/smallm_increment_lab_v3.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    assert result['counter_source_bindings']==driver.contract_source_check()
    driver.verify();cpu_proof=driver.cpu_check()
    assert 'torch' not in __import__('sys').modules
    jit={}
    for label,(folder,pin) in driver.PINS.items():
        source=ROOT/'proj_rsi/candidates'/folder/'submission.py';assert sha(source)==pin
        cu=ARTIFACT/(label+'.rq32.cu');sass=ARTIFACT/(label+'.sass');resources=ARTIFACT/(label+'.resources')
        assert pure_rq_source(source)==cu.read_text()
        record=result['jit'][label]
        assert sha(cu)==record['source_sha256'] and sha(sass)==record['SASS_sha256']
        assert record['gate']=='before prepare / first candidate math'
        actual=ops(sass.read_text())
        expected=ops((ROOT/'proj_rsi/candidates/smallm_increment'/('after.sass' if label=='increment' else 'before.sass')).read_text())
        for symbol,want in expected.items():
            assert actual[symbol]==want,(label,symbol)
            block=resources.read_text().split('Function '+symbol)[1].split('Function ')[0]
            for name,value in (('REG',168),('STACK',0),('LOCAL',0)):
                assert re.search(r'\b'+name+r'\s*:\s*'+str(value)+r'\b',block),(label,symbol,name)
        jit[label]={'generated_source_exact_from_pinned_AST':True,'source_sha256':sha(cu),'sass_sha256':sha(sass),'resource_sha256':sha(resources),
                    'selected_reference_symbols':list(expected),'recorded_so_sha256':record['so_sha256'],'actual_so_archived':False}
    assert result['jit']['pw36']['source_sha256']==result['jit']['rs32']['source_sha256']
    assert result['jit']['pw36']['so_sha256']==result['jit']['rs32']['so_sha256']
    assert result['jit']['increment']['source_sha256']!=result['jit']['rs32']['source_sha256']
    raw=[json.loads(line) for line in (ARTIFACT/'raw.jsonl').read_text().splitlines()]
    assert raw==result['raw'] and len(raw)==770
    assert all(math.isfinite(x['us']) and x['us']>0 for x in raw)
    phases=[p['phase'] for p in result['phases']]
    wantphases=[phase for label in LABELS for phase in ('compile_and_gate_'+label,'actual_full_prepare_'+label)]
    wantphases += [phase for shape in SHAPES for phase in (*[shape+'_basis_FP64_'+label for label in LABELS],shape+'_timing')]+['complete']
    assert phases==wantphases
    assert all(a['time']<=b['time'] for a,b in zip(result['phases'],result['phases'][1:]))
    baselines=result['counter_prepare_baseline'];assert len(baselines)==73
    assert all(c['dtype']=='torch.int32' and len(c['digest'])==64 and len(c['small_values'])<=32 for c in baselines)
    nonzero=[c for c in baselines if c['nonzero']]
    assert all('/p1cnt/' in c['name'] or c['name'].endswith('/rw_cnt') for c in nonzero)
    shapes={};cursorselected=0
    for shape in SHAPES:
        entry=result['shapes'][shape];rows=[x for x in raw if x['shape']==shape];assert len(rows)==154
        expected=[]
        for label in LABELS:expected += [(label,20000,'warmup')]*3
        for label in LABELS:expected += [(label,20001+i,'F1') for i in range(11)]
        for label in reversed(LABELS[:-1]):expected += [(label,20001+i,'F1') for i in range(11)]
        for i in range(30):expected += [(LABELS[j],20040+i,'paired') for j in ORDER[i%6]]
        assert [(x['label'],x['seed'],x['mode']) for x in rows]==expected
        paired=rows[64:];assert len(paired)==90
        pairs=[{x['label']:x['us'] for x in paired[i:i+3]} for i in range(0,90,3)]
        assert pairs==entry['paired_all_samples']
        assert collections.Counter(tuple(x['label'] for x in paired[i:i+3]) for i in range(0,90,3))==collections.Counter({tuple(LABELS[j] for j in o):5 for o in ORDER})
        vals=entry['validation'];assert len(vals)==20
        basis=[x for x in vals if 'full_K_basis_FP64' in x];assert len(basis)==3
        assert {x['label'] for x in basis}==set(LABELS) and all(x['full_K_basis_FP64'].startswith('ok ') and float(x['full_K_basis_FP64'][3:])<.004 for x in basis)
        mutations=[x for x in vals if 'graph_mutation' in x]
        assert {(x['label'],x['graph_mutation']) for x in mutations}=={(l,m) for l in LABELS for m in ('normal','outliers','zero','all_probes','fresh_normal')}
        assert all(x['NaN_output_fresh_workspace']=='PASS' for x in mutations)
        terminal=[x for x in vals if 'immediate_final_timed_output' in x];assert len(terminal)==2
        for x in terminal:
            seed=20011 if x['immediate_final_timed_output']=='F1' else 20069
            assert x['last_fixtures']=={l:seed for l in LABELS} and x['PASS'] is True
        metrics={};picks={}
        for label in LABELS:
            p=entry['picks'][label];path=result['prepared_policy'][label][shape]['counter_path']
            if p['rq_name']:
                assert path=='RQ' and p['selected_attrs_regs_local_shared'][0:2]==[168,0]
                expected_shared=72840 if shape=='qkv16' else 99488 if 's6' in p['rq_name'] else 83080
                assert p['selected_attrs_regs_local_shared'][2]==expected_shared
                layer=driver.SHAPES[shape][0];name=label+"/rq/"+str((layer,driver.SHAPES[shape][1]))+'/cnt'
                active=[c for c in baselines if c['name']==name];assert len(active)==1 and active[0]['nonzero']==0
                picks[label]={'path':'RQ','name':p['rq_name'],'KS':p['rq']['KS'],'resources':p['selected_attrs_regs_local_shared'],
                              'counter_storage':active[0]['storage'],'cursor_affected':p['cursor_affected_selected']}
                cursorselected+=bool(label=='increment' and p['cursor_affected_selected'])
            else:
                assert path=='GV_no_counter'
                index=int(re.match(r'\((\d+),',p['gv']).group(1))
                instances=base_gv(ROOT/'proj_rsi/candidates'/driver.PINS[label][0]/'submission.py')
                instance=instances[index];assert instance[6]==1 or (instance[9]>>26)&15
                assert not ((instance[9]>>27)&5)
                picks[label]={'path':path,'GV_instance':index,'GV_template':instance,'cursor_affected':False}
        incF=[x['us'] for x in rows[31:42]];assert len(incF)==11
        approx(stats.median(incF),entry['F1_medians_us']['increment'])
        for label in LABELS[:-1]:
            ratios=[p[label]/p['increment'] for p in pairs];deltas=[p[label]-p['increment'] for p in pairs]
            ratio=stats.median(ratios);approx(ratio,entry['paired_control_over_increment_median_ratio'][label])
            before=entry['F1_medians_us'][label+'_before'];after=entry['F1_medians_us'][label+'_after']
            beforeRaw=[x['us'] for x in rows[9:20] if x['label']==label] if label=='pw36' else [x['us'] for x in rows[20:31]]
            afterRaw=[x['us'] for x in rows[53:64]] if label=='pw36' else [x['us'] for x in rows[42:53]]
            approx(before,stats.median(beforeRaw));approx(after,stats.median(afterRaw))
            approx(min(before,after)/stats.median(incF),entry['F1_control_over_increment'][label])
            metrics[label]={'paired_control_over_increment_median_ratio':ratio,'paired_increment_gain_percent':(ratio-1)*100,
                            'paired_signed_control_minus_increment_us':stats.median(deltas),'paired_delta_distribution':summary(deltas),
                            'paired_increment_win_tie_loss':[sum(x>0 for x in deltas),sum(x==0 for x in deltas),sum(x<0 for x in deltas)],
                            'F1_before_us':before,'F1_after_us':after,'F1_increment_us':stats.median(incF),
                            'F1_signed_conservative_control_minus_increment_us':min(before,after)-stats.median(incF),
                            'F1_signed_before_minus_increment_us':before-stats.median(incF),'F1_signed_after_minus_increment_us':after-stats.median(incF)}
        shapes[shape]={'picks':picks,'paired_timing':{l:summary([p[l] for p in pairs]) for l in LABELS},'comparison':metrics,
                       'basis_results':basis,'validation_records':20,'terminal_output_oracles':6}
    assert cursorselected==3
    output={'verdict':'VALID_LOCAL_ALL5_ACTUAL_POLICY_COMPARISON; MIXED_FLAT_AND_REGRESSIONS; NO_OFFICIAL_PROMOTION',
            'job_id':job['id'],'archive':str(ARCHIVE.relative_to(ROOT)),'inventory_sha256':sha(ARCHIVE/'inventory.json'),
            'validation':{'inventory_files':len(inventory['files']),'pinned_job_files':len(job['files']),'raw_calls':770,'shapes':5,
                          'basis_FP64_checks':15,'graph_mutation_checks':75,'strict_terminal_output_oracles':30,
                          'balanced_pairs_per_shape':30,'all_state_invariant_evaluations_from_frozen_control_flow':125,
                          'counter_baseline_views':73,'cursor_actually_selected_shapes':3,'cleanup_verified':True},
            'jit':jit,'counter_contract':{'source_binding_matches':True,'CPU_positive_negative_alias_proof':cpu_proof['counter_contract_proof'],
             'baseline_nonzero_counter_views':len(nonzero),'baseline_nonzero_P1_views':sum('/p1cnt/' in c['name'] for c in nonzero),
             'nonzero_metadata':nonzero,'successful_final_digests_archived':False,
             'conclusion':'Nonzero dormant P1 state is directly observed in all three prepares and successful source-bound invariants preserve it. Exact old failed counter remains UNKNOWN.'},
            'shapes':shapes,'limits':['Actual SO binaries not collected; recorded hashes and source-bound JIT SASS/resource checks retained.',
             'Successful final counter tensor values/digests not individually archived;125 executed invariant checks inferred from frozen source and successful complete control flow.',
             'Fresh independent prepare picks differ; qkv32 and unpooler32 compare different ST, pooler32 pw36 uses non-EARLY variant, qkv16/o32 do not select cursor.',
             'One local H100 attempt does not establish tiny gains as reproducible or predict official H200 score.']}
    (OUT/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'verdict':output['verdict'],'validation':output['validation'],'per_shape':{s:{'picks':d['picks'],'comparison':d['comparison']} for s,d in shapes.items()}},indent=2))

if __name__=='__main__':main()
