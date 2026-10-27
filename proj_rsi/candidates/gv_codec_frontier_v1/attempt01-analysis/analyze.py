"""Read-only completed-data audit; preserves the actual terminal cleanup failure."""
import ast
import hashlib
import json
import math
import statistics
from pathlib import Path

D=Path(__file__).resolve().parent
ROOT=D.parents[3]
C=ROOT/'results/proj-rsi/scheduler/artifacts/research.gv.codec.frontier.two-cell.01/collection-1793134176827697877'
inventory=json.loads((C/'inventory.json').read_text())
run=C/'logs/scheduler/gv-codec-frontier-two-cell-01/run'
result=json.loads((run/'result.json').read_text())
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
for rel,pin in inventory['files'].items():
    check('artifact '+rel,hashlib.sha256((C/rel).read_bytes()).hexdigest()==pin)
check('actual internal cleanup failure retained',result['state']=='INVALID_CLEANUP')
check('exact 32MiB post-scope retained allocation',result['post_scope_cleanup']['allocated']==result['post_scope_cleanup']['reserved']==33554432)
external=inventory['cleanup_observation']
check('nonzero trial exit retained',external['terminal_exit']==1)
check('independent external cleanup',not external['cleanup_failed'] and not external['container_exists'] and not external['compute_pids'] and external['gpu_memory_mib']==0)
check('actual source identity',result['source_sha256']==inventory['dispatch']['source_pins']['proj_rsi/candidates/gv_codec_frontier_v1/native_v2_capacity32_48/native.cu'])
check('control native identity',result['actual_control_native_identity']=='EXACT_PC_PREDICATE_ENCODING')
check('no profiler/prepare',result['profiler_calls']==result['full_prepare_calls']==0)
driver=ROOT/'proj_rsi/candidates/gv_codec_frontier_v1/native_v2_capacity32_48/driver.py'
ns={}
nodes=[]
for node in ast.parse(driver.read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='LABELS'for t in node.targets):nodes.append(node)
    if isinstance(node,ast.FunctionDef)and node.name in ('schedule','basis_columns'):nodes.append(node)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-protocol','exec'),ns)
stats=json.loads((D.parent/'actual-stats-proof.json').read_text())['layers']
raw=[json.loads(line)for line in (run/'raw.jsonl').read_text().splitlines()]
check('complete182raw rows',len(raw)==182)
cells={}
for key,entry in result['fixed_cells'].items():
    config=entry['config'];layer=config['layer'];m=config['m'];n=config['n'];k=config['k']
    rows=entry['raw']
    check(layer+' full91schedule',len(rows)==91 and all({a:b for a,b in row.items()if a!='us'}==want for row,want in zip(rows,ns['schedule']())))
    check(layer+' rawJSONL identity',[{a:b for a,b in row.items()if a!='m'}for row in raw if row['m']==m]==rows)
    check(layer+' finite positive samples',all(math.isfinite(row['us']) and row['us']>0 for row in rows))
    check(layer+' C11 actual fixture and bytes',entry['pack']['actual_fixture_sha']==stats[layer]['weight_sha256'] and entry['pack']['C11_bytes']==stats[layer]['C11_total_bytes'] and entry['pack']['C12_bytes']==stats[layer]['C12_total_bytes'])
    check(layer+' exact all-weight reconstructions',entry['pack']['actual_W_bit_reconstruction']=='EXACT_BOTH_FORMATS')
    attrs=result['raw_attributes'][layer]
    expected=([64,0,0,1024,17408,1],[64,0,0,1024,20480,1]) if layer=='pooler' else ([110,0,0,128,5120,4],[108,0,0,128,6144,4])
    check(layer+' exact resource gates',(attrs['control'],attrs['frontier'])==expected)
    for arm in ns['LABELS']:
        validation=entry['validation'][arm]
        check(layer+arm+' normal/outlier/zero',[v['kind']for v in validation['task']]==['normal','outliers','zero'])
        for site in validation['task']:
            oracle=site['oracle'];check(layer+arm+site['kind']+' oracle',oracle['current_X']=='EXACT'and oracle['FP64_L2']<=.004 and oracle['worst_row']<=.02)
        check(layer+arm+'64basis columns',validation['basis']=='EXACT'and validation['basis_columns']==ns['basis_columns'](k))
    for phase,end in (('F1',43),('paired',91)):
        sites=entry['terminals'][phase]
        check(layer+phase+' own2terminals',len(sites)==2 and {s['arm']for s in sites}==set(ns['LABELS']))
        for site in sites:
            seed=next(row['seed']for row in reversed(rows[:end])if row['arm']==site['arm']and row['mode']!='warmup')
            oracle=site['oracle']
            check(layer+phase+site['arm']+' own timed output',site['seed']==seed and site['before_any_replay_or_mutation'] and oracle['current_X']=='EXACT'and oracle['FP64_L2']<=.004 and oracle['worst_row']<=.02)
    check(layer+'10retained graph mutations',len(entry['mutations'])==10)
    for site in entry['mutations']:
        check(layer+site['arm']+site['kind']+' mutation',site['NaN_output_overwritten']and site['oracle']['current_X']=='EXACT'and site['oracle']['FP64_L2']<=.004 and site['oracle']['worst_row']<=.02)
    check(layer+'graph drain/reset',entry['cleanup']=='RESET_AFTER_DRAIN')
    f1=[statistics.median(row['us']for row in rows[10+11*i:21+11*i])for i in range(3)]
    paired=[{row['arm']:row['us']for row in rows[43+2*i:45+2*i]}for i in range(24)]
    delta=statistics.median(row['control']-row['frontier']for row in paired)
    fdelta=(f1[0]+f1[2])/2-f1[1]
    check(layer+' saved contrasts reproduce',delta==entry['contrasts']['paired_median_control_minus_new_us']and fdelta==entry['contrasts']['F1_endpoint_average_control_minus_new_us'])
    units=n*k//1024;extra=units*8;unique=entry['pack']['C11_bytes'];old=entry['pack']['C12_bytes']
    cells[layer]=dict(raw_samples=91,F1_medians_us=f1,paired_control_median_us=statistics.median(row['control']for row in paired),
        paired_frontier_median_us=statistics.median(row['frontier']for row in paired),unadmitted_paired_control_minus_new_us=delta,
        unadmitted_F1_endpoint_control_minus_new_us=fdelta,paired_new_wins=sum(row['control']>row['frontier']for row in paired),
        C11_unique_format_bytes=unique,C12_payload_and_fixed_escape_bytes=old,extra_boundary_requested_bytes=extra,
        C11_format_plus_duplicate_boundary_requests=unique+extra,unique_format_fraction_saved=1-unique/old,
        source_requests_fraction_saved=1-(unique+extra)/old,measured_HBM_bytes='NOT_COLLECTED',actual_resources=attrs)
report=dict(disposition='INVALID_CLEANUP',timing_admission='NOT_ADMITTED_AS_VALID_PERFORMANCE',
    completed_numeric_and_timing_evidence=True,old_result_not_rewritten=True,checks_passed=len(checks),checks=checks,
    cells=cells,internal_cleanup=result['post_scope_cleanup'],external_cleanup=external,
    inventory_sha256=hashlib.sha256((C/'inventory.json').read_bytes()).hexdigest(),
    raw_result_sha256=hashlib.sha256((run/'result.json').read_bytes()).hexdigest(),
    root_action='Preserve completed data with post-tail failure; no identical full rerun solely for workspace cleanup; incorporate justified post-drain BLAS cache clear in next meaningful source comparison',
    causal_status='32MiB is consistent with version-matched Hopper cuBLAS workspace, and oracle BLAS calls precede all timing; allocation origin not directly measured in this attempt')
print(json.dumps(report,indent=2))
