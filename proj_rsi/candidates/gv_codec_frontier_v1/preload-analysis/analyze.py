"""One frozen C11 three-arm collection audit. CPU only; root owns disposition.

python analyze.py --collection <collection-directory> --output <new-report.json>
No queue, source, raw artifact, GPU or remote mutation.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

D=Path(__file__).resolve().parent
ROOT=D.parents[3]
TRIAL=D.with_name('native_v3_preloadedmeta_highpatch')


def protocol():
    chosen=[]
    for node in ast.parse((TRIAL/'driver.py').read_text()).body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ('LABELS','CELLS')for t in node.targets):chosen.append(node)
        if isinstance(node,ast.FunctionDef)and node.name in ('schedule','basis_columns'):chosen.append(node)
    ns={'itertools':itertools}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'<frozen-three-arm-protocol>','exec'),ns)
    return ns


def run(collection):
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    inventory=json.loads((collection/'inventory.json').read_text())
    for rel,pin in inventory['files'].items():check('artifact '+rel,sha(collection/rel)==pin)
    for rel,pin in inventory['dispatch']['source_pins'].items():check('frozen source '+rel,sha(ROOT/rel)==pin)
    result_paths=list(collection.rglob('result.json'));check('one result',len(result_paths)==1)
    result_path=result_paths[0];folder=result_path.parent
    result=json.loads(result_path.read_text())
    # Preserve setup/cleanup failures instead of treating missing samples as zero.
    if 'error'in result or result.get('state')not in ('TWO_FIXED_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED','INVALID_CLEANUP')or set(result.get('fixed_cells',{}))!={'1','16'} or any(len(row.get('raw',[]))!=131 for row in result.get('fixed_cells',{}).values()):
        return dict(disposition=result.get('state','INVALID_SETUP'),timing_admission='INCOMPLETE_NO_PERFORMANCE_ADMISSION',
            observed_result=result,external_cleanup=inventory.get('cleanup_observation'),checks=checks,checks_passed=len(checks),
            result_sha256=sha(result_path),next_action='Diagnose actual setup/correctness failure before timing conclusions; retain failed source/result')
    ns=protocol();labels=ns['LABELS'];schedule=ns['schedule']()
    check('three exact arms',labels==('control','v2','frontier'))
    check('two exact shapes',[(row[0],row[1])for row in ns['CELLS']]==[('pooler',1),('unpooler',16)])
    check('live current and v2 identity',result['actual_control_native_identity']=='EXACT_C12_AND_V2_PC_PREDICATE_ENCODING')
    check('no prepare or profiler',result['profiler_calls']==result['full_prepare_calls']==0)
    check('actual CUDA source',result['build']['CUDA_sha256']==sha(folder/'actual.cu'))
    check('actual SO',result['build']['SO_sha256']==sha(folder/'actual.so'))
    parser_nodes=[n for n in ast.parse((TRIAL/'verify_native.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='streams']
    parser={'re':re};exec(compile(ast.Module(body=parser_nodes,type_ignores=[]),'<frozen-stream-parser>','exec'),parser)
    actual=parser['streams']((folder/'actual.sass').read_text());expected=parser['streams']((TRIAL/'native.sass').read_text())
    check('six expected kernels',set(actual)==set(expected)and len(actual)==6)
    for symbol,rows in expected.items():
        if 'preload'not in symbol:check('native matched '+symbol,actual[symbol]==rows)
    raw=[json.loads(line)for line in (folder/'raw.jsonl').read_text().splitlines()]
    check('262 raw rows',len(raw)==262)
    stats=json.loads((D.parent/'actual-stats-proof.json').read_text())['layers']
    cells={}
    for layer,m,n,k,tpl,capacity in ns['CELLS']:
        entry=result['fixed_cells'][str(m)];rows=entry['raw'];config=entry['config'];attrs=result['raw_attributes'][layer]
        check(layer+' config',config==dict(layer=layer,m=m,n=n,k=k))
        check(layer+' exact131schedule',all({a:b for a,b in row.items()if a!='us'}==want for row,want in zip(rows,schedule)))
        check(layer+' rawJSONL',[{a:b for a,b in row.items()if a!='m'}for row in raw if row['m']==m]==rows)
        check(layer+' finite positive timings',all(math.isfinite(row['us'])and row['us']>0 for row in rows))
        check(layer+' full exact pack',entry['pack']['actual_W_bit_reconstruction']=='EXACT_BOTH_FORMATS')
        check(layer+' actual fixture',entry['pack']['actual_fixture_sha']==stats[layer]['weight_sha256'])
        check(layer+' capacity geometry bytes',entry['pack']['capacity']==capacity and entry['pack']['geo']==list(tpl[3:9])and entry['pack']['C11_bytes']==stats[layer]['C11_total_bytes']and entry['pack']['C12_bytes']==stats[layer]['C12_total_bytes'])
        regs=[64,64,64]if layer=='pooler'else[110,108,96]
        dynamic=[17408,20480,28160]if layer=='pooler'else[5120,6144,8000]
        for index,arm in enumerate(labels):
            reg,local,static,threads,smem,occupancy=attrs[arm]
            check(layer+arm+' resources',reg==regs[index]and local==static==0 and smem==dynamic[index]and threads==(1024 if layer=='pooler' else 128)and occupancy>=attrs['control'][-1]>=1)
            validation=entry['validation'][arm]
            check(layer+arm+' basis',validation['basis']=='EXACT'and validation['basis_columns']==ns['basis_columns'](k))
            check(layer+arm+' three validation modes',[r['kind']for r in validation['task']]==['normal','outliers','zero'])
            for site in validation['task']:
                oracle=site['oracle'];check(layer+arm+site['kind']+' oracle',oracle['current_X']=='EXACT'and oracle['FP64_L2']<=.004 and oracle['worst_row']<=.02)
        for phase,end in (('F1',59),('paired',131)):
            sites=entry['terminals'][phase]
            check(layer+phase+' three own terminals',len(sites)==3 and {s['arm']for s in sites}==set(labels))
            for site in sites:
                seed=next(row['seed']for row in reversed(rows[:end])if row['arm']==site['arm']and row['mode']!='warmup');oracle=site['oracle']
                check(layer+phase+site['arm']+' own output',site['seed']==seed and site['before_any_replay_or_mutation']and oracle['current_X']=='EXACT'and oracle['FP64_L2']<=.004 and oracle['worst_row']<=.02)
        kinds=('zero','normal','outliers','boundary_probes','fresh_normal')
        check(layer+'15unique mutations',len(entry['mutations'])==15 and {(s['arm'],s['kind'])for s in entry['mutations']}==set(itertools.product(labels,kinds)))
        for site in entry['mutations']:
            oracle=site['oracle'];check(layer+site['arm']+site['kind']+' retained graph',site['NaN_output_overwritten']and oracle['current_X']=='EXACT'and oracle['FP64_L2']<=.004 and oracle['worst_row']<=.02)
        check(layer+' graph drain/reset',entry['cleanup']=='RESET_AFTER_DRAIN')
        f1=[statistics.median(r['us']for r in rows[15+11*i:26+11*i])for i in range(4)]
        paired=[]
        for i in range(24):
            trip=rows[59+3*i:62+3*i];timings={r['arm']:r['us']for r in trip}
            paired.append(dict(pair=i,seed=trip[0]['seed'],order=[r['arm']for r in trip],us=timings,
                control_minus_v2=timings['control']-timings['v2'],control_minus_new=timings['control']-timings['frontier'],v2_minus_new=timings['v2']-timings['frontier']))
        delta=statistics.median(p['control_minus_new']for p in paired)
        fdelta=(f1[0]+f1[3])/2-f1[2]
        check(layer+' reported contrasts',delta==entry['contrasts']['paired_median_control_minus_new_us']and fdelta==entry['contrasts']['F1_endpoint_average_control_minus_new_us']and statistics.median(p['v2_minus_new']for p in paired)==entry['contrasts']['paired_median_v2_minus_new_us'])
        old_bytes=entry['pack']['C12_bytes'];new_bytes=entry['pack']['C11_bytes'];units=n*k//1024;groups=units//tpl[5]
        cells[layer]=dict(F1_medians_us=dict(zip(('control_before','v2','new','control_after'),f1)),
            paired_medians_us={arm:statistics.median(p['us'][arm]for p in paired)for arm in labels},
            paired_control_minus_new_us=delta,paired_control_minus_v2_us=statistics.median(p['control_minus_v2']for p in paired),
            paired_v2_minus_new_us=statistics.median(p['v2_minus_new']for p in paired),
            F1_endpoint_average_control_minus_new_us=fdelta,F1_conservative_min_control_minus_new_us=min(f1[0],f1[3])-f1[2],
            F1_endpoint_average_control_minus_v2_us=(f1[0]+f1[3])/2-f1[1],
            wins_vs_current=sum(p['control_minus_new']>0 for p in paired),wins_vs_v2=sum(p['v2_minus_new']>0 for p in paired),
            signed_triples=paired,actual_resources=attrs,C12_unique_bytes=old_bytes,C11_unique_bytes=new_bytes,
            V2_duplicate_boundary_word_bytes=8*units-4,new_duplicate_boundary_word_bytes=4*(groups-1),
            unique_byte_fraction_saved=1-new_bytes/old_bytes,
            new_source_word_ledger_fraction_saved=1-(new_bytes+4*(groups-1))/old_bytes,
            measured_HBM_bytes='NOT_COLLECTED',scope='Direct fixed-cell same-run comparison; no official score or policy gain')
    cleanup=result.get('post_scope_cleanup',{});external=inventory['cleanup_observation']
    evidence=cleanup.get('blas_cache_entries_workspace_bytes_entries_after')
    clean=(result['state']=='TWO_FIXED_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED'and cleanup.get('allocated')==0 and cleanup.get('clear_error')is None and cleanup.get('clear_only_after_drain_and_scope_release')is True and cleanup.get('all_run_aliases_released')is True and isinstance(evidence,list)and len(evidence)==3 and evidence[2]==0 and external['terminal_exit']==0 and not external['cleanup_failed']and not external['container_exists']and not external['compute_pids']and external['gpu_memory_mib']==0)
    disposition='INVALID_CLEANUP'
    if clean:
        effects=[r['paired_control_minus_new_us']for r in cells.values()]
        disposition='VALID_GAIN'if any(x>0 for x in effects)else'VALID_REGRESSION'if any(x<0 for x in effects)else'VALID_NO_GAIN'
    before=cleanup.get('before_blas_clear',{})
    cache_accounting_match=(isinstance(evidence,list)and len(evidence)==3 and evidence[0]>0 and before.get('allocated')==evidence[0]*evidence[1]and cleanup.get('allocated')==0 and evidence[2]==0)
    return dict(disposition=disposition,timing_admission='VALID_FIXED_CELLS_ONLY'if clean else'INTERNAL_OR_EXTERNAL_CLEANUP_FAILURE_NO_PERFORMANCE_ADMISSION',
        checks_passed=len(checks),checks=checks,cells=cells,internal_cleanup=cleanup,external_cleanup=external,
        cache_clear_counts_match_freed_remaining_bytes=cache_accounting_match,hardware=result['hardware'],
        historical_v2_scope='Attempt01 remains INVALID_CLEANUP; current same-run v2 arm has separate validity. No retroactive historical admission.',
        result_sha256=sha(result_path),inventory_sha256=sha(collection/'inventory.json'),
        next_action='Root independently validates complete receipts; retain all signs. Gain only versus v2 does not establish gain versus incumbent. Next causal question depends on the actual current comparison and resource/cache evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists(),'never overwrite a prior analysis'
    report=run(args.collection)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(disposition=report['disposition'],checks_passed=report['checks_passed'],output=str(args.output))))
