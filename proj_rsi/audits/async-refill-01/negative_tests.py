"""CPU synthetic analyzer tests. These fixtures are NEVER GPU async evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'candidates/reverse_tma_x_async_refill'),str(ROOT/'candidates/reverse_tma_x_async_trial')]
import native_x_async_support as support
spec=importlib.util.spec_from_file_location('async_analysis_under_test',HERE/'analysis.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def test():
    # Old parent timings are only arbitrary finite synthetic test numbers. No
    # resulting test document is stored as measured async result/analysis.json.
    raw=(ROOT/'audits/native-x-01/native-x-01.json').read_text()
    d=json.loads(raw.replace('"dp','"native'))
    d['status']='CPU_SYNTHETIC_NOT_HARDWARE'
    d['native_variants_storage_bits_gate']=d.pop('native_X_storage_bits_gate')
    d['GPU_pack_all65536_BF16_storage_bits'].pop('native')
    d['candidate_source_sha256']={'ds':'672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1','native':'ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377','new':'45fbc8b600c6054c43e725b6b5f33b9194b61f5b602c583a7b9bb825ab325170'}
    d['full_pipeline_in_timed_graph']={'ds':'FRESH_PACK_PLUS_MAIN_2_NODES','native':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP','new':'CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP'}
    d['attrs']['native']=copy.deepcopy(d['attrs']['new']);d['attrs']['new']['smem']=137248;d['attrs']['new']['max_dynamic_after']=137248
    for phase in d['timing']['strict_immediate_actual_timed_output_checks'].values():
        for row in phase['checks']:row['XP_current_recorded_fixture_exact']=row['graph']=='ds'
    for row in d['actual_timed_graph_checks']:row['packed_poison_overwritten_byte_exact']=row['graph']=='ds'
    d['fresh_workspace_and_output']['native']={'status':'PASS','workspace_bytes':0,'XP':'NONE'}
    d['profile']={'incumbent':[{'name':'SYNTHETIC_rq'}],'ds':[{'name':'rx_pack'},{'name':'rq_px_dp_stmx<'}],'native':[{'name':'rq_tma_x<'}],'new':[{'name':'rq_tma_x_async<'}]}
    rejected={}
    with tempfile.TemporaryDirectory(prefix='async-analyzer-cpu-') as td:
        base=Path(td);p=base/'CPU_SYNTHETIC_NOT_HARDWARE.json'
        d['actual_JIT_SASS']={}
        for label,folder in support.FOLDERS.items():
            text=(support.CANDIDATES/folder/'sass.txt').read_text();artifact=base/(label+'.sass');artifact.write_text(text)
            d['actual_JIT_SASS'][label]={'artifact':str(artifact),'sha256':hashlib.sha256(text.encode()).hexdigest(),**support.actual_binary_gate(label,text)}
        # Normalize tuples identically to runtime retained JSON.
        d=json.loads(json.dumps(d));p.write_text(json.dumps(d));A.analyze(p)
        # Fresh interpreter proves analyzer imports work without this test's
        # intentionally imported support module masking path resolution.
        code="import runpy; from pathlib import Path; ns=runpy.run_path("+repr(str(HERE/'analysis.py'))+"); ns['analyze'](Path("+repr(str(p))+"))"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True,timeout=20)
        def change(name,mutate):
            broken=copy.deepcopy(d);mutate(broken);p.write_text(json.dumps(broken))
            try:A.analyze(p)
            except (AssertionError,KeyError) as e:rejected[name]=type(e).__name__
            else:raise AssertionError('admitted synthetic corruption '+name)
        change('warm_fixture',lambda b:b['timing']['all_timed_and_warmup_calls'][0].update(input_seed=0))
        change('paired_order',lambda b:b['timing']['all_timed_and_warmup_calls'][167].update(label='wrong'))
        change('F1_raw',lambda b:b['timing']['F1_samples_us']['new'].__setitem__(0,0))
        change('F1_median',lambda b:b['timing']['F1_medians_us'].update(new=0))
        change('paired_ratio',lambda b:b['timing']['paired_new_median_ratio'].update(native=100))
        change('terminal_policy_fixture',lambda b:b['timing']['strict_immediate_actual_timed_output_checks']['F1']['checks'][0].update(input_seed=0))
        change('terminal_current_X',lambda b:b['timing']['strict_immediate_actual_timed_output_checks']['paired'].update(current_X_matches_last_actual_call_seed=0))
        change('native_fake_XP',lambda b:b['GPU_pack_all65536_BF16_storage_bits'].update(native='PASS'))
        change('mutation_duplicate',lambda b:b['actual_timed_graph_checks'].__setitem__(0,b['actual_timed_graph_checks'][1]))
        change('counter_or_immutable',lambda b:b['timing']['strict_immediate_actual_timed_output_checks']['F1'].update(W_packed_escape_immutable_counters_zero=False))
        change('missing_reset',lambda b:b.update(graph_cleanup='NOT_RESET'))
        change('wrong_source',lambda b:b['candidate_source_sha256'].update(new='wrong'))
        change('new_uses_parent_resources',lambda b:b['attrs']['new'].update(smem=137232))
        change('new_uses_parent_profile',lambda b:b['profile']['new'][0].update(name='rq_tma_x<'))
        change('SASS_hash_mismatch',lambda b:b['actual_JIT_SASS']['new'].update(sha256='wrong'))
        text=(base/'new.sass').read_text();(base/'new.sass').write_text(text.replace('LDSM.16.M88.4','LDS.INVALID',1))
        broken=copy.deepcopy(d);broken['actual_JIT_SASS']['new']['sha256']=hashlib.sha256((base/'new.sass').read_bytes()).hexdigest();p.write_text(json.dumps(broken))
        try:A.analyze(p)
        except AssertionError:rejected['mutated_SASS_with_matching_hash']='AssertionError'
        else:raise AssertionError('admitted changed async main')
    return {'status':'CPU_SYNTHETIC_ANALYZER_NEGATIVE_TESTS_PASS','fixture_status':'CPU_SYNTHETIC_NOT_HARDWARE','async_GPU_evidence_created':False,'GPU_calls':0,'fresh_interpreter_import_and_valid_synthetic_fixture':'PASS','negative_controls':rejected,'analysis_sha256':hashlib.sha256((HERE/'analysis.py').read_bytes()).hexdigest()}
if __name__=='__main__':
    r=test();(HERE/'cpu-negative-tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
