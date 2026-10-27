"""Exercise lab confirmation gates against the real extracted production helper."""
import copy
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def run():
    driver=load(ROOT/'proj_rsi/tools/pw39_stack_lab.py','pw39_lab_gate_cpu')
    proof=load(ROOT/'proj_rsi/candidates/pw39/cpu_proof.py','pw39_actual_helper_cpu')
    cases={
        'stable_A':proof.harness([100,97,98,100,98,97,100]),
        'stable_B':proof.harness([100,90,96,100,96,98,100]),
        'both_brackets_reject':proof.harness([100,80,103,100,103,102,100]),
        'single':proof.harness([100,97,100,97,100],cands=[(90,'A',{'id':'A'})]),
        'no_screen_winner':proof.harness([],screen=('incumbent',None,100)),
        'global_partial':proof.harness([100,97,98,100,98,97,100],duration=.5),
        'prepare_partial':proof.harness([100,97,98,100,98,97,100],wall=209.5,duration=.2),
        'timing_error':proof.harness([100,97,98,100,98,97,100],throw_at=2)}
    admitted={}
    for name,result in cases.items():
        record=result['record']
        entry={'layer':record['layer'],'m':record['m'],'screen_pick':record['screen_pick'],'incumbent':record['incumbent'],
               'returned_pick':result['selected'][0],'returned_us':result['selected'][2],
               'production_record':record,'raw_timing_blocks':[{'it':5,'raw_replay_us':[block['median_us']]*5} for block in record['blocks']]}
        audit={'helper_present':True,'invocations':[entry],'state_audit':{'records':[record]}}
        driver.confirmation_gate(audit);admitted[name]=audit
    negatives={}
    for name in ('wrong_block_raw_median','wrong_returned_latency','wrong_returned_winner','RQ_nonzero_status'):
        audit=copy.deepcopy(admitted['stable_A']);entry=audit['invocations'][0]
        if name=='wrong_block_raw_median':entry['raw_timing_blocks'][0]['raw_replay_us']=[999]*5
        elif name=='wrong_returned_latency':entry['returned_us']+=1
        elif name=='wrong_returned_winner':entry['returned_pick']='rq:B'
        else:entry['RQ_host_launch_status']=[{'code':-6}]
        try:driver.confirmation_gate(audit)
        except AssertionError:negatives[name]='REJECTED'
        else:raise AssertionError('negative accepted '+name)
    return {'state':'CPU_ACTUAL_HELPER_TO_LAB_GATE_PASS_GPU_UNRUN','positive_cases':list(admitted),'negative_cases':negatives,'CUDA_calls':0,'production_imports':0}
if __name__=='__main__':print(json.dumps(run(),indent=2))
