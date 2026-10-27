"""Exercise identities, all canonical binary gates, and adversarial timed-output gate CPU only."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reverse_tma_x4'))
import native_x4_support as support
import timed_output_gate_proof

def proof():
    identities=support.verify_frozen()
    modules={label:support.load(folder,'prototype',label) for label,folder in support.FOLDERS.items()}
    for label,mod in modules.items():
        assert mod.source()==(support.CANDIDATES/support.FOLDERS[label]/'extension_source.cu').read_text()
    rejected={}
    for label,wrong in [('new','native'),('native','new'),('ds','native')]:
        text=(support.CANDIDATES/support.FOLDERS[wrong]/'sass.txt').read_text()
        try:support.actual_binary_gate(label,text)
        except AssertionError as e:rejected[label+'_given_'+wrong]=repr(e)
        else:raise AssertionError('Wrong label admitted '+label+'/'+wrong)
    for label,folder in support.FOLDERS.items():
        text=(support.CANDIDATES/folder/'sass.txt').read_text()
        assert 'HGMMA.' in text
        damaged=text.replace('HGMMA.','BROKEN_HGMMA.',1)
        try:support.actual_binary_gate(label,damaged)
        except AssertionError as e:rejected[label+'_canonical_instruction_tamper']=repr(e)
        else:raise AssertionError('Canonical instruction tamper admitted '+label)
    lab=(HERE.parents[1]/'tools/native_x4_lab.py').read_text()
    assert '170016 if label=="new" else 137232 if label=="native" else 103600' in lab
    assert 'attrs["occupancy_after_optin"]==1' in lab and 'info(6)==512' in lab
    assert 'for label in ("ds",)' in lab
    assert 'source_shas=' in lab and 'UNSUPPORTED_INCUMBENT_FALLBACK_NO_ADMISSION' in lab
    assert 'range(21)' in lab and 'range(32)' in lab and 'range(5)' in lab
    terminal=timed_output_gate_proof.proof()
    (HERE/'timed-output-gate-proof.json').write_text(json.dumps(terminal,indent=2)+'\n')
    result={'status':'CPU_LAB_GATES_PASS_GPU_UNRUN','identities':identities,
            'canonical_binary_gates':{label:support.actual_binary_gate(label,(support.CANDIDATES/folder/'sass.txt').read_text()) for label,folder in support.FOLDERS.items()},
            'timed_output_adversarial_gate':terminal,'wrong_label_and_instruction_rejections':rejected,
            'predeclared_call_counts':dict(F1=7*21,paired=32*4,warmup=4*5,all_raw=7*21+32*4+4*5,terminal_oracles=8,mutations=20),
            'GPU':'UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE',
            'raw_native_X_bit_mover':'UNPROBED_NOT_AN_XP_PACK_GATE',
            'limits':'CPU proofs bind frozen sources and local cubin; actual JIT and all CUDA runtime gates remain required'}
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('canonical_binary_gates','timed_output_adversarial_gate')},indent=2))
if __name__=='__main__':proof()
