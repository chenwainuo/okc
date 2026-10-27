"""Exercise identities, all canonical binary gates, and adversarial timed-output gate CPU only."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reverse_tma_x_async_refill'))
import native_x_async_support as support
import timed_output_gate_proof

def proof():
    identities=support.verify_frozen()
    modules={label:support.load(folder,'prototype',label) for label,folder in support.FOLDERS.items()}
    for label,mod in modules.items():
        assert mod.source()==(support.CANDIDATES/support.FOLDERS[label]/'extension_source.cu').read_text()
    result={'status':'CPU_LAB_GATES_PASS_GPU_UNRUN','identities':identities,
            'canonical_binary_gates':{label:support.actual_binary_gate(label,(support.CANDIDATES/folder/'sass.txt').read_text()) for label,folder in support.FOLDERS.items()},
            'timed_output_adversarial_gate':timed_output_gate_proof.proof(),
            'GPU':'UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE',
            'raw_native_X_bit_mover':'UNPROBED_NOT_AN_XP_PACK_GATE',
            'limits':'CPU proofs bind frozen sources and local cubin; actual JIT and all CUDA runtime gates remain required'}
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('canonical_binary_gates','timed_output_adversarial_gate')},indent=2))
if __name__=='__main__':proof()
