"""Read-only frozen PW41 verification. No compiler/CUDA/prepare/API calls."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cpu_proof import proof
from lab_plan import proof as lab_plan
HERE=Path(__file__).resolve().parent

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    before={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in manifest['files']}
    assert before==manifest['files'],'frozen file hash mismatch'
    for name,sha in manifest['external_inputs'].items():
        assert hashlib.sha256((HERE.parents[2]/name).read_bytes()).hexdigest()==sha,('external input changed',name)
    result=proof();retained=json.loads((HERE/'cpu-proof.json').read_text())
    assert json.dumps(result,sort_keys=True)==json.dumps(retained,sort_keys=True)
    assert json.dumps(lab_plan(),sort_keys=True)==json.dumps(json.loads((HERE/'lab-plan.json').read_text()),sort_keys=True)
    registry=json.loads((HERE/'learning-proof.json').read_text())
    for row in registry['applied']:
        assert row['ref'] in manifest['applied_lesson_refs'] and len(row['record_sha256'])==64
    assert before=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in manifest['files']}
    print(json.dumps({'status':'PASS_READONLY_FROZEN_PW41_CPU','files':len(before),'source_sha256':manifest['source_sha256'],'old_current_GV_controls':95,'optional_new_types':4,'scored_bake_cells':5,'protocol_states':result['protocol']['total_states'],'GPU':'UNRUN','GPU_wrapper':'NOT_IMPLEMENTED_OR_ADMITTED; CPU_TEST_PLAN_ONLY'},indent=2))
if __name__=='__main__':main()
