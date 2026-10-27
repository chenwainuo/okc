"""Read-only PW40 artifact/source/protocol verification. No compiler/context/prepare."""
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cpu_proof as G
from learning_proof import proof as lessons
HERE=Path(__file__).resolve().parent

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    source=G.source_proof();learning=lessons()
    candidate=G.streams((HERE/'sass.txt').read_text());control=G.streams((HERE/'control-sass.txt').read_text())
    assert len(candidate)==20 and len(control)==19
    for args,instructions in control.items():assert candidate[args]==instructions,args
    args=(128,6,1,0,0,0,1027)
    assert candidate[args]==G.streams((HERE.parent/'pw36_wring_rebase/sass.txt').read_text())[args]
    registers=G.resources((HERE/'device-resource.log').read_text())
    assert registers[args]==168 and all(((r+7)//8)*8*384<=65536 for r in registers.values())
    env=G.execute_pure(HERE.parent/'pw36_wring_rebase/protocol_proof.py',{'program','model','lifetime','require_X','require_all_warp_reads','proof'},{'random':__import__('random')})
    protocol=env['proof']();retained=json.loads((HERE/'protocol-proof.json').read_text());assert json.dumps(protocol,sort_keys=True)==json.dumps(retained,sort_keys=True)
    print(json.dumps(dict(state='PASS_READONLY_FROZEN_PW40_VERIFY',source_sha256=source['full_candidate_sha256'],existing_SASS_controls=19,
                         validated_split_main='EXACT',protocol_states=protocol['total_reachable_states'],lessons=[r['ref'] for r in learning['applied']],GPU='UNRUN'),indent=2))
if __name__=='__main__':main()
