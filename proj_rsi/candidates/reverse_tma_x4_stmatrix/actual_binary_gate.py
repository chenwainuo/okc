"""CPU-only gate component. Accepts SASS dumped by a separate root GPU driver."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import liveness
import proofs
HERE=Path(__file__).resolve().parent
TOKEN='rq_tma_x4_stmx'
def instructions(sass):
    tables=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if TOKEN in t.splitlines()[0]]
    assert len(tables)==1
    return liveness.parse(tables[0])
def check(sass):
    canonical=(HERE/'sass.txt').read_text()
    assert instructions(sass)==instructions(canonical),'ACTUAL_JIT_MAIN_SEMANTIC_DRIFT'
    return dict(status='PASS_ACTUAL_JIT_MAIN_AND_PHYSICAL_GATES',
                canonical_sass_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                instructions=len(instructions(sass)),native_physical=proofs.physical(sass),
                producer_store=liveness.store_form(sass,TOKEN,True,dual=True),
                limit='Does not replace GPU resource/occupancy/numerical/graph/mutation gates')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sass',type=Path,required=True);args=ap.parse_args()
    print(json.dumps(check(args.sass.read_text()),indent=2))
