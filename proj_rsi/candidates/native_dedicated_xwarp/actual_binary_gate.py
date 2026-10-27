"""Fixed-candidate CPU gate for a future root-collected actual JIT SASS dump."""
import hashlib
from pathlib import Path
import re
import liveness
import proofs
HERE=Path(__file__).resolve().parent
TOKEN='rq_dedicated_x'
def instructions(sass):
    tables=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if TOKEN in t.splitlines()[0]]
    assert len(tables)==1,('dedicated-X kernel count',len(tables))
    return liveness.parse(tables[0])
def check(sass):
    canonical=(HERE/'sass.txt').read_text()
    assert instructions(sass)==instructions(canonical),'ACTUAL_JIT_MAIN_SEMANTIC_DRIFT'
    return dict(status='PASS_CANONICAL_AND_PHYSICAL_DEDICATED_X',canonical_sass_sha256=hashlib.sha256(canonical.encode()).hexdigest(),physical=proofs.physical(sass),
                limit='CPU gate only; actual CUDA attrs, numerics, source-bound current-input graphs and cleanup still mandatory')
