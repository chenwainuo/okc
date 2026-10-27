"""Focused CPU physical layout/repair ring checks. No CUDA context or JIT."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from pack import pack, unpack

D=Path(__file__).resolve().parent
rng=np.random.default_rng(4551)
fr=(rng.integers(0,256,(4,8,4,32,8),dtype=np.uint16) |
    (rng.integers(57,61,(4,8,4,32,8),dtype=np.uint16)<<8) |
    (rng.integers(0,2,(4,8,4,32,8),dtype=np.uint16)<<15))
flat=fr.reshape(-1)
flat[::97]=(flat[::97]&0x80ff)|(120<<8)
geo=(2,2,4,1,1,0)
pk=pack(fr,57,geo)
assert pk is not None and np.array_equal(unpack(pk),fr)
assert all(a.flags.c_contiguous for a in (pk['low'],pk['header'],pk['desc'],pk['records']))
assert pk['total_bytes']==pk['low'].nbytes+pk['header'].nbytes+pk['desc'].nbytes+pk['records'].nbytes
# Default capacity rejects rather than truncating; nonfinite remains invalid.
outlier=fr.copy();outlier[0,0]=120<<8
assert pack(outlier,57,geo) is None
nonfinite=fr.copy();nonfinite[0,0,0,0,0]=0x7f80
assert pack(nonfinite,57,geo) is None
# Mirror cp.async group completion and per-slot ownership, including both tails.
checks=0
for stages,chunks in ((1,8),(2,16)):
    issued=list(range(stages)); landed=-1; owners=list(range(stages))
    for u in range(chunks):
        allowed=0 if stages==1 or u+1==chunks else 1
        while len(issued)>allowed:
            landed=issued.pop(0)
        assert landed>=u and owners[u%stages]==u
        # All four steps finish before the slot is reused.
        for q in range(4):
            assert owners[u%stages]==u
            checks+=1
        if u+stages<chunks:
            owners[u%stages]=u+stages
            issued.append(u+stages)
    assert not issued
result=dict(status='CPU_PHYSICAL_PACK_AND_RING_PASS_GPU_UNRUN',physical_roundtrip_values=int(fr.size),
            record_count=pk['repairs'],bytes=pk['total_bytes'],incumbent_bytes=pk['incumbent_bytes'],
            protected_step_generations=checks,overflow_and_nonfinite_rejections=True,
            actual_CUDA_tail_admission='UNKNOWN_PENDING_STATS')
print(json.dumps(result,indent=2))
