"""Host-only mapped-local grid bound; exact frozen v2 codec/device engine.
Only for original aligner inputs whose IDs map into -1,0..31. The original
adapter remains available for generic/defensive route producers.
"""
import importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'ss_pipeline_v2/adapter.py'
spec=importlib.util.spec_from_file_location('_ss_mapped_grid_frozen_v2_engine',BASE)
_engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_engine;spec.loader.exec_module(_engine)
Bank=_engine.Bank
prepare=_engine.prepare
module=_engine.module

def mapped_local_row_bound(slots,block_m,capacity,local_experts=32):
    assert slots>0 and block_m>=8 and block_m%8==0 and capacity>=block_m and local_experts==32
    # counts include one -1 bucket and32 local buckets. Actual aligner post is
    # sum ceil(count/bm)*bm, hence a multiple ofbm <= slots+33*(bm-1).
    padded_upper=(slots+(local_experts+1)*(block_m-1))//block_m*block_m
    return min(capacity,padded_upper)

def project(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor):
    assert bank.packed.shape[0]==32,'mapped-local32 aligner contract only'
    limit=mapped_local_row_bound(out.shape[0],block_m,sorted_ids.numel())
    # Metadata-only prefix view, same storage/pointer. Every actual post row is
    # still exposed; v2 run() derives launchgrid from this smaller capacity.
    # No host read of post, device code, zeroing, routing, or math is changed.
    return _engine.project(x,bank,sorted_ids[:limit],experts,post,block_m,out,gather_divisor)
