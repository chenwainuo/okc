"""Isolated correctness-first down-projection candidate. NOT serving dispatch.

Call prepare before benchmark/capture on original checkpoint-layout local weights.
Call project with existing moe_align_block_size metadata and expanded cache2.
The baseline remains unchanged; this module does not monkeypatch anything.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import torch

@dataclass(frozen=True)
class Bank:
    packed: torch.Tensor
    scales: torch.Tensor
    globals: torch.Tensor
    grid_limit: int

@lru_cache(None)
def module():
    from sglang.kernels.jit.utils import load_jit
    import flashinfer
    import sglang
    source=Path(__file__).with_name('routed_down.cuh').resolve()
    cutlass=Path(flashinfer.__file__).resolve().parent/'data/cutlass/include'
    return load_jit('rsi_wgmma_persistent_down',hashlib.sha256(source.read_bytes()).hexdigest()[:16],
        cuda_files=[str(source)],cuda_wrappers=[('run','RoutedDownKernel::run')],
        extra_include_paths=[str(cutlass),str(Path(sglang.__file__).resolve().parent/"kernels/jit/csrc")],extra_cuda_cflags=['-O3'])

def prepare(raw_weight,raw_scale,raw_global):
    """Local original uint8[E,N,K/2], E4M3[E,N,K/16], float[E].

    Retains original packed bytes, avoiding inverse Marlin weight permutation.
    Scales use exact incumbent half/bit transform without layout permutation.
    """
    from sglang.srt.layers.quantization.marlin_utils_fp4 import nvfp4_marlin_process_global_scale
    assert raw_weight.dtype==torch.uint8 and raw_weight.ndim==3
    e,n,kp=raw_weight.shape
    assert kp==2048 and n%64==0 and tuple(raw_scale.shape)==(e,n,256)
    assert raw_scale.dtype==torch.float8_e4m3fn and tuple(raw_global.shape)==(e,)
    assert raw_weight.is_cuda and raw_scale.device==raw_weight.device==raw_global.device
    assert torch.cuda.get_device_capability(raw_weight.device)==(9,0)
    # Same steps as nvfp4_marlin_process_scales before its physical permutation.
    # The return needs byte upper8 of int16 shift1, not a numeric E4M3 cast.
    half=(raw_scale.to(torch.float16)*128).view(torch.int16)
    processed=(((half.to(torch.int32)<<1)>>8)&255).to(torch.uint8).contiguous()
    global_processed=nvfp4_marlin_process_global_scale(raw_global.to(torch.bfloat16)).contiguous()
    return Bank(raw_weight.contiguous(),processed,global_processed,
                2*torch.cuda.get_device_properties(raw_weight.device).multi_processor_count)

def project(expanded_input,bank,sorted_token_ids,expert_ids,num_post,block_m,out=None):
    """Expanded slot input/output, top_k=1; no norm or router weighting here."""
    assert expanded_input.dtype==torch.bfloat16 and expanded_input.is_contiguous()
    shape=(expanded_input.shape[0],bank.packed.shape[1])
    if out is None:
        out=torch.empty(shape,dtype=expanded_input.dtype,device=expanded_input.device)
    assert tuple(out.shape)==shape and out.dtype==torch.bfloat16 and out.is_contiguous()
    assert out.untyped_storage().data_ptr()!=expanded_input.untyped_storage().data_ptr(), 'projection input/output must not alias'
    out.zero_()  # explicit graph-captured work; do not hide this from timing
    module().run(bank.packed,bank.scales,expanded_input,bank.globals,
                 sorted_token_ids,expert_ids,num_post,out,block_m,bank.grid_limit)
    return out
