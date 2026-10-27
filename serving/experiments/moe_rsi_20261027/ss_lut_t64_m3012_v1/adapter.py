"""Isolated CPU-frozen prototype; import does not load a CUDA extension."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import torch
@dataclass(frozen=True)
class Bank:
    packed:torch.Tensor
    scales:torch.Tensor
    globals:torch.Tensor
def pack_weight(raw_weight):
    """Lossless uint8[E,N,K/2] -> int32[E,N/64,K/16,128]. CPU or GPU.
    Nibbles in a lane word: q0,q2,q1,q3,q4,q6,q5,q7 (LSB first).
    """
    e,n,kp=raw_weight.shape
    assert raw_weight.dtype==torch.uint8 and n%64==0 and kp%8==0
    # [E,nt,c,e_row,b,Q,f,a] -> [E,nt,Q,c,b,a,f,e_row].
    t=raw_weight.reshape(e,n//64,4,2,8,kp//8,2,4).permute(0,1,5,2,4,7,6,3)
    row0,row1=t[...,0],t[...,1]
    lo=(row0&15)|((row1&15)<<4)
    hi=(row0>>4)|(row1&240)
    result=torch.stack((lo,hi),dim=-1).contiguous()
    return result.reshape(e,n//64,kp//8,128,4).view(torch.int32).reshape(e,n//64,kp//8,128)

def pack_scale(processed):
    e,n,q=processed.shape
    return processed.reshape(e,n//64,64,q).permute(0,1,3,2).contiguous()

def prepare(raw_weight,raw_scale,raw_global):
    """Local original uint8[E,N,K/2], E4M3[E,N,K/16], float[E].

    Losslessly repacks original bytes, avoiding inverse Marlin weight permutation.
    Scales preserve all raw E4M3 bytes; global is BF16 rawgs, without exponent bias.
    """
    assert raw_weight.dtype==torch.uint8 and raw_weight.ndim==3
    e,n,kp=raw_weight.shape
    assert kp==2048 and n%64==0 and tuple(raw_scale.shape)==(e,n,256)
    assert raw_scale.dtype==torch.float8_e4m3fn and tuple(raw_global.shape)==(e,)
    assert raw_weight.is_cuda and raw_scale.device==raw_weight.device==raw_global.device
    assert torch.cuda.get_device_capability(raw_weight.device)==(9,0)
    # Preserve raw FP8 bytes, including signedzero/subnormals/NaNs. No lossy
    # HALF→byte Marlin scale format and no compensating exponent bias.
    processed=raw_scale.view(torch.uint8).contiguous()
    global_processed=raw_global.to(torch.bfloat16).contiguous()
    return Bank(pack_weight(raw_weight),pack_scale(processed),global_processed)
@lru_cache(None)
def module():
    from torch.utils.cpp_extension import load_inline
    kernel=Path(__file__).with_name('kernel.cuh').read_text()
    host=r"""
#include <torch/extension.h>
void run(torch::Tensor p,torch::Tensor sf,torch::Tensor x,torch::Tensor gs,torch::Tensor sorted,torch::Tensor expert,torch::Tensor post,torch::Tensor out,int64_t bm,int64_t div,int64_t stream_value){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&out.is_contiguous(),"CUDA contiguous tensors required");
 TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&out.scalar_type()==torch::kBFloat16,"BF16 boundary");
 TORCH_CHECK(p.scalar_type()==torch::kInt32&&sf.scalar_type()==torch::kUInt8&&gs.scalar_type()==torch::kBFloat16,"bank dtype");
 TORCH_CHECK(sorted.scalar_type()==torch::kInt32&&expert.scalar_type()==torch::kInt32&&post.scalar_type()==torch::kInt32,"route dtype");
 for(auto t:{p,sf,gs,sorted,expert,post,out})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous(),"same-device contiguous tensors");
 int E=p.size(0),N=p.size(1)*64,K=p.size(2)*16,rows=x.size(0),slots=out.size(0),cap=sorted.numel(),ec=expert.numel();
 TORCH_CHECK(p.dim()==4&&p.size(3)==128&&sf.dim()==4&&sf.size(0)==E&&sf.size(1)*64==N&&sf.size(2)*16==K&&sf.size(3)==64,"bank shapes");
 TORCH_CHECK(K==4096&&x.dim()==2&&x.size(1)==K&&out.dim()==2&&out.size(1)==N&&gs.numel()==E&&post.numel()==1,"geometry");
 TORCH_CHECK((div==1||div==8)&&slots==rows*div&&bm>=8&&bm%8==0&&cap>=bm&&(cap+bm-1)/bm<=ec,"route capacities");
 TORCH_CHECK(E==32&&bm==64&&((div==8&&rows==3012&&N==8192)||(div==1&&rows==24096&&N==4096)),"bounded M3012 EP8 T64 geometry");
 TORCH_CHECK(x.data_ptr()!=out.data_ptr(),"input/output alias");
 auto stream=reinterpret_cast<cudaStream_t>(stream_value);
 auto P=(const uint32_t*)p.data_ptr();auto S=(const uint8_t*)sf.data_ptr();auto X=(const moe_ss_lut_t64_m3012_v1::B*)x.data_ptr();auto G=(const moe_ss_lut_t64_m3012_v1::B*)gs.data_ptr();
 auto R=(const int32_t*)sorted.data_ptr();auto EID=(const int32_t*)expert.data_ptr();auto Z=(const int32_t*)post.data_ptr();auto Y=(moe_ss_lut_t64_m3012_v1::B*)out.data_ptr();
 #define LAUNCH(T,D) moe_ss_lut_t64_m3012_v1::launch<T,D>(P,S,X,G,R,EID,Z,Y,K,N,E,rows,slots,cap,ec,bm,stream)
 #define PICK(D) LAUNCH(64,D); // Bounded original M3012, exact blockM64 only.
 if(div==8){PICK(8)}else{PICK(1)}
 TORCH_CHECK(cudaGetLastError()==cudaSuccess,"SS projection launch failed");
}
"""
    return load_inline('moe_ss_lut_t64_m3012_v1_'+hashlib.sha256((kernel+host).encode()).hexdigest()[:16],cpp_sources='void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=kernel+host,functions=['run'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--expt-relaxed-constexpr'],verbose=True)

def project(x,bank,sorted_ids,experts,post,block_m,out,gather_divisor):
    assert block_m==64 and gather_divisor in (1,8)
    assert (gather_divisor,x.shape[0],bank.packed.shape[1]*64) in ((8,3012,8192),(1,24096,4096))
    assert x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
    assert torch.cuda.get_device_capability(x.device)==(9,0)
    assert out.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
    assert out.shape==(x.shape[0]*gather_divisor,bank.packed.shape[1]*64)
    out.zero_() # Retain empty/nonlocal exact-zero contract; included in graph and timing.
    module().run(bank.packed,bank.scales,x,bank.globals,sorted_ids,experts,post,out,block_m,gather_divisor,torch.cuda.current_stream(x.device).cuda_stream)
    return out
