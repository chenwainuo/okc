"""Frozen dual producers, fresh packed-X, two K64 stages per8-MMA group. No CUDA calls at import."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HELPER=HERE.parent/"rs32/submission.py"
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=="a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad"
spec=importlib.util.spec_from_file_location("reverse_group8_rs32_helpers",HELPER)
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
VARIANT=(128,8,1,0,0,0,20)


def source():
    old=S.RQCU
    a=old.index("template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>\nstruct RqC")
    b=old.index("typedef CUresult (*RqEnc)",a)
    cu=old[:a]+(HERE/"kernel_body.cuh").read_text()+"\n"+old[b:]
    cu=cu.replace("const uint8_t* W;","const __nv_bfloat16* X;\n  const uint32_t* XP;\n  const uint8_t* W;",1)
    cu=cu.replace("p.W = (const uint8_t*)W.data_ptr();","p.X = (const __nv_bfloat16*)x.data_ptr(); p.XP = (const uint32_t*)xp.data_ptr(); p.W = (const uint8_t*)W.data_ptr();",1)
    cu=cu.replace("if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;",
                  "if(p.m!=128 || p.N!=8192 || p.K!=4096 || p.KC!=64 || KS!=1) return -4;")
    cu=cu.replace("int64_t info) {", "int64_t info, torch::Tensor xp) {")
    cu=cu.replace("if (!info) {", "if (!info) {\n    if(!x.is_cuda() || !x.is_contiguous() || (uintptr_t)x.data_ptr()%4 || xp.scalar_type()!=torch::kInt32 || xp.numel()!=262144 || !xp.is_contiguous() || !xp.is_cuda() || xp.device()!=x.device() || (uintptr_t)xp.data_ptr()%16) return -14;",1)
    cu=cu.replace("  k<<<dim3(", "  rx_pack<<<256,256,0,at::cuda::getCurrentCUDAStream()>>>((const uint32_t*)p.X,(uint32_t*)p.XP);\n  const cudaError_t pe=cudaGetLastError();if(pe!=cudaSuccess)return(int)pe;\n  k<<<dim3(",1)
    cu=cu.replace("int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);","tx = {}; int r = 0;")
    old_info="return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;"
    new_info="""if(info==4){
      if(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess){cudaGetLastError();return -6;}
      int occupancy=0;
      if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,k,512,C::SMEM)!=cudaSuccess){cudaGetLastError();return -7;}
      return occupancy;
    }
    if(info==5)return a.maxDynamicSharedSizeBytes;
    if(info==6)return a.maxThreadsPerBlock;
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;"""
    assert cu.count(old_info)==1
    cu=cu.replace(old_info,new_info)
    cu=cu.replace("dim3(384)", "dim3(512)")
    cu+="\nint64_t packinfo(int64_t info){cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,rx_pack)!=cudaSuccess){cudaGetLastError();return -1;}return info==1?a.numRegs:info==2?(int)a.localSizeBytes:a.maxThreadsPerBlock;}\n"
    assert "p.X =" in cu and "p.KC!=64" in cu
    return cu.replace("@@WGMMA@@",S._rqwg(64)).replace("@@CASES@@","case 0: return rq_go<128,8,1,0,0,0,20>(p,x,out,(int)KS,(int)info);").replace("rq<", "rq_px_g8<")


def device_source():
    cu=source().split("typedef CUresult (*RqEnc)")[0]
    cu=cu.replace("#include <torch/extension.h>","#include <cuda_runtime.h>").replace("#include <ATen/cuda/CUDAContext.h>","")
    return cu+"\nvoid instantiate(RqP p,RqTmap x,RqTmap o){rq_px_g8<128,8,1,0,0,0,20><<<128,512,RqC<128,8,1,0,0,0,20>::SMEM>>>(p,x,o);}\n"


_ext=None
def extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        cu=source()
        _ext=load_inline("operand_reverse_group8_"+hashlib.sha256(cu.encode()).hexdigest()[:12],
          cpp_sources="int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t,torch::Tensor); int64_t packinfo(int64_t);",
          cuda_sources=cu,functions=["rqrun","packinfo"],extra_cuda_cflags=S.RQFLAGS,verbose=True)
    return _ext
