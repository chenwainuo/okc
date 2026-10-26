"""Isolated o32/o128 fixed-policy lab. Not a submission or incumbent mutation."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HELPER = HERE.parent / "rs32/submission.py"
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == "a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad", "rs32 helper source changed"
S = load(HELPER, "od_rs32_helpers")
VARIANTS = [("od32",32,8,1,0,0,0,20),("od128",128,8,1,0,0,0,20)]


def source():
    old = S.RQCU
    start = old.index("template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>\nstruct RqC")
    end = old.index("typedef CUresult (*RqEnc)", start)
    cu = old[:start] + (HERE / "kernel_body.cuh").read_text() + "\n" + old[end:]
    assert cu.count("dim3(384)") == 1
    cu = cu.replace("dim3(384)","dim3(256)")
    cu = cu.replace("if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;",
                    "if (p.m != NT || KS != 1 || p.KC != 16 || p.N != 8192) return -4;")
    assert "p.m != NT || KS != 1" in cu
    cases = "\n".join("case %d: return rq_go<%s>(p,x,out,(int)KS,(int)info);" %
                        (i,",".join(map(str,v[1:]))) for i,v in enumerate(VARIANTS))
    return cu.replace("@@WGMMA@@",S._rqwg(32)+S._rqwg(128)).replace("@@CASES@@",cases)


def device_source():
    cu = source().split("typedef CUresult (*RqEnc)")[0]
    cu = cu.replace("#include <torch/extension.h>","#include <cuda_runtime.h>").replace("#include <ATen/cuda/CUDAContext.h>","")
    for nt in (32,128):
        cu += ("\nvoid instantiate_%d(RqP p,RqTmap tx,RqTmap to) { "
               "rq<%d,8,1,0,0,0,20><<<128,256,RqC<%d,8,1,0,0,0,20>::SMEM>>>(p,tx,to); }\n") % (nt,nt,nt)
    return cu


_ext = None


def extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        cu = source()
        _ext = load_inline("od_w_"+hashlib.sha256(cu.encode()).hexdigest()[:12],
                          cpp_sources="int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t);",
                          cuda_sources=cu, functions=["rqrun"], extra_cuda_cflags=S.RQFLAGS, verbose=True)
    return _ext
