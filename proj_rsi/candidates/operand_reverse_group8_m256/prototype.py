"""M256 minimal tiled port of frozen G8. Import performs no CUDA operations."""
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=HERE.parent/'operand_reverse_group8'
PARENT_MANIFEST='2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7'
PARENT_SOURCE='e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e'
assert hashlib.sha256((PARENT/'manifest.json').read_bytes()).hexdigest()==PARENT_MANIFEST
frozen=json.loads((PARENT/'manifest.json').read_text())
for file in ('prototype.py','kernel_body.cuh'):
    assert hashlib.sha256((PARENT/file).read_bytes()).hexdigest()==frozen['files'][file]
spec=importlib.util.spec_from_file_location('frozen_group8_m256_parent',PARENT/'prototype.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
S=G.S
VARIANT=G.VARIANT

def source():
    cu=G.source();assert hashlib.sha256(cu.encode()).hexdigest()==PARENT_SOURCE
    old=(PARENT/'kernel_body.cuh').read_text();assert cu.count(old)==1
    cu=cu.replace(old,(HERE/'kernel_body.cuh').read_text())
    for a,b in (
        ('rq_px_g8<','rq_px_g8_m256<'),
        ('if(p.m!=128 ||','if(p.m!=256 ||'),
        ('xp.numel()!=262144','xp.numel()!=524288'),
        ('rx_pack<<<256,256,','rx_pack<<<dim3(256,2),256,'),
        ('dim3(p.NB * ((FL & 16) ? 2 : 1), KS)','dim3(p.NB * ((FL & 16) ? 2 : 1), 2)'),
    ):
        assert cu.count(a)==1,(a,cu.count(a));cu=cu.replace(a,b)
    return cu

def device_source():
    cu=source().split('typedef CUresult (*RqEnc)')[0]
    cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    return cu+'\nvoid instantiate(RqP p,RqTmap x,RqTmap o){rq_px_g8_m256<128,8,1,0,0,0,20><<<dim3(128,2),512,RqC<128,8,1,0,0,0,20>::SMEM>>>(p,x,o);}\n'

_ext=None
def extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        cu=source()
        _ext=load_inline('operand_reverse_group8_m256_'+hashlib.sha256(cu.encode()).hexdigest()[:12],
          cpp_sources='int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t,torch::Tensor); int64_t packinfo(int64_t);',
          cuda_sources=cu,functions=['rqrun','packinfo'],extra_cuda_cflags=S.RQFLAGS,verbose=True)
    return _ext
