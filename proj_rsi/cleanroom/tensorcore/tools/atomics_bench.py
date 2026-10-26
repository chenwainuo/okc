"""How fast is a split-K partial flush? 132 CTAs each add a 128(n) x MP(m) fp32 tile into ws[m][N] (overlapping ~7 per
n-block) with red.global.add.v4.f32, vs plain st.global.v4, vs scalar red.  Also: an empty kernel's graph time."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from torch.utils.cpp_extension import load_inline
from bench_util import time_graph, flush_buffer

src = r"""
#include <cuda_bf16.h>
template <int MODE>
__global__ void __launch_bounds__(256) flush_kernel(float* ws, int N, int MP, int nblocks) {
    int c = blockIdx.x;
    int j = (c * nblocks) / gridDim.x;           // n-block of this CTA
    int t = threadIdx.x;
    // tile 128 n x MP m; each thread does MP*128/256/4 float4 adds, coalesced along n
    float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (int i = t; i < MP * 32; i += 256) {      // 32 float4 per token row
        int tok = i / 32, n4 = (i % 32) * 4;
        float* p = ws + (size_t)tok * N + j * 128 + n4;
        if (MODE == 0) {
            asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
        } else if (MODE == 1) {
            *reinterpret_cast<float4*>(p) = v;
        } else {
            atomicAdd(p, v.x); atomicAdd(p + 1, v.y); atomicAdd(p + 2, v.z); atomicAdd(p + 3, v.w);
        }
    }
}
__global__ void empty_kernel(float* ws) { if (threadIdx.x == 999) ws[0] = 1.f; }

void run(torch::Tensor ws, int N, int MP, int nblocks, int mode, int grid) {
    float* p = ws.data_ptr<float>();
    if (mode == 0) flush_kernel<0><<<grid, 256>>>(p, N, MP, nblocks);
    else if (mode == 1) flush_kernel<1><<<grid, 256>>>(p, N, MP, nblocks);
    else if (mode == 2) flush_kernel<2><<<grid, 256>>>(p, N, MP, nblocks);
    else empty_kernel<<<grid, 256>>>(p);
}
"""
mod = load_inline("tensorcore_atomics", cpp_sources="void run(torch::Tensor ws, int N, int MP, int nblocks, int mode, int grid);",
                  cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
flush = flush_buffer()
for N, MP in [(2304, 128), (2304, 256), (4096, 128), (8192, 128), (256, 128), (256, 256)]:
    ws = torch.zeros(MP, N, device="cuda")
    nb = N // 128
    for mode, name in [(0, "red.v4"), (1, "st.v4"), (2, "red.scalar"), (3, "empty")]:
        t = time_graph(lambda: mod.run(ws, N, MP, nb, mode, 132), flush)
        print(f"N={N} MP={MP} {name:10s} {t:7.2f} us  ({132*128*MP*4/1e6:.1f} MB of partials)")
