"""Microbenchmark: which load mechanism streams N bytes per CTA fastest on 128 CTAs after an L2 flush.

tma  : 1-D cp.async.bulk copies of COPY bytes, DEPTH in flight per CTA (ring in smem)
ld   : ld.global.nc.v4 per thread, UNROLL loads in flight per thread (256 threads)
cpa  : cp.async.cg 16 B per thread into an smem ring, DEPTH commit groups in flight
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}

// ---- TMA bulk ring: copies of `copy` bytes, `depth` in flight, with `split` copies per stage (issued back to back)
__global__ void __launch_bounds__(256, 1) k_tma(const uint8_t* __restrict__ src, uint32_t* __restrict__ out,
                                                 int bytes_per_cta, int copy, int depth, int split) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bars[32];
    const int tid = threadIdx.x;
    const uint8_t* base = src + (size_t)blockIdx.x * bytes_per_cta;
    const int nst = bytes_per_cta / copy;
    if (tid == 0) {
        for (int i = 0; i < depth; ++i) mbar_init(&bars[i], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < depth && i < nst; ++i) {
            mbar_expect_tx(&bars[i], copy);
            const int piece = copy / split;
            for (int j = 0; j < split; ++j)
                tma_load_1d(smem + i * copy + j * piece, base + (size_t)i * copy + j * piece, piece, &bars[i]);
        }
    }
    uint32_t acc = 0;
    for (int s = 0; s < nst; ++s) {
        const int buf = s % depth;
        mbar_wait(&bars[buf], (s / depth) & 1);
        acc ^= reinterpret_cast<const uint32_t*>(smem + buf * copy)[tid];
        __syncthreads();
        if (tid == 0 && s + depth < nst) {
            mbar_expect_tx(&bars[buf], copy);
            const int piece = copy / split;
            for (int j = 0; j < split; ++j)
                tma_load_1d(smem + buf * copy + j * piece, base + (size_t)(s + depth) * copy + j * piece, piece, &bars[buf]);
        }
    }
    if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

// ---- plain vector loads: each thread streams 16 B per step, UNROLL loads in flight
template <int UNROLL>
__global__ void __launch_bounds__(256, 1) k_ld(const uint8_t* __restrict__ src, uint32_t* __restrict__ out, int bytes_per_cta) {
    const int tid = threadIdx.x;
    const uint4* base = reinterpret_cast<const uint4*>(src + (size_t)blockIdx.x * bytes_per_cta);
    const int n16 = bytes_per_cta / 16;
    uint32_t acc = 0;
    for (int i = tid; i < n16; i += 256 * UNROLL) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int idx = i + u * 256;
            v[u] = idx < n16 ? __ldg(base + idx) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

// ---- cp.async 16 B per thread into an smem ring of DEPTH stages of 4 KB (256 x 16 B) x CHUNK
template <int DEPTH, int CHUNK>
__global__ void __launch_bounds__(256, 1) k_cpa(const uint8_t* __restrict__ src, uint32_t* __restrict__ out, int bytes_per_cta) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x;
    const uint8_t* base = src + (size_t)blockIdx.x * bytes_per_cta;
    constexpr int STAGE = 4096 * CHUNK;
    const int nst = bytes_per_cta / STAGE;
    auto issue = [&](int s) {
        const int buf = s % DEPTH;
#pragma unroll
        for (int cch = 0; cch < CHUNK; ++cch) {
            uint32_t d = (uint32_t)__cvta_generic_to_shared(smem + buf * STAGE + cch * 4096 + tid * 16);
            const void* g = base + (size_t)s * STAGE + cch * 4096 + tid * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(g) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    for (int i = 0; i < DEPTH && i < nst; ++i) issue(i);
    uint32_t acc = 0;
    for (int s = 0; s < nst; ++s) {
        asm volatile("cp.async.wait_group %0;" ::"n"(DEPTH - 1) : "memory");
        __syncthreads();
        acc ^= reinterpret_cast<const uint32_t*>(smem + (s % DEPTH) * STAGE)[tid];
        __syncthreads();
        if (s + DEPTH < nst) issue(s + DEPTH); else asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

void run_tma(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t copy, int64_t depth, int64_t split) {
    static bool conf = false;
    if (!conf) { cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024); conf = true; }
    k_tma<<<n_cta, 256, copy * depth, at::cuda::getCurrentCUDAStream()>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta, copy, depth, split);
}
void run_ld(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t unroll) {
    auto s = at::cuda::getCurrentCUDAStream();
    if (unroll == 4) k_ld<4><<<n_cta, 256, 0, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
    else if (unroll == 8) k_ld<8><<<n_cta, 256, 0, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
    else k_ld<16><<<n_cta, 256, 0, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
}
void run_cpa(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t depth, int64_t chunk) {
    auto s = at::cuda::getCurrentCUDAStream();
    static bool conf = false;
    if (!conf) {
        cudaFuncSetAttribute(k_cpa<4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cudaFuncSetAttribute(k_cpa<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cudaFuncSetAttribute(k_cpa<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cudaFuncSetAttribute(k_cpa<8, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        conf = true;
    }
    if (depth == 4 && chunk == 2) k_cpa<4, 2><<<n_cta, 256, 4096 * 2 * 4, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
    else if (depth == 8 && chunk == 2) k_cpa<8, 2><<<n_cta, 256, 4096 * 2 * 8, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
    else if (depth == 4 && chunk == 4) k_cpa<4, 4><<<n_cta, 256, 4096 * 4 * 4, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
    else k_cpa<8, 1><<<n_cta, 256, 4096 * 8, s>>>(src.data_ptr<uint8_t>(), out.data_ptr<uint32_t>(), bytes_per_cta);
}
"""

CPP = r"""
#include <torch/extension.h>
void run_tma(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t copy, int64_t depth, int64_t split);
void run_ld(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t unroll);
void run_cpa(torch::Tensor src, torch::Tensor out, int64_t n_cta, int64_t bytes_per_cta, int64_t depth, int64_t chunk);
"""


def time_graph(fn, flush, reps=15):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    times = []
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    fview = flush.view(torch.float32)
    for _ in range(reps):
        torch.sum(fview, 0, out=sink)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--ctas", default="128")
    args = ap.parse_args()
    ext = load_inline(name="latency_mb_load", cpp_sources=[CPP], cuda_sources=[SRC],
                      functions=["run_tma", "run_ld", "run_cpa"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"])
    src = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    out = torch.zeros(1024, dtype=torch.int32, device="cuda").view(torch.uint32) if hasattr(torch, "uint32") else torch.zeros(1024, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    tiny = torch.zeros(32, device="cuda")
    ovh = time_graph(lambda: tiny.add_(1.0), flush, args.reps)
    print(f"REPLAY_OVERHEAD {ovh:.2f} us")
    for n_cta in [int(v) for v in args.ctas.split(",")]:
        for kb in (48, 96, 384):
            bpc = kb * 1024
            total_mb = n_cta * bpc / 1e6
            rows = []
            for copy_kb, depth, split in ((3, 16, 1), (6, 8, 1), (12, 4, 1), (24, 4, 1), (48, 2, 1), (24, 4, 4), (48, 2, 8), (12, 8, 1)):
                if copy_kb * 1024 > bpc or (bpc % (copy_kb * 1024)):
                    continue
                t = time_graph(lambda: ext.run_tma(src, out, n_cta, bpc, copy_kb * 1024, depth, split), flush, args.reps)
                rows.append((f"tma copy={copy_kb}K depth={depth} split={split}", t))
            for unroll in (4, 8, 16):
                t = time_graph(lambda: ext.run_ld(src, out, n_cta, bpc, unroll), flush, args.reps)
                rows.append((f"ld unroll={unroll}", t))
            for depth, chunk in ((4, 2), (8, 2), (4, 4), (8, 1)):
                if bpc % (4096 * chunk):
                    continue
                t = time_graph(lambda: ext.run_cpa(src, out, n_cta, bpc, depth, chunk), flush, args.reps)
                rows.append((f"cpa depth={depth} stage={4 * chunk}K", t))
            for name, t in rows:
                print(f"ctas={n_cta} {kb:4d}KB/cta ({total_mb:6.1f} MB) {name:32s} {t:7.2f} us  {t - ovh:6.2f} net  {total_mb / max(t - ovh, 0.01) / 1e3:.2f} TB/s", flush=True)


if __name__ == "__main__":
    main()
