# okc: problem=muse_bf16_proj
"""latency: one kernel node per call for every m <= 32.

Design (clean room):
  * Weights are re-encoded losslessly in prepare() as two planes per weight: the low byte of the bf16
    (exponent bit 0 + 7 mantissa bits) and a nibble (sign + 3 bits of exponent>>1 relative to a per-layer
    base), i.e. 12 bits/weight. Weights whose exponent falls outside the 16-code window are encoded as a
    fixed escape value and fixed up exactly from a per-CTA exception list (x[k] * (true - escape)).
    The router keeps its fp32 weights verbatim and splits them into bf16 hi + lo in-kernel.
  * The packed stream of each CTA is contiguous and is pulled by 1-D TMA bulk copies into a deep smem
    ring: every CTA issues all its resident stages at launch, so the whole GPU's HBM queue is full from
    the first microsecond; no clusters, 128 CTAs, one wave.
  * Compute is mma.sync m16n8k16 bf16 with fp32 accumulation; x fragments are read straight from L2
    (16 B per lane, k permuted consistently on both operands so every load is a full sector).
  * The 8 warps of a CTA split K; their partial sums are reduced once in smem and the output is written
    exactly once. No cross-CTA reduction anywhere.
  * m > 32 falls back to cuBLAS (torch.mm) / the hi+lo router split.
"""

import math
import os
from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace latency_ns {

constexpr int LOADER = LATENCY_LOADER;  // 0: TMA 1-D bulk per stage, 1: cp.async 16 B per thread

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define LATENCY_SM90A 1
#endif

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
#ifdef LATENCY_SM90A
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count));
#endif
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
#ifdef LATENCY_SM90A
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#ifdef LATENCY_SM90A
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}" ::"r"(a), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
#ifdef LATENCY_SM90A
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
#endif
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
#ifdef LATENCY_SM90A
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(a) : "memory");
#endif
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
#ifdef LATENCY_SM90A
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a) : "memory");
#endif
}

__device__ __forceinline__ uint4 lds128(const uint8_t* p) {
    uint4 v;
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
    return v;
}
__device__ __forceinline__ uint2 lds64(const uint8_t* p) {
    uint2 v;
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
    return v;
}

// 4 weights: L word (4 low bytes), sel = 4 nibbles (sign<<3 | (E>>1)-hb) in the low 16 bits.
__device__ __forceinline__ void decode4(uint32_t L, uint32_t sel, uint32_t lut_lo, uint32_t lut_hi,
                                        uint32_t& pair0, uint32_t& pair1) {
    uint32_t p1 = prmt(lut_lo, lut_hi, sel & 0x7777u);            // E>>1 bytes
    uint32_t p2 = prmt(0x80808080u, 0x80808080u, sel);            // 0xFF where sign set, else 0x80
    uint32_t hi = p1 | ((p2 << 1) & 0x80808080u);                  // sign | E>>1
    pair0 = prmt(L, hi, 0x5140u);
    pair1 = prmt(L, hi, 0x7362u);
}

struct Exc { uint32_t nk; float delta; };

template <int NB, int MT, int MODE>
__global__ void __launch_bounds__(256, 1)
latency_gemv(const uint8_t* __restrict__ packed, const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
             const Exc* __restrict__ exc, const int* __restrict__ exc_off,
             int m, int K, int N, int nstages, int S, uint32_t lut_lo, uint32_t lut_hi, uint32_t lut2_lo, uint32_t lut2_hi)
{
    constexpr bool F32 = MODE == 1;
    constexpr bool DUAL = MODE == 2;
    constexpr int NT = (NB + 7) / 8;
    constexpr int ROWB = F32 ? 256 : (DUAL ? 192 : 96);            // bytes per row per k64 block
    constexpr int STAGE = 8 * NB * ROWB;            // 8 k64 blocks (one per warp) of NB rows
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t bars[16];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int cta = blockIdx.x;
    const uint8_t* src = packed + (size_t)cta * (size_t)nstages * STAGE;

    if (tid == 0) {
        for (int i = 0; i < S; ++i) mbar_init(&bars[i], LOADER == 0 ? 1 : 256);
#ifdef LATENCY_SM90A
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    }
    __syncthreads();
    auto issue_stage = [&](int st) {
        const int buf = st % S;
        if (LOADER == 0) {
            if (tid == 0) {
                mbar_expect_tx(&bars[buf], STAGE);
                tma_load_1d(smem + buf * STAGE, src + (size_t)st * STAGE, STAGE, &bars[buf]);
            }
        } else {
            uint8_t* d = smem + buf * STAGE;
            const uint8_t* g = src + (size_t)st * STAGE;
            for (int off = tid * 16; off < STAGE; off += 4096) cp_async16(d + off, g + off);
            cp_async_mbar_arrive(&bars[buf]);
        }
    };
    for (int i = 0; i < S; ++i) issue_stage(i);

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[a][t][i] = 0.f;

    // x fragments for one k64 block: [mt][h][row g / g+8][4 regs]
    uint32_t xn[MT][2][2][4];
    auto load_x = [&](int blk) {
        const int k0 = blk * 64;
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = g + 8 * rr + 16 * a;
                    if (row < m) {
                        const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)row * K + k0 + h * 32 + c * 8));
                        xn[a][h][rr][0] = v.x; xn[a][h][rr][1] = v.y; xn[a][h][rr][2] = v.z; xn[a][h][rr][3] = v.w;
                    } else {
                        xn[a][h][rr][0] = xn[a][h][rr][1] = xn[a][h][rr][2] = xn[a][h][rr][3] = 0u;
                    }
                }
    };
    load_x(warp);

    for (int s = 0; s < nstages; ++s) {
        const int buf = s % S;
        const uint8_t* base = smem + buf * STAGE;
        uint32_t xc[MT][2][2][4];
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
#pragma unroll
                    for (int i = 0; i < 4; ++i) xc[a][h][rr][i] = xn[a][h][rr][i];
        if (s + 1 < nstages) load_x((s + 1) * 8 + warp);

        mbar_wait(&bars[buf], (uint32_t)((s / S) & 1));

#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int n = t * 8 + g;
            const bool has = (NB % 8 == 0) || (n < NB);
            uint32_t bf[2][2][2];   // [h][frag][b0,b1]
            if (!F32) {
#pragma unroll
                for (int pl = 0; pl < (DUAL ? 2 : 1); ++pl) {
                    uint4 L = make_uint4(0u, 0u, 0u, 0u);
                    uint2 H = make_uint2(0u, 0u);
                    const uint8_t* pb = base + pl * (8 * NB * 96);
                    const uint32_t l0 = pl ? lut2_lo : lut_lo, l1 = pl ? lut2_hi : lut_hi;
                    if (has) {
                        L = lds128(pb + warp * (NB * 64) + n * 64 + c * 16);
                        H = lds64(pb + 8 * NB * 64 + warp * (NB * 32) + n * 32 + c * 8);
                    }
                    decode4(L.x, H.x, l0, l1, bf[0][0][0], bf[0][0][1]);
                    decode4(L.y, H.x >> 16, l0, l1, bf[0][1][0], bf[0][1][1]);
                    decode4(L.z, H.y, l0, l1, bf[1][0][0], bf[1][0][1]);
                    decode4(L.w, H.y >> 16, l0, l1, bf[1][1][0], bf[1][1][1]);
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int f = 0; f < 2; ++f)
#pragma unroll
                            for (int a = 0; a < MT; ++a) {
                                uint32_t A[4] = {xc[a][h][0][2 * f], xc[a][h][1][2 * f], xc[a][h][0][2 * f + 1], xc[a][h][1][2 * f + 1]};
                                mma_bf16(acc[a][t], A, bf[h][f][0], bf[h][f][1]);
                            }
                }
            } else {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = make_uint4(0u, 0u, 0u, 0u);
                    if (has) {
                        const uint8_t* p = base + warp * (NB * 256) + n * 256 + h * 128 + c * 32;
                        w0 = lds128(p);
                        w1 = lds128(p + 16);
                    }
                    uint32_t w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                    uint32_t hi[4], lo[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const uint32_t u0 = w[2 * p], u1 = w[2 * p + 1];
                        hi[p] = prmt(u0, u1, 0x7632u);
                        const float r0 = __uint_as_float(u0 & 0xFFFF0000u), r1 = __uint_as_float(u1 & 0xFFFF0000u);
                        const float l0 = __uint_as_float(u0) - r0, l1 = __uint_as_float(u1) - r1;
                        __nv_bfloat162 lp = __floats2bfloat162_rn(l0, l1);
                        lo[p] = *reinterpret_cast<uint32_t*>(&lp);
                    }
#pragma unroll
                    for (int f = 0; f < 2; ++f)
#pragma unroll
                        for (int a = 0; a < MT; ++a) {
                            uint32_t A[4] = {xc[a][h][0][2 * f], xc[a][h][1][2 * f], xc[a][h][0][2 * f + 1], xc[a][h][1][2 * f + 1]};
                            mma_bf16(acc[a][t], A, hi[2 * f], hi[2 * f + 1]);
                            mma_bf16(acc[a][t], A, lo[2 * f], lo[2 * f + 1]);
                        }
                }
            }
        }
        __syncthreads();
        if (s + S < nstages) issue_stage(s + S);
    }

    // exceptions (weights outside the exponent window): acc[m][n] += x[m][k] * delta
    if (!F32) {
        const int e0 = exc_off[cta], e1 = exc_off[cta + 1];
        for (int e = e0 + warp; e < e1; e += 8) {
            const Exc ex = exc[e];
            const int nl = ex.nk & 0xFFFF, k = ex.nk >> 16;
            const int t = nl >> 3, col = nl & 7, cc = col >> 1, j = col & 1;
            if (cc == c) {
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    const int r0 = g + 16 * a, r1 = r0 + 8;
                    const float x0 = r0 < m ? __bfloat162float(x[(size_t)r0 * K + k]) : 0.f;
                    const float x1 = r1 < m ? __bfloat162float(x[(size_t)r1 * K + k]) : 0.f;
#pragma unroll
                    for (int tt = 0; tt < NT; ++tt)
                        if (tt == t) {
                            if (j == 0) { acc[a][tt][0] += x0 * ex.delta; acc[a][tt][2] += x1 * ex.delta; }
                            else        { acc[a][tt][1] += x0 * ex.delta; acc[a][tt][3] += x1 * ex.delta; }
                        }
                }
            }
        }
    }

    // cross-warp reduction over the K split, then one store of the output
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
    constexpr int RROWS = MT * 16, RCOLS = NT * 8;
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int r0 = g + 16 * a, col = t * 8 + 2 * c;
            float* p = red + (size_t)warp * RROWS * RCOLS;
            p[r0 * RCOLS + col] = acc[a][t][0];
            p[r0 * RCOLS + col + 1] = acc[a][t][1];
            p[(r0 + 8) * RCOLS + col] = acc[a][t][2];
            p[(r0 + 8) * RCOLS + col + 1] = acc[a][t][3];
        }
    __syncthreads();
    const int total = m * NB;
    for (int idx = tid; idx < total; idx += 256) {
        const int row = idx / NB, col = idx - row * NB;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sum += red[(size_t)w * RROWS * RCOLS + row * RCOLS + col];
        const size_t o = (size_t)row * N + (size_t)cta * NB + col;
        if (F32 || DUAL) reinterpret_cast<float*>(out)[o] = sum;
        else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(sum);
    }
}

template <int NB, int MT, int MODE>
void launch(const torch::Tensor& packed, const torch::Tensor& x, torch::Tensor& out, const torch::Tensor& exc,
            const torch::Tensor& exc_off, int m, int K, int N, int nstages, int S, const uint32_t* lut,
            int smem_bytes) {
    auto kfn = latency_gemv<NB, MT, MODE>;
    static bool configured = false;   // one per instantiation
    if (!configured) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024);
        configured = true;
    }
    const int n_cta = N / NB;
    kfn<<<n_cta, 256, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr(),
        reinterpret_cast<const Exc*>(exc.data_ptr()), exc_off.data_ptr<int>(), m, K, N, nstages, S, lut[0], lut[1], lut[2], lut[3]);
}

// ---------------------------------------------------------------------------------------------------------
// LOADER 2: warp-private cp.async rings. Warp w owns k64 block w of every stage: its weight planes, its
// x slice (m rows x 64 k, 16 B chunks, swizzled) and its exception list all arrive through its own ring of
// S slots, completion by cp.async.wait_group, refill right after the warp's own use. No CTA barrier and no
// global load on the critical path until the epilogue.
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_rt(int n) {
    switch (n) {
        case 0: cp_wait<0>(); break;  case 1: cp_wait<1>(); break;  case 2: cp_wait<2>(); break;
        case 3: cp_wait<3>(); break;  case 4: cp_wait<4>(); break;  case 5: cp_wait<5>(); break;
        case 6: cp_wait<6>(); break;  case 7: cp_wait<7>(); break;  case 8: cp_wait<8>(); break;
        case 9: cp_wait<9>(); break;  case 10: cp_wait<10>(); break; case 11: cp_wait<11>(); break;
        case 12: cp_wait<12>(); break; case 13: cp_wait<13>(); break; case 14: cp_wait<14>(); break;
        default: cp_wait<15>(); break;
    }
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ uint32_t lds32(const uint8_t* p) {
    uint32_t v, a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
    return v;
}
__device__ __forceinline__ uint32_t lds16(const uint8_t* p) {
    unsigned short v; uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
    return v;
}

__device__ unsigned long long g_stamps[1024 * 16];
__device__ __forceinline__ void stamp(int dbg, int cta, int warp, int lane, int slot) {
    if ((dbg & 16) && lane == 0 && (warp == 0 || warp == 7)) {
        unsigned long long gt, ck;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt));
        ck = clock64();
        g_stamps[(cta * 2 + (warp == 7)) * 8 + slot] = slot == 0 ? gt : ck;
        if (slot == 0) g_stamps[(cta * 2 + (warp == 7)) * 8 + 7] = ck;
    }
}

template <int NB, int MT, int MODE>
__global__ void __launch_bounds__(384, 1)
latency_gemv_w(const uint8_t* __restrict__ packed, const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
               const uint8_t* __restrict__ excw, int ecap, int ksplit, int dbg,
               int m, int K, int N, int nstages, int S, uint32_t lut_lo, uint32_t lut_hi, uint32_t lut2_lo, uint32_t lut2_hi)
{
    constexpr bool DUAL = MODE == 2;
    constexpr int NPL = DUAL ? 2 : 1;
    constexpr int NT = (NB + 7) / 8;
    constexpr int WB = NB * 96 * NPL;          // weight bytes of one warp's k64 block
    constexpr int STAGE = 8 * WB;
    constexpr int WCH = WB / 16;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t fullb[8 * 16], emptyb[8 * 16];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int cta = blockIdx.x;                    // = n_tile * ksplit + kq; a cluster = the ksplit CTAs of one n_tile
    const int ntile = cta / ksplit, kq = cta - ntile * ksplit;
    const int nst = nstages / ksplit, st0 = kq * nst;
    const int XB = m * 128;
    const int SLOT = (WB + XB + 127) & ~127;
    const int EB = ecap * 8;
    const bool prod = dbg & 32;                    // producer warps (warps 8..) feed the 8 consumer warps
    const int P = (int)(blockDim.x >> 5) - 8;
    stamp(dbg, cta, warp, lane, 0);
    const bool wl = dbg & 8;   // warp-contiguous layout: [n_tile][warp][stage][pl][L|H]
    const uint8_t* xb = reinterpret_cast<const uint8_t*>(x);

    // copy (warp w, stage st) into w's ring slot: its weight planes and its x slice
    auto issue_for = [&](int w, int st) {
        uint8_t* d = smem + 8 * EB + w * S * SLOT + (st % S) * SLOT;
        if (wl) {
            const uint8_t* gs = packed + (((size_t)ntile * 8 + w) * (size_t)nstages + st0 + st) * WB;
#pragma unroll 4
            for (int i = lane; i < WCH; i += 32) cp_async16(d + i * 16, gs + i * 16);
        } else {
            const uint8_t* gs = packed + ((size_t)ntile * (size_t)nstages + st0 + st) * STAGE;
#pragma unroll 4
            for (int i = lane; i < WCH; i += 32) {
                const int pl = NPL == 1 ? 0 : i / (NB * 6);
                const int r = i - pl * (NB * 6);
                const uint8_t* gp = gs + pl * (8 * NB * 96) +
                    (r < NB * 4 ? w * (NB * 64) + r * 16 : 8 * NB * 64 + w * (NB * 32) + (r - NB * 4) * 16);
                cp_async16(d + i * 16, gp);
            }
        }
        const int k0 = (st0 + st) * 512 + w * 64;
        if (!(dbg & 1)) for (int i = lane; i < m * 8; i += 32) {
            const int r = i >> 3, ch = i & 7;
            cp_async16(d + WB + r * 128 + ((ch ^ ((r & 1) << 2)) << 4), xb + ((size_t)r * K + k0 + ch * 8) * 2);
        }
    };

    if (prod) {
        if (tid == 0) {
            for (int i = 0; i < 8 * 16; ++i) { mbar_init(&fullb[i], 32); mbar_init(&emptyb[i], 1); }
#ifdef LATENCY_SM90A
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
        }
        __syncthreads();
    }
    uint8_t* eslot = smem + (warp < 8 ? warp : 0) * EB;
    uint8_t* ring = smem + 8 * EB + (warp < 8 ? warp : 0) * S * SLOT;
    if (warp < 8) {   // exception list of this (cta, warp): rides in the first group
        const uint8_t* es = excw + (size_t)(cta * 8 + warp) * EB;
        for (int i = lane; i < EB / 16; i += 32) cp_async16(eslot + i * 16, es + i * 16);
    }
    auto issue = [&](int st) {
        if (st < nst) issue_for(warp, st);
        cp_commit();
    };
    if (!prod) {
        for (int i = 0; i < S; ++i) issue(i);
    } else if (warp >= 8) {
        // producer: stage-major over its consumer warps; a slot is re-armed once its consumer released it
        for (int st = 0; st < nst; ++st) {
            const int slot = st % S;
            for (int w = warp - 8; w < 8; w += P) {
                if (st >= S) mbar_wait(&emptyb[w * 16 + slot], (uint32_t)(((st / S) - 1) & 1));
                issue_for(w, st);
                cp_async_mbar_arrive(&fullb[w * 16 + slot]);
            }
        }
        cp_wait<0>();
    } else {
        cp_commit();
        cp_wait<0>();
        __syncwarp();
    }
    stamp(dbg, cta, warp, lane, 1);

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[a][t][i] = 0.f;

    int ep = 0;
    if (warp < 8)
    for (int s = 0; s < nst; ++s) {
        if (prod) mbar_wait(&fullb[warp * 16 + (s % S)], (uint32_t)((s / S) & 1));
        else cp_wait_rt(S - 1);
        __syncwarp();
        if (s == 0) stamp(dbg, cta, warp, lane, 2);
        if (s == nst / 2) stamp(dbg, cta, warp, lane, 3);
        const uint8_t* base = ring + (s % S) * SLOT;
        const uint8_t* xs = base + WB;
        uint32_t xc[MT][2][2][4];
#pragma unroll
        for (int a = 0; a < MT; ++a)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = g + 8 * rr + 16 * a;
                    if (row < m && !(dbg & 1)) {
                        const uint4 v = lds128(xs + row * 128 + (((h * 4 + c) ^ ((row & 1) << 2)) << 4));
                        xc[a][h][rr][0] = v.x; xc[a][h][rr][1] = v.y; xc[a][h][rr][2] = v.z; xc[a][h][rr][3] = v.w;
                    } else {
                        xc[a][h][rr][0] = xc[a][h][rr][1] = xc[a][h][rr][2] = xc[a][h][rr][3] = 0u;
                    }
                }
        if (dbg & 2) {
            acc[0][0][0] += __uint_as_float(lds32(base + lane * 4) & 0x3fffffffu);
        } else
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int n = t * 8 + g;
            const bool has = (NB % 8 == 0) || (n < NB);
            uint32_t bf[2][2][2];
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl) {
                uint4 L = make_uint4(0u, 0u, 0u, 0u);
                uint2 H = make_uint2(0u, 0u);
                const uint8_t* pb = base + pl * (NB * 96);
                const uint32_t l0 = pl ? lut2_lo : lut_lo, l1 = pl ? lut2_hi : lut_hi;
                if (has) {
                    L = lds128(pb + n * 64 + c * 16);
                    H = lds64(pb + NB * 64 + n * 32 + c * 8);
                }
                decode4(L.x, H.x, l0, l1, bf[0][0][0], bf[0][0][1]);
                decode4(L.y, H.x >> 16, l0, l1, bf[0][1][0], bf[0][1][1]);
                decode4(L.z, H.y, l0, l1, bf[1][0][0], bf[1][0][1]);
                decode4(L.w, H.y >> 16, l0, l1, bf[1][1][0], bf[1][1][1]);
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int f = 0; f < 2; ++f)
#pragma unroll
                        for (int a = 0; a < MT; ++a) {
                            uint32_t A[4] = {xc[a][h][0][2 * f], xc[a][h][1][2 * f], xc[a][h][0][2 * f + 1], xc[a][h][1][2 * f + 1]};
                            mma_bf16(acc[a][t], A, bf[h][f][0], bf[h][f][1]);
                        }
            }
        }
        // exceptions of this (stage, block), x from the slot
        while (ep < ecap && !(dbg & 4)) {
            const uint32_t nk = lds32(eslot + ep * 8);
            const int k = (int)(nk >> 16);
            if ((k >> 9) != st0 + s) break;
            const float delta = __uint_as_float(lds32(eslot + ep * 8 + 4));
            ++ep;
            const int nl = nk & 0xFFFF, kl = k & 63;
            const int t = nl >> 3, col = nl & 7, cc = col >> 1, j = col & 1;
            if (cc == c) {
#pragma unroll
                for (int a = 0; a < MT; ++a) {
                    const int r0 = g + 16 * a, r1 = r0 + 8;
                    const float x0 = r0 < m ? __uint_as_float(lds16(xs + r0 * 128 + ((((kl >> 3) ^ ((r0 & 1) << 2))) << 4) + (kl & 7) * 2) << 16) : 0.f;
                    const float x1 = r1 < m ? __uint_as_float(lds16(xs + r1 * 128 + ((((kl >> 3) ^ ((r1 & 1) << 2))) << 4) + (kl & 7) * 2) << 16) : 0.f;
#pragma unroll
                    for (int tt = 0; tt < NT; ++tt)
                        if (tt == t) {
                            if (j == 0) { acc[a][tt][0] += x0 * delta; acc[a][tt][2] += x1 * delta; }
                            else        { acc[a][tt][1] += x0 * delta; acc[a][tt][3] += x1 * delta; }
                        }
                }
            }
        }
        __syncwarp();
        if (prod) { if (lane == 0) mbar_arrive(&emptyb[warp * 16 + (s % S)]); }
        else issue(s + S);
    }
    cp_wait<0>();
    stamp(dbg, cta, warp, lane, 4);

    __syncthreads();
    stamp(dbg, cta, warp, lane, 5);
    float* red = reinterpret_cast<float*>(smem);
    constexpr int RROWS = MT * 16, RCOLS = NT * 8;
    if (warp < 8)
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int r0 = g + 16 * a, col = t * 8 + 2 * c;
            float* p = red + (size_t)warp * RROWS * RCOLS;
            *reinterpret_cast<float2*>(p + r0 * RCOLS + col) = make_float2(acc[a][t][0], acc[a][t][1]);
            *reinterpret_cast<float2*>(p + (r0 + 8) * RCOLS + col) = make_float2(acc[a][t][2], acc[a][t][3]);
        }
    __syncthreads();
    const int total = m * NB;
    if (ksplit == 1) {
        if (tid < 256) for (int idx = tid; idx < total; idx += 256) {
            const int row = idx / NB, col = idx - row * NB;
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) sum += red[(size_t)w * RROWS * RCOLS + row * RCOLS + col];
            const size_t o = (size_t)row * N + (size_t)ntile * NB + col;
            if (DUAL) reinterpret_cast<float*>(out)[o] = sum;
            else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(sum);
        }
        stamp(dbg, cta, warp, lane, 6);
        return;
    }
    // split-K: reduce the 8 warps into red2 [m][NB], then rank kq sums its share of red2 over the cluster (DSMEM)
    float* red2 = red + 8 * RROWS * RCOLS;
    if (tid < 256) for (int idx = tid; idx < total; idx += 256) {
        const int row = idx / NB, col = idx - row * NB;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sum += red[(size_t)w * RROWS * RCOLS + row * RCOLS + col];
        red2[idx] = sum;
    }
#ifdef LATENCY_SM90A
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t r2 = (uint32_t)__cvta_generic_to_shared(red2);
    if (tid < 256) for (int idx = kq * 256 + tid; idx < total; idx += 256 * ksplit) {
        float sum = 0.f;
        for (int r = 0; r < ksplit; ++r) {
            uint32_t ra; float v;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(r2 + idx * 4), "r"(r));
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
            sum += v;
        }
        const int row = idx / NB, col = idx - row * NB;
        const size_t o = (size_t)row * N + (size_t)ntile * NB + col;
        if (DUAL) reinterpret_cast<float*>(out)[o] = sum;
        else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16(sum);
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

template <int NB, int MT, int MODE>
void launch_w(const torch::Tensor& packed, const torch::Tensor& x, torch::Tensor& out, const torch::Tensor& excw,
              int ecap, int ksplit, int dbg, int m, int K, int N, int nstages, int S, const uint32_t* lut, int smem_bytes) {
    auto kfn = latency_gemv_w<NB, MT, MODE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024);
        cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        configured = true;
    }
    const int n_cta = N / NB * ksplit;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_cta);
    cfg.blockDim = dim3((dbg & 32) ? 256 + 32 * (((dbg >> 6) & 3) + 1) : 256);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = ksplit; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = ksplit > 1 ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kfn, packed.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                       out.data_ptr(), excw.data_ptr<uint8_t>(), ecap, ksplit, dbg, m, K, N, nstages, S,
                       lut[0], lut[1], lut[2], lut[3]);
}

}  // namespace latency_ns

torch::Tensor latency_stamps() {
    auto t = torch::zeros({1024 * 16}, torch::dtype(torch::kInt64));
    cudaMemcpyFromSymbol(t.data_ptr(), latency_ns::g_stamps, sizeof(unsigned long long) * 1024 * 16);
    return t;
}

torch::Tensor latency_gemv_w_run(torch::Tensor packed, torch::Tensor x, torch::Tensor excw, int64_t ecap,
                                 int64_t ksplit, int64_t NB, int64_t N, int64_t nstages, int64_t S, int64_t lut0, int64_t lut1,
                                 int64_t lut2, int64_t lut3, int64_t smem_bytes, int64_t mode, int64_t dbg) {
    using namespace latency_ns;
    const int m = x.size(0), K = x.size(1);
    auto out = torch::empty({m, N}, x.options().dtype(mode ? torch::kFloat32 : torch::kBFloat16));
    const int MT = m > 16 ? 2 : 1;
    uint32_t lut[4] = {(uint32_t)lut0, (uint32_t)lut1, (uint32_t)lut2, (uint32_t)lut3};
#define W_CASE(nb, md)                                                                                         \
    if (NB == nb && mode == md) {                                                                              \
        if (MT == 1) launch_w<nb, 1, md>(packed, x, out, excw, ecap, ksplit, dbg, m, K, N, nstages, S, lut, smem_bytes);    \
        else launch_w<nb, 2, md>(packed, x, out, excw, ecap, ksplit, dbg, m, K, N, nstages, S, lut, smem_bytes);            \
        return out;                                                                                            \
    }
    W_CASE(8, 0) W_CASE(16, 0) W_CASE(18, 0) W_CASE(32, 0) W_CASE(36, 0) W_CASE(64, 0) W_CASE(72, 0)
    W_CASE(2, 2) W_CASE(4, 2) W_CASE(8, 2) W_CASE(16, 2)
#undef W_CASE
    TORCH_CHECK(false, "unsupported NB/mode");
}

torch::Tensor latency_gemv_run(torch::Tensor packed, torch::Tensor x, torch::Tensor exc, torch::Tensor exc_off,
                               int64_t NB, int64_t N, int64_t nstages, int64_t S, int64_t lut0, int64_t lut1,
                               int64_t lut2, int64_t lut3, int64_t smem_bytes, int64_t mode) {
    using namespace latency_ns;
    const int m = x.size(0), K = x.size(1);
    auto out = torch::empty({m, N}, x.options().dtype(mode ? torch::kFloat32 : torch::kBFloat16));
    const int MT = m > 16 ? 2 : 1;
    uint32_t lut[4] = {(uint32_t)lut0, (uint32_t)lut1, (uint32_t)lut2, (uint32_t)lut3};
#define L_CASE(nb, md)                                                                                    \
    if (NB == nb && mode == md) {                                                                         \
        if (MT == 1) launch<nb, 1, md>(packed, x, out, exc, exc_off, m, K, N, nstages, S, lut, smem_bytes); \
        else launch<nb, 2, md>(packed, x, out, exc, exc_off, m, K, N, nstages, S, lut, smem_bytes);         \
        return out;                                                                                       \
    }
    L_CASE(8, 0) L_CASE(16, 0) L_CASE(18, 0) L_CASE(32, 0) L_CASE(64, 0) L_CASE(2, 1) L_CASE(4, 1) L_CASE(2, 2) L_CASE(4, 2)
#undef L_CASE
    TORCH_CHECK(false, "unsupported NB/mode");
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor latency_stamps();
torch::Tensor latency_gemv_run(torch::Tensor packed, torch::Tensor x, torch::Tensor exc, torch::Tensor exc_off,
                               int64_t NB, int64_t N, int64_t nstages, int64_t S, int64_t lut0, int64_t lut1,
                               int64_t lut2, int64_t lut3, int64_t smem_bytes, int64_t mode);
torch::Tensor latency_gemv_w_run(torch::Tensor packed, torch::Tensor x, torch::Tensor excw, int64_t ecap,
                                 int64_t ksplit, int64_t NB, int64_t N, int64_t nstages, int64_t S, int64_t lut0, int64_t lut1,
                                 int64_t lut2, int64_t lut3, int64_t smem_bytes, int64_t mode, int64_t dbg);
"""

_EXT = None
LOADER = int(os.environ.get("LATENCY_LOADER", "2"))


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name=f"latency_gemv_v10_l{LOADER}",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            functions=["latency_gemv_run", "latency_gemv_w_run", "latency_stamps"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-std=c++17",
                               f"-DLATENCY_LOADER={LOADER}"],
            verbose=False,
        )
    return _EXT


# rows per CTA per packed stream, and how many stages (8 k64 blocks each) stay resident in smem
SMEM_BUDGET = 96 * 1024


def _perm_bf16(t, n_cta, NB, KS):
    # [N, K] -> [n_cta, stage, block(8), n, c(4), h(2), i(8)]
    return t.view(n_cta, NB, KS, 8, 2, 4, 8).permute(0, 2, 3, 1, 5, 4, 6)


def _encode_planes(W: torch.Tensor, NB: int):
    """12-bit planes of a bf16 [N, K] matrix in CTA/stage order, plus exceptions as (n, k, delta)."""
    N, K = W.shape
    n_cta, KS = N // NB, K // 512
    bits = W.view(torch.int16).to(torch.int32) & 0xFFFF
    E = (bits >> 7) & 0xFF
    Emax = int(E.max())
    hb = max(0, (Emax - 15 + 1) // 2)
    lo_e, hi_e = 2 * hb, 2 * hb + 15
    ok = (E >= lo_e) & (E <= hi_e)
    L = torch.where(ok, bits & 0xFF, torch.zeros_like(bits)).to(torch.uint8)
    Hn = torch.where(ok, ((bits >> 15) << 3) | ((E >> 1) - hb), torch.zeros_like(bits))
    Lp = _perm_bf16(L, n_cta, NB, KS).contiguous().view(n_cta, KS, 8 * NB * 64)
    Hp = _perm_bf16(Hn, n_cta, NB, KS).contiguous().view(n_cta, KS, 8, NB, 4, 8, 2)
    Hb = (Hp[..., 0] | (Hp[..., 1] << 4)).to(torch.uint8).reshape(n_cta, KS, 8 * NB * 32)
    planes = torch.cat([Lp, Hb], dim=2)
    esc = torch.tensor([lo_e << 7], dtype=torch.int16, device=W.device).view(torch.bfloat16).float()
    idx = (~ok).nonzero()
    n_i, k_i = idx[:, 0], idx[:, 1]
    delta = W.float()[n_i, k_i] - esc
    lut = [(hb + s) & 0xFF for s in range(8)]
    lut_lo = lut[0] | (lut[1] << 8) | (lut[2] << 16) | (lut[3] << 24)
    lut_hi = lut[4] | (lut[5] << 8) | (lut[6] << 16) | (lut[7] << 24)
    return planes, (n_i, k_i, delta), (lut_lo, lut_hi), hb


def _exception_table(n_cta, NB, parts, device):
    ns = torch.cat([p[0] for p in parts]) if parts else torch.zeros(0, dtype=torch.int64, device=device)
    ks = torch.cat([p[1] for p in parts]) if parts else torch.zeros(0, dtype=torch.int64, device=device)
    ds = torch.cat([p[2] for p in parts]) if parts else torch.zeros(0, dtype=torch.float32, device=device)
    if ns.numel():
        cta_i = ns // NB
        order = torch.argsort(cta_i, stable=True)
        ns, ks, ds, cta_i = ns[order], ks[order], ds[order], cta_i[order]
        nk = ((ns % NB) | (ks << 16)).to(torch.int32)
        exc = torch.stack([nk, ds.view(torch.int32)], dim=1).contiguous()
        counts = torch.bincount(cta_i, minlength=n_cta)
    else:
        exc = torch.zeros((1, 2), dtype=torch.int32, device=device)
        counts = torch.zeros(n_cta, dtype=torch.int64, device=device)
    exc_off = torch.zeros(n_cta + 1, dtype=torch.int32, device=device)
    exc_off[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return exc, exc_off, int(ns.numel())


def _warp_exception_table(n_cta, NB, parts, device, ksplit=1, KS=1):
    """Per (cta, warp) exception lists, padded to a common even capacity, sorted by k; entry = (n_local | k << 16, delta).
    Warp w of a CTA owns k64 block w of every 512-wide stage. Padding entries carry k = 0xFFFF (stage 127, never hit)."""
    ns = torch.cat([p[0] for p in parts])
    ks = torch.cat([p[1] for p in parts])
    ds = torch.cat([p[2] for p in parts])
    n_cta = n_cta * ksplit
    kq = (ks // 512) // (KS // ksplit) if ksplit > 1 else torch.zeros_like(ks)
    key = ((ns // NB) * ksplit + kq) * 8 + (ks % 512) // 64
    order = torch.argsort(key * 65536 + ks, stable=True)
    ns, ks, ds, key = ns[order], ks[order], ds[order], key[order]
    counts = torch.bincount(key, minlength=n_cta * 8)
    cap = max(2, int(counts.max()) if ns.numel() else 2)
    cap += cap & 1
    start = torch.zeros(n_cta * 8 + 1, dtype=torch.int64, device=device)
    start[1:] = torch.cumsum(counts, 0)
    rank = torch.arange(ns.numel(), device=device) - start[key]
    tab = torch.zeros((n_cta * 8 * cap, 2), dtype=torch.int32, device=device)
    tab[:, 0] = -65536  # 0xFFFF0000: k = 0xFFFF
    pos = key * cap + rank
    tab[pos, 0] = ((ns % NB) | (ks << 16)).to(torch.int64).sub(torch.where(ks >= 32768, 1 << 32, 0)).to(torch.int32)
    tab[pos, 1] = ds.view(torch.int32)
    return tab.view(torch.uint8).reshape(-1).contiguous(), cap


def _meta(packed, exc, exc_off, NB, N, K, lut, mode, n_exc, hb, rowb):
    KS = K // 512
    stage = 8 * NB * rowb
    assert packed.shape[2] == stage
    S = max(1, min(KS, SMEM_BUDGET // stage, 16))
    red = 8 * 32 * ((NB + 7) // 8 * 8) * 4
    return dict(packed=packed, exc=exc, exc_off=exc_off, NB=NB, N=N, K=K, nstages=KS, S=S,
                lut=list(lut) + [0] * (4 - len(lut)), smem=max(S * stage, red), mode=mode, n_exc=n_exc, hb=hb)


def pack_bf16(W: torch.Tensor, NB: int):
    W = W.contiguous()
    N, K = W.shape
    assert N % NB == 0 and K % 512 == 0, (N, K, NB)
    planes, exc_part, lut, hb = _encode_planes(W, NB)
    exc, exc_off, n_exc = _exception_table(N // NB, NB, [exc_part], W.device)
    meta = _meta(planes.contiguous(), exc, exc_off, NB, N, K, lut, 0, n_exc, hb, 96)
    meta["excw"], meta["ecap"] = _warp_exception_table(N // NB, NB, [exc_part], W.device)
    meta["exc_parts"] = [exc_part]
    return meta


def pack_dual(W32: torch.Tensor, NB: int):
    """fp32 [N, K] as a bf16 hi + lo pair, each 12-bit packed (the router)."""
    W32 = W32.contiguous()
    N, K = W32.shape
    hi = W32.to(torch.bfloat16)
    lo = (W32 - hi.float()).to(torch.bfloat16)
    p_hi, e_hi, lut_hi, hb_hi = _encode_planes(hi, NB)
    p_lo, e_lo, lut_lo, hb_lo = _encode_planes(lo, NB)
    exc, exc_off, n_exc = _exception_table(N // NB, NB, [e_hi, e_lo], W32.device)
    packed = torch.cat([p_hi, p_lo], dim=2).contiguous()
    meta = _meta(packed, exc, exc_off, NB, N, K, list(lut_hi) + list(lut_lo), 2, n_exc, (hb_hi, hb_lo), 192)
    meta["excw"], meta["ecap"] = _warp_exception_table(N // NB, NB, [e_hi, e_lo], W32.device)
    meta["exc_parts"] = [e_hi, e_lo]
    return meta


def pack_f32(W: torch.Tensor, NB: int):
    W = W.contiguous()
    N, K = W.shape
    n_cta, KS = N // NB, K // 512
    # [n_cta, stage, block, n, h, c, i]
    P = W.view(n_cta, NB, KS, 8, 2, 4, 8).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    packed = P.view(torch.uint8).view(n_cta, KS, 8 * NB * 256)
    dev = W.device
    return _meta(packed, torch.zeros((1, 2), dtype=torch.int32, device=dev),
                 torch.zeros(n_cta + 1, dtype=torch.int32, device=dev), NB, N, K, (0, 0), 1, 0, 0, 256)


NB_OF = {"qkv_gate": 18, "o": 64, "pooler": 32, "unpooler": 64, "unpooler_slice": 8, "router": 2}
ROUTER_MODE = 2  # 1: fp32 weights split in-kernel; 2: 12-bit packed bf16 hi + lo planes
MAX_M = 32

# LOADER 2 plan: per layer, (max m, NB, ksplit) in increasing m. ksplit > 1 = a cluster of ksplit CTAs splitting K
# for one n-tile (cuts the per-SM x stream by ksplit), reduced once through DSMEM.
PLAN = {
    "qkv_gate": [(32, 18, 1)],
    "o": [(32, 64, 1)],
    "pooler": [(32, 32, 1)],
    "unpooler": [(32, 64, 1)],
    "unpooler_slice": [(32, 8, 1)],
    "router": [(32, 2, 1)],
}


def _parse_plan(spec):
    # "qkv_gate=8:18x1,32:72x4;router=32:16x8"
    for part in filter(None, spec.split(";")):
        layer, rest = part.split("=")
        ent = []
        for e in rest.split(","):
            mm, cfg = e.split(":")
            nb, ks = cfg.split("x")
            ent.append((int(mm), int(nb), int(ks)))
        PLAN[layer] = sorted(ent)


_parse_plan(os.environ.get("LATENCY_PLAN", ""))


def _src_weight(weights, key):
    return weights["unpooler"][:1024] if key == "unpooler_slice" else weights[key]


def prepare(weights: Weights):
    ext = _ext()
    state = {"ext": ext, "orig": {k: v.contiguous() for k, v in weights.items()}, "packs": {}}
    for key, plan in PLAN.items():
        for _, nb, ks in plan:
            if (key, nb) not in state["packs"]:
                fn = pack_dual if key == "router" else pack_bf16
                state["packs"][(key, nb)] = fn(_src_weight(weights, key), nb)
            p = state["packs"][(key, nb)]
            if "packed_w" not in p:
                p["packed_w"] = _warp_layout(p)
            if ks not in p.setdefault("excw_ks", {}):
                p["excw_ks"][ks] = _warp_exception_table(p["N"] // nb, nb, p["exc_parts"], p["packed"].device, ks, p["nstages"])
    for key in NB_OF:
        state[key] = state["packs"].get((key, NB_OF[key])) or (pack_dual if key == "router" else pack_bf16)(_src_weight(weights, key), NB_OF[key])
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    for (key, nb), p in state["packs"].items():
        print(f"PACK {key} NB={nb}: hb={p['hb']} exceptions={p['n_exc']} ecap={ {k: v[1] for k, v in p['excw_ks'].items()} }")
    print(f"PLAN {PLAN}")
    return state


W_SMEM = 200 * 1024   # LOADER 2 smem budget (ring + exception lists); the epilogue reduction reuses it
W_SMAX = int(os.environ.get("LATENCY_SMAX", "16"))
DBG = int(os.environ.get("LATENCY_DBG", "0"))  # timing experiments only: 1 no x, 2 no math, 4 no exceptions


def _w_geometry(p, m, ks, ecap):
    wb = p["NB"] * 96 * (2 if p["mode"] == 2 else 1)
    slot = (wb + m * 128 + 127) // 128 * 128
    eb = 8 * ecap * 8
    S = max(1, min(p["nstages"] // ks, W_SMAX, (W_SMEM - eb) // (8 * slot)))
    mt = 2 if m > 16 else 1
    red = 8 * (16 * mt) * ((p["NB"] + 7) // 8 * 8) * 4 + (m * p["NB"] * 4 if ks > 1 else 0)
    return S, max(eb + 8 * S * slot, red)


WLAYOUT = int(os.environ.get("LATENCY_WLAYOUT", "1"))


def _warp_layout(p):
    """[n_cta, KS, pl x (L 8 blocks | H 8 blocks)] -> [n_cta, warp, KS, pl x (L | H)]: each warp's stream contiguous."""
    NB, packed = p["NB"], p["packed"]
    n_cta, KS, stage = packed.shape
    npl = stage // (8 * NB * 96)
    parts = []
    for pl in range(npl):
        q = packed[:, :, pl * 8 * NB * 96:(pl + 1) * 8 * NB * 96]
        L = q[:, :, :8 * NB * 64].reshape(n_cta, KS, 8, NB * 64)
        H = q[:, :, 8 * NB * 64:].reshape(n_cta, KS, 8, NB * 32)
        parts += [L, H]
    return torch.cat(parts, dim=3).permute(0, 2, 1, 3).contiguous()


def _run(state, key, x):
    m = x.shape[0]
    if LOADER == 2:
        for mm, nb, ks in PLAN[key]:
            if m <= mm:
                break
        p = state["packs"][(key, nb)]
        excw, ecap = p["excw_ks"][ks]
        if WLAYOUT:
            S, smem = _w_geometry(p, m, ks, ecap)
            return state["ext"].latency_gemv_w_run(p["packed_w"], x, excw, ecap, ks, nb, p["N"], p["nstages"], S,
                                                    p["lut"][0], p["lut"][1], p["lut"][2], p["lut"][3], smem, p["mode"],
                                                    DBG | 8)
        S, smem = _w_geometry(p, m, ks, ecap)
        return state["ext"].latency_gemv_w_run(p["packed"], x, excw, ecap, ks, nb, p["N"], p["nstages"], S,
                                                p["lut"][0], p["lut"][1], p["lut"][2], p["lut"][3], smem, p["mode"], DBG)
    p = state[key]
    return state["ext"].latency_gemv_run(p["packed"], x, p["exc"], p["exc_off"], p["NB"], p["N"], p["nstages"],
                                          p["S"], p["lut"][0], p["lut"][1], p["lut"][2], p["lut"][3], p["smem"], p["mode"])


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    if not x.is_contiguous():
        x = x.contiguous()
    if m <= MAX_M:
        key = layer
        if layer == "unpooler" and n == 1024:
            key = "unpooler_slice"
        return _run(state, key, x)
    w = state["orig"][layer]
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return torch.mm(x, w[:n].t())
