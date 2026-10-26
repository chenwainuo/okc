# okc: problem=muse_bf16_proj
"""bandwidth: lossless 12-bit weight stream + register-decoded mma.sync, for every load.

Encoding (prepare, GPU, lossless): a bf16 weight is sign(1) + exponent(8) + mantissa(7).  The weights are
N(0, 1/sqrt(K)) so the exponent takes one of ~16 consecutive values; we store a 4-bit exponent code
(e - base) plus the 8 other bits: 12 bits per weight, 25% fewer HBM bytes.  Weights whose exponent falls
outside the 16-code window (~7e-5 of them) are stored as a placeholder and fixed exactly in the epilogue
from a per-column escape list, so every output is bit-identical to using the bf16 weights.  The router's
fp32 weights store their top 16 bits the same way plus the low 16 mantissa bits verbatim (28 bits/weight).

The stream is laid out in mma.sync B-fragment order so a thread's 16-byte loads are exactly its fragment
bytes: decode is 5 integer ops per bf16x2 (prmt / and / shift+and / iadd3).

Kernels:
  k_small  m <= 32: one CTA per 64 output columns x K-slice (cluster split-K reduced through DSMEM),
           8 warps split the CTA's K range, B fragments decoded straight from registers, A via ldmatrix
           from a per-warp cp.async ring.  Bandwidth-bound at the 12-bit byte count.
  k_rfma   router m <= 8: same stream decoded to exact fp32, FMA.
  k_big    33 <= m <= 256: CTA tile 16*MT rows x 128 cols; weights decoded once per k16 step into a
           shared B tile; 4x2 warps run mma.sync; x tile via cp.async; cluster + global split-K.
  torch    m > 256 (and router 33..64).
"""

import math
import os

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <set>
#include <map>
#include <cstdio>
#include <tuple>
#include <utility>
namespace cg = cooperative_groups;

struct Esc { int k; float wt; float wd; int pad; };

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define BW_STUB 1
#endif

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(a), "r"(sel)); return r;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* g, int src_bytes) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(g), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(g));
}
template <int NB> __device__ __forceinline__ void copy_bytes(uint8_t* dst, const uint8_t* src) {
    static_assert(NB == 4 || NB == 8 || NB % 16 == 0, "copy size");
    if (NB == 4) cp_async4(dst, src);
    else if (NB == 8) cp_async8(dst, src);
    else {
#pragma unroll
        for (int i = 0; i < NB / 16; ++i) cp_async16(dst + 16 * i, src + 16 * i, 16);
    }
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* a, const void* p) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

// exponent nibble pair p (0..3) of an E word -> bf16x2 exponent field (<<7)
template <int P> __device__ __forceinline__ uint32_t dec_ex(uint32_t ew) {
    if (P == 0) return (ew << 7) & 0x07800780u;
    if (P == 1) return (ew << 3) & 0x07800780u;
    if (P == 2) return (ew >> 1) & 0x07800780u;
    return (ew >> 5) & 0x07800780u;
}
// SM word = bytes (k0,k1,k8,k9) each sign<<7|mantissa; HI selects (k8,k9).  Result: bf16x2.
template <bool HI> __device__ __forceinline__ uint32_t dec_pair(uint32_t smw, uint32_t ex, uint32_t base2) {
    uint32_t t = prmt(smw, HI ? 0xB3A2u : 0x9180u);   // byte, signrep(byte) per halfword
    return (t & 0x807F807Fu) + ex + base2;
}
// router: bf16x2 hi + packed low-16 mantissa bits -> bf16x2 lo = rn(w - hi)
__device__ __forceinline__ uint32_t lo_pair(uint32_t b, uint32_t lo) {
    float f0 = __uint_as_float(b << 16), f1 = __uint_as_float(b & 0xFFFF0000u);
    float w0 = __uint_as_float((b << 16) | (lo & 0xFFFFu)), w1 = __uint_as_float((b & 0xFFFF0000u) | (lo >> 16));
    __nv_bfloat162 l = __floats2bfloat162_rn(w0 - f0, w1 - f1);
    return *reinterpret_cast<uint32_t*>(&l);
}
__device__ __forceinline__ float bf16_to_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float fix_esc_bf16(float v, float xv, const Esc& e) {
    v = fmaf(-xv, e.wd, v); return fmaf(xv, e.wt, v);
}
__device__ __forceinline__ float fix_esc_hilo(float v, float xv, const Esc& e) {
    float hd = __uint_as_float(__float_as_uint(e.wd) & 0xFFFF0000u), ld = __bfloat162float(__float2bfloat16_rn(e.wd - hd));
    float ht = __uint_as_float(__float_as_uint(e.wt) & 0xFFFF0000u), lt = __bfloat162float(__float2bfloat16_rn(e.wt - ht));
    v = fmaf(-xv, hd, v); v = fmaf(-xv, ld, v); v = fmaf(xv, ht, v); return fmaf(xv, lt, v);
}

// ---------------------------------------------------------------------------------------------
// k_small: m <= 16*MT (MT = 1, 2).  grid = NBLK * S, cluster S along K.  8 warps split the CTA's K range.
// ---------------------------------------------------------------------------------------------
template <int NT, int MT, bool ROUTER, int D>
__global__ void __launch_bounds__(256, 1) k_small(const uint8_t* __restrict__ packed, const __nv_bfloat16* __restrict__ x,
                                                  void* __restrict__ out, const int* __restrict__ esc_off,
                                                  const Esc* __restrict__ esc, int m, int K, int N, uint32_t base2, int S, int diag) {
#ifndef BW_STUB
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int BN = 8 * NT, SMB = 4 * NT, EB = 2 * NT, LOB = ROUTER ? 8 * NT : 0;
    constexpr int STEP = 32 * (SMB + EB + LOB), ROWS = 16 * MT, XB = ROWS * 32, STAGE = STEP + XB;
    const bool no_x = diag & 1, no_mma = diag & 2, no_epi = diag & 4, no_clu = diag & 8;
    static_assert(9 * ROWS * BN * 4 <= 8 * D * STAGE, "warp partials + P fit in the ring");
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int r = blockIdx.x % S, nb = blockIdx.x / S;
    const int KS = K >> 4;
    const int wi = r * 8 + w;
    const int s0 = (int)(((long long)wi * KS) / (S * 8)), s1 = (int)(((long long)(wi + 1) * KS) / (S * 8));
    const int nsteps = s1 - s0;
    const uint8_t* src = packed + ((size_t)nb * KS + s0) * STEP;
    uint8_t* ring = smem + w * (D * STAGE);
    float* P = reinterpret_cast<float*>(smem) + 8 * ROWS * BN;   // [ROWS][BN], aliases the dead ring

    float acc[MT][NT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int b = 0; b < NT; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

    auto issue = [&](int s) {
        uint8_t* st = ring + (s % D) * STAGE;
        const uint8_t* gs = src + (size_t)s * STEP;
        copy_bytes<SMB>(st + lane * SMB, gs + lane * SMB);
        copy_bytes<EB>(st + 32 * SMB + lane * EB, gs + 32 * SMB + lane * EB);
        if (ROUTER) copy_bytes<(LOB ? LOB : 16)>(st + 32 * (SMB + EB) + lane * LOB, gs + 32 * (SMB + EB) + lane * LOB);
        if (no_x) return;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            int row = 16 * i + (lane >> 1), chunk = lane & 1;
            const __nv_bfloat16* gx = x + (size_t)(row < m ? row : 0) * K + (size_t)(s0 + s) * 16 + chunk * 8;
            cp_async16(st + STEP + row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4), gx, row < m ? 16 : 0);
        }
    };
#pragma unroll 1
    for (int s = 0; s < D - 1; ++s) { if (s < nsteps) issue(s); cp_commit(); }

#pragma unroll 1
    for (int s = 0; s < nsteps; ++s) {
        cp_wait<D - 2>();
        __syncwarp();
        const uint8_t* st = ring + (s % D) * STAGE;
        if (no_mma) { acc[0][0][0] += __uint_as_float(*reinterpret_cast<const uint32_t*>(st + lane * SMB)); __syncwarp(); if (s + D - 1 < nsteps) issue(s + D - 1); cp_commit(); continue; }
        uint32_t a[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            int row = 16 * mt + (lane & 15), chunk = (lane >> 4) ^ ((lane >> 2) & 1);
            ldmatrix_x4(a[mt], st + STEP + row * 32 + (chunk << 4));
        }
        const uint32_t* smw = reinterpret_cast<const uint32_t*>(st + lane * SMB);
        const uint32_t* ew = reinterpret_cast<const uint32_t*>(st + 32 * SMB + lane * EB);
        const uint32_t* low = reinterpret_cast<const uint32_t*>(st + 32 * (SMB + EB) + lane * LOB);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t sw = smw[j], e = ew[j >> 1];
            uint32_t ex0 = (j & 1) ? dec_ex<2>(e) : dec_ex<0>(e);
            uint32_t ex1 = (j & 1) ? dec_ex<3>(e) : dec_ex<1>(e);
            uint32_t b0 = dec_pair<false>(sw, ex0, base2), b1 = dec_pair<true>(sw, ex1, base2);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][j], a[mt], b0, b1);
            if (ROUTER) {
                uint32_t l0 = lo_pair(b0, low[2 * j]), l1 = lo_pair(b1, low[2 * j + 1]);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][j], a[mt], l0, l1);
            }
        }
        __syncwarp();
        if (s + D - 1 < nsteps) issue(s + D - 1);
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
    if (no_epi) {
        if (w == 0) { if (lane < 8) reinterpret_cast<__nv_bfloat16*>(out)[(size_t)nb * BN + r * 8 + lane] = __float2bfloat16_rn(acc[0][0][0]); }
        if (S > 1) cg::this_cluster().sync();
        return;
    }
    // per-warp partials -> P
    float* red = reinterpret_cast<float*>(smem) + w * (ROWS * BN);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int row = 16 * mt + g, col = 8 * j + 2 * q;
            *reinterpret_cast<float2*>(red + row * BN + col) = make_float2(acc[mt][j][0], acc[mt][j][1]);
            *reinterpret_cast<float2*>(red + (row + 8) * BN + col) = make_float2(acc[mt][j][2], acc[mt][j][3]);
        }
    __syncthreads();
    for (int idx = tid; idx < ROWS * BN; idx += 256) {
        float v = 0.f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww) v += reinterpret_cast<const float*>(smem)[ww * ROWS * BN + idx];
        P[idx] = v;
    }
    cg::cluster_group cluster = cg::this_cluster();
    if (S > 1) cluster.sync(); else __syncthreads();
    const int c0 = (r * BN) / S, c1 = ((r + 1) * BN) / S, ncols = c1 - c0;
    for (int idx = tid; idx < ROWS * ncols; idx += 256) {
        int c = c0 + idx % ncols, row = idx / ncols;
        if (row >= m) continue;
        float v = 0.f;
        if (S > 1 && !no_clu) {
            for (int rr = 0; rr < S; ++rr) v += cluster.map_shared_rank(P, rr)[row * BN + c];
        } else v = P[row * BN + c];
        int n = nb * BN + c;
        int e0 = __ldg(esc_off + n), e1 = __ldg(esc_off + n + 1);
        for (int e = e0; e < e1; ++e) {
            Esc E = esc[e];
            float xv = bf16_to_f(x + (size_t)row * K + E.k);
            v = ROUTER ? fix_esc_hilo(v, xv, E) : fix_esc_bf16(v, xv, E);
        }
        if (ROUTER) reinterpret_cast<float*>(out)[(size_t)row * N + n] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)row * N + n] = __float2bfloat16_rn(v);
    }
    if (S > 1) cluster.sync();
#endif
}

// ---------------------------------------------------------------------------------------------
// k_rfma: router, m <= 8, exact fp32 FMA.  grid = NBLK * S (NT = 8, BN = 64).
// ---------------------------------------------------------------------------------------------
template <int D>
__global__ void __launch_bounds__(256, 1) k_rfma(const uint8_t* __restrict__ packed, const __nv_bfloat16* __restrict__ x,
                                                 float* __restrict__ out, const int* __restrict__ esc_off,
                                                 const Esc* __restrict__ esc, int m, int K, int N, uint32_t base2, int S) {
#ifndef BW_STUB
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int NT = 8, BN = 64, SMB = 32, EB = 16, LOB = 64, STEP = 32 * (SMB + EB + LOB), STAGE = STEP, ROWS = 8;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int r = blockIdx.x % S, nb = blockIdx.x / S;
    const int KS = K >> 4;
    const int cs0 = (int)(((long long)r * KS) / S), cs1 = (int)(((long long)(r + 1) * KS) / S);
    const int wi = r * 8 + w;
    const int s0 = (int)(((long long)wi * KS) / (S * 8)), s1 = (int)(((long long)(wi + 1) * KS) / (S * 8));
    const int nsteps = s1 - s0;
    const int KC = (cs1 - cs0) * 16;   // k values of this CTA
    const uint8_t* src = packed + ((size_t)nb * KS + s0) * STEP;
    uint8_t* ring = smem + w * (D * STAGE);
    float* P = reinterpret_cast<float*>(smem + 8 * D * STAGE);          // [8][BN]
    float* xs = P + ROWS * BN;                                           // [8][KC] fp32

    auto issue = [&](int s) {
        uint8_t* st = ring + (s % D) * STAGE;
        const uint8_t* gs = src + (size_t)s * STEP;
        copy_bytes<SMB>(st + lane * SMB, gs + lane * SMB);
        copy_bytes<EB>(st + 32 * SMB + lane * EB, gs + 32 * SMB + lane * EB);
        copy_bytes<LOB>(st + 32 * (SMB + EB) + lane * LOB, gs + 32 * (SMB + EB) + lane * LOB);
    };
#pragma unroll 1
    for (int s = 0; s < D - 1; ++s) { if (s < nsteps) issue(s); cp_commit(); }
    // stage x (rows >= m zero)
    for (int idx = tid; idx < ROWS * KC; idx += 256) {
        int row = idx / KC, kk = idx % KC;
        xs[idx] = row < m ? __bfloat162float(x[(size_t)row * K + cs0 * 16 + kk]) : 0.f;
    }
    __syncthreads();
    float acc[ROWS][NT];
#pragma unroll
    for (int a = 0; a < ROWS; ++a)
#pragma unroll
        for (int b = 0; b < NT; ++b) acc[a][b] = 0.f;
    const int kbase = (s0 - cs0) * 16 + 2 * q;
#pragma unroll 1
    for (int s = 0; s < nsteps; ++s) {
        cp_wait<D - 2>();
        const uint8_t* st = ring + (s % D) * STAGE;
        const uint32_t* smw = reinterpret_cast<const uint32_t*>(st + lane * SMB);
        const uint32_t* ew = reinterpret_cast<const uint32_t*>(st + 32 * SMB + lane * EB);
        const uint32_t* low = reinterpret_cast<const uint32_t*>(st + 32 * (SMB + EB) + lane * LOB);
        float xv[ROWS][4];
#pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            float2 p0 = *reinterpret_cast<const float2*>(xs + row * KC + kbase + s * 16);
            float2 p1 = *reinterpret_cast<const float2*>(xs + row * KC + kbase + s * 16 + 8);
            xv[row][0] = p0.x; xv[row][1] = p0.y; xv[row][2] = p1.x; xv[row][3] = p1.y;
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t sw = smw[j], e = ew[j >> 1];
            uint32_t ex0 = (j & 1) ? dec_ex<2>(e) : dec_ex<0>(e);
            uint32_t ex1 = (j & 1) ? dec_ex<3>(e) : dec_ex<1>(e);
            uint32_t b0 = dec_pair<false>(sw, ex0, base2), b1 = dec_pair<true>(sw, ex1, base2);
            uint32_t l01 = low[2 * j], l89 = low[2 * j + 1];
            float wv[4];
            wv[0] = __uint_as_float((b0 << 16) | (l01 & 0xFFFFu));
            wv[1] = __uint_as_float((b0 & 0xFFFF0000u) | (l01 >> 16));
            wv[2] = __uint_as_float((b1 << 16) | (l89 & 0xFFFFu));
            wv[3] = __uint_as_float((b1 & 0xFFFF0000u) | (l89 >> 16));
#pragma unroll
            for (int row = 0; row < ROWS; ++row)
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) acc[row][j] = fmaf(xv[row][kk], wv[kk], acc[row][j]);
        }
        __syncwarp();
        if (s + D - 1 < nsteps) issue(s + D - 1);
        cp_commit();
    }
    cp_wait<0>();
    // reduce over q lanes
#pragma unroll
    for (int row = 0; row < ROWS; ++row)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = acc[row][j];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            acc[row][j] = v;
        }
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem) + w * (ROWS * BN);   // reuse the ring
    if (q == 0) {
#pragma unroll
        for (int row = 0; row < ROWS; ++row)
#pragma unroll
            for (int j = 0; j < NT; ++j) red[row * BN + 8 * j + g] = acc[row][j];
    }
    __syncthreads();
    for (int idx = tid; idx < ROWS * BN; idx += 256) {
        float v = 0.f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww) v += reinterpret_cast<const float*>(smem)[ww * ROWS * BN + idx];
        P[idx] = v;
    }
    cg::cluster_group cluster = cg::this_cluster();
    if (S > 1) cluster.sync(); else __syncthreads();
    const int c0 = (r * BN) / S, c1 = ((r + 1) * BN) / S, ncols = c1 - c0;
    for (int idx = tid; idx < ROWS * ncols; idx += 256) {
        int c = c0 + idx % ncols, row = idx / ncols;
        if (row >= m) continue;
        float v = 0.f;
        if (S > 1) {
            for (int rr = 0; rr < S; ++rr) v += cluster.map_shared_rank(P, rr)[row * BN + c];
        } else v = P[row * BN + c];
        int n = nb * BN + c;
        int e0 = __ldg(esc_off + n), e1 = __ldg(esc_off + n + 1);
        for (int e = e0; e < e1; ++e) {
            Esc E = esc[e];
            float xv = bf16_to_f(x + (size_t)row * K + E.k);
            v = fix_esc_bf16(v, xv, E);
        }
        out[(size_t)row * N + n] = v;
    }
    if (S > 1) cluster.sync();
#endif
}

// ---------------------------------------------------------------------------------------------
// k_big: 16*MT rows x (64*NB2) cols per CTA; grid = NBLK * G * S; cluster S along K, G outer K split.
// Warps: wm = w&3 (m quarter), wn = w>>2 (n half).  Weights decoded once per step into a shared B tile.
// ---------------------------------------------------------------------------------------------
template <int MT, int NB2, bool ROUTER, int D>
__global__ void __launch_bounds__(256, 1) k_big(const uint8_t* __restrict__ packed, const __nv_bfloat16* __restrict__ x,
                                                void* __restrict__ out, const int* __restrict__ esc_off,
                                                const Esc* __restrict__ esc, float* __restrict__ scratch,
                                                int* __restrict__ counters, int m, int K, int N, uint32_t base2, int S, int G) {
#ifndef BW_STUB
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int NT = 8, BN = 64 * NB2, SMB = 32, EB = 16, LOB = ROUTER ? 64 : 0;
    constexpr int STEP = 32 * (SMB + EB + LOB);                  // one packed n-block step
    constexpr int ROWS = 16 * MT, XB = ROWS * 32, WB = NB2 * STEP, STAGE = WB + XB;
    constexpr int BT = NB2 * 8 * 32 * 2 * 4;                      // B tile bytes (bf16x2 b0,b1 per lane per tile)
    constexpr int NBT = ROUTER ? 2 : 1;                           // hi (+lo) tiles
    constexpr int MTW = MT / 4, NTW = 4 * NB2;                    // per-warp m-tiles, n-tiles
    static_assert(MT % 4 == 0, "MT multiple of 4");
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int wm = w & 3, wn = w >> 2;
    const int r = blockIdx.x % S, gi = (blockIdx.x / S) % G, nb = blockIdx.x / (S * G);
    const int KS = K >> 4;
    const int ci = gi * S + r;
    const int s0 = (int)(((long long)ci * KS) / (S * G)), s1 = (int)(((long long)(ci + 1) * KS) / (S * G));
    const int nsteps = s1 - s0;
    static_assert(D * STAGE >= ROWS * BN * 4, "partial tile aliases the ring");
    uint8_t* ring = smem;
    uint8_t* btile = smem + D * STAGE;                             // [2][NBT][NB2][8][32] uint2
    float* P = reinterpret_cast<float*>(smem);                     // [ROWS][BN], aliases the (dead) ring

    auto issue = [&](int s) {
        uint8_t* st = ring + (s % D) * STAGE;
        for (int c = tid; c < NB2 * (STEP / 16); c += 256) {
            int b = c / (STEP / 16), off = (c % (STEP / 16)) * 16;
            const uint8_t* gs = packed + ((size_t)(nb * NB2 + b) * KS + s0 + s) * STEP + off;
            cp_async16(st + b * STEP + off, gs, 16);
        }
        for (int c = tid; c < ROWS * 2; c += 256) {
            int row = c >> 1, chunk = c & 1;
            const __nv_bfloat16* gx = x + (size_t)(row < m ? row : 0) * K + (size_t)(s0 + s) * 16 + chunk * 8;
            cp_async16(st + WB + row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4), gx, row < m ? 16 : 0);
        }
    };
    auto decode = [&](int s) {
        const uint8_t* st = ring + (s % D) * STAGE;
        uint2* bt = reinterpret_cast<uint2*>(btile + (s & 1) * NBT * BT);
        const int j = w;   // this warp decodes n-tile j of every packed n-block, lane = fragment lane
#pragma unroll
        for (int b = 0; b < NB2; ++b) {
            const uint8_t* wst = st + b * STEP;
            uint32_t sw = reinterpret_cast<const uint32_t*>(wst + lane * SMB)[j];
            uint32_t e = reinterpret_cast<const uint32_t*>(wst + 32 * SMB + lane * EB)[j >> 1];
            uint32_t ex0 = (j & 1) ? dec_ex<2>(e) : dec_ex<0>(e);
            uint32_t ex1 = (j & 1) ? dec_ex<3>(e) : dec_ex<1>(e);
            uint32_t b0 = dec_pair<false>(sw, ex0, base2), b1 = dec_pair<true>(sw, ex1, base2);
            bt[(b * 8 + j) * 32 + lane] = make_uint2(b0, b1);
            if (ROUTER) {
                const uint32_t* low = reinterpret_cast<const uint32_t*>(wst + 32 * (SMB + EB) + lane * LOB);
                uint32_t l0 = lo_pair(b0, low[2 * j]), l1 = lo_pair(b1, low[2 * j + 1]);
                bt[NB2 * 8 * 32 + (b * 8 + j) * 32 + lane] = make_uint2(l0, l1);
            }
        }
    };

    float acc[MTW][NTW][4];
#pragma unroll
    for (int a = 0; a < MTW; ++a)
#pragma unroll
        for (int b = 0; b < NTW; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll 1
    for (int s = 0; s < D - 1; ++s) { if (s < nsteps) issue(s); cp_commit(); }
    cp_wait<D - 2>();
    __syncthreads();
    if (nsteps > 0) decode(0);

#pragma unroll 1
    for (int s = 0; s < nsteps; ++s) {
        cp_wait<D - 3>();
        __syncthreads();
        if (s + 1 < nsteps) decode(s + 1);
        if (s + D - 1 < nsteps) issue(s + D - 1);
        cp_commit();
        const uint8_t* st = ring + (s % D) * STAGE;
        const uint2* bt = reinterpret_cast<const uint2*>(btile + (s & 1) * NBT * BT);
        uint32_t a[MTW][4];
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) {
            int row = 16 * (wm * MTW + mt) + (lane & 15), chunk = (lane >> 4) ^ ((lane >> 2) & 1);
            ldmatrix_x4(a[mt], st + WB + row * 32 + (chunk << 4));
        }
#pragma unroll
        for (int jj = 0; jj < NTW; ++jj) {
            int J = wn * NTW + jj;
            uint2 bb = bt[J * 32 + lane];
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt) mma16816(acc[mt][jj], a[mt], bb.x, bb.y);
            if (ROUTER) {
                uint2 ll = bt[NB2 * 8 * 32 + J * 32 + lane];
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt) mma16816(acc[mt][jj], a[mt], ll.x, ll.y);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
        for (int jj = 0; jj < NTW; ++jj) {
            int row = 16 * (wm * MTW + mt) + g, col = 8 * (wn * NTW + jj) + 2 * q;
            *reinterpret_cast<float2*>(P + row * BN + col) = make_float2(acc[mt][jj][0], acc[mt][jj][1]);
            *reinterpret_cast<float2*>(P + (row + 8) * BN + col) = make_float2(acc[mt][jj][2], acc[mt][jj][3]);
        }
    cg::cluster_group cluster = cg::this_cluster();
    if (S > 1) cluster.sync(); else __syncthreads();
    const int c0 = (r * BN) / S, c1 = ((r + 1) * BN) / S, ncols = c1 - c0;
    const int NBLK_S = (int)gridDim.x / (S * G);
    float* myscr = scratch + ((size_t)(nb * G + gi) * ROWS) * BN;      // [ROWS][BN] slice of (nb, gi)
    __shared__ int ticket_s;
    if (G > 1) {
        for (int idx = tid; idx < ROWS * ncols; idx += 256) {
            int c = c0 + idx % ncols, row = idx / ncols;
            float v = 0.f;
            for (int rr = 0; rr < S; ++rr) v += (S > 1 ? cluster.map_shared_rank(P, rr) : P)[row * BN + c];
            myscr[row * BN + c] = v;
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) ticket_s = atomicAdd(counters + nb * S + r, 1);
        __syncthreads();
        if (ticket_s != G - 1) { if (S > 1) cluster.sync(); return; }
        __threadfence();
        if (tid == 0) counters[nb * S + r] = 0;
    }
    (void)NBLK_S;
    for (int idx = tid; idx < ROWS * ncols; idx += 256) {
        int c = c0 + idx % ncols, row = idx / ncols;
        if (row >= m) continue;
        float v = 0.f;
        if (G > 1) {
            for (int gg = 0; gg < G; ++gg) v += __ldcg(scratch + ((size_t)(nb * G + gg) * ROWS + row) * BN + c);
        } else if (S > 1) {
            for (int rr = 0; rr < S; ++rr) v += cluster.map_shared_rank(P, rr)[row * BN + c];
        } else v = P[row * BN + c];
        int n = nb * BN + c;
        int e0 = __ldg(esc_off + n), e1 = __ldg(esc_off + n + 1);
        for (int e = e0; e < e1; ++e) {
            Esc E = esc[e];
            float xv = bf16_to_f(x + (size_t)row * K + E.k);
            v = ROUTER ? fix_esc_hilo(v, xv, E) : fix_esc_bf16(v, xv, E);
        }
        if (ROUTER) reinterpret_cast<float*>(out)[(size_t)row * N + n] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)row * N + n] = __float2bfloat16_rn(v);
    }
    if (S > 1) cluster.sync();
#endif
}

// ---------------------------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------------------------
template <typename... Args>
static void launch(void (*kern)(Args...), int grid, int smem, int S, cudaStream_t stream, Args... args) {
    static std::map<void*, int> max_smem;
    static std::set<std::tuple<void*, int, int, int>> seen;
    if (max_smem[(void*)kern] < smem) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        max_smem[(void*)kern] = smem;
    }
    if (S > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = S; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = (S > 1) ? 1 : 0;
    auto key = std::make_tuple((void*)kern, smem, S, grid);
    if (!seen.count(key)) {
        seen.insert(key);
        int nclu = -1, nblk = -1;
        cudaOccupancyMaxActiveClusters(&nclu, kern, &cfg);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, kern, 256, smem);
        cudaGetLastError();
        printf("CFG grid=%d S=%d smem=%d blocks/SM=%d maxActiveClusters=%d resident=%d waves=%.2f\n", grid, S, smem, nblk, nclu,
               nclu * S, (double)grid / (double)(nclu > 0 ? nclu * S : 1));
    }
    cudaError_t err = cudaLaunchKernelEx(&cfg, kern, args...);
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

void small_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                  int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t MT, bool router, int64_t D, int64_t diag) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int NT = 8, BN = 64;
    int NBLK = (int)N / BN;
    int grid = NBLK * (int)S;
    auto args = std::make_tuple((const uint8_t*)packed.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (void*)out.data_ptr(),
                                (const int*)esc_off.data_ptr(), (const Esc*)esc.data_ptr(), (int)m, (int)K, (int)N, (uint32_t)base2, (int)S, (int)diag);
#define SMALL_CASE(MT_, R_, D_)                                                                                   \
    if (MT == MT_ && router == R_ && D == D_) {                                                                   \
        constexpr int STEP = 32 * (4 * NT + 2 * NT + (R_ ? 8 * NT : 0)), STAGE = STEP + 16 * MT_ * 32;           \
        int smem = 8 * D_ * STAGE;                                                                               \
        std::apply([&](auto... a) { launch(k_small<NT, MT_, R_, D_>, grid, smem, (int)S, stream.stream(), a...); }, args); \
        return;                                                                                                  \
    }
    SMALL_CASE(1, false, 4) SMALL_CASE(2, false, 4) SMALL_CASE(1, false, 6) SMALL_CASE(2, false, 6) SMALL_CASE(1, false, 8) SMALL_CASE(2, false, 8)
    SMALL_CASE(1, true, 4) SMALL_CASE(2, true, 4)
    TORCH_CHECK(false, "no small config");
}

void rfma_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                 int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t D) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BN = 64, STEP = 32 * (32 + 16 + 64);
    int NBLK = (int)N / BN, grid = NBLK * (int)S;
    int KS = (int)K / 16;
    int KC = ((KS + (int)S - 1) / (int)S) * 16;
    auto args = std::make_tuple((const uint8_t*)packed.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (float*)out.data_ptr(),
                                (const int*)esc_off.data_ptr(), (const Esc*)esc.data_ptr(), (int)m, (int)K, (int)N, (uint32_t)base2, (int)S);
#define RFMA_CASE(D_)                                                                                            \
    if (D == D_) {                                                                                               \
        int smem = 8 * D_ * STEP + 8 * BN * 4 + 8 * KC * 4;                                                      \
        std::apply([&](auto... a) { launch(k_rfma<D_>, grid, smem, (int)S, stream.stream(), a...); }, args);   \
        return;                                                                                                  \
    }
    RFMA_CASE(3) RFMA_CASE(4) RFMA_CASE(6)
    TORCH_CHECK(false, "no rfma config");
}

void big_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                torch::Tensor scratch, torch::Tensor counters,
                int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t G, int64_t MT, int64_t NB2, bool router, int64_t D) {
    auto stream = at::cuda::getCurrentCUDAStream();
    int BN = 64 * (int)NB2;
    int NBLK = (int)N / BN, grid = NBLK * (int)G * (int)S;
    auto args = std::make_tuple((const uint8_t*)packed.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (void*)out.data_ptr(),
                                (const int*)esc_off.data_ptr(), (const Esc*)esc.data_ptr(), (float*)scratch.data_ptr(),
                                (int*)counters.data_ptr(), (int)m, (int)K, (int)N, (uint32_t)base2, (int)S, (int)G);
#define BIG_CASE(MT_, NB2_, R_, D_)                                                                              \
    if (MT == MT_ && NB2 == NB2_ && router == R_ && D == D_) {                                                   \
        constexpr int STEP = 32 * (32 + 16 + (R_ ? 64 : 0)), STAGE = NB2_ * STEP + 16 * MT_ * 32;                \
        constexpr int BT = NB2_ * 8 * 32 * 8, NBT = R_ ? 2 : 1;                                                  \
        int smem = D_ * STAGE + 2 * NBT * BT;                                                                    \
        std::apply([&](auto... a) { launch(k_big<MT_, NB2_, R_, D_>, grid, smem, (int)S, stream.stream(), a...); }, args); \
        return;                                                                                                  \
    }
    BIG_CASE(4, 2, false, 12) BIG_CASE(8, 2, false, 12) BIG_CASE(16, 2, false, 12)
    BIG_CASE(8, 1, true, 12) BIG_CASE(16, 1, true, 12)
    TORCH_CHECK(false, "no big config");
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void small_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                  int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t MT, bool router, int64_t D, int64_t diag);
void rfma_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                 int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t D);
void big_launch(torch::Tensor packed, torch::Tensor x, torch::Tensor out, torch::Tensor esc_off, torch::Tensor esc,
                torch::Tensor scratch, torch::Tensor counters,
                int64_t m, int64_t K, int64_t N, int64_t base2, int64_t S, int64_t G, int64_t MT, int64_t NB2, bool router, int64_t D);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="bandwidth_proj_v3",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["small_launch", "rfma_launch", "big_launch"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=False,
        )
    return _EXT


# ------------------------------------------------------------------------------------------------
# encoding
# ------------------------------------------------------------------------------------------------
NT = 8
BN = 64


def _pack_layer(w: torch.Tensor):
    """Lossless 12-bit (bf16) / 28-bit (fp32) stream in mma B-fragment order + escape list."""
    router = w.dtype == torch.float32
    N, K = w.shape
    assert N % BN == 0 and K % 16 == 0
    if router:
        bits = w.view(torch.int32)
        hi = (bits >> 16) & 0xFFFF
        lo = bits & 0xFFFF
    else:
        hi = w.view(torch.int16).to(torch.int32) & 0xFFFF
        lo = None
    s = hi >> 15
    e = (hi >> 7) & 0xFF
    mant = hi & 0x7F
    hist = torch.bincount(e.flatten(), minlength=256 + 16)
    win = torch.nn.functional.avg_pool1d(hist.float().view(1, 1, -1), 16, stride=1).flatten() * 16
    base = int(win.argmax())
    code = e - base
    esc = (code < 0) | (code > 15)
    sm = torch.where(esc, torch.zeros_like(hi), (s << 7) | mant).to(torch.uint8)
    code = torch.where(esc, torch.zeros_like(code), code).to(torch.uint8)
    NB, KS = N // BN, K // 16
    # SM plane: [nb][step][t=4g+q][j][kk=2h+i]  with n = nb*64 + 8j + g, k = 16 step + 8h + 2q + i
    smp = sm.view(NB, NT, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).reshape(NB, KS, 32 * NT * 4)
    # E plane: per thread NT/2 words; word wd holds tiles (2wd, 2wd+1); nibble index = 4i + 2(j%2) + h
    cp = code.view(NB, NT // 2, 2, 8, KS, 2, 4, 2).permute(0, 4, 3, 6, 1, 7, 2, 5).reshape(NB, KS, 32, NT // 2, 4, 2)
    eb = (cp[..., 0] | (cp[..., 1] << 4)).reshape(NB, KS, 32 * NT * 2)
    parts = [smp, eb]
    if router:
        lo6 = lo.view(NB, NT, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).reshape(NB, KS, 32 * NT * 4)
        lob = torch.stack([lo6 & 0xFF, lo6 >> 8], dim=-1).to(torch.uint8).reshape(NB, KS, 32 * NT * 8)
        parts.append(lob)
    packed = torch.cat(parts, dim=2).contiguous().view(-1)
    # escapes
    idx = esc.nonzero()  # sorted by n then k
    n_idx, k_idx = idx[:, 0], idx[:, 1]
    wt = w[n_idx, k_idx].float()
    if router:
        wd = ((base << 23) | lo[n_idx, k_idx]).to(torch.int32).view(torch.float32)
    else:
        wd = (torch.full_like(k_idx, (base << 7) << 16, dtype=torch.int32)).view(torch.float32)
    counts = torch.bincount(n_idx, minlength=N)
    esc_off = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    esc_off[1:] = torch.cumsum(counts, 0).to(torch.int32)
    ent = torch.zeros((max(int(idx.shape[0]), 1), 4), dtype=torch.int32, device=w.device)
    if idx.shape[0]:
        ent[:, 0] = k_idx.to(torch.int32)
        ent[:, 1] = wt.view(torch.int32)
        ent[:, 2] = wd.view(torch.int32)
    base2 = ((base << 7) | ((base << 7) << 16)) & 0xFFFFFFFF
    return {"packed": packed, "esc_off": esc_off, "esc": ent.contiguous(), "base2": base2, "n_esc": int(idx.shape[0]), "base": base}


# ------------------------------------------------------------------------------------------------
# per-shape configuration: (S cluster split, G outer split, D stages)
# ------------------------------------------------------------------------------------------------
def _small_cfg(layer, m, N, K):
    nblk = N // BN
    S = {36: 3, 128: 1, 64: 2, 16: 8, 4: 16}.get(nblk, 1)
    return {"S": S, "D": 4}


def _big_cfg(layer, m, N, K, router):
    if router:
        return {"S": 2, "G": 8 if m <= 128 else 16, "NB2": 1, "D": 12}
    nblk = N // 128
    S, G = {18: (2, 3), 32: (2, 2), 64: (2, 1)}.get(nblk, (1, 1))
    return {"S": S, "G": G, "NB2": 2, "D": BIG_D}


SMALL_D = int(os.environ.get("BW_SMALL_D", "4"))
BIG_D = int(os.environ.get("BW_BIG_D", "12"))


def prepare(weights):
    ext = _ext()
    state = {"_w": {k: v.contiguous() for k, v in weights.items()}}
    for name, w in weights.items():
        state[name] = _pack_layer(w.contiguous())
        # fp32 scratch for the big kernel's outer split-K (zero-free protocol: ticket counters only)
        N = w.shape[0]
        gmax = 16 if name == "router" else 4
        state[name]["scratch"] = torch.zeros(gmax * 256 * N, dtype=torch.float32, device=w.device)
        state[name]["counters"] = torch.zeros(1024, dtype=torch.int32, device=w.device)
    # router hi/lo fallback for unoptimised m
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    state["_router_hi_t"] = hi.t().contiguous()
    state["_router_lo_t"] = (r - hi.float()).to(torch.bfloat16).t().contiguous()
    # warm every served config (sets function attributes outside any graph capture)
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        for m in (1, 8, 16, 32, 64, 128, 256):
            Nn = 1024 if (layer == "unpooler" and m <= 8) else weights[layer].shape[0]
            x = torch.zeros((m, weights[layer].shape[1]), dtype=torch.bfloat16, device=r.device)
            kernel_fn({"x": x, "config": {"m": m, "n": Nn, "k": weights[layer].shape[1], "layer": layer}}, state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, N, K = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    L = state[layer]
    router = layer == "router"
    ext = _ext()
    if router:
        if m <= 8:
            out = torch.empty((m, N), dtype=torch.float32, device=x.device)
            ext.rfma_launch(L["packed"], x, out, L["esc_off"], L["esc"], m, K, N, L["base2"], 8, 4)
            return out
        if m <= 32:
            out = torch.empty((m, N), dtype=torch.float32, device=x.device)
            ext.small_launch(L["packed"], x, out, L["esc_off"], L["esc"], m, K, N, L["base2"], 8, 2, True, 4, 0)
            return out
        if m in (128, 256) or (64 < m <= 256):
            MT = 8 if m <= 128 else 16
            out = torch.empty((m, N), dtype=torch.float32, device=x.device)
            c = _big_cfg(layer, m, N, K, True)
            ext.big_launch(L["packed"], x, out, L["esc_off"], L["esc"], L["scratch"], L["counters"],
                           m, K, N, L["base2"], c["S"], c["G"], MT, c["NB2"], True, c["D"])
            return out
        acc = torch.mm(x, state["_router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["_router_lo_t"], out_dtype=torch.float32)
        return acc
    if m <= 32:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
        c = _small_cfg(layer, m, N, K)
        ext.small_launch(L["packed"], x, out, L["esc_off"], L["esc"], m, K, N, L["base2"], c["S"], 1 if m <= 16 else 2, False, SMALL_D, 0)
        return out
    if m <= 256:
        MT = 4 if m <= 64 else (8 if m <= 128 else 16)
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
        c = _big_cfg(layer, m, N, K, False)
        ext.big_launch(L["packed"], x, out, L["esc_off"], L["esc"], L["scratch"], L["counters"],
                       m, K, N, L["base2"], c["S"], c["G"], MT, c["NB2"], False, c["D"])
        return out
    return torch.mm(x, state["_w"][layer][:N].t())
