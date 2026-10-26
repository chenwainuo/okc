# okc: problem=muse_bf16_proj
"""contrarian: one m-adaptive kernel for every layer and every m <= 128.

Weights are re-encoded losslessly to 12 bits (sign, 4-bit exponent offset inside a 16-code window, 7-bit mantissa)
and laid out so that each lane's 48 bytes of a 16x64 block decode straight into the mma.m16n8k16 A-fragment
registers (W is the A operand; x, in swizzled smem, is the B operand). Weights outside the exponent window
(about 1 in 7000) are stored as an escape code plus an exact fp32 correction applied in the epilogue, so the
computation uses every weight at its own precision. Split-K reduces with fp32 red into a self-cleaning scratch;
the last CTA of an n-tile converts to bf16. The router runs as one kernel on the hi|lo bf16 pair (K = 16384).
"""

import math
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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace contrarian {

struct Params {
    const uint8_t* __restrict__ w;        // packed stream
    const __nv_bfloat16* __restrict__ x;  // [M][Kx]
    int M, N, KW, Kx;
    int G;        // KW / 64 : k64 groups per 64-row band
    int gps;      // groups per split
    const float* __restrict__ sscale;     // [G] output scale (2^-j) of the segment holding group g
    __nv_bfloat16* __restrict__ out;      // [M][N]
    float* __restrict__ scratch;          // [M][N] fixup accumulator (wgmma path), zero between launches
    float* __restrict__ partials;         // [splits][MT][N]
    int* __restrict__ counters;           // [2][ntiles] arrive / exit
    int ntiles;
    const int* __restrict__ exc_off;      // CSR over (band, kblock256)
    const int2* __restrict__ exc;         // (n%64 | k<<8), float bits (scaled-domain diff)
    int KB;       // KW / 256
    int splits;
    int fp32_out; // output is fp32 (router): out32 points at it
    float* __restrict__ out32;
    int mode;     // 0 normal; 1 diagnostic: stream only
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_full(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

// decode one pair of 12-bit codes into a bf16x2 register. Byte plane B = [s d3 d2 d1 d0 M6 M5 M4],
// nibble plane = [M3 M2 M1 M0]; exponent window [112,127] (E = 112 + d). 4 ops per pair.
__device__ __forceinline__ uint32_t lop3_ab_or_c(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));   // (a & b) | c
    return r;
}
__device__ __forceinline__ uint32_t decode_pair(uint32_t Lw, uint32_t Nw, uint32_t sel, uint32_t m1, uint32_t kc, uint32_t m2) {
    uint32_t w = __byte_perm(Lw, Nw, sel);            // [B1 Nb B0 Nb]
    uint32_t v = lop3_ab_or_c(w, m1, kc);             // (w & 0x8000800F) | 0x38003800
    return lop3_ab_or_c(w >> 4, m2, v);               // | ((w >> 4) & 0x07FF07F0)
}

__device__ __forceinline__ int ld_volatile(const int* p) {
    int v; asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}

// T: smem fp32 tile [MT][TILE_N] holding this CTA's (scaled) partial. splits == 1: store. Otherwise write the
// partial, wait for the tile's other split-CTAs (all co-resident: grid <= #SMs), reduce 1/splits of the tile.
template <int MT, int TILE_N, int NT>
__device__ __forceinline__ void reduce_epilogue(const Params& p, const float* T, int mrows, int n0, int bx, int split,
                                                int tid, bool fix_in_scratch) {
    const int cq = TILE_N / 4;   // float4 per row
    if (p.splits == 1) {
        for (int q = tid; q < mrows * cq; q += NT) {
            int m = q / cq, c = (q % cq) * 4;
            float4 v = *reinterpret_cast<const float4*>(T + m * TILE_N + c);
            if (p.fp32_out) *reinterpret_cast<float4*>(p.out32 + (size_t)m * p.N + n0 + c) = v;
            else {
                __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)m * p.N + n0 + c);
                o[0] = __float22bfloat162_rn(make_float2(v.x, v.y));
                o[1] = __float22bfloat162_rn(make_float2(v.z, v.w));
            }
        }
        return;
    }
    float* part = p.partials + (size_t)split * MT * p.N;
    for (int q = tid; q < mrows * cq; q += NT) {
        int m = q / cq, c = (q % cq) * 4;
        __stcg(reinterpret_cast<float4*>(part + (size_t)m * p.N + n0 + c), *reinterpret_cast<const float4*>(T + m * TILE_N + c));
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        atomicAdd(p.counters + bx, 1);
        while (ld_volatile(p.counters + bx) < p.splits) __nanosleep(64);
    }
    __syncthreads();
    __threadfence();
    const int total = mrows * cq;
    const int lo = (int)(((long long)total * split) / p.splits), hi = (int)(((long long)total * (split + 1)) / p.splits);
    for (int q = lo + tid; q < hi; q += NT) {
        int m = q / cq, c = (q % cq) * 4;
        const size_t off = (size_t)m * p.N + n0 + c;
        float4 a = __ldcg(reinterpret_cast<const float4*>(p.partials + off));
        for (int sp = 1; sp < p.splits; ++sp) {
            float4 b = __ldcg(reinterpret_cast<const float4*>(p.partials + (size_t)sp * MT * p.N + off));
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        if (fix_in_scratch) {
            float4 b = __ldcg(reinterpret_cast<const float4*>(p.scratch + off));
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            __stcg(reinterpret_cast<float4*>(p.scratch + off), make_float4(0.f, 0.f, 0.f, 0.f));
        }
        if (p.fp32_out) *reinterpret_cast<float4*>(p.out32 + off) = a;
        else {
            __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(p.out + off);
            o[0] = __float22bfloat162_rn(make_float2(a.x, a.y));
            o[1] = __float22bfloat162_rn(make_float2(a.z, a.w));
        }
    }
    __syncthreads();
    if (tid == 0) {
        int t = atomicAdd(p.counters + p.ntiles + bx, 1);
        if (t == p.splits - 1) { p.counters[bx] = 0; p.counters[p.ntiles + bx] = 0; __threadfence(); }
    }
}

constexpr int MAXE = 256;   // exception entries prefetched per 64-row band per CTA

// NMF: number of 8-token fragments (MT = 8*NMF); NW: warps per CTA (4 or 8); STAGES: pipeline depth
template <int NMF, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32, (NMF <= 4 ? (NW == 4 ? 4 : 2) : 1))
gemm_kernel(const Params p) {
    constexpr int MT = NMF * 8;
    constexpr int NT = NW * 32;
    constexpr int TILE_N = NW * 16;
    constexpr int NB = NW / 4;              // 64-row bands per CTA
    constexpr int WBYTES = NW * 1536;       // W bytes per stage
    constexpr int XBYTES = MT * 128;        // x bytes per stage
    constexpr int STAGE_BYTES = WBYTES + XBYTES;
    constexpr int RING_BYTES = STAGES * STAGE_BYTES;
    constexpr int C_BYTES = MT * TILE_N * 4;
    constexpr int CO = RING_BYTES > C_BYTES ? RING_BYTES : C_BYTES;
    constexpr int XCH = MT * 8;             // 16B chunks of x per stage
    constexpr int XPT = (XCH + NT - 1) / NT;
    extern __shared__ __align__(128) uint8_t smem[];
    float* C = reinterpret_cast<float*>(smem + CO);
    int2* sexc = reinterpret_cast<int2*>(smem + CO + C_BYTES);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bx = blockIdx.x;             // n-tile
    const int split = blockIdx.y;
    const int g0 = split * p.gps;
    const int G = p.gps;
    const int bb = warp >> 2;              // band within CTA
    const int band = bx * NB + bb;
    const uint8_t* wsrc = p.w + ((size_t)band * p.G + g0) * 6144 + (size_t)(warp & 3) * 1536 + lane * 16;
    const int n0 = bx * TILE_N;
    const uint32_t smem_base = smem_u32(smem);
    const uint32_t wdst0 = smem_base + warp * 1536 + lane * 16;
    const uint4* wq = reinterpret_cast<const uint4*>(smem + warp * 1536 + lane * 16);

    // x chunk assignment of this thread (compile-time count)
    const __nv_bfloat16* xsrc[XPT];
    uint32_t xdst[XPT];
    int xbytes[XPT];
#pragma unroll
    for (int i = 0; i < XPT; ++i) {
        int q = tid + i * NT;
        int r = q >> 3, c = q & 7;
        bool ok = (q < XCH);
        int rr = (ok && r < p.M) ? r : 0;
        xsrc[i] = p.x + (size_t)rr * p.Kx + c * 8;
        xdst[i] = smem_base + WBYTES + r * 128 + ((c ^ (r & 7)) << 4);
        xbytes[i] = (ok && r < p.M) ? 16 : 0;
    }
    const int kxmask = p.Kx - 1;
    const int kb0 = (g0 * 64) & kxmask;     // a k-slice never straddles the x wrap (router hi|lo boundary)
#pragma unroll
    for (int i = 0; i < XPT; ++i) xsrc[i] += kb0;

    auto load_stage = [&](const int g, const int slot) {
        const uint32_t sb = slot * STAGE_BYTES;
        const uint8_t* src = wsrc + (size_t)g * 6144;
        cp_async16_full(wdst0 + sb, src);
        cp_async16_full(wdst0 + sb + 512, src + 512);
        cp_async16_full(wdst0 + sb + 1024, src + 1024);
#pragma unroll
        for (int i = 0; i < XPT; ++i) {
            if (tid + i * NT < XCH) cp_async16(xdst[i] + sb, xsrc[i] + g * 64, xbytes[i]);
        }
    };

    const int eb0 = (g0 * 64) / 256, eb1 = ((g0 + G) * 64) / 256;
    int e0[NB], e1[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        e0[b] = __ldg(p.exc_off + (bx * NB + b) * p.KB + eb0);
        e1[b] = __ldg(p.exc_off + (bx * NB + b) * p.KB + eb1);
    }
    // all stage loads first; the exception entries are stored to smem while those are in flight
#pragma unroll
    for (int st = 0; st < STAGES - 2; ++st) {
        if (st < G) load_stage(st, st);
        cp_async_commit();
    }
    if (STAGES - 2 < G) load_stage(STAGES - 2, STAGES - 2);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        int cnt = e1[b] - e0[b]; if (cnt > MAXE) cnt = MAXE;
        for (int q = tid; q < cnt; q += NT) sexc[b * MAXE + q] = __ldg(p.exc + e0[b] + q);   // published by the loop's first barrier
    }
    cp_async_commit();
    for (int q = tid; q < MT * TILE_N; q += NT) C[q] = 0.f;

    float acc[NMF][4];
#pragma unroll
    for (int f = 0; f < NMF; ++f) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }

    const uint32_t SEL[4] = {0x1404u, 0x3525u, 0x1606u, 0x3727u};
    uint32_t dm1, dkc, dm2;   // decode constants kept in registers (lop3 takes one immediate at most)
    asm("mov.b32 %0, 0x8000800F;" : "=r"(dm1));
    asm("mov.b32 %0, 0x38003800;" : "=r"(dkc));
    asm("mov.b32 %0, 0x07FF07F0;" : "=r"(dm2));
    const int lrow = lane & 7;
    const int lmat = lane >> 3;
    const int tb = tid & 127;
    int ep = 0;
    const int ecnt = min(e1[bb] - e0[bb], MAXE);
    const int ekbase = g0 * 64;
    // ldmatrix lane offsets (without the stage base / chunk index)
    uint32_t lofs[NMF];
#pragma unroll
    for (int f = 0; f < NMF; ++f) {
        int r = (NMF - f >= 2) ? 8 * (f + (lmat >> 1)) + lrow : 8 * f + lrow;
        lofs[f] = WBYTES + r * 128 + (((lmat & 1) ^ (r & 7)) << 4);   // chunk index 2t is xored in below
        // NOTE: (c ^ (r&7)) with c = 2t + (lmat&1): since 2t has bit0 = 0, c ^ (r&7) = ((lmat&1) ^ (r&7)) ^ 2t
    }

    auto body = [&](const int g, const int slot) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int gn = g + STAGES - 1;
            const int sn = (slot + STAGES - 1) % STAGES;
            if (gn < G) load_stage(gn, sn);
            cp_async_commit();
        }
        const uint32_t sb = smem_base + slot * STAGE_BYTES;
        uint32_t L[8], NBW[4];
        {
            const uint4* q = wq + slot * (STAGE_BYTES / 16);
            uint4 a0 = q[0], b0 = q[32], c0 = q[64];
            L[0] = a0.x; L[1] = a0.y; L[2] = a0.z; L[3] = a0.w;
            L[4] = b0.x; L[5] = b0.y; L[6] = b0.z; L[7] = b0.w;
            NBW[0] = c0.x; NBW[1] = c0.y; NBW[2] = c0.z; NBW[3] = c0.w;
        }
        if (ep < ecnt) {
            const int kend = ekbase + (g + 1) * 64;
            while (ep < ecnt) {
                int2 ent = sexc[bb * MAXE + ep];
                int k = ent.x >> 8;
                if (k >= kend) break;
                ++ep;
                int kk = k & 63;
                int nl = (ent.x & 63) + bb * 64;
                float d = __int_as_float(ent.y);
                for (int m = tb; m < MT; m += 128) {
                    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(
                        smem + slot * STAGE_BYTES + WBYTES + m * 128 + (((kk >> 3) ^ (m & 7)) << 4) + (kk & 7) * 2);
                    C[m * TILE_N + nl] += __bfloat162float(*xp) * d;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t a[4];
#pragma unroll
            for (int pp = 0; pp < 4; ++pp) a[pp] = decode_pair(L[2 * t + (pp >> 1)], NBW[t], SEL[pp], dm1, dkc, dm2);
#pragma unroll
            for (int f = 0; f < NMF; f += 2) {
                uint32_t addr = sb + (lofs[f] ^ (2 * t << 4));
                if (f + 1 < NMF) {
                    uint32_t b0, b1, b2, b3;
                    ldmatrix_x4(addr, b0, b1, b2, b3);
                    mma_bf16(acc[f], a, b0, b1);
                    mma_bf16(acc[f + 1], a, b2, b3);
                } else {
                    uint32_t b0, b1;
                    ldmatrix_x2(addr, b0, b1);
                    mma_bf16(acc[f], a, b0, b1);
                }
            }
        }
    };
    for (int g = 0; g < G; g += STAGES) {
#pragma unroll
        for (int u = 0; u < STAGES; ++u) {
            if (g + u < G) body(g + u, u);
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    // overflow exceptions (beyond MAXE): slow path from global x
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        for (int e = e0[b] + MAXE; e < e1[b]; ++e) {
            int2 ent = p.exc[e];
            int nl = (ent.x & 63) + b * 64;
            int k = ent.x >> 8;
            int kx = k & kxmask;
            float d = __int_as_float(ent.y);
            for (int m = tid; m < MT; m += NT) {
                float xv = m < p.M ? __bfloat162float(p.x[(size_t)m * p.Kx + kx]) : 0.f;
                C[m * TILE_N + nl] += xv * d;
            }
        }
    }
    __syncthreads();

    // epilogue: (fragments + C) * scale -> smem fp32 tile T[MT][TILE_N] (reuses the ring)
    const float scale = p.sscale[g0];
    float* T = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int f = 0; f < NMF; ++f) {
        int mrow = 8 * f + 2 * (lane & 3);
        int ncol = warp * 16 + (lane >> 2);
        T[mrow * TILE_N + ncol] = (acc[f][0] + C[mrow * TILE_N + ncol]) * scale;
        T[(mrow + 1) * TILE_N + ncol] = (acc[f][1] + C[(mrow + 1) * TILE_N + ncol]) * scale;
        T[mrow * TILE_N + ncol + 8] = (acc[f][2] + C[mrow * TILE_N + ncol + 8]) * scale;
        T[(mrow + 1) * TILE_N + ncol + 8] = (acc[f][3] + C[(mrow + 1) * TILE_N + ncol + 8]) * scale;
    }
    __syncthreads();
    const int mrows = p.M < MT ? p.M : MT;
    reduce_epilogue<MT, TILE_N, NT>(p, T, mrows, n0, bx, split, tid, false);
}

template <int NMF, int NW, int STAGES>
size_t smem_t() {
    constexpr int MT = NMF * 8;
    constexpr int STAGE_BYTES = NW * 1536 + MT * 128;
    size_t ring = (size_t)STAGES * STAGE_BYTES;
    size_t tile = (size_t)MT * NW * 16 * 4;
    if (tile > ring) ring = tile;
    return ring + tile + (size_t)(NW / 4) * MAXE * 8;
}
template <int NMF, int NW, int STAGES>
int occ_t() {
    size_t smem = smem_t<NMF, NW, STAGES>();
    cudaFuncSetAttribute(gemm_kernel<NMF, NW, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gemm_kernel<NMF, NW, STAGES>, NW * 32, smem);
    return nb;
}
template <int NMF, int NW, int STAGES>
void launch_t(const Params& p, cudaStream_t stream) {
    size_t smem = smem_t<NMF, NW, STAGES>();
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_kernel<NMF, NW, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    dim3 grid(p.N / (NW * 16), p.splits);
    gemm_kernel<NMF, NW, STAGES><<<grid, NW * 32, smem, stream>>>(p);
}


// ------------------------------------------------------------------------------------------------
// wgmma (register-A) variant for MT >= 64: W decoded straight into the wgmma A fragments, x tile in
// SW128 K-major smem as the B descriptor. 2 warpgroups = 128 n-rows per CTA. Requires splits >= 2
// (exceptions are red.added into the scratch).

__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
    uint64_t d = (uint64_t)((saddr & 0x3FFFFu) >> 4);
    d |= (uint64_t)1 << 16;                 // LBO (unused for swizzled K-major)
    d |= (uint64_t)(1024 >> 4) << 32;       // SBO: 8 rows * 128B
    d |= (uint64_t)1 << 62;                 // 128B swizzle
    return d;
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

template <int MT> __device__ __forceinline__ void wgmma_ss(float* acc, uint64_t da, uint64_t db, int scale_d);
template <> __device__ __forceinline__ void wgmma_ss<64>(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(da), "l"(db), "r"(scale_d));
}
template <> __device__ __forceinline__ void wgmma_ss<128>(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
        : "l"(da), "l"(db), "r"(scale_d));
}
template <> __device__ __forceinline__ void wgmma_ss<256>(float* acc, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63]), "+f"(acc[64]), "+f"(acc[65]), "+f"(acc[66]), "+f"(acc[67]), "+f"(acc[68]), "+f"(acc[69]), "+f"(acc[70]), "+f"(acc[71]), "+f"(acc[72]), "+f"(acc[73]), "+f"(acc[74]), "+f"(acc[75]), "+f"(acc[76]), "+f"(acc[77]), "+f"(acc[78]), "+f"(acc[79]), "+f"(acc[80]), "+f"(acc[81]), "+f"(acc[82]), "+f"(acc[83]), "+f"(acc[84]), "+f"(acc[85]), "+f"(acc[86]), "+f"(acc[87]), "+f"(acc[88]), "+f"(acc[89]), "+f"(acc[90]), "+f"(acc[91]), "+f"(acc[92]), "+f"(acc[93]), "+f"(acc[94]), "+f"(acc[95]), "+f"(acc[96]), "+f"(acc[97]), "+f"(acc[98]), "+f"(acc[99]), "+f"(acc[100]), "+f"(acc[101]), "+f"(acc[102]), "+f"(acc[103]), "+f"(acc[104]), "+f"(acc[105]), "+f"(acc[106]), "+f"(acc[107]), "+f"(acc[108]), "+f"(acc[109]), "+f"(acc[110]), "+f"(acc[111]), "+f"(acc[112]), "+f"(acc[113]), "+f"(acc[114]), "+f"(acc[115]), "+f"(acc[116]), "+f"(acc[117]), "+f"(acc[118]), "+f"(acc[119]), "+f"(acc[120]), "+f"(acc[121]), "+f"(acc[122]), "+f"(acc[123]), "+f"(acc[124]), "+f"(acc[125]), "+f"(acc[126]), "+f"(acc[127])
        : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_named(int id, int count) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory"); }

template <int NMF, int STAGES>
__global__ void __launch_bounds__(256, 1)
gemm_wg_kernel(const Params p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    constexpr int MT = NMF * 8;
    constexpr int NW = 8;
    constexpr int NT = 256;
    constexpr int TILE_N = 128;
    constexpr int NB = 2;
    constexpr int WBYTES = NW * 1536;       // 12288
    constexpr int XBYTES = MT * 128;
    constexpr int STAGE_BYTES = WBYTES + XBYTES;
    constexpr int RING_BYTES = STAGES * STAGE_BYTES;
    constexpr int XCH = MT * 8;
    constexpr int XPT = (XCH + NT - 1) / NT;
    static_assert(STAGE_BYTES % 1024 == 0, "stage alignment");
    constexpr int ABUF_BYTES = 2 * 2 * 8192;   // [wg][buf] 64 rows x 128B, SW128 K-major
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    const uint32_t raw_base = smem_u32(smem_raw);
    const uint32_t smem_base = (raw_base + 1023u) & ~1023u;
    uint8_t* smem = smem_raw + (smem_base - raw_base);
    int2* sexc = reinterpret_cast<int2*>(smem + RING_BYTES + ABUF_BYTES);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bx = blockIdx.x;
    const int split = blockIdx.y;
    const int g0 = split * p.gps;
    const int G = p.gps;
    const int bb = warp >> 2;
    const int band = bx * NB + bb;
    const uint8_t* wsrc = p.w + ((size_t)band * p.G + g0) * 6144 + (size_t)(warp & 3) * 1536 + lane * 16;
    const int n0 = bx * TILE_N;
    const uint32_t wdst0 = smem_base + warp * 1536 + lane * 16;
    const uint4* wq = reinterpret_cast<const uint4*>(smem + warp * 1536 + lane * 16);

    const __nv_bfloat16* xsrc[XPT];
    uint32_t xdst[XPT];
    int xbytes[XPT];
#pragma unroll
    for (int i = 0; i < XPT; ++i) {
        int q = tid + i * NT;
        int r = q >> 3, c = q & 7;
        bool ok = (q < XCH);
        int rr = (ok && r < p.M) ? r : 0;
        xsrc[i] = p.x + (size_t)rr * p.Kx + c * 8;
        xdst[i] = smem_base + WBYTES + r * 128 + ((c ^ (r & 7)) << 4);
        xbytes[i] = (ok && r < p.M) ? 16 : 0;
    }
    const int kxmask = p.Kx - 1;
    const int kb0 = (g0 * 64) & kxmask;     // a k-slice never straddles the x wrap (router hi|lo boundary)
#pragma unroll
    for (int i = 0; i < XPT; ++i) xsrc[i] += kb0;

    auto load_stage = [&](const int g, const int slot) {
        const uint32_t sb = slot * STAGE_BYTES;
        const uint8_t* src = wsrc + (size_t)g * 6144;
        cp_async16_full(wdst0 + sb, src);
        cp_async16_full(wdst0 + sb + 512, src + 512);
        cp_async16_full(wdst0 + sb + 1024, src + 1024);
#pragma unroll
        for (int i = 0; i < XPT; ++i) {
            if (tid + i * NT < XCH) cp_async16(xdst[i] + sb, xsrc[i] + g * 64, xbytes[i]);
        }
    };

    const int eb0 = (g0 * 64) / 256, eb1 = ((g0 + G) * 64) / 256;
    int e0[NB], e1[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        e0[b] = __ldg(p.exc_off + (bx * NB + b) * p.KB + eb0);
        e1[b] = __ldg(p.exc_off + (bx * NB + b) * p.KB + eb1);
    }
    // prefetch STAGES-2 stages (one slot is kept for the in-flight wgmma group)
#pragma unroll
    for (int st = 0; st < STAGES - 3; ++st) {
        if (st < G) load_stage(st, st);
        cp_async_commit();
    }
    if (STAGES - 3 < G) load_stage(STAGES - 3, STAGES - 3);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        int cnt = e1[b] - e0[b]; if (cnt > MAXE) cnt = MAXE;
        for (int q = tid; q < cnt; q += NT) sexc[b * MAXE + q] = __ldg(p.exc + e0[b] + q);   // published by the loop's first barrier
    }
    cp_async_commit();

    float acc[NMF * 4];   // deliberately uninitialised: the first wgmma uses scale-d = 0

    const uint32_t SEL[4] = {0x1404u, 0x3525u, 0x1606u, 0x3727u};
    uint32_t dm1, dkc, dm2;   // decode constants kept in registers (lop3 takes one immediate at most)
    asm("mov.b32 %0, 0x8000800F;" : "=r"(dm1));
    asm("mov.b32 %0, 0x38003800;" : "=r"(dkc));
    asm("mov.b32 %0, 0x07FF07F0;" : "=r"(dm2));
    const int tb = tid & 127;
    int ep = 0;
    const int ecnt = min(e1[bb] - e0[bb], MAXE);
    const int ekbase = g0 * 64;
    const float scale = p.sscale[g0];
    const int mrows = p.M < MT ? p.M : MT;

    static_assert(STAGES % 2 == 0, "even stages");
    // A staging: this thread's 16 bf16x2 values go to rows r0 = 16*(warp&3) + lane/4 (+8), k = 16t + 8h + 2(lane&3)
    const uint32_t abase = smem_base + RING_BYTES + bb * 16384;
    uint32_t aofs[2];
    {
        int r0 = 16 * (warp & 3) + (lane >> 2);
        aofs[0] = r0 * 128 + (lane & 3) * 4;        // chunk c xored in below with (r0 & 7)
        aofs[1] = (r0 + 8) * 128 + (lane & 3) * 4;
    }
    const int rx0 = (lane >> 2) & 7;   // r0 & 7 == (r0+8) & 7

    auto step = [&](const int g, const int slot, const int abuf, const bool first) {
        // group g-2 retired by the wait<1> that followed commit(g-1): its A buffer and x slot are free
        cp_async_wait<STAGES - 3>();
        fence_proxy_async();                   // cp.async (generic proxy) writes -> visible to wgmma (async proxy)
        __syncthreads();
        {
            const int gn = g + STAGES - 2;
            const int sn = (slot + STAGES - 2) % STAGES;
            if (gn < G) load_stage(gn, sn);
            cp_async_commit();
        }
        const uint32_t sb = smem_base + slot * STAGE_BYTES;
        uint32_t L[8], NBW[4];
        {
            const uint4* q = wq + slot * (STAGE_BYTES / 16);
            uint4 a0 = q[0], b0 = q[32], c0 = q[64];
            L[0] = a0.x; L[1] = a0.y; L[2] = a0.z; L[3] = a0.w;
            L[4] = b0.x; L[5] = b0.y; L[6] = b0.z; L[7] = b0.w;
            NBW[0] = c0.x; NBW[1] = c0.y; NBW[2] = c0.z; NBW[3] = c0.w;
        }
        if (ep < ecnt) {
            const int kend = ekbase + (g + 1) * 64;
            while (ep < ecnt) {
                int2 ent = sexc[bb * MAXE + ep];
                int k = ent.x >> 8;
                if (k >= kend) break;
                ++ep;
                int kk = k & 63;
                int nl = (ent.x & 63) + bb * 64;
                float d = __int_as_float(ent.y) * scale;
                for (int m = tb; m < mrows; m += 128) {
                    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(
                        smem + slot * STAGE_BYTES + WBYTES + m * 128 + (((kk >> 3) ^ (m & 7)) << 4) + (kk & 7) * 2);
                    atomicAdd(p.scratch + (size_t)m * p.N + n0 + nl, __bfloat162float(*xp) * d);
                }
            }
        }
        const uint32_t ab = abase + abuf * 8192;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int pp = 0; pp < 4; ++pp) {
                uint32_t v = decode_pair(L[2 * t + (pp >> 1)], NBW[t], SEL[pp], dm1, dkc, dm2);
                // pp: 0 -> (row r0, k-half 0), 1 -> (r0+8, half 0), 2 -> (r0, half 1), 3 -> (r0+8, half 1)
                const int c = 2 * t + (pp >> 1);
                sts32(ab + aofs[pp & 1] + ((c ^ rx0) << 4), v);
            }
        }
        fence_proxy_async();
        bar_named(1 + bb, 128);               // the warpgroup's A tile is complete
#pragma unroll
        for (int i = 0; i < NMF * 4; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
        wgmma_fence();
        const uint64_t da0 = make_desc_sw128(ab);
        const uint64_t db0 = make_desc_sw128(sb + WBYTES);
#pragma unroll
        for (int t = 0; t < 4; ++t) wgmma_ss<MT>(acc, da0 + (uint64_t)(t * 2), db0 + (uint64_t)(t * 2), (first && t == 0) ? 0 : 1);
        wgmma_commit();
        wgmma_wait<1>();                       // retire group g-1; group g stays in flight during the next decode
#pragma unroll
        for (int i = 0; i < NMF * 4; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
    };

    // G is a multiple of STAGES (enforced by the host). The first step is peeled so that scale-d is a compile-time
    // constant everywhere: a runtime scale-d predicate makes ptxas serialise the whole wgmma stream (C7515).
    step(0, 0, 0, true);
#pragma unroll
    for (int u = 1; u < STAGES; ++u) step(u, u, u & 1, false);
    for (int g = STAGES; g < G; g += STAGES) {
#pragma unroll
        for (int u = 0; u < STAGES; u += 2) {
            step(g + u, u, 0, false);
            step(g + u + 1, u + 1, 1, false);
        }
    }
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < NMF * 4; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
    cp_async_wait<0>();
    __syncthreads();
    // overflow exceptions beyond MAXE (rare): from global x
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        for (int e = e0[b] + MAXE; e < e1[b]; ++e) {
            int2 ent = p.exc[e];
            int nl = (ent.x & 63) + b * 64;
            int kx = (ent.x >> 8) & kxmask;
            float d = __int_as_float(ent.y) * scale;
            for (int m = tid; m < mrows; m += NT)
                atomicAdd(p.scratch + (size_t)m * p.N + n0 + nl, __bfloat162float(p.x[(size_t)m * p.Kx + kx]) * d);
        }
    }
    // epilogue: fragments * scale -> smem tile -> red into scratch; last CTA converts
    float* T = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int f = 0; f < NMF; ++f) {
        int mrow = 8 * f + 2 * (lane & 3);
        int ncol = warp * 16 + (lane >> 2);
        T[mrow * TILE_N + ncol] = acc[4 * f] * scale;
        T[(mrow + 1) * TILE_N + ncol] = acc[4 * f + 1] * scale;
        T[mrow * TILE_N + ncol + 8] = acc[4 * f + 2] * scale;
        T[(mrow + 1) * TILE_N + ncol + 8] = acc[4 * f + 3] * scale;
    }
    __syncthreads();
    reduce_epilogue<MT, TILE_N, NT>(p, T, mrows, n0, bx, split, tid, true);
#endif
}

template <int NMF, int STAGES>
size_t smem_wg() {
    constexpr int MT = NMF * 8;
    constexpr int STAGE_BYTES = 8 * 1536 + MT * 128;
    size_t ring = (size_t)STAGES * STAGE_BYTES;
    size_t tile = (size_t)MT * 128 * 4;
    if (tile > ring) ring = tile;
    return ring + 32768 + 2 * MAXE * 8 + 64 + 1024;
}
template <int NMF, int STAGES>
int occ_wg() {
    size_t smem = smem_wg<NMF, STAGES>();
    cudaFuncSetAttribute(gemm_wg_kernel<NMF, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gemm_wg_kernel<NMF, STAGES>, 256, smem);
    return nb;
}
template <int NMF, int STAGES>
void launch_wg(const Params& p, cudaStream_t stream) {
    size_t smem = smem_wg<NMF, STAGES>();
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_wg_kernel<NMF, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    dim3 grid(p.N / 128, p.splits);
    gemm_wg_kernel<NMF, STAGES><<<grid, 256, smem, stream>>>(p);
}

}  // namespace contrarian

int64_t contrarian_occ(int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode) {
    using namespace contrarian;
#define OCC_WG(NMF_, ST_) if (nmf == NMF_ && nwarps == 8 && stages == ST_ && mode == 2) return occ_wg<NMF_, ST_>();
    OCC_WG(8, 4) OCC_WG(16, 4) OCC_WG(32, 4) OCC_WG(8, 8)
#undef OCC_WG
#define OCC(NMF_, NW_, ST_) if (nmf == NMF_ && nwarps == NW_ && stages == ST_ && mode != 2) return occ_t<NMF_, NW_, ST_>();
    OCC(1, 4, 8) OCC(2, 4, 8) OCC(4, 4, 8)
    OCC(1, 8, 6) OCC(2, 8, 6) OCC(4, 8, 6) OCC(8, 8, 4) OCC(16, 8, 4)
    OCC(1, 4, 12) OCC(2, 4, 12) OCC(4, 4, 12)
#undef OCC
    return 0;
}

void contrarian_launch(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor scratch, torch::Tensor partials,
                       torch::Tensor counters, torch::Tensor kadd, torch::Tensor exc_off, torch::Tensor exc, int64_t KW,
                       int64_t splits, int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode) {
    using namespace contrarian;
    Params p;
    p.w = w.data_ptr<uint8_t>();
    p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    p.M = (int)x.size(0);
    p.N = (int)out.size(1);
    p.KW = (int)KW;
    p.Kx = (int)x.size(1);
    p.G = p.KW / 64;
    p.gps = p.G / (int)splits;
    p.sscale = kadd.data_ptr<float>();
    p.scratch = scratch.data_ptr<float>();
    p.partials = partials.data_ptr<float>();
    p.counters = counters.data_ptr<int>();
    p.ntiles = p.N / (int)(nwarps * 16);
    p.exc_off = exc_off.data_ptr<int>();
    p.exc = reinterpret_cast<const int2*>(exc.data_ptr());
    p.KB = p.KW / 256;
    p.splits = (int)splits;
    p.fp32_out = out.scalar_type() == torch::kFloat32 ? 1 : 0;
    p.out32 = p.fp32_out ? out.data_ptr<float>() : nullptr;
    p.out = p.fp32_out ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.mode = (int)mode;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define DISPATCH_WG(NMF_, ST_) \
    if (nmf == NMF_ && nwarps == 8 && stages == ST_ && mode == 2) { launch_wg<NMF_, ST_>(p, stream); return; }
    DISPATCH_WG(8, 4) DISPATCH_WG(16, 4) DISPATCH_WG(32, 4) DISPATCH_WG(8, 8)
#undef DISPATCH_WG
#define DISPATCH(NMF_, NW_, ST_) \
    if (nmf == NMF_ && nwarps == NW_ && stages == ST_) { launch_t<NMF_, NW_, ST_>(p, stream); return; }
    DISPATCH(1, 4, 8) DISPATCH(2, 4, 8) DISPATCH(4, 4, 8)
    DISPATCH(1, 8, 6) DISPATCH(2, 8, 6) DISPATCH(4, 8, 6) DISPATCH(8, 8, 4) DISPATCH(16, 8, 4)
    DISPATCH(1, 4, 12) DISPATCH(2, 4, 12) DISPATCH(4, 4, 12)
#undef DISPATCH
    TORCH_CHECK(false, "no kernel instance for nmf/nwarps/stages");
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void contrarian_launch(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor scratch, torch::Tensor partials,
                       torch::Tensor counters, torch::Tensor kadd, torch::Tensor exc_off, torch::Tensor exc, int64_t KW,
                       int64_t splits, int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode);
int64_t contrarian_occ(int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode);
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="contrarian_v14",
            cpp_sources=[CPP_SRC],
            cuda_sources=["#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["contrarian_launch", "contrarian_occ"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"],
            verbose=False,
        )
    return _ext


# ----------------------------------------------------------------------------------------------------------------
# packing

def _pack_layer(W: torch.Tensor, segments):
    """W bf16 [N][KW] -> packed stream, kadd[G], exception CSR. segments: list of (g_start, g_end) sharing a base."""
    N, KW = W.shape
    assert N % 64 == 0 and KW % 256 == 0
    dev = W.device
    G = KW // 64
    KB = KW // 256
    bits = W.view(torch.int16).to(torch.int32) & 0xFFFF
    E = (bits >> 7) & 0xFF
    # per segment: shift exponents by j so the window [maxE-15, maxE] lands on [112, 127]
    j_g = torch.zeros(G, dtype=torch.int64, device=dev)
    for gs, ge in segments:
        maxE = int(E[:, gs * 64: ge * 64].max())
        j_g[gs:ge] = 127 - maxE
    sscale = torch.ldexp(torch.ones(G, device=dev), (-j_g).to(torch.int32)).float().contiguous()  # 2^-j
    j_k = j_g.repeat_interleave(64)  # [KW]
    d = E + j_k[None, :] - 112
    exc = (d < 0) | (E == 0)
    s = bits >> 15
    M = bits & 0x7F
    dd = torch.where(exc, torch.zeros_like(d), d)
    Lb = torch.where(exc, torch.zeros_like(d), (s << 7) | (dd << 3) | (M >> 4)).to(torch.uint8)
    Hb = torch.where(exc, torch.zeros_like(d), M & 0xF).to(torch.uint8)
    # index map (n, k) -> lane element
    n = torch.arange(N, device=dev)[:, None]
    k = torch.arange(KW, device=dev)[None, :]
    band = n // 64
    warp = (n % 64) // 16
    r = n % 16
    g = k // 64
    kk = k % 64
    t16 = kk // 16
    c = kk % 16
    lane = 4 * (r % 8) + (c % 8) // 2
    e = 4 * (c // 8) + 2 * (r // 8) + (c % 2)
    i = 8 * t16 + e
    fg = ((band * G + g) * 4 + warp)  # fragment-group index
    Lidx = fg * 1024 + (i // 16) * 512 + lane * 16 + (i % 16)
    Nidx = fg * 1024 + lane * 32 + i
    nfg = (N // 64) * G * 4
    Lplane = torch.zeros(nfg * 1024, dtype=torch.uint8, device=dev)
    Lplane[Lidx.reshape(-1)] = Lb.reshape(-1)
    Nib = torch.zeros(nfg * 1024, dtype=torch.uint8, device=dev)
    Nib[Nidx.reshape(-1)] = Hb.reshape(-1)
    Nib2 = Nib.view(-1, 2)
    Nbytes = (Nib2[:, 0] | (Nib2[:, 1] << 4)).contiguous()
    packed = torch.cat([Lplane.view(nfg, 1024), Nbytes.view(nfg, 512)], dim=1).contiguous().view(-1)
    # exceptions: escape code decodes to +2^-15 in the scaled domain; store diff = w*2^j - 2^-15
    en, ek = exc.nonzero(as_tuple=True)
    wv = W.float()[en, ek]
    wscaled = torch.ldexp(wv, j_k[ek].to(torch.int32))
    diff = (wscaled - 2.0 ** -15).float()
    key = (en // 64) * KB + ek // 256
    order = torch.argsort((en // 64) * KW + ek)
    en, ek, diff, key = en[order], ek[order], diff[order], key[order]
    counts = torch.bincount(key, minlength=(N // 64) * KB)
    off = torch.zeros((N // 64) * KB + 1, dtype=torch.int32, device=dev)
    off[1:] = torch.cumsum(counts, 0).to(torch.int32)
    ent = torch.stack([(en % 64) | (ek << 8), diff.view(torch.int32).to(torch.int64)], dim=1).to(torch.int32).contiguous()
    if ent.numel() == 0:
        ent = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    return {"packed": packed, "kadd": sscale, "exc_off": off.contiguous(), "exc": ent, "KW": KW, "N": N,
            "n_exc": int(exc.sum())}


def prepare(weights: Weights):
    _module()
    state = {"orig": {k: v.contiguous() for k, v in weights.items()}}
    dev = weights["qkv_gate"].device
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        W = weights[name].contiguous()
        G = W.shape[1] // 64
        pk = _pack_layer(W, [(0, G)])
        state[name] = pk
        print(f"PACK {name}: {pk['n_exc']} exceptions, {pk['packed'].numel()/1e6:.1f} MB", flush=True)
    r = weights["router"].float()
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    Wr = torch.cat([hi, lo], dim=1).contiguous()
    G = Wr.shape[1] // 64
    pk = _pack_layer(Wr, [(0, G // 2), (G // 2, G)])
    state["router"] = pk
    print(f"PACK router: {pk['n_exc']} exceptions, {pk['packed'].numel()/1e6:.1f} MB", flush=True)
    state["scratch"] = torch.zeros(256 * 8192, dtype=torch.float32, device=dev)
    state["partials"] = torch.zeros(8 * 1024 * 1024, dtype=torch.float32, device=dev)   # 32 MB: splits*MT*N <= 8M floats
    state["counters"] = torch.zeros(2048, dtype=torch.int32, device=dev)
    if _BAKE:
        _bake(state, dev)
    return state


import os
import statistics
import time
_OVR = os.environ.get("CONTRARIAN_CFG", "")  # "nw,splits,stages[,mode]" override for experiments
_MODE = int(os.environ.get("CONTRARIAN_MODE", "0"))
_WG = int(os.environ.get("CONTRARIAN_WG", "1"))
_BAKE = int(os.environ.get("CONTRARIAN_BAKE", "1"))
_BAKE_TORCH = int(os.environ.get("CONTRARIAN_BAKE_TORCH", "1"))   # also time cuBLAS; pick it when it is faster
_KW = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 16384}
_PICKS = {}
_NSM = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 132


def _mt_of(m):
    nmf = 1
    while nmf * 8 < m:
        nmf *= 2
    return nmf * 8


def _splits_for(layer, n, nw, target):
    tiles = n // (nw * 16)
    kw = _KW[layer]
    splits = 1
    while tiles * splits < target and (kw // (splits * 2)) % 256 == 0:
        splits *= 2
    return splits


def _candidates(layer, n, mt):
    """(nw, splits, stages, mode) candidates for a shape class; mode 0 = mma.sync kernel, 2 = wgmma kernel."""
    c = []
    if mt <= 32:
        for tgt in (128, 256):
            c.append((8, _splits_for(layer, n, 8, tgt), 6, 0))
        for tgt in (128, 256, 512):
            c.append((4, _splits_for(layer, n, 4, tgt), 8, 0))
        c.append((4, _splits_for(layer, n, 4, 256), 12, 0))
    else:
        for tgt in (128, 256):
            sp = max(2, _splits_for(layer, n, 8, tgt))
            c.append((8, sp, 4, 2))
            if mt <= 64:
                c.append((8, sp, 8, 2))
        if mt <= 128:
            c.append((8, _splits_for(layer, n, 8, 128), 4, 0))
    out = []
    for x in c:
        nw, sp, st, mode = x
        while sp >= 2 and not _cfg_ok(layer, n, mt, (nw, sp, st, mode)):
            sp //= 2
        if mode == 2 and sp < 2:
            continue
        x = (nw, sp, st, mode)
        if x not in out and _cfg_ok(layer, n, mt, x):
            out.append(x)
    return out


_OCC = {}


def _occ(nw, nmf, st, mode):
    key = (nw, nmf, st, mode)
    if key not in _OCC:
        _OCC[key] = int(_module().contrarian_occ(nw, nmf, st, mode))
    return _OCC[key]


def _cfg_ok(layer, n, mt, cfg):
    """A split>1 config spins on the tile's other CTAs, so the whole grid must be co-resident."""
    nw, sp, st, mode = cfg
    if mode == 2 and ((_KW[layer] // 64 // sp) % st != 0):
        return False
    if (_KW[layer] // sp) % 256 == 0 and sp * mt * n <= 8 * 1024 * 1024:
        if sp == 1:
            return True
        grid = (n // (nw * 16)) * sp
        return grid <= _occ(nw, mt // 8, st, mode) * _NSM
    return False


def _default_config(layer, n, mt):
    c = _candidates(layer, n, mt)
    if not c:
        raise RuntimeError(f"no valid config for {layer} n={n} mt={mt}")
    return c[0]


def _config(layer, n, mt):
    if _OVR:
        a = [int(v) for v in _OVR.split(",")]
        return (a[0], a[1], a[2], a[3] if len(a) > 3 else (2 if mt >= 64 else 0))
    key = (layer, n, mt)
    if key in _PICKS:
        return _PICKS[key]
    c = _candidates(layer, n, mt)
    return c[0] if c else None


def _time_graph(fn, flush, reps):
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


def _bake(state, dev):
    """Time every candidate config per (layer, m-tile) on synthetic inputs, keep the fastest one that also matches a
    torch reference; shapes with no passing config fall back to torch.mm in kernel_fn."""
    t0 = time.time()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        for mt in (8, 16, 32, 64, 128, 256):
            n = 1024 if (layer == "unpooler" and mt <= 8) else {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}[layer]
            k = 8192 if layer == "router" else _KW[layer]
            x = torch.randn((mt, k), generator=gen, device=dev).to(torch.bfloat16)
            w = state["orig"][layer][:n]
            ref = x.float() @ w.float().T
            tol = 1e-4 if layer == "router" else 1e-2
            out = torch.empty((mt, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=dev)
            best = None
            for cfg in _candidates(layer, n, mt):
                try:
                    out.zero_()
                    _launch(x, layer, n, state, out, cfg)
                    torch.cuda.synchronize()
                    err = float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))
                    if not (err < tol):
                        print(f"BAKE {layer} mt={mt} cfg={cfg} WRONG err={err:.3e}", flush=True)
                        continue
                    t = _time_graph(lambda: _launch(x, layer, n, state, out, cfg), flush, 11)
                except Exception as e:  # noqa: BLE001
                    print(f"BAKE {layer} mt={mt} cfg={cfg} failed: {e}", flush=True)
                    continue
                print(f"BAKE {layer:9s} mt={mt:<4d} {cfg} {t:.2f} us", flush=True)
                if best is None or t < best[0]:
                    best = (t, cfg)
            if _BAKE_TORCH:
                if layer == "router":
                    def tfn():
                        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
                        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
                        return acc
                else:
                    wt = w.t().contiguous()
                    tfn = lambda: torch.mm(x, wt)  # noqa: E731
                tt = _time_graph(tfn, flush, 11)
                print(f"TORCH {layer:9s} mt={mt:<4d} {tt:.2f} us", flush=True)
                if best is not None and tt < best[0]:
                    print(f"PICK {layer:9s} n={n:<5d} mt={mt:<4d} torch {tt:.2f} us beats {best[1]} {best[0]:.2f} us", flush=True)
                    best = None
            if best is not None:
                _PICKS[(layer, n, mt)] = best[1]
                print(f"PICK {layer:9s} n={n:<5d} mt={mt:<4d} {best[1]} {best[0]:.2f} us", flush=True)
            else:
                _PICKS[(layer, n, mt)] = None
                print(f"PICK {layer:9s} n={n:<5d} mt={mt:<4d} NONE -> torch fallback", flush=True)
    del flush
    torch.cuda.synchronize()
    print(f"BAKE {time.time() - t0:.1f}s", flush=True)


def _launch(x, layer, n, state, out, cfg):
    pk = state[layer]
    nw, splits, st, mode = cfg
    nmf = _mt_of(x.shape[0]) // 8
    if not _cfg_ok(layer, n, nmf * 8, cfg):
        raise RuntimeError(f"config {cfg} not co-resident for {layer} n={n} mt={nmf * 8}")
    _module().contrarian_launch(pk["packed"], x, out, state["scratch"], state["partials"], state["counters"], pk["kadd"],
                                pk["exc_off"], pk["exc"], pk["KW"], splits, nw, nmf, st, mode)


def _run(x, layer, n, state, out):
    mt = _mt_of(x.shape[0])
    cfg = _config(layer, n, mt)
    if cfg is None:
        w = state["orig"][layer][:n]
        if layer == "router":
            torch.mm(x, state["router_hi_t"], out_dtype=torch.float32, out=out)
            out.add_(torch.mm(x, state["router_lo_t"], out_dtype=torch.float32))
        else:
            torch.mm(x, w.T, out=out)
        return
    if not _WG and cfg[3] == 2:
        cfg = (8, cfg[1], 4, 0)
    _launch(x, layer, n, state, out, cfg)


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n, m = cfg["layer"], cfg["n"], x.shape[0]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if m <= 256:
        _run(x, layer, n, state, out)
    else:
        for r0 in range(0, m, 256):
            r1 = min(m, r0 + 256)
            _run(x[r0:r1], layer, n, state, out[r0:r1])
    return out
