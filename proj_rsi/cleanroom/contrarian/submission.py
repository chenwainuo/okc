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
    float* __restrict__ scratch;          // [M][N]
    int* __restrict__ counters;           // per n-tile
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

// decode one pair of 12-bit codes into a bf16x2 register (exponent window [112,127]; sign in nibble bit 3)
__device__ __forceinline__ uint32_t decode_pair(uint32_t Lw, uint32_t Nw, uint32_t sel) {
    uint32_t w = __byte_perm(Lw, Nw, sel);            // [N L1 N L0]
    uint32_t v = (w & 0x80FF07FFu) | 0x38003800u;     // L0, d0 bits (8-10), L1, s1 (bit 31), E high bits
    v |= (w >> 4) & 0x07000000u;                      // d1 bits -> 24-26
    v |= (w << 4) & 0x00008000u;                      // s0 -> 15
    return v;
}

constexpr int MAXE = 256;   // exception entries prefetched per 64-row band per CTA

// NMF: number of 8-token fragments (MT = 8*NMF); NW: warps per CTA (4 or 8); STAGES: pipeline depth
template <int NMF, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32, (NW == 4 ? 2 : 1))
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

    auto load_stage = [&](int g, int slot) {
        const uint32_t sb = slot * STAGE_BYTES;
        const uint8_t* src = wsrc + (size_t)g * 6144;
        cp_async16_full(wdst0 + sb, src);
        cp_async16_full(wdst0 + sb + 512, src + 512);
        cp_async16_full(wdst0 + sb + 1024, src + 1024);
        const int kb = ((g0 + g) * 64) & kxmask;
#pragma unroll
        for (int i = 0; i < XPT; ++i) {
            if (tid + i * NT < XCH) cp_async16(xdst[i] + sb, xsrc[i] + kb, xbytes[i]);
        }
    };

    // all stage loads first; exception metadata rides in the last prologue group
#pragma unroll
    for (int st = 0; st < STAGES - 2; ++st) {
        if (st < G) load_stage(st, st);
        cp_async_commit();
    }
    if (STAGES - 2 < G) load_stage(STAGES - 2, STAGES - 2);
    const int kb0 = (g0 * 64) / 256, kb1 = ((g0 + G) * 64) / 256;
    int e0[NB], e1[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        e0[b] = p.exc_off[(bx * NB + b) * p.KB + kb0];
        e1[b] = p.exc_off[(bx * NB + b) * p.KB + kb1];
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        int cnt = e1[b] - e0[b]; if (cnt > MAXE) cnt = MAXE;
        for (int q = tid; q * 2 < cnt; q += NT) {
            int ne = cnt - q * 2; int bytes = ne >= 2 ? 16 : 8;
            cp_async16(smem_u32(sexc + b * MAXE + q * 2), p.exc + e0[b] + q * 2, bytes);
        }
    }
    cp_async_commit();
    for (int q = tid; q < MT * TILE_N; q += NT) C[q] = 0.f;

    float acc[NMF][4];
#pragma unroll
    for (int f = 0; f < NMF; ++f) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }

    const uint32_t SEL[4] = {0x4140u, 0x5352u, 0x6160u, 0x7372u};
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

    for (int g = 0; g < G; ++g) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int gn = g + STAGES - 1;
            if (gn < G) load_stage(gn, gn % STAGES);
            cp_async_commit();
        }
        const int slot = g % STAGES;
        const uint32_t sb = smem_base + slot * STAGE_BYTES;
        uint32_t L[8], NBW[4];
        {
            const uint4* q = reinterpret_cast<const uint4*>(smem + slot * STAGE_BYTES + warp * 1536 + lane * 16);
            uint4 a = q[0], b = q[32], c = q[64];
            L[0] = a.x; L[1] = a.y; L[2] = a.z; L[3] = a.w;
            L[4] = b.x; L[5] = b.y; L[6] = b.z; L[7] = b.w;
            NBW[0] = c.x; NBW[1] = c.y; NBW[2] = c.z; NBW[3] = c.w;
        }
        if (p.mode == 1) { acc[0][0] += __uint_as_float(L[0] ^ L[7] ^ NBW[3]); continue; }
        // exceptions whose k falls in this group: read x from the smem tile
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
            for (int pp = 0; pp < 4; ++pp) a[pp] = decode_pair(L[2 * t + (pp >> 1)], NBW[t], SEL[pp]);
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
    if (p.splits == 1) {
        for (int q = tid; q < mrows * (TILE_N / 2); q += NT) {
            int m = q / (TILE_N / 2), c = (q % (TILE_N / 2)) * 2;
            float2 v2 = make_float2(T[m * TILE_N + c], T[m * TILE_N + c + 1]);
            if (p.fp32_out) *reinterpret_cast<float2*>(p.out32 + (size_t)m * p.N + n0 + c) = v2;
            else *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)m * p.N + n0 + c) = __float22bfloat162_rn(v2);
        }
    } else {
        for (int q = tid; q < mrows * TILE_N; q += NT) {
            int m = q / TILE_N, c = q % TILE_N;
            atomicAdd(p.scratch + (size_t)m * p.N + n0 + c, T[m * TILE_N + c]);
        }
        __threadfence();
        __syncthreads();
        __shared__ int s_last;
        if (tid == 0) {
            int t = atomicAdd(p.counters + bx, 1);
            s_last = (t == p.splits - 1);
        }
        __syncthreads();
        if (s_last) {
            __threadfence();
            for (int q = tid; q < mrows * (TILE_N / 2); q += NT) {
                int m = q / (TILE_N / 2), c = (q % (TILE_N / 2)) * 2;
                float* sp = p.scratch + (size_t)m * p.N + n0 + c;
                float2 v2 = __ldcg(reinterpret_cast<float2*>(sp));
                if (p.fp32_out) *reinterpret_cast<float2*>(p.out32 + (size_t)m * p.N + n0 + c) = v2;
                else *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)m * p.N + n0 + c) = __float22bfloat162_rn(v2);
                __stcg(reinterpret_cast<float2*>(sp), make_float2(0.f, 0.f));
            }
            if (tid == 0) p.counters[bx] = 0;
        }
    }
}

template <int NMF, int NW, int STAGES>
void launch_t(const Params& p, cudaStream_t stream) {
    constexpr int MT = NMF * 8;
    constexpr int STAGE_BYTES = NW * 1536 + MT * 128;
    size_t ring = (size_t)STAGES * STAGE_BYTES;
    size_t tile = (size_t)MT * NW * 16 * 4;
    if (tile > ring) ring = tile;
    size_t smem = ring + tile + (size_t)(NW / 4) * MAXE * 8;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_kernel<NMF, NW, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    dim3 grid(p.N / (NW * 16), p.splits);
    gemm_kernel<NMF, NW, STAGES><<<grid, NW * 32, smem, stream>>>(p);
}

}  // namespace contrarian

void contrarian_launch(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor scratch, torch::Tensor counters,
                       torch::Tensor kadd, torch::Tensor exc_off, torch::Tensor exc, int64_t KW, int64_t splits,
                       int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode) {
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
    p.counters = counters.data_ptr<int>();
    p.exc_off = exc_off.data_ptr<int>();
    p.exc = reinterpret_cast<const int2*>(exc.data_ptr());
    p.KB = p.KW / 256;
    p.splits = (int)splits;
    p.fp32_out = out.scalar_type() == torch::kFloat32 ? 1 : 0;
    p.out32 = p.fp32_out ? out.data_ptr<float>() : nullptr;
    p.out = p.fp32_out ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.mode = (int)mode;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define DISPATCH(NMF_, NW_, ST_) \
    if (nmf == NMF_ && nwarps == NW_ && stages == ST_) { launch_t<NMF_, NW_, ST_>(p, stream); return; }
    DISPATCH(1, 4, 8) DISPATCH(2, 4, 8) DISPATCH(4, 4, 8) DISPATCH(8, 4, 4) DISPATCH(16, 4, 4)
    DISPATCH(1, 8, 6) DISPATCH(2, 8, 6) DISPATCH(4, 8, 6) DISPATCH(8, 8, 4) DISPATCH(16, 8, 4)
    DISPATCH(1, 4, 12) DISPATCH(2, 4, 12) DISPATCH(4, 4, 12)
#undef DISPATCH
    TORCH_CHECK(false, "no kernel instance for nmf/nwarps/stages");
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void contrarian_launch(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor scratch, torch::Tensor counters,
                       torch::Tensor kadd, torch::Tensor exc_off, torch::Tensor exc, int64_t KW, int64_t splits,
                       int64_t nwarps, int64_t nmf, int64_t stages, int64_t mode);
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="contrarian_v1",
            cpp_sources=[CPP_SRC],
            cuda_sources=["#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["contrarian_launch"],
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
    Lb = torch.where(exc, torch.zeros_like(d), ((dd & 1) << 7) | M).to(torch.uint8)
    Hb = torch.where(exc, torch.zeros_like(d), (s << 3) | (dd >> 1)).to(torch.uint8)
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
    Wr = torch.cat([hi, lo], dim=1).contiguous()
    G = Wr.shape[1] // 64
    pk = _pack_layer(Wr, [(0, G // 2), (G // 2, G)])
    state["router"] = pk
    print(f"PACK router: {pk['n_exc']} exceptions, {pk['packed'].numel()/1e6:.1f} MB", flush=True)
    state["scratch"] = torch.zeros(128 * 8192, dtype=torch.float32, device=dev)
    state["counters"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    return state


import os
_OVR = os.environ.get("CONTRARIAN_CFG", "")  # "nw,splits,stages" override for experiments
_MODE = int(os.environ.get("CONTRARIAN_MODE", "0"))


# (layer, mt) -> (nwarps, splits, stages)
def _config(layer, n, mt):
    if _OVR:
        a = [int(v) for v in _OVR.split(",")]
        return a[0], a[1], a[2]
    if mt <= 32:
        nw, st, target = 4, 8, 256
    else:
        nw, st, target = 8, 4, 128
    tiles = n // (nw * 16)
    kw = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 16384}[layer]
    splits = 1
    while tiles * splits < target and (kw // (splits * 2)) % 256 == 0:
        splits *= 2
    return nw, splits, st


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n, m = cfg["layer"], cfg["n"], x.shape[0]
    if m > 128:
        w = state["orig"][layer][:n]
        if layer == "router":
            return torch.mm(x.float(), w.T)
        return torch.mm(x, w.T)
    pk = state[layer]
    nmf = 1
    while nmf * 8 < m:
        nmf *= 2
    mt = nmf * 8
    nw, splits, st = _config(layer, n, mt)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().contrarian_launch(pk["packed"], x, out, state["scratch"], state["counters"], pk["kadd"], pk["exc_off"], pk["exc"],
                                pk["KW"], splits, nw, nmf, st, _MODE)
    return out
