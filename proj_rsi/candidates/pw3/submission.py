# okc: problem=muse_bf16_proj
"""pw1: packed-weight streaming tensor-core GEMM for Muse Spark's dense projections, rewritten for these exact shapes.

Structure every candidate keeps:
  * PATHS: a registry of fast paths. Each path names the shapes it serves and builds a callable (x, out) -> None.
  * prepare(): compiles, packs the weights, then for every scored (layer, m) VALIDATES each applicable path
    (one-hot columns exact, random rows within tolerance) and TIMES it against the torch fallback under
    CUDA-graph replay with a cold L2. The fastest correct path wins with a 1% margin; the table is printed as
    'BAKE <layer> m=<m> <path> <us>' lines so the judge's log tail carries every variant.
  * kernel_fn(): a dict lookup on (layer, m). Anything not selected (every odd m) runs the torch fallback.
A path that fails validation is never selected, so a check cannot die on a kernel bug; it just measures the fallback.

pw3 kernel family ("pw"), out[m, n] = x[m, K] @ W[n, K]^T for qkv_gate / o / pooler / unpooler at m in {1, 8, 16, 32, 128}:
  * Weights: r6's judge-proven lossless 12-bit codec (_pack + _c12, A-fragment order, swap-AB), 1536 B units laid out
    per BN-row block as [n-block][chunk][tile-in-block][1536]; escapes fixed in registers from per-warp lists.
  * m <= 32 ("b", pw2's design): grid (NG, KS); each CTA copies x[:, K-slice] into shared memory once and streams W for
    a range of n-blocks through a W-only cp.async.bulk ring (producer warp, full/empty mbarriers, runtime depth ST).
  * m = 128 ("t", new): grid (NB, KS <= 4); per 64-k stage ONE 2D TMA copy of the x tile [128 rows x 64 k] (128B swizzle,
    rows >= m zero-filled) plus one bulk copy of the W chunk land on the same full barrier. The tensor map is encoded
    on the host per call (cudaGetDriverEntryPoint, no -lcuda) and passed as a __grid_constant__ parameter; pw-tms
    instead copies x into a fixed staging buffer first.
  * Split-K partials: red.global.add.v4.f32 into a fragment-order fp32 workspace; the last arriver per n-block converts,
    writes out, re-zeroes its slice and counter. KS == 1 stores directly.
  * prepare() first runs PROBE cons lines: the bulk kernel with its consumer work switched on step by step.
"""

from typing import TypedDict
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; n=1024 (rows 0..1023) when m <= 8
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
SCORED_M = (1, 8, 16, 32, 128, 256)
PREPARE_BUDGET_S = 240.0  # the judge allows 300 s; leave room for the final sanity pass
MARGIN = 0.99             # a path must beat the fallback by 1% to be selected
TOL = {"bf16": 4e-3, "f32": 2e-5}  # random-row relative L2 for validation (judge: 1e-2 and 1e-4, rows 5x)
_T0 = time.time()
_EXT = None


def _log(*parts):
    print(*parts, flush=True)


def _n_of(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}[layer]


# ----------------------------------------------------------------------------------------------------------------
# pw3 instances.
#  kind "b" (x-stationary bulk, pw2's design): cfg (RT, WN, WTOK, TTW, CPS, LV, AG, MINB); ST (smem ring depth) is a
#    runtime argument. LV = consumer work level (1 wait/arrive only, 2 +smem loads, 3 +decode, 4 +mma without escape
#    fix, 5 full); AG = stages per empty-barrier arrive (1 or 2). Only LV5/AG1 instances are GEMM paths; the rest are
#    PROBE-only (consumer-cost decomposition).
#  kind "t" (x streamed by one 2D TMA tile per stage, m=128): cfg (RT, WN, WTOK, TTW, ST).
#  Entry: (name, kind, cfg, ms served as a path (() = probe only), CTAs/SM, default ST)
# ----------------------------------------------------------------------------------------------------------------
PW_INSTS = [
    ("b8t1", "b", (1, 8, 1, 1, 2, 5, 1, 1), (1, 8), 1, 4),
    ("b8t2", "b", (1, 8, 1, 2, 2, 5, 1, 1), (16,), 1, 4),
    ("b8t4", "b", (1, 8, 1, 4, 2, 5, 1, 1), (32,), 1, 4),
    ("tm", "t", (2, 4, 2, 8, 6), (128,), 1, 6),
] + [("p8t%dL%d" % (t, lv), "b", (1, 8, 1, t, 2, lv, 1, 1), (), 1, 4) for t in (1, 4) for lv in (1, 2, 3, 4)] \
  + [("p16t%dL%d" % (t, lv), "b", (1, 16, 1, t, 1, lv, 1, 1), (), 1, 4) for t in (1, 4) for lv in (1, 2, 3, 4, 5)] \
  + [("p8t%dAG2" % t, "b", (1, 8, 1, t, 2, 5, 2, 1), (), 1, 4) for t in (1, 4)]
PW_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
PW_SMEM = 220 * 1024   # dynamic smem cap per CTA (opt-in max is 227 KB incl. static)
PW_SEQ = 32000         # escape-list step field (15 bits)
PW_BIDX = [i for i, e in enumerate(PW_INSTS) if e[1] == "b"]
PW_TIDX = [i for i, e in enumerate(PW_INSTS) if e[1] == "t"]

# ----------------------------------------------------------------------------------------------------------------
# CUDA sources. KERNELS_SRC is pure CUDA (no torch headers) so it can be compiled standalone for ptxas -v;
# GLUE_SRC is the torch launch glue. CUDA_SRC = KERNELS_SRC + GLUE_SRC, compiled by one load_inline.
# No -std flag: the judge's ATen needs C++20 and load_inline already sets it. sm_90a is required for bulk/TMA copies.
# ----------------------------------------------------------------------------------------------------------------
_KERNELS_TEMPLATE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x row length K; W chunk c reads x columns (c * 64) % K (K-wrap hook for a later router mode).
struct PwP {
  const uint8_t* W;            // packed [NBW][KCW][RTB][1536]
  const __nv_bfloat16* x;      // [m, K]
  __nv_bfloat16* out;          // [m, N]
  float* ws;                   // fp32 split-K workspace, fragment order [NB][RTB][TT][32][4], zero between calls
  int* cnt;                    // [NB] arrival counters, zero between calls
  const uint32_t* esc;         // escape entries, one sentinel-terminated (0xffffffff) list per consumer warp
  const int* eoff;             // [KS][NG][NWC] start of each warp's list
  long long wsn;               // workspace floats (bounds check)
  int cntn, eoffn;             // counter / offset entries (bounds check)
  int m, N, K, KCW, RTBW, NBW;
  int xrow;                    // x-stationary smem bytes per x row: KCSmax * 128 + 16
  int st;                      // smem ring depth (runtime)
  uint32_t Tb;                 // decode window base, replicated per byte
};
// opaque 128-byte CUtensorMap (encoded on the host); passed by value as a __grid_constant__ kernel parameter
struct alignas(64) PwTmap { unsigned long long v[16]; };

__device__ __forceinline__ uint4 pw_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t pw_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// m16n8k16 bf16, fp32 accumulate. A: {row g k-lo, row g+8 k-lo, row g k-hi, row g+8 k-hi} (judge-proven order)
__device__ __forceinline__ void pw_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// r6 c12 decode (variant 1)
__device__ __forceinline__ void pw_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = pw_prmt(L0, hl, 0x5140); a[1] = pw_prmt(L0, hl, 0x7362); a[2] = pw_prmt(L1, hh, 0x5140); a[3] = pw_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void pw_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  pw_dec(q0.x, q0.y, q2.x, Tb, A); pw_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  pw_dec(q1.x, q1.y, q2.z, Tb, A + 8); pw_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
__device__ __forceinline__ void pw_mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void pw_mbwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nPW_WAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra PW_WAIT_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void pw_mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void pw_mbexpect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void pw_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// one 2D TMA tile {64 k, box rows} at (k0, row0) into 1024-aligned smem, 128B-swizzled, rows past the map's m zero-filled
__device__ __forceinline__ void pw_tile(uint32_t dst, const PwTmap* tm, int k0, int row0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(row0), "r"(bar) : "memory");
}
__device__ __forceinline__ void pw_cpa(uint32_t dst, const void* src, uint32_t sz) {  // 16 B, zero-filled past sz
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void pw_red4(float4* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void pw_put(__nv_bfloat16* out, int N, int m, int n, int tok, float4 v) {
  if (tok < m) { out[(size_t)tok * N + n] = __float2bfloat16(v.x); out[(size_t)tok * N + n + 8] = __float2bfloat16(v.z); }
  if (tok + 1 < m) { out[(size_t)(tok + 1) * N + n] = __float2bfloat16(v.y); out[(size_t)(tok + 1) * N + n + 8] = __float2bfloat16(v.w); }
}
// escape fix: entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = (warp step) * RT + j
template <int RT>
__device__ __forceinline__ void pw_fix(uint32_t (&A)[RT][16], int seq0, uint32_t& nx, int& ep, const uint32_t* esc, int lane) {
#pragma unroll
  for (int j = 0; j < RT; j++) {
    while ((nx >> 17) == (uint32_t)(seq0 + j)) {
      if (lane == (int)((nx >> 12) & 31)) {
        const uint32_t r = (nx >> 8) & 15, xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
        for (int i = 0; i < 16; i++) if (i == (int)r) A[j][i] ^= xr;
      }
      nx = __ldg(esc + (++ep));
    }
  }
}
// all mma of one 64-wide chunk. Padded rows (x-stationary): row tile row g at xa + tt*8*xrow, bytes t*32 .. +32.
template <int RT, int TTW>
__device__ __forceinline__ void pw_mmas(float (&acc)[RT][TTW][4], const uint32_t (&A)[RT][16], uint32_t xa, int xrow) {
#pragma unroll
  for (int tt = 0; tt < TTW; tt++) {
    const uint4 b0 = pw_lds(xa + tt * 8 * xrow), b1 = pw_lds(xa + tt * 8 * xrow + 16);
    const uint32_t B[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++) pw_mma(acc[j][tt], A[j] + 4 * q, B[2 * q], B[2 * q + 1]);
  }
}
// 128B-swizzled TMA tile (128 B rows): row R's 16 B chunk c sits at R*128 + ((c ^ (R & 7)) << 4); R & 7 == g here
template <int RT, int TTW>
__device__ __forceinline__ void pw_mmas_sw(float (&acc)[RT][TTW][4], const uint32_t (&A)[RT][16], uint32_t xrow0, int g, int t) {
  const uint32_t o0 = ((2 * t) ^ g) << 4, o1 = ((2 * t + 1) ^ g) << 4;
#pragma unroll
  for (int tt = 0; tt < TTW; tt++) {
    const uint32_t ra = xrow0 + tt * 8 * 128;
    const uint4 b0 = pw_lds(ra + o0), b1 = pw_lds(ra + o1);
    const uint32_t B[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++) pw_mma(acc[j][tt], A[j] + 4 * q, B[2 * q], B[2 * q + 1]);
  }
}
// split-K epilogue shared by both kernels: per n-block reductions, then arrivals; the last arriver converts and re-zeroes
template <int RT, int TTW, int WTOK, int RTB, int NTH>
__device__ __forceinline__ void pw_flush(const PwP& p, float (&acc)[RT][TTW][4], int nb, int wn, int wt, int lane, int KS) {
  constexpr int TT = WTOK * TTW, BN = RTB * 16;
  const int g = lane >> 2, t = lane & 3;
  float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) {
      const int T2 = wt * TTW + tt;
      if (KS == 1) pw_put(p.out, p.N, p.m, nb * BN + (wn * RT + j) * 16 + g, T2 * 8 + 2 * t,
                          make_float4(acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]));
      else if (T2 * 8 + 2 * t < p.m)
        pw_red4(ws4 + ((wn * RT + j) * TT + T2) * 32 + lane, acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]);
      acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
    }
}
template <int TT, int RTB, int NTH>
__device__ __forceinline__ void pw_arrive(const PwP& p, int nb0, int NBR, int KS, int* s_list, int* s_n) {
  const int tid = threadIdx.x;
  constexpr int BN = RTB * 16;
  __threadfence();
  if (tid == 0) *s_n = 0;
  __syncthreads();
  for (int i = tid; i < NBR; i += NTH)
    if (atomicAdd(p.cnt + nb0 + i, 1) == KS - 1) s_list[atomicAdd(s_n, 1)] = nb0 + i;
  __syncthreads();
  const int nl = *s_n;
  if (nl == 0) return;
  __threadfence();
  for (int k = 0; k < nl; k++) {
    const int nb = s_list[k];
    float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
    for (int i = tid; i < RTB * TT * 32; i += NTH) {
      const int l = i & 31, T2 = (i >> 5) % TT, r = (i >> 5) / TT, tok = T2 * 8 + 2 * (l & 3);
      if (tok < p.m) {
        const float4 v = __ldcg(ws4 + i);
        __stcg(ws4 + i, make_float4(0.f, 0.f, 0.f, 0.f));
        pw_put(p.out, p.N, p.m, nb * BN + r * 16 + (l >> 2), tok, v);
      }
    }
    if (tid == 0) p.cnt[nb] = 0;
  }
}

// ---------------------------------------------------------------- kind "b": x-stationary, W-only bulk ring (pw2)
// Grid (NG, KS): CTA (ng, ks) owns n-blocks [ng*NB/NG, (ng+1)*NB/NG) and K stages [ks*NST/KS, (ks+1)*NST/KS).
template <int RT, int WN, int WTOK, int TTW, int CPS>
struct PwG {
  static constexpr int NWC = WN * WTOK, NTH = (NWC + 1) * 32, RTB = WN * RT, BN = RTB * 16, TT = WTOK * TTW, TOK = TT * 8;
  static constexpr int WB = CPS * RTB * 1536;  // W bytes per stage
};
__host__ __device__ inline int pw_xoff(int ST, int WB) { return (ST * WB + ST * 16 + 127) / 128 * 128; }

template <int RT, int WN, int WTOK, int TTW, int CPS, int LV, int AG, int MINB>
__global__ void __launch_bounds__(PwG<RT, WN, WTOK, TTW, CPS>::NTH, MINB) pwk(const PwP p) {
  using G = PwG<RT, WN, WTOK, TTW, CPS>;
  constexpr int NWC = G::NWC, NTH = G::NTH, RTB = G::RTB, BN = G::BN, TOK = G::TOK, WB = G::WB, TT = G::TT;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_list[256];
  __shared__ int s_n;
  const int ST = p.st;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = sbase + ST * WB, empty = full + ST * 8, xs = sbase + pw_xoff(ST, WB);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ng = blockIdx.x, ks = blockIdx.y, NG = gridDim.x, KS = gridDim.y, m = p.m, xrow = p.xrow;
  const int NB = p.N / BN, nb0 = (int)((long long)ng * NB / NG), nb1 = (int)((long long)(ng + 1) * NB / NG), NBR = nb1 - nb0;
  const int NST = p.KCW / CPS, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int SPB = s1 - s0, KCS = SPB * CPS, c0 = s0 * CPS, NSTOT = NBR * SPB;
  const int wn = warp % WN, wt = warp / WN;
  const size_t cstr = (size_t)RTB * 1536;
  if (tid == NWC * 32) {
    for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, NWC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int s = 0; s < ST && s < NSTOT; s++) {
      const int nb = nb0 + s / SPB, ss = s - (s / SPB) * SPB;
      pw_mbexpect(full + s * 8, WB);
      pw_bulk(sbase + s * WB, p.W + ((size_t)nb * p.KCW + c0 + ss * CPS) * cstr, WB, full + s * 8);
    }
  }
  {
    const int segs = KCS * 8;
    for (int e = tid; e < TOK * segs; e += NTH) {
      const int r = e / segs, sg = e - r * segs;
      const int col = ((c0 + (sg >> 3)) * 64) % p.K + (sg & 7) * 8;
      pw_cpa(xs + r * xrow + sg * 16, p.x + (size_t)(r < m ? r : 0) * p.K + col, r < m ? 16u : 0u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  __syncthreads();
  float acc[RT][TTW][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
  uint32_t sink = 0;
  if (warp == NWC) {
    // producer: stages ST.. of the W stream (lane 0 issues, the warp waits together)
    int slot = 0, round = 1, nb = nb0, ss = 0;  // cursor of stage ST (the prologue issued stages 0..ST-1)
    for (int s = 0; s < ST && s < NSTOT; s++) if (++ss == SPB) { ss = 0; nb++; }
    for (int s = ST; s < NSTOT; s++) {
      const uint32_t eb = AG == 2 ? empty + (slot >> 1) * 8 : empty + slot * 8;
      pw_mbwait(eb, (round - 1) & 1);
      if (lane == 0) {
        pw_mbexpect(full + slot * 8, WB);
        pw_bulk(sbase + slot * WB, p.W + ((size_t)nb * p.KCW + c0 + ss * CPS) * cstr, WB, full + slot * 8);
      }
      __syncwarp();
      if (++ss == SPB) { ss = 0; nb++; }
      if (++slot == ST) { slot = 0; round++; }
    }
  } else {
    int ep = __ldg(p.eoff + (ks * NG + ng) * NWC + warp);
    uint32_t nx = __ldg(p.esc + ep);
    const uint32_t xb = xs + (wt * TTW * 8 + g) * xrow + t * 32;
    int cnb = nb0, ss = 0, slot = 0;
    uint32_t phase = 0;
#pragma unroll 1
    for (int s = 0; s < NSTOT; s++) {
      pw_mbwait(full + slot * 8, phase);
      const uint32_t st = sbase + slot * WB;
#pragma unroll
      for (int cc = 0; cc < CPS; cc++) {
        const int u = ss * CPS + cc;
        if constexpr (LV >= 2) {
          uint32_t A[RT][16];
#pragma unroll
          for (int j = 0; j < RT; j++) {
            const uint32_t a = st + (cc * RTB + wn * RT + j) * 1536 + lane * 16;
            const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
            if constexpr (LV == 2) sink ^= q0.x ^ q0.y ^ q0.z ^ q0.w ^ q1.x ^ q1.y ^ q1.z ^ q1.w ^ q2.x ^ q2.y ^ q2.z ^ q2.w;
            else pw_unit(q0, q1, q2, p.Tb, A[j]);
          }
          if constexpr (LV == 3) {
#pragma unroll
            for (int j = 0; j < RT; j++)
#pragma unroll
              for (int i = 0; i < 16; i++) sink ^= A[j][i];
          }
          if constexpr (LV == 5) pw_fix<RT>(A, ((cnb - nb0) * KCS + u) * RT, nx, ep, p.esc, lane);
          if constexpr (LV >= 4) pw_mmas<RT, TTW>(acc, A, xb + u * 128, xrow);
        }
      }
      if (AG == 1 || (slot & 1) || s == NSTOT - 1) {
        __syncwarp();
        if (lane == 0) pw_mbarrive(AG == 2 ? empty + (slot >> 1) * 8 : empty + slot * 8);
      }
      if (++ss == SPB) { pw_flush<RT, TTW, WTOK, RTB, NTH>(p, acc, cnb, wn, wt, lane, KS); ss = 0; cnb++; }
      if (++slot == ST) { slot = 0; phase ^= 1; }
    }
  }
  if (LV < 4 && sink == 0x9e3779b9u) p.out[0] = __float2bfloat16(0.f);  // keeps the probe's loads/decode live
  if (KS == 1) return;
  pw_arrive<TT, RTB, NTH>(p, nb0, NBR, KS, s_list, &s_n);
}

template <int RT, int WN, int WTOK, int TTW, int CPS, int LV, int AG, int MINB>
static int pw_launch(const PwP& p, int NG, int KS, cudaStream_t st) {
  using G = PwG<RT, WN, WTOK, TTW, CPS>;
  if (p.m < 1 || p.m > G::TOK || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.KCW % CPS || p.K % 64 || (p.KCW * 64) % p.K || KS < 1 || NG < 1 || p.st < 1 || p.st > 8 || (AG == 2 && p.st % 2)) return -5;
  const int NB = p.N / G::BN, NST = p.KCW / CPS, KCSM = (NST + KS - 1) / KS * CPS, NBRM = (NB + NG - 1) / NG;
  if (KS > NST || NG > NB || NBRM > 256 || (long long)NBRM * KCSM * RT >= 32000) return -6;
  if (p.xrow < KCSM * 128 + 16 || p.xrow % 128 != 16) return -6;
  if (p.wsn < (long long)p.N * G::TOK || p.cntn < NB || p.eoffn < KS * NG * G::NWC) return -7;
  const long long smem = (long long)pw_xoff(p.st, G::WB) + (long long)G::TOK * p.xrow;
  if (smem > 220 * 1024) return -8;
  auto k = pwk<RT, WN, WTOK, TTW, CPS, LV, AG, MINB>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  k<<<dim3(NG, KS), G::NTH, (size_t)smem, st>>>(p);
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------- kind "t": x streamed by 2D TMA, m = 128
// Grid (NB, KS): CTA = one BN-row n-block x one K-slice. Stage = [x tile TOK x 64 k, 128B-swizzled][W chunk RTB x 1536],
// both completing on one full barrier (expect_tx = TOK*128 + RTB*1536).
template <int RT, int WN, int WTOK, int TTW, int ST>
struct PwT {
  static constexpr int NWC = WN * WTOK, NTH = (NWC + 1) * 32, RTB = WN * RT, BN = RTB * 16, TT = WTOK * TTW, TOK = TT * 8;
  static constexpr int XB = TOK * 128, WB = RTB * 1536, SB = (XB + WB + 1023) / 1024 * 1024;
  static constexpr int SMEM = 1024 + ST * SB + ST * 16;  // + 1024 to align the ring base for the 128B swizzle
};

template <int RT, int WN, int WTOK, int TTW, int ST>
__global__ void __launch_bounds__(PwT<RT, WN, WTOK, TTW, ST>::NTH, 1) pwt(const PwP p, const __grid_constant__ PwTmap tm) {
  using G = PwT<RT, WN, WTOK, TTW, ST>;
  constexpr int NWC = G::NWC, NTH = G::NTH, RTB = G::RTB, TT = G::TT, XB = G::XB, WB = G::WB, SB = G::SB;
  extern __shared__ __align__(128) uint8_t smem[];  // ring base re-aligned to 1024 below
  __shared__ int s_list[1];
  __shared__ int s_n;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t full = sbase + ST * SB, empty = full + ST * 8;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int nb = blockIdx.x, ks = blockIdx.y, KS = gridDim.y;
  const int NST = p.KCW, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int KCS = s1 - s0, c0 = s0;
  const int wn = warp % WN, wt = warp / WN;
  const size_t cstr = (size_t)RTB * 1536;
  if (tid == NWC * 32) {
    for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, NWC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float acc[RT][TTW][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
  if (warp == NWC) {
    if (lane == 0) {
      int slot = 0, round = 0;
      for (int s = 0; s < KCS; s++) {
        if (round > 0) pw_mbwait(empty + slot * 8, (round - 1) & 1);
        const uint32_t dst = sbase + slot * SB, bar = full + slot * 8;
        pw_mbexpect(bar, XB + WB);
        pw_tile(dst, &tm, ((c0 + s) * 64) % p.K, 0, bar);
        pw_bulk(dst + XB, p.W + ((size_t)nb * p.KCW + c0 + s) * cstr, WB, bar);
        if (++slot == ST) { slot = 0; round++; }
      }
    }
    __syncwarp();
  } else {
    int ep = __ldg(p.eoff + (ks * gridDim.x + nb) * NWC + warp);
    uint32_t nx = __ldg(p.esc + ep);
    int slot = 0;
    uint32_t phase = 0;
#pragma unroll 1
    for (int u = 0; u < KCS; u++) {
      pw_mbwait(full + slot * 8, phase);
      const uint32_t st = sbase + slot * SB;
      uint32_t A[RT][16];
#pragma unroll
      for (int j = 0; j < RT; j++) {
        const uint32_t a = st + XB + (wn * RT + j) * 1536 + lane * 16;
        const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
        pw_unit(q0, q1, q2, p.Tb, A[j]);
      }
      pw_fix<RT>(A, u * RT, nx, ep, p.esc, lane);
      pw_mmas_sw<RT, TTW>(acc, A, st + (wt * TTW * 8 + g) * 128, g, t);
      __syncwarp();
      if (lane == 0) pw_mbarrive(empty + slot * 8);
      if (++slot == ST) { slot = 0; phase ^= 1; }
    }
    pw_flush<RT, TTW, WTOK, RTB, NTH>(p, acc, nb, wn, wt, lane, KS);
  }
  if (KS == 1) return;
  pw_arrive<TT, RTB, NTH>(p, nb, 1, KS, s_list, &s_n);
}

template <int RT, int WN, int WTOK, int TTW, int ST>
static int pw_tlaunch(const PwP& p, const PwTmap& tm, int KS, cudaStream_t st) {
  using G = PwT<RT, WN, WTOK, TTW, ST>;
  if (p.m < 1 || p.m > G::TOK || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.K % 64 || p.KCW * 64 != p.K || KS < 1 || KS > p.KCW) return -5;
  const int NB = p.N / G::BN;
  if (((p.KCW + KS - 1) / KS) * RT >= 32000 || p.wsn < (long long)p.N * G::TOK || p.cntn < NB || p.eoffn < KS * NB * G::NWC) return -7;
  auto k = pwt<RT, WN, WTOK, TTW, ST>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  k<<<dim3(NB, KS), G::NTH, G::SMEM, st>>>(p, tm);
  return (int)cudaGetLastError();
}

int pw_dispatch(int inst, const PwP& p, int NG, int KS, cudaStream_t st) {
  switch (inst) {
@@BCASES@@
  }
  return -2;
}
int pw_tdispatch(int inst, const PwP& p, const PwTmap& tm, int KS, cudaStream_t st) {
  switch (inst) {
@@TCASES@@
  }
  return -2;
}
__global__ void pw_probe_empty(uint32_t* sink) { if (threadIdx.x == 0) sink[blockIdx.x] = 1; }
int pw_empty_launch(uint32_t* sink, int grid, cudaStream_t st) { pw_probe_empty<<<grid, 32, 0, st>>>(sink); return (int)cudaGetLastError(); }
"""

KERNELS_SRC = _KERNELS_TEMPLATE.replace("@@BCASES@@", "\n".join(
    "    case %d: return pw_launch<%s>(p, NG, KS, st);" % (i, ",".join(map(str, PW_INSTS[i][2]))) for i in PW_BIDX)).replace(
    "@@TCASES@@", "\n".join("    case %d: return pw_tlaunch<%s>(p, tm, KS, st);" % (i, ",".join(map(str, PW_INSTS[i][2]))) for i in PW_TIDX))

GLUE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

static_assert(sizeof(CUtensorMap) == sizeof(PwTmap), "CUtensorMap size");
typedef CUresult (*PwEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                               const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                               CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static int pw_fill(PwP& p, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
                   torch::Tensor esc, torch::Tensor eoff, int64_t xrow, int64_t st, int64_t Tb) {
  if (W.dim() != 4 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
  p.W = (const uint8_t*)W.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = (__nv_bfloat16*)out.data_ptr();
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>(); p.esc = (const uint32_t*)esc.data_ptr(); p.eoff = eoff.data_ptr<int>();
  p.wsn = ws.numel(); p.cntn = (int)cnt.numel(); p.eoffn = (int)eoff.numel();
  p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1);
  p.NBW = (int)W.size(0); p.KCW = (int)W.size(1); p.RTBW = (int)W.size(2); p.xrow = (int)xrow; p.st = (int)st;
  p.Tb = (uint32_t)(Tb & 0xffffffff);
  return 0;
}

int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t st, int64_t Tb) {
  PwP p;
  const int r = pw_fill(p, W, x, out, ws, cnt, esc, eoff, xrow, st, Tb);
  if (r) return r;
  return (int64_t)pw_dispatch((int)inst, p, (int)NG, (int)KS, at::cuda::getCurrentCUDAStream());
}

// x tensor map encoded on the host for this call's x pointer (box 64 k x box_rows rows, 128B swizzle, OOB rows zero).
// Under CUDA-graph capture the 128-byte map is baked into the kernel parameters exactly like a raw x pointer would be,
// so a replay is correct whenever the replay reuses the captured x storage (the condition every kernel argument has).
int64_t pw_run_tma(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
                   torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t box_rows, int64_t Tb) {
  static PwEncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
#else
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
#endif
    enc = (PwEncodeFn)fn;
  }
  PwP p;
  const int r = pw_fill(p, W, x, out, ws, cnt, esc, eoff, 16, 1, Tb);
  if (r) return r;
  if (((uintptr_t)p.x) % 16 || box_rows < 1 || box_rows > 256) return -21;
  PwTmap tm;
  cuuint64_t dims[2] = {(cuuint64_t)p.K, (cuuint64_t)p.m};
  cuuint64_t strides[1] = {(cuuint64_t)p.K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult e = enc((CUtensorMap*)&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p.x, dims, strides, box, estr,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (e != CUDA_SUCCESS) return -22;
  return (int64_t)pw_tdispatch((int)inst, p, tm, (int)KS, at::cuda::getCurrentCUDAStream());
}

int64_t pw_empty(torch::Tensor sink, int64_t grid) {
  if (sink.numel() < grid) return -10;
  return (int64_t)pw_empty_launch((uint32_t*)sink.data_ptr(), (int)grid, at::cuda::getCurrentCUDAStream());
}
"""

CUDA_SRC = KERNELS_SRC + GLUE_SRC

CPP_SRC = ("int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t st, int64_t Tb);\n"
           "int64_t pw_run_tma(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t box_rows, int64_t Tb);\n"
           "int64_t pw_empty(torch::Tensor sink, int64_t grid);")
FUNCTIONS = ["pw_run", "pw_run_tma", "pw_empty"]


def _ext():
    global _EXT
    if _EXT is None:
        flags = ["-O3", "-lineinfo", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"]
        _EXT = load_inline("pw_" + hex(hash(CUDA_SRC) & 0xffffffff)[2:], cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=FUNCTIONS, extra_cuda_cflags=flags, verbose=False)
    return _EXT


# ----------------------------------------------------------------------------------------------------------------
# Fallback: exactly the judge's baseline math (cuBLAS through torch; router FP32 up to 8 tokens, hi+lo pair above).
# ----------------------------------------------------------------------------------------------------------------
def _fallback(state, layer, n):
    if layer != "router":
        w = state["raw"][layer]
        def run(x, out):
            torch.matmul(x, w[:n].t(), out=out)
        return run
    w, hi_t, lo_t = state["raw"]["router"], state["router_hi_t"], state["router_lo_t"]
    def run(x, out):
        if x.shape[0] <= 8 or not x.is_cuda:
            torch.matmul(x.float(), w.t(), out=out)
        else:
            acc = torch.mm(x, hi_t, out_dtype=torch.float32)
            acc += torch.mm(x, lo_t, out_dtype=torch.float32)
            out.copy_(acc)
    return run


# ----------------------------------------------------------------------------------------------------------------
# Fast-path registry. Each entry: name, serves(layer, m) -> bool, build(state, layer, m) -> callable(x, out) or None.
# build() may return None to decline (e.g. pack failed its exactness proof). Add entries; do not touch the harness.
# ----------------------------------------------------------------------------------------------------------------
PATHS = []


def path(name, serves):
    def register(build):
        PATHS.append((name, serves, build))
        return build
    return register


# ---------------------------------------------------------------- r6's lossless 12-bit codec (judge-proven, unchanged)
def _pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


# code-word bit of each value's 3-bit offset and of its sign, per decode variant (pw uses variant 1, r6's dec1)
_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


def _c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    T, KC = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
    if not torch.equal(_unc12(data, tab, esc, dv), b):
        return None
    return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


def _unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    T, KC = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
    sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
    tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
    lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
    val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        t, c, j, l, vv, x = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()


# ---------------------------------------------------------------- pw packing, escape lists, grid table, paths
def _pw_c12(state, layer):
    cache = state.setdefault("pw_c12", {})
    if layer not in cache:
        try:
            r = _c12(_pack(state["raw"][layer]), 1)
        except Exception as error:
            _log("PW c12 %s failed: %s" % (layer, str(error).splitlines()[-1][:160] if str(error) else type(error).__name__))
            r = None
        if r is None:
            _log("PW c12 %s: pack not exact, pw declines" % layer)
        cache[layer] = r
    return cache[layer]


def _pw_layout(state, layer, rtb):
    """c12 units [T, KC, 3, 32, 16] -> [T/rtb, KC, rtb, 1536]: one n-block's K-slice is one contiguous byte range"""
    cache = state.setdefault("pw_lay", {})
    if (layer, rtb) not in cache:
        c = _pw_c12(state, layer)
        lay = None
        if c is not None:
            data = c[0]
            T, KC = data.shape[:2]
            if T % rtb == 0:
                lay = data.reshape(T // rtb, rtb, KC, 1536).permute(0, 2, 1, 3).contiguous()
        cache[(layer, rtb)] = lay
    return cache[(layer, rtb)]


def _pw_esc(esc6, n, geo, KC, NG, KS, dev):
    """per-consumer-warp escape lists for grid (NG, KS): (entries int32 incl. one sentinel per warp, offsets int32).
    geo = (RT, WN, WTOK, CPS). A warp's step i = (nb - nb0) * KCS + u walks its n-blocks in order, each over its K-slice."""
    RT, WN, WTOK, CPS = geo
    NWC, RTB = WN * WTOK, WN * RT
    NB, NST = n // (RTB * 16), KC // CPS
    G = KS * NG * NWC
    e = esc6[esc6[:, 0] < n // 16]
    if len(e):
        tile, ch, q, ln, v, xr = e.t()
        nb, r = tile // RTB, tile % RTB
        wn, j = r // RT, r % RT
        nbb = torch.arange(NG + 1, device=dev, dtype=torch.long) * NB // NG
        ng = torch.searchsorted(nbb, nb, right=True) - 1
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * NST // KS
        ks = torch.searchsorted(s0, ch // CPS, right=True) - 1
        kcs = (s0[ks + 1] - s0[ks]) * CPS
        u = ch - s0[ks] * CPS
        seq = ((nb - nbb[ng]) * kcs + u) * RT + j
        if int(seq.max()) >= PW_SEQ:
            return None
        ent = (seq << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        gw = torch.cat([(ks * NG + ng) * NWC + wt * WN + wn for wt in range(WTOK)])
        ent, seq = ent.repeat(WTOK), seq.repeat(WTOK)
        o = torch.argsort(gw * 65536 + seq)
        gw, ent = gw[o], ent[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cnt = torch.bincount(gw, minlength=G)
    else:
        cnt = torch.zeros(G, dtype=torch.long, device=dev)
    starts = torch.cumsum(cnt + 1, 0) - (cnt + 1)
    arr = torch.full((int((cnt + 1).sum()),), 0xffffffff, dtype=torch.long, device=dev)
    if len(e):
        arr[starts[gw] + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return arr.contiguous(), starts.int().contiguous()


def _pw_xoff(cfg, st):
    RT, WN, WTOK, TTW, CPS = cfg[:5]
    wb = CPS * WN * RT * 1536
    return (st * wb + st * 16 + 127) // 128 * 128


def _pw_grid(cfg, occ, st, n, K):
    """kind "b" (NG, KS, xrow): x slice + W ring must fit PW_SMEM / occ. Target one wave of 132 * occ CTAs (two waves
    for m <= 8 jobs under 16 MB of packed W). Among grids filling >= 85% of the target, minimise x re-read NG*m*K*2
    plus fp32 reductions KS*m*N*4 (weighted 2x; none when KS == 1)."""
    RT, WN, WTOK, TTW, CPS = cfg[:5]
    BN, TOK = WN * RT * 16, WTOK * TTW * 8
    NB, NST = n // BN, K // 64 // CPS
    T = 132 * occ * (2 if TOK == 8 and n * K * 1.5 < 16e6 else 1)
    cap = PW_SMEM // occ - _pw_xoff(cfg, st)
    fits = []
    for KS in range(1, NST + 1):
        xrow = -(-NST // KS) * CPS * 128 + 16
        if TOK * xrow > cap:
            continue
        NG = min(NB, max(1, T // KS))
        if -(-NB // NG) > 256 or -(-NB // NG) * (xrow - 16) // 128 * RT >= PW_SEQ:
            continue
        fits.append((NG, KS, xrow, NG * K * 2 + (2 * KS * n * 4 if KS > 1 else 0)))
    if not fits:
        return None
    full = [f for f in fits if f[0] * f[1] >= 0.85 * T]
    best = min(full, key=lambda f: f[3]) if full else max(fits, key=lambda f: f[0] * f[1])
    return best[:3]


def _pw_tks(NB, half=False):
    """kind "t" split-K: about one wave of 1-CTA/SM CTAs, capped at 4 (16 MB of red.v4 cost 6.5 us on the judge)"""
    ks = max(1, min(4, round(132 / NB)))
    return max(1, ks // 2) if half else ks


def _pw_runner(state, layer, m, idx, st=None, occ=None, half=False):
    """-> (run(x, out) -> launch code, description) or None. Workspace and counters are per call of this function."""
    name, kind, cfg, ms, occ0, st0 = PW_INSTS[idx]
    occ, st = occ or occ0, st or st0
    RT, WN, WTOK, TTW = cfg[:4]
    CPS = cfg[4] if kind == "b" else 1
    RTB, TOK = WN * RT, WTOK * TTW * 8
    raw = state["raw"][layer]
    n, K = _n_of(layer, m), raw.shape[1]
    KC = K // 64
    if n % (RTB * 16) or not (TOK - 8 < m <= TOK) or KC % CPS:
        return None
    NB = n // (RTB * 16)
    if kind == "b":
        grid = _pw_grid(cfg, occ, st, n, K)
        if grid is None:
            return None
        NG, KS, xrow = grid
    else:
        NG, KS, xrow = NB, _pw_tks(NB, half), 16
    lay = _pw_layout(state, layer, RTB)
    if lay is None:
        return None
    _, tb, esc6 = _pw_c12(state, layer)
    el = _pw_esc(esc6, n, (RT, WN, WTOK, CPS), KC, NG, KS, raw.device)
    if el is None:
        return None
    esc, eoff = el
    ws = torch.zeros((n * TOK,), dtype=torch.float32, device=raw.device)  # private: a broken variant cannot poison
    cnt = torch.zeros((NB,), dtype=torch.int32, device=raw.device)       # another's workspace
    ext = _ext()
    if kind == "b":
        def run(x, out):
            return ext.pw_run(idx, lay, x, out, ws, cnt, esc, eoff, NG, KS, xrow, st, tb)
        desc = "NG=%d KS=%d grid=%d st=%d xKB=%.0f" % (NG, KS, NG * KS, st, TOK * xrow / 1024)
    else:
        def run(x, out):
            return ext.pw_run_tma(idx, lay, x, out, ws, cnt, esc, eoff, KS, TOK, tb)
        desc = "NB=%d KS=%d grid=%d st=%d" % (NB, KS, NB * KS, st)
    return run, desc


PW_IDX = {e[0]: i for i, e in enumerate(PW_INSTS)}


def _pw_path(pname, iname, staged=False, half=False):
    idx = PW_IDX[iname]
    ms = PW_INSTS[idx][3]

    def serves(layer, m):
        return layer in PW_LAYERS and m in ms

    def build(state, layer, m):
        r = _pw_runner(state, layer, m, idx, half=half)
        if r is None:
            return None
        launch, desc = r
        base = _fallback(state, layer, _n_of(layer, m))
        _log("PW %s %s m=%d %s" % (pname, layer, m, desc))
        stage = torch.zeros((m, state["raw"][layer].shape[1]), dtype=torch.bfloat16, device=state["raw"][layer].device) if staged else None

        def run(x, out):
            if not x.is_contiguous() or x.data_ptr() % 16:
                base(x, out)
                return
            if staged:  # fixed x address for the tensor map (captured as a D2D memcpy node)
                stage.copy_(x)
                x = stage
            if launch(x, out):  # a host-side launch error code; the fallback keeps the output correct
                base(x, out)
        return run

    path(pname, serves)(build)


_pw_path("pw-b8t1", "b8t1")
_pw_path("pw-b8t2", "b8t2")
_pw_path("pw-b8t4", "b8t4")
_pw_path("pw-tm", "tm")
_pw_path("pw-tmh", "tm", half=True)
_pw_path("pw-tms", "tm", staged=True)


# ---------------------------------------------------------------- PROBE: consumer-cost decomposition (no selection)
PROBE_BUDGET_S = 15.0


def _probe(state, flush, sink):
    """The bulk x-stationary kernel on pooler at m=1 and m=32 with the consumer's work switched on step by step.
    PROBE cons m=<m> <config> L<level> <grid> <us>: L1 wait/arrive only, L2 +3x16 B smem loads per unit, L3 +decode,
    L4 +mma (escape fix compiled out, results wrong), L5 full; configs c8w1 = 8 consumer warps 1 CTA/SM (ST 4),
    c8w2 = same kernel ST 2 at 2 CTAs/SM, c16w1 = 16 consumer warps (BN 256, 1 unit per warp per stage), AG2 = L5 c8w1
    with one empty-barrier arrive per 2 stages. Same cold-L2 graph timing as the bake-off (includes ~5 us replay)."""
    t0 = time.time()
    ext = _ext()
    dev = flush.device
    sk = torch.zeros((1024,), dtype=torch.int32, device=dev)
    dummy = torch.empty((1, 1), device=dev)
    try:
        us = _time_us(lambda x, o: ext.pw_empty(sk, 132), dummy, dummy, flush, sink)
        _log("PROBE empty grid=132 %.2f" % us)
    except Exception as error:
        _log("PROBE empty ERROR %s" % str(error).splitlines()[-1][:120])
    gen = torch.Generator(device=dev).manual_seed(77)
    for m, tt in ((1, 1), (32, 4)):
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
        runs = []
        for cname, pre, st, occ in (("c8w1", "p8", 4, 1), ("c8w2", "p8", 2, 2), ("c16w1", "p16", 4, 1)):
            for lv in (1, 2, 3, 4, 5):
                iname = "b8t%d" % tt if pre == "p8" and lv == 5 else "%st%dL%d" % (pre, tt, lv)
                runs.append(("%s L%d" % (cname, lv), iname, st, occ))
        runs.append(("c8w1 AG2", "p8t%dAG2" % tt, 4, 1))
        for label, iname, st, occ in runs:
            if time.time() - t0 > PROBE_BUDGET_S:
                _log("PROBE cons m=%d %s skipped-budget" % (m, label))
                continue
            try:
                r = _pw_runner(state, "pooler", m, PW_IDX[iname], st=st, occ=occ)
                if r is None:
                    _log("PROBE cons m=%d %s declined" % (m, label))
                    continue
                launch, desc = r
                code = launch(x, out)
                if code:
                    _log("PROBE cons m=%d %s launch-error %d" % (m, label, code))
                    continue
                us = _time_us(lambda a, b: launch(a, b), x, out, flush, sink)
                _log("PROBE cons m=%d %s %s %.2f" % (m, label, desc.split(" st=")[0], us))
            except Exception as error:
                _log("PROBE cons m=%d %s ERROR %s" % (m, label, str(error).splitlines()[-1][:120]))
    _log("PROBE done in %.1fs" % (time.time() - t0))


# ----------------------------------------------------------------------------------------------------------------
# Harness: validation and cold-L2 graph timing. The judge's method: a read-only pass over a buffer larger than L2
# before every replay, paired alternation with the baseline; here each path is timed alone, median of `it` replays.
# ----------------------------------------------------------------------------------------------------------------
def _graph(fn, x, out):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x, out); fn(x, out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x, out)
    return g


def _time_us(fn, x, out, flush, sink, it=15):
    g = _graph(fn, x, out)
    times = []
    for _ in range(it):
        torch.sum(flush, 0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _validate(fn, w, m, n, f32, gen):
    """One-hot rows: every output must equal a weight column exactly (the judge's probe rows, 1e-3, but a lossless
    pack must be bit-exact). Random rows: relative L2 against FP64 within TOL. Returns (ok, message)."""
    K = w.shape[1]
    dt = torch.float32 if f32 else torch.bfloat16
    out = torch.empty((m, n), dtype=dt, device=w.device)
    exact = w[:n].t().to(dt) if f32 else w[:n].t()  # [K, n]
    rows = max(m, 1)
    eye = torch.eye(K, dtype=torch.bfloat16, device=w.device)
    # Cover every column of the weight: K one-hot rows in batches of m, scaled by a power of two like the judge.
    for start in range(0, K, rows):
        x = eye[start:start + rows]
        if x.shape[0] < rows:
            x = torch.cat((x, torch.zeros((rows - x.shape[0], K), dtype=x.dtype, device=x.device)))
        fn(x * 8.0, out)
        got = out[: min(rows, K - start)]
        want = exact[start:start + got.shape[0]] * 8.0
        if not torch.isfinite(got).all() or not torch.equal(got.to(torch.float32), want.to(torch.float32)):
            return False, "one-hot mismatch at column %d" % start
    x = torch.randn((rows, K), generator=gen, device=w.device).to(torch.bfloat16)
    fn(x, out)
    ref = x.double() @ w[:n].double().t()
    err = float((out.double() - ref).norm() / ref.norm())
    row = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
    tol = TOL["f32" if f32 else "bf16"]
    if not (err <= tol and row <= 5 * tol and torch.isfinite(out).all()):
        return False, "random rel L2 %.2e row %.2e > %.0e" % (err, row, tol)
    return True, "ok rel L2 %.1e" % err


def prepare(weights: Weights) -> dict:
    dev = weights["qkv_gate"].device
    state = {"raw": {k: w.contiguous() for k, w in weights.items()}, "sel": {}}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"], state["router_lo_t"] = hi.t().contiguous(), lo.t().contiguous()
    state["router_hi"], state["router_lo"] = hi.contiguous(), lo.contiguous()
    if not dev.type == "cuda":
        return state  # CPU: the fallback only (proj_rsi cpu-check exercises the dispatch)
    _log("PREP start device", torch.cuda.get_device_name(0))
    try:
        _ext()
        _log("PREP compiled in %.0fs" % (time.time() - _T0))
    except Exception as error:  # a compile failure leaves every shape on the fallback
        _log("PREP compile failed: %s" % str(error)[-1500:])
        return state
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)  # 256 MB read-only L2 flush
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    try:  # calibration only: never selects anything
        _probe(state, flush, sink)
    except Exception as error:
        _log("PROBE failed: %s" % str(error).splitlines()[-1][:160])
    for layer in LAYERS:
        w = state["raw"][layer]
        f32 = layer == "router"
        for m in SCORED_M:
            n, K = _n_of(layer, m), w.shape[1]
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            base = _fallback(state, layer, n)
            best_name, best_fn, best_us = "torch", base, _time_us(base, x, out, flush, sink)
            _log("BAKE %s m=%d torch %.2f" % (layer, m, best_us))
            for name, serves, build in PATHS:
                if not serves(layer, m):
                    continue
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    _log("BAKE %s m=%d %s skipped-budget" % (layer, m, name))
                    continue
                try:
                    fn = build(state, layer, m)
                    if fn is None:
                        _log("BAKE %s m=%d %s declined" % (layer, m, name))
                        continue
                    ok, msg = _validate(fn, w, m, n, f32, gen)
                    if not ok:
                        _log("BAKE %s m=%d %s INVALID %s" % (layer, m, name, msg))
                        continue
                    us = _time_us(fn, x, out, flush, sink)
                except Exception as error:
                    _log("BAKE %s m=%d %s ERROR %s" % (layer, m, name, str(error).splitlines()[-1][:160]))
                    continue
                _log("BAKE %s m=%d %s %.2f" % (layer, m, name, us))
                if us < best_us * MARGIN:
                    best_name, best_fn, best_us = name, fn, us
            state["sel"][(layer, m)] = (best_name, best_fn)
            _log("PICK %s m=%d %s %.2f" % (layer, m, best_name, best_us))
    _log("PREP done in %.0fs" % (time.time() - _T0))
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer, n, m = cfg["layer"], cfg["n"], x.shape[0]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    sel = state.get("sel", {}).get((layer, m))
    if sel is not None:
        sel[1](x, out)
    else:
        _fallback(state, layer, n)(x, out)
    return out
