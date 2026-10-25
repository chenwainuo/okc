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

pw2 kernel family ("pw", x-stationary), out[m, n] = x[m, K] @ W[n, K]^T for qkv_gate / o / pooler / unpooler at
m in {1, 8, 16, 32, 128}:
  * Weights: r6's judge-proven lossless 12-bit codec (_pack + _c12, A-fragment order, swap-AB: W rows are the mma M
    dimension, tokens the n8 dimension). A unit is one 16-row tile x one 64-wide K chunk = 1536 B, laid out per BN-row
    block as [n-block][chunk][tile-in-block][1536]. Escapes are fixed in registers from per-warp lists.
  * Grid (NG, KS): CTA (ng, ks) owns a K-slice and a RANGE of n-blocks. It copies x[:, K-slice] into shared memory once
    (cp.async, rows >= m zero-filled, padded rows so B-fragment loads are conflict-free), then streams W for each
    n-block in turn with no further x traffic.
  * W engines with no x coupling: MODE 0 "ring" = per-warp register ring of ld.global.nc.v4, D deep, no barriers;
    MODE 1 "bulk" = a producer warp streams W only through an ST-deep smem ring of 12-24 KB cp.async.bulk stages.
  * Each finished n-block: KS == 1 stores bf16 directly; else red.global.add.v4.f32 into a fragment-order fp32
    workspace. After the stream, one arrival per n-block; the last arriver converts, writes out, re-zeroes its slice
    and counter, so no memset and no second kernel.
  * prepare() first runs PROBE kernels (pure streaming, no decode/mma) and prints achieved TB/s per engine config.
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
# pw2 instances: name -> (MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB), the m it serves, CTAs/SM it is sized for.
#   MODE 0 ring: every warp streams its own W tiles through a D-deep register ring (no barriers, no producer)
#   MODE 1 bulk: one producer warp streams W only (CPS chunks x BN rows per stage) into an ST-deep smem ring
#   RT: 16-row W tiles per warp, WN: warps along N, WTOK: warps along tokens, TTW: n8 token tiles per warp
#   BN = WN*RT*16 rows per n-block, TOK = WTOK*TTW*8 token rows. x[:, K-slice] is copied into smem ONCE per CTA.
# Register budget (acc RT*TTW*4 + decoded A RT*16 + B 8 + ring D*RT*12 + 12); ptxas -v figures in hypothesis.json.
# ----------------------------------------------------------------------------------------------------------------
PW_INSTS = [
    ("r8t1d4", (0, 1, 8, 1, 1, 1, 1, 4, 2), (1, 8), 2),
    ("r8t1d8", (0, 1, 8, 1, 1, 1, 1, 8, 1), (1, 8), 1),
    ("b8t1", (1, 1, 8, 1, 1, 4, 2, 1, 1), (1, 8), 1),
    ("r8t2d4", (0, 1, 8, 1, 2, 1, 1, 4, 2), (16,), 2),
    ("r8t2d8", (0, 1, 8, 1, 2, 1, 1, 8, 1), (16,), 1),
    ("b8t2", (1, 1, 8, 1, 2, 4, 2, 1, 1), (16,), 1),
    ("r8t4d4", (0, 1, 8, 1, 4, 1, 1, 4, 2), (32,), 2),
    ("r8t4d8", (0, 1, 8, 1, 4, 1, 1, 8, 1), (32,), 1),
    ("b8t4", (1, 1, 8, 1, 4, 4, 2, 1, 1), (32,), 1),
    ("bm12", (1, 2, 4, 2, 8, 6, 1, 1, 1), (128,), 1),
    ("bm24", (1, 2, 4, 2, 8, 3, 2, 1, 1), (128,), 1),
    ("rm", (0, 1, 4, 2, 8, 1, 1, 4, 1), (128,), 1),
]
PW_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
PW_SMEM = 220 * 1024   # dynamic smem cap per CTA (opt-in max is 227 KB incl. static)
PW_SEQ = 32000         # escape-list step field (15 bits)

# ----------------------------------------------------------------------------------------------------------------
# CUDA sources. KERNELS_SRC is pure CUDA (no torch headers) so it can be compiled standalone for ptxas -v;
# GLUE_SRC is the torch launch glue. CUDA_SRC = KERNELS_SRC + GLUE_SRC, compiled by one load_inline.
# No -std flag: the judge's ATen needs C++20 and load_inline already sets it. sm_90a is required for bulk copies.
# ----------------------------------------------------------------------------------------------------------------
_KERNELS_TEMPLATE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Grid (NG, KS): CTA (ng, ks) owns n-blocks [ng*NB/NG, (ng+1)*NB/NG) and K stages [ks*NST/KS, (ks+1)*NST/KS).
// x row length K; W chunk c reads x columns (c * 64) % K, so a later router "K-wrap" (W' = [hi | lo] along K,
// KCW = 2K/64) needs no restructuring.
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
  int xrow;                    // smem bytes per x row: KCSmax * 128 + 16 (the pad keeps B loads conflict-free)
  uint32_t Tb;                 // decode window base, replicated per byte
};

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS>
struct PwG {
  static constexpr int NWC = WN * WTOK, NTH = (NWC + (MODE ? 1 : 0)) * 32, RTB = WN * RT, BN = RTB * 16;
  static constexpr int TT = WTOK * TTW, TOK = TT * 8;
  static constexpr int WB = MODE ? CPS * RTB * 1536 : 0;                     // W bytes per stage
  static constexpr int XOFF = MODE ? (ST * WB + ST * 16 + 127) / 128 * 128 : 0;  // W ring + barriers, then x
};

__device__ __forceinline__ uint4 pw_gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 pw_gl1(const void* p) {  // L1-allocating: WTOK > 1 warps share each unit
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 pw_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t pw_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// m16n8k16 bf16, fp32 accumulate. A: {row g k-lo, row g+8 k-lo, row g k-hi, row g+8 k-hi} (judge-proven order)
__device__ __forceinline__ void pw_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// r6 c12 decode (variant 1): C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i,
// bit 3 the sign of value 4 + (i + 3) % 4; high byte = (C & 0x87) + base (base <= 120, no carry)
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
// all mma of one 64-wide chunk: B fragments of token tile tt are 16 contiguous bf16 of row (tile row g) at t * 32 B
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

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS, int D, int MINB>
__global__ void __launch_bounds__(PwG<MODE, RT, WN, WTOK, TTW, ST, CPS>::NTH, MINB) pwk(const PwP p) {
  using G = PwG<MODE, RT, WN, WTOK, TTW, ST, CPS>;
  constexpr int NWC = G::NWC, NTH = G::NTH, RTB = G::RTB, BN = G::BN, TT = G::TT, TOK = G::TOK, WB = G::WB;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_list[256];
  __shared__ int s_n;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = sbase + ST * WB, empty = full + ST * 8, xs = sbase + G::XOFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ng = blockIdx.x, ks = blockIdx.y, NG = gridDim.x, KS = gridDim.y, m = p.m, xrow = p.xrow;
  const int NB = p.N / BN, nb0 = (int)((long long)ng * NB / NG), nb1 = (int)((long long)(ng + 1) * NB / NG), NBR = nb1 - nb0;
  const int NST = p.KCW / CPS, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int SPB = s1 - s0, KCS = SPB * CPS, c0 = s0 * CPS, NI = NBR * KCS, NSTOT = NBR * SPB;
  const int wn = warp % WN, wt = warp / WN;
  const size_t cstr = (size_t)RTB * 1536;  // bytes per chunk of one n-block
  // ---- W prologue first (independent of x), then the one-time x slice copy
  if constexpr (MODE == 1) {
    if (tid == NWC * 32) {
#pragma unroll
      for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, NWC); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      for (int s = 0; s < ST && s < NSTOT; s++) {
        const int nb = nb0 + s / SPB, ss = s - (s / SPB) * SPB;
        pw_mbexpect(full + s * 8, WB);
        pw_bulk(sbase + s * WB, p.W + ((size_t)nb * p.KCW + c0 + ss * CPS) * cstr, WB, full + s * 8);
      }
    }
  }
  uint4 wb[MODE ? 1 : D][RT][3];
  const uint8_t* wbase = p.W + (size_t)(wn * RT) * 1536 + lane * 16;
  int lnb = nb0, lu = 0;  // ring load cursor (n-block, chunk in slice)
  auto ldw = [&](int s) {
    const uint8_t* a = wbase + ((size_t)lnb * p.KCW + c0 + lu) * cstr;
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int i = 0; i < 3; i++) wb[s % (MODE ? 1 : D)][j][i] = WTOK > 1 ? pw_gl1(a + j * 1536 + i * 512) : pw_gnc(a + j * 1536 + i * 512);
    if (++lu == KCS) { lu = 0; lnb++; }
  };
  if constexpr (MODE == 0) {
#pragma unroll
    for (int s = 0; s < D; s++) if (s < NI) ldw(s);
  }
  {
    const int segs = KCS * 8;  // 16 B segments per row
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
  // one n-block done: KS == 1 stores bf16 directly, else fp32 vector reductions into the fragment-order workspace
  auto flush = [&](int nb) {
    float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int tt = 0; tt < TTW; tt++) {
        const int T2 = wt * TTW + tt;
        if (KS == 1) pw_put(p.out, p.N, m, nb * BN + (wn * RT + j) * 16 + g, T2 * 8 + 2 * t,
                            make_float4(acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]));
        else if (T2 * 8 + 2 * t < m)
          pw_red4(ws4 + ((wn * RT + j) * TT + T2) * 32 + lane, acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]);
        acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
      }
  };
  if (warp == NWC) {
    // producer warp (MODE 1): stages ST.. of the W stream; lane 0 issues, the warp waits together
    for (int s = ST; s < NSTOT; s++) {
      const int slot = s % ST;
      pw_mbwait(empty + slot * 8, ((s / ST) - 1) & 1);
      if (lane == 0) {
        const int nb = nb0 + s / SPB, ss = s - (s / SPB) * SPB;
        pw_mbexpect(full + slot * 8, WB);
        pw_bulk(sbase + slot * WB, p.W + ((size_t)nb * p.KCW + c0 + ss * CPS) * cstr, WB, full + slot * 8);
      }
      __syncwarp();
    }
  } else {
    int ep = __ldg(p.eoff + (ks * NG + ng) * NWC + warp);
    uint32_t nx = __ldg(p.esc + ep);
    const uint32_t xb = xs + (wt * TTW * 8 + g) * xrow + t * 32;
    if constexpr (MODE == 1) {
      int cnb = nb0, ss = 0;
#pragma unroll 1
      for (int s = 0; s < NSTOT; s++) {
        const int slot = s % ST;
        pw_mbwait(full + slot * 8, (s / ST) & 1);
        const uint32_t st = sbase + slot * WB;
#pragma unroll
        for (int cc = 0; cc < CPS; cc++) {
          const int u = ss * CPS + cc;
          uint32_t A[RT][16];
#pragma unroll
          for (int j = 0; j < RT; j++) {
            const uint32_t a = st + (cc * RTB + wn * RT + j) * 1536 + lane * 16;
            const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
            pw_unit(q0, q1, q2, p.Tb, A[j]);
          }
          pw_fix<RT>(A, ((cnb - nb0) * KCS + u) * RT, nx, ep, p.esc, lane);
          pw_mmas<RT, TTW>(acc, A, xb + u * 128, xrow);
        }
        __syncwarp();
        if (lane == 0) pw_mbarrive(empty + slot * 8);
        if (++ss == SPB) { flush(cnb); ss = 0; cnb++; }
      }
    } else {
      int cnb = nb0, cu = 0;
#pragma unroll 1
      for (int i0 = 0; i0 < NI; i0 += D) {
#pragma unroll
        for (int s = 0; s < D; s++) {
          const int i = i0 + s;
          if (i < NI) {
            uint4 cur[RT][3];
#pragma unroll
            for (int j = 0; j < RT; j++)
#pragma unroll
              for (int k = 0; k < 3; k++) cur[j][k] = wb[s][j][k];
            if (i + D < NI) ldw(s);
            uint32_t A[RT][16];
#pragma unroll
            for (int j = 0; j < RT; j++) pw_unit(cur[j][0], cur[j][1], cur[j][2], p.Tb, A[j]);
            pw_fix<RT>(A, i * RT, nx, ep, p.esc, lane);
            pw_mmas<RT, TTW>(acc, A, xb + cu * 128, xrow);
            if (++cu == KCS) { flush(cnb); cu = 0; cnb++; }
          }
        }
      }
    }
  }
  if (KS == 1) return;
  // ---- arrivals for every n-block of this CTA; the last arriver of an n-block converts, writes and re-zeroes it
  __threadfence();
  if (tid == 0) s_n = 0;
  __syncthreads();
  for (int i = tid; i < NBR; i += NTH)
    if (atomicAdd(p.cnt + nb0 + i, 1) == KS - 1) s_list[atomicAdd(&s_n, 1)] = nb0 + i;
  __syncthreads();
  const int nl = s_n;
  if (nl == 0) return;
  __threadfence();
  for (int k = 0; k < nl; k++) {
    const int nb = s_list[k];
    float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
    for (int i = tid; i < RTB * TT * 32; i += NTH) {
      const int l = i & 31, T2 = (i >> 5) % TT, r = (i >> 5) / TT, tok = T2 * 8 + 2 * (l & 3);
      if (tok < m) {
        const float4 v = __ldcg(ws4 + i);
        __stcg(ws4 + i, make_float4(0.f, 0.f, 0.f, 0.f));
        pw_put(p.out, p.N, m, nb * BN + r * 16 + (l >> 2), tok, v);
      }
    }
    if (tid == 0) p.cnt[nb] = 0;
  }
}

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS, int D, int MINB>
static int pw_launch(const PwP& p, int NG, int KS, cudaStream_t st) {
  using G = PwG<MODE, RT, WN, WTOK, TTW, ST, CPS>;
  if (p.m < 1 || p.m > G::TOK || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.KCW % CPS || p.K % 64 || (p.KCW * 64) % p.K || KS < 1 || NG < 1) return -5;
  const int NB = p.N / G::BN, NST = p.KCW / CPS, KCSM = (NST + KS - 1) / KS * CPS, NBRM = (NB + NG - 1) / NG;
  if (KS > NST || NG > NB || NBRM > 256 || (long long)NBRM * KCSM * RT >= 32000) return -6;
  if (p.xrow < KCSM * 128 + 16 || p.xrow % 128 != 16) return -6;
  if (p.wsn < (long long)p.N * G::TOK || p.cntn < NB || p.eoffn < KS * NG * G::NWC) return -7;
  const long long smem = (long long)G::XOFF + (long long)G::TOK * p.xrow;
  if (smem > 220 * 1024) return -8;
  auto k = pwk<MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  k<<<dim3(NG, KS), G::NTH, (size_t)smem, st>>>(p);
  return (int)cudaGetLastError();
}

int pw_dispatch(int inst, const PwP& p, int NG, int KS, cudaStream_t st) {
  switch (inst) {
@@CASES@@
  }
  return -2;
}

// ---------------------------------------------------------------- PROBE kernels: pure streaming, no decode, no mma
// ring: each CTA streams a contiguous range; warp w lane l reads uint4 (k * NW + w) * 32 + l, D loads in flight
template <int D>
__global__ void __launch_bounds__(256) pw_probe_ring(const uint4* buf, long long per16, uint32_t* sink) {
  const int NW = blockDim.x >> 5, w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const uint4* b = buf + blockIdx.x * per16 + w * 32 + l;
  const long long steps = per16 / (NW * 32);
  uint4 v[D];
  uint32_t a = 0;
#pragma unroll
  for (int s = 0; s < D; s++) if (s < steps) v[s] = pw_gnc(b + (long long)s * NW * 32);
#pragma unroll 1
  for (long long k0 = 0; k0 < steps; k0 += D) {
#pragma unroll
    for (int s = 0; s < D; s++) {
      const long long k = k0 + s;
      if (k < steps) {
        a ^= v[s].x ^ v[s].y ^ v[s].z ^ v[s].w;
        if (k + D < steps) v[s] = pw_gnc(b + (k + D) * NW * 32);
      }
    }
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
  if (threadIdx.x == 0) sink[gridDim.x + blockIdx.x] = 1;
}
// bulk: warp 1 lane 0 streams SZ-byte stages into an ST-deep ring; warp 0 waits, touches one uint4 per lane, releases
__global__ void __launch_bounds__(64) pw_probe_bulk(const uint8_t* buf, long long per, int SZ, int ST, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), full = sb + ST * SZ, empty = full + ST * 8;
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const int NS = (int)(per / SZ);
  const uint8_t* src = buf + blockIdx.x * per;
  if (threadIdx.x == 32) {
    for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  uint32_t a = 0;
  if (w == 1) {
    if (l == 0)
      for (int s = 0; s < NS; s++) {
        const int slot = s % ST;
        if (s >= ST) pw_mbwait(empty + slot * 8, ((s / ST) - 1) & 1);
        pw_mbexpect(full + slot * 8, SZ);
        pw_bulk(sb + slot * SZ, src + (long long)s * SZ, SZ, full + slot * 8);
      }
  } else {
    for (int s = 0; s < NS; s++) {
      const int slot = s % ST;
      pw_mbwait(full + slot * 8, (s / ST) & 1);
      const uint4 v = pw_lds(sb + slot * SZ + l * 16);
      a ^= v.x ^ v.w;
      __syncwarp();
      if (l == 0) pw_mbarrive(empty + slot * 8);
    }
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
__global__ void pw_probe_empty(uint32_t* sink) { if (threadIdx.x == 0) sink[blockIdx.x] = 1; }
__global__ void __launch_bounds__(256) pw_probe_red(float4* ws, int n4, int reps) {
  for (int r = 0; r < reps; r++)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) pw_red4(ws + i, 1.f, 1.f, 1.f, 1.f);
}
// kind 0: ring D=a, threads b; 1: bulk SZ=a, ST=b; 2: empty; 3: red, reps a over n4 = per float4s
int pw_probe_launch(int kind, int a, int b, const void* buf, long long per, int grid, uint32_t* sink, float* ws, cudaStream_t st) {
  if (kind == 0) {
    if (b != 128 && b != 256) return -1;
    if (a == 4) pw_probe_ring<4><<<grid, b, 0, st>>>((const uint4*)buf, per / 16, sink);
    else if (a == 8) pw_probe_ring<8><<<grid, b, 0, st>>>((const uint4*)buf, per / 16, sink);
    else if (a == 16) pw_probe_ring<16><<<grid, b, 0, st>>>((const uint4*)buf, per / 16, sink);
    else return -1;
  } else if (kind == 1) {
    const int sm = b * a + b * 16;
    if (a % 16 || sm > 220 * 1024) return -1;
    static int attr = 0;
    if (!attr) { if (cudaFuncSetAttribute(pw_probe_bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024) != cudaSuccess) { cudaGetLastError(); return -3; } attr = 1; }
    pw_probe_bulk<<<grid, 64, sm, st>>>((const uint8_t*)buf, per, a, b, sink);
  } else if (kind == 2) {
    pw_probe_empty<<<grid, 32, 0, st>>>(sink);
  } else if (kind == 3) {
    pw_probe_red<<<grid, 256, 0, st>>>((float4*)ws, (int)per, a);
  } else return -1;
  return (int)cudaGetLastError();
}
"""

KERNELS_SRC = _KERNELS_TEMPLATE.replace("@@CASES@@", "\n".join(
    "    case %d: return pw_launch<%s>(p, NG, KS, st);" % (i, ",".join(map(str, cfg))) for i, (_, cfg, _, _) in enumerate(PW_INSTS)))

GLUE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t Tb) {
  if (W.dim() != 4 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
  PwP p;
  p.W = (const uint8_t*)W.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = (__nv_bfloat16*)out.data_ptr();
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>(); p.esc = (const uint32_t*)esc.data_ptr(); p.eoff = eoff.data_ptr<int>();
  p.wsn = ws.numel(); p.cntn = (int)cnt.numel(); p.eoffn = (int)eoff.numel();
  p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1);
  p.NBW = (int)W.size(0); p.KCW = (int)W.size(1); p.RTBW = (int)W.size(2); p.xrow = (int)xrow; p.Tb = (uint32_t)(Tb & 0xffffffff);
  return (int64_t)pw_dispatch((int)inst, p, (int)NG, (int)KS, at::cuda::getCurrentCUDAStream());
}

int64_t pw_probe(int64_t kind, int64_t a, int64_t b, torch::Tensor buf, int64_t per, int64_t grid, torch::Tensor sink, torch::Tensor ws) {
  if (sink.numel() < 2 * grid) return -10;
  return (int64_t)pw_probe_launch((int)kind, (int)a, (int)b, buf.data_ptr(), (long long)per, (int)grid,
                                  (uint32_t*)sink.data_ptr(), ws.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
}
"""

CUDA_SRC = KERNELS_SRC + GLUE_SRC

CPP_SRC = ("int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t Tb);\n"
           "int64_t pw_probe(int64_t kind, int64_t a, int64_t b, torch::Tensor buf, int64_t per, int64_t grid, torch::Tensor sink, torch::Tensor ws);")
FUNCTIONS = ["pw_run", "pw_probe"]


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


def _pw_esc(esc6, n, cfg, KC, NG, KS, dev):
    """per-consumer-warp escape lists for grid (NG, KS): (entries int32 incl. one sentinel per warp, offsets int32).
    A warp's step i = (nb - nb0) * KCS + u walks its n-blocks in order, each over its K-slice."""
    MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB = cfg
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


def _pw_xoff(cfg):
    MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB = cfg
    wb = CPS * WN * RT * 1536 if MODE else 0
    return (ST * wb + ST * 16 + 127) // 128 * 128 if MODE else 0


def _pw_grid(cfg, occ, n, K):
    """(NG, KS, xrow): x slice + W ring must fit PW_SMEM / occ. Target one wave of 132 * occ CTAs (two waves for m <= 8
    jobs under 16 MB of packed W, so the per-CTA ramp hides). Among grids filling >= 85% of the target, minimise the
    L2 side traffic: x re-read NG * m*K*2 plus fp32 reductions KS * m*N*4 (weighted 2x; none when KS == 1)."""
    MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB = cfg
    BN, TOK = WN * RT * 16, WTOK * TTW * 8
    NB, NST = n // BN, K // 64 // CPS
    T = 132 * occ * (2 if TOK == 8 and n * K * 1.5 < 16e6 else 1)
    cap = PW_SMEM // occ - _pw_xoff(cfg)
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


def _pw_make(idx):
    name, cfg, ms, occ = PW_INSTS[idx]
    MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB = cfg
    RTB, TOK = WN * RT, WTOK * TTW * 8

    def serves(layer, m):
        return layer in PW_LAYERS and m in ms

    def build(state, layer, m):
        raw = state["raw"][layer]
        n, K = _n_of(layer, m), raw.shape[1]
        KC = K // 64
        if n % (RTB * 16) or not (TOK - 8 < m <= TOK) or KC % CPS:
            return None
        grid = _pw_grid(cfg, occ, n, K)
        if grid is None:
            return None
        NG, KS, xrow = grid
        NB = n // (RTB * 16)
        lay = _pw_layout(state, layer, RTB)
        if lay is None:
            return None
        _, tb, esc6 = _pw_c12(state, layer)
        el = _pw_esc(esc6, n, cfg, KC, NG, KS, raw.device)
        if el is None:
            return None
        esc, eoff = el
        ws = torch.zeros((n * TOK,), dtype=torch.float32, device=raw.device)  # per path and shape: a broken path
        cnt = torch.zeros((NB,), dtype=torch.int32, device=raw.device)       # cannot poison another's workspace
        ext = _ext()
        base = _fallback(state, layer, n)
        _log("PW %s %s m=%d NG=%d KS=%d grid=%d xKB=%.0f" % (name, layer, m, NG, KS, NG * KS, TOK * xrow / 1024))

        def run(x, out):
            if not x.is_contiguous() or x.data_ptr() % 16:
                base(x, out)
                return
            r = ext.pw_run(idx, lay, x, out, ws, cnt, esc, eoff, NG, KS, xrow, tb)
            if r:  # a host-side launch error code; the fallback keeps the output correct
                base(x, out)
        return run

    path("pw-%s" % name, serves)(build)


for _i in range(len(PW_INSTS)):
    _pw_make(_i)


# ---------------------------------------------------------------- PROBE: pure streaming engines, no decode, no mma
PROBE_BUDGET_S = 10.0


def _probe(state, flush, sink):
    """Achieved HBM bandwidth per stream engine under the bake-off's cold-L2 graph timing. One line per variant:
    PROBE <engine> <config> <us> <TB/s>, TB/s = bytes / (us - 4) (graph replay cost removed)."""
    t0 = time.time()
    ext = _ext()
    lay = _pw_layout(state, "pooler", 8)
    dev = flush.device
    buf = lay if lay is not None else state["raw"]["pooler"]
    total = buf.numel() * buf.element_size()
    sk = torch.zeros((2 * 132 * 8,), dtype=torch.int32, device=dev)
    ws = torch.zeros((128 * 4096,), dtype=torch.float32, device=dev)
    dummy = torch.empty((1, 1), device=dev)

    def one(label, kind, a, b, per, grid, nbytes):
        if time.time() - t0 > PROBE_BUDGET_S:
            return
        try:
            if ext.pw_probe(kind, a, b, buf, per, grid, sk, ws):
                _log("PROBE %s skip" % label)
                return
            us = _time_us(lambda x, o: ext.pw_probe(kind, a, b, buf, per, grid, sk, ws), dummy, dummy, flush, sink)
            _log("PROBE %s %.2f %.2f" % (label, us, nbytes / max(us - 4.0, 0.05) / 1e6))
        except Exception as error:
            _log("PROBE %s ERROR %s" % (label, str(error).splitlines()[-1][:120]))

    one("empty grid=132", 2, 0, 0, 0, 132, 0)
    for d in (4, 8, 16):
        for wpc in (4, 8):
            for c in (1, 2, 4):
                grid = 132 * c
                per = total // grid // (wpc * 512) * (wpc * 512)
                one("ring D=%d warps=%d ctas/sm=%d" % (d, wpc, c), 0, d, wpc * 32, per, grid, per * grid)
    for sz in (16384, 32768):
        for st in (2, 4, 6):
            for c in (1, 2):
                if c * st * sz > 220 * 1024:
                    _log("PROBE bulk stage=%dK ST=%d ctas/sm=%d skip-smem" % (sz >> 10, st, c))
                    continue
                grid = 132 * c
                per = total // grid // sz * sz
                one("bulk stage=%dK ST=%d ctas/sm=%d" % (sz >> 10, st, c), 1, sz, st, per, grid, per * grid)
    per = (6 << 20) // 264 // 4096 * 4096
    one("6mb ring D=8 warps=8 grid=264", 0, 8, 256, per, 264, per * 264)
    one("red 16MB v4 grid=264 (8 x 2MB)", 3, 8, 0, 128 * 4096 // 4, 264, 16 << 20)
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
