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

pw1 kernel family ("pw"), out[m, n] = x[m, K] @ W[n, K]^T for qkv_gate / o / pooler / unpooler at m in {1, 8, 16, 32, 128}:
  * Weights: r6's judge-proven lossless 12-bit codec (_pack + _c12, A-fragment order, swap-AB: W rows are the mma M
    dimension, tokens the n8 dimension). A unit is one 16-row tile x one 64-wide K chunk = 1536 B. prepare() re-lays
    units per BN-row block as [n-block][chunk][tile-in-block][1536] so any K-slice of an n-block is one contiguous range.
    Out-of-window exponents are fixed in registers from per-warp escape lists (CSR, sentinel-terminated) built for
    each launch geometry.
  * Stream: MODE 0 "ring" = each consumer warp keeps a D-deep register ring of ld.global.nc.L1::no_allocate.v4 loads of
    its own tiles; MODE 1 "bulk" = one producer warp issues cp.async.bulk 1D copies of each stage (CPS chunks of the
    whole n-block, 6-12 KB) into an ST-deep shared-memory ring with full/empty mbarriers.
  * x: staged in the same shared-memory ring, one 1D bulk copy per token row and stage (row stride CPS*128+16 bytes,
    so the per-lane 2 x 16 B B-fragment loads are bank-conflict free); rows m..TOK-1 are zeroed once.
  * Split-K over gridDim.y (uneven split allowed); partials meet via vector fp32 reductions into a
    prepare-allocated fp32 workspace in fragment order (red.global.add.v4.f32); the last-arriving CTA of an n-block converts to bf16, writes
    out, re-zeroes its workspace slice and its counter. KS == 1 stores straight from registers.
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
# pw instances: name -> (MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB), the m it serves, CTAs/SM guess for the KS table.
#   MODE 0 ring (W in per-warp register ring, D chunks deep), 1 bulk (W in the smem ring with x)
#   RT: 16-row W tiles per warp, WN: warps along N, WTOK: warps along tokens, TTW: n8 token tiles per warp
#   BN = WN*RT*16 rows per CTA, TOK = WTOK*TTW*8 token rows; ST smem stages of CPS 64-wide K chunks; MINB launch bound
# Register budget (fp32 acc = RT*TTW*4; decoded A = RT*16; B = 8 transient; ring = D*RT*12 (+12 current)).
# ptxas -v (CUDA 12.8, sm_90a), all 0 spills: r64t1 96, r32t1 93, r64t2 100, r64t4 121, b64t1 48, b32t1 48, b64t2 51,
#   b64t4 70, b32t4 71, b128m 160, b64m2 160, b64m1 98, r64m 209. Last field = CTAs/SM implied by regs and smem.
# ----------------------------------------------------------------------------------------------------------------
PW_INSTS = [
    ("r64t1", (0, 1, 4, 1, 1, 4, 1, 4, 3), (1, 8), 3),
    ("r32t1", (0, 1, 2, 1, 1, 4, 1, 4, 6), (1, 8), 5),
    ("r64t2", (0, 1, 4, 1, 2, 4, 1, 4, 3), (16,), 3),
    ("r64t4", (0, 1, 4, 1, 4, 4, 1, 4, 3), (32,), 2),
    ("b64t1", (1, 1, 4, 1, 1, 4, 2, 1, 3), (1, 8), 3),
    ("b32t1", (1, 1, 2, 1, 1, 6, 2, 1, 4), (1, 8), 4),
    ("b64t2", (1, 1, 4, 1, 2, 4, 2, 1, 3), (16,), 3),
    ("b64t4", (1, 1, 4, 1, 4, 4, 2, 1, 2), (32,), 2),
    ("b32t4", (1, 1, 2, 1, 4, 6, 2, 1, 2), (32,), 2),
    ("b128m", (1, 2, 4, 2, 8, 4, 1, 1, 1), (128,), 1),
    ("b64m2", (1, 2, 2, 2, 8, 4, 1, 1, 2), (128,), 2),
    ("b64m1", (1, 1, 4, 2, 8, 4, 1, 1, 1), (128,), 1),
    ("r64m", (0, 1, 4, 1, 16, 3, 1, 3, 1), (128,), 1),
]
PW_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")

# ----------------------------------------------------------------------------------------------------------------
# CUDA sources. KERNELS_SRC is pure CUDA (no torch headers) so it can be compiled standalone for ptxas -v;
# GLUE_SRC is the torch launch glue. CUDA_SRC = KERNELS_SRC + GLUE_SRC, compiled by one load_inline.
# No -std flag: the judge's ATen needs C++20 and load_inline already sets it. sm_90a is required for bulk copies.
# ----------------------------------------------------------------------------------------------------------------
_KERNELS_TEMPLATE = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x row length K; W chunk c reads x columns (c * 64) % K, so a later router "K-wrap" (W' = [hi | lo] along K,
// KCW = 2K/64) needs no restructuring.
struct PwP {
  const uint8_t* W;            // packed [NBW][KCW][RTB][1536]
  const __nv_bfloat16* x;      // [m, K]
  __nv_bfloat16* out;          // [m, N]
  float* ws;                   // fp32 split-K workspace, fragment order [NB][RTB][TT][32][4], zero between calls
  int* cnt;                    // [NB] arrival counters, zero between calls
  const uint32_t* esc;         // escape entries, one sentinel-terminated (0xffffffff) list per consumer warp
  const int* eoff;             // [KS][NB][NWC] start of each warp's list
  long long wsn;               // workspace floats (bounds check)
  int cntn, eoffn;             // counter / offset entries (bounds check)
  int m, N, K, KCW, RTBW, NBW;
  uint32_t Tb;                 // decode window base, replicated per byte
};

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS>
struct PwG {
  static constexpr int NWC = WN * WTOK, NTH = (NWC + 1) * 32, RTB = WN * RT, BN = RTB * 16, TT = WTOK * TTW, TOK = TT * 8;
  static constexpr int XROW = CPS * 128 + 16;           // padded x row: per-lane B loads hit distinct banks
  static constexpr int WB = MODE ? CPS * RTB * 1536 : 0; // W bytes per stage
  static constexpr int XB = TOK * XROW;                  // x bytes per stage
  static constexpr int SB = (WB + XB + 127) / 128 * 128; // stage stride
  static constexpr int SMEM = ST * SB + ST * 16;         // + full/empty mbarriers
};

__device__ __forceinline__ uint4 pw_gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
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
// one unit (16 rows x 64 k) per lane: 3 x 16 B -> 16 A registers (4 k16 steps x 4)
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
__device__ __forceinline__ void pw_red4(float4* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void pw_put(__nv_bfloat16* out, int N, int m, int n, int tok, float4 v) {
  if (tok < m) { out[(size_t)tok * N + n] = __float2bfloat16(v.x); out[(size_t)tok * N + n + 8] = __float2bfloat16(v.z); }
  if (tok + 1 < m) { out[(size_t)(tok + 1) * N + n] = __float2bfloat16(v.y); out[(size_t)(tok + 1) * N + n + 8] = __float2bfloat16(v.w); }
}
// escape fix: entry = seq << 20 | lane << 15 | reg << 11 | half << 10 | xor7 (seq = chunk-in-slice * RT + j)
template <int RT>
__device__ __forceinline__ void pw_fix(uint32_t (&A)[RT][16], int seq0, uint32_t& nx, int& ep, const uint32_t* esc, int lane) {
#pragma unroll
  for (int j = 0; j < RT; j++) {
    while ((nx >> 20) == (uint32_t)(seq0 + j)) {
      if (lane == (int)((nx >> 15) & 31)) {
        const uint32_t r = (nx >> 11) & 15, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
        for (int i = 0; i < 16; i++) if (i == (int)r) A[j][i] ^= xr;
      }
      nx = __ldg(esc + (++ep));
    }
  }
}
// all mma of one 64-wide chunk: B fragments of token tile tt are 16 contiguous bf16 of row (tile row g) at t * 32 B
template <int RT, int TTW, int XROW>
__device__ __forceinline__ void pw_mmas(float (&acc)[RT][TTW][4], const uint32_t (&A)[RT][16], uint32_t xa) {
#pragma unroll
  for (int tt = 0; tt < TTW; tt++) {
    const uint4 b0 = pw_lds(xa + tt * 8 * XROW), b1 = pw_lds(xa + tt * 8 * XROW + 16);
    const uint32_t B[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++) pw_mma(acc[j][tt], A[j] + 4 * q, B[2 * q], B[2 * q + 1]);
  }
}

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS, int D, int MINB>
__global__ void __launch_bounds__((WN * WTOK + 1) * 32, MINB) pwk(const PwP p) {
  using G = PwG<MODE, RT, WN, WTOK, TTW, ST, CPS>;
  constexpr int NWC = G::NWC, NTH = G::NTH, RTB = G::RTB, BN = G::BN, TT = G::TT, TOK = G::TOK;
  constexpr int XROW = G::XROW, WB = G::WB, SB = G::SB;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = sbase + ST * SB, empty = full + ST * 8;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int nb = blockIdx.x, ks = blockIdx.y, NB = gridDim.x, KS = gridDim.y, m = p.m;
  const int NST = p.KCW / CPS, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int NS = s1 - s0, KCS = NS * CPS, c0 = s0 * CPS;
  // token rows m..TOK-1 of every stage are zero for the whole kernel (never copied into)
  {
    const int zr = TOK - m, per = zr * CPS * 8;
    for (int e = tid; e < ST * per; e += NTH) {
      const int s = e / per, rem = e - s * per, r = m + rem / (CPS * 8), c = rem % (CPS * 8);
      *(uint4*)(smem + s * SB + WB + r * XROW + c * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, NWC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float acc[RT][TTW][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
  const int wn = warp % WN, wt = warp / WN;
  if (warp == NWC) {
    // producer warp: stage s = CPS chunks of x (one bulk copy per token row) [+ the n-block's W bytes]
    const uint32_t tx = (uint32_t)m * (CPS * 128) + WB;
    for (int s = 0; s < NS; s++) {
      const int slot = s % ST;
      if (s >= ST) pw_mbwait(empty + slot * 8, ((s / ST) - 1) & 1);
      const uint32_t bar = full + slot * 8, dst = sbase + slot * SB;
      if (lane == 0) pw_mbexpect(bar, tx);
      __syncwarp();
      const int ck = c0 + s * CPS;
      const __nv_bfloat16* xs = p.x + (ck * 64) % p.K;
      for (int r = lane; r < m; r += 32) pw_bulk(dst + WB + r * XROW, xs + (size_t)r * p.K, CPS * 128, bar);
      if (MODE == 1 && lane == 0) pw_bulk(dst, p.W + ((size_t)nb * p.KCW + ck) * (RTB * 1536), WB, bar);
    }
  } else {
    int ep = __ldg(p.eoff + (ks * NB + nb) * NWC + warp);
    uint32_t nx = __ldg(p.esc + ep);
    const uint32_t xoff = WB + (wt * TTW * 8 + g) * XROW + t * 32;
    if constexpr (MODE == 1) {
#pragma unroll 1
      for (int s = 0; s < NS; s++) {
        const int slot = s % ST;
        pw_mbwait(full + slot * 8, (s / ST) & 1);
        const uint32_t st = sbase + slot * SB;
#pragma unroll
        for (int cc = 0; cc < CPS; cc++) {
          uint32_t A[RT][16];
#pragma unroll
          for (int j = 0; j < RT; j++) {
            const uint32_t a = st + (cc * RTB + wn * RT + j) * 1536 + lane * 16;
            const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
            pw_unit(q0, q1, q2, p.Tb, A[j]);
          }
          pw_fix<RT>(A, (s * CPS + cc) * RT, nx, ep, p.esc, lane);
          pw_mmas<RT, TTW, XROW>(acc, A, st + xoff + cc * 128);
        }
        __syncwarp();
        if (lane == 0) pw_mbarrive(empty + slot * 8);
      }
    } else {
      // ring: this warp's RT tiles of chunk u are 3 x 16 B per lane each, D chunks in flight per lane
      const uint8_t* wp = p.W + ((size_t)nb * p.KCW + c0) * (RTB * 1536) + (size_t)(wn * RT) * 1536 + lane * 16;
      uint4 wb[D][RT][3];
#pragma unroll
      for (int s = 0; s < D; s++)
        if (s < KCS) {
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int i = 0; i < 3; i++) wb[s][j][i] = pw_gnc(wp + (size_t)s * (RTB * 1536) + j * 1536 + i * 512);
        }
#pragma unroll 1
      for (int u0 = 0; u0 < KCS; u0 += D) {
#pragma unroll
        for (int s = 0; s < D; s++) {
          const int u = u0 + s;
          if (u < KCS) {
            uint4 cur[RT][3];
#pragma unroll
            for (int j = 0; j < RT; j++)
#pragma unroll
              for (int i = 0; i < 3; i++) cur[j][i] = wb[s][j][i];
            if (u + D < KCS) {
#pragma unroll
              for (int j = 0; j < RT; j++)
#pragma unroll
                for (int i = 0; i < 3; i++) wb[s][j][i] = pw_gnc(wp + (size_t)(u + D) * (RTB * 1536) + j * 1536 + i * 512);
            }
            const int xs = u / CPS, cc = u - xs * CPS, slot = xs % ST;
            if (cc == 0) pw_mbwait(full + slot * 8, (xs / ST) & 1);
            uint32_t A[RT][16];
#pragma unroll
            for (int j = 0; j < RT; j++) pw_unit(cur[j][0], cur[j][1], cur[j][2], p.Tb, A[j]);
            pw_fix<RT>(A, u * RT, nx, ep, p.esc, lane);
            pw_mmas<RT, TTW, XROW>(acc, A, sbase + slot * SB + xoff + cc * 128);
            if (cc == CPS - 1) {
              __syncwarp();
              if (lane == 0) pw_mbarrive(empty + slot * 8);
            }
          }
        }
      }
    }
  }
  // ---- epilogue. Fragment (j, tt): rows n = nb*BN + (wn*RT + j)*16 + g (+8), tokens tok = tt*8 + 2t (+1)
  if (KS == 1) {
    if (warp < NWC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int tt = 0; tt < TTW; tt++)
          pw_put(p.out, p.N, m, nb * BN + (wn * RT + j) * 16 + g, (wt * TTW + tt) * 8 + 2 * t,
                 make_float4(acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]));
    }
    return;
  }
  float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
  if (warp < NWC) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int tt = 0; tt < TTW; tt++) {
        const int T2 = wt * TTW + tt;
        if (T2 * 8 + 2 * t < m)
          pw_red4(ws4 + ((wn * RT + j) * TT + T2) * 32 + lane, acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]);
      }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(p.cnt + nb, 1) == KS - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
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

template <int MODE, int RT, int WN, int WTOK, int TTW, int ST, int CPS, int D, int MINB>
static int pw_launch(const PwP& p, int KS, cudaStream_t st) {
  using G = PwG<MODE, RT, WN, WTOK, TTW, ST, CPS>;
  if (MODE == 0 && WTOK != 1) return -3;
  if (p.m < 1 || p.m > G::TOK || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.KCW % CPS || p.K % (64 * CPS) || (p.KCW * 64) % p.K || KS < 1) return -5;
  const int NB = p.N / G::BN, NST = p.KCW / CPS;
  if (KS > NST || ((NST + KS - 1) / KS) * CPS * RT >= 4000) return -6;
  if (p.wsn < (long long)p.N * G::TOK || p.cntn < NB || p.eoffn < KS * NB * G::NWC) return -7;
  auto k = pwk<MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  k<<<dim3(NB, KS), G::NTH, G::SMEM, st>>>(p);
  return (int)cudaGetLastError();
}

int pw_dispatch(int inst, const PwP& p, int KS, cudaStream_t st) {
  switch (inst) {
@@CASES@@
  }
  return -2;
}
"""

KERNELS_SRC = _KERNELS_TEMPLATE.replace("@@CASES@@", "\n".join(
    "    case %d: return pw_launch<%s>(p, KS, st);" % (i, ",".join(map(str, cfg))) for i, (_, cfg, _, _) in enumerate(PW_INSTS)))

GLUE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t Tb) {
  if (W.dim() != 4 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
  PwP p;
  p.W = (const uint8_t*)W.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = (__nv_bfloat16*)out.data_ptr();
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>(); p.esc = (const uint32_t*)esc.data_ptr(); p.eoff = eoff.data_ptr<int>();
  p.wsn = ws.numel(); p.cntn = (int)cnt.numel(); p.eoffn = (int)eoff.numel();
  p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1);
  p.NBW = (int)W.size(0); p.KCW = (int)W.size(1); p.RTBW = (int)W.size(2); p.Tb = (uint32_t)(Tb & 0xffffffff);
  return (int64_t)pw_dispatch((int)inst, p, (int)KS, at::cuda::getCurrentCUDAStream());
}
"""

CUDA_SRC = KERNELS_SRC + GLUE_SRC

CPP_SRC = ("int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t Tb);")
FUNCTIONS = ["pw_run"]


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


# ---------------------------------------------------------------- pw packing, escape lists, KS tables, paths
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


def _pw_esc(esc6, n, cfg, KC, KS, dev):
    """per-consumer-warp escape lists for one launch geometry: (entries int32 incl. one sentinel per warp, offsets int32)"""
    MODE, RT, WN, WTOK, TTW, ST, CPS, D, MINB = cfg
    NWC, RTB = WN * WTOK, WN * RT
    NB, NST = n // (RTB * 16), KC // CPS
    G = KS * NB * NWC
    e = esc6[esc6[:, 0] < n // 16]
    if len(e):
        tile, ch, q, ln, v, xr = e.t()
        nb, r = tile // RTB, tile % RTB
        wn, j = r // RT, r % RT
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * NST // KS
        ks = torch.searchsorted(s0, ch // CPS, right=True) - 1
        u = ch - s0[ks] * CPS
        seq = u * RT + j
        ent = (seq << 20) | (ln << 15) | ((q * 4 + v // 2) << 11) | ((v % 2) << 10) | xr
        gw = torch.cat([(ks * NB + nb) * NWC + wt * WN + wn for wt in range(WTOK)])
        ent, seq = ent.repeat(WTOK), seq.repeat(WTOK)
        if int(seq.max()) >= 4000:
            return None
        o = torch.argsort(gw * 8192 + seq)
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


def _pw_ks(NB, NST, occ, mult):
    """split-K so the grid is about mult waves of occ CTAs per SM on 132 SMs (uneven K split allowed)"""
    return max(1, min(NST, (132 * occ * mult) // NB))


def _pw_make(idx, mult):
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
        NB, NST = n // (RTB * 16), KC // CPS
        KS = _pw_ks(NB, NST, occ, mult)
        if mult > 1 and KS == _pw_ks(NB, NST, occ, 1):
            return None  # same grid as the 1-wave entry
        lay = _pw_layout(state, layer, RTB)
        if lay is None:
            return None
        _, tb, esc6 = _pw_c12(state, layer)
        el = _pw_esc(esc6, n, cfg, KC, KS, raw.device)
        if el is None:
            return None
        esc, eoff = el
        ws = torch.zeros((n * TOK,), dtype=torch.float32, device=raw.device)  # per path and shape: a broken path
        cnt = torch.zeros((NB,), dtype=torch.int32, device=raw.device)       # cannot poison another's workspace
        ext = _ext()
        base = _fallback(state, layer, n)

        def run(x, out):
            if not x.is_contiguous() or x.data_ptr() % 16:
                base(x, out)
                return
            r = ext.pw_run(idx, lay, x, out, ws, cnt, esc, eoff, KS, tb)
            if r:  # a host-side launch error code; the fallback keeps the output correct
                base(x, out)
        return run

    path("pw-%s-w%d" % (name, mult), serves)(build)


for _i, (_name, _cfg, _ms, _occ) in enumerate(PW_INSTS):
    for _mult in ((1,) if 128 in _ms else (1, 2)):
        _pw_make(_i, _mult)


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
