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

pw4 kernel family ("pw"), out[m, n] = x[m, K] @ W[n, K]^T for qkv_gate / o / pooler / unpooler at m in {1, 8, 16, 32, 128}:
  * Weights: r6's judge-proven lossless 12-bit codec (_pack + _c12, A-fragment order, swap-AB), 1536 B units laid out
    per BN-row block as [n-block][chunk][tile-in-block][1536]; escapes fixed in registers from per-warp lists.
  * m <= 32 ("b"): grid (NG, KS); each CTA copies x[:, K-slice] into shared memory once and streams W for a range of
    n-blocks through a W-only cp.async.bulk ring (producer warp, full/empty mbarriers, runtime depth ST). 8- and
    16-consumer-warp variants.
  * m = 128 / 256 (pw9, kind "w"): wgmma.mma_async m64nNk16 with A = decoded W in registers, B = the x tile in smem
    (one 2D TMA copy per 64-k stage, 128B swizzle, K-major descriptor), N = tokens (128 or 256); BN = 64 or 128 W rows
    per CTA (1 or 2 consumer warpgroups + a producer warpgroup); split-K clusters (1, KS, 1) with a DSMEM reduce and a
    transposed 16-B-row store. See the design note below.
  * pw4 skeleton fixes kept (producer first issue, escape prefetch, consumer-only epilogue barrier).
  * pw6: the m <= 32 paths come in two epilogues measured side by side: "g" = pw4's global red.v4 + counter + last-arriver
    finalize, "c" = cluster split-K with a DSMEM reduce of the KS partial n-blocks.
  * pw7: the bulk kernel rewritten as pwc (SASS-audit edits, numerics unchanged): ring depth as a template parameter with
    the stage loop unrolled by it; hoisted W / x base addresses with immediate offsets; chunk-major x slice; x fragments
    loaded before the full-barrier wait; escape compare against a loop-carried step; empty-barrier arrive right after the
    decode; lane and Tb pinned in registers; RT=2 cluster paths for m=16/32 (two W units share one x fragment set).
  * pw8: the router as one bf16 GEMM over K' = 2K: W' = [hi | lo] (two 12-bit packs, one contiguous range per n-block),
    x read at chunk c % (K/64), fp32 output; cluster (KS 8) and global (KS 16 / 32) epilogues, m in {1, 8, 16, 32, 128}.
  * prepare() prints STATS lines (bf16 weight statistics for the codec decision), then PROBE carve lines (pw6).

pw9 design note -- wgmma fragment-layout check (PTX ISA 9.4 sec. 9.7.17.5.1.1.1 fig. 151/152; CUTLASS SM90 RS GMMA
traits ALayout_64x16 = ((4,8,4),(2,2,2)):((128,1,16),(64,8,512)) vs SM80 m16n8k16 A ((4,8),(2,2,2)):((32,1),(16,8,128));
proj_rsi/audits/wgmma-layout.md):
  * Register A of wgmma m64nNk16 .bf16 is, per warp w of the warpgroup, exactly the mma.sync m16n8k16 A layout on rows
    16w..16w+15: R0 = (row g, k 2t,2t+1), R1 = (row g+8, same k), R2 = (row g, k 2t+8,+9), R3 = (row g+8, k 2t+8,+9).
    pw_dec's register order a[0..3] per k16 step therefore feeds wgmma without any register permutation.
  * BUT the r6 pack (_pack) assigns PERMUTED k to those slots (lane t, step q holds k = 16t + 4q + 0..3 of the
    64-chunk), which was legal only because mma.sync's B came from registers loaded with the same permutation. wgmma
    reads B (x) from smem in natural k order, so the wgmma paths use _pack_nat: slot (q, lane g*4+t, v = (kh, half, e))
    holds W[16T + 8 half + g][64c + 16q + 8 kh + 2t + e]. Codec, decoder and escape format are unchanged; the CPU
    emulator confirms the r6 pack fails on this path (rel. error 1.4) and the natural pack is bit-exact.
  * D of m64nNk16 .f32: d[4i..4i+3] = (row g, tok 8i+2t), (g, 8i+2t+1), (g+8, 8i+2t), (g+8, 8i+2t+1) = the mma.sync C
    fragment of token tile i, so the fragment-order staging / reduce indexing carries over.
  * B descriptor (K-major, 128B swizzle): start (addr & 0x3FFFF) >> 4, LBO field 1 (unused), SBO 1024 B (field 64),
    base offset 0 (1024-aligned stage bases), mode 1; +32 B (+2 encoded) per k16 step inside the 128-B row.
  * Async-read traps handled: ring slots released only after the wgmma.wait_group covering the chunk; A registers
    double buffered and pinned with asm volatile("" : "+r") after the retiring wait; accumulators fenced around each
    batch; no generic stores into B tiles (rows >= m are TMA zero fill); immediates scale-a/b = 1, trans-b = 0.
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
#  kind "b" (x-stationary bulk, pw7 kernel pwc): cfg (RT, WN, WTOK, TTW, CPS, ST, CLU, MINB); ST = ring depth (template);
#    CLU = 1: cluster split-K with a DSMEM reduce (grid NB x KS, clusters of KS CTAs along K); CLU = 0: global red.v4.
#    cfg[8] RTR = 1: router mode (two packs hi|lo along K, decode base switch, fp32 output).
#  kind "w" (wgmma register-A, m = 128 / 256): cfg (NWG, NT, ST, RTR), see the design note in the module docstring.
#  Entry: (name, kind, cfg, ms served as a path (() = probe only), CTAs/SM, default ST)
# ----------------------------------------------------------------------------------------------------------------
PW_INSTS = [
    ("b16t1g", "b", (1, 16, 1, 1, 1, 4, 0, 1, 0), (1, 8), 1, 4),   # pw4 epilogue: global red.v4 + counters
    ("b16t2g", "b", (1, 16, 1, 2, 1, 4, 0, 1, 0), (16,), 1, 4),
    ("b16t4g", "b", (1, 16, 1, 4, 1, 4, 0, 1, 0), (32,), 1, 4),
    ("b16t1c", "b", (1, 16, 1, 1, 1, 4, 1, 1, 0), (1, 8), 1, 4),   # cluster split-K, DSMEM reduce (pw6)
    ("b16t2c", "b", (1, 16, 1, 2, 1, 4, 1, 1, 0), (16,), 1, 4),
    ("b16t4c", "b", (1, 16, 1, 4, 1, 4, 1, 1, 0), (32,), 1, 4),
    ("r2t2c", "b", (2, 8, 2, 1, 1, 4, 1, 1, 0), (16,), 1, 4),      # RT=2: two W units per warp share one x fragment set
    ("r2t4c", "b", (2, 8, 2, 2, 1, 4, 1, 1, 0), (32,), 1, 4),
    ("b8t1", "b", (1, 8, 1, 1, 2, 4, 0, 1, 0), (1, 8), 1, 4),      # unpooler n=1024 only
    ("b8t1c", "b", (1, 8, 1, 1, 2, 4, 1, 1, 0), (1, 8), 1, 4),
    # router (RTR = 1): W' = [hi | lo] along K (K' = 2K, two 12-bit packs, decode base switches at chunk K/64),
    # x read at chunk c % (K/64), fp32 output; BN = 32 (2 consumer warps), 3 KB stages (one chunk), 8 deep
    ("rc1", "b", (1, 2, 1, 1, 1, 8, 1, 1, 1), (1, 8), 1, 8),       # cluster, KS 8 -> 8 x 8 = 64 CTAs
    ("rc4", "b", (1, 2, 1, 4, 1, 8, 1, 1, 1), (16, 32), 1, 8),
    ("rg1", "b", (1, 2, 1, 1, 1, 8, 0, 1, 1), (1, 8), 1, 8),       # global, KS 16 / 32 -> 128 / 256 CTAs
    ("rg4", "b", (1, 2, 1, 4, 1, 8, 0, 1, 1), (16, 32), 1, 8),
    ("rg128", "b", (1, 2, 2, 8, 1, 8, 0, 1, 1), (128,), 1, 8),     # m=128: 4 consumer warps (2 x 64 tokens)
    # wgmma, register-A (kind "w"): cfg (NWG consumer warpgroups (BN = 64*NWG), NT tokens per wgmma, ST, RTR)
    ("w2n128", "w", (2, 128, 4, 0), (128,), 1, 4),
    ("w1n128", "w", (1, 128, 4, 0), (128,), 1, 4),
    ("w2n256", "w", (2, 256, 4, 0), (256,), 1, 4),
    ("w1n256", "w", (1, 256, 4, 0), (256,), 1, 4),
    ("wr", "w", (1, 256, 4, 1), (128, 256), 1, 4),                  # router (K' = 2K, fp32 out) at m = 128 / 256
]
PW_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
PW_SMEM = 220 * 1024   # dynamic smem cap per CTA (opt-in max is 227 KB incl. static)
PW_SEQ = 32000         # escape-list step field (15 bits)
PW_BIDX = [i for i, e in enumerate(PW_INSTS) if e[1] == "b"]
PW_WIDX = [i for i, e in enumerate(PW_INSTS) if e[1] == "w"]

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
  int flags, stb, xtok;        // probe-only switches (pwk<..., PRB=1>); must be 0 for the GEMM paths
  uint32_t Tb;                 // decode window base, replicated per byte
  uint32_t Tb2;                // router: window base of the lo pack (W chunks >= kx); = Tb otherwise
  int kx;                      // x chunks (K/64): W chunk c reads x chunk c % kx; router chunks >= kx are the lo pack
  int of;                      // output is fp32 (router) instead of bf16
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
__device__ __forceinline__ void pw_put(__nv_bfloat16* out, int N, int m, int n, int tok, float4 v);
// store one accumulator fragment (rows n, n+8; tokens tok, tok+1) as bf16, or as fp32 for the router
__device__ __forceinline__ void pw_putp(const PwP& p, int m, int n, int tok, float4 v) {
  if (p.of) {
    float* o = (float*)p.out;
    if (tok < m) { o[(size_t)tok * p.N + n] = v.x; o[(size_t)tok * p.N + n + 8] = v.z; }
    if (tok + 1 < m) { o[(size_t)(tok + 1) * p.N + n] = v.y; o[(size_t)(tok + 1) * p.N + n + 8] = v.w; }
  } else pw_put((__nv_bfloat16*)p.out, p.N, m, n, tok, v);
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
// split-K epilogue shared by both kernels. pw_flush: per n-block bf16 store (KS == 1) or fp32 vector reductions, no
// barrier. pw_arrive: run by the CONSUMER warps only (named barrier 1 over NT = consumer threads), so the producer warp
// never waits for the epilogue; one arrival per owned n-block, the last arriver converts, writes and re-zeroes.
template <int RT, int TTW, int WTOK, int RTB>
__device__ __forceinline__ void pw_flush(const PwP& p, float (&acc)[RT][TTW][4], int nb, int wn, int wt, int lane, int KS, int m) {
  constexpr int TT = WTOK * TTW, BN = RTB * 16;
  const int g = lane >> 2, t = lane & 3;
  float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) {
      const int T2 = wt * TTW + tt;
      if (KS == 1) pw_putp(p, m, nb * BN + (wn * RT + j) * 16 + g, T2 * 8 + 2 * t,
                          make_float4(acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]));
      else if (T2 * 8 + 2 * t < m)
        pw_red4(ws4 + ((wn * RT + j) * TT + T2) * 32 + lane, acc[j][tt][0], acc[j][tt][1], acc[j][tt][2], acc[j][tt][3]);
      acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
    }
}
__device__ __forceinline__ void pw_csync() {  // all non-exited threads of the cluster; release/acquire on shared::cluster
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t pw_crank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ float4 pw_ldc(uint32_t local, uint32_t rank) {  // 16 B from peer `rank`'s smem at the same offset
  uint32_t ra; float4 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local), "r"(rank));
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ void pw_cbar(int nt) { asm volatile("bar.sync 1, %0;\n" :: "r"(nt) : "memory"); }
template <int TT, int RTB, int NT>
__device__ __forceinline__ void pw_arrive(const PwP& p, int nb0, int NBR, int KS, int m, int* s_list, int* s_n) {
  const int tid = threadIdx.x;  // 0 .. NT-1 (consumer warps are the low warps)
  constexpr int BN = RTB * 16;
  __threadfence();
  if (tid == 0) *s_n = 0;
  pw_cbar(NT);
  for (int i = tid; i < NBR; i += NT)
    if (atomicAdd(p.cnt + nb0 + i, 1) == KS - 1) s_list[atomicAdd(s_n, 1)] = nb0 + i;
  pw_cbar(NT);
  const int nl = *s_n;
  if (nl == 0) return;
  __threadfence();
  for (int k = 0; k < nl; k++) {
    const int nb = s_list[k];
    float4* ws4 = (float4*)p.ws + (size_t)nb * (RTB * TT * 32);
    for (int i = tid; i < RTB * TT * 32; i += NT) {
      const int l = i & 31, T2 = (i >> 5) % TT, r = (i >> 5) / TT, tok = T2 * 8 + 2 * (l & 3);
      if (tok < m) {
        const float4 v = __ldcg(ws4 + i);
        __stcg(ws4 + i, make_float4(0.f, 0.f, 0.f, 0.f));
        pw_putp(p, m, nb * BN + r * 16 + (l >> 2), tok, v);
      }
    }
    if (tid == 0) p.cnt[nb] = 0;
  }
}

// ---------------------------------------------------------------- kind "b": x-stationary, W-only bulk ring (pw7)
// Grid (NG, KS): CTA (ng, ks) owns n-blocks [ng*NB/NG, (ng+1)*NB/NG) and K stages [ks*NST/KS, (ks+1)*NST/KS).
// CLU = 1: NG == NB (one n-block per CTA) and, for KS > 1, clusters of KS CTAs along K reduce over DSMEM (pw6).
// CLU = 0: per-n-block red.global.add.v4 + counters + last-arriver finalize on the consumer warps (pw4).
// x slice layout (pw7): chunk-major [chunk][row][144 B] (128 B of x + 16 B pad), so the per-lane B-fragment loads
// (row g at byte t*32 and +16) hit 16-B bank groups (g + 2t) mod 8, distinct per quarter-warp, and tt*8*144 and the
// stage offsets are immediates. ST (ring depth) is a template parameter: the stage loop is unrolled by ST.
template <int RT, int WN, int WTOK, int TTW, int CPS, int ST>
struct PwC {
  static constexpr int NWC = WN * WTOK, NTH = (NWC + 1) * 32, RTB = WN * RT, BN = RTB * 16, TT = WTOK * TTW, TOK = TT * 8;
  static constexpr int WB = CPS * RTB * 1536;                          // W bytes per stage
  static constexpr int FULL = ST * WB, EMPTY = ST * WB + ST * 8;       // barrier offsets
  static constexpr int XOFF = (ST * WB + ST * 16 + 127) / 128 * 128;   // x slice offset
  static constexpr int XCH = TOK * 144;                                // x bytes per 64-wide chunk
  static constexpr int FRAG = RTB * TT * 32;                           // float4 fragments per n-block
};
// escape fix against a loop-carried nxs = nx >> 17 (updated only when an entry is consumed)
template <int RT>
__device__ __forceinline__ void pw_fix2(uint32_t (&A)[RT][16], int seq0, uint32_t& nx, uint32_t& nxs, int& ep, const uint32_t* esc, int lane) {
#pragma unroll
  for (int j = 0; j < RT; j++) {
    while (nxs == (uint32_t)(seq0 + j)) {
      if (lane == (int)((nx >> 12) & 31)) {
        const uint32_t r = (nx >> 8) & 15, xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
        for (int i = 0; i < 16; i++) if (i == (int)r) A[j][i] ^= xr;
      }
      nx = __ldg(esc + (++ep));
      nxs = nx >> 17;
    }
  }
}
// mma of one chunk with prefetched B fragments (xb[tt][0..1] = the lane's 16 bf16 of token row tt*8+g)
template <int RT, int TTW>
__device__ __forceinline__ void pw_mmas_r(float (&acc)[RT][TTW][4], const uint32_t (&A)[RT][16], const uint4 (&xb)[TTW][2]) {
#pragma unroll
  for (int tt = 0; tt < TTW; tt++) {
    const uint32_t B[8] = {xb[tt][0].x, xb[tt][0].y, xb[tt][0].z, xb[tt][0].w, xb[tt][1].x, xb[tt][1].y, xb[tt][1].z, xb[tt][1].w};
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++) pw_mma(acc[j][tt], A[j] + 4 * q, B[2 * q], B[2 * q + 1]);
  }
}

template <int RT, int WN, int WTOK, int TTW, int CPS, int ST, int CLU, int MINB, int RTR>
__global__ void __launch_bounds__(PwC<RT, WN, WTOK, TTW, CPS, ST>::NTH, MINB) pwc(const PwP p) {
  using G = PwC<RT, WN, WTOK, TTW, CPS, ST>;
  constexpr int NWC = G::NWC, NTH = G::NTH, RTB = G::RTB, BN = G::BN, TOK = G::TOK, TT = G::TT, WB = G::WB, XCH = G::XCH;
  constexpr int XPF = (!CLU && TTW >= 4) ? 0 : TTW;  // x tiles loaded before the full wait (global m=32: at the 96-reg cap)
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_list[256];
  __shared__ int s_n;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t xs = sbase + G::XOFF;
  const int tid = threadIdx.x, warp = tid >> 5;
  int lane = tid & 31;
  asm volatile("" : "+r"(lane));  // keep lane in a register (no S2R rematerialisation in the loop)
  const int g = lane >> 2, t = lane & 3;
  const int ng = blockIdx.x, ks = blockIdx.y, NG = gridDim.x, KS = gridDim.y, m = p.m;
  const int NB = p.N / BN, nb0 = (int)((long long)ng * NB / NG), nb1 = (int)((long long)(ng + 1) * NB / NG), NBR = nb1 - nb0;
  const int NST = p.KCW / CPS, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int SPB = s1 - s0, KCS = SPB * CPS, c0 = s0 * CPS, NSTOT = NBR * SPB;
  const int wn = warp % WN, wt = warp / WN;
  const size_t cstr = (size_t)RTB * 1536;
  auto issue = [&](int s, int slot) {
    const int nb = nb0 + s / SPB, ss = s - (s / SPB) * SPB;
    pw_mbexpect(sbase + G::FULL + slot * 8, WB);
    pw_bulk(sbase + slot * WB, p.W + ((size_t)nb * p.KCW + c0 + ss * CPS) * cstr, WB, sbase + G::FULL + slot * 8);
  };
  // (1) producer: barriers, then the first ST W stages before anything else in the CTA
  if (tid == NWC * 32) {
#pragma unroll
    for (int i = 0; i < ST; i++) { pw_mbinit(sbase + G::FULL + i * 8, 1); pw_mbinit(sbase + G::EMPTY + i * 8, NWC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int s = 0; s < ST && s < NSTOT; s++) issue(s, s);
  }
  // (2) consumers: escape-list cursor loads now, overlapping the x copy
  int ep = 0;
  uint32_t nx = 0xffffffffu;
  if (warp < NWC) { ep = __ldg(p.eoff + (ks * NG + ng) * NWC + warp); nx = __ldg(p.esc + ep); }
  // (3) x slice copy into the chunk-major layout (rows >= m zero-filled)
  {
    const int segs = KCS * 8 * TOK;  // e -> (chunk c, row r, 16-B segment sg)
    for (int e = tid; e < segs; e += NTH) {
      const int sg = e & 7, r = (e >> 3) % TOK, c = (e >> 3) / TOK;
      const int col = ((c0 + c) * 64) % p.K + sg * 8;
      pw_cpa(xs + (c * TOK + r) * 144 + sg * 16, p.x + (size_t)(r < m ? r : 0) * p.K + col, r < m ? 16u : 0u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  __syncthreads();
  float acc[RT][TTW][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int tt = 0; tt < TTW; tt++) acc[j][tt][0] = acc[j][tt][1] = acc[j][tt][2] = acc[j][tt][3] = 0.f;
  if (warp == NWC) {
    // producer: stages ST.. (lane 0 issues, the warp waits together)
    int slot = 0, round = 1;
    for (int s = ST; s < NSTOT; s++) {
      pw_mbwait(sbase + G::EMPTY + slot * 8, (round - 1) & 1);
      if (lane == 0) issue(s, slot);
      __syncwarp();
      if (++slot == ST) { slot = 0; round++; }
    }
    if (!CLU || KS == 1) return;
  } else {
    uint32_t Tb = p.Tb, Tb2 = p.Tb2;
    asm volatile("" : "+r"(Tb), "+r"(Tb2));
    const uint32_t wl = sbase + (wn * RT) * 1536 + lane * 16;            // + slot*WB + (cc*RTB + j)*1536 + i*512
    const uint32_t xl = xs + (wt * TTW * 8 + g) * 144 + t * 32;           // + chunk*XCH + tt*8*144 (+16)
    uint32_t nxs = nx >> 17;
    int cnb = nb0, ss = 0;
    uint32_t ph = 0;
    // one stage: slot k (an immediate where the loop below is unrolled), stage index s
    auto stage = [&](const int k, const int s) {
        const int ub = (CLU ? s : ss) * CPS;  // x chunk of cc = 0 within this CTA's K slice
        // x fragments first (the x slice was written before the CTA barrier); XPF token tiles before the wait, the
        // rest after it (XPF < TTW only where the register cap is tight)
        uint4 xb[CPS][TTW][2];
#pragma unroll
        for (int cc = 0; cc < CPS; cc++)
#pragma unroll
          for (int tt = 0; tt < XPF; tt++) {
            const uint32_t xa = xl + (ub + cc) * XCH + tt * 8 * 144;
            xb[cc][tt][0] = pw_lds(xa); xb[cc][tt][1] = pw_lds(xa + 16);
          }
        pw_mbwait(sbase + G::FULL + k * 8, ph);
#pragma unroll
        for (int cc = 0; cc < CPS; cc++)
#pragma unroll
          for (int tt = XPF; tt < TTW; tt++) {
            const uint32_t xa = xl + (ub + cc) * XCH + tt * 8 * 144;
            xb[cc][tt][0] = pw_lds(xa); xb[cc][tt][1] = pw_lds(xa + 16);
          }
        uint32_t A[CPS][RT][16];
#pragma unroll
        for (int cc = 0; cc < CPS; cc++)
#pragma unroll
          for (int j = 0; j < RT; j++) {
            const uint32_t a = wl + k * WB + (cc * RTB + j) * 1536;
            const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
            pw_unit(q0, q1, q2, RTR ? (c0 + ub + cc >= p.kx ? Tb2 : Tb) : Tb, A[cc][j]);  // router: lo pack past kx
          }
        // the W slot is dead once decoded: release it before the escape fix and the mma
        __syncwarp();
        if (lane == 0) pw_mbarrive(sbase + G::EMPTY + k * 8);
#pragma unroll
        for (int cc = 0; cc < CPS; cc++) {
          pw_fix2<RT>(A[cc], (s * CPS + cc) * RT, nx, nxs, ep, p.esc, lane);
          pw_mmas_r<RT, TTW>(acc, A[cc], xb[cc]);
        }
        if constexpr (!CLU) {
          if (++ss == SPB) { pw_flush<RT, TTW, WTOK, RTB>(p, acc, cnb, wn, wt, lane, KS, m); ss = 0; cnb++; }
        }
    };
    if constexpr (CLU) {
#pragma unroll 1
      for (int s0 = 0; s0 < NSTOT; s0 += ST) {
#pragma unroll
        for (int k = 0; k < ST; k++) {
          if (s0 + k >= NSTOT) break;
          stage(k, s0 + k);
        }
        ph ^= 1;
      }
    } else {  // global epilogue: the per-n-block flush inlined ST times would exceed the 96-register cap at m=32
      int k = 0;
#pragma unroll 1
      for (int s = 0; s < NSTOT; s++) {
        stage(k, s);
        if (++k == ST) { k = 0; ph ^= 1; }
      }
    }
    if constexpr (CLU) {
      if (KS == 1) { pw_flush<RT, TTW, WTOK, RTB>(p, acc, nb0, wn, wt, lane, 1, m); return; }
    } else {
      if (KS == 1) return;
      pw_arrive<TT, RTB, NWC * 32>(p, nb0, NBR, KS, m, s_list, &s_n);
      return;
    }
  }
  if constexpr (CLU) {
    // every consumer has passed its last full wait (all W bytes landed and were decoded) and the producer issued
    // its last stage: after this barrier the W ring area is free for the partial sums
    __syncthreads();
    if (warp < NWC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int tt = 0; tt < TTW; tt++) {
          const uint32_t a = sbase + ((((wn * RT + j) * TT + wt * TTW + tt) * 32) + lane) * 16;
          asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "r"(a), "f"(acc[j][tt][0]), "f"(acc[j][tt][1]), "f"(acc[j][tt][2]), "f"(acc[j][tt][3]) : "memory");
        }
    }
    pw_csync();  // (1) every CTA of the cluster has its partials in smem
    constexpr int FRAG = G::FRAG;
    const int r = (int)pw_crank(), per = FRAG / KS;
    for (int i = r * per + tid; i < (r + 1) * per; i += NTH) {
      const int l = i & 31, T2 = (i >> 5) % TT, tok = T2 * 8 + 2 * (l & 3);
      if (tok < m) {
        float4 v[8];
#pragma unroll
        for (int q = 0; q < 8; q++) if (q < KS) v[q] = pw_ldc(sbase + i * 16, q);  // all peer loads before the adds
        float4 a = v[0];
#pragma unroll
        for (int q = 1; q < 8; q++) if (q < KS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; }
        pw_putp(p, m, nb0 * BN + ((i >> 5) / TT) * 16 + (l >> 2), tok, a);
      }
    }
    pw_csync();  // (2) no CTA exits (or frees its smem) while a peer still reads it
  }
}

template <int RT, int WN, int WTOK, int TTW, int CPS, int ST, int CLU, int MINB, int RTR>
static int pw_launch(const PwP& p, int NG, int KS, cudaStream_t st) {
  using G = PwC<RT, WN, WTOK, TTW, CPS, ST>;
  if (p.m < 1 || p.m > G::TOK || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.KCW % CPS || p.K % 64 || (p.KCW * 64) % p.K || KS < 1 || NG < 1 || p.st != ST || p.flags || p.stb || p.xtok) return -5;
  const int NB = p.N / G::BN, NST = p.KCW / CPS, KCSM = (NST + KS - 1) / KS * CPS, NBRM = (NB + NG - 1) / NG;
  if (KS > NST || NG > NB || NBRM > 256 || (long long)NBRM * KCSM * RT >= 32000) return -6;
  if (CLU && (NG != NB || KS > 8 || G::FRAG % KS || G::XOFF < G::FRAG * 16)) return -6;
  if (p.of != RTR || (!RTR && p.Tb2 != p.Tb) || p.kx * 64 != p.K) return -5;
  if (p.wsn < (long long)p.N * G::TOK || p.cntn < NB || p.eoffn < KS * NG * G::NWC) return -7;
  const long long smem = (long long)G::XOFF + (long long)KCSM * G::XCH;
  if (smem > 220 * 1024) return -8;
  auto k = pwc<RT, WN, WTOK, TTW, CPS, ST, CLU, MINB, RTR>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  if (!CLU || KS == 1) {
    k<<<dim3(NG, KS), G::NTH, (size_t)smem, st>>>(p);
    return (int)cudaGetLastError();
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NG, KS); cfg.blockDim = dim3(G::NTH); cfg.dynamicSmemBytes = (size_t)smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = KS; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  static int ck_key[8] = {0}, ck_ok[8] = {0};  // cached cudaOccupancyMaxActiveClusters per (KS, smem)
  const int key = KS * 1000000 + (int)smem;
  int ok = -1;
  for (int i = 0; i < 8; i++) if (ck_key[i] == key) ok = ck_ok[i];
  if (ok < 0) {
    int nc = 0;
    const cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, k, &cfg);
    if (e != cudaSuccess) cudaGetLastError();
    ok = e == cudaSuccess && nc > 0;
    for (int i = 0; i < 8; i++) if (!ck_key[i]) { ck_key[i] = key; ck_ok[i] = ok; break; }
  }
  if (!ok) return -9;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, k, p);
  if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
  return 0;
}

// ---------------------------------------------------------------- kind "w": wgmma with A from registers (m = 128 / 256)
// out[tok][n] = sum_k W[n][k] x[tok][k]: wgmma M = W rows (64 per warpgroup; warp i of a warpgroup owns rows 16i..16i+15,
// its A fragment is the mma.sync m16n8k16 A layout), N = tokens (NT = 128 or 256), K = 16 per instruction. A comes from
// the decoded 12-bit pack (natural k order within each 64-chunk: _pack_nat); B = the x tile [NT tokens x 64 k] that one
// 2D TMA copy lands in smem with 128B swizzle (K-major B: descriptor LBO 16 B (unused), SBO 1024 B, swizzle mode 1).
// CTA = NWG consumer warpgroups (BN = 64*NWG W rows) + 1 producer warpgroup (one lane issues; the warpgroup exits after
// its last stage). Stage = [x tile NT*128 B][W chunk RTB*1536 B], one full barrier (expect_tx = both). A chunk's slot is
// released only after wgmma.wait_group shows its MMAs complete (x is read asynchronously). A fragments are double
// buffered so the decode of chunk u overlaps the MMAs of chunk u-1. Split-K over gridDim.y as clusters (1, KS, 1): each
// CTA parks its fp32 partial in the drained ring (fragment order), rank r sums its rectangle of the n-block over DSMEM,
// transposes it through smem and stores 16-B rows of out (bf16, or fp32 for the router).
template <int NWG, int NT, int ST, int RTR>
struct PwW {
  static constexpr int NWC = 4 * NWG, NTH = (NWC + 4) * 32, BN = 64 * NWG, RTB = BN / 16, TT = NT / 8, ND = NT / 2;
  static constexpr int XB = NT * 128, WB = RTB * 1536, SB = (XB + WB + 1023) / 1024 * 1024;
  static constexpr int SMEM = 1024 + ST * SB + ST * 16;
  static constexpr int FRAG = RTB * TT * 32;  // float4 fragments of one n-block partial
};
// smem matrix descriptor: start >> 4 (bits 0-13), LBO >> 4 (16-29), SBO >> 4 (32-45), base offset 0 (49-51; stage
// bases are 1024-aligned), swizzle mode 1 = 128B (62-63). K-major B, 128B swizzle: 8-row groups 1024 B apart.
__device__ __forceinline__ uint64_t pw_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void pw_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void pw_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void pw_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int ND> __device__ __forceinline__ void pw_fence_acc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void pw_fence_a(uint32_t (&A)[1][16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[0][i]) :: "memory");
}
template <int NT> __device__ __forceinline__ void pw_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc);
@@WGMMA@@

template <int NWG, int NT, int ST, int RTR>
__global__ void __launch_bounds__(PwW<NWG, NT, ST, RTR>::NTH, 1) pwg(const PwP p, const __grid_constant__ PwTmap tm) {
  using G = PwW<NWG, NT, ST, RTR>;
  constexpr int NWC = G::NWC, BN = G::BN, RTB = G::RTB, TT = G::TT, ND = G::ND, XB = G::XB, WB = G::WB, SB = G::SB;
  constexpr int FRAG = G::FRAG, NC = NWC * 32;
  extern __shared__ __align__(128) uint8_t smem[];  // ring base re-aligned to 1024 below
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t full = sbase + ST * SB, empty = full + ST * 8;
  const int tid = threadIdx.x, warp = tid >> 5;
  int lane = tid & 31;
  asm volatile("" : "+r"(lane));
  const int nb = blockIdx.x, ks = blockIdx.y, NB = gridDim.x, KS = gridDim.y, m = p.m;
  const int NST = p.KCW, s0 = (int)((long long)ks * NST / KS), s1 = (int)((long long)(ks + 1) * NST / KS);
  const int KCS = s1 - s0, c0 = s0;
  const size_t cstr = (size_t)RTB * 1536;
  auto issue = [&](int s, int slot) {
    const uint32_t bar = full + slot * 8, dst = sbase + slot * SB;
    pw_mbexpect(bar, XB + WB);
    pw_tile(dst, &tm, ((c0 + s) * 64) % p.K, 0, bar);
    pw_bulk(dst + XB, p.W + ((size_t)nb * p.KCW + c0 + s) * cstr, WB, bar);
  };
  if (tid == NWC * 32) {  // producer lane: barriers, then the first ST stages before anything else
    for (int i = 0; i < ST; i++) { pw_mbinit(full + i * 8, 1); pw_mbinit(empty + i * 8, NWC); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int s = 0; s < ST && s < KCS; s++) issue(s, s);
  }
  int ep = 0;
  uint32_t nx = 0xffffffffu;
  if (warp < NWC) { ep = __ldg(p.eoff + (ks * NB + nb) * NWC + warp); nx = __ldg(p.esc + ep); }
  __syncthreads();
  if (warp >= NWC) {
    // producer warpgroup: one lane streams the remaining stages, then the whole warpgroup exits
    if (NWG == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == NWC && lane == 0) {
      int slot = 0, round = 1;
      for (int s = ST; s < KCS; s++) {
        pw_mbwait(empty + slot * 8, (round - 1) & 1);
        issue(s, slot);
        if (++slot == ST) { slot = 0; round++; }
      }
    }
    return;
  }
  if (NWG == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  float D[ND];
#pragma unroll
  for (int i = 0; i < ND; i++) D[i] = 0.f;
  uint32_t Tb = p.Tb, Tb2 = p.Tb2;
  asm volatile("" : "+r"(Tb), "+r"(Tb2));
  uint32_t nxs = nx >> 17;
  const uint32_t wl = sbase + XB + warp * 1536 + lane * 16;  // this warp's 16-row tile in a stage's W chunk
  uint32_t A0[1][16], A1[1][16];
  auto decode = [&](const int u, uint32_t (&A)[1][16]) {
    const int slot = u % ST;
    pw_mbwait(full + slot * 8, (u / ST) & 1);
    const uint32_t a = wl + slot * SB;
    const uint4 q0 = pw_lds(a), q1 = pw_lds(a + 512), q2 = pw_lds(a + 1024);
    pw_unit(q0, q1, q2, RTR ? (c0 + u >= p.kx ? Tb2 : Tb) : Tb, A[0]);
    pw_fix2<1>(A, u, nx, nxs, ep, p.esc, lane);
  };
  auto mma = [&](const int u, uint32_t (&A)[1][16]) {
    const uint64_t d0 = pw_desc(sbase + (u % ST) * SB);
    pw_fence_acc<ND>(D);
    pw_wg_fence();
#pragma unroll
    for (int q = 0; q < 4; q++) pw_wgmma<NT>(D, A[0] + 4 * q, d0 + 2 * q);  // +32 B per k16 step inside the 128-B row
    pw_wg_commit();
    pw_fence_acc<ND>(D);
  };
  auto release = [&](const int u, uint32_t (&A)[1][16]) {  // chunk u's MMAs are complete: its A set and slot are free
    pw_fence_a(A);
    __syncwarp();
    if (lane == 0) pw_mbarrive(empty + (u % ST) * 8);
  };
  decode(0, A0);
  mma(0, A0);
  int u = 1;
#pragma unroll 1
  for (; u + 1 < KCS; u += 2) {
    decode(u, A1); mma(u, A1); pw_wg_wait<1>(); release(u - 1, A0);
    decode(u + 1, A0); mma(u + 1, A0); pw_wg_wait<1>(); release(u, A1);
  }
  if (u < KCS) {
    decode(u, A1); mma(u, A1); pw_wg_wait<1>(); release(u - 1, A0);
    pw_wg_wait<0>(); release(u, A1);
  } else {
    pw_wg_wait<0>(); release(u - 1, A0);
  }
  pw_fence_acc<ND>(D);
  // ---- epilogue, consumer warps only (named barrier 1). Every slot was released after its MMAs completed and every
  // stage was consumed, so after this barrier the ring area is free.
  pw_cbar(NC);
  const uint32_t stg = sbase;
#pragma unroll
  for (int j = 0; j < TT; j++) {  // D[4j..4j+3]: rows g / g+8 of this warp's tile, tokens 8j + 2t, +1 (= mma C layout)
    const uint32_t a = stg + (((warp * TT + j) * 32) + lane) * 16;
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "r"(a), "f"(D[4 * j]), "f"(D[4 * j + 1]), "f"(D[4 * j + 2]), "f"(D[4 * j + 3]) : "memory");
  }
  if (KS > 1) pw_csync(); else pw_cbar(NC);
  // rank r owns a rectangle of the n-block: whole row tiles (KS <= RTB) or one tile x a token range (KS > RTB)
  const int r = KS > 1 ? (int)pw_crank() : 0;
  int tile0, ntile, tt0, ntt;
  if (KS <= RTB) { ntile = RTB / KS; tile0 = r * ntile; tt0 = 0; ntt = TT; }
  else { const int sp = KS / RTB; ntile = 1; tile0 = r / sp; ntt = TT / sp; tt0 = (r % sp) * ntt; }
  const int rows = ntile * 16, esz = p.of ? 4 : 2;
  const uint32_t tb = stg + FRAG * 16;  // transpose buffer [ntt*8 tokens][rows]
  for (int e = tid; e < ntile * ntt * 32; e += NC) {
    const int l = e & 31, jj = (e >> 5) % ntt, ti = (e >> 5) / ntt;
    const int idx = ((tile0 + ti) * TT + tt0 + jj) * 32 + l, tok = (tt0 + jj) * 8 + 2 * (l & 3);
    if (tok < m) {
      float4 v[8];
#pragma unroll
      for (int q = 0; q < 8; q++)
        if (q < KS) {
          if (KS > 1) v[q] = pw_ldc(stg + idx * 16, q);
          else { const uint4 w = pw_lds(stg + idx * 16); v[q] = make_float4(__uint_as_float(w.x), __uint_as_float(w.y), __uint_as_float(w.z), __uint_as_float(w.w)); }
        }
      float4 a = v[0];
#pragma unroll
      for (int q = 1; q < 8; q++) if (q < KS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; }
      const int rl = ti * 16 + (l >> 2), tl = jj * 8 + 2 * (l & 3);
      const float vv[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int c = 0; c < 4; c++) {  // c: (tok + (c & 1), row + 8 * (c >> 1))
        if (tok + (c & 1) < m) {
          const uint32_t o = tb + ((tl + (c & 1)) * rows + rl + 8 * (c >> 1)) * esz;
          if (esz == 4) asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(o), "f"(vv[c]) : "memory");
          else { const __nv_bfloat16 h = __float2bfloat16(vv[c]); asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(o), "h"(*(const unsigned short*)&h) : "memory"); }
        }
      }
    }
  }
  pw_cbar(NC);
  const int segs = rows * esz / 16;  // 16-B pieces of one token's row segment
  for (int e = tid; e < ntt * 8 * segs; e += NC) {
    const int tl = e / segs, sg = e - tl * segs, tok = tt0 * 8 + tl;
    if (tok < m) {
      const uint4 v = pw_lds(tb + tl * rows * esz + sg * 16);
      uint8_t* dst = (uint8_t*)p.out + ((size_t)tok * p.N + (size_t)nb * BN + tile0 * 16) * esz + sg * 16;
      *(uint4*)dst = v;
    }
  }
  if (KS > 1) pw_csync();  // peers may still read this CTA's staged partial
}

template <int NWG, int NT, int ST, int RTR>
static int pw_wlaunch(const PwP& p, const PwTmap& tm, int KS, cudaStream_t st) {
  using G = PwW<NWG, NT, ST, RTR>;
  if (p.m < 1 || p.m > NT || p.N % G::BN || p.RTBW != G::RTB || p.NBW * G::BN < p.N) return -4;
  if (p.K % 64 || (p.KCW * 64) % p.K || p.kx * 64 != p.K || KS < 1 || KS > 8 || KS > p.KCW) return -5;
  if (p.of != RTR || (!RTR && p.Tb2 != p.Tb)) return -5;
  if (KS <= G::RTB ? (G::RTB % KS) : (KS % G::RTB || G::TT % (KS / G::RTB))) return -6;
  const int NB = p.N / G::BN, KCSM = (p.KCW + KS - 1) / KS;
  if (KCSM >= 32000 || p.eoffn < KS * NB * G::NWC) return -7;
  const int ntile = KS <= G::RTB ? G::RTB / KS : 1, ntt = KS <= G::RTB ? G::TT : G::TT / (KS / G::RTB);
  if ((long long)G::FRAG * 16 + (long long)ntt * 8 * ntile * 16 * (RTR ? 4 : 2) > (long long)ST * G::SB) return -8;
  auto k = pwg<NWG, NT, ST, RTR>;
  static int attr = 0;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    attr = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NB, KS); cfg.blockDim = dim3(G::NTH); cfg.dynamicSmemBytes = G::SMEM; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = KS; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = KS > 1 ? 1 : 0;
  if (KS > 1) {
    static int ck_key[8] = {0}, ck_ok[8] = {0};  // cached cudaOccupancyMaxActiveClusters per KS
    int ok = -1;
    for (int i = 0; i < 8; i++) if (ck_key[i] == KS) ok = ck_ok[i];
    if (ok < 0) {
      int nc = 0;
      const cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, k, &cfg);
      if (e != cudaSuccess) cudaGetLastError();
      ok = e == cudaSuccess && nc > 0;
      for (int i = 0; i < 8; i++) if (!ck_key[i]) { ck_key[i] = KS; ck_ok[i] = ok; break; }
    }
    if (!ok) return -9;
  }
  const cudaError_t e = cudaLaunchKernelEx(&cfg, k, p, tm);
  if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
  return 0;
}

int pw_dispatch(int inst, const PwP& p, int NG, int KS, cudaStream_t st) {
  switch (inst) {
@@BCASES@@
  }
  return -2;
}
int pw_wdispatch(int inst, const PwP& p, const PwTmap& tm, int KS, cudaStream_t st) {
  switch (inst) {
@@WCASES@@
  }
  return -2;
}
__global__ void pw_probe_empty(uint32_t* sink) { if (threadIdx.x == 0) sink[blockIdx.x] = 1; }
// pure stream reference (pw2's bulk probe): warp 1 lane 0 streams SZ-byte stages into an ST-deep ring; warp 0 waits,
// touches one uint4 per lane, releases
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
// carveout probes: an empty kernel with opted-in dynamic smem; the pw2 register-ring stream with unused dynamic smem;
// a 256 MB read-only flush kernel with or without dynamic smem
__device__ __forceinline__ uint4 pw_gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__global__ void __launch_bounds__(256) pw_probe_empty2(uint32_t* sink) { if (threadIdx.x == 0) sink[blockIdx.x] = 1; }
__global__ void __launch_bounds__(256) pw_probe_ring8(const uint4* buf, long long per16, uint32_t* sink) {
  constexpr int D = 8;
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
}
__global__ void __launch_bounds__(1024) pw_probe_flush(const float4* buf, long long n4, uint32_t* sink) {
  float a = 0.f;
  for (long long i = blockIdx.x * 1024ll + threadIdx.x; i < n4; i += (long long)gridDim.x * 1024) {
    const float4 v = __ldg(buf + i); a += v.x + v.y + v.z + v.w;
  }
  if (a == 1.2345f) sink[blockIdx.x] = 1;
}
// kind 0 empty (grid x 256, smem), 1 ring8 (grid x 256, smem; per = bytes per CTA), 2 flush (132 x 1024, smem; per = float4s)
int pw_carve_launch(int kind, const void* buf, long long per, int grid, int smem, uint32_t* sink, cudaStream_t st) {
  static int attr = 0;
  if (!attr) {
    if (cudaFuncSetAttribute(pw_probe_empty2, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024) != cudaSuccess ||
        cudaFuncSetAttribute(pw_probe_ring8, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024) != cudaSuccess ||
        cudaFuncSetAttribute(pw_probe_flush, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024) != cudaSuccess) { cudaGetLastError(); return -3; }
    attr = 1;
  }
  if (smem < 0 || smem > 220 * 1024) return -1;
  if (kind == 0) pw_probe_empty2<<<grid, 256, smem, st>>>(sink);
  else if (kind == 1) pw_probe_ring8<<<grid, 256, smem, st>>>((const uint4*)buf, per / 16, sink);
  else if (kind == 2) pw_probe_flush<<<grid, 1024, smem, st>>>((const float4*)buf, per, sink);
  else return -1;
  return (int)cudaGetLastError();
}
int pw_pure_launch(const void* buf, long long per, int SZ, int ST, int grid, uint32_t* sink, cudaStream_t st) {
  const int sm = ST * SZ + ST * 16;
  if (SZ % 16 || sm > 220 * 1024) return -1;
  static int attr = 0;
  if (!attr) { if (cudaFuncSetAttribute(pw_probe_bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024) != cudaSuccess) { cudaGetLastError(); return -3; } attr = 1; }
  pw_probe_bulk<<<grid, 64, sm, st>>>((const uint8_t*)buf, per, SZ, ST, sink);
  return (int)cudaGetLastError();
}
int pw_empty_launch(uint32_t* sink, int grid, cudaStream_t st) { pw_probe_empty<<<grid, 32, 0, st>>>(sink); return (int)cudaGetLastError(); }
"""

def _wgmma_src(n):
    """pw_wgmma<n>: one wgmma.mma_async m64n<n>k16 f32 += bf16 (A: 4 registers) x bf16 (B: smem descriptor);
    scale-d = 1 (accumulate; D is zeroed with a wgmma.fence before the first chunk), scale-a = scale-b = 1, trans-b = 0."""
    nd = n // 2
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("template <> __device__ __forceinline__ void pw_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(1));\n}\n")
            % (n, nd, nd + 5, n, d, a, nd + 4, outs))


KERNELS_SRC = _KERNELS_TEMPLATE.replace("@@BCASES@@", "\n".join(
    "    case %d: return pw_launch<%s>(p, NG, KS, st);" % (i, ",".join(map(str, PW_INSTS[i][2]))) for i in PW_BIDX)).replace(
    "@@WCASES@@", "\n".join("    case %d: return pw_wlaunch<%s>(p, tm, KS, st);" % (i, ",".join(map(str, PW_INSTS[i][2]))) for i in PW_WIDX)).replace(
    "@@WGMMA@@", _wgmma_src(128) + _wgmma_src(256))

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
  p.flags = 0; p.stb = 0; p.xtok = 0;
  p.Tb = (uint32_t)(Tb & 0xffffffff); p.Tb2 = p.Tb; p.kx = p.K / 64;
  p.of = out.scalar_type() == torch::kFloat32;
  return 0;
}

int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t st, int64_t Tb,
               int64_t flags, int64_t stb, int64_t xtok, int64_t Tb2) {
  PwP p;
  const int r = pw_fill(p, W, x, out, ws, cnt, esc, eoff, xrow, st, Tb);
  if (r) return r;
  p.Tb2 = (uint32_t)(Tb2 & 0xffffffff); p.flags = (int)flags; p.stb = (int)stb; p.xtok = (int)xtok;
  return (int64_t)pw_dispatch((int)inst, p, (int)NG, (int)KS, at::cuda::getCurrentCUDAStream());
}

// x tensor map encoded on the host for this call's x pointer (box 64 k x box_rows rows, 128B swizzle, OOB rows zero).
// Under CUDA-graph capture the 128-byte map is baked into the kernel parameters exactly like a raw x pointer would be,
// so a replay is correct whenever the replay reuses the captured x storage (the condition every kernel argument has).
int64_t pw_run_tma(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
                   torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t box_rows, int64_t Tb, int64_t Tb2) {
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
  p.Tb2 = (uint32_t)(Tb2 & 0xffffffff);
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
  return (int64_t)pw_wdispatch((int)inst, p, tm, (int)KS, at::cuda::getCurrentCUDAStream());
}

int64_t pw_pure(torch::Tensor buf, int64_t per, int64_t SZ, int64_t ST, int64_t grid, torch::Tensor sink) {
  if (sink.numel() < grid || per * grid > buf.numel() * buf.element_size()) return -10;
  return (int64_t)pw_pure_launch(buf.data_ptr(), (long long)per, (int)SZ, (int)ST, (int)grid, (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}

int64_t pw_carve(int64_t kind, torch::Tensor buf, int64_t per, int64_t grid, int64_t smem, torch::Tensor sink) {
  if (sink.numel() < grid) return -10;
  return (int64_t)pw_carve_launch((int)kind, buf.data_ptr(), (long long)per, (int)grid, (int)smem, (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}

int64_t pw_empty(torch::Tensor sink, int64_t grid) {
  if (sink.numel() < grid) return -10;
  return (int64_t)pw_empty_launch((uint32_t*)sink.data_ptr(), (int)grid, at::cuda::getCurrentCUDAStream());
}
"""

CUDA_SRC = KERNELS_SRC + GLUE_SRC

CPP_SRC = ("int64_t pw_run(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t NG, int64_t KS, int64_t xrow, int64_t st, int64_t Tb, int64_t flags, int64_t stb, int64_t xtok, int64_t Tb2);\n"
           "int64_t pw_run_tma(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, "
           "torch::Tensor esc, torch::Tensor eoff, int64_t KS, int64_t box_rows, int64_t Tb, int64_t Tb2);\n"
           "int64_t pw_pure(torch::Tensor buf, int64_t per, int64_t SZ, int64_t ST, int64_t grid, torch::Tensor sink);\n"
           "int64_t pw_carve(int64_t kind, torch::Tensor buf, int64_t per, int64_t grid, int64_t smem, torch::Tensor sink);\n"
           "int64_t pw_empty(torch::Tensor sink, int64_t grid);")
FUNCTIONS = ["pw_run", "pw_run_tma", "pw_pure", "pw_carve", "pw_empty"]


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


def _pack_nat(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in NATURAL k order, for wgmma (B = x from smem in
    natural k order): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e].
    Same register order as _pack (pw_dec output a0..a3 = {row g k-lo, row g+8 k-lo, row g k-hi, row g+8 k-hi}); only
    the k assigned to each slot differs (_pack permutes k so mma.sync B can be read as 16 contiguous bf16 per lane)."""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


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
def _pw_c12(state, layer, nat=False):
    if nat:  # natural-k packs for the wgmma kernel, cached under their own key
        cache = state.setdefault("pw_c12nat", {})
        if layer not in cache:
            pk = _pack_nat
            try:
                if layer == "router":
                    w = state["raw"]["router"]
                    hi = w.to(torch.bfloat16)
                    lo = (w - hi.float()).to(torch.bfloat16)
                    a, b = _c12(pk(hi), 1), _c12(pk(lo), 1)
                    r = None
                    if a is not None and b is not None:
                        eb = b[2].clone()
                        eb[:, 1] += a[0].shape[1]
                        r = (torch.cat((a[0], b[0]), 1).contiguous(), (a[1], b[1]), torch.cat((a[2], eb)))
                else:
                    r = _c12(pk(state["raw"][layer]), 1)
            except Exception as error:
                _log("PW c12nat %s failed: %s" % (layer, str(error).splitlines()[-1][:160] if str(error) else type(error).__name__))
                r = None
            if r is None:
                _log("PW c12nat %s: pack not exact, wgmma paths decline" % layer)
            cache[layer] = r
        return cache[layer]
    cache = state.setdefault("pw_c12", {})
    if layer not in cache and layer == "router":
        # W' = [hi | lo] along K: each pack proven exact by _c12 on its own; escapes of lo shifted by K/64 chunks
        try:
            w = state["raw"]["router"]
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            a, b = _c12(_pack(hi), 1), _c12(_pack(lo), 1)
            r = None
            if a is not None and b is not None:
                eb = b[2].clone()
                eb[:, 1] += a[0].shape[1]
                r = (torch.cat((a[0], b[0]), 1).contiguous(), (a[1], b[1]), torch.cat((a[2], eb)))
        except Exception as error:
            _log("PW c12 router failed: %s" % (str(error).splitlines()[-1][:160] if str(error) else type(error).__name__))
            r = None
        if r is None:
            _log("PW c12 router: pack not exact, pw declines")
        cache[layer] = r
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


def _pw_layout(state, layer, rtb, nat=False):
    """c12 units [T, KC, 3, 32, 16] -> [T/rtb, KC, rtb, 1536]: one n-block's K-slice is one contiguous byte range"""
    cache = state.setdefault("pw_lay", {})
    if (layer, rtb, nat) not in cache:
        c = _pw_c12(state, layer, nat)
        lay = None
        if c is not None:
            data = c[0]
            T, KC = data.shape[:2]
            if T % rtb == 0:
                lay = data.reshape(T // rtb, rtb, KC, 1536).permute(0, 2, 1, 3).contiguous()
        cache[(layer, rtb, nat)] = lay
    return cache[(layer, rtb, nat)]


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


def _pw_xb(TOK, xrow):
    """x-slice bytes of the chunk-major layout: KCSmax chunks x TOK rows x 144 B (xrow = KCSmax*128 + 16)"""
    return TOK * ((xrow - 16) // 128) * 144


def _pw_cgrid(cfg, occ, st, n, K):
    """cluster paths: grid (NB, KS), one n-block per CTA, KS in {8, 4, 2, 1} (portable cluster sizes, dividing the
    16 row tiles of a CTA) -- the largest with NB*KS <= 132*occ whose x slice + W ring fits; (NG, KS, xrow) or None"""
    RT, WN, WTOK, TTW, CPS = cfg[:5]
    RTB, TOK = WN * RT, WTOK * TTW * 8
    NB, NST = n // (RTB * 16), K // 64 // CPS
    for KS in (8, 4, 2, 1):
        if KS > NST or (RTB * WTOK * TTW * 32) % KS or (NB * KS > 132 * occ and KS > 1):
            continue
        xrow = -(-NST // KS) * CPS * 128 + 16
        if _pw_xoff(cfg, st) + _pw_xb(TOK, xrow) <= PW_SMEM // occ and _pw_xoff(cfg, st) >= RTB * WTOK * TTW * 32 * 16:
            return NB, KS, xrow
    return None


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
        if _pw_xb(TOK, xrow) > cap:
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


def _pw_wks(NB, KC, RTB, TT, mode):
    """wgmma split-K: KS in {1, 2, 4, 8} (portable clusters) whose rank rectangles tile the n-block; mode 'a' = the
    largest with NB*KS <= 132 (one wave at 1 CTA/SM), mode 'b' = the next larger one; None if there is none"""
    ok = [k for k in (1, 2, 4, 8) if k <= KC and (RTB % k == 0 if k <= RTB else (k % RTB == 0 and TT % (k // RTB) == 0))]
    a = max([k for k in ok if NB * k <= 132] or [ok[0]])
    if mode == "a":
        return a
    b = [k for k in ok if k > a]
    return b[0] if b else None


def _pw_wrunner(state, layer, m, idx, mode):
    """kind "w" (wgmma) launch closure: (run(x, out) -> launch code, description) or None"""
    name, kind, cfg, ms, occ, st = PW_INSTS[idx]
    NWG, NT, ST, RTR = cfg
    BN, NWC = 64 * NWG, 4 * NWG
    RTB, TT = BN // 16, NT // 8
    if bool(RTR) != (layer == "router") or not (1 <= m <= NT):
        return None
    raw = state["raw"][layer]
    n, K = _n_of(layer, m), raw.shape[1] * (2 if RTR else 1)
    KC, NB = K // 64, n // BN
    if n % BN:
        return None
    KS = _pw_wks(NB, KC, RTB, TT, mode)
    if KS is None:
        return None
    lay = _pw_layout(state, layer, RTB, nat=True)
    if lay is None:
        return None
    _, tb, esc6 = _pw_c12(state, layer, nat=True)
    tb, tb2 = tb if RTR else (tb, tb)
    el = _pw_esc(esc6, n, (1, NWC, 1, 1), KC, NB, KS, raw.device)
    if el is None:
        return None
    esc, eoff = el
    dws = torch.zeros((4,), dtype=torch.float32, device=raw.device)
    dcnt = torch.zeros((4,), dtype=torch.int32, device=raw.device)
    ext = _ext()

    def run(x, out):
        return ext.pw_run_tma(idx, lay, x, out, dws, dcnt, esc, eoff, KS, NT, tb, tb2)
    return run, "%dx%d" % (NB, KS)


def _pw_wpath(pname, iname, mode, layers):
    idx = PW_IDX[iname]
    ms = PW_INSTS[idx][3]

    def serves(layer, m):
        return layer in layers and m in ms

    def build(state, layer, m):
        r = _pw_wrunner(state, layer, m, idx, mode)
        if r is None:
            return None
        launch, desc = r
        base = _fallback(state, layer, _n_of(layer, m))
        _log("PW %s %s m=%d %s" % (pname, layer, m, desc))

        def run(x, out):
            if not x.is_contiguous() or x.data_ptr() % 16:
                base(x, out)
                return
            if launch(x, out):  # a host-side launch error code; the fallback keeps the output correct
                base(x, out)
        return run

    path(pname, serves)(build)


def _pw_runner(state, layer, m, idx, st=None, occ=None, half=False, grid=None, flags=0, stb=0, xtok=0):
    """-> (run(x, out) -> launch code, description, info dict) or None. Workspace and counters are private to the call.
    grid / flags / stb / xtok are probe overrides (PRB instances only)."""
    name, kind, cfg, ms, occ0, st0 = PW_INSTS[idx]
    occ, st = occ or occ0, st or st0
    RT, WN, WTOK, TTW = cfg[:4]
    CPS = cfg[4] if kind == "b" else 1
    RTB, TOK = WN * RT, WTOK * TTW * 8
    raw = state["raw"][layer]
    rtr = kind == "b" and cfg[8]
    if rtr != (layer == "router"):
        return None
    n, K = _n_of(layer, m), raw.shape[1] * (2 if rtr else 1)  # router: K' = 2K over W' = [hi | lo]
    KC = K // 64
    if n % (RTB * 16) or KC % CPS or not (m <= max(TOK, xtok) and (xtok or rtr or m > TOK - 8)):
        return None
    NB, NST = n // (RTB * 16), KC // CPS
    if kind == "b":
        if grid is None and cfg[6]:
            g = _pw_cgrid(cfg, occ, st, n, K)
            if g is None:
                return None
            NG, KS, xrow = g
        elif grid is None:
            g = _pw_grid(cfg, occ, st, n, K)
            if g is None:
                return None
            NG, KS, xrow = g
        else:
            NG, KS = grid
            xrow = -(-NST // KS) * CPS * 128 + 16
    else:
        NG, KS, xrow = NB, _pw_tks(NB, half), 16
    lay = _pw_layout(state, layer, RTB)
    if lay is None:
        return None
    _, tb, esc6 = _pw_c12(state, layer)
    tb, tb2 = tb if rtr else (tb, tb)
    el = _pw_esc(esc6, n, (RT, WN, WTOK, CPS), KC, NG, KS, raw.device)
    if el is None:
        return None
    esc, eoff = el
    ws = torch.zeros((n * TOK,), dtype=torch.float32, device=raw.device)  # private: a broken variant cannot poison
    cnt = torch.zeros((NB,), dtype=torch.int32, device=raw.device)       # another's workspace
    ext = _ext()
    wb = stb or CPS * RTB * 1536
    kcs = -(-NST // KS) * CPS
    nbr = -(-NB // NG)
    info = {"NG": NG, "KS": KS, "grid": NG * KS, "st": st, "stage_bytes": wb, "nb_per_cta": nbr,
            "stages_per_cta": nbr * -(-(kcs * RTB * 1536) // wb), "w_bytes_per_cta": nbr * kcs * RTB * 1536,
            "smem": (_pw_xoff(cfg, st) if kind == "b" else 0) + _pw_xb(TOK, xrow)}
    if kind == "b":
        def run(x, out):
            return ext.pw_run(idx, lay, x, out, ws, cnt, esc, eoff, NG, KS, xrow, st, tb, flags, stb, xtok, tb2)
        desc = "%dx%d" % (NG, KS)
    else:
        def run(x, out):
            return ext.pw_run_tma(idx, lay, x, out, ws, cnt, esc, eoff, KS, TOK, tb)
        desc = "NB=%d KS=%d grid=%d st=%d" % (NB, KS, NB * KS, st)
    return run, desc, info


PW_IDX = {e[0]: i for i, e in enumerate(PW_INSTS)}


def _pw_path(pname, iname, staged=False, half=False, layers=PW_LAYERS, grid=None):
    idx = PW_IDX[iname]
    ms = PW_INSTS[idx][3]

    def serves(layer, m):
        return layer in layers and m in ms

    def build(state, layer, m):
        r = _pw_runner(state, layer, m, idx, half=half, grid=grid)
        if r is None:
            return None
        launch, desc, _ = r
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


for _p in ("b16t1g", "b16t2g", "b16t4g", "b16t1c", "b16t2c", "b16t4c", "r2t2c", "r2t4c"):
    _pw_path("pw-" + _p, _p)
_pw_path("pw-b8t1", "b8t1", layers=("unpooler",))
_pw_path("pw-b8t1c", "b8t1c", layers=("unpooler",))
for _w in ("w2n128", "w1n128", "w2n256", "w1n256"):  # wgmma m = 128 / 256, two split-K choices each
    for _mode in ("a", "b"):
        _pw_wpath("pw-%s%s" % (_w, _mode), _w, _mode, PW_LAYERS)
for _mode in ("a", "b"):
    _pw_wpath("pw-wr%s" % _mode, "wr", _mode, ("router",))
for _t in (1, 4):  # router: cluster (KS 8 auto) and global (KS 16 / 32) epilogues
    _pw_path("pw-rtc%d" % _t, "rc%d" % _t, layers=("router",))
    _pw_path("pw-rtg16-%d" % _t, "rg%d" % _t, layers=("router",), grid=(8, 16))
    _pw_path("pw-rtg32-%d" % _t, "rg%d" % _t, layers=("router",), grid=(8, 32))
_pw_path("pw-rtg128", "rg128", layers=("router",), grid=(8, 32))


# ---------------------------------------------------------------- PROBE: L1/shared-memory carveout (no selection)
PROBE_BUDGET_S = 15.0


def _time_us_flush(fn, x, out, flush_fn, it=15):
    """_time_us with a custom cold-L2 flush callable (the harness's _time_us is left as is)"""
    g = _graph(fn, x, out)
    times = []
    for _ in range(it):
        flush_fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _probe(state, flush, sink):
    """Does a change of dynamic shared memory (and so of the L1/smem carveout) between the flush kernel and ours cost
    launch time? PROBE carve <name> <us>, timed exactly like the bake-off (256 MB torch.sum flush before each replay)
    unless the line says otherwise."""
    t0 = time.time()
    ext = _ext()
    dev = flush.device
    sk = torch.zeros((4096,), dtype=torch.int32, device=dev)
    dummy = torch.empty((1, 1), device=dev)

    def timed(label, fn, flush_fn=None):
        if time.time() - t0 > PROBE_BUDGET_S:
            _log("PROBE carve %s skipped-budget" % label)
            return
        try:
            if fn(dummy, dummy):
                _log("PROBE carve %s launch-error" % label)
                return
            us = _time_us(fn, dummy, dummy, flush, sink) if flush_fn is None else _time_us_flush(fn, dummy, dummy, flush_fn)
            _log("PROBE carve %s %.2f" % (label, us))
        except Exception as error:
            _log("PROBE carve %s ERROR %s" % (label, str(error).splitlines()[-1][:120]))

    for smem in (0, 64 << 10, 200 << 10):
        timed("empty 132x256 smem=%dK" % (smem >> 10), lambda a, b, sm=smem: ext.pw_carve(0, sk, 0, 132, sm, sk))
    timed("empty 132x256 smem=200K x2", lambda a, b: ext.pw_carve(0, sk, 0, 132, 200 << 10, sk) or ext.pw_carve(0, sk, 0, 132, 200 << 10, sk))
    lay = _pw_layout(state, "pooler", 16)
    if lay is None:
        _log("PROBE carve no pooler pack")
        return
    total = lay.numel()
    bulk = {}
    for name, sz, stg in (("8K", 4096, 2), ("16K", 8192, 2), ("32K", 16384, 2), ("64K", 16384, 4), ("100K", 25600, 4), ("160K", 40960, 4)):
        per = total // 132 // sz * sz
        bulk[name] = (lambda a, b, per=per, sz=sz, stg=stg: ext.pw_pure(lay, per, sz, stg, 132, sk))
        timed("bulk ring=%s (%dKx%d) grid=132 MB=%.1f" % (name, sz >> 10, stg, per * 132 / 1e6), bulk[name])
    per = total // 528 // 4096 * 4096
    for smem in (0, 48 << 10):  # 4 CTAs/SM x 48 KB fits; 64 KB would drop residency to 3 CTAs/SM
        timed("ring D=8 8w x 528 smem=%dK MB=%.1f" % (smem >> 10, per * 528 / 1e6), lambda a, b, sm=smem: ext.pw_carve(1, lay, per, 528, sm, sk))
    n4 = flush.numel() // 4
    timed("bulk ring=64K flush=torch.sum", bulk["64K"])
    timed("bulk ring=64K flush=own-kernel smem=200K", bulk["64K"], lambda: ext.pw_carve(2, flush, n4, 132, 200 << 10, sk))
    timed("bulk ring=64K flush=own-kernel smem=0", bulk["64K"], lambda: ext.pw_carve(2, flush, n4, 132, 0, sk))
    _log("PROBE done in %.1fs" % (time.time() - t0))


# ---------------------------------------------------------------- STATS: bf16 weight statistics (codec headroom)
def _stats_line(name, w16):
    """one STATS line for a bf16 tensor: exponent spread / windows, escape rates, mantissa trailing zeros, zeros,
    subnormals, Shannon entropy (bits/weight) of the 16-bit value, the 8-bit exponent and the 7-bit mantissa"""
    b = w16.contiguous().view(torch.int16).reshape(-1).int() & 0xffff
    nw = b.numel()
    e, man = (b >> 7) & 0xff, b & 0x7f
    he = torch.bincount(e, minlength=256).double()
    hm = torch.bincount(man, minlength=128).double()
    hv = torch.bincount(b, minlength=65536).double()

    def ent(h):
        pr = h[h > 0] / nw
        return float(-(pr * torch.log2(pr)).sum())

    hc = he.cpu()
    cs = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(hc, 0)))
    best = [float((cs[L:] - cs[:-L]).max()) / nw for L in range(1, 257)]  # best coverage of an L-code window
    win = lambda frac: next(L for L in range(1, 257) if best[L - 1] >= frac - 1e-12)
    mode = int(hc.argmax())
    lo8 = max(0, mode - 4)
    out8 = 1.0 - float(hc[lo8:lo8 + 8].sum()) / nw
    kv = [("distinct_exp", int((hc > 0).sum())), ("win99", win(0.99)), ("win999", win(0.999)), ("win9999", win(0.9999)),
          ("win100", win(1.0)), ("exp_mode", mode), ("esc8_centred", out8), ("esc8_best", 1.0 - best[7]), ("esc16_best", 1.0 - best[15]),
          ("mlow1z", float(((man & 1) == 0).double().mean())), ("mlow2z", float(((man & 3) == 0).double().mean())),
          ("mlow3z", float(((man & 7) == 0).double().mean())), ("zero", float(((b & 0x7fff) == 0).double().mean())),
          ("subnormal", float(((e == 0) & (man != 0)).double().mean())),
          ("H16", ent(hv)), ("Hexp", ent(he)), ("Hman", ent(hm))]
    _log("STATS %s " % name + " ".join("%s=%s" % (k, v if isinstance(v, int) else "%.4f" % v) for k, v in kv))


def _stats(state):
    t0 = time.time()
    try:
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            _stats_line(layer, state["raw"][layer])
        _stats_line("router_hi", state["router_hi"])
        _stats_line("router_lo", state["router_lo"])
    except Exception as error:
        _log("STATS failed: %s" % str(error).splitlines()[-1][:160])
    _log("STATS done in %.2fs" % (time.time() - t0))


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


def _validate(fn, w, m, n, f32, gen, exact_w=None):
    """One-hot rows: every output must equal a weight column exactly (the judge's probe rows, 1e-3, but a lossless
    pack must be bit-exact). Random rows: relative L2 against FP64 within TOL. Returns (ok, message).
    exact_w (router only): the one-hot reference is hi.float() + lo.float(), which a hi/lo pair reproduces exactly."""
    K = w.shape[1]
    dt = torch.float32 if f32 else torch.bfloat16
    out = torch.empty((m, n), dtype=dt, device=w.device)
    ew = w if exact_w is None else exact_w
    exact = ew[:n].t().to(dt) if f32 else ew[:n].t()  # [K, n]
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
    state["router_hl"] = hi.float() + lo.float()  # exact in fp32; the router paths' one-hot reference
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
    _stats(state)  # weight statistics for the next codec decision (before the probes and the BAKE table)
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
                    ok, msg = _validate(fn, w, m, n, f32, gen, exact_w=state["router_hl"] if f32 else None)
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
