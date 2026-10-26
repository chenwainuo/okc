# okc: problem=muse_bf16_proj
"""tensorcore: TMA + mbarrier + wgmma warp-specialised GEMM over a 12-bit lossless packing of W.

out[m][n] = sum_k x[m][k] W[n][k] is computed as D[n][m] = W[n][:] . x[m][:], i.e. W is the wgmma A operand (fed from
registers, decoded on the fly from the packed stream) and x is the B operand (K-major, TMA-loaded into 128B-swizzled
smem).  Each weight is stored in 12 bits: 1 sign, 7 mantissa, 4-bit exponent code relative to a per-layer 16-code
window; the ~1e-4 of weights outside the window are stored as a placeholder and corrected exactly by a sparse fix-up
list.  Persistent one-wave grid, stream-K over (128-row n-block, 64-wide k-chunk) units, fp32 split-K partials
reduced with red.global.add.v4 into a workspace; the last CTA to finish an n-block converts it to bf16 and re-zeroes
the workspace (so the workspace is always zero between calls)."""

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


BN, BK = 128, 64
WBYTES = BN * BK * 12 // 8  # 12288 bytes per unit
GRID = 132
CLUSTER = int(os.environ.get("TC_CLUSTER", "1"))
CLUSTER_MAX_MP = 128  # no smem room for a 128 KB slot at MP = 256; CLUSTER = CTAs per cluster for the DSMEM pre-reduction (1 = off)
PDL = int(os.environ.get("TC_PDL", "1"))  # programmatic dependent launch of the convert kernel
# largest m served by the packed-W kernel per layer; above it cuBLAS (torch) is faster on the H100 (its split-K tail of
# fp32 atomics costs 2-5 us at m = 128/256, see DESIGN.md)
KERNEL_MAX_M = {"qkv_gate": 64, "o": 64, "pooler": 64, "unpooler": 64, "router": 256}
KERNEL_MAX_M = {k: int(os.environ.get("TC_MAX_M", v)) for k, v in KERNEL_MAX_M.items()}
MODE = 0  # ladder: 0 full, 3 no flush, 4 red.add flush only, 5 escapes only (no red.add), 6 no x tile (no flush), 7 no wgmma (no flush), 8 no flush + convert kernel
DEDICATED_MAX_MP = 32  # MP <= this: dedicated producer warp (288 threads); above: thread 0 issues loads (256 threads)
MPS = [8, 16, 32, 64, 128, 256]
STAGES = {8: 11, 16: 10, 32: 9, 64: 8, 128: 4, 256: 4}      # cluster of 2 reserves a 256*MP*2-byte slot (MP<=128)
DSTAGES = {8: 8, 16: 7, 32: 6, 64: 5, 128: 3, 256: 3}  # router (hi+lo: 24 KB of W per stage)
ROUTER_GRID = {8: 128, 16: 128, 32: 128, 64: 128, 128: 128, 256: 128}

# ----------------------------------------------------------------------------------------------------------------------
# CUDA source
# ----------------------------------------------------------------------------------------------------------------------

def _wgmma_asm(n: int) -> str:
    """wgmma.mma_async m64n{n}k16 f32 += bf16(A regs) * bf16(B smem desc), scale-d from a predicate."""
    nacc = n // 2
    d = ", ".join(f"%{i}" for i in range(nacc))
    a = ", ".join(f"%{nacc + i}" for i in range(4))
    desc = f"%{nacc + 4}"
    sc = f"%{nacc + 5}"
    body = (f"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sc}, 0;\\n"
            f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, {{{a}}}, {desc}, p, 1, 1, 0;\\n}}")
    outs = ", ".join(f'"+f"(acc[{i}])' for i in range(nacc))
    return (f'__device__ __forceinline__ void wgmma_n{n}(float* acc, const uint32_t* a, uint64_t desc, int scale_d) {{\n'
            f'  asm volatile("{body}" : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n}}\n')


CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace tensorcore {

constexpr int BN = 128, BK = 64, WBYTES = 12288, ECAP = 64;   // escape capacity per CTA (the last 4 em slots hold offsets)
template <int MP> constexpr int nthreads() { return MP <= DEDICATED_MAX_MP ? 288 : 256; }  // dedicated producer warp for small MP

struct Params {
  const uint8_t* __restrict__ packed;
  float* __restrict__ ws;                 // [MP][N] fp32, zero on entry and on exit
  void* __restrict__ out;                 // [m][N] bf16 (fp32 when DUAL)
  const __nv_bfloat16* __restrict__ x;    // [m][K]
  int* __restrict__ counters;             // [NB]
  const int* __restrict__ expected;       // [NB]
  const int* __restrict__ esc_off;        // [U+1]
  const int* __restrict__ esc_idx;        // unit<<13 | n_local<<6 | k_local
  const float* __restrict__ esc_delta;
  int N, K, m, NB, KC, U, G, cl;   // cl: cluster size (1 or 2) for the DSMEM pre-reduction of split-K partials
  uint32_t scale2, scale2b;   // bf16x2 exponent scales of the (hi, lo) streams
  int pdl;    // programmatic dependent launch of the convert kernel
  int mode;   // 0 full, 3 no flush, 4 flush red.add only, 5 +escapes (no counters/convert)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar);
  asm volatile("{\n.reg .pred p;\nLAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@p bra DONE;\nbra LAB_WAIT;\nDONE:\n}" :: "r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint32_t mapa_cluster(uint32_t addr, int rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
__device__ __forceinline__ void st_cluster_v4(uint32_t raddr, float a, float b, float c, float d) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(raddr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ uint32_t ld_cluster_acquire(uint32_t raddr) {
  uint32_t v; asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(raddr) : "memory"); return v;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t raddr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(raddr) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar);
  asm volatile("{\n.reg .pred p;\nLAB_WAIT:\n"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
               "@p bra DONE;\nbra LAB_WAIT;\nDONE:\n}" :: "r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {
  uint64_t d = (uint64_t)((smem_addr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;             // LBO (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;   // SBO: 8 rows x 128B
  d |= (uint64_t)1 << 62;             // 128B swizzle
  return d;
}

WGMMA_FUNCS

template <int MP> __device__ __forceinline__ void wgmma_mp(float* acc, const uint32_t* a, uint64_t desc, int sc) {
  if constexpr (MP == 8) wgmma_n8(acc, a, desc, sc);
  else if constexpr (MP == 16) wgmma_n16(acc, a, desc, sc);
  else if constexpr (MP == 32) wgmma_n32(acc, a, desc, sc);
  else if constexpr (MP == 64) wgmma_n64(acc, a, desc, sc);
  else if constexpr (MP == 128) wgmma_n128(acc, a, desc, sc);
  else wgmma_n256(acc, a, desc, sc);
}

// 12-bit -> bf16x2 decode of one register pair.  B: 4 bytes (sign<<7 | mant7) of weights 2p,2p+1 at bytes (0,1) or
// (2,3); X: the nibble word shifted so the pair's exponent codes sit at bits 7-10 / 23-26.
__device__ __forceinline__ uint32_t decode_pair(uint32_t B, uint32_t sel, uint32_t X, uint32_t scale2) {
  uint32_t pre = __byte_perm(B, 0u, sel);                     // {signrep(b), b} per 16-bit lane
  uint32_t x2 = (X & 0x07800780u) | 0x30003000u;              // exponent = e4 | 96
  uint32_t lane = (pre & 0x807F807Fu) | (x2 & 0x7F807F80u);
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&lane);
  __nv_bfloat162 s = *reinterpret_cast<__nv_bfloat162*>(&scale2);
  v = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void decode_step2(uint32_t B0, uint32_t B1, uint32_t Nw, uint32_t scale2, uint32_t* a) {
  a[0] = decode_pair(B0, 0x9180u, Nw << 7, scale2);
  a[1] = decode_pair(B0, 0xB3A2u, Nw << 3, scale2);
  a[2] = decode_pair(B1, 0x9180u, Nw >> 1, scale2);
  a[3] = decode_pair(B1, 0xB3A2u, Nw >> 5, scale2);
}

template <int MP, int STAGES, bool DUAL>
__global__ void __launch_bounds__(nthreads<MP>(), 1) gemm_kernel(const __grid_constant__ CUtensorMap xmap, Params p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int XBYTES = MP * 128;
  constexpr int WB = DUAL ? 2 * WBYTES : WBYTES;   // packed bytes per unit
  constexpr int NACC = MP / 2;
  constexpr int GS = (MP >= 256) ? 2 : 4;   // k16 steps per wgmma group
  constexpr int NG = 4 / GS;                // groups per unit
  constexpr bool DB = (MP < 256);           // double-buffered A registers (MP=256: single, the other WG covers)
  constexpr bool DEDICATED = (MP <= DEDICATED_MAX_MP);  // dedicated producer warp (288 threads); else thread 0 of WG0 issues loads
                                            // (256 threads: ptxas caps registers at 168 above 256 threads)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = smem;                                    // STAGES x XBYTES (1024-aligned)
  uint8_t* wsm = smem + STAGES * XBYTES;                 // STAGES x WB
  uint64_t* full = (uint64_t*)(wsm + STAGES * WB);      // STAGES
  uint64_t* empty = full + STAGES;                       // STAGES
  float* scratch = (float*)(empty + STAGES);            // 2 warpgroups x 8 tokens x 68 floats
  int* em = (int*)(scratch + 2 * 8 * 68);               // ECAP escape indices of the current n-block
  float* ed = (float*)(em + ECAP);                       // ECAP escape deltas
  __nv_bfloat16* xe = (__nv_bfloat16*)(ed + ECAP);       // ECAP x MP x values of the escapes' columns
  uint64_t* slots_bar = (uint64_t*)(xe + ECAP * MP);     // the cluster follower arrives here after writing its slot
  float* slot = (float*)(slots_bar + 2);                 // reserved: the follower's fragment-layout partial (256 x NACC floats)

  const int c = blockIdx.x;
  auto ustart = [&](int cc) { return (int)(((long long)cc * p.U) / p.G); };
  const int unit0 = ustart(c), unit1 = ustart(c + 1);
  const int crank = c % p.cl, cbase = c - crank;
  // cluster members holding a partial of n-block j: ranks [lead, lead+nmem)
  auto members = [&](int j, int& lead, int& nmem) {
    lead = -1; nmem = 0;
    for (int r = 0; r < p.cl; ++r) {
      if (ustart(cbase + r) < (j + 1) * p.KC && ustart(cbase + r + 1) > j * p.KC) { if (lead < 0) lead = r; ++nmem; }
    }
  };
  const int wgi = threadIdx.x >> 7;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 2); }
    int lead, nmem; members((unit1 - 1) / p.KC, lead, nmem);
    mbar_init(slots_bar, (lead == crank && nmem > 1) ? nmem - 1 : 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (p.cl > 1) cluster_sync_all();

  auto issue_load = [&](const int u, const int st) {
    const int kc = u % p.KC;
    if (p.mode == 6) { mbar_expect_tx(&full[st], WB); }
    else { mbar_expect_tx(&full[st], WB + XBYTES); tma_load_2d(xs + st * XBYTES, &xmap, kc * BK, 0, &full[st]); }
    tma_load_1d(wsm + st * WB, p.packed + (size_t)u * WB, WB, &full[st]);
  };
  if constexpr (DEDICATED) {
    if (wgi == 2) {
      // ---------------- producer warp (warp 8): one elected thread issues the TMA loads ----------------
      if (threadIdx.x == 256) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
        int stage = 0; uint32_t round = 0;
        for (int u = unit0; u < unit1; ++u) {
          mbar_wait(&empty[stage], (round & 1) ^ 1);
          issue_load(u, stage);
          if (++stage == STAGES) { stage = 0; ++round; }
        }
      }
      return;
    }
  }
  int pstage = 0; uint32_t pround = 0;   // integrated producer state (thread 0 only)
  if constexpr (!DEDICATED) {
    if (threadIdx.x == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
      for (int i = 0; i < STAGES && unit0 + i < unit1; ++i) {
        issue_load(unit0 + i, pstage);
        if (++pstage == STAGES) { pstage = 0; ++pround; }
      }
    }
  }

  // ---------------- consumer warpgroups ----------------
  const int tid = threadIdx.x;              // 0..255
  const int wg = tid >> 7;                  // 0,1: rows wg*64..
  const int t = tid & 127;
  const int warp = t >> 5, lane = t & 31;
  float acc[NACC];
  uint32_t a[DB ? 2 : 1][GS][4];
  int stage = 0; uint32_t round = 0;
  int prev_stage = -1;

  int* eoff = em + ECAP - 4;   // [e_base, e_mid, e_end] of the CTA's <= 2 n-blocks, loaded at prefetch time
  auto flush_block = [&](int j, int ulo, int uhi, int cnt) {
    // (1) escape fix-ups: the owning quad adds delta * x[tok][k] (prefetched into xe) into its accumulators
    if (cnt <= ECAP - 4) {
      const int eb = (ulo == unit0 ? eoff[0] : eoff[1]) - eoff[0], ee = (uhi == unit1 ? eoff[2] : eoff[1]) - eoff[0];
      for (int e = eb; e < ee; ++e) {
        const int nl = (em[e] >> 6) & 127;
        if ((nl >> 6) == wg && ((nl & 63) >> 4) == warp && (nl & 7) == (lane >> 2)) {
          const float d = ed[e];
          const __nv_bfloat162* xr = reinterpret_cast<const __nv_bfloat162*>(xe + e * MP) + (lane & 3);
          if (nl & 8) {
            #pragma unroll
            for (int g = 0; g < MP / 8; ++g) { float2 xv = __bfloat1622float2(xr[4 * g]); acc[4 * g + 2] += d * xv.x; acc[4 * g + 3] += d * xv.y; }
          } else {
            #pragma unroll
            for (int g = 0; g < MP / 8; ++g) { float2 xv = __bfloat1622float2(xr[4 * g]); acc[4 * g] += d * xv.x; acc[4 * g + 1] += d * xv.y; }
          }
        }
      }
    }
    if (p.mode == 5) return;
    // (1b) cluster pre-reduction: the follower pushes its fragment-layout tile into the leader's reserved slot (DSMEM)
    // right away (a shared n-block is the follower's first block and the leader's last one), the leader adds it at its
    // own flush after the follower's remote mbarrier arrival.
    if (p.cl > 1) {
      int lead, nmem; members(j, lead, nmem);
      if (nmem > 1) {
        if (crank != lead) {
          uint32_t ra = mapa_cluster(smem_u32(slot) + (uint32_t)((crank - lead - 1) * 256 * NACC * 4 + tid * 16), lead);
          #pragma unroll
          for (int q = 0; q < NACC / 4; ++q) { st_cluster_v4(ra, acc[4 * q], acc[4 * q + 1], acc[4 * q + 2], acc[4 * q + 3]); ra += 256 * 16; }
          asm volatile("bar.sync 1, 256;" ::: "memory");
          if (tid == 0) mbar_arrive_remote(mapa_cluster(smem_u32(slots_bar), lead));
          return;
        }
        mbar_wait_cluster(slots_bar, 0);
        for (int f = 0; f < nmem - 1; ++f) {
          uint32_t sl = smem_u32(slot) + f * 256 * NACC * 4 + tid * 16;
          #pragma unroll
          for (int q = 0; q < NACC / 4; ++q) {
            float4 v;
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(sl) : "memory");
            acc[4 * q] += v.x; acc[4 * q + 1] += v.y; acc[4 * q + 2] += v.z; acc[4 * q + 3] += v.w;
            sl += 256 * 16;
          }
        }
      }
    }
    // (2) partial accumulators -> ws via red.add.v4, through a per-warpgroup smem transpose (64 n x 8 tokens per step)
    float* sc = scratch + wg * (8 * 68);
    const int wn = warp * 16 + (lane >> 2), wtok = 2 * (lane & 3);
    const int rtok = t >> 4, rn4 = (t & 15) * 4;
    float* dst = p.ws + (size_t)rtok * p.N + j * BN + wg * 64 + rn4;
    const size_t dstep = (size_t)8 * p.N;
    #pragma unroll
    for (int g = 0; g < MP / 8; ++g) {
      sc[wtok * 68 + wn] = acc[4 * g];
      sc[(wtok + 1) * 68 + wn] = acc[4 * g + 1];
      sc[wtok * 68 + wn + 8] = acc[4 * g + 2];
      sc[(wtok + 1) * 68 + wn + 8] = acc[4 * g + 3];
      if (wg == 0) asm volatile("bar.sync 2, 128;" ::: "memory"); else asm volatile("bar.sync 3, 128;" ::: "memory");
      float4 val = *reinterpret_cast<const float4*>(sc + rtok * 68 + rn4);
      asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(dst), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
      dst += dstep;
      if (wg == 0) asm volatile("bar.sync 2, 128;" ::: "memory"); else asm volatile("bar.sync 3, 128;" ::: "memory");
    }
    // (3) overflow path: too many escapes for the smem buffer -> scalar atomics
    if (cnt > ECAP - 4) {
      const int e0 = p.esc_off[ulo], e1 = p.esc_off[uhi];
      for (int e = e0 + (tid >> 5); e < e1; e += 8) {
        const int idx = p.esc_idx[e];                 // unit<<13 | n_local<<6 | k_local
        const float delta = p.esc_delta[e];
        const int uu = idx >> 13;
        const int n = j * BN + ((idx >> 6) & 127);
        const int k = (uu % p.KC) * BK + (idx & 63);
        for (int tok = lane; tok < p.m; tok += 32) {
          float xv = __bfloat162float(p.x[(size_t)tok * p.K + k]);
          atomicAdd(p.ws + (size_t)tok * p.N + n, delta * xv);
        }
      }
    }
  };
  // escape prefetch for the CTA's whole unit range, once at kernel start (overlaps the pipeline fill): metadata, then
  // the bf16 x columns of the escapes (8 independent loads in flight per thread)
  auto prefetch_escapes = [&](int ulo, int uhi) -> int {
    const int e0 = p.esc_off[ulo], cnt = p.esc_off[uhi] - e0;
    if (tid == 0) {
      const int jmid = ((unit1 - 1) / p.KC) * p.KC;   // first unit of the CTA's last n-block
      eoff[0] = e0; eoff[1] = p.esc_off[jmid > ulo ? jmid : ulo]; eoff[2] = e0 + cnt;
    }
    if (cnt <= ECAP - 4) {
      for (int e = tid; e < cnt; e += 256) { em[e] = p.esc_idx[e0 + e]; ed[e] = p.esc_delta[e0 + e]; }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      for (int i0 = tid; i0 < cnt * MP; i0 += 256 * 8) {
        __nv_bfloat16 v[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
          const int i = i0 + b * 256;
          const int e = i / MP, tok = i % MP;
          v[b] = __float2bfloat16(0.f);
          if (i < cnt * MP && tok < p.m) {
            const int idx = em[e];
            const int k = ((idx >> 13) % p.KC) * BK + (idx & 63);
            v[b] = p.x[(size_t)tok * p.K + k];
          }
        }
        #pragma unroll
        for (int b = 0; b < 8; ++b) { const int i = i0 + b * 256; if (i < cnt * MP) xe[i] = v[b]; }
      }
    }
    return cnt;
  };
  const int cnt_all = (p.mode == 0 || p.mode == 5) ? prefetch_escapes(unit0, unit1) : 0;
  // one group of GS k16-steps: decode into ab, issue, then wait so that <=1 group stays in flight (double-buffered A)
  // or 0 (single-buffered: the other consumer warpgroup keeps the tensor pipe busy meanwhile).
  auto group_body = [&](const int g, const uint8_t* wp, const uint32_t xaddr, const uint32_t scale2, uint32_t (*ab)[4]) {
    uint32_t vb[2 * GS], vn[GS];
    {
      // byte words of this group's steps (16B per 2 steps) and its nibble words
      if constexpr (GS == 4) {
        uint4 q0 = *reinterpret_cast<const uint4*>(wp); uint4 q1 = *reinterpret_cast<const uint4*>(wp + 4096);
        vb[0] = q0.x; vb[1] = q0.y; vb[2] = q0.z; vb[3] = q0.w; vb[4] = q1.x; vb[5] = q1.y; vb[6] = q1.z; vb[7] = q1.w;
        uint4 q2 = *reinterpret_cast<const uint4*>(wp + 8192); vn[0] = q2.x; vn[1] = q2.y; vn[2] = q2.z; vn[3] = q2.w;
      } else {
        uint4 q0 = *reinterpret_cast<const uint4*>(wp + g * 4096);
        vb[0] = q0.x; vb[1] = q0.y; vb[2] = q0.z; vb[3] = q0.w;
        uint2 q2 = *reinterpret_cast<const uint2*>(wp + 8192 + g * 8); vn[0] = q2.x; vn[1] = q2.y;
      }
    }
    if constexpr (!DB) wgmma_wait<0>();
    #pragma unroll
    for (int s = 0; s < GS; ++s) decode_step2(vb[2 * s], vb[2 * s + 1], vn[s], scale2, ab[s]);
    if (p.mode == 7) { scratch[tid] = __uint_as_float(ab[0][0] ^ ab[GS - 1][3]); return; }
    wgmma_fence();
    #pragma unroll
    for (int s = 0; s < GS; ++s) wgmma_mp<MP>(acc, ab[s], make_desc_b(xaddr + (g * GS + s) * 32), 1);
    wgmma_commit();
    if constexpr (DB) wgmma_wait<1>();
  };
  auto unit_body = [&](const int u, const int bufsel) {
    mbar_wait(&full[stage], round & 1);
    const uint8_t* wp = wsm + stage * WB + tid * 16;
    const uint32_t xaddr = smem_u32(xs + stage * XBYTES);
    if constexpr (DUAL) {
      // hi stream into a[0], lo stream (same B, same accumulators) into a[1]
      #pragma unroll
      for (int g = 0; g < NG; ++g) {
        group_body(g, wp, xaddr, p.scale2, a[0]);
        group_body(g, wp + WBYTES, xaddr, p.scale2b, a[DB ? 1 : 0]);
      }
    } else if constexpr (NG == 1) {
      if (DB && bufsel) group_body(0, wp, xaddr, p.scale2, a[DB ? 1 : 0]); else group_body(0, wp, xaddr, p.scale2, a[0]);
    } else {
      #pragma unroll
      for (int g = 0; g < NG; ++g) {
        if (DB && (g & 1)) group_body(g, wp, xaddr, p.scale2, a[DB ? 1 : 0]); else group_body(g, wp, xaddr, p.scale2, a[0]);
      }
    }
    // the previous unit's wgmmas are complete (<=1 group pending), so its stage can be refilled
    if (prev_stage >= 0 && t == 0) mbar_arrive(&empty[prev_stage]);
    prev_stage = stage;
    if (++stage == STAGES) { stage = 0; ++round; }
    if constexpr (!DEDICATED) {
      const int un = u + STAGES - 1;
      if (tid == 0 && un < unit1 && u > unit0) {
        mbar_wait(&empty[pstage], (pround & 1) ^ 1);
        issue_load(un, pstage);
        if (++pstage == STAGES) { pstage = 0; ++pround; }
      }
    }
  };
  for (int j = unit0 / p.KC; j * p.KC < unit1; ++j) {
    const int ulo = max(unit0, j * p.KC), uhi = min(unit1, (j + 1) * p.KC);
    #pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
    if constexpr (NG == 1) {
      for (int u = ulo; u < uhi; u += 2) {
        unit_body(u, 0);
        if (u + 1 < uhi) unit_body(u + 1, 1);
      }
    } else {
      for (int u = ulo; u < uhi; ++u) unit_body(u, 0);
    }
    wgmma_wait<0>();
    if (prev_stage >= 0 && t == 0) mbar_arrive(&empty[prev_stage]);
    prev_stage = -1;
    if (p.mode == 0 || p.mode == 5) asm volatile("bar.sync 1, 256;" ::: "memory");   // xe visible to all
    if (p.mode == 0 || (p.mode >= 4 && p.mode <= 5)) flush_block(j, ulo, uhi, cnt_all);
  }
#endif  // __CUDA_ARCH_FEAT_SM90_ALL
}

// ws (fp32 [m][N], contiguous) -> out (bf16, or fp32 copy), and re-zero ws
template <bool DUAL>
__global__ void convert_kernel(float* __restrict__ ws, void* __restrict__ out, int total4) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total4) return;
  float4 val = __ldcg(reinterpret_cast<const float4*>(ws) + i);
  if constexpr (DUAL) {
    reinterpret_cast<float4*>(out)[i] = val;
  } else {
    __nv_bfloat162 lo = __floats2bfloat162_rn(val.x, val.y);
    __nv_bfloat162 hi = __floats2bfloat162_rn(val.z, val.w);
    uint2 o; o.x = *reinterpret_cast<uint32_t*>(&lo); o.y = *reinterpret_cast<uint32_t*>(&hi);
    reinterpret_cast<uint2*>(out)[i] = o;
  }
  __stcg(reinterpret_cast<float4*>(ws) + i, make_float4(0.f, 0.f, 0.f, 0.f));
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn get_encode() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q);
    TORCH_CHECK(ptr != nullptr, "cuTensorMapEncodeTiled not found");
    fn = (EncodeTiledFn)ptr;
  }
  return fn;
}

template <int MP, int STAGES, bool DUAL>
static void launch_mp(const CUtensorMap& map, const Params& p, size_t smem, cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gemm_kernel<MP, STAGES, DUAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  if (p.cl > 1) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.G); cfg.blockDim = dim3(nthreads<MP>()); cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = p.cl; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, gemm_kernel<MP, STAGES, DUAL>, map, p);
    TORCH_CHECK(e == cudaSuccess, "cluster launch failed: ", cudaGetErrorString(e));
  } else {
    gemm_kernel<MP, STAGES, DUAL><<<p.G, nthreads<MP>(), smem, stream>>>(map, p);
  }
  if (p.mode == 0 || p.mode == 8) {
    const int total4 = p.m * p.N / 4;
    if (p.pdl) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3((total4 + 255) / 256); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr; cfg.numAttrs = 1;
      cudaError_t e = cudaLaunchKernelEx(&cfg, convert_kernel<DUAL>, p.ws, p.out, total4);
      TORCH_CHECK(e == cudaSuccess, "PDL launch failed: ", cudaGetErrorString(e));
    } else {
      convert_kernel<DUAL><<<(total4 + 255) / 256, 256, 0, stream>>>(p.ws, p.out, total4);
    }
  }
}

void gemm(int MP, int stages, torch::Tensor packed, torch::Tensor x, torch::Tensor ws, torch::Tensor out,
          torch::Tensor counters, torch::Tensor expected, torch::Tensor esc_off, torch::Tensor esc_idx,
          torch::Tensor esc_delta, int N, int K, int m, int NB, int KC, int U, int G, int cl, int64_t scale2, int64_t scale2b,
          int dual, int pdl, int mode) {
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)m};
  cuuint64_t gstride[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {(cuuint32_t)BK, (cuuint32_t)MP};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstride, box, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  Params p;
  p.packed = packed.data_ptr<uint8_t>();
  p.ws = ws.data_ptr<float>();
  p.out = out.data_ptr();
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.counters = counters.data_ptr<int>();
  p.expected = expected.data_ptr<int>();
  p.esc_off = esc_off.data_ptr<int>();
  p.esc_idx = esc_idx.data_ptr<int>();
  p.esc_delta = esc_delta.data_ptr<float>();
  p.N = N; p.K = K; p.m = m; p.NB = NB; p.KC = KC; p.U = U; p.G = G; p.cl = cl;
  p.scale2 = (uint32_t)scale2;
  p.scale2b = (uint32_t)scale2b;
  p.pdl = pdl;
  p.mode = mode;
  size_t smem = 1024 + (size_t)stages * (MP * 128 + (dual ? 2 : 1) * WBYTES) + 2 * stages * 8 + 2 * 8 * 68 * 4 + ECAP * 8 + (size_t)ECAP * MP * 2 + 16;
  smem += (size_t)(cl - 1) * 256 * (MP / 2) * 4;
  TORCH_CHECK(smem <= 232448, "smem too large: ", smem);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (!dual) {
    switch (MP) {
      case 8: launch_mp<8, STAGES_8, false>(map, p, smem, stream); break;
      case 16: launch_mp<16, STAGES_16, false>(map, p, smem, stream); break;
      case 32: launch_mp<32, STAGES_32, false>(map, p, smem, stream); break;
      case 64: launch_mp<64, STAGES_64, false>(map, p, smem, stream); break;
      case 128: launch_mp<128, STAGES_128, false>(map, p, smem, stream); break;
      case 256: launch_mp<256, STAGES_256, false>(map, p, smem, stream); break;
      default: TORCH_CHECK(false, "bad MP");
    }
  } else {
    switch (MP) {
      case 8: launch_mp<8, DSTAGES_8, true>(map, p, smem, stream); break;
      case 16: launch_mp<16, DSTAGES_16, true>(map, p, smem, stream); break;
      case 32: launch_mp<32, DSTAGES_32, true>(map, p, smem, stream); break;
      case 64: launch_mp<64, DSTAGES_64, true>(map, p, smem, stream); break;
      case 128: launch_mp<128, DSTAGES_128, true>(map, p, smem, stream); break;
      case 256: launch_mp<256, DSTAGES_256, true>(map, p, smem, stream); break;
      default: TORCH_CHECK(false, "bad MP");
    }
  }
}

}  // namespace tensorcore

void tensorcore_gemm(int MP, int stages, torch::Tensor packed, torch::Tensor x, torch::Tensor ws, torch::Tensor out,
                     torch::Tensor counters, torch::Tensor expected, torch::Tensor esc_off, torch::Tensor esc_idx,
                     torch::Tensor esc_delta, int N, int K, int m, int NB, int KC, int U, int G, int cl, int64_t scale2, int64_t scale2b,
                     int dual, int pdl, int mode) {
  tensorcore::gemm(MP, stages, packed, x, ws, out, counters, expected, esc_off, esc_idx, esc_delta, N, K, m, NB, KC, U, G, cl, scale2, scale2b, dual, pdl, mode);
}
"""

CPP_SRC = """
void tensorcore_gemm(int MP, int stages, torch::Tensor packed, torch::Tensor x, torch::Tensor ws, torch::Tensor out,
                     torch::Tensor counters, torch::Tensor expected, torch::Tensor esc_off, torch::Tensor esc_idx,
                     torch::Tensor esc_delta, int N, int K, int m, int NB, int KC, int U, int G, int cl, int64_t scale2, int64_t scale2b,
                     int dual, int pdl, int mode);
"""

_ext = None


def _build():
    global _ext
    if _ext is None:
        src = CUDA_SRC.replace("WGMMA_FUNCS", "\n".join(_wgmma_asm(n) for n in MPS))
        for mp, st in DSTAGES.items():
            src = src.replace(f"DSTAGES_{mp},", f"{st},")
        for mp, st in STAGES.items():
            src = src.replace(f"STAGES_{mp},", f"{st},")
        src = src.replace("DEDICATED_MAX_MP", str(DEDICATED_MAX_MP))
        _ext = load_inline(
            name="tensorcore_gemm_v23",
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["tensorcore_gemm"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"]
            + (["-Xptxas", "-v"] if os.environ.get("TC_PTXAS_V") else []),
            verbose=bool(os.environ.get("TC_PTXAS_V")),
        )
    return _ext


# ----------------------------------------------------------------------------------------------------------------------
# prepare: lossless 12-bit packing in the wgmma A-fragment order
# ----------------------------------------------------------------------------------------------------------------------

def _fragment_index(N: int, K: int, device) -> torch.Tensor:
    """Flat index n*K+k of every weight in stream order [unit][thread][step][i]."""
    NB, KC = N // BN, K // BK
    j = torch.arange(NB, device=device).view(NB, 1, 1, 1, 1)
    c = torch.arange(KC, device=device).view(1, KC, 1, 1, 1)
    tid = torch.arange(256, device=device).view(1, 1, 256, 1, 1)
    s = torch.arange(4, device=device).view(1, 1, 1, 4, 1)
    i = torch.arange(8, device=device).view(1, 1, 1, 1, 8)
    t = tid % 128
    n = j * BN + (tid // 128) * 64 + (t // 32) * 16 + ((t % 32) // 4) + 8 * ((i // 2) % 2)
    k = c * BK + s * 16 + (t % 4) * 2 + (i % 2) + 8 * (i // 4)
    return (n * K + k).reshape(-1)


def pack_layer(W: torch.Tensor, G: int = GRID) -> dict:
    """W bf16 [N][K] -> packed stream + escape list + partition tables."""
    assert W.dtype == torch.bfloat16
    N, K = W.shape
    dev = W.device
    NB, KC = N // BN, K // BK
    bits = W.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).to(torch.int64)
    win = torch.nn.functional.avg_pool1d(hist.view(1, 1, -1).float(), 16, stride=1).view(-1) * 16
    base = int(win.argmax())
    inw = (e >= base) & (e < base + 16)
    e4 = torch.where(inw, e - base, torch.zeros_like(e))
    byte = torch.where(inw, ((bits >> 8) & 0x80) | (bits & 0x7F), torch.zeros_like(e)).to(torch.uint8)
    nib = e4.to(torch.uint8)
    placeholder = 2.0 ** (base - 127)
    idx = _fragment_index(N, K, dev)
    bg = byte.flatten()[idx].view(NB * KC, 256, 4, 8)
    ng = nib.flatten()[idx].view(NB * KC, 256, 4, 8).to(torch.int32)
    pos = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=dev, dtype=torch.int32)
    nw = (ng << pos).sum(-1).to(torch.int32)  # [U, 256, 4]
    v01 = bg.reshape(NB * KC, 256, 32)  # bytes of steps 0..3 (v0 = 0..15, v1 = 16..31)
    v2 = nw.contiguous().view(torch.uint8).view(NB * KC, 256, 16)
    stream = torch.empty(NB * KC, 3, 256, 16, dtype=torch.uint8, device=dev)
    stream[:, 0] = v01[:, :, :16]
    stream[:, 1] = v01[:, :, 16:]
    stream[:, 2] = v2
    packed = stream.reshape(-1).contiguous()
    # escapes
    esc = (~inw).nonzero()
    en, ek = esc[:, 0], esc[:, 1]
    unit = (en // BN) * KC + ek // BK
    order = torch.argsort(unit)
    en, ek, unit = en[order], ek[order], unit[order]
    delta = (W.float()[en, ek] - placeholder).contiguous()
    esc_idx = ((unit << 13) | ((en % BN) << 6) | (ek % BK)).to(torch.int32).contiguous()
    cnt = torch.bincount(unit, minlength=NB * KC)
    esc_off = torch.zeros(NB * KC + 1, dtype=torch.int64, device=dev)
    esc_off[1:] = torch.cumsum(cnt, 0)
    esc_off = esc_off.to(torch.int32).contiguous()
    scale = 2.0 ** (base - 96)
    sb = torch.tensor([scale], dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF
    scale2 = sb | (sb << 16)
    return {"packed": packed, "esc_off": esc_off, "esc_idx": esc_idx, "esc_delta": delta, "scale2": scale2,
            "base": base, "N": N, "K": K, "NB": NB, "KC": KC, "n_esc": int(esc.shape[0])}


def pack_router(W: torch.Tensor) -> dict:
    """fp32 W -> bf16 hi + lo pair, each 12-bit packed, interleaved per unit [hi | lo]; merged escape list."""
    hi = W.to(torch.bfloat16)
    lo = (W - hi.float()).to(torch.bfloat16)
    a, b = pack_layer(hi), pack_layer(lo)
    U = a["NB"] * a["KC"]
    packed = torch.stack([a["packed"].view(U, WBYTES), b["packed"].view(U, WBYTES)], 1).reshape(-1).contiguous()
    idx = torch.cat([a["esc_idx"], b["esc_idx"]])
    delta = torch.cat([a["esc_delta"], b["esc_delta"]])
    unit = idx >> 13
    order = torch.argsort(unit, stable=True)
    idx, delta, unit = idx[order].contiguous(), delta[order].contiguous(), unit[order]
    cnt = torch.bincount(unit, minlength=U)
    off = torch.zeros(U + 1, dtype=torch.int64, device=W.device)
    off[1:] = torch.cumsum(cnt, 0)
    return {"packed": packed, "esc_off": off.to(torch.int32).contiguous(), "esc_idx": idx, "esc_delta": delta,
            "scale2": a["scale2"], "scale2b": b["scale2"], "base": (a["base"], b["base"]), "N": a["N"], "K": a["K"],
            "NB": a["NB"], "KC": a["KC"], "n_esc": a["n_esc"] + b["n_esc"]}


def _partition_tables(NB: int, KC: int, G: int, device) -> tuple:
    """Stream-K partition: CTA c handles units [c*U//G, (c+1)*U//G); expected[j] = #CTAs touching n-block j."""
    U = NB * KC
    G = min(G, U)
    starts = [(c * U) // G for c in range(G + 1)]
    expected = []
    for j in range(NB):
        lo, hi = j * KC, (j + 1) * KC
        expected.append(sum(1 for c in range(G) if starts[c] < hi and starts[c + 1] > lo))
    return torch.tensor(expected, dtype=torch.int32, device=device), U, G


def prepare(weights: Weights) -> dict:
    _build()
    state = {"w": {name: w.contiguous() for name, w in weights.items()}}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["layers"] = {}
    for name in ["qkv_gate", "o", "pooler", "unpooler", "router"]:
        W = weights[name]
        info = pack_router(W) if name == "router" else pack_layer(W)
        N, K = W.shape
        info["ws"] = torch.zeros(256, N, dtype=torch.float32, device=W.device)
        info["counters"] = torch.zeros(info["NB"], dtype=torch.int32, device=W.device)
        info["parts"] = {}
        for n_rows in ([N, 1024] if name == "unpooler" else [N]):
            nb = n_rows // BN
            for mp in MPS:
                g = ROUTER_GRID[mp] if name == "router" else GRID
                exp_t, U, G = _partition_tables(nb, info["KC"], g, W.device)
                info["parts"][(n_rows, mp)] = (exp_t, U, G, nb)
        state["layers"][name] = info
        print(f"PACK {name}: base={info['base']} escapes={info['n_esc']} ({info['n_esc'] / (N * K):.2e})", flush=True)
    torch.cuda.synchronize()
    return state


def _mp_for(m: int) -> int:
    for mp in MPS:
        if m <= mp:
            return mp
    return 0


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n, m = cfg["layer"], cfg["n"], cfg["m"]
    dual = layer == "router"
    mp = _mp_for(m)
    if mp == 0 or m > KERNEL_MAX_M[layer]:
        if dual:
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc
        return x @ state["w"][layer][:n].T
    info = state["layers"][layer]
    exp_t, U, G, NB = info["parts"][(n, mp)]
    out = torch.empty(m, n, dtype=torch.float32 if dual else torch.bfloat16, device=x.device)
    _ext.tensorcore_gemm(mp, (DSTAGES if dual else STAGES)[mp], info["packed"], x, info["ws"], out, info["counters"], exp_t,
                         info["esc_off"], info["esc_idx"], info["esc_delta"], n, info["K"], m, NB, info["KC"], U, G,
                         CLUSTER if mp <= CLUSTER_MAX_MP else 1, info["scale2"], info.get("scale2b", 0), int(dual), PDL, MODE)
    return out
