# okc: problem=muse_bf16_proj
"""Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
pw11 ("gvx"): r6-c12 unchanged (every r6 path is still baked and kept as the per-shape baseline "r6"), plus gv geometries
with 64-144 rows per CTA (x fetched once per 128 rows instead of once per 16) and a cross-CTA K split of 2/4/8, reduced
either through cluster DSMEM (path suffix a, r6's own S > 1 epilogue) or through an fp32 workspace + last-arriving CTA
(suffix b, V bit 2, no cluster); suffix -s = x in a shared-memory slab instead of through L1; the router gets the
12-bit codec on both hi and lo. Each new path is validated (one-hot x8 exact, random rows) and timed against the r6
choice in a second bake (BAKE / PICK lines); kernel_fn takes the pw11 pick when it exists, else r6's dispatch.
pw12: the same gv kernel with W streamed through a private per-warp cp.async.bulk smem ring (V bit 3, paths gb-*);
decode, mma, escapes, reduction and epilogue unchanged. PROBE gvbulk lines compare it with r6 on pooler m=1.
pw18 (= pw17 with codec E v3): r6 unchanged (its 12-bit paths, bake and dispatch are SASS-identical) plus codec E paths
(gv FMT 4, "ge-*"). The code length is fixed by a code's first 3 bits (LOP3 + PRMT on an 8-byte table), so the per-lane
bit-position chain has no memory access. The exponent pair value comes from a 256-entry smem LUT (single or lane-replicated)
off the chain. There are 1, 2 or 4 independent substreams per lane, and refill words come through a conflict-free
[substream][slot][lane] 4 B cp.async ring and are prefetched. 11.07 / 11.17 / 11.32 bits/weight (K 1 / 2 / 4) against 12
(proj_rsi/tools/CODEC_E_LAYOUT.md v3). Each pack is proven bit-exact by the reference unpacker before any E kernel runs. Each
path is validated like every second-bake path and timed against r6's pick (BAKE / PICK). PROBE ecod lines time the best E path
against r6 on pooler m=1 and m=16, alternated.
pw29: pw23 plus wgmma router paths at m = 128 / 256 (rwk, its own sm_90a extension), baked against pw23's rs / rp / rv
paths and r6's cuBLASLt path.
pw31 (scored merge): pw27h (P1: flag-zeroed split-K router m = 8/16/32 and unpooler m <= 8 on r6's gv, rr6 at router
m = 1) + pw29h (wgmma split-K router m = 128 / 256 with self-zeroing); the two regions are disjoint (3-way merge on pw23).
pw34 = pw33 + rs128 (audits/m128-plan.md): pw23 unchanged plus a swap-AB register-A split-K wgmma kernel for m exactly 128 / 256 on the
pooler, qkv_gate, unpooler and o layers: 128 W rows x all tokens per CTA, two math warpgroups decode the lossless 12-bit
pack (pw10 _pack_nat + _c12, re-blocked to 12 KB per 128-row k64 stage) into wgmma A registers, x is the smem B operand
(2D TMA, SW128), one full barrier per ring slot shared by a W bulk lane and an x TMA lane, K split over KS CTAs on a
one-wave grid, partials summed by cp.reduce.async.bulk.tensor bf16 add into out, which the kernel zeroes itself (each
CTA its slice, release counter, polled off the critical path; no memset node, no cluster). Built in a background thread
during r6's phase; BAKE / PICK lines time it against r6's own path per (layer, m); any failure leaves r6's choices."""
from typing import TypedDict
import time
import torch
_T0 = time.time()
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


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WPC
#define WPC 16
#endif
#ifndef V7
#define V7 0
#endif
#ifndef RSN
#define RSN 2
#endif
#ifndef CUN
#define CUN 2
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }

struct Q {
  const uint8_t* A; const uint8_t* A2; const __nv_bfloat16* x; void* out; float* ws; int* cnt;
  int M, N, K, MT, KC, TB, f32;
};

__device__ __forceinline__ void put(const Q& q, int tok, int row, float v) {
  if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
}

// Each warp streams an equal contiguous range of (tile, unit) work through its own RS-deep TMA ring; a unit is CU
// consecutive 64-wide K chunks (2 KB of A fragments each; the router's lo half rides along). Partial tiles are
// summed with fp32 red.add into ws; a PDL-launched finalize kernel converts ws to the output and clears it.
template <int NT, bool HL, int RS, int CU>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  constexpr int SB = (HL ? 2 : 1) * CU * 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * RS * SB;
  const uint32_t bars = (uint32_t)__cvta_generic_to_shared(smem) + WPC * RS * SB + wid * RS * 8;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");  // let the finalize kernel get scheduled early
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const int KU = q.KC / CU;
  const long long U = (long long)q.TB * q.MT * KU;
  const long long L0 = U * gw / W, Le = U * (gw + 1) / W;
  if (L0 >= Le) return;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const long long tile = pu / KU; const int cu = (int)(pu - tile * KU), mt = (int)(tile % q.MT);
      const size_t off = ((size_t)mt * q.KC + cu * CU) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d), "l"(q.A + off), "n"(CU * 2048), "r"(bar) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d + CU * 2048), "l"(q.A2 + off), "n"(CU * 2048), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4 * CU; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      if (HL) {
#pragma unroll
        for (int k = 0; k < 4 * CU; k++) cp16(d + CU * 2048 + k * 512 + lane * 16, q.A2 + off + k * 512 + lane * 16);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
    } else {
#if !TMA
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + slot * 8) : "memory");
#endif
    }
    pu++;
  };
#pragma unroll
  for (int i = 0; i < RS - 1; i++) issue(i);
  uint4 b0[NT][2], b1[NT][2];
  auto ldb = [&](long long cg, uint4 (&b)[NT][2]) {  // cg: global chunk index tile * KC + c
    const long long tile = cg / q.KC; const int c = (int)(cg - tile * q.KC), tb = (int)(tile / q.MT);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + c * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  const long long CE = Le * CU;  // chunk range end
  ldb(L0 * CU, b0);
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const long long tile = u / KU; const int u0 = (int)(u - tile * KU);
    const int u1 = (int)min((long long)KU, u0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int uu = u0; uu < u1; uu++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
#pragma unroll
      for (int i = 0; i < CU; i++) {
        const long long cg = u * CU + i;
        if (cg + 1 < CE) ldb(cg + 1, b1);
        const uint32_t sa = ring + slot * SB + i * 2048 + lane * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4 a = lds(sa + j * 512);
          uint4 h; if (HL) h = lds(sa + CU * 2048 + j * 512);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
            mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
            if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
#pragma unroll
        for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
      }
    }
    const int tb = (int)(tile / q.MT), mt = (int)(tile - (long long)tb * q.MT), row0 = mt * 16 + g, mrem = min(32, q.M - tb * 32);
    float* w = q.ws + (size_t)tb * 32 * q.N;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float f[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
      const int tk = n * 8 + 2 * t;
      if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[2]); }
      if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[3]); }
    }
#if V7
    {  // the last warp to add into this tile writes it out and clears ws (no finalize kernel)
      auto wof = [&](long long uu) { return (int)(((uu + 1) * W - 1) / U); };  // warp streaming unit uu
      const long long t0u = tile * KU, t1u = t0u + KU - 1;
      const int nc = wof(t1u) - wof(t0u) + 1;
      __syncwarp();
      int last = 0;
      if (lane == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(q.cnt + tile) : "memory"); last = v == nc - 1; }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int e = lane; e < 16 * mrem; e += 32) {
          const int tok = e >> 4, row = mt * 16 + (e & 15);
          if (row < q.N) { float* wp = w + (size_t)tok * q.N + row; put(q, tb * 32 + tok, row, __ldcg(wp)); *wp = 0.f; }
        }
        if (lane == 0) q.cnt[tile] = 0;
      }
    }
#endif
  }
}

// ws (fp32 sums, [M][N]) -> out; ws cleared for the next call. Launched as a programmatic dependent.
__global__ void __launch_bounds__(256) fin(Q q) {
#if TMA
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  const int n4 = q.M * q.N / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4* p = (float4*)q.ws + i;
    const float4 v = __ldcg(p);
    *p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q.f32) ((float4*)q.out)[i] = v;
    else { __nv_bfloat162 o[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)}; ((uint2*)q.out)[i] = *(uint2*)o; }
  }
}

static int g_grid = 0;
template <int NT, bool HL> static void go(const Q& q, cudaStream_t st) {
  constexpr int RS = HL ? 3 : RSN, CU = HL ? 1 : CUN;
  const size_t sm = (size_t)WPC * RS * (HL ? 2 : 1) * CU * 2048 + WPC * RS * 8;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(skinny<NT, HL, RS, CU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  skinny<NT, HL, RS, CU><<<g_grid, WPC * 32, sm, st>>>(q);
}
void run(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = (q.M + 31) / 32;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = q.M >= 32 ? 4 : (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go<1, true>(q, st); else if (nt <= 2) go<2, true>(q, st); else go<4, true>(q, st); }
  else { if (nt <= 1) go<1, false>(q, st); else if (nt <= 2) go<2, false>(q, st); else go<4, false>(q, st); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(std::min(g_grid * 4, (q.M * q.N / 4 + 255) / 256)); cfg.blockDim = dim3(256); cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  cfg.attrs = at; cfg.numAttrs = major >= 9 ? 1 : 0;
  if (!V7) cudaLaunchKernelEx(&cfg, fin, q);
}

// ---------------------------------------------------------------- v6: deep per-CTA TMA ring (stream over 16 KB stages)
// A stage is 8 consecutive 2 KB units (one 16-row tile, 8 K chunks); CTA c streams stages [ST c / G, ST (c + 1) / G)
// through an NS-deep ring filled by a producer warp, so most of a CTA's bytes are in flight at once. Consumer warp w
// takes unit w of every stage; at a tile's end the 8 partials meet in smem, and a tile shared by CTAs is summed in
// ws with fp32 atomics, the last arriver writing the output.
#ifndef NSG
#define NSG 12
#endif
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
template <int NT, bool HL>
__global__ void __launch_bounds__(288, 1) gemv(Q q) {
  constexpr int SB = (HL ? 2 : 1) * 16384, NS = HL ? NSG / 2 : NSG, CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = base + NS * SB, empty = full + NS * 8;
  float* red = (float*)(smem + NS * SB + NS * 16);
  if (threadIdx.x < NS) { mbinit(full + threadIdx.x * 8, TMA ? 1 : 32); mbinit(empty + threadIdx.x * 8, 8); }
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  __syncthreads();
  const int SPT = q.KC / 8;  // stages per tile
  const int ST = q.MT * SPT, G = gridDim.x, c = blockIdx.x;
  const int s0 = (int)((long long)ST * c / G), s1 = (int)((long long)ST * (c + 1) / G);
  if (wid == 8) {  // producer
#if !TMA
    for (int i = s0; i < s1; i++) {
      const int k = i - s0, slot = k % NS;
      if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
      for (int o = lane * 16; o < 16384; o += 512) {
        cp16(base + slot * SB + o, q.A + (size_t)i * 16384 + o);
        if (HL) cp16(base + slot * SB + 16384 + o, q.A2 + (size_t)i * 16384 + o);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(full + slot * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    if (lane == 0) {
      for (int i = s0; i < s1 && TMA; i++) {
        const int k = i - s0, slot = k % NS;
        if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
#if TMA
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full + slot * 8), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB), "l"(q.A + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB + 16384), "l"(q.A2 + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
#endif
      }
    }
    return;
  }
  const int tb = 0;
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  uint4 b0[NT][2];
  auto ldb = [&](int i, uint4 (&b)[NT][2]) {
    const int ch = (i % SPT) * 8 + wid;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + ch * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  if (s0 < s1) ldb(s0, b0);
  for (int i = s0; i < s1; i++) {
    const int k = i - s0, slot = k % NS;
    uint4 b1[NT][2];
    if (i + 1 < s1) ldb(i + 1, b1);
    mbwait(full + slot * 8, (k / NS) & 1);
    const uint32_t sa = base + slot * SB + wid * 2048 + lane * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4 a = lds(sa + j * 512);
      uint4 h; if (HL) h = lds(sa + 16384 + j * 512);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
        mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
    __syncwarp();
    if (lane == 0) mbarrive(empty + slot * 8);
#pragma unroll
    for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
    const int tile = i / SPT;
    if (i + 1 == s1 || (i + 1) % SPT == 0) {  // tile ends here (for this CTA): reduce the 8 warps' partials
#pragma unroll
      for (int n = 0; n < NT; n++) {
        float f[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
          for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
        ((float4*)red)[(n * 8 + wid) * 32 + lane] = make_float4(f[0], f[1], f[2], f[3]);
#pragma unroll
        for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      const int t0s = tile * SPT, t1s = t0s + SPT;
      const bool own = t0s >= s0 && t1s <= s1;
      const int row0 = tile * 16 + g;
      for (int n = wid; n < NT; n += 8) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w2 = 0; w2 < 8; w2++) { const float4 v = ((float4*)red)[(n * 8 + w2) * 32 + lane]; sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w; }
        const int tk = n * 8 + 2 * t;
        const float fv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int tok = tk + (e & 1), row = row0 + (e >> 1) * 8;
          if (tok >= q.M || row >= q.N) continue;
          if (own) put(q, tok, row, fv[e]);
          else atomicAdd(q.ws + (size_t)tok * q.N + row, fv[e]);
        }
      }
      if (!own) {
        __threadfence();
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        __shared__ int lastf;
        if (threadIdx.x == 0) {
          auto cta = [&](int st2) { return (int)(((long long)(st2 + 1) * G - 1) / ST); };  // CTA streaming stage st2
          const int nc = cta(t1s - 1) - cta(t0s) + 1;
          lastf = atomicAdd(q.cnt + tile, 1) == nc - 1;
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (lastf) {
          __threadfence();
          for (int e = threadIdx.x; e < 16 * q.M; e += 256) {
            const int tok = e >> 4, row = tile * 16 + (e & 15);
            if (row < q.N) { float* wp = q.ws + (size_t)tok * q.N + row; put(q, tok, row, __ldcg(wp)); *wp = 0.f; }
          }
          if (threadIdx.x == 0) q.cnt[tile] = 0;
        }
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
    }
  }
}
template <int NT, bool HL> static void go2(const Q& q, cudaStream_t st) {
  constexpr int NS = HL ? NSG / 2 : NSG;
  const size_t sm = (size_t)NS * (HL ? 32768 : 16384) + NS * 16 + 8 * 32 * 16 * NT;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(gemv<NT, HL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  gemv<NT, HL><<<g_grid, 288, sm, st>>>(q);
}
void run2(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = 1;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go2<1, true>(q, st); else if (nt <= 2) go2<2, true>(q, st); else go2<4, true>(q, st); }
  else { if (nt <= 1) go2<1, false>(q, st); else if (nt <= 2) go2<2, false>(q, st); else go2<4, false>(q, st); }
}
// ---------------------------------------------------------------- router (N 256, K 8192, m <= 32)
// CTA (ks, mt) owns row tile mt and 16 consecutive 64-wide K chunks, one per warp. Every warp loads its hi + lo A
// fragments and its x chunk straight into registers in one round trip (64 KB per CTA in flight); the 16 warp partials
// meet in smem and the last of the KS CTAs on a tile sums their partials (no atomics on the output, no second kernel).
#define RPART (1 << 20)
#define RCNT 2048
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int NT>
__global__ void __launch_bounds__(512, 1) rtr(Q q) {
  __shared__ float red[16 * 512];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ks = blockIdx.x, mt = blockIdx.y, KS = gridDim.x, c = ks * 16 + w;
  const uint8_t* ah = q.A + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  const uint8_t* al = q.A2 + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  uint4 a[4], h[4], b[NT][2];
#pragma unroll
  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const uint4* xr = (const uint4*)(q.x + (size_t)min(n * 8 + g, q.M - 1) * q.K + c * 64) + t * 2;
    b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
      mma(acc[n], a[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      mma(acc[n], h[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
    }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    float* r = red + w * 512 + g * 32 + n * 8 + 2 * t;
    *(float2*)r = make_float2(acc[n][0], acc[n][1]);
    *(float2*)(r + 8 * 32) = make_float2(acc[n][2], acc[n][3]);
  }
  __syncthreads();
  const int row = tid >> 5, tok = tid & 31;  // this thread's output (row of the tile, token)
  float s = 0.f;
  if (tok < NT * 8) {
#pragma unroll
    for (int ww = 0; ww < 16; ww++) s += red[ww * 512 + tid];
  }
  float* P = q.ws + RPART + (size_t)mt * KS * 512;
  P[ks * 512 + tid] = s;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(q.cnt + RCNT + mt, 1) == KS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  float tot = 0.f;
  for (int k2 = 0; k2 < KS; k2++) tot += __ldcg(P + k2 * 512 + tid);
  if (tok < q.M) ((float*)q.out)[(size_t)tok * q.N + mt * 16 + row] = tot;
  if (tid == 0) q.cnt[RCNT + mt] = 0;
}
void run3(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = (const uint8_t*)A2.data_ptr();
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)(N / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 1;
  const dim3 grid(q.KC / 16, q.MT);
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (nt <= 1) rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);
}
"""

# ---------------------------------------------------------------- v13/v14: register-direct swap-AB GEMV on 12-bit-packed BF16
# A unit is one 16-row tile x one 64-wide K chunk in mma.sync A-fragment order, per lane 3 x 16 B (c12) or 4 x 16 B (bf16):
# c12 keeps each value's low byte (exponent LSB + mantissa) and a nibble (sign + 3-bit offset of the upper 7 exponent bits
# from a per-matrix window); out-of-window values are fixed from a per-warp escape list (exact). FMT 2/3: the router's
# BF16 hi + lo pair (bf16 / c12), fp32 out. Warp (wr, wk) of CTA (bx, by) streams tiles (bx * WR + wr) * RT + j, chunks
# [(by * WK + wk) * KU, + KU) straight into registers; x comes from registers (XS 0) or a CTA-shared smem slab (XS 1);
# partials meet in smem, then across the S CTAs of a cluster through DSMEM (sm_90) or a last-arriver workspace (older GPUs).
GVCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };

__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p);  // pw16 x load policy (defined below)
__device__ __forceinline__ uint4 gx(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p) {
  uint4 v;
  if constexpr (XP == 1) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else if constexpr (XP == 2) asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else v = gx(p);
  return v;
}
__device__ __forceinline__ uint4 lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// V & 1 == 0: nibble v = offset of value v (byte-permute table T0:T1 = base .. base + 7), signs at bits 3, 11, 19, 27 (values 0-3)
// and 7, 15, 23, 31 (values 4-7): 11 ops / 8 values
__device__ __forceinline__ void dec0(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hl = prmt(T0, T1, cm) | ((C << 4) & 0x80808080u), hh = prmt(T0, T1, cm >> 16) | (C & 0x80808080u);
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
// V & 1 == 1: C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i, bit 3 the sign of value 4 + (i + 3) % 4,
// so (C & 0x87878787) + base and the same on C rotated by 4 are the high bytes of values 0-3 and 4-7 (window base <= 120)
__device__ __forceinline__ void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
template <typename O>
__device__ __forceinline__ void emit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  sto(o, v.x); sto(o + 8, v.z);
  if (tok + 1 < p.m) { sto(o + p.N, v.y); sto(o + p.N + 8, v.w); }
}
// pw27: split-K partial added into an output the caller zeroed (a memset node in the same graph); fp32 (router) or bf16
__device__ __forceinline__ void reda(float* d, float v) { atomicAdd(d, v); }
__device__ __forceinline__ void reda2(__nv_bfloat16* d, float a, float b) {  // native packed bf16x2 reduction (no return value)
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" :: "l"(d), "r"(*(const uint32_t*)&h) : "memory");
}
template <typename O>
__device__ __forceinline__ void redemit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  if constexpr (std::is_same<O, float>::value) {
    reda(o, v.x); reda(o + 8, v.z);
    if (tok + 1 < p.m) { reda(o + p.N, v.y); reda(o + p.N + 8, v.w); }
  } else {  // bf16: rows (g, g+1) of lanes l, l ^ 4 (same tokens, same liveness) as one native packed bf16x2 red.add
    const unsigned am = __activemask();
    const float px = __shfl_xor_sync(am, v.x, 4), py = __shfl_xor_sync(am, v.y, 4), pz = __shfl_xor_sync(am, v.z, 4), pw = __shfl_xor_sync(am, v.w, 4);
    if (((l >> 2) & 1) == 0) {  // even row g: the pair starts at an even column (4-B aligned: row0 % 16 == 0, N even)
      reda2(o, v.x, px); reda2(o + 8, v.z, pz);
      if (tok + 1 < p.m) { reda2(o + p.N, v.y, py); reda2(o + p.N + 8, v.w, pw); }
    }
  }
}
__device__ __forceinline__ void add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __forceinline__ bool live(int idx, int MT, int m) { return ((idx >> 5) % MT) * 8 + 2 * (idx & 3) < m; }

// ---- pw18: codec E v3, routine copied from proj_rsi/tools/codec_e.py CODEC_E_CUH ----
// ---- codec E v3 device decode (proj_rsi/tools/CODEC_E_LAYOUT.md v3) ------------------------------------------------
// K = 1, 2 or 4 independent substreams per lane; substream s owns steps j in [s*4/K, (s+1)*4/K) of every unit.
// Per lane and substream: lo, hi = 64-bit bit buffer (next code at bit 0 of lo); V = valid bits mod 128 (bits 8+ carry
// garbage from V -= len; only (V & 0x60) and (V & 31) are read); pb = bytes taken from the substream's lane stream;
// iss = words issued into the ring; nw = prefetched refill word.
// Code length: the first 3 bits of every code determine its length, so len = byte (lo & 7) of the 8-byte table (t0, t1):
// LOP3 + PRMT. The bit-position chain (LOP3 -> PRMT -> SHF) has no memory access. The value (both exponents) comes from
// lut[lo & 0xff] (LDS, off the chain); A = lop3(e, 0x807f807f, plane bytes, 0xb8) (bits 0-6 and 15 from the plane).
// Stream: per segment, K substream regions of P * 32 words; word q of lane l of substream s at soff + (s * P + q) * 32 + l
// (4 B pieces: coalesced 128 B per warp, padding to a word). Ring per warp and substream: ES = ED / K word slots x 32
// lanes ([s][slot][lane], 128 B per slot: one bank per lane, conflict-free); word q in slot q % ES.
// Per unit: ce_issue for each substream (NI = 4 / K predicated 4 B cp.async), one commit_group, wait_group(2) (the
// prologue issues 3 rounds); then ce_unit. A substream's first refill word of the unit is read after the wait, the
// others are prefetched one step ahead (same landed set, no issue in between).
struct CeS { uint32_t lo, hi, V, pb, iss, nw; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f): immLut b ? c : a = 0xb8
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// smem address of the word at substream byte pb: rl = ring_s + lane * 4, ring_s aligned to ES * 128
template <int ES>
__device__ __forceinline__ uint32_t ce_addr(uint32_t rl, uint32_t pb) { return rl | ((pb << 5) & ((ES - 1) << 7)); }
// NI predicated 4 B cp.async; gsrc = stream + (soff[gw] + s * P * 32 + lane) * 4 (word 0 of this lane and substream)
template <int ES, int NI>
__device__ __forceinline__ void ce_issue(CeS& b, uint32_t P, const uint8_t* gsrc, uint32_t rl) {
#pragma unroll
  for (int k = 0; k < NI; k++) {
    const bool go = b.iss < P && b.iss < (b.pb >> 2) + ES;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4; }\n"
                 :: "r"(rl | ((b.iss % ES) << 7)), "l"(gsrc + (size_t)b.iss * 128), "r"((uint32_t)go));
    b.iss += go;
  }
}
__device__ __forceinline__ void ce_refill(CeS& b, uint32_t w) {
  const bool need = (b.V & 0x60u) == 0u;
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// one unit: the K substreams' chains interleaved (independent), 16 A registers in r6 order A[4 j + i];
// rl = ring of substream 0 + lane * 4 (substream s at + s * ES * 128); LS = LUT index shift (2 single, 7 lane-replicated)
template <int ES, int LS, int K>
__device__ __forceinline__ void ce_unit(CeS* b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint4 q0, uint4 q1, uint32_t* A) {
  constexpr int NS = 4 / K;
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int t = 0; t < NS; t++) {
#pragma unroll
    for (int s = 0; s < K; s++) {
      const uint32_t rs = rl + s * ES * 128;
      ce_refill(b[s], t == 0 ? ce_lds(ce_addr<ES>(rs, b[s].pb)) : b[s].nw);
      if (t + 1 < NS) b[s].nw = ce_lds(ce_addr<ES>(rs, b[s].pb));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int s = 0; s < K; s++) {
        const int j = s * NS + t;
        const uint32_t len = ce_prmt(t0, t1, b[s].lo & 7u);
        const uint32_t e = ce_lds(lut + ((b[s].lo & 0xffu) << LS));
        b[s].lo = __funnelshift_r(b[s].lo, b[s].hi, len); b[s].hi = __funnelshift_r(b[s].hi, 0u, len); b[s].V -= len;
        A[4 * j + i] = ce_sel(e, ce_prmt(L[2 * j + (i >> 1)], 0u, (i & 1) ? 0x3322u : 0x1100u));
      }
  }
}
// escapes (r6 walk, unchanged except the XOR field): A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1))
// pw20: one step t (4 A registers) of ce_unit<ES, LS, 1>, identical operations in identical order (refill, prefetch for
// step t + 1, 4 pairs), so a caller can decode and consume a unit step by step with 4 A registers live instead of 16
template <int ES, int LS>
__device__ __forceinline__ void ce_step1(CeS& b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint32_t Lw0, uint32_t Lw1, int t, uint32_t* A) {
  ce_refill(b, t == 0 ? ce_lds(ce_addr<ES>(rl, b.pb)) : b.nw);
  if (t < 3) b.nw = ce_lds(ce_addr<ES>(rl, b.pb));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t len = ce_prmt(t0, t1, b.lo & 7u);
    const uint32_t e = ce_lds(lut + ((b.lo & 0xffu) << LS));
    b.lo = __funnelshift_r(b.lo, b.hi, len); b.hi = __funnelshift_r(b.hi, 0u, len); b.V -= len;
    A[i] = ce_sel(e, ce_prmt((i >> 1) ? Lw1 : Lw0, 0u, (i & 1) ? 0x3322u : 0x1100u));
  }
}

// pw12: per-warp private bulk-copy ring for W (V bit 3). V bits 4-5: slots NS = 2 << code; bits 6-7: chunks per slot
// CU = 1 << code; bit 8: refill without fence.proxy.async (the slot's values are consumed by the decode, in program
// order, before the __syncwarp and the refill; CUTLASS-style release). Lane 0 of each warp fills its own slots with cp.async.bulk (one copy per row tile and matrix, CU
// contiguous 64-k chunks) on a per-slot mbarrier; the warp waits on it, reads its fragments with ld.shared.v4 at the
// offsets r6's ld.global.v4 used, and lane 0 refills the slot after the warp has decoded its last chunk.
template <int FMT, int MT, int WR, int WK, int KU, int S, int RT, int XS, int V>
struct GL {
  static constexpr int NL = FMT == 4 ? 2 : ((FMT & 1) ? 3 : 4), NA = (FMT == 2 || FMT == 3) ? 2 : 1, NW = WR * WK, NI = WR * RT * MT * 32;
  // pw18 codec E v3 (FMT 4): V bits 22-23 = log2 substreams CK (1/2/4); bit 19 = ring 32 words per lane (else 16), split
  // into ES = CED / CK slots per substream; bit 20 = LUT replicated per lane (32 KB, conflict-free); smem = per-warp ring
  // (2048-aligned) + LUT, after the slab / reduction buffers
  static constexpr int CK = 1 << ((V >> 22) & 3), CED = ((V >> 19) & 1) ? 32 : 16, CES = CED / CK;
  static constexpr int CLUT = FMT == 4 ? (((V >> 20) & 1) ? 32768 : 1024) : 0;
  static constexpr int CAL = CES * 128 > 2048 ? CES * 128 : 2048;  // ring alignment (ring_s | slot offset)
  static constexpr int CRING = FMT == 4 ? NW * 32 * CED * 4 + CAL : 0;  // + CAL: alignment slack
  static constexpr bool BK = (V & 8) != 0;
  static constexpr int CU = BK ? 1 << ((V >> 6) & 3) : 1, G = KU / CU, NS0 = BK ? 2 << ((V >> 4) & 3) : 1;
  static constexpr int NS = NS0 < G ? NS0 : G, SB = RT * NA * CU * NL * 512;
  static constexpr int SLAB = XS == 1 ? WK * KU * MT * 8 * 128 : 0, REDB = (WK > 1 || S > 1) ? (WK + 1) * NI * 16 : 0;
  static constexpr int BAR = BK ? ((NW * NS * 8 + 127) / 128) * 128 : 0, RING = BK ? NW * NS * SB : 0;
  static constexpr int SMEM = SLAB + REDB + BAR + RING + CLUT + CRING;
};
__device__ __forceinline__ void gxwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nGXW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra GXW;\n}\n" :: "r"(b), "r"(ph) : "memory");
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv(G p) {
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}

#include <type_traits>

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv_empty_guard(G p) {
  static_assert(FMT==1 && MT==1 && ST==1 && WK==16 && KU==8 && S==1 && RT==1 && XS==2 && ((WR==2 && V==1) || (WR==1 && V==8193)), "PW42 approved QKV empty-escape geometries only");
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](auto REPAIR, const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if constexpr (decltype(REPAIR)::value && (C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if constexpr (decltype(REPAIR)::value && (C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (nx == ~0u) {
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(std::false_type{}, u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(std::false_type{}, u, u % ST);
  }
  } else {
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(std::true_type{}, u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(std::true_type{}, u, u % ST);
  }
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}

__global__ void p1zero(uint4* o, int n16) {  // P1 (V bit 28): the PDL primary, lets the gv grid launch at once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) o[i] = make_uint4(0u, 0u, 0u, 0u);
}
template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st, int occ) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = []() {
    if constexpr (V & 1073741824) return gv_empty_guard<FMT, MT, ST, WR, WK, KU, S, RT, XS, (V & ~1073741824)>;
    else return gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }();
  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;
  const size_t sm = (XS == 1 ? (size_t)WK * KU * MT * 8 * 128 : 0) + ((WK > 1 || S > 1) ? (size_t)(WK + 1) * NI * 16 : 0)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::BAR + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::RING)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CLUT + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CRING);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if ((V >> 17) & 1) { cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 0); cudaGetLastError(); }  // pw16
    if (S > 8) { cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaGetLastError(); }
    once = true;
  }
  static int cc = 0;
  if (!cc) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S + (((V >> 29) & 1) ? 1 : 0)); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  const bool clu = S > 1 && cc >= 9 && !(V & 4) && !((V >> 26) & 15);  // pw11: V bit 2 = global last-arriver reduction, no cluster
  if (occ == 2) { cudaFuncAttributes fa; if (cudaFuncGetAttributes(&fa, k) != cudaSuccess) { cudaGetLastError(); return -1; } return fa.numRegs; }  // pw14
  if (occ) {  // pw11: how many CTAs of this launch can be co-resident (clusters: cudaOccupancyMaxActiveClusters)
    int n = 0;
    if (clu) {
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { cudaGetLastError(); return -1; }
      return (int64_t)n * S;
    }
    int dev, sms; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WR * WK * 32, sm) != cudaSuccess) { cudaGetLastError(); return -1; }
    return (int64_t)n * sms;
  }
  if (clu) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }
  if constexpr (((V >> 28) & 1) != 0) {
    const int n16 = (int)(((size_t)p.m * p.N * sizeof(typename std::conditional<FMT == 2 || FMT == 3, float, __nv_bfloat16>::type)) / 16);
    p1zero<<<(n16 + 255) / 256 < 8 ? (n16 + 255) / 256 : 8, 256, 0, st>>>((uint4*)p.out, n16);
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
  }
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R, int NT = 512>
__global__ void __launch_bounds__(NT) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[NT / 32][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  constexpr int NI = 2048 / NT;  // float4 per thread per row (K = 8192)
  float4 wv[R][NI];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < NI; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * NT); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < NI; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * NT);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

// ---------------------------------------------------------------- pw22: router m = 128 / 256 as a split-K GEMM
// out[tok][n] (fp32) = sum_k x[tok][k] (hi[n][k] + lo[n][k]), hi / lo = the BF16 split of the FP32 router weights, both
// products accumulated in fp32 by the same mma.sync (as SGLang above 8 tokens). CTA (n tile of BN, K split of KR, token
// block of 128): every x row slice (128 x KR) and hi / lo row slice (BN x KR) is requested up front with 16-B cp.async
// into XOR-swizzled smem (16-B chunk c of row r at r*KR*2 + ((c ^ (r & 7)) << 4): ldmatrix rows hit distinct banks),
// one wait, then KR/16 k16 steps of ldmatrix.x4 + mma.sync m16n8k16 (A = x rows, B = W rows: both K-major). 8 warps:
// warp w owns tokens 32 (w & 3) .. +31 and n 0.5 BN (w >> 2) .. +0.5 BN. Rows >= m are zero-filled (src-size 0).
// RED 0: red.global.add.v2.f32 into an output zeroed by the caller (a memset node in the same graph);
// RED 1: fp32 partial to a workspace slot, per-(token block, n tile) counter, the last arriving CTA sums the KS slots.
__device__ __forceinline__ void rs_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rs_ldm4(uint32_t a, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void rs_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(a), "f"(x), "f"(y) : "memory");
}
template <int BN, int KR, int RED>
__global__ void __launch_bounds__(256, 1) rsk(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                            float4* ws, int* cnt, int m, int N, int K) {
  constexpr int TOK = 128, CH = KR / 8, WN = BN / 2, NT8 = WN / 8, RB = KR * 2;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOK * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOK;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < TOK * CH; i += 256) {
    const int r = i / CH, c = i - r * CH, tok = t0 + r;
    rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
  }
  for (int i = tid; i < BN * CH; i += 256) {
    const int r = i / CH, c = i - r * CH;
    const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
    const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
    rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  const int wt = w & 3, wn = w >> 2;
  float acc[2][NT8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;                 // A: row (lane & 15) of the m16 tile, k chunk + (lane >> 4)
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;  // B: n row of the n16 pair, k chunk + (j & 1)
#pragma unroll 4
  for (int q = 0; q < KR / 16; q++) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
      const int r = wt * 32 + mi * 16 + ar, c = 2 * q + ac;
      rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int nb = 0; nb < NT8 / 2; nb++) {
        const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
        uint32_t b[4];
        rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
        }
      }
  }
  if constexpr (RED == 0) {
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) {
        const int tok = t0 + wt * 32 + mi * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
        if (tok < m) rs_red2(out + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
        if (tok + 8 < m) rs_red2(out + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
      }
  } else {
    constexpr int NF = 2 * NT8 * 32 * 8;  // float4 fragments per CTA partial (8 warps)
    float4* slot = ws + ((size_t)(tb * NTL + nt) * KS + ks) * NF;
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) __stcg(slot + ((w * 2 + mi) * NT8 + j) * 32 + lane, make_float4(acc[mi][j][0], acc[mi][j][1], acc[mi][j][2], acc[mi][j][3]));
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float4* base = ws + (size_t)(tb * NTL + nt) * KS * NF;
    for (int i = tid; i < NF; i += 256) {
      float4 v = __ldcg(base + i);
      for (int k2 = 1; k2 < KS; k2++) { const float4 u = __ldcg(base + (size_t)k2 * NF + i); v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
      const int l = i & 31, f = i >> 5, j = f % NT8, mi = (f / NT8) & 1, ww = f / NT8 / 2;
      const int tok = t0 + (ww & 3) * 32 + mi * 16 + (l >> 2), n = n0 + (ww >> 2) * WN + j * 8 + 2 * (l & 3);
      if (tok < m) { out[(size_t)tok * N + n] = v.x; out[(size_t)tok * N + n + 1] = v.y; }
      if (tok + 8 < m) { out[(size_t)(tok + 8) * N + n] = v.z; out[(size_t)(tok + 8) * N + n + 1] = v.w; }
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;  // self-cleaning for graph replay
  }
}
// ---------------------------------------------------------------- pw23: pipelined router split-K (rsk2)
// rsk with (1) the K slice loaded in NSTG commit groups, all issued up front; stage s is multiplied after
// cp.async.wait_group(NSTG - 1 - s) + a CTA barrier, so the mma of stage s overlaps the landing of s + 1 ..;
// (2) TOKS = 128 or 256 tokens per CTA: at m = 256 one CTA holds both 128-token blocks and every ldmatrix'd W fragment
// feeds MI = TOKS / 64 m16 tiles per warp (the W chain is not doubled); (3) RED 0 = red.add into an output zeroed by a
// memset node; RED 2 = red.add into a persistent fp32 workspace (zero at prepare), then per output tile a counter: the
// CTA that completes the tile copies it ws -> out and re-zeroes its ws tile and the counter (self-cleaning).
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
__global__ void __launch_bounds__(256, MINB) rsk2(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                                 float* ws, int* cnt, int m, int N, int K) {
  constexpr int CH = KR / 8, CS = CH / NSTG, WN = BN / 2, NT8 = WN / 8, RB = KR * 2, MI = TOKS / 64, QS = KR / 16 / NSTG;
  static_assert(CH % NSTG == 0 && NT8 % 2 == 0, "stages / n tiles");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOKS * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOKS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {  // stage s = 16-B chunks [s CS, (s + 1) CS) of every row
    for (int i = tid; i < TOKS * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS, tok = t0 + r;
      rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
    }
    for (int i = tid; i < BN * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS;
      const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
      const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
      rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  const int wt = w & 3, wn = w >> 2;
  float acc[MI][NT8][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    if (s == 0) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG - 1) : "memory");
    else if (s == 1) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 1 ? NSTG - 2 : 0) : "memory");
    else if (s == 2) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 2 ? NSTG - 3 : 0) : "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll 2
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {  // warp wt owns tokens 32 wt .. of each 128-token block: mi = (block, m16)
        const int r = (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + ar, c = 2 * q + ac;
        rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int nb = 0; nb < NT8 / 2; nb++) {
          const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
          uint32_t b[4];
          rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
          for (int mi = 0; mi < MI; mi++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
          }
        }
    }
  }
  float* dst = RED == 0 ? out : ws;
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int j = 0; j < NT8; j++) {
      const int tok = t0 + (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
      if (tok < m) rs_red2(dst + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
      if (tok + 8 < m) rs_red2(dst + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
    }
  if constexpr (RED == 2) {  // the CTA completing this (token block, n tile) copies it to out and re-zeroes it
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int rows = min(TOKS, m - t0);
    for (int i = tid; i < rows * (BN / 4); i += 256) {
      const int r = i / (BN / 4), c4 = i - r * (BN / 4);
      float4* src = (float4*)(ws + (size_t)(t0 + r) * N + n0) + c4;
      const float4 v = __ldcg(src);
      __stcg(src, make_float4(0.f, 0.f, 0.f, 0.f));
      ((float4*)(out + (size_t)(t0 + r) * N + n0))[c4] = v;
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;
  }
}
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
static int rs2_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float* ws, long long wsn, int* cnt, int cntn,
                  int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (TOKS + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + TOKS - 1) / TOKS, KS = K / KR, NTL = N / BN;
  if (RED == 2 && ((long long)TB * TOKS * N > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk2<BN, KR, NSTG, TOKS, RED, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk2<BN, KR, NSTG, TOKS, RED, MINB><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
// variant: 0 = pw22's rs64k32 (in-run reference); 5-11 = rsk2 (see RSP in the Python part)
template <int BN, int KR, int RED>
static int rs_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
                 int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (128 + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + 127) / 128, KS = K / KR, NTL = N / BN;
  if (RED == 1 && ((long long)TB * NTL * KS * (BN / 16) * 2 * 32 * 8 > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk<BN, KR, RED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk<BN, KR, RED><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
int rs_launch(int v, const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
              int m, int N, int K, cudaStream_t st) {
  switch (v) {
    case 0: return rs_go<64, 256, 0>(x, hi, lo, out, ws, wsn, cnt, cntn, m, N, K, st);
    case 5: return rs2_go<64, 256, 4, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 6: return rs2_go<64, 256, 2, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 7: return rs2_go<64, 128, 2, 128, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 8: return rs2_go<64, 256, 4, 128, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 9: return rs2_go<64, 256, 4, 256, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 10: return rs2_go<64, 128, 2, 256, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 11: return rs2_go<64, 256, 4, 256, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
  }
  return -2;
}
// ---------------------------------------------------------------- pw27 PROBE ladder for the 6-8 MB regime (no paths)
// pr_ldg: G CTAs x 256 threads, every 16-B load of the CTA's share issued at launch (NL per thread, ld.global.nc
// .L1::no_allocate), then RED 0: nothing; 1: 64 fp32 atomicAdds per CTA into a zeroed 8 KB output; 2: cluster-8 DSMEM
// sum of 64 floats per CTA, rank 0 writes. pr_blk: G CTAs, one lane issues the CTA's share as CHK-byte cp.async.bulk
// copies on one mbarrier at launch, then waits.
template <int NL, int RED>
__global__ void __launch_bounds__(256) pr_ldg(const uint4* buf, long long n16, float* out, uint32_t* sink) {
  const long long T = (long long)gridDim.x * 256, t = (long long)blockIdx.x * 256 + threadIdx.x;
  uint4 v[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) { const long long j = t + (long long)i * T; v[i] = j < n16 ? gnc(buf + j) : make_uint4(0u, 0u, 0u, 0u); }
  uint32_t a = 0;
#pragma unroll
  for (int i = 0; i < NL; i++) a ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
  if constexpr (RED == 1) {
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) atomicAdd(out + (blockIdx.x * 64 + threadIdx.x) % 2048, (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u));
  } else if constexpr (RED == 2) {
    __shared__ float red[64];
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) red[threadIdx.x] = (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u);
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    if (rk == 0 && threadIdx.x < 64) {
      float s = 0.f;
      const uint32_t la = (uint32_t)__cvta_generic_to_shared(red + threadIdx.x);
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t ra; float f;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(q));
        asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(f) : "r"(ra) : "memory");
        s += f;
      }
      out[(blockIdx.x / 8 * 64 + threadIdx.x) % 2048] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
template <int CHK>
__global__ void __launch_bounds__(32) pr_blk(const uint8_t* buf, long long per, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), bar = sb + (uint32_t)per;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"((uint32_t)per) : "memory");
    for (long long o = 0; o < per; o += CHK)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sb + (uint32_t)o), "l"(buf + blockIdx.x * per + o), "r"(CHK), "r"(bar) : "memory");
  }
  __syncwarp();
  asm volatile("{\n.reg .pred p;\nPRB_W:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PRB_W;\n}\n" :: "r"(bar) : "memory");
  uint32_t a; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(a) : "r"(sb + threadIdx.x * 4));
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
// kind: 0-3 ldg NL 16 / 8 / 4 / 2 (no reduction), 4 ldg NL 8 + red, 5 ldg NL 8 + cluster-8 DSMEM, 6 bulk 8 KB, 7 bulk 16 KB
int pr_launch(int kind, int G, const void* buf, long long bytes, float* out, uint32_t* sink, cudaStream_t st) {
  const long long n16 = bytes / 16;
  switch (kind) {
    case 0: pr_ldg<16, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 1: pr_ldg<8, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 2: pr_ldg<4, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 3: pr_ldg<2, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 4: pr_ldg<8, 1><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 5: {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(G); cfg.blockDim = dim3(256); cfg.stream = st;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = 8; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaLaunchKernelEx(&cfg, pr_ldg<8, 2>, (const uint4*)buf, n16, out, sink) != cudaSuccess) { cudaGetLastError(); return -3; }
      break;
    }
    case 6: case 7: {
      const long long per = bytes / G;
      const int chk = kind == 6 ? 8192 : 16384;
      if (per % chk || per > 200 * 1024) return -4;
      auto k = kind == 6 ? pr_blk<8192> : pr_blk<16384>;
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per + 64) != cudaSuccess) { cudaGetLastError(); return -6; }
      k<<<G, 32, (int)per + 64, st>>>((const uint8_t*)buf, per, sink);
      break;
    }
    default: return -2;
  }
  return (int)cudaGetLastError();
}
int64_t rrrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const float4* w = (const float4*)W.data_ptr(); const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr(); float* o = out.data_ptr<float>();
  if (K != 8192) return -1;
  switch (inst) {
    case 0: if (m > 1) return -1; rr<1, 1><<<N, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 1: if (m > 1) return -1; rr<1, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 2: if (m > 8) return -1; rr<8, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 3: if (m > 8) return -1; rr<8, 4><<<N / 4, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 4: if (m > 1) return -1; rr<1, 1, 256><<<N, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 5: if (m > 1) return -1; rr<1, 2, 256><<<N / 2, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 6: if (m > 1) return -1; rr<1, 1, 1024><<<N, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    case 7: if (m > 1) return -1; rr<1, 2, 1024><<<N / 2, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    default: return -2;
  }
  const cudaError_t r = cudaGetLastError();
  return (int64_t)r;
}

static int64_t gvgo(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U, int O) {
  G p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = out.data_ptr();
  p.esc = (const uint32_t*)esc.data_ptr(); p.ws = (float4*)ws.data_ptr(); p.cnt = cnt.data_ptr<int>();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64;
  p.T0 = (uint32_t)(T & 0xffffffff); p.T1 = (uint32_t)((uint64_t)T >> 32); p.U0 = (uint32_t)(U & 0xffffffff); p.U1 = (uint32_t)((uint64_t)U >> 32);
  auto st = at::cuda::getCurrentCUDAStream();
  const int occ = O;
  switch (inst) {
@@CASES@@
  }
  return -2;
}
int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 0);
}
int64_t gvocc(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 1);
}
int64_t prrun(int64_t kind, int64_t G, torch::Tensor buf, torch::Tensor out, torch::Tensor sink) {
  return (int64_t)pr_launch((int)kind, (int)G, buf.data_ptr(), (long long)buf.numel() * buf.element_size(), out.data_ptr<float>(),
                            (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}
int64_t rsrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0)) return -10;
  return (int64_t)rs_launch((int)v, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)hi.data_ptr(), (const __nv_bfloat16*)lo.data_ptr(),
                            out.data_ptr<float>(), (float4*)ws.data_ptr(), (long long)(ws.numel() / 4), cnt.data_ptr<int>(), (int)cnt.numel(),
                            (int)x.size(0), (int)out.size(1), (int)x.size(1), at::cuda::getCurrentCUDAStream());
}
int64_t gvregs(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 2);
}
"""
# candidate launch geometries (WR, WK, KU, S, RT, XS) per (layer key, MT); FMT is c12 (1/3) with a bf16 (0/2) twin of the
# first geometry as the fallback when a 12-bit pack fails
# candidates per (layer key, MT): (WR, WK, KU, S, RT, XS) -> kernel variants V (bit 0: 9-op add decode, else 11-op table
# decode; bit 1: rolled K loop); H200 v15 judge: the v14 kernel (V 0) was faster at deep K, the rolled/9-op one (V 3) at m32
_0, _1, _01 = (0,), (1,), (0, 1)
GVC = {  # H200 v16 log: V1 (unrolled, 9-op decode) wins most shapes, V3 (rolled) only pooler m1 at 1024 threads and qkv m16
    # pw33 (H100 lab, audits/p2-h100.md): short per-warp chains with many K-warps. WR = 2 row tiles per CTA (x shared by
    # the two row-tile warps through L1, no split-K): 72 CTAs x 32 warps (KU 8, ST 1) or x 16 warps (KU 16, x in
    # registers, ST 2); and one tile per CTA x 16 warps (KU 8, ST 1, x through L1) at 2 CTAs/SM (V bits 12-14 = minBlocks 2)
    ("qkv_gate", 1): [((1, 8, 16, 1, 1, 0), _1), ((1, 4, 16, 2, 1, 0), _01), ((1, 4, 8, 4, 1, 0), _1), ((2, 16, 8, 1, 1, 2), _1), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,))],
    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), _1), ((1, 4, 8, 4, 1, 2), (1, 3)), ((1, 4, 16, 2, 1, 2), _01), ((1, 4, 16, 2, 1, 2, 1), _1), ((1, 4, 8, 4, 1, 2, 1), _1),
                      ((2, 8, 16, 1, 1, 0, 2), _1), ((2, 16, 8, 1, 1, 2), (3,)), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,))],
    ("qkv_gate", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _01), ((2, 4, 16, 2, 1, 2), _1), ((1, 4, 8, 4, 1, 2, 1), _01), ((1, 4, 8, 4, 1, 2), _1), ((1, 4, 16, 2, 1, 2, 1), _1)],
    ("o", 1): [((1, 8, 2, 1, 1, 0), _01), ((2, 4, 4, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 0), _1)],
    ("o", 2): [((1, 8, 2, 1, 1, 0), _01), ((2, 4, 4, 1, 1, 0), _1), ((2, 4, 4, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2), _1), ((1, 4, 4, 1, 1, 2, 1), _1)],
    ("o", 4): [((1, 4, 4, 1, 1, 2), _01), ((2, 4, 4, 1, 1, 2), _01), ((1, 8, 2, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 2, 1), _01), ((1, 8, 2, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2, 1), _1)],
    ("pooler", 1): [((2, 8, 16, 1, 1, 0), _01), ((2, 16, 8, 1, 1, 0), (3,)), ((1, 8, 16, 1, 1, 0), _1)],
    ("pooler", 2): [((2, 8, 16, 1, 1, 0), _01), ((1, 8, 16, 1, 1, 2), _1), ((1, 8, 16, 1, 1, 0), _1)],
    ("pooler", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _1), ((2, 4, 16, 2, 1, 2), _1)],
    ("unpooler_s", 1): [((1, 16, 4, 1, 1, 0), _01), ((1, 8, 2, 4, 1, 0), _1), ((1, 8, 4, 2, 1, 0), _1)],
    ("unpooler", 2): [((1, 4, 16, 1, 1, 2), _01), ((1, 4, 16, 1, 1, 0), _1), ((1, 8, 8, 1, 1, 0), _1)],
    ("unpooler", 4): [((1, 4, 16, 1, 1, 2), _01), ((2, 4, 16, 1, 1, 2), _1), ((1, 8, 8, 1, 1, 2), _1)],
    # router: plain BF16 hi + lo fragments (FMT 2), every warp's units in flight at once, split-K over a 16-CTA cluster
    ("router", 1): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0), ((1, 8, 2, 8, 1, 0), _0)],
    ("router", 2): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0), ((1, 8, 2, 8, 1, 0), _0)],
    ("router", 4): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0), ((1, 8, 2, 8, 1, 0), _0)],
}
GVRB = []  # [(128, 4), (256, 4)]
GVB = {1: 1, 8: 1, 16: 2, 32: 4}  # bucket m -> MT (the kernel also runs m > 8 MT as token blocks in grid.x; slower than cuBLAS)


def _gvst(geo):  # a 7th geometry field forces the register-ring depth
    if len(geo) == 7:
        return geo[6]
    wr, wk, ku, s, rt, xs = geo
    return 1 if wr * wk >= 32 else 3 if xs == 1 and rt == 1 else 2


def _gvinst():
    out = []
    for (key, mt), geos in GVC.items():
        for k, (geo, vs) in enumerate(geos):
            if key == "router":
                out.append((2, mt, _gvst(geo)) + geo[:6] + (0,))
                continue
            out += [(1, mt, _gvst(geo)) + geo[:6] + (v,) for v in vs] + ([(0, mt, _gvst(geo)) + geo[:6] + (0,)] if k == 0 else [])
    return list(dict.fromkeys(out))


# pw11 paths: (name, layer, buckets, (FMT, WR, WK, KU, S, RT, XS), V). V 1 = 9-op decode with r6's epilogue (S > 1:
# cluster DSMEM), V 5 = same kernel with the global last-arriver reduction (no cluster). Rows per CTA = 16 WR RT.
GX = True
GXP = []
GX11 = False  # pw17: pw11's wide-row gx paths never won (checks 824f32ce, f1e6a186, f03e2e77); not built
for _ms, _xs in (((1, 8), 0), ((16, 32), 2)):
    for _wr in (9, 8):  # 144 rows: 16 x 8 = 128 CTAs; 128 rows: 2304 = 18 x 128 -> 144 CTAs
        for _ab, _v in (("a", 1), ("b", 5)):
            if _ab == "a":  # pw12 trim: q144a (cluster 8 declined on the judge, 120 < 128) and q128a dropped
                continue
            _ms2 = tuple(m_ for m_ in _ms if not (_wr == 8 and m_ == 32))  # pw12 trim: q128b m=32 (L1-x, MT 4)
            if _ms2:
                GXP.append(("gx-q%d%s" % (16 * _wr, _ab), "qkv_gate", _ms2, (1, _wr, 1, 16, 8, 1, _xs), _v))
for _ab, _v in (("a", 1), ("b", 5)):
    if _ab == "b":  # pw12 trim: gx-q144a-s dropped (cluster 8 declined)
        GXP.append(("gx-q144%s-s" % _ab, "qkv_gate", (32,), (1, 9, 1, 16, 8, 1, 1), _v))
    GXP.append(("gx-p128%s" % _ab, "pooler", (16,), (1, 8, 2, 16, 4, 1, 2), _v))  # pw12 trim: m=32 L1-x (8 B spill) dropped
    GXP.append(("gx-p128%s-s" % _ab, "pooler", (32,), (1, 8, 2, 16, 4, 1, 1), _v))
    GXP.append(("gx-u128%s" % _ab, "unpooler", (16,), (1, 8, 2, 16, 2, 1, 2), _v))  # pw12 trim: m=32 L1-x dropped
    GXP.append(("gx-u128%s-s" % _ab, "unpooler", (32,), (1, 8, 2, 16, 2, 1, 1), _v))
GXP.append(("gx-o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1))
GXP.append(("gx-r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5))  # pw12 trim: r12c8 / r12c16 dropped
if not GX11:
    GXP = []


# pw12: the same kernels with W through a per-warp cp.async.bulk smem ring (V bit 3; slots 4/8; chunks per slot 1/2).
# Geometries: r6's judge-picked one per (layer, bucket) (job 23cecdef1bb54c9e), r6's 16-warp pooler alternative, and
# pw11's wide-row global-reduction geometries. An 8th geo field forces r6's register x-ring depth ST.
def _bv(v, ns, cu=1, fence=False):
    return v | 8 | ({2: 0, 4: 1, 8: 2, 16: 3}[ns] << 4) | ({1: 0, 2: 1, 4: 2}[cu] << 6) | (0 if fence else 256)


GB = True
for _nm, _ly, _ms, _g, _v, _vars in (
        ("q", "qkv_gate", (1, 8), (1, 1, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1))),
        ("q144b", "qkv_gate", (1, 8), (1, 9, 1, 16, 8, 1, 0), 5, ((8, 1),)),
        ("q", "qkv_gate", (16,), (1, 1, 4, 8, 4, 1, 2, 1), 1, ((4, 1), (8, 1))),
        ("q144b", "qkv_gate", (16, 32), (1, 9, 1, 16, 8, 1, 2), 5, ((8, 1),)),
        ("q", "qkv_gate", (32,), (1, 1, 4, 16, 2, 1, 2, 1), 1, ((4, 1), (8, 1))),
        ("o", "o", (1, 8), (1, 1, 8, 2, 1, 1, 0), 0, ((4, 1),)),
        ("o", "o", (16,), (1, 1, 8, 2, 1, 1, 0), 1, ((4, 1),)),
        ("o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1, ((4, 1),)),
        ("p", "pooler", (1, 8), (1, 2, 16, 8, 1, 1, 0), 3, ((4, 1),)),
        ("p16w", "pooler", (1, 8, 16), (1, 2, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1), (4, 2))),
        ("p", "pooler", (32,), (1, 1, 8, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
        ("p128b", "pooler", (16, 32), (1, 8, 2, 16, 4, 1, 2), 5, ((4, 1),)),
        ("u", "unpooler", (1, 8), (1, 1, 16, 4, 1, 1, 0), 1, ((4, 1),)),
        ("u", "unpooler", (16,), (1, 1, 4, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
        ("u", "unpooler", (32,), (1, 1, 4, 16, 1, 1, 2), 0, ((4, 1), (8, 1))),
        ("u128b", "unpooler", (16, 32), (1, 8, 2, 16, 2, 1, 2), 5, ((4, 1),)),
        ("r", "router", (1, 8, 16), (2, 1, 8, 2, 8, 1, 0), 0, ((4, 1),)),
        ("r", "router", (32,), (2, 1, 4, 2, 16, 1, 0), 0, ((4, 1),)),
        ("r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5, ((4, 1),))):
    for _ns, _cu in _vars if GB else ():
        if True:  # pw21: no bulk-ring paths (gb-q-n4 never won)
            continue
        GXP.append(("gb-%s-n%d%s" % (_nm, _ns, "c%d" % _cu if _cu > 1 else ""), _ly, _ms, _g, _bv(_v, _ns, _cu)))


# pw18: codec E v3 paths (FMT 4) on pw17's geometries (r6's judge-picked qkv 1.8.16, r6's 512-thread pooler 2.8.16,
# unpooler 1.4.16; qkv m=16 on 1.8.16 since pw17's 1.4.8 S4 cluster was declined on occupancy). V: base | bits 22-23 =
# log2 substreams (independent decode chains per lane) | bit 20 = LUT replicated per lane (32 KB, conflict-free) | bit 19
# = 32-word ring (not baked). Per (layer, m): K1 single LUT, K1 / K2 / K4 replicated. m=32, o and the router: r6.
GE = False  # pw21: codec E closed (pw17-pw20); r6 + router m=128/256 paths only
_ER, _EK2, _EK4 = 1 << 20, 1 << 22, 2 << 22
_GEG = {("qkv_gate", 1): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 8): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 16): (4, 1, 8, 16, 1, 1, 0),
        ("pooler", 1): (4, 2, 8, 16, 1, 1, 0), ("pooler", 8): (4, 2, 8, 16, 1, 1, 0), ("pooler", 16): (4, 2, 8, 16, 1, 1, 0),
        ("unpooler", 1): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 8): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 16): (4, 1, 4, 16, 1, 1, 2)}
_GEV = (("", 0), ("r", _ER), ("r-k2", _ER | _EK2), ("r-k4", _ER | _EK4))
_GEN = {"qkv_gate": "q", "pooler": "p512", "unpooler": "u"}
# qkv (256 threads) carries the __launch_bounds__ minBlocks 2 override (V bits 12-14): r6's 144 CTAs need 2 per SM to stay in
# one wave on 132 SMs, which caps registers at 128. qkv and pooler m=16 (MT 2) K2 / K4 spill at 128 and are not built.
GEP = [("ge-%s%s" % (_GEN[ly] if not (ly == "unpooler" and m <= 8) else "u16", sfx), ly, (m,), g, 1 | b | (2 << 12 if ly == "qkv_gate" else 0))
       for (ly, m), g in _GEG.items() for sfx, b in _GEV if not (ly != "unpooler" and m == 16 and b & (3 << 22))] if GE else []
# pw20 "lite" codec E paths at m = 1 / 8 (V bit 24): one substream, single LUT, plane ring ST 1 (8th geometry field), rolled
# K loop, one accumulator set, each unit decoded and consumed one k16 step at a time (4 A registers live): 64 registers,
# 0 spills (ptxas, sm_90a), so 1024 threads run per SM like r6's own m=1 geometry. __launch_bounds__ minBlocks (V bits
# 12-14) = 1024 / CTA threads. Geometries: pooler 2.16.8 (r6's m=1 geometry, 128 CTAs x 1024) and 1.16.8 (256 x 512, 2/SM);
# qkv 1.16.8 (144 x 512) and 1.8.16 (r6's, 144 x 256, up to 4/SM); unpooler (n = 1024) 1.8.8 (64 x 256).
_EL = 1 << 24
GEPL = [("ge-l%s" % nm, ly, (m,), g, 1 | 2 | ((1024 // nt) << 12) | _EL)
        for nm, ly, g, nt in (("p1k", "pooler", (4, 2, 16, 8, 1, 1, 0, 1), 1024), ("p16", "pooler", (4, 1, 16, 8, 1, 1, 0, 1), 512),
                              ("q16", "qkv_gate", (4, 1, 16, 8, 1, 1, 0, 1), 512), ("q8", "qkv_gate", (4, 1, 8, 16, 1, 1, 0, 1), 256),
                              ("u8", "unpooler", (4, 1, 8, 8, 1, 1, 0, 1), 256))
        for m in (1, 8)] if GE else []
GEP += GEPL
GXP += GEP

# pw27 (P1, the 6-8 MB regime): split-K without a cluster, reduced by red.add into an output zeroed by a memset node in
# the same graph (gv V bit 26), every W chunk of a warp requested at launch (ring depth ST = KU, 8th geometry field).
# Router: plain BF16 hi + lo (FMT 2), as r6 at m <= 32 (r6: cluster 8 / 16 DSMEM, 4.35-6.29 judge-us). Unpooler (rows
# 0..1023, K 4096) and o at m <= 8: r6's 12-bit pack (FMT 1, dec1).
_RD = 1 << 26
_ZF, _PZ, _XZ = 1 << 27, 1 << 28, 1 << 29
GRD = []
for _nm, _g, _ms, _vs in (("rk-s8", (2, 1, 8, 2, 8, 1, 0, 2), (8, 16, 32), "z"),           # 16 x 8 = 128 CTAs x 256 thr
                          ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16, 32), "z"),      # 256 CTAs x 256 thr, one chunk per warp
                          ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16), "x"),          # critic (pw27h): the x variant at m=32 is 272 CTAs > 264 resident slots
                          ("rk-s32", (2, 1, 4, 1, 32, 1, 0, 1), (8, 16), "x")):           # 512 CTAs x 128 thr
    for _c in _vs:
        GRD.append((_nm + _c, "router", _ms, _g, {"z": _ZF, "x": _XZ, "p": _PZ}[_c]))  # P1 self-zeroing variants
for _nm, _ly, _g, _vs in (("uk-s4", "unpooler", (1, 1, 8, 2, 4, 1, 0, 2), "zx"),    # n=1024: 64 x 4 = 256 CTAs x 256 thr
                          ("uk-s4w4", "unpooler", (1, 1, 4, 4, 4, 1, 0, 4), "x")):    # 256 CTAs x 128 thr
    for _c in _vs:
        GRD.append((_nm + _c, _ly, (1, 8), _g, 1 | {"z": _ZF, "x": _XZ}[_c]))
GXP += GRD


# pw14 PROBE ladder on r6's own gv (judge-picked geometries of job 23cecdef1bb54c9e), never registered as paths:
# (layer, MT, geo (FMT, WR, WK, KU, S, RT, XS[, ST]), V, step). LAD (V bits 9-11) 1..4 = steps L0..L3; L4 = r6's real
# pick, timed separately. capNN = the full kernel with the __launch_bounds__ minBlocks override (V bits 12-14).
GL_P = (1, 2, 16, 8, 1, 1, 0)
GL_P4 = (1, 1, 8, 16, 1, 1, 2)
GL_Q = (1, 1, 8, 16, 1, 1, 0)
GL_Q4 = (1, 1, 4, 16, 2, 1, 2, 1)
GLAD = [("pooler", 1, GL_P, 3 | 1 << 9, "L0"), ("pooler", 1, GL_P, 3 | 4 << 9, "L3"),
        ("qkv_gate", 1, GL_Q, 1 | 1 << 9, "L0"), ("qkv_gate", 1, GL_Q, 1 | 4 << 9, "L3")]
GLAD += [(ly, 4, g, 1 | k << 9, "L%d" % (k - 1)) for ly, g in (("pooler", GL_P4), ("qkv_gate", GL_Q4)) for k in (1, 2, 3, 4)]
GLAD += [("pooler", 4, GL_P4, 1 | 4 << 12, "cap64"), ("pooler", 4, GL_P4, 1 | 1 << 12, "cap255"),
         ("qkv_gate", 4, GL_Q4, 1 | 1 << 12, "cap255"),
         ("pooler", 4, GL_P4, 1 | 8 | 2 << 4 | 256 | 1 << 9, "L0-bulk8")]
GLAD = []  # pw16: the pw14 ladder is not built (its result is in audits/r6-lineage.md)

# pw16: x cache-policy experiment on r6's own m=16/32 geometries (judge picks). V bits 15-16 = x load policy XP
# (0 r6: ld.global.nc, L1-allocating; 1: ld.global.cg, L1 bypass = r6's L1 reuse removed; 2: ld.global.nc.L1::evict_last);
# V bit 17 = PreferredSharedMemoryCarveout 0 (largest L1 the kernel's smem allows). W loads stay .nc.L1::no_allocate.
GCX = (("-xcg", 1 << 15), ("-xel", 2 << 15), ("-L1", 1 << 17), ("-xelL1", 2 << 15 | 1 << 17))
for _nm, _ly, _m, _g, _v in (("p", "pooler", 16, (1, 2, 8, 16, 1, 1, 0), 1), ("p", "pooler", 32, (1, 1, 8, 16, 1, 1, 2), 1),
                             ("u", "unpooler", 16, (1, 1, 4, 16, 1, 1, 2), 1), ("u", "unpooler", 32, (1, 1, 4, 16, 1, 1, 2), 0),
                             ("q", "qkv_gate", 16, (1, 1, 4, 8, 4, 1, 2, 1), 1), ("q", "qkv_gate", 32, (1, 1, 4, 16, 2, 1, 2, 1), 1)):
    for _sfx, _b in ():  # pw17: pw16's x-policy paths all lost (GCX); not built
        GXP.append(("gc-%s%s" % (_nm, _sfx), _ly, (_m,), _g, _v | _b))


def _gxinst(geo7, v, bm):
    fmt, wr, wk, ku, s, rt, xs = geo7[:7]
    return (fmt, GVB.get(bm, 16), geo7[7] if len(geo7) > 7 else _gvst((wr, wk, ku, s, rt, xs)), wr, wk, ku, s, rt, xs, v)


GVI = list(dict.fromkeys(_gvinst() + [_gxinst(g, v, bm) for _, _, ms, g, v in GXP for bm in ms]
                        + [_gxinst(g, v, {1: 1, 2: 16, 4: 32}[mt]) for _, mt, g, v, _ in GLAD]))
GV_EMPTY_TRAIT = 1 << 30
GV_EMPTY_BASES = ((1, 1, 1, 2, 16, 8, 1, 1, 2, 1), (1, 1, 1, 1, 16, 8, 1, 1, 2, 8193))
GVI += [c[:-1] + (c[-1] | GV_EMPTY_TRAIT,) for c in GV_EMPTY_BASES]

_gvmod = None


def _gvext():
    global _gvmod
    if _gvmod is None:
        cases = "\n".join("    case %d: return gvl<%s>(p, st, occ);" % (i, ",".join(map(str, c))) for i, c in enumerate(GVI))
        _gvmod = load_inline("gv20_%x" % (hash(tuple(GVI)) & 0xffffffff), cpp_sources="int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rsrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t prrun(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                             cuda_sources=GVCU.replace("@@CASES@@", cases), functions=["gvrun", "gvocc", "gvregs", "rrrun", "rsrun", "prrun"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _gvmod


# code-word bit of each value's 3-bit offset and of its sign, per decode variant (see dec0 / dec1)
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


def _gvesc(esc, T, geo):
    """per-warp escape lists for one launch geometry (32 entries per warp, sorted by (unit, tile)); None if a list overflows.
    esc: [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
    wr, wk, ku, s, rt, xs = geo
    gx, nw = T // (wr * rt), wr * wk
    out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=esc.device)
    e = esc[esc[:, 0] < T]
    if len(e):
        t, c, j, l, v, x, h = e.t()
        gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
        seq = (c % ku) * rt + t % rt
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | (h << 9) | x)[o]
        rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


def _bk(m):
    return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m == 128 else 256 if m == 256 else 0


def _tm(fn, flush, it=15):
    """median replay time (ms) of a CUDA graph of fn, each replay after a read-only pass over a buffer larger than L2
    (the judge's cold-L2 method; a write flush leaves L2 dirty and penalises bandwidth-heavy kernels unevenly)"""
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    if _FL[0] is not flush:
        _FL[0], _FL[1] = flush, torch.empty((), dtype=flush.dtype, device=flush.device)
    ts = []
    for _ in range(it):
        torch.sum(flush, 0, out=_FL[1]); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


_FL = [None, None]


def _gvok(st, plan, exact, w, mr):
    """exact on every one-hot column, close on random rows"""
    ext, n, K = _gvext(), exact.shape[0], exact.shape[1]
    mm = (8 if plan[0] in (1002, 1003) else 1) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
    dt = exact.dtype
    eye = torch.eye(K, dtype=torch.bfloat16, device=w.device).view(K // mm, mm, K)
    got = torch.empty((K // mm, mm, n), dtype=dt, device=w.device)
    for i in range(K // mm):
        if _gvcall(st, plan, eye[i], got[i]):
            return False
    if not torch.equal(got.view(K, n), exact.t()):
        return False
    x = torch.randn((max(mm, mr), K), device=w.device).to(torch.bfloat16)
    out = torch.empty((max(mm, mr), n), dtype=dt, device=w.device)
    _gvcall(st, plan, x, out)
    ref = x.double() @ w.double().T
    return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < (2e-5 if dt == torch.float32 else 4e-3)


RRB = {1: (0, 6, 7)}  # P1: rr2/rr3 (no cross-CTA reduction) re-measured at m=8 on the H100; fp32 router kernels per bucket (m8 rr: 7.7-14.9 us on the H200, v12 4.7)


def _name(i):
    return "rr%d" % (i - 1000) if i >= 1000 else ".".join(map(str, GVI[i]))


def _gvcall(st, p, x, out):
    if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 26) & 1:  # pw27 red.add split-K: the output accumulates from zero
        out.zero_()
    if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 27) & 5:  # P1: per-instance ticket / zero-flag counters (self-zeroing)
        c = st.setdefault("p1cnt", {}).get(p[0])
        if c is None:
            c = st["p1cnt"][p[0]] = torch.zeros((2048,), dtype=torch.int32, device=x.device)
        return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], c, p[3], p[6])
    if p[0] == -7:  # pw22 router split-K
        return _rscall(st, p, x, out)
    if p[0] >= 1000:
        return _gvext().rrrun(p[0] - 1000, p[1], x, out)
    return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])


def _gvee_choices(layer, bm, geo, vs, c12):
    if not c12:
        return (0,)
    extra = tuple(v | GV_EMPTY_TRAIT for v in vs
                  if layer == "qkv_gate" and bm in (1, 8)
                  and (1, 1, _gvst(geo)) + geo[:6] + (v,) in GV_EMPTY_BASES)
    return tuple(vs) + extra


def _gvprep(st, flush):
    dev = flush.device
    st["gvws"] = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
    st["gvcnt"] = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
    noesc = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
    st["gvlog"] = []
    ok = {}
    z = torch.zeros(1, device=dev)
    off = _tm(lambda: z.zero_(), flush, 31)
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        hl = layer == "router"
        frs = list(st[layer]) if hl else [st[layer][0]]
        css = [[_c12(f, dv) for f in frs] if GVC12 and not hl else [None] for dv in (0, 1)]
        c12 = all(c is not None for cs in css for c in cs)
        if c12:
            esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(css[1])])
            st.setdefault("gvc12", {})[layer] = (css, esc)  # pw11: reused by the second bake
        Tf, KC = frs[0].shape[:2]
        w = st[layer + "_raw"]
        for bm, mt in list(GVB.items()) + (GVRB if hl else []):  # router m128/256 as token blocks: 1.6x slower than v12 on the 3090
            key = "unpooler_s" if layer == "unpooler" and bm <= 8 else layer
            n = 1024 if key == "unpooler_s" else Tf * 16
            T, wn = n // 16, w[:n]
            exact = (wn.to(torch.bfloat16).float() + (wn - wn.to(torch.bfloat16).float()).to(torch.bfloat16).float()) if hl else wn
            cands = []
            for k, (geo, vs) in enumerate(GVC.get((key, mt), [])):
                if not c12 and k and not hl:
                    break
                if geo[1] * geo[2] * geo[3] != KC or T % (geo[0] * geo[4]):
                    continue
                e = _gvesc(esc, T, geo[:6]) if c12 else noesc
                for v in _gvee_choices(layer, bm, geo, vs, c12):
                    fmt = 1 if c12 else 2 if hl else 0
                    i = GVI.index((fmt, mt, _gvst(geo)) + geo[:6] + (v,))
                    if e is None:
                        continue
                    cs = css[v & 1]
                    src = [c[0] for c in cs] if fmt & 1 else frs
                    plan = (i, src[0], e, cs[0][1] if fmt & 1 else 0, n, src[-1], cs[-1][1] if fmt & 1 else 0)
                    if (i, n, bm > 8 * mt) not in ok:
                        ok[(i, n, bm > 8 * mt)] = _gvok(st, plan, exact, wn, bm)
                    if ok[(i, n, bm > 8 * mt)]:
                        cands.append(plan)
            for i in RRB.get(bm, ()) if hl else ():
                plan = (1000 + i, wn.contiguous(), noesc, 0, n, wn, 0)
                if (plan[0], n, 0) not in ok:
                    ok[(plan[0], n, 0)] = _gvok(st, plan, wn, wn, bm)
                if ok[(plan[0], n, 0)]:
                    cands.append(plan)
            x = torch.randn((bm, KC * 64), device=dev).to(torch.bfloat16)
            dt = torch.float32 if hl else torch.bfloat16
            out = torch.empty((bm, n), dtype=dt, device=dev)
            t12 = _tm(lambda: _v12(x, layer, n, st), flush)
            res = [(layer, bm, "v12", t12 - off)]
            tt = []
            for p in cands:
                tt.append((_tm(lambda: _gvcall(st, p, x, out), flush, 9), p))
                res.append((layer, bm, _name(p[0]), tt[-1][0] - off))
            tt.sort(key=lambda r: r[0])
            best = (t12, None)
            for _, p in tt[:3]:  # re-time the leaders, alternating with v12
                a = _tm(lambda: _gvcall(st, p, x, out), flush, 21)
                t12 = min(t12, _tm(lambda: _v12(x, layer, n, st), flush, 21))
                if a < best[0]:
                    best = (a, p)
            if best[1] is not None and best[0] > t12 * GVMARGIN:
                best = (t12, None)
            st["gvlog"] += res
            print("gv %s m%d v12 %.2f best %s %.2f" % (layer, bm, (t12 - off) * 1e3, _name(best[1][0]) if best[1] is not None else "v12", (best[0] - off) * 1e3), flush=True)
            print("  " + " ".join("%s:%.2f" % (r[2], r[3] * 1e3) for r in res), flush=True)
            if best[1] is not None:
                st["gv"][(layer, bm)] = best[1]
            if P1ALT and hl and bm == 1 and best[1] is not None and best[1][0] >= 1001:  # P1: the rr pick vs r6's rr0, alternated
                p0 = next((p_ for p_ in cands if p_[0] == 1000), None)
                if p0 is not None:
                    ta, tb = [], []
                    for _ in range(3):
                        ta.append(_tm(lambda: _gvcall(st, best[1], x, out), flush, 21)); tb.append(_tm(lambda: _gvcall(st, p0, x, out), flush, 21))
                    print("PROBE p1alt router m=1 %s %.2f rr0=%.2f delta=%+.2f" % (_name(best[1][0]), min(ta) * 1e3, min(tb) * 1e3, (min(ta) - min(tb)) * 1e3), flush=True)

_mod = None
# host-only (g++): the judge's nvcc crashes on these STL containers under -std=c++20
LTCPP = r"""
// ---------------------------------------------------------------- cuBLASLt with an algorithm picked in prepare()
// layout bit 0: row-major descriptors with x as A (else column-major, W as A); bit 1: W pre-transposed [K, N] (packed)
// algorithm handle = layout * 100000 + index into that plan's list (heuristic results, then configs added by lt_add)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulHeuristicResult_t> algos; };
static cublasLtHandle_t g_lt = nullptr;
static void* g_ltws = nullptr;
static const size_t LTWS = 64ull << 20;
static std::map<std::tuple<int, int, int, int, int>, LtPlan> g_plans;
static cublasLtHandle_t lth() { if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_ltws, LTWS); } return g_lt; }
static LtPlan& ltplan(int m, int n, int k, int f32, int lay) {
  auto key = std::make_tuple(m, n, k, f32, lay);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  lth();
  const bool row = lay & 1, pk = lay & 2;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  LtPlan p;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.c, ct, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {p.a, p.b, p.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&p.c, ct, n, m, n);
  }
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = LTWS;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.algos.resize(32);
  int cnt = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 32, p.algos.data(), &cnt) != CUBLAS_STATUS_SUCCESS) cnt = 0;
  p.algos.resize(cnt);
  cublasLtMatmulPreferenceDestroy(pref);
  return g_plans.emplace(key, p).first->second;
}
int64_t lt_algos(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay) { return (int64_t)ltplan(m, n, k, f32, lay).algos.size(); }
std::vector<int64_t> lt_caps(int64_t f32, int64_t id) {
  cublasLtMatmulAlgo_t a;
  std::vector<int64_t> r;
  if (cublasLtMatmulAlgoInit(lth(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                             f32 ? CUDA_R_32F : CUDA_R_16BF, f32 ? CUDA_R_32F : CUDA_R_16BF, (int)id, &a) != CUBLAS_STATUS_SUCCESS) return r;
  for (auto cap : {CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_ALGO_CAP_STAGES_IDS}) {
    size_t bytes = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, nullptr, 0, &bytes);
    std::vector<uint32_t> v(bytes / 4);
    if (bytes) cublasLtMatmulAlgoCapGetAttribute(&a, cap, v.data(), bytes, &bytes);
    r.push_back((int64_t)v.size());
    for (auto x : v) r.push_back(x);
  }
  for (auto cap : {CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK}) {
    uint32_t v = 0; size_t w = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, &v, sizeof(v), &w);
    r.push_back(v);
  }
  return r;
}
static bool ltset(cublasLtMatmulAlgo_t& a, int attr, int64_t v) {
  if (attr >= 7) { uint16_t x = (uint16_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; }
  uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
}
// cfg: CUBLASLT_ALGO_CONFIG_{ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID}
int64_t lt_add(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay, std::vector<int64_t> cfg) {
  if (cfg.size() != 9) return -1;
  LtPlan& p = ltplan(m, n, k, f32, lay);
  cublasLtMatmulAlgo_t a;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
  for (int i = 1; i < 9; i++) if (!ltset(a, i, cfg[i]) && cfg[i]) return -1;
  if (cfg[2] > 1 && cfg[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return -1;
  uint64_t fl = 0; size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &fl, sizeof(fl), &w) != CUBLAS_STATUS_SUCCESS) return -1;
  if ((fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F || (fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) return -1;
  cublasLtMatmulHeuristicResult_t r{};
  if (cublasLtMatmulAlgoCheck(g_lt, p.op, p.a, p.b, p.c, p.c, &a, &r) != CUBLAS_STATUS_SUCCESS || r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > LTWS) return -1;
  r.algo = a;
  p.algos.push_back(r);
  return lay * 100000 + (int64_t)p.algos.size() - 1;
}
std::vector<int64_t> lt_cfg(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t code) {
  LtPlan& p = ltplan(m, n, k, f32, code / 100000);
  std::vector<int64_t> r;
  const auto& a = p.algos.at(code % 100000).algo;
  for (int i = 0; i < 9; i++) {
    size_t w = 0;
    if (i >= 7) { uint16_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
    else { uint32_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
  }
  return r;
}
int64_t lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, int64_t n, int64_t code) {
  const int m = x.size(0), k = x.size(1), f32 = out.scalar_type() == torch::kFloat32, lay = (int)(code / 100000);
  LtPlan& p = ltplan(m, (int)n, k, f32, lay);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (lay & 1) ? x.data_ptr() : wp;
  const void* b = (lay & 1) ? wp : x.data_ptr();
  const auto& al = p.algos.at(code % 100000);
  return (int64_t)cublasLtMatmul(g_lt, p.op, &alpha, a, p.a, b, p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                 &al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());
}
"""
_DBG = []
RTR = True
GV = True
GVC12 = True
GVMARGIN = 0.99
TUNE = True  # prepare(): time torch vs cuBLASLt algorithms per cuBLAS-served shape
V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _ext():
    global _mod
    if _mod is None:
        wpc = next((int(d[6:]) for d in _DBG if d.startswith("-DWPC=")), 16 if _big() else 8)
        _mod = load_inline("skinny_v20_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources=LTCPP + "void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                           cuda_sources=CUDA, functions=["run", "run2", "run3", "lt_algos", "lt_mm", "lt_caps", "lt_add", "lt_cfg"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if _big() else ["-DNSG=4"]) + [d for d in _DBG if not d.startswith("-DWPC=")], extra_ldflags=_ltflags(),
                           verbose=False)
    return _mod


def _ltflags():
    """link the process's cuBLASLt (torch's nvidia-cublas wheel ships only the versioned soname)"""
    import glob, os
    try:
        import nvidia.cublas as nc
        d = os.path.join(list(nc.__path__)[0], "lib")
    except Exception:
        d = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
    so = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
    return ["-L" + d, "-l:" + os.path.basename(so[0]) if so else "-lcublasLt", "-Wl,-rpath," + d]


def _tune(st, flush):
    """pick, per cuBLAS-served shape, the fastest of torch's default and every cuBLASLt heuristic algorithm (timed here,
    on this GPU, as the judge times: CUDA graph replays from a cold L2); m 128/256 get the pinned-config search below"""
    ext = _ext()
    tm = lambda fn, it=12: _tm(fn, flush, it)
    dev = flush.device
    jobs = []
    for layer, (N, K) in (("qkv_gate", (2304, 8192)), ("o", (8192, 1024)), ("pooler", (4096, 8192)), ("unpooler", (8192, 4096))):
        for m in (1, 8, 16, 32, 128, 256):
            if _cublas(layer, m):
                jobs.append((layer, m, 1024 if layer == "unpooler" and m <= 8 else N, K, st[layer + "_raw"], st.get(layer + "_t"), torch.bfloat16))
    for m in (128, 256):
        jobs.append(("router", m, 512, 8192, st["router_hl"], st["router_hl_t"], torch.float32))
    t0 = time.time()
    for layer, m, n, K, W, Wt, dt in jobs:
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=dt, device=dev)
        ref = torch.mm(x.float(), W[:n].float().T)
        f32 = int(dt == torch.float32)
        torch_fn = (lambda: torch.mm(x, W[:n].T, out_dtype=dt)) if f32 else (lambda: torch.mm(x, W[:n].T))
        ok = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c) == 0 and _close(out, ref)
        run = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c)
        if m < 128 or not LTSEARCH or Wt is None:
            best = (tm(torch_fn), -1)
            for i in range(ext.lt_algos(m, n, K, f32, 0)):
                if ok(i):
                    best = min(best, (tm(lambda: run(i)), i))
            if best[1] >= 0:
                st["lt"][(layer, m)] = best[1]
            continue
        st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0)


LTSEARCH = True
LTBUDGET = 22.0  # seconds for the whole m128/256 search (10 shapes)
LTW = {"qkv_gate": 1.6, "o": 0.3, "pooler": 1.0, "unpooler": 0.7, "router": 1.6}  # v18 H200: o never beat torch
# H200 picks reported for another submission's pinned search: (layout, epilogue, CUBLASLT_ALGO_CONFIG_* fields) -> seeds
_LTSEED = [(2, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (3, [66, 20, 1, 0, 0, 4, 35, 0, 2]),
           (0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 4, 35, 0, 2]),
           (3, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 1, 35, 0, 2]),
           (2, [66, 15, 1, 0, 0, 0, 35, 0, 4]), (2, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (3, [66, 18, 1, 0, 0, 2, 35, 0, 5]),
           (3, [66, 18, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 2, 35, 0, 5]), (2, [66, 18, 1, 0, 0, 0, 35, 0, 2]),
           (2, [66, 18, 1, 0, 0, 6, 35, 0, 5])]


def _close(out, ref):
    torch.cuda.synchronize()
    return bool(torch.isfinite(out).all()) and float((out.float() - ref).norm() / ref.norm()) < 1e-2


def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0):
    """heuristic algorithms of all four layouts + seeds + an algo-66 (sm90 nvjet) sweep over tile / stages / cluster, then
    split-K / custom option / cluster / stages refinement of the leaders; all checked numerically and timed (judge timer)"""
    if not _LTDONE and not _LT0:
        _LT0.append(time.time())
    t0 = _LT0[0]
    order = [(l, mm) for l in ("qkv_gate", "o", "pooler", "unpooler", "router") for mm in (128, 256)]
    wsum = sum(LTW[l] for l, _ in order)
    dl = t0 + LTBUDGET * sum(LTW[l] for l, mm in order[:order.index((layer, m)) + 1]) / wsum  # cumulative, weighted
    njob = wsum / LTW[layer]
    tried, res = set(), []
    def tryc(code):
        if code < 0 or code in tried:
            return
        tried.add(code)
        if ok(code):
            res.append((_tm(lambda: run(code), flush, 5), code))
    def addc(lay, cfg):
        c = ext.lt_add(m, n, K, f32, lay, cfg)
        if c >= 0:
            cfg2 = tuple(ext.lt_cfg(m, n, K, f32, c))
            if (lay, cfg2) in seen:
                return
            seen.add((lay, cfg2))
            tryc(c)
    seen = set()
    t_torch = _tm(torch_fn, flush, 9)
    for lay in range(4):
        for i in range(min(ext.lt_algos(m, n, K, f32, lay), 8)):
            seen.add((lay, tuple(ext.lt_cfg(m, n, K, f32, lay * 100000 + i))))
            tryc(lay * 100000 + i)
    for lay, cfg in _LTSEED:
        addc(lay, cfg)
    caps = ext.lt_caps(f32, 66)
    if caps:
        nt = caps[0]; tiles = caps[1:1 + nt]; ns = caps[1 + nt]; stages = caps[2 + nt:2 + nt + ns]
        cmax, ksup, swz = caps[2 + nt + ns], caps[3 + nt + ns], caps[4 + nt + ns]
        stg = [35] if 35 in stages else stages[:1] or [0]
        for lay in (2, 0, 3, 1):
            for tile in tiles:
                if time.time() > dl - 0.4 * LTBUDGET / njob:
                    break
                for cl in (2, 5):
                    addc(lay, [66, tile, 1, 0, 0, 0, stg[0], 0, cl])
        res.sort()
        for _, code in res[:4]:
            if time.time() > dl:
                break
            lay, cfg = code // 100000, list(ext.lt_cfg(m, n, K, f32, code))
            if cfg[0] != 66:
                continue
            for cu in range(min(cmax, 7) + 1):
                for cl in (0, 2, 3, 4, 5, 6):
                    if time.time() > dl:
                        break
                    addc(lay, cfg[:5] + [cu] + cfg[6:8] + [cl])
            if ksup:
                for sk in (2, 3, 4):
                    addc(lay, cfg[:2] + [sk, 2] + cfg[4:])
            for sg in stages[::max(1, len(stages) // 6)]:
                if time.time() > dl:
                    break
                addc(lay, cfg[:6] + [sg] + cfg[7:])
    res.sort()
    best = (t_torch, -1)
    for _, code in res[:3]:  # re-time the leaders against torch's default
        a = _tm(lambda: run(code), flush, 15)
        t_torch = min(t_torch, _tm(torch_fn, flush, 15))
        if a < best[0]:
            best = (a, code)
    if best[1] >= 0 and best[0] > t_torch * 0.99:
        best = (t_torch, -1)
    _LTDONE.append((layer, m))
    cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, f32, best[1]))))
    print("lt %s m%d heur %.2f best %s %.2f (%d tried, %.1fs)" % (layer, m, t_torch * 1e3, cfg, best[0] * 1e3, len(tried), time.time() - t0), flush=True)
    return best[1] if best[1] >= 0 else None


_LTDONE, _LT0 = [], []


def _cublas(layer, m):
    return layer != "router" and (m > 1 or layer == "o") and not (layer == "unpooler" and m == 1)


def _pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


def prepare(weights):
    st = {}
    if weights["qkv_gate"].device.type != "cuda":  # pw11: CPU (proj_rsi cpu-check) runs the plain torch math
        return {"cpu": True, "raw": dict(weights)}
    rqt = _rqstart() if RQ else None  # rs128: compile overlapped with r6's phase (CPU only)
    rqt2 = _rqstart(RQPROD2, _rqmod2, _rqthr2, "rq32") if RQ and RQS else None  # rs32: m = 16 / 32 instances
    for k, w in weights.items():
        if k == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            st[k] = (_pack(hi), _pack(lo))
            st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N]: one cuBLAS mm for m > 32
            st["router_hl"] = torch.cat((hi, lo)).contiguous()  # [2N, K] for cuBLASLt
        else:
            st[k] = (_pack(w), None)
            st[k + "_t"] = w.t().contiguous()  # [K, N] for the packed cuBLASLt layouts
        st[k + "_raw"] = w
    dev = weights["qkv_gate"].device
    st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    _ext()
    st["lt"] = {}
    if TUNE:
        _tune(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
    st["gv"] = {}
    if GV:
        _gvprep(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
    print("PREP r6 done in %.0fs" % (time.time() - _T0), flush=True)
    st["sel"] = {}
    st["rq"] = {}  # rs128 before the gx / router phases: its gain is the larger
    if RQ and rqt is not None:
        try:
            _rqbake(st, rqt)
        except Exception as error:  # the rs128 bake can only add m = 128 / 256 paths; r6's choices stand on any failure
            st["rq"] = {}
            print("PREP rq bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        finally:
            torch.cuda.synchronize()
    print("PREP phase rq done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
    if GV and GX:
        try:
            _gxbake(st)
        except Exception as error:  # the second bake can only add paths; r6's choices stand on any failure
            st["sel"] = {}
            print("PREP gx bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    print("PREP phase gx done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
    if RQ and RQS and rqt2 is not None:
        try:
            _rqsbake(st, rqt2)
        except Exception as error:  # can only add (layer, 16 / 32) rq paths; the incumbent stands on any failure
            for k_ in [k_ for k_ in st["rq"] if k_[1] in (16, 32)]:
                del st["rq"][k_]
            print("PREP rq32 bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        finally:
            torch.cuda.synchronize()
    print("PREP phase rq32 done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
    st.pop("rq_pk", None)
    if PR27 and GV and GX and time.time() - _T0 < PREPARE_BUDGET_S - 30:
        try:
            _pr27(st)
        except Exception as error:
            print("PROBE rd8 failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    if GV and GX and time.time() - _T0 < PREPARE_BUDGET_S - 20:
        try:
            _rsbake(st)
        except Exception as error:  # can only add router paths; r6's choices stand on any failure
            for b_ in (128, 256):
                st["sel"].pop(("router", b_), None)
            print("PREP rs bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        finally:  # critic (pw21): never leave a capture open or a side stream busy for the judge's own captures
            if torch.cuda.is_current_stream_capturing():
                print("PROBE l2scratch capture_left_open", flush=True)
            torch.cuda.synchronize()
    print("PREP phase router done at %.0fs" % (time.time() - _T0), flush=True)  # pw36: phase timestamps
    st.pop("gvc12", None)
    st.pop("gepk", None)  # pw17: unpicked codec E packs (a picked plan keeps its own pack alive)
    print("PREP done in %.0fs" % (time.time() - _T0), flush=True)
    return st


# ---------------------------------------------------------------- pw11 second bake (template conventions)
PREPARE_BUDGET_S = 240.0
MARGIN = 0.99
TOL = {"bf16": 4e-3, "f32": 2e-5}
_EYE = {}


def _time_us(fn, x, out, flush, sink, it=15):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x, out); fn(x, out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x, out)
    times = []
    for _ in range(it):
        torch.sum(flush, 0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _validate(fn, w, m, n, f32, gen, exact=None, m_odd=None):
    """template _validate: one-hot rows x8 bit-exact against the weight (for the router: against hi + lo, which is what
    the lossless 12-bit hi/lo pack carries), random rows within TOL of FP64 on the true weight; plus random rows at an
    odd m of the same bucket (kernel_fn serves the whole bucket with this plan)."""
    K = w.shape[1]
    dt = torch.float32 if f32 else torch.bfloat16
    out = torch.empty((m, n), dtype=dt, device=w.device)
    exact = (exact if exact is not None else w)[:n].t().to(dt)  # [K, n]
    rows = max(m, 1)
    if K not in _EYE:
        _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
    eye = _EYE[K]
    for start in range(0, K, rows):
        x = eye[start:start + rows]
        if x.shape[0] < rows:
            x = torch.cat((x, torch.zeros((rows - x.shape[0], K), dtype=x.dtype, device=x.device)))
        fn(x, out)
        got = out[: min(rows, K - start)]
        want = exact[start:start + got.shape[0]] * 8.0
        if not torch.isfinite(got).all() or not torch.equal(got.to(torch.float32), want.to(torch.float32)):
            return False, "one-hot mismatch at column %d" % start
    tol = TOL["f32" if f32 else "bf16"]
    for mm in (rows,) + ((m_odd,) if m_odd else ()):
        x = torch.randn((mm, K), generator=gen, device=w.device).to(torch.bfloat16)
        o = torch.empty((mm, n), dtype=dt, device=w.device)
        fn(x, o)
        ref = x.double() @ w[:n].double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
            return False, "random m=%d rel L2 %.2e row %.2e > %.0e" % (mm, err, row, tol)
    return True, "ok"


# ---------------------------------------------------------------- pw18: codec E v3 packer + reference unpacker
# (copied verbatim from proj_rsi/tools/codec_e.py; spec proj_rsi/tools/CODEC_E_LAYOUT.md v3)
ESC_MAX = 4e-4          # target escape rate (weights); keeps the 32-entry per-warp list far from overflow
LMAX = 8                # code length limit: 4 pairs per step <= 32 bits (one refill per step suffices)
PK = 3                  # v3: the first PK bits of a code (LSB-first) determine its length (8-entry byte table, one PRMT)


def _package_merge(cnt, L):
    """optimal prefix-code lengths (max length L) for positive counts (python list)."""
    n = len(cnt)
    if n == 1:
        return [1]
    leaves = sorted((float(q), (i,)) for i, q in enumerate(cnt))
    cur = list(leaves)
    for _ in range(L - 1):
        pk = [(cur[2 * i][0] + cur[2 * i + 1][0], cur[2 * i][1] + cur[2 * i + 1][1]) for i in range(len(cur) // 2)]
        cur = sorted(leaves + pk, key=lambda t: t[0])
    ln = [0] * n
    for _, syms in cur[:2 * n - 2]:
        for s in syms:
            ln[s] += 1
    return ln


def _canonical_lsb(lengths):
    """canonical codes, bit-reversed: the code of length l is the low l bits of the LSB-first peek window."""
    order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
    code, prev, out = 0, None, [0] * len(lengths)
    for s in order:
        l = lengths[s]
        if prev is not None:
            code = (code + 1) << (l - prev)
        prev = l
        out[s] = int(format(code, "0%db" % l)[::-1], 2)
    return out


def _geo(geo):
    wr, wk, ku, s, rt = geo[:5]
    return wr, wk, ku, s, rt


def _to_segments(x, T, KC, geo):
    """[T, KC, 4, 32, 4, ...] (tile, chunk, step j, lane, reg-in-step i) -> [NSEG, 32, KU*RT*16, ...] in segment
    (gw) order and lane-stream order (u, jt, r = 4 j + i)."""
    wr, wk, ku, s, rt = _geo(geo)
    gx = T // (wr * rt)
    rest = x.shape[5:]
    x = x.reshape(gx, wr, rt, s, wk, ku, 4, 32, 4, *rest)
    nd = len(rest)
    x = x.permute(3, 0, 4, 1, 7, 5, 2, 6, 8, *range(9, 9 + nd))  # by, bx, wk, wr, lane, u, jt, j, i
    return x.reshape(s * gx * wk * wr, 32, ku * rt * 16, *rest)


def _gvesc_e(t, c, j, l, v, x, T, geo, dev):
    """r6's _gvesc structure with an 8-bit exponent XOR (bits 0-7; the kernel applies x << (7 + 16 * (v & 1)))."""
    wr, wk, ku, s, rt = _geo(geo)
    gx, nw = T // (wr * rt), wr * wk
    out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=dev)
    if len(t):
        gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
        seq = (c % ku) * rt + t % rt
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


def _prefix_code(cnt, k=PK, L=LMAX):
    """optimal prefix code (max length L) in which the first k bits of every code determine its length: codes of length
    <= k, and buckets (one k-bit prefix each) of 2^(Lb - k) codes of length Lb. cnt: python list (any order).
    Returns (lengths, codes LSB-first, tab[2^k] = length per k-bit window)."""
    n, P = len(cnt), 2 ** k
    order = sorted(range(n), key=lambda s: (-cnt[s], s))
    p = [float(cnt[s]) for s in order]
    cum = [0.0]
    for x in p:
        cum.append(cum[-1] + x)
    INF = float("inf")
    D = [[INF] * (P + 1) for _ in range(n + 1)]
    B = [[None] * (P + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n):
        for u in range(P + 1):
            d = D[i][u]
            if d == INF:
                continue
            for l in range(1, k + 1):
                w = 2 ** (k - l)
                if u + w <= P and d + p[i] * l < D[i + 1][u + w]:
                    D[i + 1][u + w], B[i + 1][u + w] = d + p[i] * l, (i, u, "s", l)
            if u + 1 <= P:
                for lb in range(k + 1, L + 1):
                    j = min(n, i + 2 ** (lb - k))
                    v = d + (cum[j] - cum[i]) * lb
                    if v < D[j][u + 1]:
                        D[j][u + 1], B[j][u + 1] = v, (i, u, "b", lb)
    u = min(range(P + 1), key=lambda q: D[n][q])
    assert D[n][u] < INF, "no prefix code fits"
    items, i = [], n
    while i > 0:
        pi, pu, kind, l = B[i][u]
        items.append((kind, l, order[pi:i]))
        i, u = pi, pu
    items.reverse()
    # canonical assignment in the k-bit prefix space (virtual length: l for short codes, k for buckets), then bit-reversed
    items.sort(key=lambda it: it[1] if it[0] == "s" else k)
    lengths, codes, tab = [0] * n, [0] * n, [L] * P
    code, prev = 0, None
    for kind, l, syms in items:
        vl = l if kind == "s" else k
        if prev is not None:
            code = (code + 1) << (vl - prev)
        prev = vl
        pre = int(format(code, "0%db" % vl)[::-1], 2)
        if kind == "s":
            lengths[syms[0]], codes[syms[0]] = l, pre
            for w in range(P):
                if w & (2 ** l - 1) == pre:
                    tab[w] = l
        else:
            tab[pre] = l
            for j, s in enumerate(syms):
                lengths[s], codes[s] = l, pre | (j << k)
    return lengths, codes, tab


def _wpp(ilp):
    """words per ring piece. v3: 4 B pieces at every ilp (padding to a word, not 16 B: ilp 1/2/4 pad 0.12/0.22/0.36
    bits/w against 0.16/0.55/0.36 with 16 B pieces); the code supports 16 B pieces with per-lane word rotation"""
    return 1


def _substreams(x, ilp):
    """[NSEG, 32, KU*RT*16, ...] (unit, register) -> [NSEG, ilp, 32, KU*RT*16/ilp, ...]: substream s holds steps
    j in [s*4/ilp, (s+1)*4/ilp) of every unit, in unit order"""
    nseg, _, n = x.shape[:3]
    rest = x.shape[3:]
    x = x.reshape(nseg, 32, n // 16, ilp, 16 // ilp, *rest)
    x = x.permute(0, 3, 1, 2, 4, *range(5, 5 + len(rest)))
    return x.reshape(nseg, ilp, 32, n // ilp, *rest)


def pack_e(fr, geo, layer="?", verbose=True, ilp=1):
    """v3. r6 fragments [T, KC, 4, 32, 8] bf16 + gv geometry (WR, WK, KU, S, RT, ...) -> dict or None (escape overflow).
    keys: plane uint8 [T, KC, 2, 32, 16]; lut int32 [258] (256 value entries + the 8-byte length table); stream int32;
    soff int32 [NSEG + 1] (in 4 B words; a segment = ilp substream regions of P * 32 pieces of 16 B, 4 B at ilp 4);
    esc int32 [NSEG * 32]; ilp; stats dict."""
    t0 = time.time()
    dev = fr.device
    T, KC = fr.shape[:2]
    wr, wk, ku, s, rt = _geo(geo)
    assert wk * ku * s == KC and T % (wr * rt) == 0 and ilp in (1, 2, 4), (geo, T, KC, ilp)
    b = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff
    e8 = (b >> 7) & 0xff
    plane = ((b >> 8) & 0x80) | (b & 0x7f)
    n = b.numel()

    cnt = torch.bincount(e8.flatten(), minlength=256)
    val = torch.argsort(cnt, descending=True, stable=True)
    rank = torch.empty(256, dtype=torch.long, device=dev)
    rank[val] = torch.arange(256, device=dev)
    rk = rank[e8].view(T, KC, 4, 32, 4, 2)
    pid = rk[..., 0] * 256 + rk[..., 1]
    pc = torch.bincount(pid.flatten(), minlength=65536)
    order = torch.argsort(pc, descending=True, stable=True)
    npos = int((pc > 0).sum())
    pcf = pc.view(256, 256)
    N = None
    for cand in list(range(16, 256, 8)) + [256]:
        cand = min(cand, npos)
        C = torch.zeros(65536, dtype=torch.bool, device=dev)
        C[order[:cand]] = True
        C = C.view(256, 256)
        one = C.any(1)[:, None] | C.any(0)[None, :]
        esc = int((pcf * (~C) * (2 - one.long())).sum())
        if esc <= ESC_MAX * n or cand == npos:
            N = cand
            break
    syms = order[:N]
    C = torch.zeros(65536, dtype=torch.bool, device=dev)
    C[syms] = True
    Cm = C.view(256, 256)
    a_i = torch.arange(256, device=dev)
    rowbest = torch.where(Cm, pcf, -1).argmax(1)
    colbest = torch.where(Cm, pcf, -1).argmax(0)
    A, B = torch.meshgrid(a_i, a_i, indexing="ij")
    ph = torch.where(Cm, A * 256 + B,
                     torch.where(Cm.any(1)[:, None], A * 256 + rowbest[:, None],
                                 torch.where(Cm.any(0)[None, :], colbest[None, :] * 256 + B, syms[0].expand(256, 256))))
    ph = ph.flatten()
    eff = torch.bincount(ph[pid.flatten()], minlength=65536)
    symlist = syms.tolist()
    ln, code, tab = _prefix_code([max(int(eff[x]), 1) for x in symlist])
    sym_len = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_code = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_len[syms] = torch.tensor(ln, device=dev)
    sym_code[syms] = torch.tensor(code, device=dev)
    x = torch.arange(256, device=dev)
    sv = torch.tensor(symlist, device=dev)
    ent = (val[sv // 256] << 7) | (val[sv % 256] << 23) | torch.tensor(ln, device=dev)
    hit = (x[None, :] & (2 ** torch.tensor(ln, device=dev)[:, None] - 1)) == torch.tensor(code, device=dev)[:, None]
    assert bool((hit.sum(0) <= 1).all())
    lut = (hit.long() * ent[:, None]).sum(0)
    tabv = sum(t << (8 * i) for i, t in enumerate(tab))
    lut = torch.cat((lut, torch.tensor([tabv & 0xffffffff, tabv >> 32], device=dev)))

    php = ph[pid]
    pr = torch.stack((php // 256, php % 256), -1)
    mis = pr != rk
    ei = mis.nonzero()
    t_, c_, j_, l_, i_, h_ = ei.unbind(1)
    xr = (val[rk[mis]] ^ val[pr[mis]]).long()
    escl = _gvesc_e(t_, c_, j_, l_, 2 * i_ + h_, xr, T, geo, dev)
    if escl is None:
        if verbose:
            print("STATS_E %s escape list overflow (esc=%.5f)" % (layer, len(ei) / n))
        return None

    L = _substreams(_to_segments(sym_len[php], T, KC, geo), ilp)                 # [NSEG, ilp, 32, NP]
    V = _substreams(_to_segments(sym_code[php], T, KC, geo), ilp)
    nseg = L.shape[0]
    wpp = _wpp(ilp)                                                              # words per piece: 16 B, or 4 B at ilp 4
    off = torch.cumsum(L, 3) - L
    tot = L.sum(3)                                                               # [NSEG, ilp, 32]
    P = (tot.flatten(1).max(1).values + 32 * wpp - 1) // (32 * wpp)              # pieces per lane per substream
    soff = torch.zeros(nseg + 1, dtype=torch.long, device=dev)                   # in words (4 B)
    soff[1:] = torch.cumsum(P * 32 * ilp * wpp, 0)
    nwords = int(soff[-1])
    wd, sh = off >> 5, off & 31
    lane = torch.arange(32, device=dev)[None, None, :, None]
    sub = torch.arange(ilp, device=dev)[None, :, None, None]
    rot = ((lane >> 3) & 3) if wpp == 4 else 0
    gword = lambda w: soff[:-1, None, None, None] + ((sub * P[:, None, None, None] + w // wpp) * 32 + lane) * wpp + ((w % wpp) ^ rot)
    words = torch.zeros(nwords + 4, dtype=torch.long, device=dev)
    words.scatter_add_(0, gword(wd).flatten(), ((V << sh) & 0xffffffff).flatten())
    spill = (sh + L) > 32
    words.scatter_add_(0, gword(wd + 1)[spill], (V >> (32 - sh))[spill])
    words = words[:nwords]
    stream = torch.where(words >= 2 ** 31, words - 2 ** 32, words).int()

    pl = plane.to(torch.uint8)
    p0 = pl[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = pl[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    planeb = torch.stack((p0, p1), 2).contiguous()
    lut32 = torch.where(lut >= 2 ** 31, lut - 2 ** 32, lut).int()

    stream_bits = nwords * 32
    pad_bits = stream_bits - int(tot.sum())
    total_bits = 8 * n + stream_bits + 32 * (nseg + 1) + 32 * 258 + 32 * escl.numel()
    st = dict(bits=total_bits / n, esc=len(ei) / n, maxpad=pad_bits / stream_bits, maxpad_bits=pad_bits / n,
              N=N, code_bits=float(tot.sum()) / n, secs=time.time() - t0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        st["secs"] = time.time() - t0
    if verbose:
        print("STATS_E %s ilp%d bits=%.4f esc=%.5f maxpad=%.4f (code %.4f, pad %.4f bits/w, alphabet %d, len tab %s, %.2f s)"
              % (layer, ilp, st["bits"], st["esc"], st["maxpad"], 8 + st["code_bits"], st["maxpad_bits"], N,
                 "".join(map(str, tab)), st["secs"]))
    return dict(plane=planeb, lut=lut32, stream=stream, soff=soff.int(), esc=escl, ilp=ilp, stats=st)


def unpack_e(pk, T, KC, geo, ED=16):
    """v3 reference decoder: the device routine's logic per lane and substream (64-bit buffer, refill at step start when
    (V & 0x60) == 0, length = tab[lo & 7] (no memory on the chain), value = lut[lo & 0xff], V -= length), all lanes of
    all segments at once. Replays the per-substream smem ring protocol (ED words per lane in total, ED / K slots per
    substream; prologue: 3 rounds of NI = 4 / K predicated issues per substream + commit; per unit: one round + commit +
    wait_group(2)) and asserts every refill word (incl. the kernel's prefetched ones) is read from a landed, not yet
    overwritten piece. Returns bf16 bit patterns [T, KC, 4, 32, 8]."""
    dev = pk["stream"].device
    wr, wk, ku, s, rt = _geo(geo)
    K = pk["ilp"]
    NS = 4 // K
    M32 = 0xffffffff
    words = pk["stream"].long() & M32
    lut = pk["lut"].long() & M32
    tab = torch.tensor([((int(lut[256 + i // 4]) >> (8 * (i % 4))) & 255) for i in range(8)], device=dev)
    lut = lut[:256]
    soff = pk["soff"].long()
    nseg = soff.numel() - 1
    wpp = _wpp(K)
    P = (soff[1:] - soff[:-1]) // (32 * K * wpp)
    lane = torch.arange(32, device=dev)[None, None, :]
    sub = torch.arange(K, device=dev)[None, :, None]
    rot = ((lane >> 3) & 3) if wpp == 4 else 0
    z = torch.zeros(nseg, K, 32, dtype=torch.long, device=dev)
    lo, hi, Vr, ptr, iss = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    NU = ku * rt
    E = torch.zeros(nseg, K, 32, NU, NS, 4, dtype=torch.long, device=dev)
    Pl = P[:, None, None].expand_as(z)
    groups = []
    ES, NI = ED // K, NS                                                         # slots per substream, issues per unit

    def issue():
        nonlocal iss
        for _ in range(NI):
            go = (iss < Pl) & (iss < ptr // wpp + ES)
            iss = iss + go.long()
        groups.append(iss.clone())

    for _ in range(3):
        issue()
    for un in range(NU):
        issue()
        landed = groups[len(groups) - 1 - 2]
        for t in range(NS):
            need = (Vr & 0x60) == 0
            pc = ptr // wpp                                                    # ptr counts words
            inb = pc < Pl
            rd = need & inb                                                    # words the decode uses (other reads are masked)
            assert bool((pc < landed)[rd].all()), "ring: piece not landed"
            assert bool((pc + ES > iss - 1)[rd].all()), "ring: slot overwritten"
            gi = soff[:-1, None, None] + ((sub * P[:, None, None] + pc) * 32 + lane) * wpp + ((ptr % wpp) ^ rot)
            w = torch.where(need & inb, words[gi.clamp(max=words.numel() - 1)], 0)
            at = Vr & 31
            lo = torch.where(need, lo | ((w << at) & M32), lo)
            hi = torch.where(need, hi | ((w >> (32 - at)) * (at > 0)), hi)
            Vr = torch.where(need, (Vr + 32) & M32, Vr)
            ptr = ptr + need.long()
            for q in range(4):
                n_ = tab[lo & 7]
                E[..., un, t, q] = lut[lo & 0xff]
                lo, hi = ((lo >> n_) | (hi << (32 - n_))) & M32, hi >> n_
                Vr = (Vr - n_) & M32
    # [nseg, K, 32, NU, NS, 4] -> [nseg, 32, NU, 4 j, 4 i] -> fragment order
    E = E.permute(0, 2, 3, 1, 4, 5).reshape(nseg, 32, NU, 4, 4)
    gx = T // (wr * rt)
    E = E.view(s, gx, wk, wr, 32, ku, rt, 4, 4).permute(1, 3, 6, 0, 2, 5, 7, 4, 8).reshape(T, KC, 4, 32, 4)
    pl = pk["plane"].long()
    pb = pl.view(T, KC, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 4, 32, 8)
    lowh, hih = pb[..., 0::2], pb[..., 1::2]
    dup0, dup1 = lowh | (lowh << 8), hih | (hih << 8)
    m = 0x807f
    v0 = (E & ~m & 0xffff) | (dup0 & m)
    v1 = ((E >> 16) & ~m & 0xffff) | (dup1 & m)
    vals = torch.stack((v0, v1), -1).reshape(T, KC, 4, 32, 8)
    esc = pk["esc"].long() & M32
    ent = esc[esc != M32]
    if len(ent):
        gwi = (esc != M32).nonzero().flatten() // 32
        seq, ln_, r, half, x = ent >> 20, (ent >> 15) & 31, (ent >> 11) & 15, (ent >> 10) & 1, ent & 0xff
        nw = wr * wk
        wwr, wwk, g2 = gwi % nw % wr, gwi % nw // wr, gwi // nw
        bx, by = g2 % gx, g2 // gx
        t = (bx * wr + wwr) * rt + seq % rt
        c = (by * wk + wwk) * ku + seq // rt
        v = (r % 4) * 2 + half
        vals[t, c, r // 4, ln_, v] ^= x << 7
    return vals.int()


def _frag_bits(fr):
    return fr.reshape(*fr.shape[:2], 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff


def _gepack(st, layer, inst, n):
    """codec E v3 pack for one geometry and substream count, proven lossless by the reference unpacker (incl. the
    per-substream ring protocol at the instance's ring size) before any kernel uses it. pack_e runs once per (layer,
    rows, geometry, substreams). Returns the pack dict or None (escape overflow, >= 12 bits, or a failed proof)."""
    fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
    ed = 32 if (v >> 19) & 1 else 16
    ilp = 1 << ((v >> 22) & 3)
    geo = (wr, wk, ku, s, rt)
    cache = st.setdefault("gepk", {})
    key = (layer, n, geo, ilp, ed)
    if key in cache:
        return cache[key]
    t0 = time.time()
    fr = st[layer][0][: n // 16]  # unpooler at m <= 8 serves rows 0..1023 only
    T, KC = fr.shape[:2]
    tag = "%s n=%d g%d.%d.%d.%d.%d" % (layer, n, wr, wk, ku, s, rt)
    base = (layer, n, geo, ilp)
    if base not in cache:
        cache[base] = pack_e(fr, geo, tag, ilp=ilp)
        if fr.is_cuda:
            torch.cuda.synchronize()
        print("PREP gepack %s ilp%d pack_e %.2fs" % (tag, ilp, time.time() - t0), flush=True)
    pk = cache[base]
    if pk is not None:
        if pk["stats"]["bits"] >= 12.0:
            print("GE %s ilp%d: %.3f bits >= 12, not served" % (tag, ilp, pk["stats"]["bits"]), flush=True)
            pk = None
        else:
            try:
                good = torch.equal(unpack_e(pk, T, KC, geo, ed), _frag_bits(fr))
            except AssertionError as err:  # ring protocol violated for this data at this ring size
                print("GE %s ilp%d ed%d: %s, not served" % (tag, ilp, ed, err), flush=True)
                good = False
            if not good:
                print("GE %s ilp%d ed%d: reference unpack failed, not served" % (tag, ilp, ed), flush=True)
                pk = None
    cache[key] = pk
    if fr.is_cuda:
        torch.cuda.synchronize()
    print("PREP gepack %s ilp%d ed%d %s in %.2fs" % (tag, ilp, ed, "proven bit-exact" if pk is not None else "declined", time.time() - t0), flush=True)
    return pk


def _gxplan(st, layer, inst, n, packs):
    fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
    if fmt == 4:  # codec E v3: W = plane, W2 = stream, esc = 8-bit XOR list, T = LUT (+ length table) pointer, U = soff pointer
        pk = _gepack(st, layer, inst, n)
        if pk is None:
            return None
        return (GVI.index(inst), pk["plane"], pk["esc"], pk["lut"].data_ptr(), n, pk["stream"], pk["soff"].data_ptr(), pk)
    if not fmt & 1:  # plain bf16 fragments (r6's FMT 0 / router hi+lo FMT 2): no escapes
        frs = list(st[layer]) if fmt == 2 else [st[layer][0]]
        return (GVI.index(inst), frs[0], st["gvnoesc"], 0, n, frs[-1], 0)
    if layer not in packs:
        return None
    css, esc = packs[layer]
    e = _gvesc(esc, n // 16, (wr, wk, ku, s, rt, xs))
    if e is None:
        return None  # an escape list overflows (32 per warp): the path is not registered
    cs = css[v & 1]
    src = [c[0] for c in cs]
    return (GVI.index(inst), src[0], e, cs[0][1], n, src[-1], cs[-1][1])


def _gxbake(st):
    ext = _gvext()
    dev = st["qkv_gate_raw"].device
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    packs = dict(st.get("gvc12", {}))
    frs = list(st["router"])  # 12-bit hi and lo: the lo half must fit the escape lists too (checked per geometry)
    rc = [[_c12(f, dv) for f in frs] for dv in (0, 1)]
    if all(c is not None for cs in rc for c in cs):
        packs["router"] = (rc, torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(rc[1])]))
        print("GX router c12 escapes hi %d lo %d" % (len(rc[1][0][2]), len(rc[1][1][2])), flush=True)
    st["gvnoesc"] = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
    res_all = {}
    rw = st["router_raw"]
    rexact = rw.to(torch.bfloat16).float() + (rw - rw.to(torch.bfloat16).float()).to(torch.bfloat16).float()
    todo = []
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        for m in (1, 8, 16, 32, 128, 256):
            ps = [(nm, g, v) for nm, ly, ms, g, v in GXP if ly == layer and m in ms]
            if ps:
                todo.append((layer, m, ps))
    for layer, m, ps in todo:
        f32 = layer == "router"
        w = st[layer + "_raw"]
        K = w.shape[1]
        n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        r6 = lambda xx, oo, layer=layer, n=n: _r6(xx, layer, n, st)
        best = ("r6", None, _time_us(r6, x, out, flush, sink))
        print("BAKE %s m=%d r6 %.2f" % (layer, m, best[2]), flush=True)
        for name, geo, v in ps:
            if time.time() - _T0 > PREPARE_BUDGET_S:
                print("BAKE %s m=%d %s skipped-budget" % (layer, m, name), flush=True)
                continue
            try:
                inst = _gxinst(geo, v, m)
                plan = _gxplan(st, layer, inst, n, packs)
                if plan is None:
                    print("BAKE %s m=%d %s declined-%s" % (layer, m, name, "pack" if inst[0] == 4 else "escapes"), flush=True)
                    continue
                a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                grid = ((m + 8 * inst[1] - 1) // (8 * inst[1])) * (n // (16 * inst[3] * inst[7])) * inst[6]
                res = int(ext.gvocc(plan[0], *a))
                if inst[6] > 1 and not (v & 4) and not ((v >> 26) & 15) and res < grid:  # cluster variant: the whole grid must be co-resident
                    print("BAKE %s m=%d %s declined-occ %d<%d" % (layer, m, name, res, grid), flush=True)
                    continue
                if _gvcall(st, plan, x, out) != 0:
                    print("BAKE %s m=%d %s ERROR launch" % (layer, m, name), flush=True)
                    continue
                fn = lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo)
                ok, msg = _validate(fn, w, m, n, f32, gen, rexact if f32 else None, {8: 5, 16: 9, 32: 17, 128: 121, 256: 250}.get(m))
                if not ok:
                    print("BAKE %s m=%d %s INVALID %s" % (layer, m, name, msg), flush=True)
                    continue
                us = _time_us(fn, x, out, flush, sink)
            except Exception as error:
                print("BAKE %s m=%d %s ERROR %s" % (layer, m, name, str(error).splitlines()[-1][:160]), flush=True)
                continue
            print("BAKE %s m=%d %s %.2f%s" % (layer, m, name, us, "" if res >= grid else " (res %d<%d)" % (res, grid)), flush=True)
            res_all[(layer, m, name)] = (us, fn)
            if us < best[2] * MARGIN:
                best = (name, plan, us)
        if best[1] is not None:
            st["sel"][(layer, m)] = best[1]
        print("PICK %s m=%d %s %.2f" % (layer, m, best[0], best[2]), flush=True)
        if P1ALT and best[1] is not None and layer in ("router", "unpooler") and m <= 32 and time.time() - _T0 < PREPARE_BUDGET_S - 30:
            fb = lambda xx, oo, plan=best[1]: _gvcall(st, plan, xx, oo)
            tb, tr = [], []
            for _ in range(3):  # P1: the pick vs r6, alternated (the bake's 15-replay medians drift ~0.3 us between runs)
                tb.append(_time_us(fb, x, out, flush, sink, 21)); tr.append(_time_us(r6, x, out, flush, sink, 21))
            print("PROBE p1alt %s m=%d %s %.2f r6=%.2f delta=%+.2f" % (layer, m, best[0], min(tb), min(tr), min(tb) - min(tr)), flush=True)
        if layer == "pooler" and m in (1, 16):  # PROBE ecod: the best codec E path vs r6, alternated (pw17 kill test)
            cand = sorted((v_[0], k[2], v_[1]) for k, v_ in res_all.items() if k[:2] == (layer, m) and k[2].startswith("ge-"))
            if not cand:
                print("PROBE ecod pooler m=%d none" % m, flush=True)
            elif time.time() - _T0 > PREPARE_BUDGET_S:
                print("PROBE ecod pooler m=%d skipped-budget" % m, flush=True)
            else:
                tb, tr = [], []
                for _ in range(3):
                    tb.append(_time_us(cand[0][2], x, out, flush, sink, 21))
                    tr.append(_time_us(r6, x, out, flush, sink, 21))
                print("PROBE ecod pooler m=%d %s %.2f r6=%.2f delta=%+.2f" % (m, cand[0][1], min(tb), min(tr), min(tb) - min(tr)), flush=True)


# pw29: router m = 128 / 256 on wgmma (rwk). Own load_inline extension with sm_90a flags (pw24 lesson); r6's gv
# extension and pw23's rsk / rsk2 are untouched. CTA = 64 router rows (the wgmma M, W as A) x TOKS tokens (the wgmma N:
# m64n128k16 at 128 tokens, m64n256k16 or two n128 sets at 256) x a K slice of KR. One warpgroup (128 threads): every
# 16-B cp.async of the slice issued up front in NSTG commit groups (or a ring of NBUF 64-k slots when NBUF < NSTG),
# x / hi / lo tiles in 128B-swizzle K-major atoms (8 rows x 128 B, SBO 1024) read by both wgmma descriptors; per k16
# step two wgmmas (hi, lo) into the same fp32 accumulators. Epilogue: accumulators -> smem [tok][68] -> 16-B
# red.global.add.v4.f32 into the output zeroed by a memset node (half pw23's v2 op count). Emulated bit-exact by
# proj_rsi/tools/rw_emu.py (layouts, descriptors, ring reuse, epilogue, partial second block).
RWCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdint.h>
@@WG@@
__device__ __forceinline__ void rw_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rw_cpwait(int n) {  // n is a constant after unrolling: the switch folds
  switch (n) {
    case 7: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
  }
}
// K-major, 128B swizzle: start >> 4, LBO 1 (unused), SBO 1024 B (8 rows x 128 B), base offset 0, mode 1 << 62
__device__ __forceinline__ uint64_t rw_desc(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void rw_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
template <int NA> __device__ __forceinline__ void rw_keep(float* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int WN> __device__ __forceinline__ void rw_mma(float* d, uint64_t a, uint64_t b) {
  if constexpr (WN == 128) rw_mma128(d, a, b); else rw_mma256(d, a, b);
}
// slot = one 64-k chunk: x rows [0, TOKS), hi rows [TOKS, TOKS + 64), lo rows [TOKS + 64, TOKS + 128), 128 B each;
// 16-B column c of row r at slot + r * 128 + ((c ^ (r & 7)) << 4) (TOKS % 8 == 0: the hi / lo tiles keep the phase).
// MODE bits: 1 loads only (ladder), 2 no reductions (ladder), 4 self-zeroing (no memset node: the first CTA of each
// output tile to arrive in a call zeroes the tile and releases a per-tile flag; the others acquire it before their
// reds; per-tile arrival counters are monotonic across calls, so nothing is reset). CL > 1: cluster of CL K-splits
// pre-reduces its partials through DSMEM, each CTA then reds 1/CL of the token rows.
__device__ __forceinline__ int rw_ldacq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void rw_strel(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long rw_gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ void rw_csync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
__global__ void __launch_bounds__(128, MINB) rwk(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ hi,
                                                 const __nv_bfloat16* __restrict__ lo, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();  // an unexpected plain sm_90 pass compiles to a stub instead of failing the extension
#else
  constexpr bool RING = NBUF < NSTG;
  constexpr int QT = KR / 16, QS = QT / NSTG, CPS = 2 * QS, NSLOT = RING ? NBUF : KR / 64;
  constexpr int ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68, LI = ROWS * CPS / 128;
  static_assert(KR % 64 == 0 && QT % NSTG == 0 && (ROWS * CPS) % 128 == 0, "stages");
  static_assert(RING ? QS == 4 : NBUF == NSTG, "a ring stage is one 64-k chunk");
  static_assert(TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB, "epilogue staging");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  constexpr bool ST = (MODE & 16) != 0;  // MODE 16: globaltimer stamps by thread 0 into cnt[8192..] (8 x u64 per CTA)
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  auto load = [&](int s) {
#pragma unroll
    for (int it = 0; it < LI; it++) {
      const int i = tid + it * 128, r = i / CPS, cc = s * CPS + i % CPS, c = cc & 7;
      const uint32_t dst = base + ((cc >> 3) % NSLOT) * SB + r * 128 + ((c ^ (r & 7)) << 4);
      if (r < TOKS) {
        const int tok = t0 + r;
        rw_cpa(dst, x + (size_t)min(tok, m - 1) * K + k0 + cc * 8, tok < m ? 16u : 0u);
      } else if (r < TOKS + 64) {
        rw_cpa(dst, hi + (size_t)(n0 + r - TOKS) * K + k0 + cc * 8, 16u);
      } else {
        rw_cpa(dst, lo + (size_t)(n0 + r - TOKS - 64) * K + k0 + cc * 8, 16u);
      }
    }
  };
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
#pragma unroll
  for (int s = 0; s < (RING ? NBUF : NSTG); s++) {
    load(s);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  const int rows = min(TOKS, m - t0);
  int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // per-tile arrival counter (monotonic)
  int arrived = 0;
  if constexpr ((MODE & 4) != 0) {  // self-zeroing: each of the G K-split CTAs zeroes 1/G of the tile, then arrives
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    for (int i = e0 + tid; i < e1; i += 128) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    if (tid == 0) asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
  }
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    rw_cpwait(RING ? (s == 0 ? NBUF - 1 : NBUF - 2) : NSTG - 1 - s);  // stage s landed (this thread's copies)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");     // generic (cp.async) writes -> wgmma reads
    __syncthreads();
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      const uint32_t sb = base + ((q >> 2) % NSLOT) * SB + (q & 3) * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        if constexpr (!(MODE & 1)) {  // MODE 1 (ladder): loads only, no wgmma
          rw_mma<WN>(d + h * (WN / 2), dh, db);
          rw_mma<WN>(d + h * (WN / 2), dl, db);
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if constexpr (RING) {
      if (s >= 1) {  // stage s - 1's wgmmas done in every warp -> refill its slot with stage s - 1 + NBUF
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        __syncthreads();
        if (s - 1 + NBUF < NSTG) load(s - 1 + NBUF);
        asm volatile("cp.async.commit_group;\n" ::: "memory");  // possibly empty: keeps the group count uniform
        rw_keep<NA>(d);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      }
    }
  }
  if constexpr ((MODE & 4) != 0) {  // every slice of this call's tile zeroed (arrivals of a call are contiguous); polled
    if (tid == 0) {                 // while the last wgmmas run, ordered before the reds by the staging barriers
      const int G = (int)gridDim.y, target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  __syncthreads();  // every warp's wgmmas are done before the staging overwrites the tiles
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {  // D: d[4i + e] = (row 16w + g (+8 for e >= 2), token 8i + 2t (+1 for odd e))
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr (CL > 1) rw_csync(); else __syncthreads();
  if (ST && tid == 0) ts[5] = rw_gt();
  const int nr = (MODE & 2) ? 0 : rows;  // MODE 2 (ladder): no reductions
  if constexpr (CL == 1) {
    // MODE 8: rotate the row order by K split, so the G CTAs of a tile (finishing together) hit different lines
    const int rot = (MODE & 8) ? (int)blockIdx.y * (nr / (int)gridDim.y + 1) : 0;
    for (int i = tid; i < nr * 16; i += 128) {
      int r = (i >> 4) + rot;
      r = (MODE & 8) ? r % nr : r;
      const int c4 = i & 15;
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
    }
  } else {
    uint32_t rk;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    uint32_t rem[CL];
#pragma unroll
    for (int q = 0; q < CL; q++) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rem[q]) : "r"(base), "r"(q));
    const int mine = nr > (int)rk ? (nr - (int)rk + CL - 1) / CL : 0;  // token rows rk, rk + CL, ...
    for (int i = tid; i < mine * 16; i += 128) {
      const int r = (int)rk + CL * (i >> 4), c4 = i & 15;
      const uint32_t off = (uint32_t)(r * SP + 4 * c4) * 4u;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < CL; q++) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rem[q] + off) : "memory");
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
      }
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, a);
    }
    rw_csync();  // no CTA leaves while a peer may still read its staging
  }
  if constexpr (ST) {
    __syncthreads();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t sm; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(sm));
      dbg[7] = sm;
    }
  }
#endif
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  if (N % 64 || K % KR || m < 1 || (K / KR) % CL || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = CL > 1 ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, x, hi, lo, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_occ(int m, int N, int K) {  // max co-resident clusters (CL > 1) or CTAs per SM (CL == 1)
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  int n = -1;
  if (CL == 1) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, 128, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
// ---- pw29m: rtk = warp-specialised TMA router. Warps 0-3: the consumer warpgroup (wgmma, self-zeroing, epilogue);
// warp 4 lane 0: producer issuing per 64-k stage three 2D TMA boxes (x {64, TOKS}, hi {64, 64}, lo {64, 64}, 128B
// swizzle, x rows >= m zero-filled) into an NBUF-slot ring guarded by full / empty mbarriers. Slot layout, descriptors,
// wgmma and epilogue are rwk's. MODE: 4 self-zeroing, 16 stamps (as rwk).
struct alignas(64) RtMap { unsigned long long v[16]; };
__device__ __forceinline__ void rt_mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void rt_mbwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRT_WAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RT_WAIT_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rt_mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rt_mbexpect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rt_tile(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ void rt_tile_mc(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void rt_rarrive(uint32_t bar, uint32_t rank) {  // arrive on the same mbarrier in CTA `rank`
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(r) : "memory");
}
__device__ __forceinline__ void rt_cbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }  // consumer warpgroup only
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
__global__ void __launch_bounds__((MODE & 4) ? 192 : 160, MINB) rtk(const __grid_constant__ RtMap tx, const __grid_constant__ RtMap th,
                                                 const __grid_constant__ RtMap tl, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();
#else
  constexpr int NSTG = KR / 64, ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68;
  constexpr int NSLOT = NBUF < NSTG ? NBUF : NSTG;
  static_assert(KR % 64 == 0 && TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB && SB % 1024 == 0, "rtk");
  constexpr bool ST = (MODE & 16) != 0;
  constexpr bool MC = (MODE & 32) != 0;  // cluster of the 4 n-tiles: each CTA multicasts a quarter of the x box to all 4
  static_assert(!MC || TOKS % 32 == 0, "mc");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  const uint32_t bars = base + NSLOT * SB;  // full[NSLOT], empty[NSLOT]
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  if ((MODE & 64) && tid == 128) {  // MODE 64: fetch the three tensor maps before the barrier setup
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&th) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tl) : "memory");
  }
  if (tid == 0) {
    for (int i = 0; i < NSLOT; i++) { rt_mbinit(bars + 8 * i, 1); rt_mbinit(bars + 8 * (NSLOT + i), MC ? 16 : 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if constexpr (MC) rw_csync(); else __syncthreads();  // MC: every CTA's barriers exist before any multicast / remote arrive
  const int rows = min(TOKS, m - t0);
  if (tid >= 160) {  // MODE 4: the zeroing warp (off the consumers' critical path): zero 1/G of the tile, arrive with
    int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // release, poll the call's G arrivals,
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    const int ln = tid - 160;                                                // then meet the consumers before the reds
    for (int i = e0 + ln; i < e1; i += 32) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncwarp();
    if (ln == 0) {
      int arrived;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
      const int target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
    asm volatile("bar.sync 2, 160;\n" ::: "memory");
    return;
  }
  if (tid >= 128) {  // producer warp
    if (tid == 128) {
      for (int s = 0; s < NSTG; s++) {
        const int sl = s % NSLOT;
        if (s >= NSLOT) rt_mbwait(bars + 8 * (NSLOT + sl), (uint32_t)((s / NSLOT - 1) & 1));
        const uint32_t fb = bars + 8 * sl, dst = base + sl * SB;
        rt_mbexpect(fb, (uint32_t)SB);
        if constexpr (MC) {
          uint32_t rk;
          asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
          rt_tile_mc(dst + rk * (TOKS / 4) * 128, &tx, k0 + 64 * s, t0 + (int)rk * (TOKS / 4), fb, (uint16_t)0xF);
        } else {
          rt_tile(dst, &tx, k0 + 64 * s, t0, fb);
        }
        rt_tile(dst + TOKS * 128, &th, k0 + 64 * s, n0, fb);
        rt_tile(dst + (TOKS + 64) * 128, &tl, k0 + 64 * s, n0, fb);
      }
    }
    return;
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int s = 0; s < NSTG; s++) {
    const int sl = s % NSLOT;
    rt_mbwait(bars + 8 * sl, (uint32_t)((s / NSLOT) & 1));
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint32_t sb = base + sl * SB + q * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        rw_mma<WN>(d + h * (WN / 2), dh, db);
        rw_mma<WN>(d + h * (WN / 2), dl, db);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (s >= 1) {
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");  // stage s - 1 retired in this warp
      rw_keep<NA>(d);
      if ((tid & 31) == 0 && s - 1 + NSLOT < NSTG) {
        if constexpr (MC) {
#pragma unroll
          for (uint32_t q = 0; q < 4; q++) rt_rarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT), q);
        } else {
          rt_mbarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT));
        }
      }
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  rt_cbar();
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr ((MODE & 4) != 0) asm volatile("bar.sync 2, 160;\n" ::: "memory");  // staged + this call's tile zeroed
  else rt_cbar();
  if (ST && tid == 0) ts[5] = rw_gt();
  for (int i = tid; i < rows * 16; i += 128) {
    const int r = i >> 4, c4 = i & 15;
    rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
  }
  if constexpr (ST) {
    rt_cbar();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t smid; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
      dbg[7] = smid;
    }
  }
#endif
}
typedef CUresult (*RtEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                               const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                               CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RtMap), "CUtensorMap size");
static int rt_map(RtMap& tm, const void* p, int rows, int K, int box_rows) {
  static RtEncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RtEncodeFn)fn;
  }
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  if (N % 64 || K % KR || m < 1 || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  static bool once = false;
  static int slots = 0;  // co-resident CTAs (self-zeroing waits across the K splits of a tile: never launch a 2-wave grid)
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    int per = 0, dev = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kern, (MODE & 4) ? 192 : 160, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    slots = per * sms;
    once = true;
  }
  if ((MODE & 4) && (MODE & 32)) return -31;  // multicast clusters + self-zeroing: co-residency not guaranteed
  if ((MODE & 4) && (N / 64) * (K / KR) * ((m + TOKS - 1) / TOKS) > slots) return -30;
  constexpr bool MC = (MODE & 32) != 0;
  RtMap tx, th, tl;
  int r = rt_map(tx, x, m, K, MC ? TOKS / 4 : TOKS);
  if (!r) r = rt_map(th, hi, N, K, 64);
  if (!r) r = rt_map(tl, lo, N, K, 64);
  if (r) return r;
  if (MC && N / 64 != 4) return -4;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3((MODE & 4) ? 192 : 160, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 4; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = MC ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, tx, th, tl, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_occ(int m, int N, int K) {  // co-resident clusters of 4 (MODE 32) or CTAs per SM
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  const int nt = (MODE & 4) ? 192 : 160;
  int n = -1;
  if (!(MODE & 32)) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, nt, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(nt, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 4; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
int64_t rwocc(int64_t v, int64_t m, int64_t N, int64_t K) {
  switch (v) {
@@RTOCC@@
@@RWOCC@@
  }
  return -2;
}
int64_t rwrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0) || !x.is_contiguous() || !out.is_contiguous()) return -10;
  if (((uintptr_t)out.data_ptr() & 15) || ((uintptr_t)x.data_ptr() & 15)) return -11;
  if (cnt.numel() < 128 * 64 * 2) return -12;
  if (v >= 20000 && cnt.numel() < 16384 + 2 * 8 * 1024) return -13;
  const __nv_bfloat16 *xp = (const __nv_bfloat16*)x.data_ptr(), *h = (const __nv_bfloat16*)hi.data_ptr(), *l = (const __nv_bfloat16*)lo.data_ptr();
  float* o = out.data_ptr<float>();
  int* c = cnt.data_ptr<int>();
  const int m = x.size(0), N = out.size(1), K = x.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (v) {
@@RWCASES@@
@@RTCASES@@
  }
  return -2;
}
"""
RWFLAGS = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-DNDEBUG"]
# variant -> rwk<KR, NSTG, NBUF, TOKS, WN, MINB>
RWV = {100: (256, 4, 4, 128, 128, 1), 101: (256, 8, 8, 128, 128, 1), 102: (128, 4, 4, 128, 128, 2), 103: (512, 8, 6, 128, 128, 1),
       110: (256, 4, 4, 256, 256, 1), 111: (256, 8, 8, 256, 256, 1), 112: (128, 4, 4, 256, 256, 2), 113: (256, 8, 8, 256, 128, 1),
       114: (512, 8, 4, 256, 256, 1)}
RWV = {k: v + (0, 1) for k, v in RWV.items()}  # + (MODE, CL)
RWV.update({104: (256, 4, 3, 128, 128, 2, 0, 1), 105: (256, 4, 2, 128, 128, 2, 0, 1), 106: (256, 4, 2, 128, 128, 2, 4, 1),
            120: RWV[100][:6] + (4, 1), 121: RWV[100][:6] + (0, 2), 123: RWV[100][:6] + (4, 2), 126: RWV[101][:6] + (4, 1),
            127: RWV[102][:6] + (4, 1), 107: (256, 4, 3, 128, 128, 2, 4, 1), 140: (256, 4, 3, 128, 128, 2, 4, 2),
            141: (256, 4, 3, 128, 128, 2, 0, 2), 142: RWV[101][:6] + (4, 2),
            150: RWV[101][:6] + (12, 1), 151: (256, 4, 3, 128, 128, 2, 12, 1), 152: (256, 4, 4, 128, 128, 1, 12, 1),
            130: RWV[113][:6] + (4, 1), 131: RWV[113][:6] + (0, 2), 136: RWV[112][:6] + (4, 1)})
RWV.update({108: (512, 8, 3, 128, 128, 2, 0, 1)})
RWST = (101, 104, 107, 103, 108, 113, 130)  # MODE 16 (globaltimer stamps) instances: variant + 20000
RWV.update({v + 20000: RWV[v][:6] + (RWV[v][6] | 16, RWV[v][7]) for v in RWST})
# pw29m rtk variants: (KR, NBUF, TOKS, WN, MINB, MODE)
RTV = {200: (512, 6, 128, 128, 1, 4), 201: (256, 3, 128, 128, 2, 4), 202: (256, 4, 256, 128, 1, 4), 203: (512, 4, 256, 128, 1, 4),
       204: (1024, 6, 128, 128, 1, 4), 206: (512, 6, 128, 128, 1, 0), 210: (256, 4, 128, 128, 1, 4), 211: (256, 3, 128, 128, 2, 4),
       230: (512, 6, 128, 128, 1, 32), 231: (256, 3, 128, 128, 2, 32), 232: (256, 4, 128, 128, 1, 32),
       240: (512, 6, 128, 128, 1, 68), 241: (256, 3, 128, 128, 2, 68), 242: (256, 4, 128, 128, 1, 68)}
# 220-223 (multicast + self-zeroing) deadlock when not every cluster is co-resident (the zeroing wait spans clusters): not built
RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in list(RTV.items())})
RWLAD = True  # pw29h ladder instances: variant + 1000 * MODE (1 = loads only, 2 = no reds, 3 = loads only, no reds)
if RWLAD:
    RWV.update({k + 1000 * md: RWV[k][:6] + (md, 1) for k in (101, 104) for md in (1, 2, 3)})


def _rwwg(n):
    """the SS wgmma m64nNk16 (bf16 in, fp32 accumulate in place) as an inline-asm device function"""
    r = n // 2
    regs = ", ".join("%%%d" % i for i in range(r))
    outs = ", ".join('"+f"(d[%d])' % i for i in range(r))
    return ('__device__ __forceinline__ void rw_mma%d(float* d, uint64_t a, uint64_t b) {\n'
            '#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n'
            '  asm volatile("wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, 1, 1, 1, 0, 0;\\n"\n'
            '    : %s : "l"(a), "l"(b));\n'
            '#endif\n}\n') % (n, n, regs, r, r + 1, outs)


def _rwsrc():
    use = {v: c for v, c in RWV.items() if RWALL or v in {p[1] for p in RSP}}
    ruse = {v: c for v, c in RTV.items() if RWALL or v in {p[1] for p in RSP}}
    rtc = "\n".join("    case %d: return rt_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), 64 + i) for i, (v, c) in enumerate(sorted(RTV.items())) if v in ruse)
    cases = "\n".join("    case %d: return rw_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), i) for i, (v, c) in enumerate(sorted(RWV.items())) if v in use)
    occ = "\n".join("    case %d: return rw_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(use.items()))
    return RWCU.replace("@@WG@@", _rwwg(128) + _rwwg(256)).replace("@@RWCASES@@", cases).replace("@@RWOCC@@", occ).replace("@@RTCASES@@", rtc).replace("@@RTOCC@@", "\n".join("    case %d: return rt_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(ruse.items())))


_rwmod = []


def _rwext():
    if not _rwmod:
        import zlib
        t = time.time()
        src = _rwsrc()
        import os
        arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"  # pw29h: torch otherwise adds compute_90 and ptxas rejects wgmma there
        try:
            _rwmod.append(load_inline("rw29_%x" % zlib.crc32(src.encode()), cpp_sources="int64_t rwrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rwocc(int64_t, int64_t, int64_t, int64_t);",
                                      cuda_sources=src, functions=["rwrun", "rwocc"], extra_cuda_cflags=RWFLAGS, verbose=False))
        finally:
            if arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        print("PREP rw compiled in %.0fs (at %.0fs)" % (time.time() - t, time.time() - _T0), flush=True)
    return _rwmod[0]


# pw22/pw23: router m = 128 / 256 split-K kernels. (name, variant, zero-fill node, served m). rs64k32 = pw22's winner
# (rsk BN 64 x KR 256, one wait), kept as the in-run reference. pw23 (rsk2): the K slice in NSTG commit groups issued up
# front, stage s multiplied once it lands; rp64k32s4 / s2 = rs64k32's geometry with 4 / 2 stages; rp64k64x2 = BN 64 x
# KR 128, 64 KB, 2 CTAs/SM (4 x 64 = 256 CTAs); rp64k32sc = s4 with the self-cleaning reduction (red.add into a
# persistent zeroed workspace, the CTA completing a tile copies it out and re-zeroes it: no memset node). m = 256: one
# CTA holds both 128-token blocks (TOKS 256: every W fragment feeds 4 m16 tiles): rv64k32 (192 KB, 128 CTAs), rv64k64x2
# (96 KB, 256 CTAs at 2/SM), rv64k32sc (self-cleaning).
RSPALL = [("rs64k32", 0, True, (128, 256)), ("rp64k32s4", 5, True, (128,)), ("rp64k32s2", 6, True, (128,)),
       ("rp64k64x2", 7, True, (128,)), ("rp64k32sc", 8, False, (128,)),
       ("rv64k32", 9, True, (256,)), ("rv64k64x2", 10, True, (256,)), ("rv64k32sc", 11, False, (256,)),
       # pw29 (rwk, wgmma; variant >= 100 = RWV): rw64k32s4 / s8 = 128 CTAs, 4 / 8 stages; rw64k64s4x2 = KR 128, 256
       # CTAs at 2/SM (also m=256 as two 128-token blocks, 512 CTAs); rw64k16r6 = KR 512 (64 CTAs), ring of 6 slots;
       # rwv* = both 128-token blocks in one CTA (n256, or h2 = two n128 sets), rwv64k16r4 = 64 CTAs, ring of 4
       ("rw64k32s4", 100, True, (128,)), ("rw64k32s8", 101, True, (128, 256)), ("rw64k64s4x2", 102, True, (128, 256)),
       ("rw64k16r6", 103, True, (128,)), ("rwv64k32s4", 110, True, (256,)), ("rwv64k32s8", 111, True, (256,)),
       ("rwv64k64s4x2", 112, True, (256,)), ("rwv64k32s8h2", 113, True, (256,)), ("rwv64k16r4", 114, True, (256,)),
       # pw29h: z = self-zeroing (no memset node), cN = cluster of N K-splits pre-reducing through DSMEM
       ("rw64k32s4z", 120, False, (128,)), ("rw64k32s4c2", 121, True, (128,)), ("rw64k32s4zc2", 123, False, (128,)),
       ("rw64k32s8z", 126, False, (128, 256)), ("rw64k64s4x2z", 127, False, (128, 256)),
       ("rw64k32r3x2", 104, True, (128, 256)), ("rw64k32r2x2", 105, True, (128, 256)), ("rw64k32r2x2z", 106, False, (128, 256)),
       ("rw64k32r3x2z", 107, False, (128, 256)), ("rw64k16r3x2", 108, True, (128, 256)), ("rw64k32r3x2zc2", 140, False, (128, 256)), ("rw64k32r3x2c2", 141, True, (128, 256)), ("rw64k32s8zc2", 142, False, (128,)),
       ("rw64k32s8zr", 150, False, (128, 256)), ("rw64k32r3x2zr", 151, False, (128, 256)), ("rw64k32s4zr", 152, False, (128,)),
       ("rt64k16b6z", 200, False, (128, 256)), ("rt64k32b3x2z", 201, False, (128, 256)), ("rtv64k32b4z", 202, False, (256,)),
       ("rtv64k16b4z", 203, False, (256,)), ("rt64k8b6z", 204, False, (128, 256)), ("rt64k16b6", 206, True, (128, 256)),
       ("rt64k32b4z", 210, False, (128, 256)), ("rt64k32b3x2bz", 211, False, (128,)),
       ("rt64k16b6m", 230, True, (128, 256)), ("rt64k32b3x2m", 231, True, (128, 256)), ("rt64k32b4m", 232, True, (128,)),
       ("rt64k16b6zp", 240, False, (128, 256)), ("rt64k32b3x2zp", 241, False, (128, 256)), ("rt64k32b4zp", 242, False, (128,)),
       ("rwv64k32s8h2z", 130, False, (256,)), ("rwv64k32s8h2c2", 131, True, (256,)), ("rwv64k64s4x2z", 136, False, (256,))]

# pw29h shipped bake set (H100 rbench, judge-like read flush): pw23's rp64k32s4 as the in-run incumbent, pw29's
# rw64k32s8 as the wgmma reference, the self-zeroing winners; every self-zeroing grid is one co-resident wave.
RSP = [p for p in RSPALL if p[0] in ("rw64k32s8z", "rw64k32r3x2z", "rt64k32b4z", "rt64k32b3x2bz", "rt64k32b3x2z", "rt64k16b6z")]
# pw35: m = 128: rw64k32s8z (the judge's current pick) vs the rtk paths; m = 256: rw64k32r3x2z vs rtk, all against r6 (F1).
# One co-resident wave per self-zeroing grid (rt_go also declines otherwise).
RSP = [(nm, v, zf, {"rw64k32s8z": (128,), "rt64k32b4z": (128,), "rt64k32b3x2bz": (128,), "rw64k32r3x2z": (256,),
                    "rt64k16b6z": (256,)}.get(nm, ms)) for nm, v, zf, ms in RSP]
RWALL = False  # rbench sets True: build every RWV instance, not only the shipped ones


P1ALT = True
PR27 = False  # P1: the rd8 ladder was read on the H100 (audits/p1-h100.md)


def _pr27(st):
    """PROBE rd8 (pw27, no paths): the floor of an 8 MB read with every load issued at launch, in the bake's timing
    (cold-L2 graph replay, in-prepare): ld.global 16 B x NL per thread at G = 128 / 132 / 264 / 528 CTAs x 256 threads;
    + 64 fp32 red.adds per CTA into a memset-zeroed 8 KB output (memset in the graph); + a cluster-8 DSMEM reduce;
    cp.async.bulk 8 / 16 KB chunks per CTA. 'PROBE rd8 <name> <us> <tb/s> tbs'."""
    dev = st["pooler_raw"].device
    ext = _gvext()
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink32 = torch.empty((), dtype=torch.float32, device=dev)
    nb = 8 << 20
    buf = torch.randint(0, 255, (nb,), dtype=torch.uint8, device=dev)
    out = torch.zeros((2048,), dtype=torch.float32, device=dev)
    sink = torch.zeros((1024,), dtype=torch.int32, device=dev)
    x0 = torch.empty((1, 1), device=dev)
    zero = lambda: out.zero_()
    for name, kind, G, z in (("ldg-g128-nl16", 0, 128, False), ("ldg-g132-nl16", 0, 132, False), ("ldg-g264-nl8", 1, 264, False),
                             ("ldg-g528-nl4", 2, 528, False), ("ldg-g1056-nl2", 3, 1056, False), ("ldg-g264-red", 4, 264, True),
                             ("ldg-g264-c8", 5, 264, False), ("blk8k-g128", 6, 128, False), ("blk16k-g128", 7, 128, False),
                             ("blk8k-g256", 6, 256, False)):
        try:
            def fn(xx, oo, kind=kind, G=G, z=z):
                if z:
                    zero()
                return ext.prrun(kind, G, buf, out, sink)
            code = fn(x0, out)
            if code:
                print("PROBE rd8 %s launch-error %d" % (name, code), flush=True)
                continue
            us = _time_us(fn, x0, out, flush, sink32)
            print("PROBE rd8 %s %.2f %.2f tbs" % (name, us, nb / us / 1e6), flush=True)
        except Exception as error:
            print("PROBE rd8 %s ERROR %s" % (name, str(error).splitlines()[-1][:120]), flush=True)
    print("PROBE rd8 empty-memset %.2f" % _time_us(lambda xx, oo: zero(), x0, out, flush, sink32), flush=True)


def _rscall(st, p, x, out):
    if p[2]:
        out.zero_()  # a memset node in the captured graph (red.add accumulates into it)
    if p[1] >= 100:  # pw29 wgmma router (own sm_90a extension); 16-B cp.async sources: never let it decline at serve time
        return _rwext().rwrun(p[1], x if x.data_ptr() % 16 == 0 else x.clone(), st["rs_hi"], st["rs_lo"], out, st["rw_cnt"])
    return _gvext().rsrun(p[1], x, st["rs_hi"], st["rs_lo"], out, st["rs_ws"], st["rs_cnt"])  # ws / cnt: rsk2 RED 2 only


def _rsbake(st):
    dev = st["router_raw"].device
    st["rs_hi"], st["rs_lo"] = st["router_hl"][:256].contiguous(), st["router_hl"][256:].contiguous()
    st["rs_ws"] = torch.zeros((256 * 256,), dtype=torch.float32, device=dev)  # RED 2: [<= 256 tokens][256] fp32, zero between calls
    st["rs_cnt"] = torch.zeros((64,), dtype=torch.int32, device=dev)
    st["rw_cnt"] = torch.zeros((128 * 64 * 2,), dtype=torch.int32, device=dev)  # pw29h self-zeroing: monotonic, never reset
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(4242)
    w = st["router_raw"]
    rexact = st["rs_hi"].float() + st["rs_lo"].float()
    n, K = w.shape
    rw = False
    # F1 baseline rule (pw29m): r6 timed before the heavy work (the rw build) and again after the variants; the lower wins
    r6pre = {}
    for m in (128, 256):
        xx = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        oo = torch.empty((m, n), dtype=torch.float32, device=dev)
        r6pre[m] = _time_us(lambda a, b: _r6(a, "router", n, st), xx, oo, flush, sink)
    if time.time() - _T0 < PREPARE_BUDGET_S - 60:  # pw29: build the wgmma router extension; a failure drops only its paths
        try:
            _rwext()
            rw = True
        except Exception as error:
            print("PREP rw build failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    else:
        print("PREP rw skipped-budget", flush=True)
    for m in (128, 256):
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)
        r6 = lambda xx, oo: _r6(xx, "router", n, st)
        best = ("r6", None, _time_us(r6, x, out, flush, sink))
        r6base0 = best[2]
        print("BAKE router m=%d r6 %.2f" % (m, best[2]), flush=True)
        for name, v, zf, ms in RSP:
            if m not in ms or (v >= 100 and not rw):
                continue
            if time.time() - _T0 > PREPARE_BUDGET_S:
                print("BAKE router m=%d %s skipped-budget" % (m, name), flush=True)
                continue
            plan = (-7, v, zf, None, n, None, None)
            fn = lambda xx, oo, plan=plan: _rscall(st, plan, xx, oo)
            try:
                if fn(x, out) != 0:
                    print("BAKE router m=%d %s ERROR launch" % (m, name), flush=True)
                    continue
                ok, msg = _validate(fn, w, m, n, True, gen, rexact, {128: 121, 256: 250}[m])
                if not ok:
                    print("BAKE router m=%d %s INVALID %s" % (m, name, msg), flush=True)
                    continue
                us = _time_us(fn, x, out, flush, sink)
            except Exception as error:
                print("BAKE router m=%d %s ERROR %s" % (m, name, str(error).splitlines()[-1][:160]), flush=True)
                continue
            print("BAKE router m=%d %s %.2f" % (m, name, us), flush=True)
            if us < best[2] * MARGIN:
                best = (name, plan, us)
        r6post = _time_us(r6, x, out, flush, sink)
        r6base = min(r6pre[m], r6post, (best[2] if best[1] is None else r6base0))
        print("BAKE router m=%d r6 F1 pre %.2f post %.2f base %.2f" % (m, r6pre[m], r6post, r6base), flush=True)
        if best[1] is not None and not best[2] < r6base * MARGIN:
            best = ("r6", None, r6base)
        if m == 128:  # PROBE rszero: the cost of the zeroing memset node alone (on the critical path of the red variants)
            print("PROBE rszero router m=128 %.2f" % _time_us(lambda xx, oo: oo.zero_(), x, out, flush, sink), flush=True)
        if best[1] is not None:
            st["sel"][("router", m)] = best[1]
        print("PICK router m=%d %s %.2f" % (m, best[0], best[2]), flush=True)
    # critic (pw23): the self-cleaning workspace/counters are shared by every RED-2 path; a declined or failed path
    # may have left them dirty, so restore the all-zero invariant before any picked path serves the judge.
    st["rs_ws"].zero_(); st["rs_cnt"].zero_(); torch.cuda.synchronize()


def _gxladder(st):
    """PROBE gvlad: where r6's gv time goes as m grows (pw14). Each step is timed with the bake's _time_us."""
    ext = _gvext()
    dev = st["qkv_gate_raw"].device
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(4321)
    packs = dict(st.get("gvc12", {}))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for layer in ("pooler", "qkv_gate"):
        for m in (1, 8, 16, 32):
            if time.time() - _T0 > PREPARE_BUDGET_S:
                print("PROBE gvlad %s m=%d skipped-budget" % (layer, m), flush=True)
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            for ly, mt, geo, v, step in GLAD:
                if ly != layer or GVB[m] != mt:
                    continue
                try:
                    inst = _gxinst(geo, v, m)
                    plan = _gxplan(st, layer, inst, n, packs)
                    if plan is None:
                        print("PROBE gvlad %s m=%d %s declined" % (layer, m, step), flush=True)
                        continue
                    a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                    if _gvcall(st, plan, x, out) != 0:
                        print("PROBE gvlad %s m=%d %s launch-error" % (layer, m, step), flush=True)
                        continue
                    us = _time_us(lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo), x, out, flush, sink)
                    print("PROBE gvlad %s m=%d %s %.2f regs=%d ctas/sm=%.2f" % (layer, m, step, us, int(ext.gvregs(plan[0], *a)), int(ext.gvocc(plan[0], *a)) / sms), flush=True)
                except Exception as error:
                    print("PROBE gvlad %s m=%d %s ERROR %s" % (layer, m, step, str(error).splitlines()[-1][:120]), flush=True)
            us = _time_us(lambda xx, oo: _r6(xx, layer, n, st), x, out, flush, sink)
            p = st["gv"].get((layer, _bk(m)))
            info = ""
            if p is not None and p[0] < 1000:
                a = (p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])
                info = " %s regs=%d ctas/sm=%.2f" % (_name(p[0]), int(ext.gvregs(p[0], *a)), int(ext.gvocc(p[0], *a)) / sms)
            print("PROBE gvlad %s m=%d L4 %.2f%s" % (layer, m, us, info or " v12"), flush=True)


def kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    if "cpu" in state:
        w = state["raw"][layer]
        return x.float() @ w[:n].float().t() if layer == "router" else x @ w[:n].t()
    rp = state.get("rq", {}).get((layer, x.shape[0]))  # rs128: m exactly 128 / 256, picked in the bake
    if rp is not None and rp["n"] == n:
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        if _rqcall(state, rp, x.contiguous(), out) == 0:
            return out
    p = state["sel"].get((layer, _bk(x.shape[0])))
    if p is not None and p[4] == n:
        out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        _gvcall(state, p, x.contiguous(), out)
        return out
    return _r6(x, layer, n, state)


def _r6(x, layer, n, state):
    """r6-c12's kernel_fn, unchanged"""
    p = state["gv"].get((layer, _bk(x.shape[0])))
    if p is not None and p[4] == n:
        out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        _gvcall(state, p, x.contiguous(), out)
        return out
    return _v12(x, layer, n, state)


def _v12(x, layer, n, state):
    m = x.shape[0]
    if _cublas(layer, m):  # cuBLAS beats the skinny kernel here (judge v7: 0.45-0.9x); algorithm tuned in prepare()
        i = state["lt"].get((layer, m))
        if i is None:
            return x @ state[layer + "_raw"][:n].T
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _ext().lt_mm(x.contiguous(), state[layer + "_raw"], state[layer + "_t"], out, n, i)
        return out
    if layer == "router" and m > 32:
        i = state["lt"].get(("router", m))
        if i is None:
            a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
        else:
            a = torch.empty((m, 2 * n), dtype=torch.float32, device=x.device)
            _ext().lt_mm(x.contiguous(), state["router_hl"], state["router_hl_t"], a, 2 * n, i)
        return a[:, :n] + a[:, n:]
    A, A2 = state[layer]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if layer == "router" and m <= 32 and RTR:
        _ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
        return out
    (_ext().run2 if V6 and m <= 32 else _ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
    return out


# ================================================================ RS128 (audits/m128-plan.md)
# swap-AB register-A split-K kernel for m = 128 / 256: CTA = 128 W rows x NT tokens x K/KS; two math warpgroups (W rows
# 0-63 / 64-127 of the block) decode the lossless 12-bit pack (pw10 _pack_nat + _c12 fragments, re-blocked to 128-row
# blocks: one 12,288 B cp.async.bulk per k64 stage) into wgmma A registers; x is the smem B operand (2D TMA, K-major
# SW128); producer warpgroup with a W lane (warp 8) and an x lane (warp 9) on split rings (W slot released at the LDS,
# x slot after the wait_group that covers its wgmmas); setmaxnreg 40 / 232; partials reduced across the KS CTAs of a row
# block by cp.reduce.async.bulk.tensor bf16 add into a zeroed out (memset node first); stmatrix.trans epilogue; no cluster.
# PRB rungs (no epilogue unless 0): 1 W only | 2 W + x (waits / arrives only) | 3 + LDS + decode into a sink |
# 4 + wgmma (full mainloop) | 5 bare W stream (producer self-paced, no consumer) | 6 bare W + x stream |
# 7 epilogue only (no mainloop: D = constant; memset + reduce-add cost) | 8 decode + wgmma on resident slot 0, no TMA
# and no barriers (SM-side rate) | 10 = 8 with constant A (no LDS / decode).
RQV = [  # (name, NT, ST, GS, PRB, WM, INS, FL): ST k64 slots (W 12 KB + x NT*128 B each), GS k64 stages per consumer
    # round, WM 0 wait_group 1 + lagged x release / 1 decode-next under the wgmma + wait_group 0, FL bit 0 L2 hints,
    # bit 1 in-kernel zeroing (no memset node)
    ("f12", 128, 8, 1, 5, 0, 0, 0),
    ("f12x", 128, 8, 1, 6, 0, 0, 0),
    ("s0", 128, 8, 1, 1, 0, 0, 0),
    ("s1", 128, 8, 1, 2, 0, 0, 0),
    ("s2", 128, 8, 1, 3, 0, 0, 0),
    ("s3", 128, 8, 1, 4, 0, 0, 0),
    ("s3g2", 128, 8, 2, 4, 0, 0, 0),
    ("s3m1", 128, 8, 1, 4, 1, 0, 0),
    ("s3m2", 128, 8, 1, 4, 2, 0, 0),
    ("s3m2s7", 128, 7, 1, 4, 2, 0, 0),
    ("s3s6", 128, 6, 1, 4, 0, 0, 0),
    ("s3s5", 128, 5, 1, 4, 0, 0, 0),
    ("s3s4", 128, 4, 1, 4, 0, 0, 0),
    ("s3s6h", 128, 6, 1, 4, 0, 0, 1),
    ("s3s5h", 128, 5, 1, 4, 0, 0, 1),
    ("s3s6g2", 128, 6, 2, 4, 0, 0, 0),
    ("s1s6", 128, 6, 1, 2, 0, 0, 0),
    ("s1s5", 128, 5, 1, 2, 0, 0, 0),
    ("s4s6z", 128, 6, 1, 0, 0, 0, 2),
    ("s4s5z", 128, 5, 1, 0, 0, 0, 2),
    ("s4s6hz", 128, 6, 1, 0, 0, 0, 3),
    ("s4s5hz", 128, 5, 1, 0, 0, 0, 3),
    ("s4s5hzi", 128, 5, 1, 0, 0, 1, 3),
    ("s4hz", 128, 8, 1, 0, 0, 0, 3),
    ("n256s5hz", 256, 5, 1, 0, 0, 0, 3),
    ("n256s4hz", 256, 4, 1, 0, 0, 0, 3),
    ("n256s5z", 256, 5, 1, 0, 0, 0, 2),
    ("s3s7", 128, 7, 1, 4, 0, 0, 0),
    ("s3h", 128, 8, 1, 4, 0, 0, 1),
    ("s3m2h", 128, 8, 1, 4, 2, 0, 1),
    ("s4m2z", 128, 8, 1, 0, 2, 0, 3),
    ("r8m2", 128, 8, 1, 8, 2, 0, 0),
    ("s3m2i", 128, 8, 1, 4, 2, 1, 0),
    ("s3g2m1", 128, 8, 2, 4, 1, 0, 0),
    ("s4", 128, 8, 1, 0, 0, 0, 1),
    ("s4m1", 128, 8, 1, 0, 1, 0, 1),
    ("s4z", 128, 8, 1, 0, 0, 0, 3),
    ("s4m1z", 128, 8, 1, 0, 1, 0, 3),
    ("s4g2m1z", 128, 8, 2, 0, 1, 0, 3),
    ("ep", 128, 8, 1, 7, 0, 0, 0),
    ("epst", 128, 8, 1, 7, 0, 0, 4),
    ("ep0", 128, 8, 1, 7, 0, 0, 8),
    ("s4st", 128, 8, 1, 0, 0, 0, 5),
    ("s4zi", 128, 8, 1, 0, 0, 1, 3),
    ("r8", 128, 8, 1, 8, 0, 0, 0),
    ("r8m1", 128, 8, 1, 8, 1, 0, 0),
    ("s3i", 128, 8, 1, 4, 0, 1, 0),
    ("s3m1i", 128, 8, 1, 4, 1, 1, 0),
    ("s4m1zi", 128, 8, 1, 0, 1, 1, 3),
    # rs32: NT 32 / 16 (m exactly 32 / 16) and HALF (FL bit 4: 64 W rows, one math warpgroup; bit 2 = plain store, KS 1)
    ("t32s5z", 32, 5, 1, 0, 0, 0, 2),
    ("t32s5hz", 32, 5, 1, 0, 0, 0, 3),
    ("t32s8z", 32, 8, 1, 0, 0, 0, 2),
    ("t32s8hz", 32, 8, 1, 0, 0, 0, 3),
    ("t32s12z", 32, 12, 1, 0, 0, 0, 2),
    ("t32s12hz", 32, 12, 1, 0, 0, 0, 3),
    ("t32g2s8z", 32, 8, 2, 0, 0, 0, 2),
    ("t32m2s8z", 32, 8, 1, 0, 2, 0, 2),
    ("t32s8zi", 32, 8, 1, 0, 0, 1, 2),
    ("h32s8", 32, 8, 1, 0, 0, 0, 20),
    ("h32s8z", 32, 8, 1, 0, 0, 0, 18),
    ("h32s12", 32, 12, 1, 0, 0, 0, 20),
    ("h32s12z", 32, 12, 1, 0, 0, 0, 18),
    ("t16s5z", 16, 5, 1, 0, 0, 0, 2),
    ("t16s5hz", 16, 5, 1, 0, 0, 0, 3),
    ("t16s8z", 16, 8, 1, 0, 0, 0, 2),
    ("t16s8hz", 16, 8, 1, 0, 0, 0, 3),
    ("t16s12z", 16, 12, 1, 0, 0, 0, 2),
    ("t16s12hz", 16, 12, 1, 0, 0, 0, 3),
    ("t16g2s8z", 16, 8, 2, 0, 0, 0, 2),
    ("t16m2s8z", 16, 8, 1, 0, 2, 0, 2),
    ("t16s8zi", 16, 8, 1, 0, 0, 1, 2),
    ("h16s8", 16, 8, 1, 0, 0, 0, 20),
    ("h16s8z", 16, 8, 1, 0, 0, 0, 18),
    ("h16s12", 16, 12, 1, 0, 0, 0, 20),
    ("h16s12z", 16, 12, 1, 0, 0, 0, 18),
    ("t32s3z", 32, 3, 1, 0, 0, 0, 2),
    ("t32s3hz", 32, 3, 1, 0, 0, 0, 3),
    ("t32s4z", 32, 4, 1, 0, 0, 0, 2),
    ("t32s4hz", 32, 4, 1, 0, 0, 0, 3),
    ("t32s5hzi", 32, 5, 1, 0, 0, 1, 3),
    ("t16s3z", 16, 3, 1, 0, 0, 0, 2),
    ("t16s3hz", 16, 3, 1, 0, 0, 0, 3),
    ("t16s4z", 16, 4, 1, 0, 0, 0, 2),
    ("t16s4hz", 16, 4, 1, 0, 0, 0, 3),
    ("t16s5hzi", 16, 5, 1, 0, 0, 1, 3),
    ("t32s5hze", 32, 5, 1, 0, 0, 0, 35),
    ("t32s5ze", 32, 5, 1, 0, 0, 0, 34),
    ("t32s6hze", 32, 6, 1, 0, 0, 0, 35),
    ("t32s6hz", 32, 6, 1, 0, 0, 0, 3),
    ("t32s5hzei", 32, 5, 1, 0, 0, 1, 35),
    ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
    ("t16s5ze", 16, 5, 1, 0, 0, 0, 34),
    ("t16s6hze", 16, 6, 1, 0, 0, 0, 35),
    ("t16s6hz", 16, 6, 1, 0, 0, 0, 3),
    ("t16s5hzei", 16, 5, 1, 0, 0, 1, 35),
    ("h128s8e", 128, 8, 1, 0, 0, 0, 52),
    ("s4s5hze", 128, 5, 1, 0, 0, 0, 35),
    ("h128s5", 128, 5, 1, 0, 0, 0, 20),
    ("h128s5z", 128, 5, 1, 0, 0, 0, 18),
    ("h128s8", 128, 8, 1, 0, 0, 0, 20),
    ("h128s8z", 128, 8, 1, 0, 0, 0, 18),
    ("s4s5zEH", 128, 5, 1, 0, 0, 0, 65538),
    ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),
]
RQI = {v[0]: i for i, v in enumerate(RQV)}
RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # judge 1e-2 (rows 5x)

RQCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int SMEM = 1024 + ST * (XB + WB) + 24 * ST + 16;
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
__global__ void __launch_bounds__(384, 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && (FL == 65538 || FL == 65539)), "EH backport approved M128/ST5/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  uint32_t ec = warp < NW ? __ldg(el + lane) : 0u;
  if (tid == (EARLY ? 256 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  }
  if (tid == 320 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  if (warp >= 8) {  // ---------------- producer warpgroup: W lane (warp 8), x lane (warp 9)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == 8 && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;  // EARLY: stages 0 .. ST-1 already issued (slot 0 of round 1 next)
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s++) {
        if (s >= ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, WB);
        if (HINT) rq_bulkh(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > ST ? KCS - ST : 0); s < KCS; s++) rq_wait(full + 8 * (s % ST), (s / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == 9 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;  // EARLY: stages 0 .. ST-1 already issued (slot 0 of round 1 next)
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s++) {
        if (s >= ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : xempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, XB);
        if (HINT) rq_tileh(xring + slot * XB, &tx, (s0 + s) * 64, 0, full + 8 * slot, pol);
        else rq_tile(xring + slot * XB, &tx, (s0 + s) * 64, 0, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > ST ? KCS - ST : 0); s < KCS; s++) rq_wait(full + 8 * (s % ST), (s / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= 10) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - 320, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : s % ST;
        if (!RES) rq_wait(full + 8 * slot, (s / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(wempty + 8 * ((GS * G + j) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(xempty + 8 * ((GS * G + j) % ST));
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : s % ST) * XB) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""


def _rqwg(n):
    """rq_wgmma<n>: wgmma.mma_async m64n<n>k16 f32 += bf16 (A: 4 registers) x bf16 (B: smem descriptor), scale-d from a
    register (0 on the CTA's first k16: no accumulator zero-fill), scale-a = scale-b = 1, trans-b = 0"""
    nd = n // 2
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
            % (n, nd, nd + 5, n, d, a, nd + 4, outs))


def _rqss(n):
    nd = n // 2
    d = ", ".join("%%%d" % i for i in range(nd))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("template <> __device__ __forceinline__ void rq_wgss<%d>(float (&D)[%d], uint64_t da, uint64_t db, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
             "    : %s\n    : \"l\"(da), \"l\"(db), \"r\"(scd));\n}\n")
            % (n, nd, nd + 2, n, d, nd, nd + 1, outs))


def _rqsrc(vs):
    cases = "\n".join("    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);" % (i, ", ".join(str(a) for a in v[1:8]))
                      for i, v in enumerate(vs))
    return RQCU.replace("@@WGMMA@@", _rqwg(16) + _rqwg(32) + _rqwg(128) + _rqwg(256) + _rqss(128) + _rqss(256)).replace("@@CASES@@", cases)


RQFLAGS = ["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"]
_rqmod = [None, None]
_rqmod2 = [None, None]  # rs32: the m = 16 / 32 instances, a second extension built on its own thread


def _rqext(vs=None, verbose=False, slot=None):
    vs = RQV if vs is None else vs
    src = _rqsrc(vs)
    slot = _rqmod if slot is None else slot
    if slot[0] is None or slot[1] != src:
        import os
        tag = ("rq128_%x" if slot is _rqmod else "rq32_%x") % (hash(src) & 0xffffffff)
        flags = RQFLAGS + (["-Xptxas=-v"] if verbose else [])
        slot[0] = load_inline(tag + ("v" if verbose else ""), cpp_sources="int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                                cuda_sources=src, functions=["rqrun"], extra_cuda_cflags=flags, verbose=verbose)
        slot[1] = src
    return slot[0]


def _pack_nat(w):
    """pw10 _pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


def _rqpack(w):
    """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
    N, K = w.shape
    c = _c12(_pack_nat(w), 1)
    if c is None or N % 128:
        return None
    data, tab, esc = c
    T, KC = data.shape[:2]
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    return lay, tab, esc


def _rqunpack(lay, tab, esc, N, K):
    """independent check of the re-blocking: unpack [NB][KC][8][1536] with _unc12 and undo _pack_nat -> bf16 [N, K]"""
    NB, KC = lay.shape[:2]
    data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
    v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
    W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
    return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)


def _rqesc(esc6, NB, KC, KS, dev):
    """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
    G = KS * NB * 8
    if len(esc6):
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        gw = (ks * NB + nb) * 8 + wt
        o = torch.argsort(gw * 65536 + u)
        gw, ent = gw[o], ent[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
    else:
        cap = 32
    arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
    if len(esc6):
        arr[gw * cap + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)


RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
# the extension the submission builds: the box's favoured instances plus the judge's pick (s4s5z: no L2 hints won on the
# H200 at all three m=128 shapes, job 846e25d9df09433f); <= 6 timed paths per layer
RQPROD = ["s4s5z", "s4s5hz", "s4hz", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]
RQBAKE = {128: ["s4s5z", "s4s5hz", "s4hz", "s4s5zEH", "s4s5hzEH"], 256: ["n256s4hz", "n256s5hz", "n256s5z"]}
RQLAYERS = ("pooler", "qkv_gate", "unpooler", "o")
RQ = True
_rqthr = []


def _rqstart(prod=None, slot=None, thr=None, tag="rq"):
    """compile the RS128 extension in a background thread while r6's phase runs (CPU-only work; never fatal)"""
    import threading
    prod = RQPROD if prod is None else prod
    thr = _rqthr if thr is None else thr

    def build():
        try:
            _rqext([RQV[RQI[nm]] for nm in prod], slot=slot)
            thr.append("ok")
            thr.append(time.time() - _T0)
        except Exception as error:
            thr.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
    t = threading.Thread(target=build, daemon=True)
    t.start()
    print("PREP %s thread started at %.0fs" % (tag, time.time() - _T0), flush=True)
    return t


def _rqcall(st, p, x, out):
    return p.get("mod", _rqmod[0]).rqrun(p["i"], p["lay"], x, out, p["esc"], p["cnt"], p["KS"], p["tab"], 0, st["rq_e"], st["rq_sink"], 0)


def _rqval(fn, w, m, n, gen, tol):
    """one-hot x8 rows bit-exact against W over every K column (the split-K bf16 reduce-add must keep them exact),
    random rows within the KS-dependent tolerance (judge: 1e-2, rows 5x)"""
    K = w.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
    exact = w[:n].t()
    if K not in _EYE:
        _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
    eye = _EYE[K]
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
        if fn(x.contiguous(), out) != 0:
            return False, "launch error"
        got = out[: min(m, K - start)]
        want = exact[start:start + got.shape[0]] * 8.0
        if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
            return False, "one-hot mismatch at column %d" % start
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=w.device)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=w.device)[: K // 100]] *= dist  # the task's 'outliers'
        x = x.to(torch.bfloat16)
        o = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        fn(x, o)
        ref = x.double() @ w[:n].double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e > %.0e" % (err, row, tol)
    return True, "ok %.1e" % err


def _rqbake(st, thr):
    """RS128 bake: per (layer, m in 128 / 256) time r6's own path and each RS128 variant (graph replays from a cold L2,
    the judge's method), pick a variant only if it validates and beats r6 by MARGIN; kernel_fn serves picks at m exactly"""
    st["rq"] = {}
    if time.time() - _T0 > PREPARE_BUDGET_S - 45:
        print("BAKE rq skipped-budget %.0fs" % (time.time() - _T0), flush=True)
        return
    thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 45 - (time.time() - _T0)))
    if not _rqthr or _rqthr[0] != "ok" or _rqmod[0] is None:
        print("BAKE rq build %s" % (_rqthr[0] if _rqthr else "not finished"), flush=True)
        return
    print("PREP rq thread ready at %.0fs (bake starts at %.0fs)" % (_rqthr[1], time.time() - _T0), flush=True)
    names = [v[0] for v in [RQV[RQI[nm]] for nm in RQPROD]]
    dev = st["pooler_raw"].device
    st["rq_e"] = torch.empty(0, dtype=torch.int64, device=dev)
    st["rq_sink"] = torch.zeros(4, dtype=torch.int32, device=dev)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(12828)
    for layer in RQLAYERS:
        if time.time() - _T0 > PREPARE_BUDGET_S - 30:
            print("BAKE rq %s skipped-budget" % layer, flush=True)
            continue
        w = st[layer + "_raw"]
        n, K = w.shape
        xs, r6a = {}, {}
        for m in (128, 256):  # r6 timed before the pack (critic F1: the first r6 timing after the pack read ~4 us high)
            xs[m] = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            o6 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            r6a[m] = _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[m], o6, flush, sink)
        pk = _rqpack(w)
        if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
            print("BAKE rq %s pack not exact" % layer, flush=True)
            continue
        lay, tab, esc6 = pk
        st.setdefault("rq_pk", {})[layer] = pk  # rs32: the small-m phase (after gx) reuses the pack
        KS = RQKS[layer]
        esc, _ = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)
        cnt = torch.zeros(2 * lay.shape[0], dtype=torch.int32, device=dev)
        # discarded warm-up: the first path timed after the pack reads ~3 us high (box: r6 31.3 vs 27.4, and an rq path
        # 27.8 vs its 24-25), which would bias whichever path is timed first
        _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[128], torch.empty((128, n), dtype=torch.bfloat16, device=dev), flush, sink, it=7)
        keep = False
        for m in (128, 256):
            x = xs[m]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            cands = []
            for nm in RQBAKE[m]:
                if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                    print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, nm, time.time() - _T0), flush=True)
                    continue
                p = {"i": names.index(nm), "lay": lay, "esc": esc, "cnt": cnt, "KS": KS, "tab": tab, "n": n}
                fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                try:
                    r = fn(x, out)
                    torch.cuda.synchronize()
                    if r != 0:
                        print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, nm, r), flush=True)
                        continue
                    ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                    torch.cuda.synchronize()
                    if int(cnt.abs().sum()):  # the zero handshake must leave its counters at 0 after every launch
                        cnt.zero_()
                        ok, msg = False, "counters not reset"
                    if not ok:
                        print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, nm, msg), flush=True)
                        continue
                    us = _time_us(fn, x, out, flush, sink)
                except Exception as error:
                    print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, nm, str(error).splitlines()[-1][:160]), flush=True)
                    continue
                print("BAKE %s m=%d rq:%s %.2f (%s) at %.0fs" % (layer, m, nm, us, msg, time.time() - _T0), flush=True)
                cands.append((us, nm, p))
            r6b = _time_us(lambda xx, oo: _r6(xx, layer, n, st), x, out, flush, sink)  # r6 re-timed after the variants
            best = ("r6", None, min(r6a[m], r6b))
            print("BAKE %s m=%d r6 %.2f (before pack %.2f, after variants %.2f) at %.0fs" % (layer, m, best[2], r6a[m], r6b, time.time() - _T0), flush=True)
            for us, nm, p in sorted(cands, key=lambda c: c[0]):
                if us < best[2] * MARGIN:
                    best = ("rq:" + nm, p, us)
                break
            if best[1] is not None:
                st["rq"][(layer, m)] = best[1]
                keep = True
            print("PICK %s m=%d %s %.2f" % (layer, m, best[0], best[2]), flush=True)
        if not keep:
            del lay, esc, cnt
    torch.cuda.synchronize()


# ---------------------------------------------------------------- rs32: rq at m exactly 32 / 16 (NT = m)
# <= 4 timed paths per (layer, m): (variant, KS). Baked after gx against the incumbent kernel_fn would serve without rq
# (st["sel"] if gx picked the shape, else r6's gv / cuBLAS choice); F1 baseline rule: one discarded warm-up timing, the
# incumbent timed before and after the variants, the lower kept, both printed.
RQS = True
RQSBAKE = {  # box lab t1-t3: ST 5-6 best at both NT (3/4 and 8/12 slower); hints (h) and EARLY (e) -0.2..-0.3; qkv KS 6
    # (108 CTAs) -0.1..-0.3 vs KS 7; o: HALF (64 rows, KS 1, EARLY) only helps at m = 128 (h128s8e 8.54 vs cuBLAS 8.90)
    ("qkv_gate", 32): [("t32s5hze", 6), ("t32s5hze", 7), ("t32s5hz", 6), ("t32s6hze", 6)],
    ("qkv_gate", 16): [("t16s5hze", 6), ("t16s5ze", 6), ("t16s5hze", 7), ("t16s5ze", 7)],
    ("o", 32): [("t32s5hze", 2), ("t32s6hze", 2), ("t32s5ze", 2)],
    ("o", 16): [("t16s5hze", 2), ("t16s6hze", 2), ("t16s5ze", 2)],
    ("o", 128): [("h128s8e", 1), ("h128s5", 1)],
    ("pooler", 32): [("t32s5hze", 4), ("t32s6hze", 4), ("t32s5hz", 4)],
    ("pooler", 16): [("t16s5hze", 4), ("t16s6hze", 4), ("t16s5hz", 4)],
    ("unpooler", 32): [("t32s5hze", 2), ("t32s6hze", 2), ("t32s6hz", 2)],
    ("unpooler", 16): [("t16s5hze", 2), ("t16s6hze", 2), ("t16s5hz", 2)],
}
RQPROD2 = list(dict.fromkeys(nm for ps in RQSBAKE.values() for nm, _ in ps))
_rqthr2 = []


def _rqsbake(st, thr):
    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
        print("BAKE rq32 skipped-budget %.0fs" % (time.time() - _T0), flush=True)
        return
    thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 30 - (time.time() - _T0)))
    if not _rqthr2 or _rqthr2[0] != "ok" or _rqmod2[0] is None:
        print("BAKE rq32 build %s" % (_rqthr2[0] if _rqthr2 else "not finished"), flush=True)
        return
    print("PREP rq32 thread ready at %.0fs" % _rqthr2[1], flush=True)
    names = RQPROD2
    dev = st["pooler_raw"].device
    st.setdefault("rq_e", torch.empty(0, dtype=torch.int64, device=dev))
    st.setdefault("rq_sink", torch.zeros(4, dtype=torch.int32, device=dev))
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(3216)
    print("PREP rq32 bake starts at %.0fs" % (time.time() - _T0), flush=True)
    for layer in RQLAYERS:
        if time.time() - _T0 > PREPARE_BUDGET_S - 30:
            print("BAKE rq32 %s skipped-budget" % layer, flush=True)
            continue
        w = st[layer + "_raw"]
        n, K = w.shape
        pk = st.get("rq_pk", {}).get(layer)
        if pk is None:
            pk = _rqpack(w)
            if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                print("BAKE rq32 %s pack not exact" % layer, flush=True)
                continue
        lay, tab, esc6 = pk
        escs = {}
        cnt = torch.zeros(4 * lay.shape[0], dtype=torch.int32, device=dev)
        for m in (32, 16, 128):
            ps = RQSBAKE.get((layer, m), [])
            if not ps:
                continue
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            sp = st["sel"].get((layer, _bk(m)))
            rp0 = st["rq"].get((layer, m))  # m = 128: an rq pick of the m=128 phase is the incumbent
            if rp0 is not None:
                inc, iname = (lambda xx, oo, rp0=rp0: _rqcall(st, rp0, xx, oo)), "rq128"
            elif sp is not None and sp[4] == n:
                inc, iname = (lambda xx, oo, sp=sp: _gvcall(st, sp, xx, oo)), "sel"
            else:
                inc, iname = (lambda xx, oo: _r6(xx, layer, n, st)), "r6"
            _time_us(inc, x, out, flush, sink, it=5)  # discarded warm-up
            ia = _time_us(inc, x, out, flush, sink)
            cands = []
            for nm, KS in ps[:4]:
                if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                    print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, nm, time.time() - _T0), flush=True)
                    continue
                if KS not in escs:
                    escs[KS] = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)[0]
                p = {"i": names.index(nm), "lay": lay, "esc": escs[KS], "cnt": cnt, "KS": KS, "tab": tab, "n": n, "mod": _rqmod2[0]}
                fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                tag = "%s-k%d" % (nm, KS)
                try:
                    r = fn(x, out)
                    torch.cuda.synchronize()
                    if r != 0:
                        print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, tag, r), flush=True)
                        continue
                    ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                    torch.cuda.synchronize()
                    if int(cnt.abs().sum()):
                        cnt.zero_()
                        ok, msg = False, "counters not reset"
                    if not ok:
                        print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, tag, msg), flush=True)
                        continue
                    us = _time_us(fn, x, out, flush, sink)
                except Exception as error:
                    print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, tag, str(error).splitlines()[-1][:160]), flush=True)
                    continue
                print("BAKE %s m=%d rq:%s %.2f (%s) at %.0fs" % (layer, m, tag, us, msg, time.time() - _T0), flush=True)
                cands.append((us, tag, p))
            ib = _time_us(inc, x, out, flush, sink)
            best = (iname, None, min(ia, ib))
            print("BAKE %s m=%d %s %.2f (before variants %.2f, after variants %.2f) at %.0fs" % (layer, m, iname, best[2], ia, ib, time.time() - _T0), flush=True)
            for us, tag, p in sorted(cands, key=lambda c: c[0]):
                if us < best[2] * MARGIN:
                    best = ("rq:" + tag, p, us)
                break
            if best[1] is not None:
                st["rq"][(layer, m)] = best[1]
            print("PICK %s m=%d %s %.2f" % (layer, m, best[0], best[2]), flush=True)
