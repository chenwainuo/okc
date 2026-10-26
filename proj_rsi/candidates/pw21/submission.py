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
against r6 on pooler m=1 and m=16, alternated."""
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
  const int bz = blockIdx.x, bx = blockIdx.y, by = blockIdx.z, GX = gridDim.y;
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
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
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

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st, int occ) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
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
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  const bool clu = S > 1 && cc >= 9 && !(V & 4);  // pw11: V bit 2 = global last-arriver reduction, no cluster
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
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R>
__global__ void __launch_bounds__(512) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[16][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  float4 wv[R][4];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < 4; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * 512); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * 512);
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
    for (int i = 0; i < 16; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
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
// pw21 PROBE l2scratch only: decode r6's 12-bit pack (dec1) of row tiles [t0, t0 + nt) into natural bf16 rows
// [nt*16][K] (escapes not applied: a timing probe, the output feeds no path). One warp per (tile, 64-k chunk) unit:
// row g gets (A0, A2, A4, A6) at k 16t .. 16t+7 and (A8, A10, A12, A14) at 16t+8 .., row g+8 the odd registers.
__global__ void __launch_bounds__(256) gvdec(const uint4* W, __nv_bfloat16* out, uint32_t Tb, int KC, int t0, int nt) {
  const int lane = threadIdx.x & 31, wi = blockIdx.x * 8 + (threadIdx.x >> 5), g = lane >> 2, t = lane & 3;
  if (wi >= nt * KC) return;
  const int tl = wi / KC, c = wi - tl * KC;
  const uint4* q = W + ((size_t)(t0 + tl) * KC + c) * 96 + lane;
  const uint4 q0 = gnc(q), q1 = gnc(q + 32), q2 = gnc(q + 64);
  uint32_t A[16];
  dec1(q0.x, q0.y, q2.x, Tb, 0, A); dec1(q0.z, q0.w, q2.y, Tb, 0, A + 4);
  dec1(q1.x, q1.y, q2.z, Tb, 0, A + 8); dec1(q1.z, q1.w, q2.w, Tb, 0, A + 12);
  const size_t K = (size_t)KC * 64;
  uint4* r0 = (uint4*)(out + (size_t)(tl * 16 + g) * K + c * 64 + t * 16);
  uint4* r1 = (uint4*)(out + (size_t)(tl * 16 + g + 8) * K + c * 64 + t * 16);
  r0[0] = make_uint4(A[0], A[2], A[4], A[6]); r0[1] = make_uint4(A[8], A[10], A[12], A[14]);
  r1[0] = make_uint4(A[1], A[3], A[5], A[7]); r1[1] = make_uint4(A[9], A[11], A[13], A[15]);
}
int64_t gvdecrun(torch::Tensor W, torch::Tensor out, int64_t Tb, int64_t t0, int64_t nt) {
  const int KC = (int)W.size(1);
  const int units = (int)nt * KC;
  gvdec<<<(units + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)W.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (uint32_t)(Tb & 0xffffffff), KC, (int)t0, (int)nt);
  return (int64_t)cudaGetLastError();
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
    ("qkv_gate", 1): [((1, 8, 16, 1, 1, 0), _1), ((1, 4, 16, 2, 1, 0), _01), ((1, 4, 8, 4, 1, 0), _1)],
    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), _1), ((1, 4, 8, 4, 1, 2), (1, 3)), ((1, 4, 16, 2, 1, 2), _01), ((1, 4, 16, 2, 1, 2, 1), _1), ((1, 4, 8, 4, 1, 2, 1), _1)],
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

# pw21: router at m = 128 / 256 on r6's gv structure (V bit 25: token-tile loop, x at use through L1, MT 16 = 128 tokens
# per CTA, m = 256 as two token blocks in grid.x), plain BF16 hi + lo fragments (FMT 2) exactly as r6's router at m <= 32
# (no decode; the 12-bit FMT 3 variant spills at every register cap and is not built). Split-K: S CTAs along K reduced by
# r6's cluster DSMEM epilogue, or V bit 2 = pw11's global last-arriver (no cluster). The 8th geometry field is the W
# register-ring depth ST; V bits 12-14 = __launch_bounds__ minBlocks (ptxas: 1 -> 255 regs, 3 -> 168 regs, 0 spills).
_TL = 1 << 25
GRP = [("rt-c16", "router", (128, 256), (2, 2, 2, 4, 16, 1, 2, 1), _TL | 1 << 12),      # 8 x 16 CTAs x 128 thr, cluster 16
       ("rt-c16s2", "router", (128, 256), (2, 2, 2, 4, 16, 1, 2, 2), _TL | 3 << 12),    # + W ring depth 2, 3 CTAs/SM
       ("rt-c8", "router", (128, 256), (2, 2, 4, 4, 8, 1, 2, 1), _TL | 1 << 12),        # 8 x 8 CTAs x 256 thr, cluster 8
       ("rt-g16s2", "router", (128, 256), (2, 2, 2, 4, 16, 1, 2, 2), _TL | 3 << 12 | 4)]  # global last-arriver, no cluster
GXP += GRP

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
_gvmod = None


def _gvext():
    global _gvmod
    if _gvmod is None:
        cases = "\n".join("    case %d: return gvl<%s>(p, st, occ);" % (i, ",".join(map(str, c))) for i, c in enumerate(GVI))
        _gvmod = load_inline("gv20_%x" % (hash(tuple(GVI)) & 0xffffffff), cpp_sources="int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t gvdecrun(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                             cuda_sources=GVCU.replace("@@CASES@@", cases), functions=["gvrun", "gvocc", "gvregs", "rrrun", "gvdecrun"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
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
    mm = (1 if plan[0] < 1002 else 8) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
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


RRB = {1: (0, 1)}  # fp32 router kernels per bucket (m8 rr: 7.7-14.9 us on the H200, v12 4.7)


def _name(i):
    return "rr%d" % (i - 1000) if i >= 1000 else ".".join(map(str, GVI[i]))


def _gvcall(st, p, x, out):
    if p[0] >= 1000:
        return _gvext().rrrun(p[0] - 1000, p[1], x, out)
    return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])


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
                for v in vs if c12 else (0,):
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
    if GV and GX:
        try:
            _gxbake(st)
        except Exception as error:  # the second bake can only add paths; r6's choices stand on any failure
            st["sel"] = {}
            print("PREP gx bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    if GV and GX and time.time() - _T0 < PREPARE_BUDGET_S - 20:
        try:
            _l2probe(st)
        except Exception as error:
            print("PROBE l2scratch declined %s" % str(error).splitlines()[-1][:160].replace(" ", "_"), flush=True)
        finally:  # critic (pw21): never leave a capture open or a side stream busy for the judge's own captures
            if torch.cuda.is_current_stream_capturing():
                print("PROBE l2scratch capture_left_open", flush=True)
            torch.cuda.synchronize()
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
                if inst[6] > 1 and not (v & 4) and res < grid:  # cluster variant: the whole grid must be co-resident
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


def _l2probe(st, R=512):
    """PROBE l2scratch (pw21, no path): pooler m=128 as a two-stream pipeline captured in one CUDA graph. Stream A decodes
    r6's 12-bit pack of R rows into one of two bf16 scratch chunks (R x 8192 x 2 B = 8 MB each, written just before use, so
    L2-resident); stream B runs cuBLAS on the previous chunk (torch.mm into a [chunks, m, R] stage), then one permuted
    copy writes out. Escapes are not applied (timing only). Lines: the pipeline end to end, plain cuBLAS (r6's own m=128
    path), the decode chunks alone, cuBLAS on the chunks alone."""
    dev = st["pooler_raw"].device
    if "pooler" not in st.get("gvc12", {}):
        print("PROBE l2scratch declined no_12-bit_pack", flush=True)
        return
    css, _ = st["gvc12"]["pooler"]
    data, tab = css[1][0][0], css[1][0][1]
    T, KC = data.shape[:2]
    n, K = T * 16, KC * 64
    NCH, tpc = n // R, R // 16
    ext = _gvext()
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    m = 128
    x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    buf = [torch.empty((R, K), dtype=torch.bfloat16, device=dev) for _ in range(2)]
    stage = torch.empty((NCH, m, R), dtype=torch.bfloat16, device=dev)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    ev_d = [torch.cuda.Event() for _ in range(NCH)]
    ev_m = [torch.cuda.Event() for _ in range(NCH)]

    def pipe(xx, oo):
        cur = torch.cuda.current_stream()
        sa.wait_stream(cur); sb.wait_stream(cur)
        for i in range(NCH):
            with torch.cuda.stream(sa):
                if i >= 2:
                    sa.wait_event(ev_m[i - 2])  # the chunk buffer is free once cuBLAS consumed it
                ext.gvdecrun(data, buf[i % 2], tab, i * tpc, tpc)
                ev_d[i].record(sa)
            with torch.cuda.stream(sb):
                sb.wait_event(ev_d[i])
                torch.mm(xx, buf[i % 2].t(), out=stage[i])
                ev_m[i].record(sb)
        cur.wait_stream(sa); cur.wait_stream(sb)
        oo.copy_(stage.permute(1, 0, 2).reshape(m, n))

    def dec_only(xx, oo):
        for i in range(NCH):
            ext.gvdecrun(data, buf[i % 2], tab, i * tpc, tpc)

    def mm_only(xx, oo):
        for i in range(NCH):
            torch.mm(xx, buf[i % 2].t(), out=stage[i])

    tp = _time_us(pipe, x, out, flush, sink, 21)
    tt = _time_us(lambda xx, oo: _r6(xx, "pooler", n, st), x, out, flush, sink, 21)
    td = _time_us(dec_only, x, out, flush, sink, 21)
    tm = _time_us(mm_only, x, out, flush, sink, 21)
    print("PROBE l2scratch pooler m=128 chunks=%d %.2f torch=%.2f decode=%.2f mmchunks=%.2f" % (NCH, tp, tt, td, tm), flush=True)


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
