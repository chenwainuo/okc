
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdint.h>
__device__ __forceinline__ void rw_mma128(float* d, uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));
#endif
}
__device__ __forceinline__ void rw_mma256(float* d, uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b));
#endif
}

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
    case 200: return rt_occ<512, 6, 128, 128, 1, 4>(m, N, K);
    case 201: return rt_occ<256, 3, 128, 128, 2, 4>(m, N, K);
    case 210: return rt_occ<256, 4, 128, 128, 1, 4>(m, N, K);
    case 211: return rt_occ<256, 3, 128, 128, 2, 4>(m, N, K);
    case 107: return rw_occ<256, 4, 3, 128, 128, 2, 4, 1>(m, N, K);
    case 126: return rw_occ<256, 8, 8, 128, 128, 1, 4, 1>(m, N, K);
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
    case 107: return rw_go<256, 4, 3, 128, 128, 2, 4, 1>(xp, h, l, o, m, N, K, c, 7, st);
    case 126: return rw_go<256, 8, 8, 128, 128, 1, 4, 1>(xp, h, l, o, m, N, K, c, 17, st);
    case 200: return rt_go<512, 6, 128, 128, 1, 4>(xp, h, l, o, m, N, K, c, 64, st);
    case 201: return rt_go<256, 3, 128, 128, 2, 4>(xp, h, l, o, m, N, K, c, 65, st);
    case 210: return rt_go<256, 4, 128, 128, 1, 4>(xp, h, l, o, m, N, K, c, 70, st);
    case 211: return rt_go<256, 3, 128, 128, 2, 4>(xp, h, l, o, m, N, K, c, 71, st);
  }
  return -2;
}
