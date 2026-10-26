#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint4 rs_ld4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 rs_ld2(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n"
               : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t rs_x(const __nv_bfloat16* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t rs_prmt(uint32_t a, uint32_t b, int c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
struct RsWords { uint4 low; uint2 code, extra; uint32_t base; bool wide; };
template<int Q> __device__ __forceinline__ uint2 rs_decode(const RsWords& v) {
  if (v.wide) {
    if constexpr (Q == 0) return make_uint2(v.low.x, v.low.y);
    if constexpr (Q == 1) return make_uint2(v.low.z, v.low.w);
    if constexpr (Q == 2) return v.code;
    if constexpr (Q == 3) return v.extra;
  }
  const uint32_t L = Q == 0 ? v.low.x : Q == 1 ? v.low.y : Q == 2 ? v.low.z : v.low.w;
  const uint32_t C = Q < 2 ? v.code.x : v.code.y;
  const uint32_t H = (((C >> ((Q & 1) * 4)) & 0x07070707u) + v.base)
                   | ((Q & 1) ? ((C << 4) & 0x80808080u) : (C & 0x80808080u));
  return make_uint2(rs_prmt(L, H, 0x5140), rs_prmt(L, H, 0x7362));
}
__device__ __forceinline__ void rs_mma(float4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
}
template<int Q, int COHORT>
__device__ __forceinline__ void rs_step(float4& d, const __nv_bfloat16* xp, uint2 hi, uint2 lo) {
  const __nv_bfloat16* p = xp + COHORT * 64 + Q * 16;
  const uint32_t a0 = rs_x(p), a1 = rs_x(p + 64), a2 = rs_x(p + 8), a3 = rs_x(p + 72);
  rs_mma(d, a0, a1, a2, a3, hi);
  rs_mma(d, a0, a1, a2, a3, lo);
}
template<int Q>
__device__ __forceinline__ void rs_both(float4& d0, float4& d1, const __nv_bfloat16* xp, const RsWords& hi, const RsWords& lo) {
  const uint2 h = rs_decode<Q>(hi), l = rs_decode<Q>(lo);
  rs_step<Q, 0>(d0, xp, h, l);
  rs_step<Q, 2>(d1, xp, h, l);
}

extern "C" __global__ __launch_bounds__(1024, 1)
void router_small_kernel(const uint8_t* packed, const uint32_t* headers, const __nv_bfloat16* x, float* out, int m) {
  __shared__ float red[32][8][2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 32 + warp;
  const uint8_t* p = packed + (size_t)gw * 2048;
  RsWords hi, lo;
  // First24B/lane independent of prepared format; request both planes before waiting on the header.
  hi.low = rs_ld4(p + lane * 16);
  hi.code = rs_ld2(p + 512 + lane * 8);
  lo.low = rs_ld4(p + 1024 + lane * 16);
  lo.code = rs_ld2(p + 1536 + lane * 8);
  const uint32_t h = __ldg(headers + gw);
  hi.base = (h & 127u) * 0x01010101u; hi.wide = (h & 128u) != 0;
  lo.base = ((h >> 8) & 127u) * 0x01010101u; lo.wide = (h & 32768u) != 0;
  hi.extra = make_uint2(0, 0); lo.extra = make_uint2(0, 0);
  if (hi.wide) hi.extra = rs_ld2(p + 768 + lane * 8);
  if (lo.wide) lo.extra = rs_ld2(p + 1792 + lane * 8);
  // PTX A row g/g+8: same token g, two distinct K groups. B columns contain four groups*two experts.
  const __nv_bfloat16* xp = x + (size_t)min(g, m - 1) * 8192 + warp * 256 + t * 2;
  float4 d0 = make_float4(0, 0, 0, 0), d1 = make_float4(0, 0, 0, 0);
  rs_both<0>(d0, d1, xp, hi, lo);
  rs_both<1>(d0, d1, xp, hi, lo);
  rs_both<2>(d0, d1, xp, hi, lo);
  rs_both<3>(d0, d1, xp, hi, lo);
  // C col2*t+expert; select A's matching K group, discard all off-diagonal products.
  const float4 d = t < 2 ? d0 : d1;
  float v0 = (t & 1) ? d.z : d.x, v1 = (t & 1) ? d.w : d.y;
  v0 += __shfl_xor_sync(0xffffffffu, v0, 1); v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
  v0 += __shfl_xor_sync(0xffffffffu, v0, 2); v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
  if (t == 0) { red[warp][g][0] = v0; red[warp][g][1] = v1; }
  __syncthreads();
  if (tid < 16 && tid / 2 < m) {
    const int tok = tid / 2, expert = tid & 1;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < 32; w++) sum += red[w][tok][expert];
    out[(size_t)tok * 256 + blockIdx.x * 2 + expert] = sum;
  }
}

#ifdef ROUTER_SMALL_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int64_t router_small_run(torch::Tensor packed, torch::Tensor headers, torch::Tensor x, torch::Tensor out) {
  if (!packed.is_cuda() || !headers.is_cuda() || !x.is_cuda() || !out.is_cuda() ||
      packed.scalar_type() != at::kByte || headers.scalar_type() != at::kUInt32 ||
      x.scalar_type() != at::kBFloat16 || out.scalar_type() != at::kFloat ||
      packed.device() != x.device() || headers.device() != x.device() || out.device() != x.device() ||
      x.dim() != 2 || out.dim() != 2 || x.size(1) != 8192 || x.size(0) < 1 || x.size(0) > 8 || !x.is_contiguous() ||
      out.size(0) != x.size(0) || out.size(1) != 256 || !out.is_contiguous() ||
      !packed.is_contiguous() || !headers.is_contiguous() || packed.numel() != 8388608 || headers.numel() != 4096 ||
      ((uintptr_t)packed.data_ptr() & 15) || ((uintptr_t)x.data_ptr() & 3)) return -1;
  router_small_kernel<<<128, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(packed.data_ptr<uint8_t>(), headers.data_ptr<uint32_t>(),
      (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), (int)x.size(0));
  return (int64_t)cudaGetLastError();
}
int64_t router_small_regs() { cudaFuncAttributes a; auto r=cudaFuncGetAttributes(&a, router_small_kernel); return r == cudaSuccess ? a.numRegs : -(int64_t)r; }
int64_t router_small_local() { cudaFuncAttributes a; auto r=cudaFuncGetAttributes(&a, router_small_kernel); return r == cudaSuccess ? a.localSizeBytes : -(int64_t)r; }
int64_t router_small_occ() { int n=0; auto r=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, router_small_kernel, 1024, 0); return r == cudaSuccess ? n : -(int64_t)r; }
#endif
