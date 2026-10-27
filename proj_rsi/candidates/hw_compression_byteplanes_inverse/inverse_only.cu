// CPU microcompile only: exactly the proposed per-lane32BF16 inverse; no GEMM/host launch.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t bp_prmt(uint32_t lo, uint32_t hi, uint32_t s) {
  uint32_t out; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(lo), "r"(hi), "r"(s)); return out;
}

__device__ __forceinline__ void bp8(uint32_t l0, uint32_t l1, uint32_t h0, uint32_t h1, uint32_t* a) {
  a[0] = bp_prmt(l0, h0, 0x5140); a[1] = bp_prmt(l0, h0, 0x7362);
  a[2] = bp_prmt(l1, h1, 0x5140); a[3] = bp_prmt(l1, h1, 0x7362);
}

extern "C" __global__ __launch_bounds__(256)
void hwc_inverse32_only(const uint4* input, uint4* output) {
  const unsigned t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const unsigned base = warp * 128 + lane;
  const uint4 l01 = input[base], l23 = input[base + 32];
  const uint4 h01 = input[base + 64], h23 = input[base + 96];
  uint32_t a[16];
  bp8(l01.x, l01.y, h01.x, h01.y, a);
  bp8(l01.z, l01.w, h01.z, h01.w, a + 4);
  bp8(l23.x, l23.y, h23.x, h23.y, a + 8);
  bp8(l23.z, l23.w, h23.z, h23.w, a + 12);
#pragma unroll
  for (int q = 0; q < 4; ++q)
    output[t * 4 + q] = make_uint4(a[4*q], a[4*q+1], a[4*q+2], a[4*q+3]);
}
