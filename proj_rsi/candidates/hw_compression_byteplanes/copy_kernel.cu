#include <cuda_runtime.h>
#include <stdint.h>

static constexpr uint32_t kVectors = 67108864u / 16u;
static constexpr int kBlocks = 128;
static constexpr int kThreads = 256;

extern "C" __global__ __launch_bounds__(kThreads)
void okc_hwc_copy64m(const uint4* __restrict__ src, uint4* __restrict__ dst) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < kVectors; i += kBlocks * kThreads) {
    uint32_t a, b, c, d;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "l"(src + i) : "memory");
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(dst + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
}

extern "C" cudaError_t okc_hwc_launch(const void* src, void* dst, cudaStream_t stream) {
  okc_hwc_copy64m<<<kBlocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(src), static_cast<uint4*>(dst));
  return cudaGetLastError();
}

extern "C" cudaError_t okc_hwc_attributes(cudaFuncAttributes* a, int* occupancy) {
  cudaError_t e = cudaFuncGetAttributes(a, okc_hwc_copy64m);
  if (e != cudaSuccess) return e;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, okc_hwc_copy64m, kThreads, 0);
}
