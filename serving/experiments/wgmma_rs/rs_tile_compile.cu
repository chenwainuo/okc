// Compile-only instruction/precision feasibility probe. NO launcher or serving dispatch.
// Caller must supply one warpgroup (128 threads), a valid shared-B descriptor,
// and per-lane weight/scale fragments mapped to GMMA A registers. Packing and
// descriptor construction remain unimplemented and this is NOT runnable evidence.
#include <cstdint>
#include <sgl_kernel/scalar_type.hpp>
#include "gemm/marlin/dequant.h"
#include <cute/arch/mma_sm90_gmma.hpp>

extern "C" __device__ __noinline__ void nvfp4_group16_rs_tile(
    const int* lane_weight_words, const int* lane_scale_words,
    uint64_t activation_smem_descriptor, float* lane_accumulators) {
  using namespace sglang;
  namespace marlin = sglang::device::marlin;
  nv_bfloat162 weights[4], scales[4];
  // Exact current NVFP4 specialization: skip_flop=true. Its exponent
  // compensation lives in processed scales/global-scale, not here.
  marlin::dequant<nv_bfloat162, host::kFE2M1f.id(), true>(lane_weight_words[0], weights);
  marlin::dequant<nv_bfloat162, host::kFE2M1f.id(), true>(lane_weight_words[1], weights + 2);
  marlin::dequant_fp8_scales<nv_bfloat162, host::kFE4M3fn.id()>(lane_scale_words[0], scales);
  marlin::dequant_fp8_scales<nv_bfloat162, host::kFE4M3fn.id()>(lane_scale_words[1], scales + 2);
  // Caller repeats the same processed group16 scale for each weight belonging
  // to that channel/K group. Per-element form preserves the BF16 multiply.
  #pragma unroll
  for (int i = 0; i < 4; ++i) weights[i] = __hmul2(weights[i], scales[i]);
  auto* a = reinterpret_cast<const uint32_t*>(weights);
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  cute::SM90::GMMA::MMA_64x8x16_F32BF16BF16_RS<
      cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>::fma(
      a[0], a[1], a[2], a[3], activation_smem_descriptor,
      c0, c1, c2, c3, cute::SM90::GMMA::ScaleOut::Zero);
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  lane_accumulators[0] = c0; lane_accumulators[1] = c1;
  lane_accumulators[2] = c2; lane_accumulators[3] = c3;
}
