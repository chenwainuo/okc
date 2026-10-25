#pragma once
// Isolated NVFP4 group16 correctness oracle; no serving dispatch or timing.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

using Bf16 = cute::bfloat16_t;
using Op = cute::SM90::GMMA::MMA_64x8x16_F32BF16BF16_RS<
    cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>;
using Traits = cute::MMA_Traits<Op>;
using AMap = Traits::ALayout;
using CMap = Traits::CLayout;
using BMap = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_SW32_Atom<Bf16>{}, cute::Shape<cute::_8,cute::_16>{}));
static_assert(cute::size(AMap{}) == 64*16);
static_assert(cute::size(CMap{}) == 64*8);
static_assert(cute::cosize(BMap{}) == 8*16);
constexpr bool coalesced_amap_valid() {
  for(int tid=0;tid<128;++tid) for(int v=0;v<8;++v) {
    int row=(tid/4)%8+16*(tid/32)+8*((v/2)%2);
    int k=2*(tid%4)+(v%2)+8*(v/4);
    if(AMap{}(cute::make_coord(tid,v))!=row+64*k)return false;
  }
  return true;
}
static_assert(coalesced_amap_valid(),"Packedlane map differs from pinnedCUTLASS");


#include <vector>
#include <sgl_kernel/scalar_type.hpp>
#include "gemm/marlin/dequant.h"
#include <sgl_kernel/tensor.h>
__global__ void routed_nvfp4_down(const uint32_t* packed, const uint8_t* scales,
    const Bf16* activation, int K, int N, int E, int rows,
    const int32_t* sorted, const int32_t* experts, const int32_t* post,
    int sorted_capacity, int expert_capacity, int block_m,
    const Bf16* globals, Bf16* final_out) {
  int position=int(blockIdx.y)*8;
  if(position>=sorted_capacity || position>=post[0]) return;
  int eb=position/block_m;
  if(eb>=expert_capacity) return;
  int expert=experts[eb];
  if(expert<0 || expert>=E) return;
  int channel=int(blockIdx.x)*64;
  packed+=(int64_t(expert)*(N/64)+channel/64)*(K/16)*128;
  scales+=(int64_t(expert)*(N/64)+channel/64)*(K/16)*64;
  Bf16 global_scale=globals[expert];
  namespace marlin = sglang::device::marlin;
  __shared__ __align__(128) Bf16 sb[128];
  auto b=cute::make_tensor(cute::make_smem_ptr(sb), BMap{});
  uint64_t desc=cute::SM90::GMMA::make_gmma_desc<cute::SM90::GMMA::Major::K>(b);
  int tid=threadIdx.x;
  float c[4]={0,0,0,0};
  for(int base=0;base<K;base+=16) {
    int route=sorted[position+tid/16];
    b(tid/16,tid%16)=(route>=0 && route<rows)?activation[int64_t(route)*K+base+tid%16]:Bf16(0.0f);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    // 512B packedpayload per64x16tile, contiguous4B perlane. No widening.
    uint32_t payload=packed[(base/16)*128+tid];
    int row0=((tid/4)%8)+16*(tid/32);
    uint32_t scale0=scales[(base/16)*64+row0];
    uint32_t scale1=scales[(base/16)*64+row0+8];
    uint32_t sc=scale0 | scale0<<16 | scale1<<8 | scale1<<24;
    nv_bfloat162 fs[2];
    marlin::dequant_fp8_scales<nv_bfloat162,sglang::host::kFE4M3fn.id()>(sc,fs);
    alignas(16) nv_bfloat162 a[4];
    #pragma unroll
    for(int h=0;h<2;++h) {
      uint32_t halfword=payload>>(h*16);
      uint32_t w=((halfword&0xff)<<8)|((halfword&0xff00)<<16);
      marlin::dequant<nv_bfloat162,sglang::host::kFE2M1f.id(),true>(w,a+h*2);
      a[h*2]=__hmul2(a[h*2],fs[0]);
      a[h*2+1]=__hmul2(a[h*2+1],fs[1]);
    }
    auto* ar=reinterpret_cast<uint32_t*>(a);
    // Ordering before each operation covers newly decoded register operands
    // and the accumulators carried from the prior completed group.
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    Op::fma(ar[0],ar[1],ar[2],ar[3],desc,c[0],c[1],c[2],c[3],
            base==0?cute::SM90::GMMA::ScaleOut::Zero:cute::SM90::GMMA::ScaleOut::One);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    __syncthreads(); // no overwrite of sharedB until all GMMA consumers done
  }
  #pragma unroll
  for(int v=0;v<4;++v) {
    int linear=CMap{}(cute::make_coord(tid,v));
    int route=sorted[position+linear/64];
    if(route<0 || route>=rows) continue;
    int64_t idx=int64_t(route)*N+channel+linear%64;
    // Preserve incumbent cast-before-global-scale ordering, BF16 arithmetic.
    nv_bfloat16 rounded=__float2bfloat16_rn(c[v]);
    nv_bfloat16 gs=*reinterpret_cast<nv_bfloat16*>(&global_scale);
    nv_bfloat16 result=__hmul(rounded,gs);
    final_out[idx]=*reinterpret_cast<Bf16*>(&result);
  }
}


struct RoutedDownKernel {
  static void run(tvm::ffi::TensorView packed,tvm::ffi::TensorView scales,
      tvm::ffi::TensorView activation,tvm::ffi::TensorView globals,
      tvm::ffi::TensorView sorted,tvm::ffi::TensorView experts,
      tvm::ffi::TensorView post,tvm::ffi::TensorView output,int64_t block_m) {
    using namespace sglang::host;
    auto E=SymbolicSize{"experts"},NT=SymbolicSize{"output_tiles"},Q=SymbolicSize{"k_groups"};
    auto R=SymbolicSize{"rows"},S=SymbolicSize{"sorted_capacity"},T=SymbolicSize{"expert_capacity"};
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({E,NT,Q,128}).with_dtype<int32_t>().with_device(device).verify(packed);
    int64_t k=Q.unwrap()*16,n=NT.unwrap()*64;
    RuntimeCheck(k==4096 && n%64==0,"requires K4096,N multiple64");
    RuntimeCheck(block_m>=8 && block_m%8==0,"requires alignment block divisible8");
    TensorMatcher({E,NT,Q,64}).with_dtype<uint8_t>().with_device(device).verify(scales);
    TensorMatcher({R,k}).with_dtype<nv_bfloat16>().with_device(device).verify(activation);
    TensorMatcher({E}).with_dtype<nv_bfloat16>().with_device(device).verify(globals);
    TensorMatcher({R,n}).with_dtype<nv_bfloat16>().with_device(device).verify(output);
    TensorMatcher({S}).with_dtype<int32_t>().with_device(device).verify(sorted);
    TensorMatcher({T}).with_dtype<int32_t>().with_device(device).verify(experts);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(post);
    RuntimeCheck(S.unwrap()>=block_m && S.unwrap()/block_m<=T.unwrap(),"invalid metadata capacities");
    RuntimeCheck(R.unwrap()>0 && R.unwrap()<2147483647 && S.unwrap()<2147483647,"row capacity unsupported");
    // Python zeros output on the same stream before this launch: nonlocal
    // expert slots must remain zero for the existing finalizer.
    LaunchKernel(dim3(n/64,S.unwrap()/8),128,device.unwrap())(
      routed_nvfp4_down,static_cast<const uint32_t*>(packed.data_ptr()),
      static_cast<const uint8_t*>(scales.data_ptr()),static_cast<const Bf16*>(activation.data_ptr()),
      int(k),int(n),int(E.unwrap()),int(R.unwrap()),
      static_cast<const int32_t*>(sorted.data_ptr()),static_cast<const int32_t*>(experts.data_ptr()),
      static_cast<const int32_t*>(post.data_ptr()),int(S.unwrap()),int(T.unwrap()),int(block_m),
      static_cast<const Bf16*>(globals.data_ptr()),static_cast<Bf16*>(output.data_ptr()));
  }
};
