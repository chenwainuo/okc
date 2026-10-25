// Single CTA correctness probe only. No performance claim or serving dispatch.
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

// A is row-major [output_channel=64,K=16], B [token=8,K=16].
// ALayout/CLayout return logical column-major matrix indices, NOT global
// row-major byte offsets: explicitly divide/modulo by M=64 when gathering.
__global__ void rs_bf16_tile(const Bf16* weights, const Bf16* activation, float* out) {
  __shared__ __align__(128) Bf16 sb[128];
  int tid = threadIdx.x;
  auto b = cute::make_tensor(cute::make_smem_ptr(sb), BMap{});
  int token = tid / 16, k = tid % 16;
  b(token, k) = activation[token*16+k];
  __syncthreads();
  // Generic shared stores must be visible to the asynchronous GMMA proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  alignas(16) Bf16 a[8];
  #pragma unroll
  for (int v=0; v<8; ++v) {
    int linear = AMap{}(cute::make_coord(tid,v));
    a[v] = weights[(linear % 64)*16 + linear/64];
  }
  auto* ar = reinterpret_cast<const uint32_t*>(a);
  uint64_t desc = cute::SM90::GMMA::make_gmma_desc<cute::SM90::GMMA::Major::K>(b);
  float c[4] = {0,0,0,0};
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  Op::fma(ar[0],ar[1],ar[2],ar[3],desc,c[0],c[1],c[2],c[3],
          cute::SM90::GMMA::ScaleOut::Zero);
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  #pragma unroll
  for (int v=0; v<4; ++v) {
    int linear = CMap{}(cute::make_coord(tid,v));
    out[(linear%64)*8 + linear/64] = c[v];
  }
}

static void check(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr,"CUDA failure %s: %s\n",where,cudaGetErrorString(e));
    std::exit(2);
  }
}
#define CK(expr) check((expr),#expr)

static bool cpu_layout_validation() {
  std::set<int> ai, ci, bi;
  for(int t=0;t<128;++t) {
    for(int v=0;v<8;++v) ai.insert(AMap{}(cute::make_coord(t,v)));
    for(int v=0;v<4;++v) ci.insert(CMap{}(cute::make_coord(t,v)));
  }
  for(int n=0;n<8;++n) for(int k=0;k<16;++k) bi.insert(BMap{}(cute::make_coord(n,k)));
  return ai.size()==1024 && *ai.begin()==0 && *ai.rbegin()==1023 &&
         ci.size()==512 && *ci.begin()==0 && *ci.rbegin()==511 &&
         bi.size()==128 && *bi.begin()==0 && *bi.rbegin()==127;
}

static void inputs(int trial, std::array<Bf16,1024>& a, std::array<Bf16,128>& b,
                   std::array<float,512>& ref) {
  // CPU-generated dyadic values are exactly BF16 representable; all products
  // and the 16-element sums are exactly FP32 representable. Expected error=0.
  for(int m=0;m<64;++m) for(int k=0;k<16;++k)
    a[m*16+k]=Bf16(float(((m*7+k*11+trial*13)%31)-15)/8.0f);
  for(int n=0;n<8;++n) for(int k=0;k<16;++k)
    b[n*16+k]=Bf16(float(((n*13+k*3+trial*17)%29)-14)/8.0f);
  for(int m=0;m<64;++m) for(int n=0;n<8;++n) {
    float acc=0;
    for(int k=0;k<16;++k) acc += float(a[m*16+k])*float(b[n*16+k]);
    ref[m*8+n]=acc;
  }
}

int main(int argc, char** argv) {
  bool cpu_only=argc==2 && std::strcmp(argv[1],"--cpu-only")==0;
  if(argc>1 && !cpu_only) { std::fprintf(stderr,"usage: %s [--cpu-only]\n",argv[0]); return 2; }
  if(!cpu_layout_validation()) { std::fprintf(stderr,"layout not bijective\n"); return 1; }
  std::array<Bf16,1024> a; std::array<Bf16,128> b;
  std::array<float,512> ref, out;
  double checksum=0;
  if(cpu_only) {
    for(int trial=0;trial<3;++trial) { inputs(trial,a,b,ref); for(float x:ref) checksum+=x; }
    std::printf("{\"mode\":\"cpu-only\",\"layout_bijections_pass\":true,\"reference_checksum\":%.9g,\"gpu_executed\":false}\n",checksum);
    return 0;
  }
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0) { std::fprintf(stderr,"requires SM90a Hopper\n"); return 2; }
  Bf16 *da,*db; float* dc;
  CK(cudaMalloc(&da,sizeof(a))); CK(cudaMalloc(&db,sizeof(b))); CK(cudaMalloc(&dc,sizeof(out)));
  float max_error=0;
  for(int trial=0;trial<3;++trial) {
    inputs(trial,a,b,ref);
    CK(cudaMemcpy(da,a.data(),sizeof(a),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),sizeof(b),cudaMemcpyHostToDevice));
    CK(cudaMemset(dc,0xff,sizeof(out))); // unwritten rows remain NaN and fail
    rs_bf16_tile<<<1,128>>>(da,db,dc); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(),dc,sizeof(out),cudaMemcpyDeviceToHost));
    for(int i=0;i<512;++i) {
      float err=std::abs(out[i]-ref[i]);
      if(!std::isfinite(out[i]) || err!=0) {
        std::fprintf(stderr,"mismatch trial=%d channel=%d token=%d got=%.9g ref=%.9g\n",trial,i/8,i%8,out[i],ref[i]);
        CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); return 1;
      }
      max_error=std::fmax(max_error,err);
    }
  }
  CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
  std::printf("{\"mode\":\"gpu-correctness\",\"layout_bijections_pass\":true,\"cases\":3,\"outputs_checked\":1536,\"max_abs_error\":%.9g,\"pass\":true}\n",max_error);
}
