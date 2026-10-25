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

#include <vector>
#include <sgl_kernel/scalar_type.hpp>
#include "gemm/marlin/dequant.h"
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


// Canonical packed [64,K/2] nibble layout, processed E4M3 [64,K/16].
// These are not Marlin's checkpoint interleave. Mapping to this lossless
// representation is a future model-load transformation, not quantization.
__global__ void rs_nvfp4_tile(const uint8_t* packed, const uint8_t* scales,
    const Bf16* activation, int K, Bf16 global_scale,
    float* accum_out, Bf16* final_out, Bf16* decoded_out) {
  namespace marlin = sglang::device::marlin;
  __shared__ __align__(128) Bf16 sb[128];
  auto b=cute::make_tensor(cute::make_smem_ptr(sb), BMap{});
  uint64_t desc=cute::SM90::GMMA::make_gmma_desc<cute::SM90::GMMA::Major::K>(b);
  int tid=threadIdx.x;
  float c[4]={0,0,0,0};
  for(int base=0;base<K;base+=16) {
    b(tid/16,tid%16)=activation[(tid/16)*K+base+tid%16];
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint32_t q[8],s[8];
    int indices[8];
    #pragma unroll
    for(int v=0;v<8;++v) {
      int linear=AMap{}(cute::make_coord(tid,v));
      int m=linear%64, k=base+linear/64;
      indices[v]=m*K+k;
      uint8_t byte=packed[(m*K+k)/2];
      q[v]=(byte>>((k%2)*4))&15;
      s[v]=scales[m*(K/16)+base/16];
    }
    alignas(16) nv_bfloat162 a[4];
    #pragma unroll
    for(int h=0;h<2;++h) {
      int i=h*4;
      uint32_t w=q[i]<<8 | q[i+1]<<24 | q[i+2]<<12 | q[i+3]<<28;
      uint32_t sc=s[i] | s[i+1]<<16 | s[i+2]<<8 | s[i+3]<<24;
      nv_bfloat162 fs[2];
      marlin::dequant<nv_bfloat162,sglang::host::kFE2M1f.id(),true>(w,a+h*2);
      marlin::dequant_fp8_scales<nv_bfloat162,sglang::host::kFE4M3fn.id()>(sc,fs);
      a[h*2]=__hmul2(a[h*2],fs[0]);
      a[h*2+1]=__hmul2(a[h*2+1],fs[1]);
    }
    auto* elements=reinterpret_cast<Bf16*>(a);
    #pragma unroll
    for(int v=0;v<8;++v) decoded_out[indices[v]]=elements[v];
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
    int idx=(linear%64)*8+linear/64;
    accum_out[idx]=c[v];
    // Preserve incumbent cast-before-global-scale ordering, BF16 arithmetic.
    nv_bfloat16 rounded=__float2bfloat16_rn(c[v]);
    nv_bfloat16 gs=*reinterpret_cast<nv_bfloat16*>(&global_scale);
    nv_bfloat16 result=__hmul(rounded,gs);
    final_out[idx]=*reinterpret_cast<Bf16*>(&result);
  }
}

static float raw_fp4(uint8_t code) {
  static const float lut[8]={0,.5f,1,1.5f,2,3,4,6};
  return (code&8)?-lut[code&7]:lut[code&7];
}
static float raw_e4m3(uint8_t code) {
  int exp=(code>>3)&15, mant=code&7;
  float mag=exp?std::ldexp(1.0f+mant/8.0f,exp-7):std::ldexp(float(mant),-9);
  return (code&128)?-mag:mag;
}
struct Fixture {
  std::vector<uint8_t> packed,processed;
  std::vector<Bf16> activation,decoded;
  std::array<float,512> accum;
  std::array<Bf16,512> final;
  Bf16 global;
};
static Fixture fixture(int K,int trial) {
  Fixture f; f.packed.resize(64*K/2); f.processed.resize(64*K/16);
  f.activation.resize(8*K); f.decoded.resize(64*K);
  // Processed global scale from nvfp4_marlin_process_global_scale:
  // exponent_bias=(1<<(8-1))-(1<<(2-1))=126; subtract7 =>119.
  f.global=Bf16(std::ldexp(1.0f,119));
  for(int m=0;m<64;++m) for(int g=0;g<K/16;++g) {
    // All eight mantissas of positive normalized E4M3, scale1..1.875.
    // Half(scale*128) upper byte after <<1 equals raw code+120 here.
    uint8_t raw=uint8_t(0x38+((m+3*g+trial)%8));
    f.processed[m*(K/16)+g]=uint8_t(raw+120);
    for(int j=0;j<16;++j) {
      int k=g*16+j,idx=m*K+k;
      uint8_t code=uint8_t((m*7+k*11+trial*3)%16);
      f.packed[idx/2]|=code<<((k%2)*4);
      // Independent raw-format decode; no device bit extraction or fragment
      // packing algorithm reused. 2^-119 compensates processed global scale.
      f.decoded[idx]=Bf16(std::ldexp(raw_fp4(code)*raw_e4m3(raw),-119));
    }
  }
  for(int n=0;n<8;++n) for(int k=0;k<K;++k)
    f.activation[n*K+k]=Bf16(float(((n*13+k*3+trial*17)%15)-7)/8.0f);
  for(int m=0;m<64;++m) for(int n=0;n<8;++n) {
    // Double oracle avoids inheriting a particular device summation order;
    // selected dyadic values have exactFP32 products and sums even atK4096.
    double acc=0;
    for(int k=0;k<K;++k) acc+=double(float(f.decoded[m*K+k]))*float(f.activation[n*K+k]);
    f.accum[m*8+n]=float(acc);
    if(double(f.accum[m*8+n])!=acc) { std::fprintf(stderr,"fixture sum not exactFP32\n"); std::exit(3); }
    f.final[m*8+n]=Bf16(float(Bf16(f.accum[m*8+n]))*float(f.global));
  }
  return f;
}
int main(int argc,char**argv) {
  bool cpu_only=argc==2 && std::strcmp(argv[1],"--cpu-only")==0;
  if(argc>1&&!cpu_only) { std::fprintf(stderr,"usage: %s [--cpu-only]\n",argv[0]);return 2; }
  if(!cpu_layout_validation()) return 1;
  if(!cpu_only) { cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.minor!=0)return 2; }
  int cases=0,decoded_checks=0,output_checks=0; double checksum=0,abs_checksum=0;
  for(int K:{16,64,128,4096}) for(int trial=0;trial<3;++trial) {
    Fixture f=fixture(K,trial);
    for(Bf16 x:f.final) { checksum+=float(x);abs_checksum+=std::abs(float(x)); }
    if(cpu_only) { ++cases;continue; }
    uint8_t *dp,*ds; Bf16 *da,*dd,*df;float*dc;
    CK(cudaMalloc(&dp,f.packed.size()));CK(cudaMalloc(&ds,f.processed.size()));
    CK(cudaMalloc(&da,f.activation.size()*2));CK(cudaMalloc(&dd,f.decoded.size()*2));
    CK(cudaMalloc(&df,1024));CK(cudaMalloc(&dc,2048));
    CK(cudaMemcpy(dp,f.packed.data(),f.packed.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds,f.processed.data(),f.processed.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da,f.activation.data(),f.activation.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemset(dd,0xff,f.decoded.size()*2));CK(cudaMemset(df,0xff,1024));CK(cudaMemset(dc,0xff,2048));
    rs_nvfp4_tile<<<1,128>>>(dp,ds,da,K,f.global,dc,df,dd);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    std::vector<Bf16> gotd(f.decoded.size()); std::array<Bf16,512> gotf;std::array<float,512> gotc;
    CK(cudaMemcpy(gotd.data(),dd,gotd.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gotf.data(),df,1024,cudaMemcpyDeviceToHost));CK(cudaMemcpy(gotc.data(),dc,2048,cudaMemcpyDeviceToHost));
    CK(cudaFree(dp));CK(cudaFree(ds));CK(cudaFree(da));CK(cudaFree(dd));CK(cudaFree(df));CK(cudaFree(dc));
    for(size_t i=0;i<gotd.size();++i) {
      if(!std::isfinite(float(gotd[i])) || float(gotd[i])!=float(f.decoded[i])) {
        std::fprintf(stderr,"decode mismatch K=%d trial=%d idx=%zu got=%g ref=%g\n",K,trial,i,float(gotd[i]),float(f.decoded[i]));return 1;
      } ++decoded_checks;
    }
    for(int i=0;i<512;++i) {
      if(!std::isfinite(gotc[i]) || gotc[i]!=f.accum[i] || !std::isfinite(float(gotf[i])) || float(gotf[i])!=float(f.final[i])) {
        std::fprintf(stderr,"GEMM/epilogue mismatch K=%d trial=%d idx=%d raw=%g ref=%g final=%g ref=%g\n",K,trial,i,gotc[i],f.accum[i],float(gotf[i]),float(f.final[i]));return 1;
      } ++output_checks;
    } ++cases;
  }
  if(abs_checksum==0) { std::fprintf(stderr,"degenerate zero fixture\n");return 3; }
  std::printf("{\"mode\":\"%s\",\"cases\":%d,\"decoded_checks\":%d,\"output_checks\":%d,\"reference_checksum\":%.9g,\"reference_abs_checksum\":%.9g,\"pass\":true}\n",cpu_only?"cpu-only":"gpu-correctness",cases,decoded_checks,output_checks,checksum,abs_checksum);
}
