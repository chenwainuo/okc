#pragma once
// New CPU prototype: two K64 operand slots, producer/math role separation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace moe_ss_v2 {
using B=__nv_bfloat16;
__device__ __forceinline__ uint32_t addr(const void* p){return uint32_t(__cvta_generic_to_shared(p));}
__device__ __forceinline__ int sw(int r,int k){return r*64+(((k/8)^(r&7))*8)+(k&7);}
__device__ __forceinline__ uint64_t desc(uint32_t p){return uint64_t(p>>4)|(uint64_t(1)<<16)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
__device__ __forceinline__ void init(uint32_t p){asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(p):"memory");}
__device__ __forceinline__ void arrive(uint32_t p){asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"::"r"(p):"memory");}
__device__ __forceinline__ void wait(uint32_t p,int phase){
 asm volatile("{ .reg .pred ok; WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ok, [%0], %1; @!ok bra WAIT; }"::"r"(p),"r"(phase):"memory");
}
// Canonical NVFP4 numeric representation, no compensating 2^-119 bias.
// Pure bit helpers are host/device so CPU tests compile the exact functions.
__host__ __device__ __forceinline__ uint16_t e2m1_bits(uint32_t code){
 uint32_t m=code&7u,sign=(code&8u)<<12;
 return uint16_t(sign|(m>=2u?(0x3f00u+(m<<6)):(m?0x3f00u:0u)));
}
__host__ __device__ __forceinline__ uint16_t raw_scale_bits(uint32_t code){
 uint32_t e=(code>>3)&15u,m=code&7u,sign=(code&128u)<<8;
 if(e==15u&&m==7u)return uint16_t(sign|0x7fc0u); // FP8 NaNs remain nonfinite.
 if(e)return uint16_t(sign|((e+120u)<<7)|(m<<4));
 uint32_t sub=m>=4u?(0x3c00u+((m-4u)<<5)):m==3u?0x3bc0u:m==2u?0x3b80u:m==1u?0x3b00u:0u;
 return uint16_t(sign|sub); // includes signedzero and all seven FP8 subnormals.
}
__device__ __forceinline__ void dequant(int q,__nv_bfloat162* f){
 uint32_t u=uint32_t(q);
 uint32_t a=uint32_t(e2m1_bits((u>>8)&15u))|(uint32_t(e2m1_bits((u>>24)&15u))<<16);
 uint32_t b=uint32_t(e2m1_bits((u>>12)&15u))|(uint32_t(e2m1_bits((u>>28)&15u))<<16);
 f[0]=*reinterpret_cast<__nv_bfloat162*>(&a);f[1]=*reinterpret_cast<__nv_bfloat162*>(&b);
}
__device__ __forceinline__ void scale(int q,__nv_bfloat162* f){
 B a=__ushort_as_bfloat16(raw_scale_bits(uint32_t(q)&255u));
 B b=__ushort_as_bfloat16(raw_scale_bits((uint32_t(q)>>8)&255u));
 f[0]=__bfloat162bfloat162(a);f[1]=__bfloat162bfloat162(b);
}
template<int T> __device__ __forceinline__ void mma(float(&c)[T/2],uint64_t a,uint64_t b,int accumulate);
template<> __device__ __forceinline__ void mma<8>(float(&c)[4],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0; }" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "l"(a),"l"(b),"r"(accumulate):"memory");
}
template<> __device__ __forceinline__ void mma<16>(float(&c)[8],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %10, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0; }" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "l"(a),"l"(b),"r"(accumulate):"memory");
}
template<> __device__ __forceinline__ void mma<32>(float(&c)[16],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0; }" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]) : "l"(a),"l"(b),"r"(accumulate):"memory");
}
template<> __device__ __forceinline__ void mma<64>(float(&c)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0; }" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]) : "l"(a),"l"(b),"r"(accumulate):"memory");
}
template<int T,int DIV> __global__ __launch_bounds__(256) void project(
 const uint32_t* packed,const uint8_t* scales,const B* input,const B* globals,
 const int32_t* sorted,const int32_t* experts,const int32_t* post,
 B* output,int K,int N,int E,int input_rows,int output_slots,int capacity,int expert_capacity,int block_m){
 static_assert(T==8||T==16||T==32||T==64);static_assert(DIV==1||DIV==8);
 constexpr int AW=4096,BW=T*64,STAGE=AW+BW;
 extern __shared__ __align__(1024) unsigned char memory[];
 B* ring=reinterpret_cast<B*>(memory);
 uint32_t full=addr(memory+2*STAGE*2),empty=full+16;
 int position=int(blockIdx.y)*T,channel=int(blockIdx.x)*64,tid=int(threadIdx.x);
 if(position>=capacity||position>=post[0])return;
 int eb=position/block_m;if(eb>=expert_capacity)return;
 int expert=experts[eb];if(expert<0||expert>=E)return;
 packed+=(int64_t(expert)*(N/64)+channel/64)*(K/16)*128;
 scales+=(int64_t(expert)*(N/64)+channel/64)*(K/16)*64;
 if(tid==0){init(full);init(full+8);init(empty);init(empty+8);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 __syncthreads();
 if(tid>=128){
  int lane=tid-128;
  for(int s=0;s<K/64;++s){
   int slot=s&1;if(s>=2)wait(empty+8*slot,((s/2)-1)&1);
   B* a=ring+slot*STAGE;B* b=a+AW;
#pragma unroll
   for(int step=0;step<4;++step){
    uint32_t payload=packed[(s*4+step)*128+lane];int row0=(lane/4)%8+16*(lane/32);
    uint32_t s0=scales[(s*4+step)*64+row0],s1=scales[(s*4+step)*64+row0+8];
    uint32_t sc=s0|(s0<<16)|(s1<<8)|(s1<<24);__nv_bfloat162 fs[2],f[4];scale(sc,fs);
#pragma unroll
    for(int h=0;h<2;++h){uint32_t hw=payload>>(16*h);uint32_t q=((hw&255)<<8)|((hw&0xff00)<<16);dequant(q,f+2*h);f[2*h]=__hmul2(f[2*h],fs[0]);f[2*h+1]=__hmul2(f[2*h+1],fs[1]);}
    B* values=reinterpret_cast<B*>(f);
#pragma unroll
    for(int v=0;v<8;++v){int row=row0+8*((v/2)%2),k=2*(lane%4)+(v%2)+8*(v/4);a[sw(row,step*16+k)]=values[v];}
   }
   for(int linear=lane;linear<BW;linear+=128){
    int token=linear/64,k=linear%64;int route=position+token<capacity?sorted[position+token]:output_slots;
    b[sw(token,k)]=(route>=0&&route<output_slots&&route/DIV<input_rows)?input[int64_t(route/DIV)*K+s*64+k]:__float2bfloat16(0.f);
   }
   asm volatile("fence.proxy.async.shared::cta; bar.sync 2,128;":::"memory");
   if(lane==0)arrive(full+8*slot);
  }
  return;
 }
 float c[T/2]={};
 for(int s=0;s<K/64;++s){
  int slot=s&1;wait(full+8*slot,(s/2)&1);
  uint64_t a=desc(addr(ring+slot*STAGE)),b=desc(addr(ring+slot*STAGE+AW));
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll
  for(int k=0;k<4;++k)mma<T>(c,a+2*k,b+2*k,(s|k)!=0);
  asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;":::"memory");
  // All math warps finish async shared reads before the elected release.
  asm volatile("bar.sync 3,128;":::"memory");if(tid==0)arrive(empty+8*slot);
 }
 B gs=globals[expert];
#pragma unroll
 for(int v=0;v<T/2;++v){
  int row=(tid/4)%8+16*(tid/32)+8*((v/2)%2),token=2*(tid%4)+(v%2)+8*(v/4);
  int route=position+token<capacity?sorted[position+token]:output_slots;if(route<0||route>=output_slots)continue;
  B rounded=__float2bfloat16_rn(c[v]);output[int64_t(route)*N+channel+row]=__hmul(rounded,gs);
 }
}
constexpr int shared_bytes(int t){return 2*(4096+t*64)*2+32+1024;}
template<int T,int DIV> inline void launch(const uint32_t* p,const uint8_t* s,const B* x,const B* g,const int32_t* r,const int32_t* e,const int32_t* z,B* y,int K,int N,int E,int rows,int slots,int capacity,int ec,int bm,cudaStream_t stream){
 project<T,DIV><<<dim3(N/64,(capacity+T-1)/T),256,shared_bytes(T),stream>>>(p,s,x,g,r,e,z,y,K,N,E,rows,slots,capacity,ec,bm);
}
} // namespace moe_ss_v2

#include <torch/extension.h>
void run(torch::Tensor p,torch::Tensor sf,torch::Tensor x,torch::Tensor gs,torch::Tensor sorted,torch::Tensor expert,torch::Tensor post,torch::Tensor out,int64_t bm,int64_t div,int64_t stream_value){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&out.is_contiguous(),"CUDA contiguous tensors required");
 TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&out.scalar_type()==torch::kBFloat16,"BF16 boundary");
 TORCH_CHECK(p.scalar_type()==torch::kInt32&&sf.scalar_type()==torch::kUInt8&&gs.scalar_type()==torch::kBFloat16,"bank dtype");
 TORCH_CHECK(sorted.scalar_type()==torch::kInt32&&expert.scalar_type()==torch::kInt32&&post.scalar_type()==torch::kInt32,"route dtype");
 for(auto t:{p,sf,gs,sorted,expert,post,out})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous(),"same-device contiguous tensors");
 int E=p.size(0),N=p.size(1)*64,K=p.size(2)*16,rows=x.size(0),slots=out.size(0),cap=sorted.numel(),ec=expert.numel();
 TORCH_CHECK(p.dim()==4&&p.size(3)==128&&sf.dim()==4&&sf.size(0)==E&&sf.size(1)*64==N&&sf.size(2)*16==K&&sf.size(3)==64,"bank shapes");
 TORCH_CHECK(K==4096&&x.dim()==2&&x.size(1)==K&&out.dim()==2&&out.size(1)==N&&gs.numel()==E&&post.numel()==1,"geometry");
 TORCH_CHECK((div==1||div==8)&&slots==rows*div&&bm>=8&&bm%8==0&&cap>=bm&&(cap+bm-1)/bm<=ec,"route capacities");
 TORCH_CHECK(x.data_ptr()!=out.data_ptr(),"input/output alias");
 auto stream=reinterpret_cast<cudaStream_t>(stream_value);
 auto P=(const uint32_t*)p.data_ptr();auto S=(const uint8_t*)sf.data_ptr();auto X=(const moe_ss_v2::B*)x.data_ptr();auto G=(const moe_ss_v2::B*)gs.data_ptr();
 auto R=(const int32_t*)sorted.data_ptr();auto EID=(const int32_t*)expert.data_ptr();auto Z=(const int32_t*)post.data_ptr();auto Y=(moe_ss_v2::B*)out.data_ptr();
 #define LAUNCH(T,D) moe_ss_v2::launch<T,D>(P,S,X,G,R,EID,Z,Y,K,N,E,rows,slots,cap,ec,bm,stream)
 #define PICK(D) if(bm%64==0){LAUNCH(64,D);}else if(bm%32==0){LAUNCH(32,D);}else if(bm%16==0){LAUNCH(16,D);}else{LAUNCH(8,D);}
 if(div==8){PICK(8)}else{PICK(1)}
 TORCH_CHECK(cudaGetLastError()==cudaSuccess,"SS projection launch failed");
}

namespace moe_ss=moe_ss_v2;
namespace ss_screen {
template<int T,int D> __global__ void mover(const uint32_t* P,const uint8_t* S,const moe_ss::B* X,const int32_t* R,moe_ss::B* A,moe_ss::B* B,int N,int K,int rows,int slots,int capacity,int position,int stage,int expert,int nt){
 __shared__ moe_ss::B a[4096];__shared__ moe_ss::B b[T*64];int lane=threadIdx.x;
 P+=(int64_t(expert)*(N/64)+nt)*(K/16)*128;S+=(int64_t(expert)*(N/64)+nt)*(K/16)*64;
 for(int step=0;step<4;++step){uint32_t payload=P[(stage*4+step)*128+lane];int row0=(lane/4)%8+16*(lane/32);uint32_t s0=S[(stage*4+step)*64+row0],s1=S[(stage*4+step)*64+row0+8];uint32_t sc=s0|(s0<<16)|(s1<<8)|(s1<<24);__nv_bfloat162 fs[2],f[4];moe_ss::scale(sc,fs);
 for(int h=0;h<2;++h){uint32_t hw=payload>>(16*h);uint32_t q=((hw&255)<<8)|((hw&0xff00)<<16);moe_ss::dequant(q,f+2*h);f[2*h]=__hmul2(f[2*h],fs[0]);f[2*h+1]=__hmul2(f[2*h+1],fs[1]);}
 auto v=reinterpret_cast<moe_ss::B*>(f);for(int i=0;i<8;++i){int row=row0+8*((i/2)%2),k=2*(lane%4)+(i%2)+8*(i/4);a[moe_ss::sw(row,step*16+k)]=v[i];}}
 for(int i=lane;i<T*64;i+=128){int r=i/64,k=i%64,route=position+r<capacity?R[position+r]:slots;b[moe_ss::sw(r,k)]=(route>=0&&route<slots&&route/D<rows)?X[int64_t(route/D)*K+stage*64+k]:__float2bfloat16(0.f);}
 __syncthreads();for(int i=lane;i<4096;i+=128)A[i]=a[moe_ss::sw(i/64,i%64)];for(int i=lane;i<T*64;i+=128)B[i]=b[moe_ss::sw(i/64,i%64)];
}
template<int T,int D> int64_t info(int field){cudaFuncAttributes a;auto k=moe_ss::project<T,D>;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;if(field==1)return a.numRegs;if(field==2)return a.localSizeBytes;if(field==3)return a.sharedSizeBytes;if(field==4)return a.maxThreadsPerBlock;if(field==5)return a.maxDynamicSharedSizeBytes;if(field==6){int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,256,moe_ss::shared_bytes(T))!=cudaSuccess)return -2;return n;}return -3;}
}
int64_t ss_info(int64_t t,int64_t d,int64_t f){
 #define CASE(T,D) if(t==T&&d==D)return ss_screen::info<T,D>(f);
 CASE(8,1) CASE(16,1) CASE(32,1) CASE(64,1) CASE(8,8) CASE(16,8) CASE(32,8) CASE(64,8)
 #undef CASE
 return -9;
}
void ss_probe(torch::Tensor p,torch::Tensor sf,torch::Tensor x,torch::Tensor r,torch::Tensor a,torch::Tensor b,int64_t t,int64_t d,int64_t pos,int64_t stage,int64_t expert,int64_t nt,int64_t slots,int64_t stream_value){
 auto stream=reinterpret_cast<cudaStream_t>(stream_value);
 #define CASE(T,D) if(t==T&&d==D){ss_screen::mover<T,D><<<1,128,0,stream>>>((uint32_t*)p.data_ptr(),(uint8_t*)sf.data_ptr(),(moe_ss::B*)x.data_ptr(),(int32_t*)r.data_ptr(),(moe_ss::B*)a.data_ptr(),(moe_ss::B*)b.data_ptr(),p.size(1)*64,p.size(2)*16,x.size(0),slots,r.numel(),pos,stage,expert,nt);TORCH_CHECK(cudaGetLastError()==cudaSuccess,"mover probe launch");return;}
 CASE(8,1) CASE(16,1) CASE(32,1) CASE(64,1) CASE(8,8) CASE(16,8) CASE(32,8) CASE(64,8)
 #undef CASE
 TORCH_CHECK(false,"probe dispatch");
}
