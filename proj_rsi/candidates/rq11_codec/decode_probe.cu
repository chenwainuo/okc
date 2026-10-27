// Offline compile/SASS probe only. No host launcher, extension, GPU or benchmark.
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {
  uint32_t d; asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s)); return d;
}
__device__ __forceinline__ uint4 lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(a):"memory"); return v;
}
__device__ __forceinline__ uint32_t lds1(uint32_t a) {
  uint32_t v; asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(a):"memory"); return v;
}
__device__ __forceinline__ void stock_dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t tb,uint32_t* a) {
  uint32_t hl=(c&0x87878787u)+tb,hh=(__funnelshift_r(c,c,4)&0x87878787u)+tb;
  a[0]=prmt(l0,hl,0x5140);a[1]=prmt(l0,hl,0x7362);a[2]=prmt(l1,hh,0x5140);a[3]=prmt(l1,hh,0x7362);
}
// Four packed three-bit indexes -> four PRMT selector nibbles. Bit 2
// selects the signed high-byte table; bits 1:0 select base+offset.
__device__ __forceinline__ uint32_t spread(uint32_t c) {
  uint32_t x=(c&0x3fu)|((c&0xfc0u)<<2);
  return (x&0x0707u)|((x&0x3838u)<<1);
}
__device__ __forceinline__ void rq11_dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t tb,uint32_t* a) {
  uint32_t hl=prmt(tb,tb|0x80808080u,spread(c));
  uint32_t hh=prmt(tb,tb|0x80808080u,spread(c>>12));
  a[0]=prmt(l0,hl,0x5140);a[1]=prmt(l0,hl,0x7362);a[2]=prmt(l1,hh,0x5140);a[3]=prmt(l1,hh,0x7362);
}
template<bool RQ11> __device__ __forceinline__ void body(uint32_t* out,uint32_t tb) {
  extern __shared__ __align__(128) uint8_t bytes[];
  uint32_t s=static_cast<uint32_t>(__cvta_generic_to_shared(bytes));
  uint32_t lane=threadIdx.x&31;
  uint4 q0=lds(s+16*lane),q1=lds(s+512+16*lane);
  uint32_t a[16];
  if constexpr(RQ11) {
    uint32_t h0=lds1(s+1024+4*lane),h1=lds1(s+1152+4*lane),h2=lds1(s+1280+4*lane);
    rq11_dec(q0.x,q0.y,h0,tb,a);
    rq11_dec(q0.z,q0.w,__funnelshift_r(h0,h1,24),tb,a+4);
    rq11_dec(q1.x,q1.y,__funnelshift_r(h1,h2,16),tb,a+8);
    rq11_dec(q1.z,q1.w,h2>>8,tb,a+12);
  } else {
    uint4 q2=lds(s+1024+16*lane);
    stock_dec(q0.x,q0.y,q2.x,tb,a);stock_dec(q0.z,q0.w,q2.y,tb,a+4);
    stock_dec(q1.x,q1.y,q2.z,tb,a+8);stock_dec(q1.z,q1.w,q2.w,tb,a+12);
  }
  #pragma unroll
  for(int i=0;i<16;i++)out[16*threadIdx.x+i]=a[i];
}
// Shared bytes deliberately unspecified: these kernels are never executed.
// tb: stock B repeated in all bytes; rq11 [B,B+1,B+2,B+3].
extern "C" __global__ void stock_probe(uint32_t* out,uint32_t tb){body<false>(out,tb);}
extern "C" __global__ void rq11_probe(uint32_t* out,uint32_t tb){body<true>(out,tb);}
