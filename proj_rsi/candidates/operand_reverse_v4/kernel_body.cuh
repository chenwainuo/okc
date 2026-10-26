// One fixed unpooler128 policy. X register-A; decoded W shared-B. Full K, plain output.
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL> struct RqC {
  static constexpr int WST=6,BST=8,WB=6144,BB=8192,ND=32;
  static constexpr int SMEM=1024+WST*WB+BST*BB+8*WST+16*BST;
};
// Same8B nc/no_allocate load; template displacement is a PTX immediate, not a new pointer calculation.
template<int BYTE_OFFSET>
__device__ __forceinline__ uint2 rev4_ldg(uint64_t base) {
  uint2 q; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1},[%2+%3];\n"
    : "=r"(q.x),"=r"(q.y):"l"(base),"n"(BYTE_OFFSET):"memory");return q;
}
__device__ __forceinline__ void rev4_demux(const uint2 (&q)[8],uint32_t (&a)[16],int t) {
#pragma unroll
  for(int j=0;j<4;++j)
#pragma unroll
    for(int half=0;half<2;++half){
      // Explicit two-shuffle/two-select map; compiler convergence fallbacks are separate CFG paths.
      // Snapshot both inputs before outputs, so tied input/output registers are safe.
      asm volatile("{\n.reg .b32 x,y,u,v;\n.reg .pred p;\n"
        "mov.b32 x,%2; mov.b32 y,%3;\n"
        "shfl.sync.bfly.b32 u,x,1,31,0xffffffff;\n"
        "shfl.sync.bfly.b32 v,y,1,31,0xffffffff;\n"
        "setp.ne.u32 p,%4,0;\n"
        "selp.b32 %0,v,x,p;\nselp.b32 %1,y,u,p;\n}\n"
        :"=r"(a[4*j+half]),"=r"(a[4*j+half+2])
        :"r"(q[2*j+half].x),"r"(q[2*j+half].y),"r"(t&1));
    }
}
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL>
__global__ void __launch_bounds__(384,1)
rq_v4(const RqP p,const __grid_constant__ RqTmap tx,const __grid_constant__ RqTmap to){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  static_assert(NT==128&&ST==8&&GS==1&&PRB==0&&WM==0&&INS==0&&FL==20,"one policy only");
  constexpr int KC=64,WST=6,BST=8,WB=6144,BB=8192;
  extern __shared__ __align__(1024) uint8_t scratch[];
  uint32_t raw=(uint32_t)__cvta_generic_to_shared(scratch),base=(raw+1023u)&~1023u;
  uint32_t bring=base+WST*WB,wfull=bring+BST*BB,bfull=wfull+8*WST,bempty=bfull+8*BST;
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,w=warp&3,t=lane&3,g=lane>>2,wg=warp>>2;
  int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;
  uint32_t ec=warp>=8?__ldg(p.esc+(size_t)(nb*8+4*half+w)*p.ecap+lane):0u;
  uint2 q0[8],q1[8],q2[8],q3[8];
  // Shape guard already enforcesK4096. Integer addresses for inactive producer
  // warps are never dereferenced. Active rows0..119 plus halfrow8 remain0..127.
  const uint64_t xrow=(uint64_t)p.X+((uint32_t)(wg*64+w*16+g)<<13)
                     +((uint32_t)(2*(t&2)+8*(t&1))<<1);
  auto prefetch=[&](int s,uint2 (&q)[8]){
    const uint64_t addr=xrow+((uint32_t)s<<7);
    q[0]=rev4_ldg<0>(addr);     q[1]=rev4_ldg<65536>(addr);
    q[2]=rev4_ldg<32>(addr);    q[3]=rev4_ldg<65568>(addr);
    q[4]=rev4_ldg<64>(addr);    q[5]=rev4_ldg<65600>(addr);
    q[6]=rev4_ldg<96>(addr);    q[7]=rev4_ldg<65632>(addr);
  };
  if(warp<8){prefetch(0,q0);prefetch(1,q1);prefetch(2,q2);prefetch(3,q3);}
  if(tid==256){
    for(int s=0;s<WST;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(wfull+8*s));
    for(int s=0;s<BST;++s){
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(bfull+8*s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0],8;\n"::"r"(bempty+8*s));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
    for(int s=0;s<WST;++s){
      rq_expect(wfull+8*s,WB);
      rq_bulk(base+s*WB,p.W+((size_t)nb*KC+s)*12288+half*WB,WB,wfull+8*s);
    }
  }
  __syncthreads();
  if(warp>=8){
    const uint32_t* el=p.esc+(size_t)(nb*8+4*half+w)*p.ecap;
    uint32_t nx=__shfl_sync(0xffffffffu,ec,0),nxs=nx>>17;int ei=0;
#pragma unroll 1
    for(int s=0;s<KC;++s){
      rq_wait(wfull+8*(s%WST),(s/WST)&1);
      if(s>=BST)rq_wait(bempty+8*(s%BST),((s/BST)&1)^1);
      uint32_t a[16],wa=base+(s%WST)*WB+w*1536+lane*16;
      uint4 x0=rq_lds(wa),x1=rq_lds(wa+512),x2=rq_lds(wa+1024);
      rq_unit(x0,x1,x2,p.Tb,a);rq_fix(a,s,nx,nxs,ei,ec,el,lane);
#pragma unroll
      for(int k=0;k<4;++k)
#pragma unroll
        for(int i=0;i<4;++i){
          int row=16*w+8*(i%2)+g;
          uint32_t dst=bring+(s%BST)*BB+row*128+(((2*k+i/2)^(row&7))*16)+4*t;
          asm volatile("st.shared.u32 [%0],%1;\n"::"r"(dst),"r"(a[k*4+i]):"memory");
        }
      asm volatile("fence.proxy.async.shared::cta;\nbar.sync 3,128;\n":::"memory");
      if(tid==256){
        rq_arrive(bfull+8*(s%BST));
        if(s+WST<KC){
          rq_expect(wfull+8*(s%WST),WB);
          rq_bulk(base+(s%WST)*WB,p.W+((size_t)nb*KC+s+WST)*12288+half*WB,WB,wfull+8*(s%WST));
        }
      }
    }
    return;
  }
  float d[32]={};uint32_t a0[16],a1[16];
  auto release=[&](int s){__syncwarp();if(lane==0)rq_arrive(bempty+8*(s%BST));};
  auto round=[&](int s,uint2 (&q)[8],uint32_t (&a)[16],uint32_t (&old)[16]){
    rev4_demux(q,a,t);rq_wait(bfull+8*(s%BST),(s/BST)&1);
    rq_facc<32>(d);rq_fa(a);asm volatile("wgmma.fence.sync.aligned;\n":::"memory");
    uint64_t bd=rq_desc(bring+(s%BST)*BB);
#pragma unroll
    for(int k=0;k<4;++k)rq_wgmma<64>(d,a+4*k,bd+2*k,1);
    asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");rq_facc<32>(d);
    if(s+4<KC)prefetch(s+4,q);
    asm volatile("wgmma.wait_group.sync.aligned 1;\n":::"memory");
    if(s>0){rq_fa(old);release(s-1);}
  };
#pragma unroll 1
  for(int s=0;s<KC;s+=4){round(s,q0,a0,a1);round(s+1,q1,a1,a0);round(s+2,q2,a0,a1);round(s+3,q3,a1,a0);}
  asm volatile("wgmma.wait_group.sync.aligned 0;\n":::"memory");rq_facc<32>(d);rq_fa(a0);rq_fa(a1);release(KC-1);
  asm volatile("bar.sync 1,256;\n":::"memory");
  // Native D[token][channel], every pair observed; a single combined128x64 slab.
#pragma unroll
  for(int j=0;j<8;++j){
    int row=wg*64+w*16+g;
    uint32_t v0=rq_pk(d[4*j],d[4*j+1]),v1=rq_pk(d[4*j+2],d[4*j+3]);
    uint32_t dst=bring+row*128+((j^(row&7))*16)+4*t;
    asm volatile("st.shared.u32 [%0],%1;\nst.shared.u32 [%2],%3;\n"::"r"(dst),"r"(v0),"r"(dst+8*128),"r"(v1):"memory");
  }
  asm volatile("fence.proxy.async.shared::cta;\nbar.sync 2,256;\n":::"memory");
  if(tid==0){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,0}],[%2];\n"::"l"((unsigned long long)&to),"r"(cb),"r"(bring):"memory");
    asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;\n":::"memory");
  }
#endif
}
