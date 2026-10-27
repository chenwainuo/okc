// One fixed unpooler128 policy. X register-A; decoded W shared-B. Full K, plain output.
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL> struct RqC {
  static constexpr int WST=6,BST=8,WB=6144,BB=8192,ND=32;
  static constexpr int SMEM=1024+WST*WB+BST*BB+8*WST+16*BST;
};
// Every call writes all1MiB in this kernel, then consumes it in rq_px on the same stream.
__global__ void rx_pack(const uint32_t* x,uint32_t* packed) {
  const uint32_t idx=blockIdx.x*256u+threadIdx.x; //65536threads, fourwords/thread.
  const uint32_t v=idx&3u,lane=(idx>>2)&31u,chunk=(idx>>7)&3u,warp=(idx>>9)&7u,s=idx>>12;
  const uint32_t row=64u*(warp>>2)+16u*(warp&3u)+(lane>>2)+8u*(v&1u);
  const uint32_t k=64u*s+16u*chunk+8u*((v>>1)&1u)+2u*(lane&3u);
  const uint32_t source=row*2048u+(k>>1);
#pragma unroll
  for(int u=0;u<4;++u)packed[idx+65536u*u]=x[source+512u*u];
}
template<int BYTE_OFFSET>
__device__ __forceinline__ uint4 px_ldg(uint64_t base) {
  uint4 q;asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4+%5];\n"
    :"=r"(q.x),"=r"(q.y),"=r"(q.z),"=r"(q.w):"l"(base),"n"(BYTE_OFFSET):"memory");return q;
}
template<int BYTE_OFFSET>
__device__ __forceinline__ void px_vec(uint64_t base,uint32_t* q){
  const uint4 v=px_ldg<BYTE_OFFSET>(base);q[0]=v.x;q[1]=v.y;q[2]=v.z;q[3]=v.w;
}
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL>
__global__ void __launch_bounds__(512,1)
rq_px_fp(const RqP p,const __grid_constant__ RqTmap tx,const __grid_constant__ RqTmap to){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  static_assert(NT==128&&ST==8&&GS==1&&PRB==0&&WM==0&&INS==0&&FL==20,"one policy only");
  constexpr int KC=64,WST=6,BST=8,WB=6144,BB=8192;
  extern __shared__ __align__(1024) uint8_t scratch[];
  uint32_t raw=(uint32_t)__cvta_generic_to_shared(scratch),base=(raw+1023u)&~1023u;
  uint32_t bring=base+WST*WB,wfull=bring+BST*BB,bfull=wfull+8*WST,bempty=bfull+8*BST;
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,w=warp&3,t=lane&3,g=lane>>2,wg=warp>>2;
  int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;
  uint32_t ec=warp>=8?__ldg(p.esc+(size_t)(nb*8+4*half+w)*p.ecap+lane):0u;
  uint32_t q0[16],q1[16],q2[16],q3[16];
  const uint64_t xbase=(uint64_t)p.XP+uint32_t(warp)*2048u+uint32_t(lane)*16u;
  auto prefetch=[&](int s,uint32_t (&q)[16]){
    const uint64_t addr=xbase+(uint32_t(s)<<14);
    px_vec<0>(addr,q);px_vec<512>(addr,q+4);
    px_vec<1024>(addr,q+8);px_vec<1536>(addr,q+12);
  };
  if(warp<8){prefetch(0,q0);prefetch(1,q1);}
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
    const int owner=(warp-8)/4;
    const uint32_t* el=p.esc+(size_t)(nb*8+4*half+w)*p.ecap;
    uint32_t nx=__shfl_sync(0xffffffffu,ec,0),nxs=nx>>17;int ei=0;
#pragma unroll 1
    for(int s=owner;s<KC;s+=2){
      rq_wait(wfull+8*(s%WST),(s/WST)&1);
      if(s>=BST)rq_wait(bempty+8*(s%BST),((s/BST)&1)^1);
      uint32_t a[16],wa=base+(s%WST)*WB+w*1536+lane*16;
      uint4 x0=rq_lds(wa),x1=rq_lds(wa+512),x2=rq_lds(wa+1024);
      rq_unit(x0,x1,x2,p.Tb,a);
      // Each independent warp cursor skips the other producer's earlier stages.
      while(nxs<(uint32_t)s){
        ++ei;if((ei&31)==0)ec=__ldg(el+ei+lane);
        nx=__shfl_sync(0xffffffffu,ec,ei&31);nxs=nx>>17;
      }
      rq_fix(a,s,nx,nxs,ei,ec,el,lane);
#pragma unroll
      for(int k=0;k<4;++k)
#pragma unroll
        for(int i=0;i<4;++i){
          int row=16*w+8*(i%2)+g;
          uint32_t dst=bring+(s%BST)*BB+row*128+(((2*k+i/2)^(row&7))*16)+4*t;
          asm volatile("st.shared.u32 [%0],%1;\n"::"r"(dst),"r"(a[k*4+i]):"memory");
        }
      asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
      if(owner==0)asm volatile("bar.sync 3,128;\n":::"memory");
      else asm volatile("bar.sync 4,128;\n":::"memory");
      if(tid==256+128*owner){
        rq_arrive(bfull+8*(s%BST));
        if(s+WST<KC){
          rq_expect(wfull+8*(s%WST),WB);
          rq_bulk(base+(s%WST)*WB,p.W+((size_t)nb*KC+s+WST)*12288+half*WB,WB,wfull+8*(s%WST));
        }
      }
    }
    return;
  }
  float d[32]={};
  auto pin=[&](uint32_t (&a)[16]){
#pragma unroll
    for(int i=0;i<16;++i)asm volatile("mov.b32 %0,%0;\n":"+r"(a[i])::"memory");
  };
  auto release=[&](int s){__syncwarp();if(lane==0)rq_arrive(bempty+8*(s%BST));};
  // One register fence covers the current/next A banks, with no generic D access between groups.
  auto issue=[&](int s,uint32_t (&a)[16]){
    uint64_t bd=rq_desc(bring+(s%BST)*BB);
#pragma unroll
    for(int k=0;k<4;++k)rq_wgmma<64>(d,a+4*k,bd+2*k,1);
    asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;\n":::"memory");
  };
  auto pair=[&](int b,uint32_t (&current)[16],uint32_t (&next)[16],uint32_t (&old2)[16],uint32_t (&old1)[16],auto future,auto old){
    rq_wait(bfull+8*(b%BST),(b/BST)&1);
    rq_wait(bfull+8*((b+1)%BST),((b+1)/BST)&1);
    rq_facc<32>(d);rq_fa(current);rq_fa(next);
    asm volatile("wgmma.fence.sync.aligned;\n":::"memory");
    // b-2 was retired by the previous round; modulo4 it is the b+2 bank.
    if constexpr(decltype(future)::value)prefetch(b+2,old2);
    issue(b,current);
    if constexpr(decltype(old)::value)pin(old1);
    // Only after group b wait1 retires b-1 may its bank become b+3.
    if constexpr(decltype(future)::value)prefetch(b+3,old1);
    issue(b+1,next);
    pin(current);
    if constexpr(decltype(old)::value)release(b-1);
    release(b);
  };
  // Separate startup/tail so every steady-state future load is unconditional.
  pair(0,q0,q1,q2,q3,std::true_type{},std::false_type{});pair(2,q2,q3,q0,q1,std::true_type{},std::true_type{});
#pragma unroll 1
  for(int b=4;b<60;b+=4){pair(b,q0,q1,q2,q3,std::true_type{},std::true_type{});pair(b+2,q2,q3,q0,q1,std::true_type{},std::true_type{});}
  pair(60,q0,q1,q2,q3,std::true_type{},std::true_type{});pair(62,q2,q3,q0,q1,std::false_type{},std::true_type{});
  asm volatile("wgmma.wait_group.sync.aligned 0;\n":::"memory");rq_facc<32>(d);
  rq_fa(q0);rq_fa(q1);rq_fa(q2);rq_fa(q3);release(KC-1);
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
