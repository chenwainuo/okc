// One fixed unpooler128 policy. X register-A; decoded W shared-B. Full K, plain output.
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL> struct RqC {
  static constexpr int WST=6,BST=8,WB=6144,BB=8192,ND=32;
  static constexpr int SMEM=1024+WST*WB+BST*BB+8*WST+16*BST+848+4*16384+4*8;
};
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL>
__global__ void __launch_bounds__(512,1)
rq_tma_x4_stmx(const RqP p,const __grid_constant__ RqTmap tx,const __grid_constant__ RqTmap to){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  static_assert(NT==128&&ST==8&&GS==1&&PRB==0&&WM==0&&INS==0&&FL==20,"one policy only");
  constexpr int KC=64,WST=6,BST=8,WB=6144,BB=8192;
  extern __shared__ __align__(1024) uint8_t scratch[];
  uint32_t raw=(uint32_t)__cvta_generic_to_shared(scratch),base=(raw+1023u)&~1023u;
  uint32_t bring=base+WST*WB,wfull=bring+BST*BB,bfull=wfull+8*WST,bempty=bfull+8*BST;
  uint32_t xring=base+103424, xfull=xring+65536;
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,w=warp&3,t=lane&3,g=lane>>2,wg=warp>>2;
  int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;
  uint32_t ec=warp>=8?__ldg(p.esc+(size_t)(nb*8+4*half+w)*p.ecap+lane):0u;
  uint32_t q0[16],q1[16],q2[16],q3[16];
  auto prefetch=[&](int s,uint32_t (&q)[16]){
    const uint32_t slot=s&3, xb=xring+slot*16384;
    rq_wait(xfull+8*slot,(s/4)&1);
    // Matrices: top-left, bottom-left, top-right, bottom-right of16x16.
    const int row=64*wg+16*w+(lane&7)+8*((lane>>3)&1);
#pragma unroll
    for(int k=0;k<4;++k){
      const uint32_t addr=xb+row*128+(((2*k+(lane>>4))^(row&7))*16);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        :"=r"(q[4*k]),"=r"(q[4*k+1]),"=r"(q[4*k+2]),"=r"(q[4*k+3]):"r"(addr):"memory");
    }
    // All math readers retire generic shared reads before async TMA reuse.
    asm volatile("fence.proxy.async.shared::cta;\nbar.sync 5,256;\n":::"memory");
    if(tid==0 && s+4<KC){
      rq_expect(xfull+8*slot,16384);
      rq_tile(xb,&tx,(s+4)*64,0,xfull+8*slot);
    }
  };
  if(tid==256){
    for(int s=0;s<WST;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(wfull+8*s));
    for(int s=0;s<BST;++s){
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(bfull+8*s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0],8;\n"::"r"(bempty+8*s));
    }
    for(int s=0;s<4;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(xfull+8*s));
    asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
    for(int s=0;s<4;++s){
      rq_expect(xfull+8*s,16384);
      rq_tile(xring+s*16384,&tx,s*64,0,xfull+8*s);
    }
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
      for(int k=0;k<4;++k){
        // PTX x4 address-provider lane is distinct from each data lane.
        const int row=16*w+8*((lane>>3)&1)+(lane&7),chunk=2*k+(lane>>4);
        const uint32_t dst=bring+(s%BST)*BB+row*128+((chunk^(row&7))*16);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
          ::"r"(dst),"r"(a[4*k]),"r"(a[4*k+1]),"r"(a[4*k+2]),"r"(a[4*k+3]):"memory");
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
  prefetch(0,q0);prefetch(1,q1);prefetch(2,q2);prefetch(3,q3);
  float d[32]={};
  auto release=[&](int s){__syncwarp();if(lane==0)rq_arrive(bempty+8*(s%BST));};
  auto round=[&](int s,uint32_t (&a)[16],uint32_t (&retired)[16]){
    rq_wait(bfull+8*(s%BST),(s/BST)&1);
    rq_facc<32>(d);rq_fa(a);asm volatile("wgmma.fence.sync.aligned;\n":::"memory");
    uint64_t bd=rq_desc(bring+(s%BST)*BB);
#pragma unroll
    for(int k=0;k<4;++k)rq_wgmma<64>(d,a+4*k,bd+2*k,1);
    asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");rq_facc<32>(d);
    // Current A stays immutable. Only group(s-1) is retired by wait_group1.
    asm volatile("wgmma.wait_group.sync.aligned 1;\n":::"memory");
    if(s>0){
      rq_fa(retired);release(s-1);
      if(s+3<KC)prefetch(s+3,retired);  // (s+3)%4 == (s-1)%4, now safe
    }
  };
#pragma unroll 1
  for(int s=0;s<KC;s+=4){round(s,q0,q3);round(s+1,q1,q0);round(s+2,q2,q1);round(s+3,q3,q2);}
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
