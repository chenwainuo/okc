
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const __nv_bfloat16* X;
  const uint32_t* XP;
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
template <> __device__ __forceinline__ void rq_wgmma<64>(float (&D)[32], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}


// One fixed unpooler128 policy. X register-A; decoded W shared-B. Full K, plain output.
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL> struct RqC {
  static constexpr int WST=6,BST=8,WB=6144,BB=8192,ND=32;
  static constexpr int SMEM=1024+WST*WB+BST*BB+8*WST+16*BST+848+2*16384+4*8;
};
template<int NT,int ST,int GS,int PRB,int WM,int INS,int FL>
__global__ void __launch_bounds__(512,1)
rq_tma_x_async(const RqP p,const __grid_constant__ RqTmap tx,const __grid_constant__ RqTmap to){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  static_assert(NT==128&&ST==8&&GS==1&&PRB==0&&WM==0&&INS==0&&FL==20,"one policy only");
  constexpr int KC=64,WST=6,BST=8,WB=6144,BB=8192;
  extern __shared__ __align__(1024) uint8_t scratch[];
  uint32_t raw=(uint32_t)__cvta_generic_to_shared(scratch),base=(raw+1023u)&~1023u;
  uint32_t bring=base+WST*WB,wfull=bring+BST*BB,bfull=wfull+8*WST,bempty=bfull+8*BST;
  uint32_t xring=base+103424, xfull=xring+32768, xempty=xfull+16;
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,w=warp&3,t=lane&3,g=lane>>2,wg=warp>>2;
  int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;
  uint32_t ec=warp>=8?__ldg(p.esc+(size_t)(nb*8+4*half+w)*p.ecap+lane):0u;
  uint32_t q0[16],q1[16],q2[16],q3[16];
  auto prefetch=[&](int s,uint32_t (&q)[16]){
    const uint32_t slot=s&1, xb=xring+slot*16384;
    rq_wait(xfull+8*slot,(s/2)&1);
    // Matrices: top-left, bottom-left, top-right, bottom-right of16x16.
    const int row=64*wg+16*w+(lane&7)+8*((lane>>3)&1);
#pragma unroll
    for(int k=0;k<4;++k){
      const uint32_t addr=xb+row*128+(((2*k+(lane>>4))^(row&7))*16);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        :"=r"(q[4*k]),"=r"(q[4*k+1]),"=r"(q[4*k+2]),"=r"(q[4*k+3]):"r"(addr):"memory");
    }
    // Each lane orders its generic reads; lane0 releases the wholewarp.
    asm volatile("fence.proxy.async.shared::cta;\n":::"memory");
    uint32_t fullmask;asm volatile("mov.u32 %0,0xffffffff;" :"=r"(fullmask));
    asm volatile("bar.warp.sync %0;\n"::"r"(fullmask):"memory");
    if(lane==0)rq_arrive(xempty+8*slot);
  };
  if(tid==256){
    for(int s=0;s<WST;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(wfull+8*s));
    for(int s=0;s<BST;++s){
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(bfull+8*s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0],8;\n"::"r"(bempty+8*s));
    }
    for(int s=0;s<2;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n"::"r"(xfull+8*s));
    for(int s=0;s<2;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],8;\n"::"r"(xempty+8*s));
    asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
    for(int s=0;s<2;++s){
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
      // One elected thread owns this parity's Xslot; acquire before overwrite.
      if(tid==256+128*owner && s+2<KC){
        rq_wait(xempty+8*(s&1),(s/2)&1);
        rq_expect(xfull+8*(s&1),16384);
        rq_tile(xring+(s&1)*16384,&tx,(s+2)*64,0,xfull+8*(s&1));
      }
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

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  auto k = rq_tma_x_async<NT, ST, GS, PRB, WM, INS, FL>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if(info==4){
      if(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess){cudaGetLastError();return -6;}
      int occupancy=0;
      if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,k,512,C::SMEM)!=cudaSuccess){cudaGetLastError();return -7;}
      return occupancy;
    }
    if(info==5)return a.maxDynamicSharedSizeBytes;
    if(info==6)return a.maxThreadsPerBlock;
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if(p.m!=128 || p.N!=8192 || p.K!=4096 || p.KC!=64 || KS!=1) return -4;
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(512), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info, torch::Tensor xp) {
  RqP p = {};
  if (!info) {
    if(!x.is_cuda() || !x.is_contiguous() || (uintptr_t)x.data_ptr()%4) return -14;
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    p.X = (const __nv_bfloat16*)x.data_ptr(); p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
case 0: return rq_go<128,8,1,0,0,0,20>(p,x,out,(int)KS,(int)info);
  }
  return -2;
}
