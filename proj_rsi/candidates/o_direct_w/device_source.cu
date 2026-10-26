
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
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
template <> __device__ __forceinline__ void rq_wgmma<32>(float (&D)[16], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgmma<128>(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}


// Fixed policy: HALF64 channels, KS1, Q-prefetch4, X ring8, plain TMA store.
// All codec, escape, descriptor, and stmatrix helpers are inherited unchanged from rs32.
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
struct RqC {
  static constexpr int XB = NT * 128, WB = 0, ND = NT / 2;
  static constexpr int SMEM = 1024 + ST * XB + 16 * ST;
};
__device__ __forceinline__ uint4 od_ldg(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
__global__ void __launch_bounds__(256,1)
rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  static_assert((NT == 32 || NT == 128) && ST == 8 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && FL == 20,
                "Only two shape instances of one fixed policy are allowed");
  constexpr int XB = NT * 128, ND = NT / 2, KC = 16;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem), base = (raw + 1023u) & ~1023u;
  uint32_t full = base + ST * XB, empty = full + 8 * ST;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int nb = blockIdx.x / 2, half = blockIdx.x & 1, cb = blockIdx.x * 64;
  const uint8_t* wp = p.W + ((size_t)nb * KC * 8 + half * 4 + (warp & 3)) * 1536 + lane * 16;
  const uint32_t* el = p.esc + (size_t)(nb * 8 + half * 4 + (warp & 3)) * p.ecap;
  uint32_t ec = warp < 4 ? __ldg(el + lane) : 0u;
  uint4 q0[3], q1[3], q2[3], q3[3];
  auto prefetch = [&](int s, uint4 (&q)[3]) {
    const uint8_t* a = wp + s * 12288;
    q[0] = od_ldg(a); q[1] = od_ldg(a + 512); q[2] = od_ldg(a + 1024);
  };
  // W requests are issued before the first CTA barrier and independently of X's full barrier.
  if (warp < 4) { prefetch(0,q0); prefetch(1,q1); prefetch(2,q2); prefetch(3,q3); }
  if (tid == 128) {
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0],4;\n" :: "r"(empty + s * 8));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      rq_expect(full + s * 8, XB);
      rq_tile(base + s * XB, &tx, s * 64, 0, full + s * 8);
    }
  }
  __syncthreads();
  if (warp >= 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (tid == 128) {
#pragma unroll
      for (int s = ST; s < KC; ++s) {
        rq_wait(empty + (s % ST) * 8, ((s / ST) & 1) ^ 1);
        rq_expect(full + (s % ST) * 8, XB);
        rq_tile(base + (s % ST) * XB, &tx, s * 64, 0, full + (s % ST) * 8);
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  float d[ND] = {};
  uint32_t a0[16], a1[16], nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  int ei = 0;
  auto release = [&](int s) {
    __syncwarp();
    if (lane == 0) rq_arrive(empty + (s % ST) * 8);
  };
  auto round = [&](int s, uint4 (&q)[3], uint32_t (&a)[16], uint32_t (&old)[16]) {
    rq_unit(q[0],q[1],q[2],p.Tb,a);
    rq_fix(a,s,nx,nxs,ei,ec,el,lane);
    // Only X participates in this dependency; W is already decoded from prefetched GPRs.
    rq_wait(full + (s % ST) * 8, (s / ST) & 1);
    rq_facc<ND>(d); rq_fa(a);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    uint64_t xd = rq_desc(base + (s % ST) * XB);
#pragma unroll
    for (int k = 0; k < 4; ++k) rq_wgmma<NT>(d,a + k * 4,xd + k * 2,1);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(d);
    if (s + 4 < KC) prefetch(s + 4,q);
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    if (s > 0) { rq_fa(old); release(s - 1); }
  };
#pragma unroll 1
  for (int s = 0; s < KC; s += 4) {
    round(s,q0,a0,a1); round(s+1,q1,a1,a0);
    round(s+2,q2,a0,a1); round(s+3,q3,a1,a0);
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rq_facc<ND>(d); rq_fa(a0); rq_fa(a1); release(KC-1);
  // Same HALF KS1 epilogue as rs32: one plain store; no memset/reduce/counter/finaliser.
  asm volatile("bar.sync 1,128;\n" ::: "memory");
  int w = warp & 3, j = lane >> 3, rr = lane & 7;
#pragma unroll
  for (int pp = 0; pp < NT / 16; ++pp) {
    int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
    uint32_t dst = base + tok * 128 + ((ch ^ rr) << 4);
    uint32_t r0 = rq_pk(d[4*i0],d[4*i0+1]), r1 = rq_pk(d[4*i0+2],d[4*i0+3]);
    uint32_t r2 = rq_pk(d[4*i0+4],d[4*i0+5]), r3 = rq_pk(d[4*i0+6],d[4*i0+7]);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0],{%1,%2,%3,%4};\n"
                 :: "r"(dst),"r"(r0),"r"(r1),"r"(r2),"r"(r3) : "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;\nbar.sync 2,128;\n" ::: "memory");
  if (tid == 0) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,0}],[%2];\n"
                 :: "l"((unsigned long long)&to),"r"(cb),"r"(base) : "memory");
    asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;\n" ::: "memory");
  }
#endif
}


void instantiate_32(RqP p,RqTmap tx,RqTmap to) { rq<32,8,1,0,0,0,20><<<128,256,RqC<32,8,1,0,0,0,20>::SMEM>>>(p,tx,to); }

void instantiate_128(RqP p,RqTmap tx,RqTmap to) { rq<128,8,1,0,0,0,20><<<128,256,RqC<128,8,1,0,0,0,20>::SMEM>>>(p,tx,to); }
