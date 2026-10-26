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
