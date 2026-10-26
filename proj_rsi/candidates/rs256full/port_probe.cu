// One fixed geometry. Diagnostic only: no dispatch, prepare, matmul API, or submission.
// Build: nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v port_probe.cu -o /tmp/rs256full_port_probe
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int XBYTES = 32768, WBYTES = 49152, PAYLOAD = XBYTES + WBYTES;
constexpr int SMEM = 2 * PAYLOAD + 1024 + 64;
constexpr int CTAS = 128, THREADS = 384, ND = 32;
struct alignas(64) TensorMap { unsigned long long v[16]; };
static_assert(sizeof(TensorMap) == sizeof(CUtensorMap), "tensor map size");
struct Stamp {
  unsigned long long cb[2], ce[2], pb, pe;
  unsigned int writes, progress, smid, writes_at_consumer_end[2];
};
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__device__ __forceinline__ void waitbar(uint32_t b, unsigned ph) {
  asm volatile("{.reg .pred p; W_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n"
               "@!p bra W_%=;}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void expect(uint32_t b, unsigned n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t d, const void* s, unsigned n, uint32_t b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];\n"
               :: "r"(d), "l"(s), "r"(n), "r"(b) : "memory");
}
__device__ __forceinline__ void tile(uint32_t d, const TensorMap* tm, int k, uint32_t b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,0}],[%3];\n"
               :: "r"(d), "l"((unsigned long long)tm), "r"(k), "r"(b) : "memory");
}
__device__ __forceinline__ uint4 lds(uint32_t a) {
  uint4 q;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];\n"
               : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "r"(a) : "memory");
  return q;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d; asm("prmt.b32 %0,%1,%2,%3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
// Same dec1 operations and original [k64][8 warps][1536] pack as rs32/RQ.
__device__ __forceinline__ void dec(uint32_t a, uint32_t b, uint32_t c, uint32_t tb, uint32_t* o) {
  uint32_t hl = (c & 0x87878787u) + tb;
  uint32_t hh = (__funnelshift_r(c, c, 4) & 0x87878787u) + tb;
  o[0] = prmt(a, hl, 0x5140); o[1] = prmt(a, hl, 0x7362);
  o[2] = prmt(b, hh, 0x5140); o[3] = prmt(b, hh, 0x7362);
}
__device__ __forceinline__ void unit(uint4 q0, uint4 q1, uint4 q2, uint32_t tb, uint32_t* a) {
  dec(q0.x, q0.y, q2.x, tb, a); dec(q0.z, q0.w, q2.y, tb, a + 4);
  dec(q1.x, q1.y, q2.z, tb, a + 8); dec(q1.z, q1.w, q2.w, tb, a + 12);
}
__device__ __forceinline__ uint64_t desc(uint32_t p) {
  return ((p & 0x3ffffu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void mma(float (&d)[ND], const uint32_t* a, uint64_t b, int scd) {
  asm volatile("{.reg .pred p; setp.ne.b32 p,%37,0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "{%32,%33,%34,%35},%36,p,1,1,0;}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scd));
}
template<int N> __device__ __forceinline__ void fregs(uint32_t (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(a[i]) :: "memory");
}
__device__ __forceinline__ void facc(float (&d)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

// MODE 1: RS + actual packed-W LDS/dec1; 2: TMA only;
// MODE 3: same RS/LDS/dec1 plus paced concurrent 80 KiB TMA fills into disjoint slot 1.
// All operands reside in slot 0. Producer never overwrites a consumer operand.
template<int MODE>
__global__ __launch_bounds__(THREADS, 1)
void probe(const uint8_t* input, float* output, uint8_t* snapshots, Stamp* stamps, int iters, uint32_t tb,
           const __grid_constant__ TensorMap tx) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t scratch[];
  uint32_t raw = (uint32_t)__cvta_generic_to_shared(scratch);
  uint32_t base = (raw + 1023u) & ~1023u;
  uint32_t initbar = base + 2 * PAYLOAD, pbar = initbar + 8, seq = pbar + 8, completed = seq + 4;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  Stamp& st = stamps[blockIdx.x];
  if (tid == 256) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n" :: "r"(initbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n" :: "r"(pbar));
    asm volatile("st.shared.u32 [%0],0;\n" :: "r"(seq) : "memory");
    asm volatile("st.shared.u32 [%0],0;\n" :: "r"(completed) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    expect(initbar, PAYLOAD);
    bulk(base, input, PAYLOAD, initbar);
  }
  __syncthreads();
  waitbar(initbar, 0);
  __syncthreads();
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (tid == 256) {
      unsigned sm;
      asm volatile("mov.u32 %0,%%smid;\n" : "=r"(sm));
      st.smid = sm;
      st.pb = clock64();
      if constexpr (MODE == 2 || MODE == 3) {
        for (int r = 0; r < iters; ++r) {
          if constexpr (MODE == 3) {
            if (r >= 2) {
              unsigned s;
              do {
                asm volatile("ld.acquire.cta.shared.u32 %0,[%1];\n" : "=r"(s) : "r"(seq) : "memory");
                if (s < (unsigned)(r - 1)) __nanosleep(64);
              } while (s < (unsigned)(r - 1));
            }
          }
          expect(pbar, PAYLOAD);
          bulk(base + PAYLOAD + XBYTES, input + XBYTES, WBYTES, pbar);
#pragma unroll
          for (int j = 0; j < 4; ++j) tile(base + PAYLOAD + j * 8192, &tx, j * 64, pbar);
          waitbar(pbar, r & 1);
          asm volatile("st.release.cta.shared.u32 [%0],%1;\n" :: "r"(completed), "r"(r + 1) : "memory");
        }
        st.writes = iters;
      } else st.writes = 0;
      st.pe = clock64();
    }
    asm volatile("bar.sync 1,384;\n" ::: "memory");
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int wg = warp / 4;
    float d[ND] = {};
    uint32_t a0[4][16], a1[4][16];
    auto load = [&](uint32_t (&a)[4][16]) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t wa = base + XBYTES + j * 12288 + warp * 1536 + lane * 16;
        uint4 q0 = lds(wa), q1 = lds(wa + 512), q2 = lds(wa + 1024);
        unit(q0, q1, q2, tb, a[j]);
      }
    };
    if constexpr (MODE != 2) {
      if ((tid & 127) == 0) st.cb[wg] = clock64();
      load(a0);
      auto round = [&](int r, uint32_t (&a)[4][16], uint32_t (&an)[4][16]) {
        facc(d);
#pragma unroll
        for (int j = 0; j < 4; ++j) fregs(a[j]);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint64_t xd = desc(base + j * 8192);
#pragma unroll
          for (int q = 0; q < 4; ++q) mma(d, a[j] + q * 4, xd + 2 * q, 1);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        facc(d);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        // The previous group's A has retired; it is safe to overwrite the other buffer.
        if (r > 0) {
#pragma unroll
          for (int j = 0; j < 4; ++j) fregs(an[j]);
        }
        if (tid == 0) asm volatile("st.release.cta.shared.u32 [%0],%1;\n" :: "r"(seq), "r"(r + 1) : "memory");
        if (r + 1 < iters) load(an);
      };
#pragma unroll 1
      for (int r = 0; r < iters; r += 2) {
        round(r, a0, a1);
        round(r + 1, a1, a0);
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      facc(d);
      if ((tid & 127) == 0) {
        st.ce[wg] = clock64();
        unsigned n;
        asm volatile("ld.acquire.cta.shared.u32 %0,[%1];\n" : "=r"(n) : "r"(completed) : "memory");
        st.writes_at_consumer_end[wg] = n;
      }
    }
    asm volatile("bar.sync 1,384;\n" ::: "memory");
    // Every accumulator is externally observed, outside BOTH measured windows.
    // Never replace this with a D[0] sink.
#pragma unroll
    for (int i = 0; i < ND; ++i)
      output[((size_t)blockIdx.x * 256 + tid) * ND + i] = MODE == 2 ? 0.0f : d[i];
  }
  __syncthreads();
  if (tid == 256) {
    asm volatile("ld.shared.u32 %0,[%1];\n" : "=r"(st.progress) : "r"(seq) : "memory");
  }
  if constexpr (MODE == 2 || MODE == 3) {
    // Full byte-for-byte producer result, read only after its completion wait and CTA barrier.
    // Outside both producer and consumer timestamp windows.
    for (int i = tid; i < PAYLOAD / 4; i += THREADS) {
      uint32_t v;
      asm volatile("ld.shared.u32 %0,[%1];\n" : "=r"(v) : "r"(base + PAYLOAD + i * 4) : "memory");
      ((uint32_t*)snapshots)[(size_t)blockIdx.x * (PAYLOAD / 4) + i] = v;
    }
  }
#endif
}

template<int MODE> void launch(const uint8_t* in, float* out, uint8_t* snaps, Stamp* st, int iters, const TensorMap& tx) {
  CK(cudaFuncSetAttribute(probe<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  probe<MODE><<<CTAS, THREADS, SMEM>>>(in, out, snaps, st, iters, 0x3f3f3f3fu, tx);
  CK(cudaGetLastError());
}
static void run(int mode, const uint8_t* in, float* out, uint8_t* snaps, Stamp* st, int iters, const TensorMap& tx) {
  switch (mode) {
    case 1: launch<1>(in, out, snaps, st, iters, tx); break;
    case 2: launch<2>(in, out, snaps, st, iters, tx); break;
    case 3: launch<3>(in, out, snaps, st, iters, tx); break;
    default: exit(2);
  }
}
static double median(std::vector<double> a) {
  std::sort(a.begin(), a.end()); return a.size() % 2 ? a[a.size()/2] : (a[a.size()/2-1] + a[a.size()/2]) / 2;
}
template<int MODE> static void info() {
  cudaFuncAttributes a;
  CK(cudaFuncGetAttributes(&a, probe<MODE>));
  int occ;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, probe<MODE>, THREADS, SMEM));
  printf("ATTR mode=%d regs=%d local=%zu dynamic_smem=%d ctas_per_sm=%d\n", MODE, a.numRegs, a.localSizeBytes, SMEM, occ);
  if (a.localSizeBytes || occ != 1) { fprintf(stderr, "REJECT spills or occupancy\n"); exit(3); }
}
// Same 2D {64 K,64 tokens}/SW128 map as RQ. Four boxes make a K256 stage.
static TensorMap make_map(void* input) {
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr));
  if (!fn) { fprintf(stderr, "tensor map encoder unavailable\n"); exit(2); }
  TensorMap tx;
  cuuint64_t dims[2] = {256,64}, strides[1] = {512};
  cuuint32_t box[2] = {64,64}, step[2] = {1,1};
  auto e = ((Encode)fn)((CUtensorMap*)&tx, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, input, dims, strides, box, step,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (e != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)e); exit(2); }
  return tx;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 128;
  int reps = argc > 2 ? atoi(argv[2]) : 9;
  if (iters < 16 || iters > 1024 || iters % 2 || reps < 3 || reps > 31) return 2;
  cudaDeviceProp dp;
  CK(cudaGetDeviceProperties(&dp, 0));
  if (dp.major != 9 || dp.multiProcessorCount < CTAS) { fprintf(stderr, "requires SM90a and >=128 SMs\n"); return 2; }
  printf("GEOMETRY gpu=%s sms=%d ctas=%d threads=%d token_tile=64 channel_tile=128 merged_k=256 stages=2 iters=%d reps=%d\n",
         dp.name, dp.multiProcessorCount, CTAS, THREADS, iters, reps);
  std::vector<uint8_t> hi(PAYLOAD, 0);
  for (int i = 0; i < XBYTES; i += 2) { hi[i] = 0x80; hi[i + 1] = 0x3f; }
  for (int j = 0; j < 4; ++j) for (int w = 0; w < 8; ++w)
    memset(hi.data() + XBYTES + j * 12288 + w * 1536, 0x80, 1024);
  uint8_t *di, *ds;
  float* dout;
  Stamp* dst;
  constexpr size_t count = (size_t)CTAS * 256 * ND;
  CK(cudaMalloc(&di, PAYLOAD)); CK(cudaMalloc(&ds, (size_t)CTAS * PAYLOAD));
  CK(cudaMalloc(&dout, count * sizeof(float))); CK(cudaMalloc(&dst, CTAS * sizeof(Stamp)));
  CK(cudaMemcpy(di, hi.data(), PAYLOAD, cudaMemcpyHostToDevice));
  const TensorMap tx = make_map(di);
  info<1>(); info<2>(); info<3>();
  std::vector<Stamp> hs(CTAS);
  std::vector<float> ho(count);
  std::vector<uint8_t> snap((size_t)CTAS * PAYLOAD);
  std::vector<double> batch[4], producer[4], overlap[4], coverage[4];
  const char* names[4] = {"UNUSED", "RS_LDS_DEC", "TMA", "COMBINED"};
  // Interleaved and rotated order; warmups excluded. No HBM flush: only 80KiB unique source bytes.
  for (int rep = -2; rep < reps; ++rep) for (int j = 0; j < 3; ++j) {
    int mode = 1 + (j + (rep < 0 ? 0 : rep)) % 3;
    CK(cudaMemset(dst, 0, CTAS * sizeof(Stamp)));
    run(mode, di, dout, ds, dst, iters, tx);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hs.data(), dst, CTAS * sizeof(Stamp), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho.data(), dout, count * sizeof(float), cudaMemcpyDeviceToHost));
    float want = mode == 2 ? 0.0f : (float)(256 * iters);
    for (size_t q = 0; q < count; ++q) if (ho[q] != want) {
      fprintf(stderr, "INVALID mode=%d accumulator=%zu got=%g want=%g\n", mode, q, ho[q], want); return 4;
    }
    if (mode == 2 || mode == 3) {
      CK(cudaMemcpy(snap.data(), ds, snap.size(), cudaMemcpyDeviceToHost));
      for (int c = 0; c < CTAS; ++c) if (memcmp(snap.data() + (size_t)c * PAYLOAD, hi.data(), PAYLOAD)) {
        fprintf(stderr, "INVALID mode=%d producer_snapshot_cta=%d\n", mode, c); return 4;
      }
    }
    std::vector<double> cs, ps, ov, cov;
    std::vector<unsigned> sms;
    for (const auto& s : hs) {
      if (s.writes != (unsigned)((mode == 2 || mode == 3) ? iters : 0) ||
          s.progress != (unsigned)(mode == 2 ? 0 : iters)) {
        fprintf(stderr, "INVALID mode=%d writes=%u progress=%u\n", mode, s.writes, s.progress); return 4;
      }
      sms.push_back(s.smid);
      ps.push_back((s.pe - s.pb) / (double)iters);
      if (mode != 2) {
        cs.push_back(std::max(s.ce[0] - s.cb[0], s.ce[1] - s.cb[1]) / (double)iters);
        if (mode == 3) {
          auto b = std::max(s.pb, std::min(s.cb[0], s.cb[1]));
          auto e = std::min(s.pe, std::max(s.ce[0], s.ce[1]));
          auto dur = std::max(s.ce[0], s.ce[1]) - std::min(s.cb[0], s.cb[1]);
          ov.push_back(e > b ? (e - b) / (double)dur : 0.0);
          cov.push_back(std::min(s.writes_at_consumer_end[0], s.writes_at_consumer_end[1]) / (double)iters);
        }
      }
    }
    std::sort(sms.begin(), sms.end());
    auto unique = std::unique(sms.begin(), sms.end()) - sms.begin();
    if (unique != CTAS) { fprintf(stderr, "INVALID only %td distinct SMs for %d CTAs\n", unique, CTAS); return 4; }
    if (rep >= 0) {
      double c = cs.empty() ? 0.0 : median(cs), p = median(ps), o = ov.empty() ? 0.0 : median(ov);
      double v = cov.empty() ? 0.0 : median(cov);
      batch[mode].push_back(c); producer[mode].push_back(p); overlap[mode].push_back(o);
      coverage[mode].push_back(v);
      printf("SAMPLE rep=%d mode=%d name=%s consumer_clk_per_k256=%.3f producer_clk_per_80KiB=%.3f overlap=%.5f writes_per_consumer_round=%.5f all_accumulators=EXACT producer_bytes=%s distinct_sms=%td\n",
             rep, mode, names[mode], c, p, o, v, mode == 1 ? "NA" : "EXACT", unique);
    }
  }
  for (int mode = 1; mode < 4; ++mode)
    printf("RESULT mode=%d name=%s consumer_clk_per_k256=%.3f producer_clk_per_80KiB=%.3f overlap=%.5f writes_per_consumer_round=%.5f\n", mode, names[mode], median(batch[mode]), median(producer[mode]), median(overlap[mode]), median(coverage[mode]));
  printf("RATIO combined_over_consumer=%.6f consumer_over_tensor_floor=%.6f\n", median(batch[3]) / median(batch[1]), median(batch[1]) / 1024.0);
  printf("PASS synthetic only; no end-to-end matmul, escape workload, judge, or speedup claim\n");
  CK(cudaFree(di)); CK(cudaFree(ds)); CK(cudaFree(dout)); CK(cudaFree(dst));
  return 0;
}
