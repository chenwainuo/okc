"""P1 reduction-mechanism probe (H100 box): 8 MB read by G CTAs + an S-way cross-CTA reduction of E fp32 outputs.
Modes: 0 no reduction (each CTA stores its own slot) | 1 red.add into a torch-zeroed output (fill kernel node) |
2 red.add, cudaMemsetAsync node | 3 PDL: zero kernel + programmatic-serialised main kernel (griddepcontrol.wait before
reds) | 4 in-kernel zero by split 0 of each tile (warp 0, zero+fence before its loads) + ticket/flag, no memset node |
5 cluster-S DSMEM | 6 last-arriver ws (self-cleaning counter) | 7 red into persistent acc + last-arriver copy & re-zero |
8 like 4 but the flag poll is prefetched mid-stream.
Timing = the bake's _time_us (read flush queued ahead, events around one replay, median)."""
import sys, time, torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ldvol(const int* p) { int v; asm volatile("ld.volatile.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p)); return v; }
__global__ void zk(float* o, int n) {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = 0.f;
}
template <int MODE, int NL, int S>
__global__ void __launch_bounds__(256) rk(const uint4* buf, float* out, float* ws, int* cnt, int E) {
  constexpr int NT = 256;
  constexpr bool XZ = MODE == 17 || MODE == 18 || MODE == 20;  // extra zeroer CTAs: blocks [0, tiles) zero one tile each and raise its flag
  const int G = XZ ? gridDim.x * S / (S + 1) : gridDim.x, tiles = G / S, b = XZ ? (int)blockIdx.x - tiles : (int)blockIdx.x;
  const int tid = threadIdx.x, PER = E / tiles;
  if (XZ && b < 0) {
    float* oz = out + (size_t)blockIdx.x * PER;
    for (int i = tid; i < PER; i += NT) oz[i] = 0.f;
    __threadfence(); __syncthreads();
    if (tid == 0) atomicAdd(cnt + 2 * blockIdx.x + 1, 1);
    return;
  }
  const int s = b % S, tile = b / S;
  const uint4* src = buf + (size_t)b * NT * NL;
  float* ot = out + (size_t)tile * PER;
  __shared__ int tk; __shared__ int last;
  if ((MODE == 4 || MODE == 8 || MODE >= 14) && tid < 32) {
    if (tid == 0) tk = atomicAdd(cnt + 2 * tile, 1);
    if (s == 0 && !XZ) { for (int i = tid; i < PER; i += 32) ot[i] = 0.f; __threadfence(); __syncwarp(); if (tid == 0) atomicAdd(cnt + 2 * tile + 1, 1); }
  }
  uint4 v[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) v[i] = gnc(src + i * NT + tid);
  float acc = 0.f; int pre = 0;
#pragma unroll
  for (int i = 0; i < NL; i++) {
    acc += __uint_as_float((v[i].x ^ v[i].y ^ v[i].z ^ v[i].w) & 0x3f7fffffu);
    if (MODE == 8 && i == NL / 2 && tid == 0) pre = ldvol(cnt + 2 * tile + 1);
    if ((MODE == 15 || MODE == 17) && i == NL / 2 && tid == 0) pre = ldacq(cnt + 2 * tile + 1);
    if ((MODE == 19 || MODE == 20) && i == NL / 2 && tid == 0) pre = ldvol(cnt + 2 * tile + 1);
  }
  if (MODE == 0) { for (int i = tid; i < PER; i += NT) ws[(size_t)b * PER + i] = acc + i; return; }
  if (MODE == 9) { for (int i = tid; i < PER; i += NT) atomicAdd(ot + i, acc + i); return; }       // reds, no zero (timing only)
  if (MODE == 10) { for (int i = tid; i < PER; i += NT) ot[i] = acc + i; return; }                // racing plain stores
  if (MODE == 11) { for (int i = tid; i < PER; i += NT) atomicAdd(ws + (size_t)b * PER + i, acc + i); return; }  // uncontended reds
  if (MODE == 12) { if (acc == 1.2345f) ws[tid] = acc; return; }                                   // no epilogue
  if (MODE == 13) { for (int i = tid; i < PER; i += NT) __stcg(ws + (size_t)b * PER + i, acc + i); __threadfence(); return; }  // stores + fence
  if (MODE <= 3) {
#if __CUDA_ARCH__ >= 900
    if (MODE == 3) asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    for (int i = tid; i < PER; i += NT) atomicAdd(ot + i, acc + i);
    return;
  }
  if (MODE >= 14) {
    __syncthreads();
    if (tid == 0 && MODE != 16) {
      const int target = tk / S + 1;
      if (MODE >= 19) { if (pre < target) while (ldvol(cnt + 2 * tile + 1) < target) {} }
      else if (!((MODE == 15 || MODE == 17) && pre >= target)) while (ldacq(cnt + 2 * tile + 1) < target) {}
    }
    __syncthreads();
    for (int i = tid; i < PER; i += NT) atomicAdd(ot + i, acc + i);
    return;
  }
  if (MODE == 4 || MODE == 8) {
    __syncthreads();
    if (tid == 0) {
      const int target = tk / S + 1;
      if (!(MODE == 8 && pre >= target)) while (ldvol(cnt + 2 * tile + 1) < target) {}
      __threadfence();
    }
    __syncthreads();
    for (int i = tid; i < PER; i += NT) atomicAdd(ot + i, acc + i);
    return;
  }
  if (MODE == 5) {
#if __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smb[]; float* sm = (float*)smb;
    for (int i = tid; i < PER; i += NT) sm[i] = acc + i;
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    const int per = PER / S;
    const uint32_t ca = (uint32_t)__cvta_generic_to_shared(sm);
    for (int i = s * per + tid; i < (s + 1) * per; i += NT) {
      float t = 0.f;
#pragma unroll
      for (int q = 0; q < S; q++) {
        uint32_t ra; float f;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + i * 4), "r"(q));
        asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(f) : "r"(ra) : "memory");
        t += f;
      }
      ot[i] = t;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#endif
    return;
  }
  if (MODE == 6) {
    for (int i = tid; i < PER; i += NT) __stcg(ws + (size_t)b * PER + i, acc + i);
    __threadfence(); __syncthreads();
    if (tid == 0) last = atomicAdd(cnt + 2048 + tile, 1) == S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = tid; i < PER; i += NT) { float t = 0.f; for (int q = 0; q < S; q++) t += __ldcg(ws + (size_t)(tile * S + q) * PER + i); ot[i] = t; }
    if (tid == 0) cnt[2048 + tile] = 0;
    return;
  }
  if (MODE == 7) {
    float* at = ws + (size_t)tile * PER;
    for (int i = tid; i < PER; i += NT) atomicAdd(at + i, acc + i);
    __threadfence(); __syncthreads();
    if (tid == 0) last = atomicAdd(cnt + 2048 + tile, 1) == S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = tid; i < PER; i += NT) { ot[i] = __ldcg(at + i); __stcg(at + i, 0.f); }
    if (tid == 0) cnt[2048 + tile] = 0;
    return;
  }
}
template <int MODE, int NL, int S>
static int launch(torch::Tensor buf, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int E) {
  const int G = (int)(buf.numel() / (256 * NL * 16));
  auto st = at::cuda::getCurrentCUDAStream().stream();
  float* o = out.data_ptr<float>();
  if (MODE == 1) out.narrow(0, 0, E).zero_();
  if (MODE == 2) cudaMemsetAsync(o, 0, E * 4, st);
  if (MODE == 3) zk<<<4, 256, 0, st>>>(o, E);
  cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
  cfg.gridDim = dim3((MODE == 17 || MODE == 18 || MODE == 20) ? G + G / S : G); cfg.blockDim = dim3(256); cfg.stream = st;
  const int PER = E / (G / S);
  cfg.dynamicSmemBytes = MODE == 5 ? PER * 4 : 0;
  if (MODE == 5) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
    if (S > 8) cudaFuncSetAttribute(rk<MODE, NL, S>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); }
  if (MODE == 3) { at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1; }
  cudaError_t r = cudaLaunchKernelEx(&cfg, rk<MODE, NL, S>, (const uint4*)buf.data_ptr(), o, ws.data_ptr<float>(), cnt.data_ptr<int>(), E);
  if (r != cudaSuccess) { cudaGetLastError(); return (int)r; }
  return 0;
}
// bulk-read floor at few CTAs: CTA b streams its 8 MB / G slice in 16 KB cp.async.bulk chunks through a NS-slot smem ring
__global__ void __launch_bounds__(128) bk(const uint8_t* buf, int per, int ns, float* outp) {
  extern __shared__ __align__(128) uint8_t smb[];
  uint8_t* sm = smb;
  __shared__ __align__(8) uint64_t bar[16];
  const int CH = 16384, nch = per / CH, tid = threadIdx.x;
  const uint8_t* src = buf + (size_t)blockIdx.x * per;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm), bb = (uint32_t)__cvta_generic_to_shared(bar);
  if (tid == 0) {
    for (int k = 0; k < ns; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bb + k * 8));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int k = 0; k < ns && k < nch; k++) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bb + k * 8), "r"(CH) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(sb + k * CH), "l"(src + (size_t)k * CH), "r"(CH), "r"(bb + k * 8) : "memory");
    }
  }
  __syncthreads();
  float acc = 0.f;
  for (int c = 0; c < nch; c++) {
    const int k = c % ns; const uint32_t ph = (c / ns) & 1;
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bb + k * 8), "r"(ph) : "memory");
    for (int i = tid; i < CH / 16; i += 128) { uint4 v = ((uint4*)(sm + k * CH))[i]; acc += __uint_as_float((v.x ^ v.y ^ v.z ^ v.w) & 0x3f7fffffu); }
    __syncthreads();
    if (tid == 0 && c + ns < nch) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bb + k * 8), "r"(CH) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(sb + k * CH), "l"(src + (size_t)(c + ns) * CH), "r"(CH), "r"(bb + k * 8) : "memory");
    }
  }
  if (acc == 1.2345f) outp[tid] = acc;
}
int runbk(int G, int ns, torch::Tensor buf, torch::Tensor out) {
  const int per = (int)(buf.numel() / G) / 16384 * 16384;
  cudaFuncSetAttribute(bk, cudaFuncAttributeMaxDynamicSharedMemorySize, ns * 16384);
  bk<<<G, 128, ns * 16384, at::cuda::getCurrentCUDAStream().stream()>>>((const uint8_t*)buf.data_ptr(), per, ns, out.data_ptr<float>());
  cudaError_t r = cudaGetLastError(); return (int)r;
}
typedef int (*LF)(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
#define ROW(M) launch<M, 8, 8>, launch<M, 16, 8>, launch<M, 8, 16>, launch<M, 16, 16>
static LF tab[] = {ROW(0), ROW(1), ROW(2), ROW(3), ROW(4), ROW(5), ROW(6), ROW(7), ROW(8), ROW(9), ROW(10), ROW(11), ROW(12), ROW(13), ROW(14), ROW(15), ROW(16), ROW(17), ROW(18), ROW(19), ROW(20)};
int run(int mode, int nl, int s, torch::Tensor buf, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int E) {
  return tab[mode * 4 + (nl == 16 ? 1 : 0) + (s == 16 ? 2 : 0)](buf, out, ws, cnt, E);
}
"""
CPP = "int runbk(int G, int ns, torch::Tensor buf, torch::Tensor out); int run(int mode, int nl, int s, torch::Tensor buf, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int E);"


def build():
    return load_inline("p1probe", CPP, cuda_sources=SRC, functions=["run", "runbk"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)


def tus(fn, flush, sink, it=41):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(it):
        torch.sum(flush, 0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def main():
    ext = build()
    if "--build" in sys.argv:
        print("built"); return
    dev = "cuda"
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev); sink = torch.empty((), device=dev)
    buf = torch.randint(0, 255, (8 << 20,), dtype=torch.uint8, device=dev)
    out = torch.zeros(1 << 16, device=dev); ws = torch.zeros(1 << 20, device=dev); cnt = torch.zeros(1 << 12, dtype=torch.int32, device=dev)
    z = torch.zeros(1, device=dev)
    if "--bulk" in sys.argv:
        for G, ns in ((32, 12), (48, 12), (64, 8), (64, 12), (66, 12), (128, 4), (128, 8), (132, 8), (256, 2), (256, 4)):
            r = ext.runbk(G, ns, buf, out); torch.cuda.synchronize()
            print("BULK G%d ns%d MB %.2f %s" % (G, ns, G * ((8 << 20) // G // 16384) * 16384 / 2**20, ("err%d" % r) if r else "%.2f" % tus(lambda: ext.runbk(G, ns, buf, out), flush, sink)), flush=True)
        return
    modes = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else list(range(9))
    for rep in range(2):
        print("empty-fill %.2f" % tus(lambda: z.zero_(), flush, sink))
        print("memset8k %.2f" % tus(lambda: out[:2048].zero_(), flush, sink))
        for E in (2048, 8192):
            for nl in (8, 16):
                for S in (8, 16):
                    row = []; cnt.zero_()
                    for mode in modes:
                        code = ext.run(mode, nl, S, buf, out, ws, cnt, E)
                        if code:
                            row.append("m%d:err%d" % (mode, code)); continue
                        torch.cuda.synchronize()
                        row.append("m%d:%.2f" % (mode, tus(lambda: ext.run(mode, nl, S, buf, out, ws, cnt, E), flush, sink)))
                    print("E%d G%d S%d  %s" % (E, (8 << 20) // (256 * nl * 16), S, " ".join(row)), flush=True)
    if "--check" not in sys.argv: return
    # correctness of the reduction modes: sum check vs mode 0 partials
    for mode in (14, 19, 20):
        for nl, S, E in ((8, 8, 2048), (16, 16, 8192)):
            cnt.zero_(); ext.run(0, nl, S, buf, out, ws, cnt, E); torch.cuda.synchronize()
            G = (8 << 20) // (256 * nl * 16); tiles = G // S; PER = E // tiles
            ref = ws[: G * PER].view(tiles, S, PER).sum(1).flatten().clone()
            out.fill_(123.0)
            for _ in range(3):
                ext.run(mode, nl, S, buf, out, ws if mode not in (6, 7) else torch.zeros(1 << 20, device=dev), cnt, E)
            torch.cuda.synchronize()
            ok = torch.allclose(out[:E], ref, rtol=1e-5, atol=1e-3)
            print("CHECK mode %d nl %d S %d: %s" % (mode, nl, S, "ok" if ok else "BAD %g" % (out[:E] - ref).abs().max()))


if __name__ == "__main__":
    main()
