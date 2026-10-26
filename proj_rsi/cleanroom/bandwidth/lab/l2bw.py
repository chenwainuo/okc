"""L2->SM bandwidth: all SMs re-read a 4 MB buffer; also HBM read rate on a 64 MB buffer; also red.global.add.f32 throughput."""
import torch, statistics
from torch.utils.cpp_extension import load_inline
src = r'''
#include <cuda_runtime.h>
#include <cstdint>
__global__ void rd(const int4* __restrict__ p, size_t n16, int passes, int* sink) {
    int4 acc = make_int4(0,0,0,0);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int ps = 0; ps < passes; ++ps)
      for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        int4 v = __ldcg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
      }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *sink = 1;
}
__global__ void redadd(float* __restrict__ p, size_t n, int reps) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int r = 0; r < reps; ++r)
      for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(p + i, 1.0f);
}
void run_rd(torch::Tensor t, int passes, int blocks, torch::Tensor sink) {
    rd<<<blocks, 256>>>((const int4*)t.data_ptr(), t.numel() / 16, passes, (int*)sink.data_ptr());
}
void run_red(torch::Tensor t, int reps, int blocks) {
    redadd<<<blocks, 256>>>((float*)t.data_ptr(), t.numel(), reps);
}
'''
mod = load_inline("bandwidth_l2bw", cpp_sources="void run_rd(torch::Tensor, int, int, torch::Tensor); void run_red(torch::Tensor, int, int);",
                  cuda_sources=src, functions=["run_rd", "run_red"], extra_cuda_cflags=["-O3"], verbose=False)
sink = torch.zeros(1, dtype=torch.int32, device="cuda")
def timeit(fn, reps=10):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)
for mb in (2, 4, 8, 16):
    t = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
    for blocks in (132, 264, 528):
        passes = max(1, 512 // mb)
        us = timeit(lambda: mod.run_rd(t, passes, blocks, sink))
        print(f"L2 read {mb} MB x{passes} passes, {blocks} blocks: {us:.1f} us -> {mb*passes*1.048576/us*1e3/1e3:.2f} TB/s")
t = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
for blocks in (132, 264, 528, 1056):
    us = timeit(lambda: mod.run_rd(t, 1, blocks, sink))
    print(f"HBM read 512 MB, {blocks} blocks: {us:.1f} us -> {512*1.048576/us*1e3/1e3:.2f} TB/s")
for n in (1 << 16, 1 << 18, 1 << 20):
    f = torch.zeros(n, dtype=torch.float32, device="cuda")
    for blocks in (132, 528):
        us = timeit(lambda: mod.run_red(f, 8, blocks))
        print(f"red.add f32 {n} addrs x8 reps, {blocks} blocks: {us:.1f} us -> {n*8/us/1e3:.1f} G atomics/s")
