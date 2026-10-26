"""contrarian config sweep: python sweep.py <layer> <m> "nw,splits,stages;nw,splits,stages;..." [reps]"""
import os, sys, statistics, importlib.util
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj")); sys.path.insert(0, str(ROOT / "proj_rsi/tools"))
import reference
from h100_bench import flush_buffer, torch_baseline, load_submission


def time_graph(fn, flush, reps=20):
    """min/median replay time (us) of fn() as a CUDA graph, L2 flushed before each replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    times = []
    sink = torch.zeros(1, dtype=torch.float32, device="cuda")
    fview = flush.view(torch.float32)
    for _ in range(reps):
        torch.sum(fview, 0, out=sink)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return "min %6.2f med %6.2f" % (min(times), statistics.median(times))

READ_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void readk(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
#pragma unroll 8
    for (; i < n; i += stride) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}
void read_floor(torch::Tensor t, torch::Tensor out, int64_t blocks, int64_t threads) {
    readk<<<(int)blocks, (int)threads, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(t.data_ptr()), t.numel() / 16, out.data_ptr<uint32_t>());
}
"""

def read_floor_fn(nbytes):
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(name="contrarian_readfloor", cpp_sources=["#include <torch/extension.h>\nvoid read_floor(torch::Tensor t, torch::Tensor out, int64_t blocks, int64_t threads);"],
                      cuda_sources=["#include <ATen/cuda/CUDAContext.h>\n" + READ_SRC], functions=["read_floor"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda").view(torch.uint32) if hasattr(torch, "uint32") else torch.zeros(4, dtype=torch.int32, device="cuda")
    return lambda: ext.read_floor(buf, out, 264, 512)


def main():
    layer, m, cfgs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(";")
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    weights = reference.generate_weights(1234)
    mod = load_submission(Path(__file__).with_name("submission.py"))
    state = mod.prepare(weights)
    flush = flush_buffer()
    data = reference.generate_input(m, layer, 1234 + m)
    tiny = torch.zeros(32, device="cuda")
    print("overhead %s" % time_graph(lambda: tiny.add_(1.0), flush, reps))
    tb = time_graph(torch_baseline(layer, data, weights), flush, reps)
    print("torch %s" % tb)
    for mb in (12.6, 28.3, 50.3):
        fn = read_floor_fn(int(mb * 1e6))
        print("readfloor %.1fMB %s" % (mb, time_graph(fn, flush, reps)))
    for c in cfgs:
        if c.startswith("mode"):
            mod._MODE = int(c[4]); c = c[6:]
        mod._OVR = c
        out = mod.kernel_fn(data, state); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, weights)
        t = time_graph(lambda: mod.kernel_fn(data, state), flush, reps)
        print("cfg %-14s %s  %s %s" % (c, t, "ok" if ok else "FAIL", msg[:60]), flush=True)

if __name__ == "__main__":
    main()
