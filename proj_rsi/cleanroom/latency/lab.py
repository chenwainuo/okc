"""latency lab: check + graph-time chosen (layer, m) shapes of submission.py, with NB/S overrides.

Usage (in the container, under the GPU lock):
  python proj_rsi/cleanroom/latency/lab.py --layers router,o --ms 1,8,16,32 [--nb router=4,o=32] [--reps 15]
"""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402

H100_HW = {"hbm_bytes_per_s": 3.35e12, "bf16_dense_flops_per_s": 989e12}


def time_graph(fn, flush, reps=20):
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
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(Path(__file__).with_name("submission.py")))
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler,router")
    ap.add_argument("--ms", default="1,8,16,32")
    ap.add_argument("--nb", default="")
    ap.add_argument("--smem", type=int, default=0)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--notime", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("submission", args.sub)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for kv in filter(None, args.nb.split(",")):
        k, v = kv.split("=")
        mod.NB_OF[k] = int(v)
    if args.smem:
        mod.SMEM_BUDGET = args.smem * 1024
    weights = reference.generate_weights(args.seed)
    t0 = time.time()
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    print(f"PREPARE {time.time() - t0:.1f}s NB={mod.NB_OF}", flush=True)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    tiny = torch.zeros(32, device="cuda")
    print(f"REPLAY_OVERHEAD {time_graph(lambda: tiny.add_(1.0), flush, args.reps):.2f} us", flush=True)
    for layer in args.layers.split(","):
        for m in [int(v) for v in args.ms.split(",")]:
            data = reference.generate_input(m, layer, args.seed + m, dist="normal")
            out = mod.kernel_fn(data, state)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, weights)
            line = f"{layer:9s} m={m:<4d} {'ok ' if ok else 'FAIL'} {msg}"
            if not args.notime:
                us = time_graph(lambda: mod.kernel_fn(data, state), flush, args.reps)
                sol = reference.speed_of_light_ns(m, layer, H100_HW) / 1e3
                line += f" | {us:7.2f} us  sol {sol:6.2f}  x{us / sol:.2f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
