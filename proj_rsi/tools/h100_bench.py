"""Local GPU harness for muse_bf16_proj candidates (run inside the CUDA 12.8 torch container on the H100 box).

Usage: python proj_rsi/tools/h100_bench.py <submission.py> [--shapes decode1,verify16,...] [--layers pooler,...]
       [--reps 20] [--seed 1234] [--official] [--json out.json] [--no-torch-baseline]

What it does: generates the task's weights (muse_bf16_proj/reference.py), runs the candidate's prepare() (which
prints its own BAKE/PICK/PROBE lines), validates every served shape with the task's own check_implementation on the
judge-like cases, then times each (load, layer) shape the way the judge does: the call captured as a CUDA graph,
replayed after an L2 flush, median of paired replays. Prints a table with the H100 speed-of-light and, unless
disabled, torch's own time for the same shape as a cuBLAS reference. Times are kernel-side (graph replay) and
include the ~3-5 us replay overhead the judge also pays; compare within one run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402  (the task's own reference module)

LOADS = {"decode1": 1, "decode16": 16, "decode32": 32, "verify1": 8, "verify16": 128, "verify32": 256}
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]
OFFICIAL_M = [1, 7, 33, 130, 5, 8, 9, 512, 1000]
H100_HW = {"hbm_bytes_per_s": 3.35e12, "bf16_dense_flops_per_s": 989e12}


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flush_buffer():
    return torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def time_graph(fn, flush, reps=20):
    """Median replay time in microseconds of fn() captured as a CUDA graph, with an L2 flush before each replay."""
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
    for _ in range(reps):
        flush.zero_()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def torch_baseline(layer, data, weights):
    w = weights[layer]
    x = data["x"]
    if layer == "router":
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        hi_t, lo_t = hi.t().contiguous(), lo.t().contiguous()

        def fn():
            acc = torch.mm(x, hi_t, out_dtype=torch.float32)
            acc += torch.mm(x, lo_t, out_dtype=torch.float32)
            return acc
        return fn
    n = reference.out_features(layer, x.shape[0])
    wt = w[:n].t().contiguous()

    def fn():
        return torch.mm(x, wt)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--shapes", default=",".join(LOADS))
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--official", action="store_true", help="also validate the judge's official m cases")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-torch-baseline", action="store_true")
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)
    weights = reference.generate_weights(args.seed)
    mod = load_submission(args.submission)
    t0 = time.time()
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    print(f"PREPARE {time.time() - t0:.1f}s", flush=True)

    layers = args.layers.split(",")
    shapes = args.shapes.split(",")
    flush = flush_buffer()
    rows = []
    # fixed cost of one graph replay on this box (the judge pays its own; subtract to compare kernel-side time)
    tiny = torch.zeros(32, device="cuda")
    overhead = time_graph(lambda: tiny.add_(1.0), flush, args.reps)
    read6 = torch.empty(6 << 20, dtype=torch.uint8, device="cuda")
    sink6 = torch.zeros((), dtype=torch.int64, device="cuda")
    read6_us = time_graph(lambda: torch.sum(read6, 0, dtype=torch.int64, out=sink6), flush, args.reps)
    print(f"REPLAY_OVERHEAD {overhead:.2f} us  (empty graph replay)   READ6MB {read6_us:.2f} us (torch.sum of 6 MB)", flush=True)
    rows.append({"replay_overhead_us": overhead, "read6mb_us": read6_us})
    # correctness on the judge-like cases first
    ms = sorted({LOADS[s] for s in shapes} | (set(OFFICIAL_M) if args.official else set()))
    for layer in layers:
        for m in ms:
            if layer == "unpooler" and m > 8 and m not in (16, 32, 128, 256, 33, 130, 512, 1000):
                pass
            data = reference.generate_input(m, layer, args.seed + m, dist="normal")
            out = mod.kernel_fn(data, state)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, out, weights)
            print(f"CHECK {layer:9s} m={m:<4d} {'ok ' if ok else 'FAIL'} {msg}", flush=True)
            if not ok:
                rows.append({"layer": layer, "m": m, "ok": False, "msg": msg})
    # timing
    print("%-9s %-9s %4s %9s %9s %9s %7s" % ("load", "layer", "m", "yours_us", "torch_us", "sol_us", "x_sol"), flush=True)
    for load in shapes:
        m = LOADS[load]
        for layer in layers:
            data = reference.generate_input(m, layer, args.seed + m, dist="normal")
            yours = time_graph(lambda: mod.kernel_fn(data, state), flush, args.reps)
            tb = float("nan")
            if not args.no_torch_baseline:
                tb = time_graph(torch_baseline(layer, data, weights), flush, args.reps)
            sol = reference.speed_of_light_ns(m, layer, H100_HW) / 1e3
            rows.append({"load": load, "layer": layer, "m": m, "yours_us": yours, "torch_us": tb, "sol_us": sol})
            print("%-9s %-9s %4d %9.2f %9.2f %9.2f %7.2f" % (load, layer, m, yours, tb, sol, yours / sol), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
