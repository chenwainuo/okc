"""Same-process dry-run A/B: compile/prepare first, then cold-L2 paired graphs and CUPTI.

Run only under the root scheduler's GPU lock. LATENCY_DBG is overridden for the A/B.
"""
import argparse
import hashlib
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402

CASES = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"),
         (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"),
         (9, "router", "outliers"), (512, "qkv_gate", "outliers"),
         (1000, "unpooler", "outliers")]


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler,router")
    ap.add_argument("--ms", default="1,8,32")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--profile-reps", type=int, default=5)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    path = Path(__file__).with_name("submission.py")
    spec = importlib.util.spec_from_file_location("latency_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LOADER == 3, "this experiment targets LOADER 3"
    print(f"SOURCE {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"DEVICE {torch.cuda.get_device_name()} torch={torch.__version__}", flush=True)
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    start = time.time()
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    print(f"PREPARE {time.time() - start:.2f}s (outside timing)", flush=True)
    if args.prepare_only:
        return
    if args.official:
        for i, (m, layer, dist) in enumerate(CASES):
            data = reference.generate_input(m, layer, 100 + i, dist=dist)
            for dbg in (0, 512):
                mod.DBG = dbg
                out = mod.kernel_fn(data, state)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, out, weights)
                print(f"OFFICIAL dbg={dbg} {layer} m={m} {dist}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
                if not ok:
                    raise AssertionError(msg)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    sink = torch.zeros((), dtype=torch.float32, device="cuda")

    def cold():
        torch.sum(flush.view(torch.float32), 0, out=sink)
        torch.cuda.synchronize()

    tiny = torch.zeros(32, device="cuda")
    empty = capture(lambda: tiny.add_(1))
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def elapsed(graph):
        cold()
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1000

    overhead = statistics.median(elapsed(empty) for _ in range(args.reps))
    print(f"REPLAY_OVERHEAD {overhead:.3f}us (single tiny kernel)", flush=True)
    ratios = []
    for layer in args.layers.split(","):
        for m in map(int, args.ms.split(",")):
            data = reference.generate_input(m, layer, 1234 + m)
            graphs = {}
            for dbg in (0, 512):
                mod.DBG = dbg
                out = mod.kernel_fn(data, state)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, out, weights)
                print(f"CHECK dbg={dbg} {layer} m={m}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
                if not ok:
                    raise AssertionError(msg)
                graphs[dbg] = capture(lambda: mod.kernel_fn(data, state))
            samples = {0: [], 512: []}
            paired = []
            for rep in range(args.reps):
                for dbg in ((0, 512) if rep % 2 == 0 else (512, 0)):
                    samples[dbg].append(elapsed(graphs[dbg]))
                paired.append(samples[0][-1] / samples[512][-1])
            warm, nowarm = (statistics.median(samples[dbg]) for dbg in (0, 512))
            ratio = statistics.median(paired)
            ratios.append(ratio)
            print(f"GRAPH {layer} m={m} warm={warm:.3f}us nowarm={nowarm:.3f}us ratio={ratio:.4f} "
                  f"net_warm={warm-overhead:.3f}us net_nowarm={nowarm-overhead:.3f}us", flush=True)
            if args.profile_reps:
                for dbg in (0, 512):
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(args.profile_reps):
                            cold()
                            graphs[dbg].replay()
                            torch.cuda.synchronize()
                    durations = [ev.time_range.elapsed_us() for ev in prof.events()
                                 if ev.device_type.name == "CUDA" and "latency_gemv_t" in ev.name]
                    assert len(durations) == args.profile_reps, (dbg, len(durations))
                    print(f"KERNEL {layer} m={m} dbg={dbg} {statistics.median(durations):.3f}us "
                          f"n={len(durations)}", flush=True)
            del graphs
    print(f"SUMMARY paired_graph_geomean={statistics.geometric_mean(ratios):.5f} shapes={len(ratios)}", flush=True)


if __name__ == "__main__":
    main()
