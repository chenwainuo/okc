"""Kernel-only durations via CUPTI (torch.profiler) for chosen shapes, graph-replayed after an L2 flush."""
import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(Path(__file__).with_name("submission.py")))
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler,router")
    ap.add_argument("--ms", default="1,8,16,32")
    ap.add_argument("--nb", default="")
    ap.add_argument("--smem", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--torch", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("submission", args.sub)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for kv in filter(None, args.nb.split(",")):
        k, v = kv.split("=")
        mod.NB_OF[k] = int(v)
    if args.smem:
        mod.SMEM_BUDGET = args.smem * 1024
    weights = reference.generate_weights(1234)
    state = mod.prepare(weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    fview = flush.view(torch.float32)
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    for layer in args.layers.split(","):
        for m in [int(v) for v in args.ms.split(",")]:
            data = reference.generate_input(m, layer, 1234 + m, dist="normal")
            fns = [("mine", lambda: mod.kernel_fn(data, state))]
            if args.torch:
                w = weights[layer]
                n = reference.out_features(layer, m)
                wt = w[:n].t().contiguous() if layer != "router" else None
                if layer != "router":
                    fns.append(("torch", lambda: torch.mm(data["x"], wt)))
            for name, fn in fns:
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
                durs = {}
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.reps):
                        torch.sum(fview, 0, out=sink)
                        torch.cuda.synchronize()
                        g.replay()
                        torch.cuda.synchronize()
                for ev in prof.events():
                    if ev.device_type.name == "CUDA" and "sum" not in ev.name and "reduce" not in ev.name.lower():
                        durs.setdefault(ev.name[:40], []).append(ev.time_range.elapsed_us())
                desc = "  ".join(f"{k}: {statistics.median(v):.2f}us(n={len(v)})" for k, v in durs.items())
                print(f"{layer:9s} m={m:<3d} {name:5s} {desc}", flush=True)


if __name__ == "__main__":
    main()
