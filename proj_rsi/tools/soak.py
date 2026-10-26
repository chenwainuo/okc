"""Low-priority GPU soak: alternate two candidates over every shape in short locked bursts, forever.

Keeps the GPU busy whenever nobody else holds /tmp/gpu.lock (each burst takes the lock, times all shapes once per
candidate, releases, sleeps briefly), and accumulates a high-sample within-run comparison of A vs B.

Usage (inside the container): python proj_rsi/tools/soak.py <A.py> <B.py> [--lock /tmp/gpu.lock] [--out logs/soak.md]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_bench import LOADS, LAYERS, flush_buffer, load_submission, time_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--lock", default="/tmp/gpu.lock")
    ap.add_argument("--out", type=Path, default=Path("logs/soak.md"))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()

    weights = reference.generate_weights(1234)
    mods = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        mod = load_submission(path)
        mods[tag] = (mod, mod.prepare(weights))
        torch.cuda.synchronize()
    print("prepared both", flush=True)
    shapes = [(load, m, layer) for load, m in LOADS.items() for layer in LAYERS]
    inputs = {(m, layer): reference.generate_input(m, layer, 1234 + m) for _, m, layer in shapes}
    flush = flush_buffer()
    samples = {(load, layer): {"A": [], "B": []} for load, _, layer in shapes}
    lock_fd = open(args.lock, "w")
    rounds = 0
    while True:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            order = ("A", "B") if rounds % 2 == 0 else ("B", "A")
            for load, m, layer in shapes:
                data = inputs[(m, layer)]
                for tag in order:
                    mod, state = mods[tag]
                    samples[(load, layer)][tag].append(time_graph(lambda: mod.kernel_fn(data, state), flush, args.reps))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        rounds += 1
        lines = ["| load | layer | A med | B med | A-B | n |", "|---|---|---|---|---|---|"]
        for load, _, layer in shapes:
            a, b = samples[(load, layer)]["A"], samples[(load, layer)]["B"]
            lines.append("| %s | %s | %.2f | %.2f | %+.2f | %d |" % (load, layer, statistics.median(a), statistics.median(b),
                                                                   statistics.median(a) - statistics.median(b), len(a)))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("# soak A=%s B=%s rounds=%d\n\n%s\n" % (args.a, args.b, rounds, "\n".join(lines)))
        args.out.with_suffix(".json").write_text(json.dumps({"%s/%s" % k: v for k, v in samples.items()}))
        print("round %d done" % rounds, flush=True)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
