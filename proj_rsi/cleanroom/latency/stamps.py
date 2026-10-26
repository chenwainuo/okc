"""In-kernel phase stamps of the LOADER 2 kernel (LATENCY_DBG must include 16).

Per CTA, warp 0 and warp 7 lane 0 record clock64 at: issue done (1), first stage ready (2), mid stage ready (3),
all stages consumed (4), after the CTA barrier (5), stores issued (6); slot 0 is %globaltimer at entry (launch skew).
Prints medians / max over CTAs in us (clock64 at the measured SM clock).
"""
import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="qkv_gate,router,o,pooler")
    ap.add_argument("--ms", default="1,32")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--ghz", type=float, default=1.98)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("submission", Path(__file__).with_name("submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DBG & 16
    weights = reference.generate_weights(1234)
    state = mod.prepare(weights)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    sink = torch.zeros((), dtype=torch.float32, device="cuda")
    for layer in args.layers.split(","):
        for m in [int(v) for v in args.ms.split(",")]:
            data = reference.generate_input(m, layer, 1234 + m, dist="normal")
            n_cta = 128
            runs = []
            for _ in range(args.reps):
                torch.sum(flush.view(torch.float32), 0, out=sink)
                torch.cuda.synchronize()
                mod.kernel_fn(data, state)
                torch.cuda.synchronize()
                st = state["ext"].latency_stamps().view(1024, 2, 8)[:n_cta]
                runs.append(st)
            cols = {}
            for st in runs:
                gt0 = st[:, 0, 0]
                skew = (gt0 - gt0.min()).double() / 1e3
                cols.setdefault("skew", []).append((float(skew.median()), float(skew.max())))
                for w in (0, 1):
                    ck0 = st[:, w, 7]
                    for slot in range(1, 7):
                        d = (st[:, w, slot] - ck0).double() / (args.ghz * 1e3)
                        cols.setdefault(f"w{7 * w}.{slot}", []).append((float(d.median()), float(d.max())))
            parts = []
            for k, v in cols.items():
                med = statistics.median(a for a, _ in v)
                mx = statistics.median(b for _, b in v)
                parts.append(f"{k}={med:.2f}/{mx:.2f}")
            print(f"{layer:9s} m={m:<3d} " + " ".join(parts), flush=True)


if __name__ == "__main__":
    main()
