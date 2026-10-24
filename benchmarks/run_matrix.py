#!/usr/bin/env python3
"""Run independent changes, their combination, and the actual serving method.

Inside the challenge container with the checkout on PYTHONPATH:
  python3 benchmarks/run_matrix.py
  python3 benchmarks/run_matrix.py --prefill

Prefill runs use two pairs by default and are regression diagnostics. The geometric
mean of local kernel speedups is not an estimate of the official serving score.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    variants = {
        "local-buckets": "muse_moe_w4a16/candidate_local_buckets.py",
        "fused-finalize": "benchmarks/candidate_fused_finalize.py",
        "combined": "benchmarks/candidate_combined.py",
        "skip-nonlocal": "benchmarks/candidate_skip_nonlocal.py",
        "full-stack": "benchmarks/candidate_full_stack.py",
        "serving-method": "benchmarks/serving_moe.py",
    }
    pairs = args.pairs if args.pairs is not None else (2 if args.prefill else 20)
    batches = [753, 3012, 16384] if args.prefill else [1, 8, 16, 32, 128, 256]
    for name, candidate in variants.items():
        suffix = "prefill" if args.prefill else "decode-verify"
        extra = ["--profile"] if args.profile else []
        if name == "serving-method":
            extra += ["--baseline", str(ROOT / "benchmarks/serving_stock_moe.py")]
        subprocess.run([
            sys.executable, str(ROOT / "benchmarks/moe_local.py"),
            "--candidate", str(ROOT / candidate), "--pairs", str(pairs),
            "--output", str(ROOT / "results" / f"{name}-{suffix}.json"),
            *extra,
            "--batches", *map(str, batches),
        ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
