#!/usr/bin/env python3
"""Isolate graph-safe Marlin tile policies; generated code is saved with results.

Keep the original alignment path and enable small-batch fusion for all candidates.
Compare fixed supported block sizes and a global-expert-density heuristic against
the official stock kernel. This is a local diagnostic, not a serving score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", default=["global-density", "8", "16", "32", "48", "64"],
                        choices=["global-density", "8", "16", "32", "48", "64"])
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--tag", default="", help="Suffix for a separate experiment's results")
    args = parser.parse_args()
    source = (ROOT / "muse_moe_w4a16/baseline.py").read_text()
    expected = json.loads((ROOT / "fixture-hashes.json").read_text())["baseline.py"]
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise SystemExit("Baseline changed; inspect before generating candidates")
    source = source.replace("slot_expert_ids=local_ids)",
                            "slot_expert_ids=local_ids, fuse_small_tokens=True)")
    directory = ROOT / "results/tile-candidates"
    directory.mkdir(parents=True, exist_ok=True)
    for policy in args.policies:
        if policy == "global-density":
            code = source.replace('num_tokens * TOP_K / bank["count"] / block_size_m',
                                  'num_tokens * TOP_K / N_EXPERTS / block_size_m')
        else:
            code = source.replace("BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)",
                                  f"BLOCK_SIZE_M_CANDIDATES = ({policy},)")
        assert code != source
        candidate = directory / f"block-{policy}.py"
        candidate.write_text("# Generated local tile experiment; see sweep_tiles.py.\n" + code)
        subprocess.run([
            sys.executable, str(ROOT / "benchmarks/moe_local.py"),
            "--candidate", str(candidate), "--pairs", str(args.pairs),
            "--output", str(ROOT / "results" / f"tiles-{policy}{('-' + args.tag) if args.tag else ''}.json"),
            "--batches", *map(str, args.batches),
        ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
