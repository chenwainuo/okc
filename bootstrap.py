#!/usr/bin/env python3
"""Fetch official challenge fixtures and pinned SGLang; apply the experiment stack."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"


def main():
    problem = ROOT / "muse_moe_w4a16"
    if not (problem / "baseline.py").exists():
        okc = shutil.which("okc") or str(Path.home() / ".okc/bin/okc")
        if not Path(okc).exists():
            raise SystemExit("Install the official CLI first: curl -fsSL https://openkernelchallenge.dev/install.sh | sh")
        subprocess.run([okc, "init", "muse_moe_w4a16"], cwd=ROOT, check=True)
    hashes = json.loads((ROOT / "fixture-hashes.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((problem / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Official fixture changed: {name}. Review it before updating fixture-hashes.json.")
    checkout = ROOT / "serving/sglang"
    checkout.mkdir(parents=True, exist_ok=True)

    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(checkout), *args], check=check,
                              capture_output=True, text=True)

    if not (checkout / ".git").exists():
        git("init")
        git("remote", "add", "origin", "https://github.com/sgl-project/sglang.git")
        git("fetch", "--depth=1", "origin", BASE)
        git("checkout", "-b", "my-patch", "FETCH_HEAD")
    if git("rev-parse", "HEAD").stdout.strip() != BASE:
        raise SystemExit("SGLang checkout is not at the pinned base; refusing to overwrite it.")
    patch = str(ROOT / "serving/patch.diff")
    if git("apply", "--reverse", "--check", patch, check=False).returncode != 0:
        if git("status", "--porcelain").stdout.strip():
            raise SystemExit("SGLang has other edits; refusing to overwrite them.")
        git("apply", "--check", patch)
        git("apply", patch)

    baseline = (problem / "baseline.py").read_text()
    anchor = 'local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])'
    assert baseline.count(anchor) == 1
    variants = {
        "candidate_local_buckets.py": (True, False, False),
        "candidate_fused_finalize.py": (False, True, False),
        "candidate_combined.py": (True, True, False),
        "candidate_skip_nonlocal.py": (False, False, True),
        "candidate_full_stack.py": (True, True, True),
    }
    for name, (local, fused, skip) in variants.items():
        replacement = 'local_ids, block_size_m, bank["count"])' if local else anchor
        if skip:
            replacement = replacement[:-1] + ", ignore_invalid_expert=True)"
        code = baseline.replace(anchor, replacement)
        if fused:
            code = code.replace("slot_expert_ids=local_ids)",
                                "slot_expert_ids=local_ids, fuse_small_tokens=True)")
        path = problem / name if name == "candidate_local_buckets.py" else ROOT / "benchmarks" / name
        path.write_text("# Generated local benchmark variant; see bootstrap.py.\n" + code)
    print("Pinned sources and experimental variants are ready. No GPU result is implied.")


if __name__ == "__main__":
    main()
