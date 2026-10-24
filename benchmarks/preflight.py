#!/usr/bin/env python3
"""Check the single-GPU experiment environment before allocating expert weights."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"


def main():
    report = {"python": sys.version, "errors": [], "warnings": []}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            report["errors"].append("No CUDA device visible; run this in the GPU container on the H200 host.")
        else:
            prop = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            report["gpu"] = dict(name=prop.name, capability=[prop.major, prop.minor],
                                 free_bytes=free, total_bytes=total)
            if "H200" not in prop.name:
                report["warnings"].append("Hardware differs from the judge's H200; timing is not directly comparable.")
            # Fail on missing runtime imports before expensive weight generation.
            for name in ("triton", "sgl_kernel", "flashinfer", "sglang",
                         "sglang.srt.layers.muse_spark_fp4_moe"):
                module = importlib.import_module(name)
                report[name] = getattr(module, "__file__", None)
            source = Path(report["sglang.srt.layers.muse_spark_fp4_moe"]).resolve()
            if not source.is_relative_to(ROOT / "serving/sglang/python"):
                report["errors"].append("SGLang is not loaded from the patched checkout; check PYTHONPATH.")
        report["checkout_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT / "serving/sglang"), "rev-parse", "HEAD"], text=True).strip()
        if report["checkout_commit"] != BASE:
            report["errors"].append("Checkout is not based on the challenge's required commit.")
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
