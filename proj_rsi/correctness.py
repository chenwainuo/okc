"""CPU correctness for a submission, using the problem's own reference.

CUDA-only submissions return status=not_runnable. That is not a pass and not a fail
of the numerics; the judge is the authority for kernels that do not run here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / "muse_bf16_proj"

OFFICIAL_CASES = (
    {"m": 1, "layer": "qkv_gate", "dist": "normal"},
    {"m": 7, "layer": "pooler", "dist": "normal"},
    {"m": 33, "layer": "o", "dist": "normal"},
    {"m": 130, "layer": "unpooler", "dist": "normal"},
    {"m": 5, "layer": "unpooler", "dist": "normal"},
    {"m": 8, "layer": "router", "dist": "normal"},
    {"m": 9, "layer": "router", "dist": "outliers"},
    {"m": 512, "layer": "qkv_gate", "dist": "outliers"},
    {"m": 1000, "layer": "unpooler", "dist": "outliers"},
)

SMOKE_CASES = (
    {"m": 1, "layer": "qkv_gate", "dist": "normal"},
    {"m": 5, "layer": "unpooler", "dist": "normal"},
    {"m": 8, "layer": "router", "dist": "normal"},
    {"m": 9, "layer": "router", "dist": "outliers"},
)


def _load_reference():
    if str(PROBLEM) not in sys.path:
        sys.path.insert(0, str(PROBLEM))
    import reference  # type: ignore

    return reference


def _load_submission(path: Path):
    spec = importlib.util.spec_from_file_location(f"proj_rsi_candidate_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_submission(path: Path, cases=SMOKE_CASES, device: str = "cpu", seed: int = 5151) -> dict:
    path = Path(path)
    try:
        reference = _load_reference()
        submission = _load_submission(path)
        weights = reference.generate_weights(seed, device=device)
        state = submission.prepare(weights)
    except Exception as error:  # import, compile, or missing CUDA
        return {
            "path": str(path),
            "status": "not_runnable",
            "passed": None,
            "error": f"{type(error).__name__}: {error}",
            "cases": [],
        }

    results = []
    for index, case in enumerate(cases):
        data = reference.generate_input(
            case["m"], case["layer"], seed + 1000 + index, dist=case["dist"], device=device
        )
        try:
            output = submission.kernel_fn(data, state)
            ok, message = reference.check_implementation(data, output, weights)
        except Exception as error:
            ok, message = False, f"{type(error).__name__}: {error}"
        results.append({**case, "ok": ok, "message": message})
    passed = all(item["ok"] for item in results)
    return {
        "path": str(path),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "cases": results,
    }
