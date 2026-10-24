"""Local benchmark adapter for the original pinned serving class.

Reads the immutable git object, independently of the patched worktree. Its
finalizer uses the original default path; the new opt-in flag defaults to False.
This harness is not submitted to either challenge track.
"""
from pathlib import Path
import subprocess
from types import ModuleType

from serving_moe import prepare_layer, kernel_fn

BASE = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
FILE = "python/sglang/srt/layers/muse_spark_fp4_moe.py"


def prepare(weights):
    checkout = Path(__file__).resolve().parents[1] / "serving/sglang"
    source = subprocess.check_output(["git", "-C", str(checkout), "show", f"{BASE}:{FILE}"])
    module = ModuleType("stock_muse_spark_fp4_moe")
    exec(compile(source, f"{BASE}:{FILE}", "exec"), module.__dict__)
    return prepare_layer(weights, module.MuseSpark1xFp4Experts)
