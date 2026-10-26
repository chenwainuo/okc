"""Build-only import; no Driver initialization until a clone/query is requested."""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = HERE.parent / "pw36/submission.py"
FROZEN = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"


def extension():
    from torch.utils.cpp_extension import load
    source = HERE / "allocator.cpp"
    name = "okc_hwcomp_" + hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name, sources=[str(source)], extra_include_paths=["/usr/local/cuda/include"],
                extra_cflags=["-O2", "-std=c++17"], extra_ldflags=["-ldl"],
                with_cuda=False, verbose=True)


def baseline_module():
    import importlib.util
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == FROZEN
    spec = importlib.util.spec_from_file_location("hwcomp_frozen_pw36", BASELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
