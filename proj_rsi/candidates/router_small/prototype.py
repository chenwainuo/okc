"""One m8 full-K tensor-folding prototype; no full challenge submission/automatic bake."""
import hashlib
import importlib.util
import os
from pathlib import Path

import torch

from packing import pack

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "pw36" / "submission.py"
PARENT_SHA256 = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
_module = None


def load_parent():
    actual = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise RuntimeError(f"Frozen pw36 changed: {actual}")
    spec = importlib.util.spec_from_file_location("router_small_frozen_pw36", PARENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extension():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load_inline
        src = (HERE / "kernel.cu").read_text()
        before = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            _module = load_inline(
                "router_small_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources="int64_t router_small_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); int64_t router_small_regs(); int64_t router_small_local(); int64_t router_small_occ();",
                cuda_sources=src,
                functions=["router_small_run", "router_small_regs", "router_small_local", "router_small_occ"],
                extra_cuda_cflags=["-O3", "-lineinfo", "-DROUTER_SMALL_TORCH", "-gencode=arch=compute_90a,code=sm_90a"],
                verbose=False)
        finally:
            if before is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = before
    return _module


def prepare(w):
    ext = extension()  # Actual Torch host/device compilation is a mandatory trial gate.
    resources = {"registers": int(ext.router_small_regs()), "local_bytes": int(ext.router_small_local()),
                 "CTAs_per_SM": int(ext.router_small_occ()), "threads": 1024, "shared_bytes": 2048}
    assert 0 < resources["registers"] <= 64 and resources["local_bytes"] == 0
    assert resources["CTAs_per_SM"] >= 1
    data, headers, stats = pack(w)
    return {"data": data.to(w.device), "headers": headers.to(w.device), "packing": stats,
            "resources": resources, "device": w.device}


def into(state, x, out):
    rc = extension().router_small_run(state["data"], state["headers"], x, out)
    if rc != 0:
        raise RuntimeError(f"prototype launch declined rc={rc}")
    return out


def call(state, x):
    out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
    return into(state, x, out)
