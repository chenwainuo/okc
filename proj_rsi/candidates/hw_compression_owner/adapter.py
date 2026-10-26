"""Corrected close ownership only; allocation and math policy unchanged from frozen v1."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def extension():
    from torch.utils.cpp_extension import load
    path = HERE / "allocator.cpp"
    name = "okc_hwcomp_owner_" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return load(name, sources=[str(path)], extra_include_paths=["/usr/local/cuda/include"],
                extra_cflags=["-O2", "-std=c++17"], extra_ldflags=["-ldl"],
                with_cuda=False, verbose=True)
