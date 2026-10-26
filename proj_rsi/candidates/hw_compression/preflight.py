"""CPU-only full host extension build/import + identity/ABI/source gates. No CUDA calls."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile

import torch
import adapter as A


def main():
    assert hashlib.sha256(A.BASELINE.read_bytes()).hexdigest() == A.FROZEN
    for p in A.HERE.glob("*.py"):
        ast.parse(p.read_text(), filename=str(p))
    source = A.HERE / "allocator.cpp"
    text = source.read_text()
    assert "__global__" not in text and "cudaMalloc" not in text
    assert "cuMemGetAllocationPropertiesFromHandle" in text and "at::from_blob" in text
    assert "cuStreamIsCapturing" in text and "cuStreamSynchronize" in text
    assert "src.is_cuda()" in text and "at::kByte" in text
    name = "okc_hwcomp_preflight"
    tmp = Path(tempfile.mkdtemp(prefix="okc-hwcomp-cpu-"))
    out = tmp / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    base = Path(torch.__file__).resolve().parent
    command = ["g++", "-shared", "-fPIC", "-O2", "-std=c++17",
               "-DTORCH_EXTENSION_NAME=" + name,
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
               "-I" + str(base / "include"), "-I" + str(base / "include/torch/csrc/api/include"),
               "-I" + sysconfig.get_path("include"), "-I/usr/local/cuda/include",
               str(source), "-L" + str(base / "lib"), "-Wl,-rpath," + str(base / "lib"),
               "-ltorch_python", "-ltorch", "-ltorch_cpu", "-lc10", "-ldl", "-o", str(out)]
    built = subprocess.run(command, text=True, capture_output=True)
    (tmp / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        print(built.stderr)
        raise RuntimeError("CPU host build failed; see " + str(tmp / "build.log"))
    spec = importlib.util.spec_from_file_location(name, out)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    abi = dict(module.abi())  # Header constants only; does not initialize/resolve Driver.
    assert abi["CUDA_VERSION"] == 12080 and abi["COMP_GENERIC"] == 1 and abi["COMP_NONE"] == 0
    try:
        module.WeightClone(torch.ones(32, dtype=torch.uint8), 0, True)
    except RuntimeError as error:
        assert "only exact contiguous CUDA uint8" in str(error)
    else:
        raise AssertionError("CPU tensor must be rejected before Driver initialization")
    result = {"state": "CPU_FULL_HOST_BUILD_IMPORT_PASS_GPU_UNRUN", "torch_cpu": torch.__version__,
              "abi": abi, "baseline_sha256": A.FROZEN, "build_command": command,
              "build_directory": str(tmp), "no_CUDA_API_invoked": True,
              "CPU_pointer_rejected_before_Driver": True,
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(A.HERE.iterdir()) if p.suffix in (".cpp", ".py", ".md")}}
    (A.HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    files = ["allocator.cpp", "adapter.py", "lab.py", "preflight.py", "preflight.json", "README.md"]
    manifest = {"state": "CPU_PASS_GPU_UNRUN_FROZEN_FOR_REVIEW", "baseline_sha256": A.FROZEN,
                "sha256": {name: hashlib.sha256((A.HERE / name).read_bytes()).hexdigest() for name in files}}
    (A.HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
