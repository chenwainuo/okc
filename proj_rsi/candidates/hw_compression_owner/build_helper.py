"""Actual corrected Driver/from_blob host-module CPU build/import. No CUDA API invoked."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile

import torch
HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "hw_compression"


def main():
    old = (V1 / "allocator.cpp").read_text()
    new = (HERE / "allocator.cpp").read_text()
    guard = '''    TORCH_CHECK(!tensor.defined() || (tensor.use_count() == 1 && tensor.storage().use_count() == 1),
                "cannot close: external tensor/alias references remain; destroy graph owners and state copies first");
    tensor = at::Tensor();
'''
    replacement = '''    // Release our Tensor first: a preserved THPVariable wrapper can keep TensorImpl count2
    // with no live external alias. Actual Tensor/Storage aliases still retain the deleter's Region.
    tensor = at::Tensor();
'''
    counts_start = new.index("  py::dict counts() const {")
    counts_end = new.index("  void close() {", counts_start)
    normalized = new[:counts_start] + new[counts_end:]
    normalized = normalized.replace(replacement, guard)
    normalized = normalized.replace('''      .def("info", &WeightClone::info).def("counts", &WeightClone::counts)
      .def("close", &WeightClone::close);''', '''      .def("info", &WeightClone::info).def("close", &WeightClone::close);''')
    assert normalized == old, "changes outside close ordering and counts diagnostic"
    manifest = json.loads((V1 / "manifest.json").read_text())
    assert all(hashlib.sha256((V1 / n).read_bytes()).hexdigest() == h for n, h in manifest["sha256"].items())
    name = "okc_owner_fixed_build"
    base = Path(torch.__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="okc-owner-helper-cpu-"))
    target = tmp / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    command = ["g++", "-shared", "-fPIC", "-O2", "-std=c++17", "-DTORCH_EXTENSION_NAME=" + name,
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
               "-I" + str(base / "include"), "-I" + str(base / "include/torch/csrc/api/include"),
               "-I" + sysconfig.get_path("include"), "-I/usr/local/cuda/include", str(HERE / "allocator.cpp"),
               "-L" + str(base / "lib"), "-Wl,-rpath," + str(base / "lib"),
               "-ltorch_python", "-ltorch", "-ltorch_cpu", "-lc10", "-ldl", "-o", str(target)]
    p = subprocess.run(command, text=True, capture_output=True)
    (tmp / "build.log").write_text(p.stdout + p.stderr)
    assert not p.returncode, str(tmp / "build.log") + "\n" + p.stderr
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    abi = dict(module.abi())
    try:
        module.WeightClone(torch.ones(32, dtype=torch.uint8), 0, True)
    except RuntimeError as e:
        assert "only exact contiguous CUDA uint8" in str(e)
    else:
        raise AssertionError("CPU pointer accepted")
    result = {"status": "CPU_BUILD_IMPORT_NORMALIZED_SOURCE_PASS", "CUDA_API_calls": 0,
              "baseline_v1_unchanged": True, "only_changes": ["close drops owner Tensor before Region guard", "counts diagnostic"],
              "abi": abi, "source_sha256": hashlib.sha256(new.encode()).hexdigest(),
              "build_directory": str(tmp), "build_command": command, "torch": torch.__version__}
    (HERE / "helper-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
