"""No GPU launch: prove v2 is host-only, rerun CPU mapping, compile both, compare SASS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
OLD = HERE / "port_probe.cu"
NEW = HERE / "port_probe_v2.cu"
FROZEN = "cc46c06739966806e4bd86ca078824e6d768068e8bac9426937b0f2b5fbf1774"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def without_host_info(src):
    # Only these host-only differences are authorized. Everything else must be byte-identical.
    src = src.replace("port_probe_v2.cu", "port_probe.cu").replace("/tmp/rs256full_port_probe_v2", "/tmp/rs256full_port_probe")
    return re.sub(r"template<int MODE> static void info\(\) \{.*?\n\}\n(?=// Same 2D)", "<HOST_INFO>\n", src, flags=re.S)


def compile_one(source, binary, nvcc, cuobjdump):
    cmd = [nvcc, "-arch=sm_90a", "-O3", "-std=c++17", "-Xptxas=-v", str(source), "-o", str(binary)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log
    assert not re.search(r"C75\d\d|\bwarning\b|\berror\b", log, re.I), log
    assert log.count("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads") == 3, log
    sass = subprocess.check_output([cuobjdump, "--dump-sass", str(binary)], text=True)
    functions = {}
    stats = {}
    for section in re.split(r"Function : ", sass)[1:]:
        name = section.splitlines()[0].strip()
        mode = int(re.search(r"probeILi(\d)E", name).group(1))
        # Drop file/ELF headers, instruction addresses, and machine encoding comments.
        instructions = re.findall(r"/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;\s*/\*", section)
        assert instructions, name
        functions[mode] = "\n".join(instructions) + "\n"
        stats[mode] = {"name": name, "instructions": len(instructions),
                       "normalized_instruction_sha256": sha(functions[mode].encode()),
                       "HGMMA": section.count("HGMMA."), "LDS128": section.count("LDS.128"),
                       "UTMALDG": section.count("UTMALDG"), "UBLKCP": section.count("UBLKCP"),
                       "WARPGROUP_ARRIVE": section.count("WARPGROUP.ARRIVE")}
        assert not re.search(r"\b(?:LDL|STL)\b", section), name
        if mode in (1, 3):
            assert stats[mode]["HGMMA"] == 32 and stats[mode]["WARPGROUP_ARRIVE"] == 2, stats
            assert stats[mode]["LDS128"] >= 24, stats
        if mode in (2, 3):
            assert stats[mode]["UTMALDG"] >= 4 and stats[mode]["UBLKCP"] >= 2, stats
    assert set(functions) == {1, 2, 3}
    return functions, {"source_sha256": sha(source.read_bytes()), "ptxas": log, "sass": stats,
                       "compile_command": cmd[:-1] + ["<temporary-binary>"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    assert sha(OLD.read_bytes()) == FROZEN, "port01 source changed"
    assert without_host_info(OLD.read_text()) == without_host_info(NEW.read_text()), "unauthorized source delta"
    spec = importlib.util.spec_from_file_location("rs256full_preflight", HERE / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SRC = NEW
    cpu = module.proof()
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuobjdump = str(Path(nvcc).with_name("cuobjdump"))
    with tempfile.TemporaryDirectory(prefix="rs256full-v2-") as tmp:
        old_sass, old_build = compile_one(OLD, Path(tmp) / "port01", nvcc, cuobjdump)
        new_sass, new_build = compile_one(NEW, Path(tmp) / "v2", nvcc, cuobjdump)
        assert old_sass == new_sass, "GPU instructions changed"
    result = {
        "status": "HOST_SETUP_FIX_ONLY_CPU_AND_SASS_PASS_GPU_UNRUN",
        "port01_status": "INVALID_OCCUPANCY_QUERY_BEFORE_DYNAMIC_SHARED_OPTIN_NO_PROBE_KERNELS",
        "source_delta": "host info() sets/prints dynamic shared opt-in before occupancy query; build-comment names updated",
        "normalized_device_source_identical": True,
        "normalized_SASS_all_three_kernels_identical": True,
        "runtime_attribute_values": "UNMEASURED_NO_GPU_LAUNCH",
        "nvcc": subprocess.check_output([nvcc, "--version"], text=True).splitlines()[-1],
        "CPU_proof": cpu, "port01_build": old_build, "v2_build": new_build,
        "files": {p.name: sha(p.read_bytes()) for p in [OLD, NEW, HERE / "preflight.py", Path(__file__), HERE / "port01.json"]},
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
