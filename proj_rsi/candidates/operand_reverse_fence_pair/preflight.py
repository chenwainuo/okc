"""CPU/source/device-only gate. A failing compiler schedule is retained, never run.

No extension loading, CUDA context, remote access or GPU launch is performed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import prototype as P
from schedule_proof import proof

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "operand_reverse_dual_producer"
EXPECTED_PARENT = "97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b"


def static_proof():
    body = (HERE / "kernel_body.cuh").read_text()
    parent = (PARENT / "kernel_body.cuh").read_text()
    # Only the startup and math region, scratch allocation and kernel name differ.
    lo = "  float d[32]={};"
    hi = '  asm volatile("bar.sync 1,256;\\n":::"memory");'
    restored = body[:body.index(lo)] + parent[parent.index(lo):parent.index(hi)] + body[body.index(hi):]
    restored = restored.replace("rq_px_fp(", "rq_px_dp(")
    restored = restored.replace("+16*BST+512;", "+16*BST;")
    restored = restored.replace("prefetch(0,q0);prefetch(1,q1);}",
                                "prefetch(0,q0);prefetch(1,q1);prefetch(2,q2);prefetch(3,q3);}")
    assert restored == parent
    source = P.source()
    original = source.replace(body, parent).replace("rq_px_fp<", "rq_px_dp<")
    assert hashlib.sha256(original.encode()).hexdigest() == EXPECTED_PARENT
    # base may consume up to1023B alignment slack. All barrier arrays precede
    # scratch; eight math warps own disjoint64B ranges. Producer/output never use it.
    barriers_end = 6 * 6144 + 8 * 8192 + 8 * 6 + 16 * 8
    assert barriers_end == 102576 and barriers_end % 16 == 0
    assert 1023 + barriers_end + 8 * 64 == 104111 < 104112
    spans = [(barriers_end + 64 * w, barriers_end + 64 * (w + 1)) for w in range(8)]
    assert all(spans[w][1] == spans[w + 1][0] for w in range(7))
    spec = importlib.util.spec_from_file_location("fence_pair_retained_bit_models", HERE.parent / "operand_reverse_packed_x/preflight.py")
    previous = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE.parent / "operand_reverse_packed_x"))
    try:
        spec.loader.exec_module(previous)
    finally:
        sys.path.pop(0)
    ls = importlib.util.spec_from_file_location("fence_pair_parent_physical_checker", PARENT / "liveness.py")
    checker = importlib.util.module_from_spec(ls)
    ls.loader.exec_module(checker)
    parent_physical = checker.check((PARENT / "sass.txt").read_text())
    rejected = False
    try:
        checker.check((HERE.parent / "operand_reverse_packed_x/failed-q4a2/sass.txt").read_text())
    except AssertionError as error:
        rejected = True
        old_bad_error = str(error)
    assert rejected, "Known premature A overwrite must remain rejected"
    return {"restored_exact_DP_source": EXPECTED_PARENT,
            "schedule": proof(), "W_bits": previous.wproof(),
            "output_map": previous.outputproof(), "X_pack_map": previous.xproof(),
            "retained_parent_physical_gate": parent_physical["status"],
            "retained_bad_lifetime_negative": {"rejected": rejected, "error": old_bad_error},
            "scratch": {"bytes": 512, "dynamic_smem": 104112, "relative_spans": spans,
                        "maximum_address_including_alignment": 104111,
                        "logical_extra_shared_writes_bytes": 128 * 8 * 64 * 64,
                        "vector_store_instructions_per_warp_stage": 4},
            "limits": "Retained W proof checks exact bits/mapping, not runtime CSR reload interleavings. This is source-level proof; actual codegen gates remain mandatory."}


def main():
    result = {"state": "CPU_STARTED_GPU_UNRUN", "static": static_proof()}
    source = P.source()
    (HERE / "extension_source.cu").write_text(source)
    (HERE / "device_source.cu").write_text(P.device_source())
    cmd = ["/usr/local/cuda-12.8/bin/nvcc", *P.S.RQFLAGS, "--cubin", "--ptxas-options=-v",
           str(HERE / "device_source.cu"), "-o", str(HERE / "device.sm90a.cubin")]
    run = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
    text = run.stdout + run.stderr
    (HERE / "resource.log").write_text(text)
    assert run.returncode == 0, text
    sass = subprocess.check_output(["/usr/local/cuda-12.8/bin/cuobjdump", "--dump-sass", str(HERE / "device.sm90a.cubin")], text=True, timeout=20)
    (HERE / "sass.txt").write_text(sass)
    main_sass = sass.split("Function : ")[1].split("Function : ")[0]
    result.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(), compile_command=cmd,
                  main_registers=int(re.search(r"Used (\d+) registers", text)[1]),
                  static_HGMMA=main_sass.count("HGMMA."), static_ARRIVE=main_sass.count("WARPGROUP.ARRIVE"),
                  static_volatile_pin_STS128=main_sass.count("STS.128"),
                  stack_spill_records=re.findall(r"\d+ bytes (?:stack frame|spill stores|spill loads)", text),
                  diagnostic_codes=sorted(set(re.findall(r"C75\d\d", text))), gpu_calls=0)
    bad = bool(result["diagnostic_codes"] or re.search(r"\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)", text))
    bad |= result["main_registers"] > 128
    # Three statically materialized4group regions: startup, steadyloop, tail.
    bad |= result["static_HGMMA"] != 48 or result["static_ARRIVE"] != 6
    result["state"] = "CPU_CODEGEN_FAILED_RETAINED_NO_GPU" if bad else "CODEGEN_REQUIRES_PHYSICAL_LIFETIME_REVIEW"
    (HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "static"}, indent=2))
    if bad:
        raise SystemExit(1)
    # Even a future compile pass is not GPU authorization. A generalized actual
    # JIT per-register lifetime/fence checker + full lab would still be needed.


if __name__ == "__main__":
    main()
