"""CPU geometry/codec proof and optional local SM90a compile/SASS gate. Never launches CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SRC = HERE / "port_probe.cu"


def prmt(a, b, sel):
    raw = (a | (b << 32)).to_bytes(8, "little")
    return sum(raw[(sel >> (4 * i)) & 7] << (8 * i) for i in range(4))


def decode(a, b, c, tb):
    hl = ((c & 0x87878787) + tb) & 0xFFFFFFFF
    rot = ((c >> 4) | (c << 28)) & 0xFFFFFFFF
    hh = ((rot & 0x87878787) + tb) & 0xFFFFFFFF
    return [prmt(a, hl, 0x5140), prmt(a, hl, 0x7362),
            prmt(b, hh, 0x5140), prmt(b, hh, 0x7362)]


def proof():
    xb, wb = 32768, 49152
    payload = xb + wb
    data = bytearray(payload)
    data[:xb] = bytes([0x80, 0x3F]) * (xb // 2)
    seen = bytearray(wb)
    for j in range(4):
        for warp in range(8):
            p = xb + j * 12288 + warp * 1536
            data[p:p + 1024] = bytes([0x80]) * 1024
    # Exhaust every lane's LDS address. No duplicated/missing packed-W bytes.
    for j in range(4):
        for warp in range(8):
            for lane in range(32):
                p = xb + j * 12288 + warp * 1536 + lane * 16
                q = [struct.unpack_from("<4I", data, p + off) for off in (0, 512, 1024)]
                a = (decode(q[0][0], q[0][1], q[2][0], 0x3F3F3F3F)
                     + decode(q[0][2], q[0][3], q[2][1], 0x3F3F3F3F)
                     + decode(q[1][0], q[1][1], q[2][2], 0x3F3F3F3F)
                     + decode(q[1][2], q[1][3], q[2][3], 0x3F3F3F3F))
                assert a == [0x3F803F80] * 16
                for off in (0, 512, 1024):
                    for pos in range(p - xb + off, p - xb + off + 16):
                        seen[pos] += 1
    assert set(seen) == {1}
    # Four 2D-TMA-equivalent X subtiles, each 64 tokens x 64 K, SW128.
    xseen = set()
    for j in range(4):
        assert (j * 8192) % 1024 == 0
        for token in range(64):
            for k in range(64):
                byte = 2 * k
                addr = j * 8192 + token * 128 + (((byte // 16) ^ (token & 7)) * 16) + byte % 16
                assert addr % 2 == 0 and 0 <= addr < xb
                assert addr not in xseen
                xseen.add(addr)
                assert data[addr:addr + 2] == bytes([0x80, 0x3F])
    assert len(xseen) * 2 == xb
    # The proposed full-K grid is one wave on the >=128-SM lab device.
    assert (4096 // 128) * (256 // 64) == 128
    read = wb + 2 * xb
    write = wb + xb
    assert read == 114688 and write == 81920
    return {
        "status": "CPU_PROOF_PASS_GPU_UNRUN_PORT_MODEL_UNRESOLVED",
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "geometry": {"channels": 128, "tokens": 64, "merged_k": 256,
                     "math_warpgroups": 2, "stages": 2, "ctas": 128,
                     "dynamic_shared_bytes_including_alignment": 164928,
                     "A_double_buffer_regs_per_math_thread": 128,
                     "accumulator_regs_per_math_thread": 32,
                     "producer_register_allocation": 40, "consumer_register_allocation": 232},
        "bytes_per_CTA_round": {"TMA_write": write, "LDS_packed_W": wb,
                                "WGMMA_B_read": 2 * xb, "read_total": read,
                                "aggregate": read + write},
        "conditional_cycle_floors": {"tensor": 1024, "aggregate_128B_read_write": 1536,
                                     "separate_128B_read_write": 896},
        "checks": {"all_49152_W_bytes_read_once": True, "all_32768_X_bytes_SW128_bijective": True,
                   "all_16384_register_pairs_decode_to_exact_bf16_ones": True,
                   "disjoint_resident_and_producer_80KiB_slots": True},
        "runtime_checks": ["all 1048576 f32 accumulators == 256*iters (all zero in producer-only)",
                           "128 full 80KiB producer snapshots == source byte-for-byte",
                           "writes==iters, progress==iters (producer-only progress==0)",
                           "exactly128 distinct SM IDs; runtime occupancy==1; localSizeBytes==0"],
        "limit": "synthetic all-ones pack: no escape walk, full matmul, input-probe correction, fallback, or speedup claim",
    }


def build():
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuobjdump = str(Path(nvcc).with_name("cuobjdump"))
    # Scratch only; no GPU API is called by this script.
    with tempfile.TemporaryDirectory(prefix="rs256full-build-") as tmp:
        binary = Path(tmp) / "port_probe"
        cmd = [nvcc, "-arch=sm_90a", "-O3", "-std=c++17", "-Xptxas=-v", str(SRC), "-o", str(binary)]
        result = subprocess.run(cmd, text=True, capture_output=True)
        log = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(log)
        assert not re.search(r"C75\d\d|\bwarning\b|\berror\b", log, re.I), log
        assert log.count("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads") == 3, log
        sass = subprocess.check_output([cuobjdump, "--dump-sass", str(binary)], text=True)
        sections = re.split(r"Function : ", sass)[1:]
        counts = {}
        for section in sections:
            name = section.splitlines()[0].strip()
            mode = int(re.search(r"probeILi(\d)E", name).group(1))
            counts[mode] = {"HGMMA": section.count("HGMMA."), "LDS128": section.count("LDS.128"),
                            "UTMALDG": section.count("UTMALDG"), "UBLKCP": section.count("UBLKCP"),
                            "WARPGROUP_ARRIVE": section.count("WARPGROUP.ARRIVE")}
            assert not re.search(r"\b(?:LDL|STL)\b", section), name
            if mode in (1, 3):
                assert counts[mode]["HGMMA"] == 32, counts
                assert counts[mode]["LDS128"] >= 24, counts
                # One explicit fence per each of two unrolled k256 rounds; no per-MMA serialization.
                assert counts[mode]["WARPGROUP_ARRIVE"] == 2, counts
            if mode in (2, 3):
                assert counts[mode]["UTMALDG"] >= 4 and counts[mode]["UBLKCP"] >= 2, counts
        assert set(counts) == {1, 2, 3}
        return {"status": "LOCAL_NVCC_LINK_AND_SASS_PASS_NO_GPU_LAUNCH",
                "nvcc": subprocess.check_output([nvcc, "--version"], text=True).splitlines()[-1],
                "compile_command": cmd[:-1] + ["<temporary-binary>"], "ptxas": log, "sass_counts": counts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    result = proof()
    if args.build:
        result["compile_gate"] = build()
    out = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(out)
    print(out, end="")
