"""Frozen packed-X pipeline: one two-node counter/PC capture, no timing claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "proj_rsi/candidates/operand_reverse_packed_x"
MANIFEST_SHA = "0420abd392bcb8fae64508f8c247bccec8f7fd7873777e6138e81543a48c94f4"
SOURCE_SHA = "2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a"
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "STARTED", "purpose": "COUNTERS_PC_ONLY_NO_PERFORMANCE_CLAIM"}
    graph = None

    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha(HERE / "manifest.json") == MANIFEST_SHA
        manifest = json.loads((HERE / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            assert sha(HERE / name) == expected, name
        mod = load(HERE / "prototype.py", "profile_frozen_packed_x")
        assert hashlib.sha256(mod.source().encode()).hexdigest() == SOURCE_SHA
        ext = mod.extension()
        binary = Path(ext.__file__).resolve()
        assert binary.is_file() and binary.suffix == ".so"
        dump = subprocess.check_output([shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump",
                                        "--dump-sass", str(binary)], text=True, timeout=20)
        artifact = args.json.with_suffix(".jit.sass")
        artifact.write_text(dump)
        result.update(source_sha256=SOURCE_SHA, manifest_sha256=MANIFEST_SHA,
                      JIT_artifact=str(artifact), JIT_sha256=hashlib.sha256(dump.encode()).hexdigest())
        save()
        result["physical_A4_gate"] = load(HERE / "liveness.py", "profile_packed_liveness").check(dump)
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        sink = torch.zeros(4, device="cuda", dtype=torch.int32)

        def info(code):
            return int(ext.rqrun(0, empty, empty, empty, empty, empty, 0, 0, 0, empty, sink, code, empty))

        assert info(1) == 122 and info(2) == 0 and info(4) >= 1
        assert int(ext.packinfo(1)) == 22 and int(ext.packinfo(2)) == 0
        result["resources"] = {"main_regs": 122, "pack_regs": 22, "local": 0, "dynamic_shared": info(3)}
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        w = weights["unpooler"]
        packed, table, esc6 = mod.S._rqpack(w)
        assert torch.equal(mod.S._rqunpack(packed, table, esc6, *w.shape), w.view(torch.int16).int() & 65535)
        esc, _ = mod.S._rqesc(esc6, packed.shape[0], packed.shape[1], 1, "cuda")
        immutable = [(value, value.clone()) for value in (w, packed, esc)]
        data = reference.generate_input(128, "unpooler", 5237)
        x = data["x"].clone()
        xp = torch.empty(262144, device="cuda", dtype=torch.int32)
        out = torch.empty((128, 8192), device="cuda", dtype=torch.bfloat16)
        cnt = torch.zeros(4 * packed.shape[0], device="cuda", dtype=torch.int32)

        def call():
            rc = int(ext.rqrun(0, packed, x, out, esc, cnt, 1, table, 0, empty, sink, 0, xp))
            assert rc == 0, rc

        call(); call(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        assert len({x.data_ptr(), xp.data_ptr(), out.data_ptr()}) == 3
        xp.fill_(0x5A5A5A5A)
        out.fill_(float("nan"))
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        flush_sink = torch.empty((), device="cuda", dtype=torch.float32)
        torch.sum(flush, 0, out=flush_sink)
        torch.cuda.synchronize()
        print("PACKED_PROFILE_BEGIN two_nodes", flush=True)
        torch.cuda.nvtx.range_push("profile")
        try:
            graph.replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        # No extra replay or workspace rewrite before these exact selected-output checks.
        assert torch.equal(x, data["x"])
        ok, message = reference.check_implementation({"x": x, "config": data["config"]}, out, {"unpooler": w})
        assert ok, message
        idx = torch.arange(262144, device="cuda", dtype=torch.int64)
        v, lane, chunk, warp, stage = idx & 3, (idx >> 2) & 31, (idx >> 7) & 3, (idx >> 9) & 7, idx >> 12
        row = 64 * (warp // 4) + 16 * (warp % 4) + (lane >> 2) + 8 * (v & 1)
        k = 64 * stage + 16 * chunk + 8 * ((v >> 1) & 1) + 2 * (lane & 3)
        assert torch.equal(xp, x.view(torch.int32).reshape(-1)[row * 2048 + k // 2])
        assert all(torch.equal(value, before) for value, before in immutable)
        assert not int(cnt.abs().sum())
        result.update(status="COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED",
                      immediate_selected_output_oracle=message, packed_bytes="EXACT_ALL_1MiB",
                      output_poison_overwritten=True, input_weights_pack_esc_immutable=True,
                      counters="ZERO", profile_nodes_expected=["rx_pack", "rq_px"])
        print("PACKED_PROFILE_END two_nodes", flush=True)
        save()
    except Exception as error:
        result.update(status="INVALID_COUNTER_DIAGNOSTIC", error=repr(error))
        save()
        raise
    finally:
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
            result["graph_cleanup"] = "RESET_AFTER_DRAIN"
            save()
    print("PACKED_COUNTER_DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()
