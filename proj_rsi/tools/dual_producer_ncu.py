"""Frozen DP: one two-node cold counter/PC capture, no timing claim.

Root only, after independent review; inner180/outer240, existing profiling capability:
  ncu --replay-mode kernel --graph-profiling node --target-processes all \
    --cache-control all --clock-control none --nvtx --nvtx-include profile/ \
    --launch-count 2 --kill no --section LaunchStats --section SourceCounters \
    --section MemoryWorkloadAnalysis_Tables --section SchedulerStats \
    --section WarpStateStats --section SpeedOfLight \
    --export /okc/logs/dual-producer-ncu01 -f \
    python3 /okc/proj_rsi/tools/dual_producer_ncu.py \
    --json /okc/logs/dual-producer-ncu01-driver.json

SourceCounters already contains Instructions Executed and Predicated-On Thread
Instructions Executed. Static PC map below is derived from actual JIT before
math launches. The raw counter dataset is analyzed separately after root capture.
"""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "proj_rsi/candidates/operand_reverse_dual_producer"
MANIFEST_SHA = "281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241"
SOURCE_SHA = "97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b"
BASE_DRIVER_SHA = "a3e81cc0adece02ffbf7f28250bb2b798642b94ccdfc63762ea90054b9b74d4d"
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def counter_questions():
    return {
        "variant": "DP: math warps0..7; producer0=8..11 even stages; producer1=12..15 odd",
        "capture": "ONE fresh packed-X+DP graph, two selected nodes; no parent prepare or timing",
        "sections_requested_externally": ["LaunchStats", "SourceCounters", "MemoryWorkloadAnalysis_Tables",
                                           "SchedulerStats", "WarpStateStats", "SpeedOfLight"],
        "questions": [
            "Do DP math Bfull TRYWAIT PCs still dominate samples after the measured producer-parallelism gain?",
            "At exact failed-test entry/retry branch PCs, count Predicated-On Thread Instructions Executed; compare with Thread Instructions Executed for failed-lane fraction.",
            "Separate actual failed tests/outlined-loop entry from TRYWAIT-site samples; successful first TRYWAIT may itself block until publication.",
            "Compare current math X LDG, WGMMA fence/commit/wait PCs with Bfull and current producer Wfull/Bempty/bar3/bar4/CSR/store PCs.",
            "Producer common PCs combine8warps and both owners; only source-proved owner-specific bar3/bar4 predicates distinguish those owners here.",
        ],
        "preregistered_checks": [
            "Bind report shader/source/launch to captured actual-JIT map; never reuse PX/old-DP hardcoded PC addresses.",
            "For failed branches use Predicated-On Thread Instructions Executed, not Instructions Executed (which ignores predication).",
            "Report failed-lane fraction as predicated-on thread count / total thread count at that branch when denominator exists.",
            "Division by32 may be labeled warp-equivalent normalization only; per-thread try_wait without .sync does not formally prove per-instance uniform predicates.",
            "Source/model predicts each of4math fast-test sites executes16iterations×8mathwarps×128CTAs=16384warp instructions; use as coverage sanity check, not branch-taken count.",
            "SourceCounters sampled TRYWAIT stalls alone do not prove producer starvation, initial barrier readiness, causal time or intrinsic instruction latency.",
            "try_wait is potentially blocking until completion/system limit and has no explicit timeHint here: even no failed lanes cannot establish initially-ready Bfull.",
            "Keep pack and DP main counters separate; node replay/cache flushing is profiling instrumentation, not whole-pipeline performance evidence.",
        ],
        "limits": "One profiled graph gives counters/PC samples, not new F1/paired gain or a confidence interval. More producer warps alter denominators; samples are not microseconds.",
    }


def pc_map(sass, physical_gate):
    """Map actual main fast/outlined Bfull predicates/targets, plus distinct role sites."""
    tables = [t for t in re.split(r"\n\s*Function : ", sass)[1:] if "rq_px_dp" in t.splitlines()[0]]
    assert len(tables) == len(physical_gate["binaries"]) > 0
    mapped = []
    for table, gate in zip(tables, physical_gate["binaries"]):
        rows = [(int(m[1], 16), m[2].strip()) for m in re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);", table)]
        lo, hi = gate["loop_PC_range"]
        slow_targets = gate["outlined_wait_paths_no_A_writes"]
        assert len(slow_targets) == 4
        waits = []
        for target in slow_targets:
            entries = [(pc, ins, re.fullmatch(r"(@!?P\d+)\s+BRA 0x([0-9a-f]+)", ins))
                       for pc, ins in rows if lo <= pc <= hi]
            entries = [(pc, ins, m) for pc, ins, m in entries if m and int(m[2], 16) == target]
            assert len(entries) == 1
            entry_pc, entry_ins, branch = entries[0]
            assert branch[1].startswith("@!P"), "Bfull entry must branch on failed TRYWAIT predicate"
            index = next(i for i, (pc, ins) in enumerate(rows) if pc == entry_pc)
            fast_tests = [(pc, ins) for pc, ins in rows[max(0, index - 12):index] if "SYNCS.PHASECHK" in ins and "TRYWAIT" in ins]
            assert fast_tests
            fast_pc, fast_ins = fast_tests[-1]
            predicate = re.search(r"TRYWAIT (P\d+),", fast_ins)[1]
            assert branch[1] == "@!" + predicate
            block = []
            for pc, ins in rows:
                if pc < target:
                    continue
                block.append((pc, ins))
                ret = re.fullmatch(r"BRA 0x([0-9a-f]+)", ins)
                if ret and lo <= int(ret[1], 16) <= hi:
                    assert int(ret[1], 16) == entry_pc + 16
                    break
            assert block and re.fullmatch(r"BRA 0x([0-9a-f]+)", block[-1][1])
            retries = [(pc, ins) for pc, ins in block if re.fullmatch(r"@!" + predicate + r" BRA 0x" + format(target, "x"), ins)]
            slow_tests = [(pc, ins) for pc, ins in block if "SYNCS.PHASECHK" in ins and "TRYWAIT" in ins]
            assert len(retries) == len(slow_tests) == 1
            assert re.search(r"TRYWAIT (P\d+),", slow_tests[0][1])[1] == predicate
            waits.append({"fast_try_PC": hex(fast_pc), "fast_try": fast_ins,
                          "failed_entry_branch_PC": hex(entry_pc), "failed_entry_branch": entry_ins,
                          "outlined_target_PC": hex(target), "slow_try_PC": hex(slow_tests[0][0]),
                          "slow_try": slow_tests[0][1], "failed_retry_branch_PC": hex(retries[0][0]),
                          "failed_retry_branch": retries[0][1], "ready_return": block[-1][1]})
        bars = {}
        for barrier, owner in [(3, "producer0_even_warps8..11"), (4, "producer1_odd_warps12..15")]:
            sites = [(pc, ins) for pc, ins in rows if re.search(r"BAR\.SYNC[^ ]* 0x" + str(barrier) + r", 0x80$", ins)]
            assert len(sites) == 1
            bars[owner] = {"PC": hex(sites[0][0]), "instruction": sites[0][1]}
        fence = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and "WARPGROUP.ARRIVE" in ins]
        wait1 = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and "WARPGROUP.DEPBAR.LE" in ins]
        assert len(fence) == len(wait1) == 4
        xloads = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and re.search(r"\bLDG\.[^ ]*\.128(?:\.| )", ins)]
        all_xloads = [(pc, ins) for pc, ins in rows if re.search(r"\bLDG\.[^ ]*\.128(?:\.| )", ins)]
        assert len(xloads) == 16 and len(all_xloads) == 32, "unexpected fixed packed-X vector-load map"
        mapped.append({"function": table.splitlines()[0], "math_warps": [0, 7], "producer_warps": [8, 15],
                       "math_Bfull_waits": waits, "producer_owner_specific_barriers": bars,
                       "math_fence_PCs": [hex(pc) for pc, ins in fence],
                       "math_retirement_wait1_PCs": [hex(pc) for pc, ins in wait1],
                       "math_loop_X_LDG128_PCs": [hex(pc) for pc, ins in xloads],
                       "math_loop_X_LDG128_instructions": [ins for pc, ins in xloads],
                       "initial_plus_loop_X_LDG128_site_count": len(all_xloads),
                       "limit": "TRYWAIT is per-thread/potentially blocking; predicate counts show failed lanes, not initial readiness or exact warp retries without uniformity proof."})
    return mapped


def cpu_check():
    """Read-only hashes/source/CPU-SASS map negative controls; no build/API/GPU call."""
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(ROOT / "proj_rsi/tools/packed_x_ncu.py") == BASE_DRIVER_SHA
    assert sha(HERE / "manifest.json") == MANIFEST_SHA
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert all(Path(name).name == name and sha(HERE / name) == expected for name, expected in manifest["files"].items())
    mod = load(HERE / "prototype.py", "cpu_dual_counter_frozen")
    assert hashlib.sha256(mod.source().encode()).hexdigest() == SOURCE_SHA
    assert "if(info==6)return a.maxThreadsPerBlock" in mod.source()
    assert "k,512,C::SMEM" in mod.source() and "dim3(512)" in mod.source()
    checker = load(HERE / "liveness.py", "cpu_dual_counter_liveness")
    sass = (HERE / "sass.txt").read_text(); gate = checker.check(sass)
    mapping = pc_map(sass, gate)
    assert all(len(m["math_Bfull_waits"]) == 4 and len(m["producer_owner_specific_barriers"]) == 2 for m in mapping)
    first = mapping[0]["math_Bfull_waits"][0]["failed_entry_branch"]
    try:
        pc_map(sass.replace(first, first.replace("@!", "@"), 1), gate)
    except AssertionError:
        pass
    else:
        raise AssertionError("PC map accepted failed-branch predicate inversion")
    return {"state": "CPU_FROZEN_BINDING_AND_PHYSICAL_PC_MAP_PASS_GPU_UNRUN", "CUDA_API_calls": 0,
            "manifest_sha256": MANIFEST_SHA, "source_sha256": SOURCE_SHA,
            "Bfull_fast_and_retry_sites": mapping, "negative_predicate_inversion_caught": True,
            "counter_preregistration": counter_questions()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "STARTED", "purpose": "COUNTERS_PC_ONLY_NO_PERFORMANCE_CLAIM"}
    graph = None
    result["math_launches_started"] = False
    result["phase"] = "SOURCE_BINDING"

    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha(HERE / "manifest.json") == MANIFEST_SHA
        manifest = json.loads((HERE / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            assert sha(HERE / name) == expected, name
        mod = load(HERE / "prototype.py", "profile_frozen_dual_producer")
        assert hashlib.sha256(mod.source().encode()).hexdigest() == SOURCE_SHA
        ext = mod.extension()
        result["phase"] = "JIT_ARCHIVE_BEFORE_PHYSICAL_GATE"
        binary = Path(ext.__file__).resolve()
        assert binary.is_file() and binary.suffix == ".so"
        dump = subprocess.check_output([shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump",
                                        "--dump-sass", str(binary)], text=True, timeout=20)
        artifact = args.json.with_suffix(".jit.sass")
        artifact.write_text(dump)
        result.update(source_sha256=SOURCE_SHA, manifest_sha256=MANIFEST_SHA,
                      JIT_artifact=str(artifact), JIT_sha256=hashlib.sha256(dump.encode()).hexdigest())
        save()
        result["physical_A4_gate"] = load(HERE / "liveness.py", "profile_dual_liveness").check(dump)
        result["Bfull_and_role_PC_map"] = pc_map(dump, result["physical_A4_gate"])
        result["counter_preregistration"] = counter_questions()
        result["phase"] = "RESOURCE_GATE"
        save()
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        sink = torch.zeros(4, device="cuda", dtype=torch.int32)

        def info(code):
            return int(ext.rqrun(0, empty, empty, empty, empty, empty, 0, 0, 0, empty, sink, code, empty))

        attrs = {"main_regs": info(1), "main_local": info(2), "dynamic_shared": info(3),
                 "max_dynamic_before": info(5), "main_max_threads": info(6), "threads": 512}
        assert attrs["main_regs"] == 122 and attrs["main_local"] == 0
        assert attrs["dynamic_shared"] == 103600 and attrs["main_max_threads"] == 512
        attrs["occupancy_after_optin"] = info(4)
        attrs["max_dynamic_after"] = info(5)
        assert attrs["occupancy_after_optin"] >= 1 and attrs["max_dynamic_after"] >= attrs["dynamic_shared"]
        attrs["rounded_register_pool"] = ((attrs["main_regs"] + 7) // 8) * 8 * 512
        assert attrs["rounded_register_pool"] == 65536
        attrs["pack_regs"] = int(ext.packinfo(1)); attrs["pack_local"] = int(ext.packinfo(2))
        attrs["pack_max_threads"] = int(ext.packinfo(3))
        assert attrs["pack_regs"] == 22 and attrs["pack_local"] == 0 and attrs["pack_max_threads"] >= 256
        assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b", dump)
        result["resources"] = attrs
        result["phase"] = "LAB_INPUTS_AND_CAPTURE"
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
            result["math_launches_started"] = True
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
        result["phase"] = "SELECTED_TWO_NODE_GRAPH"
        save()
        print("DUAL_PRODUCER_PROFILE_BEGIN two_nodes", flush=True)
        torch.cuda.nvtx.range_push("profile")
        try:
            graph.replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        # No extra replay or workspace rewrite before these exact selected-output checks.
        assert torch.equal(x, data["x"])
        assert bool(torch.isfinite(out).all()), "selected DP output retains NaN/Inf"
        ok, message = reference.check_implementation({"x": x, "config": data["config"]}, out, {"unpooler": w})
        assert ok, message
        idx = torch.arange(262144, device="cuda", dtype=torch.int64)
        v, lane, chunk, warp, stage = idx & 3, (idx >> 2) & 31, (idx >> 7) & 3, (idx >> 9) & 7, idx >> 12
        row = 64 * (warp // 4) + 16 * (warp % 4) + (lane >> 2) + 8 * (v & 1)
        k = 64 * stage + 16 * chunk + 8 * ((v >> 1) & 1) + 2 * (lane & 3)
        assert torch.equal(xp, x.view(torch.int32).reshape(-1)[row * 2048 + k // 2])
        assert all(torch.equal(value, before) for value, before in immutable)
        assert not int(cnt.abs().sum())
        result.update(status="COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED", phase="IMMEDIATE_ORACLES_PASSED",
                      immediate_selected_output_oracle=message, packed_bytes="EXACT_ALL_1MiB",
                      output_poison_overwritten=True, input_weights_pack_esc_immutable=True,
                      counters="ZERO", profile_nodes_expected=["rx_pack", "rq_px_dp"])
        print("DUAL_PRODUCER_PROFILE_END two_nodes", flush=True)
        save()
    except Exception as error:
        result.update(status="INVALID_COUNTER_DIAGNOSTIC", error=repr(error),
                      no_math_launched=not result["math_launches_started"])
        save()
        raise
    finally:
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
            result["graph_cleanup"] = "RESET_AFTER_DRAIN"
            save()
    print("DUAL_PRODUCER_COUNTER_DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()
