"""Frozen DP/G8 four-node cold counter capture; root executes after independent GO.
Bound: inner240 seconds / outer300 seconds. No timing or optimization claim.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PINS = {
 "DP": ("operand_reverse_dual_producer", "281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241", "97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b"),
 "G8": ("operand_reverse_group8", "2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7", "e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e")}
PARENTS = {"dual_producer_ncu.py": "65e8d9fda91c2b281f7a8b72a693b0f94f37ef1fd4d61fb2160b991a2dc65525", "packed_x_ncu.py": "a3e81cc0adece02ffbf7f28250bb2b798642b94ccdfc63762ea90054b9b74d4d"}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def bind(variant):
    directory, manifest_sha, source_sha = PINS[variant]
    here = ROOT / "proj_rsi/candidates" / directory
    assert sha(here / "manifest.json") == manifest_sha
    manifest = json.loads((here / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert Path(name).name == name and sha(here / name) == expected, name
    mod = load(here / "prototype.py", "compare_frozen_" + variant)
    assert hashlib.sha256(mod.source().encode()).hexdigest() == source_sha
    assert "if(info==6)return a.maxThreadsPerBlock" in mod.source()
    assert "k,512,C::SMEM" in mod.source() and "dim3(512)" in mod.source()
    checker = load(here / "liveness.py", "compare_liveness_" + variant)
    return here, mod, checker

def role_map(rows, loop_lo):
    instructions = dict(rows)
    cta_bar = next(pc for pc, ins in rows if re.search(r"BAR\.SYNC[^ ]* 0x0$", ins))
    branches = [(pc, int(m[1], 16), ins) for pc, ins in rows
                if cta_bar < pc < loop_lo and (m := re.fullmatch(r"@P\d+ BRA 0x([0-9a-f]+)", ins)) and int(m[1], 16) > loop_lo]
    assert len(branches) == 1
    role_pc, producer, role_ins = branches[0]
    succ = {}
    for pc, ins in rows:
        plain = re.sub(r"^@!?P\d+\s+", "", ins)
        branch = re.search(r"\bBRA(?:\.[A-Z]+)* .*?(0x[0-9a-f]+)$", plain)
        if plain == "EXIT": targets = [pc + 16] if ins.startswith("@") else []
        elif branch: targets = [int(branch[1], 16)] + ([pc + 16] if ins.startswith("@") or plain.startswith("BRA.DIV") else [])
        else: targets = [pc + 16]
        succ[pc] = {q for q in targets if q in instructions}
    def reach(start):
        seen, todo = set(), [start]
        while todo:
            pc = todo.pop()
            if pc in seen: continue
            seen.add(pc); todo.extend(succ[pc] - seen)
        return seen
    math, prod = reach(role_pc + 16), reach(producer)
    assert not math & prod and math and prod
    return {"role_branch_PC": hex(role_pc), "role_branch": role_ins,
            "math_start": hex(role_pc + 16), "producer_start": hex(producer),
            "math_PCs": [hex(pc) for pc in sorted(math)], "producer_PCs": [hex(pc) for pc in sorted(prod)],
            "common_setup_unreachable_PCs": [hex(pc) for pc in sorted(set(instructions) - math - prod)]}

def store_proxy_gate(rows, roles):
    prod = {int(p, 16) for p in roles["producer_PCs"]}
    body = [(pc, ins) for pc, ins in rows if pc in prod]
    proxy = [(pc, ins) for pc, ins in body if "FENCE.VIEW.ASYNC.S" in ins]
    stores = [(pc, ins) for pc, ins in body if re.search(r"\bSTS(?:\.| )", ins)]
    assert stores and proxy and max(pc for pc, _ in stores) < proxy[-1][0]
    bars = [(pc, ins) for pc, ins in body if re.search(r"BAR\.SYNC[^ ]* 0x[34], 0x80$", ins)]
    assert len(bars) == 2 and all(pc > proxy[-1][0] for pc, _ in bars)
    assert {re.match(r"(@!?P\d+)", ins)[1] for _, ins in bars} == {"@P0", "@!P0"}
    return {"producer_scalar_STS_PCs": [hex(pc) for pc, _ in stores], "proxy_sites": [(hex(pc), ins) for pc, ins in proxy],
            "owner_barriers": [(hex(pc), ins) for pc, ins in bars], "scope": "source fixed scalar stores precede proxy and complementary owner barriers; not a general memory-model proof"}

def pc_map(sass, physical_gate, variant):
    """Map actual main fast/outlined Bfull predicates/targets, plus distinct role sites."""
    tables = [t for t in re.split(r"\n\s*Function : ", sass)[1:] if ("rq_px_dp" if variant == "DP" else "rq_px_g8") in t.splitlines()[0]]
    assert len(tables) == len(physical_gate["binaries"]) > 0
    mapped = []
    for table, gate in zip(tables, physical_gate["binaries"]):
        rows = [(int(m[1], 16), m[2].strip()) for m in re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);", table)]
        lo, hi = gate["loop_PC_range"]
        slow_targets = gate["outlined_wait_paths_no_A_writes" if variant == "DP" else "outlined_entry_PCs"]
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
                          "failed_retry_branch": retries[0][1], "ready_return": block[-1][1],
                          "outlined_instructions": [{"PC": hex(pc), "instruction": ins} for pc, ins in block]})
        bars = {}
        for barrier, owner in [(3, "producer0_even_warps8..11"), (4, "producer1_odd_warps12..15")]:
            sites = [(pc, ins) for pc, ins in rows if re.search(r"BAR\.SYNC[^ ]* 0x" + str(barrier) + r", 0x80$", ins)]
            assert len(sites) == 1
            bars[owner] = {"PC": hex(sites[0][0]), "instruction": sites[0][1]}
        fence = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and "WARPGROUP.ARRIVE" in ins]
        wait1 = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and "WARPGROUP.DEPBAR.LE" in ins]
        assert len(fence) == len(wait1) == (4 if variant == "DP" else 2)
        xloads = [(pc, ins) for pc, ins in rows if lo <= pc <= hi and re.search(r"\bLDG\.[^ ]*\.128(?:\.| )", ins)]
        all_xloads = [(pc, ins) for pc, ins in rows if re.search(r"\bLDG\.[^ ]*\.128(?:\.| )", ins)]
        assert len(xloads) == 16 and len(all_xloads) == 32, "unexpected fixed packed-X vector-load map"
        roles = role_map(rows, lo)
        stores = store_proxy_gate(rows, roles)
        mapped.append({"roles": roles, "store_proxy_gate": stores, "function": table.splitlines()[0], "math_warps": [0, 7], "producer_warps": [8, 15],
                       "math_Bfull_waits": waits, "producer_owner_specific_barriers": bars,
                       "math_fence_PCs": [hex(pc) for pc, ins in fence],
                       "math_retirement_wait1_PCs": [hex(pc) for pc, ins in wait1],
                       "math_loop_X_LDG128_PCs": [hex(pc) for pc, ins in xloads],
                       "math_loop_X_LDG128_instructions": [ins for pc, ins in xloads],
                       "initial_plus_loop_X_LDG128_site_count": len(all_xloads),
                       "limit": "TRYWAIT is per-thread/potentially blocking; predicate counts show failed lanes, not initial readiness or exact warp retries without uniformity proof."})
    return mapped

def preregistration():
    return {"purpose": "COUNTERS_PC_ONLY_NO_TIME_DECOMPOSITION", "bounds_seconds": {"inner": 240, "outer": 300},
            "questions": ["Reduced X lookahead exposes L2 A-load dependencies at retained fences?", "Coarser B-pair retirement increases producer Bempty backpressure?", "Wait samples move to other consumers without pipeline gain?"],
            "sections": ["LaunchStats", "SourceCounters", "MemoryWorkloadAnalysis_Tables", "SchedulerStats", "WarpStateStats", "SpeedOfLight"],
            "sample_sufficiency": {"math_per_variant": 200, "producer_per_variant": 200, "leading_phase": 30},
            "interpretation": ["Retain full parent profile denominators, math/prod/common partitions and all outlined paths.",
                               "Predicated-On Thread Instructions Executed counts failed lanes, not initial readiness; TRYWAIT can block and per-thread predicates need not be uniform.",
                               "No exact warp count from division by32; only warp-equivalent normalization without independent uniformity proof.",
                               "SourceInstructionsExecuted can differ across HW replay passes; retain source/raw totals independently.",
                               "Per-node cold replay differs from normal pack-to-main consumption. No PC fraction to microseconds; no NCU duration pipeline benchmark.",
                               "Both have four Bfull fast/outlined sites,16loop/32total LDG128; DP has4fence/wait1 sites, G8 has2."]}

def cpu_check():
    assert all(sha(ROOT / "proj_rsi/tools" / name) == expected for name, expected in PARENTS.items())
    tree = ast.parse(Path(__file__).read_text())
    cpu_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"cpu_check", "bind", "pc_map", "role_map", "store_proxy_gate"}]
    assert not any(isinstance(n, ast.Attribute) and n.attr in {"cuda", "extension", "prepare", "load_inline"} for node in cpu_nodes for n in ast.walk(node))
    result = {"state": "CPU_BINDING_MAP_PASS_GPU_UNRUN", "CUDA_API_calls": 0, "extension_builds": 0, "variants": {}}
    for variant in PINS:
        here, mod, checker = bind(variant)
        sass = (here / "sass.txt").read_text()
        gate = checker.check(sass)
        mapping = pc_map(sass, gate, variant)
        first = mapping[0]["math_Bfull_waits"][0]["failed_entry_branch"]
        try:
            pc_map(sass.replace(first, first.replace("@!", "@"), 1), gate, variant)
        except AssertionError:
            pass
        else:
            raise AssertionError("predicate-inversion negative control accepted")
        result["variants"][variant] = {"manifest_sha256": PINS[variant][1], "source_sha256": PINS[variant][2],
                                        "physical_gate": gate, "PC_map": mapping, "negative_predicate_inversion_caught": True}
    result["preregistration"] = preregistration()
    return result

def run(args):
    import torch
    sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
    import reference
    result = {"status": "STARTED", "phase": "SOURCE_BINDING", "math_launches_started": False,
              "driver_sha256": sha(__file__), "preregistration": preregistration(), "variants": {}}
    graphs = []
    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    try:
        assert all(sha(ROOT / "proj_rsi/tools" / name) == expected for name, expected in PARENTS.items())
        modules = {}
        # Archive every actual binary before any math launch or resource/input work.
        for variant in PINS:
            here, mod, checker = bind(variant)
            ext = mod.extension()
            binary = Path(ext.__file__).resolve()
            assert binary.is_file() and binary.suffix == ".so"
            dump = subprocess.check_output([shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)], text=True, timeout=20)
            artifact = args.json.with_suffix("." + variant + ".jit.sass")
            artifact.write_text(dump)
            entry = {"source_sha256": PINS[variant][2], "manifest_sha256": PINS[variant][1], "JIT_artifact": str(artifact), "JIT_sha256": hashlib.sha256(dump.encode()).hexdigest()}
            result["variants"][variant] = entry
            save()
            entry["physical_gate"] = checker.check(dump)
            entry["PC_map"] = pc_map(dump, entry["physical_gate"], variant)
            assert not re.search(r"\b(?:LDL|STL|USETMAXREG)\b", dump)
            modules[variant] = (mod, ext)
            save()
        result["phase"] = "RESOURCE_GATE"
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        sink = torch.zeros(4, device="cuda", dtype=torch.int32)
        for variant, (mod, ext) in modules.items():
            def info(code):
                return int(ext.rqrun(0, empty, empty, empty, empty, empty, 0, 0, 0, empty, sink, code, empty))
            attrs = {"main_regs": info(1), "main_local": info(2), "dynamic_shared": info(3), "max_dynamic_before": info(5), "main_max_threads": info(6), "threads": 512}
            assert (attrs["main_regs"], attrs["main_local"], attrs["dynamic_shared"], attrs["main_max_threads"]) == (122, 0, 103600, 512)
            attrs["occupancy_after_optin"] = info(4)
            attrs["max_dynamic_after"] = info(5)
            assert attrs["occupancy_after_optin"] == 1 and attrs["max_dynamic_after"] >= 103600
            attrs["rounded_register_pool"] = ((attrs["main_regs"] + 7) // 8) * 8 * 512
            assert attrs["rounded_register_pool"] == 65536
            attrs.update(pack_regs=int(ext.packinfo(1)), pack_local=int(ext.packinfo(2)), pack_max_threads=int(ext.packinfo(3)))
            assert attrs["pack_regs"] == 22 and attrs["pack_local"] == 0 and attrs["pack_max_threads"] >= 256
            result["variants"][variant]["resources"] = attrs
        save()
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        w = weights["unpooler"]
        data = reference.generate_input(128, "unpooler", 5237)
        work = {}
        for variant, (mod, ext) in modules.items():
            packed, table, esc6 = mod.S._rqpack(w)
            assert torch.equal(mod.S._rqunpack(packed, table, esc6, *w.shape), w.view(torch.int16).int() & 65535)
            esc, _ = mod.S._rqesc(esc6, packed.shape[0], packed.shape[1], 1, "cuda")
            x = data["x"].clone()
            xp = torch.empty(262144, device="cuda", dtype=torch.int32)
            out = torch.empty((128, 8192), device="cuda", dtype=torch.bfloat16)
            cnt = torch.zeros(4 * packed.shape[0], device="cuda", dtype=torch.int32)
            immutable = [(v, v.clone()) for v in (w, packed, esc, x)]
            def call(ext=ext, packed=packed, table=table, esc=esc, x=x, xp=xp, out=out, cnt=cnt):
                result["math_launches_started"] = True
                rc = int(ext.rqrun(0, packed, x, out, esc, cnt, 1, table, 0, empty, sink, 0, xp))
                assert rc == 0, rc
            call(); call(); torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph): call()
            work[variant] = (x, xp, out, cnt, immutable, graph)
        ptrs = [v.data_ptr() for ws in work.values() for v in ws[:4]]
        assert len(set(ptrs)) == len(ptrs), "workspaces must be fresh and disjoint"
        for x, xp, out, cnt, immutable, graph in work.values():
            xp.fill_(0x5A5A5A5A); out.fill_(float("nan"))
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        flush_sink = torch.empty((), device="cuda", dtype=torch.float32)
        torch.sum(flush, 0, out=flush_sink); torch.cuda.synchronize()
        result["phase"] = "SELECTED_FOUR_NODE_RANGE"
        save()
        print("GROUP8_COMPARE_PROFILE_BEGIN four_nodes_DPpack_DPmain_G8pack_G8main", flush=True)
        torch.cuda.nvtx.range_push("profile")
        try:
            work["DP"][-1].replay()
            work["G8"][-1].replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        # All exact selected graph outputs and immutable inputs are checked before any replay/workspace mutation.
        for variant, (x, xp, out, cnt, immutable, graph) in work.items():
            assert bool(torch.isfinite(out).all()), variant + " selected output retains poison/Inf"
            ok, message = reference.check_implementation({"x": x, "config": data["config"]}, out, {"unpooler": w})
            assert ok, (variant, message)
            idx = torch.arange(262144, device="cuda", dtype=torch.int64)
            v, lane, chunk, warp, stage = idx & 3, (idx >> 2) & 31, (idx >> 7) & 3, (idx >> 9) & 7, idx >> 12
            row = 64 * (warp // 4) + 16 * (warp % 4) + (lane >> 2) + 8 * (v & 1)
            k = 64 * stage + 16 * chunk + 8 * ((v >> 1) & 1) + 2 * (lane & 3)
            assert torch.equal(xp, x.view(torch.int32).reshape(-1)[row * 2048 + k // 2])
            assert torch.equal(x, data["x"]) and all(torch.equal(value, before) for value, before in immutable)
            assert not int(cnt.abs().sum())
            result["variants"][variant]["selected_oracles"] = {"output": message, "XP": "EXACT_ALL_1MiB", "W_pack_esc_X": "IMMUTABLE", "cnt": "ZERO", "poison_overwritten": True}
            save()
        result.update(status="COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED", phase="IMMEDIATE_ORACLES_PASSED", profile_nodes_expected=["DP_rx_pack", "rq_px_dp", "G8_rx_pack", "rq_px_g8"])
        print("GROUP8_COMPARE_PROFILE_END four_nodes", flush=True)
        save()
    except BaseException as error:
        result.update(status="INVALID_COUNTER_DIAGNOSTIC", error=repr(error), no_math_launched=not result["math_launches_started"])
        save()
        raise
    finally:
        cleanup = []
        for graph in graphs:
            try:
                torch.cuda.synchronize(); graph.reset(); cleanup.append("RESET_AFTER_DRAIN")
            except BaseException as error:
                cleanup.append(repr(error))
                result["status"] = "INVALID_CLEANUP"
        result["graph_cleanup"] = cleanup
        save()
    print("GROUP8_COMPARE_COUNTER_DRIVER_DONE", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_check:
        result = cpu_check()
        result["driver_sha256"] = sha(__file__)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"state": result["state"], "driver_sha256": result["driver_sha256"], "CUDA_API_calls": 0}))
    else:
        run(args)

if __name__ == "__main__":
    main()
