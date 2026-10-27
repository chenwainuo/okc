"""Frozen DP/native-X three-node cold counter capture; root executes after independent GO.
Bound: inner240 seconds / outer300 seconds. No timing or optimization claim.
"""
import argparse
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
 "NX": ("reverse_tma_x", "025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc", "ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377")}
PARENTS = {"group8_compare_ncu.py": "245b78898a16c2087b00924cd809cd33fb74bfd7c19c9012bd7d040005485702"}
TRIAL_SHA = "a73cff33a3d87ca925f9bd025c70ad1bfeeac1c94799e4fcd23de45a9e8f4821"

def support_module():
    trial = ROOT / "proj_rsi/candidates/reverse_tma_x_trial"
    assert sha(trial / "manifest.json") == TRIAL_SHA
    support = load(trial / "native_x_support.py", "profile_native_support")
    support.verify_lab()
    support.verify_frozen()
    sys.path.insert(0, str(ROOT / "proj_rsi/candidates/reverse_tma_x"))
    # Bind proofs' absolute liveness import to the native frozen dependency.
    sys.modules["liveness"] = load(ROOT / "proj_rsi/candidates/reverse_tma_x/liveness.py", "profile_native_liveness")
    return support

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
    """Actual CFG roles, source-address-qualified wait sites and unclassified residue.

    No PC constants. Exact canonical equality is a prerequisite, not an inference
    from line ordering. Address constants below derive from frozen region sizes.
    """
    support = support_module()
    token = "rq_px_dp" if variant == "DP" else "rq_tma_x"
    table = support.main_table(sass, token)
    rows = support.instructions(table)
    lo, hi = physical_gate["binaries"][0]["loop_PC_range"]
    roles = role_map(rows, lo)
    math = {int(p, 16) for p in roles["math_PCs"]}
    prod = {int(p, 16) for p in roles["producer_PCs"]}
    # Frozen source defines: W6*6144, B8*8192, 6 Wfull8B,
    # 8 Bfull8B, 8 Bempty8B; native Xfull=103424+32768.
    wfull = 6*6144 + 8*8192
    bfull = wfull + 6*8
    bempty = bfull + 8*8
    xfull = 103424 + 2*16384
    source = (ROOT / "proj_rsi/candidates" / PINS[variant][0] / "kernel_body.cuh").read_text()
    assert "wfull=bring+BST*BB,bfull=wfull+8*WST,bempty=bfull+8*BST" in source
    if variant == "NX":
        assert "xring=base+103424, xfull=xring+32768" in source
        assert "rq_wait(xfull+8*slot,(s/2)&1)" in source
    waits = []; classified = set()
    for index, (pc, ins) in enumerate(rows):
        if "SYNCS.PHASECHK" not in ins or "TRYWAIT" not in ins: continue
        role = "math" if pc in math else "producer" if pc in prod else "common"
        offset_match = re.search(r"\[R\d+\+URZ\+0x([0-9a-f]+)\]", ins)
        offset = int(offset_match[1], 16) if offset_match else None
        kind = ("math_Xfull" if role == "math" and variant == "NX" and offset in (xfull, xfull+8)
                else "math_Bfull" if role == "math" and offset == bfull
                else "producer_Wfull" if role == "producer" and offset == wfull
                else "producer_Bempty" if role == "producer" and offset == bempty else "unclassified_wait")
        predicate = re.search(r"TRYWAIT (P\d+),", ins)[1]
        # Match branch by predicate, stopping at a predicate rewrite/next wait.
        branch = None
        for q, text in rows[index+1:]:
            if "TRYWAIT" in text: break
            m = re.fullmatch(r"(@!?P\d+) BRA 0x([0-9a-f]+)", text)
            if m and m[1].lstrip("@!") == predicate:
                assert m[1] == "@!" + predicate, "failed-wait branch must negate TRYWAIT"
                branch = {"PC": hex(q), "instruction": text, "target_PC": hex(int(m[2],16))}
                break
            if re.match(r"(?:ISETP|PLOP3).*\b" + predicate + r",", text): break
        assert branch is not None, (hex(pc), ins, "no predicate-bound failed branch")
        waits.append({"PC":hex(pc),"instruction":ins,"role":role,"classification":kind,
                      "source_region_offset":offset,"failed_branch":branch,
                      "placement":"loop" if lo<=pc<=hi else "startup_or_outlined"})
        if kind != "unclassified_wait": classified.add(pc)
    by_pc = {pc:ins for pc,ins in rows}
    for wait in waits:
        target = int(wait["failed_branch"]["target_PC"],16)
        pc = int(wait["PC"],16)
        if target <= pc:
            wait["wait_path"] = "outlined_retry"
            continue
        tests = [(q,ins) for q,ins in rows if target <= q <= target+32 and "TRYWAIT" in ins]
        assert len(tests) == 1 and tests[0][1] == wait["instruction"]
        wait["wait_path"] = "fast_entry_to_outlined"
        wait["outlined_try_PC"] = hex(tests[0][0])
        for q,ins in rows:
            if q <= tests[0][0]: continue
            ret = re.fullmatch(r"BRA 0x([0-9a-f]+)",ins)
            if ret:
                assert int(ret[1],16) == int(wait["failed_branch"]["PC"],16)+16
                wait["ready_return"] = {"PC":hex(q),"instruction":ins}
                break
        assert "ready_return" in wait
    assert all(w["classification"] != "unclassified_wait" for w in waits)
    sites = {}
    patterns = {"math_bar5": r"BAR\.SYNC[^ ]* 0x5, 0x100$", "math_LDSM": r"LDSM\.16\.M88\.4",
                "math_TMA": r"UTMALDG\.2D", "math_LDG128": r"\bLDG\.[^ ]*\.128(?:\.| )", "math_fence": r"WARPGROUP\.ARRIVE",
                "math_wait1": r"WARPGROUP\.DEPBAR\.LE.*0x1$", "math_wait0": r"WARPGROUP\.DEPBAR\.LE.*0x0$", "math_proxy": r"FENCE\.VIEW\.ASYNC\.S",
                "producer_owner_barriers": r"BAR\.SYNC[^ ]* 0x[34], 0x80$"}
    for key, pattern in patterns.items():
        owned = prod if key.startswith("producer") else math
        sites[key] = [{"PC":hex(pc),"instruction":ins} for pc,ins in rows if pc in owned and re.search(pattern,ins)]
        classified.update(int(x["PC"],16) for x in sites[key])
    if variant == "NX":
        assert [len(sites[k]) for k in ("math_LDSM","math_bar5","math_TMA")] == [32,8,8]
        assert not any(re.search(r"\bLDG\.",i) for p,i in rows if p in math)
    else:
        assert len(sites["math_LDG128"]) == 16
        sites["common_setup_LDG128"] = [{"PC":hex(pc),"instruction":ins} for pc,ins in rows if pc not in math|prod and re.search(patterns["math_LDG128"],ins)]
        assert len(sites["common_setup_LDG128"]) == 16
        assert sum(lo <= int(site["PC"],16) <= hi for site in sites["math_LDG128"]) == 16
    assert len(sites["math_fence"]) == len(sites["math_wait1"]) == 4
    assert len(sites["math_wait0"]) == 1
    stores = store_proxy_gate(rows, roles)
    return [{"function":table.splitlines()[0],"roles":roles,"loop_PC_range":[lo,hi],
             "source_region_offsets":{"Wfull":wfull,"Bfull":bfull,"Bempty":bempty,"Xfull":xfull if variant=="NX" else None},
             "waits":waits,"sites":sites,"store_proxy_gate":stores,
             "unclassified_instruction_PCs":[hex(p) for p,i in rows if p not in classified],
             "limits":"Classifications require exact canonical main and frozen source. Retain all residuals; TRYWAIT can block; failed predicate lanes are not initial readiness."}]

def preregistration():
    return {"purpose":"COUNTERS_PC_ONLY_NO_TIME_DECOMPOSITION", "bounds_seconds":{"inner":240,"outer":300},
            "questions":["Does math Xfull/bar5/LDSM/TMA refill introduce dependency or barrier stalls?", "Does producer Wfull/Bempty backpressure dominate retained samples?", "Do memory traffic and scheduler counters support either hypothesis?"],
            "sections":["LaunchStats","SourceCounters","MemoryWorkloadAnalysis_Tables","SchedulerStats","WarpStateStats","SpeedOfLight"],
            "sample_sufficiency":{"math_per_variant":200,"producer_per_variant":200,"leading_phase":30},
            "interpretation":["Retain raw counts and full denominators, math/producer/common partitions and unclassified residuals.",
              "PC fractions are not microseconds. NCU durations are not pipeline benchmarks.",
              "TRYWAIT may block; predicated instruction counts are failed lanes, not initial readiness or warp retries.",
              "No exact warp count from divide-by32 without independent uniformity proof.",
              "Per-node cache-all replay differs from normal DP pack-to-main consumption; counter differences alone do not establish speed causality.",
              "Native raw hardware X TMA/LDSM storage-bit roundtrip remains UNPROBED."]}

def _cpu_check_impl():
    assert all(sha(ROOT / "proj_rsi/tools" / name)==expected for name,expected in PARENTS.items())
    support = support_module()
    result = {"state":"CPU_BINDING_MAP_PASS_GPU_UNRUN", "CUDA_API_calls":0,"extension_builds":0,"variants":{}}
    for variant in PINS:
        here, mod, checker = bind(variant)
        sass = (here / "sass.txt").read_text()
        canonical = support.actual_binary_gate("dp" if variant=="DP" else "new",sass)
        gate = checker.check(sass)
        mapping = pc_map(sass,gate,variant)
        first = mapping[0]["waits"][0]["failed_branch"]["instruction"]
        for label, changed in (("predicate_inversion",sass.replace(first,first.replace("@!","@"),1)),
                               ("PC_binding",sass.replace("/*" + format(int(mapping[0]["roles"]["math_start"],16),"04x") + "*/", "/*" + format(int(mapping[0]["roles"]["math_start"],16)+1,"04x") + "*/",1)),
                               ("role_branch",sass.replace(mapping[0]["roles"]["role_branch"],"BRA 0x0",1))):
            assert changed != sass,(label,"mutation did not bind")
            try:
                support.actual_binary_gate("dp" if variant=="DP" else "new",changed)
            except AssertionError: pass
            else: raise AssertionError(label+" negative control accepted")
        # Mapper itself must reject failed-predicate inversion, independently of canonical gate.
        try: pc_map(sass.replace(first,first.replace("@!","@"),1),gate,variant)
        except AssertionError: pass
        else: raise AssertionError("map predicate negative control accepted")
        result["variants"][variant] = {"manifest_sha256":PINS[variant][1],"source_sha256":PINS[variant][2],
            "canonical_gate":canonical,"physical_gate":gate,"PC_map":mapping,
            "negative_controls":["predicate_inversion","PC_binding","role_branch","mapper_predicate_inversion"]}
    result["preregistration"] = preregistration()
    return result

def cpu_check():
    # Imports are permitted; executing a torch CUDA API or build routine is not.
    # A Python call-profile tripwire fails before any such body can execute.
    import torch
    assert not torch.cuda._initialized
    import types
    build_name = "torch.utils.cpp_extension"
    previous_build = sys.modules.get(build_name)
    stub = types.ModuleType(build_name)
    def forbid_build(*args, **kwargs):
        raise AssertionError("CPU mode attempted extension build")
    stub.load_inline = forbid_build
    sys.modules[build_name] = stub
    previous = sys.getprofile()
    def reject_cuda_call(frame, event, arg):
        if event != "call": return
        filename = frame.f_code.co_filename.replace("\\", "/")
        assert "/torch/cuda/" not in filename and not filename.endswith("cpp_extension.py"), ("CPU mode attempted CUDA/build API",filename,frame.f_code.co_name)
    sys.setprofile(reject_cuda_call)
    try:
        result = _cpu_check_impl()
    finally:
        sys.setprofile(previous)
        if previous_build is None: sys.modules.pop(build_name,None)
        else: sys.modules[build_name] = previous_build
    assert not torch.cuda._initialized
    result["CPU_runtime_guard"] = "PASS_NO_TORCH_CUDA_OR_CPP_EXTENSION_CALLS_NO_CONTEXT_INITIALIZED"
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
        support = support_module()
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
            entry["canonical_gate"] = support.actual_binary_gate("dp" if variant == "DP" else "new", dump)
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
            assert (attrs["main_regs"], attrs["main_local"], attrs["dynamic_shared"], attrs["main_max_threads"]) == (122, 0, 137232 if variant == "NX" else 103600, 512)
            attrs["occupancy_after_optin"] = info(4)
            attrs["max_dynamic_after"] = info(5)
            assert attrs["occupancy_after_optin"] == 1 and attrs["max_dynamic_after"] >= attrs["dynamic_shared"]
            attrs["rounded_register_pool"] = ((attrs["main_regs"] + 7) // 8) * 8 * 512
            assert attrs["rounded_register_pool"] == 65536
            if variant == "DP":
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
            xp = torch.empty(262144 if variant == "DP" else 0, device="cuda", dtype=torch.int32)
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
        ptrs = [v.data_ptr() for ws in work.values() for v in ws[:4] if v.numel()]
        assert len(set(ptrs)) == len(ptrs), "workspaces must be fresh and disjoint"
        for x, xp, out, cnt, immutable, graph in work.values():
            xp.fill_(0x5A5A5A5A); out.fill_(float("nan"))
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        flush_sink = torch.empty((), device="cuda", dtype=torch.float32)
        torch.sum(flush, 0, out=flush_sink); torch.cuda.synchronize()
        result["phase"] = "SELECTED_THREE_NODE_RANGE"
        save()
        print("NATIVE_X_COMPARE_PROFILE_BEGIN three_nodes_DPpack_DPmain_NXmain", flush=True)
        torch.cuda.nvtx.range_push("profile")
        try:
            work["DP"][-1].replay()
            work["NX"][-1].replay()
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
            if variant == "DP":
                assert torch.equal(xp, x.view(torch.int32).reshape(-1)[row * 2048 + k // 2])
            else:
                assert xp.numel() == 0
            assert torch.equal(x.view(torch.uint8), data["x"].view(torch.uint8)) and all(torch.equal(value.view(torch.uint8), before.view(torch.uint8)) for value, before in immutable)
            assert not int(cnt.abs().sum())
            result["variants"][variant]["selected_oracles"] = {"output": message, "XP": "EXACT_ALL_1MiB" if variant == "DP" else "NONE_RAW_TMA_LDSM_BITS_UNPROBED", "W_pack_esc_X": "IMMUTABLE", "cnt": "ZERO", "poison_overwritten": True}
            save()
        result.update(status="COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED", phase="IMMEDIATE_ORACLES_PASSED", profile_nodes_expected=["DP_rx_pack", "rq_px_dp", "rq_tma_x"])
        print("NATIVE_X_COMPARE_PROFILE_END three_nodes", flush=True)
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
    print("NATIVE_X_COMPARE_COUNTER_DRIVER_DONE", flush=True)

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
