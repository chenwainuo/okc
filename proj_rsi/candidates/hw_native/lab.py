"""Conditional root-owned unpooler128 allocation lab; no new CUDA kernel.

Requires a separately reviewed, hash-pinned ownership-fixed allocator. Never imports
the invalid packed pilot's allocator implicitly. This is a lab, not a submission.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "proj_rsi/candidates/pw36/submission.py"
PARENT_SHA = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
LAYER, M, N, K = "unpooler", 128, 8192, 4096
LABELS = ("ordinary_lt", "generic_lt", "none_lt", "incumbent")
HELPER_DIR = HERE.parent / "hw_compression_owner"
HELPER_MANIFEST_SHA = "8f32b91ee4146b2c9534360897d5953850f57b370c1ae653b5b04d325d80c781"
HELPER_ALLOCATOR_SHA = "7f2dea1ecdcbff20efd1daad1a4250f775bbf9cfb8b9a76a16f257c1268140f8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_helper_binding(directory, manifest_sha):
    """Hash-only binding gate; no helper import/build or Driver/CUDA call."""
    directory = Path(directory).resolve()
    assert directory == HELPER_DIR.resolve(), "only the independently reviewed lifetime-proven helper is admitted"
    assert manifest_sha == HELPER_MANIFEST_SHA, "unreviewed helper manifest"
    manifest_path = directory / "manifest.json"
    assert len(manifest_sha) == 64 and digest(manifest_path) == manifest_sha, "helper manifest drift"
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest["sha256"]
    assert {"adapter.py", "allocator.cpp"} <= set(hashes), "helper must bind allocator+adapter"
    assert hashes["allocator.cpp"] == HELPER_ALLOCATOR_SHA, "ownership-fixed allocator identity drift"
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        assert path.parent == directory and path.is_file(), "helper manifest path escape"
        assert digest(path) == expected, f"helper {name} drift"
    return directory, hashes


def load_fixed_helper(directory, manifest_sha):
    """No allocator source copying and no implicit mutable/v1 dependency."""
    directory, hashes = verify_helper_binding(directory, manifest_sha)
    adapter = load_file(directory / "adapter.py", "hw_native_fixed_" + manifest_sha[:12])
    ext = adapter.extension()  # Build before importing pw36/starting its prepare clock.
    for name in ("WeightClone", "capabilities", "pointer_properties", "abi"):
        assert hasattr(ext, name), "fixed helper API missing " + name
    assert hasattr(ext.WeightClone, "counts"), "ownership-fixed helper counts() required"
    return ext, {"directory": str(directory), "manifest_sha256": manifest_sha,
                 "sha256": hashes, "ABI": dict(ext.abi())}


def selected_source(state):
    code = state["lt"].get((LAYER, M))
    if not isinstance(code, int) or code < 0:
        return None
    layout = code // 100000
    assert layout in range(4)
    key = LAYER + ("_t" if layout & 2 else "_raw")
    source = state[key]
    shape = (K, N) if layout & 2 else (N, K)
    assert source.dtype == torch.bfloat16 and source.is_contiguous() and tuple(source.shape) == shape
    return code, layout, key, source


class Case:
    """All graph/output/state aliases are dropped before any external owner closes."""
    def __init__(self, B, state, label, code, key=None, owner=None):
        self.B, self.label, self.code = B, label, code
        self.state = state if owner is None else dict(state)
        self.owner = owner
        if owner is not None:
            # dtype view only: no cast/copy/repack; state owns the only long-lived BF16 view.
            self.state[key] = owner.tensor.view(torch.bfloat16)
            assert self.state[key].data_ptr() == owner.info()["ptr"]
            assert self.state[key].shape == state[key].shape
            assert self.state[key].stride() == state[key].stride()
        self.graph = self.hold = self.out = None

    @property
    def config(self):
        return {"layer": LAYER, "m": M, "n": N, "k": K}

    def call(self, x):
        assert self.state["lt"][(LAYER, M)] == self.code, "algorithm changed after freeze"
        if self.label == "incumbent":
            self.hold = self.B.kernel_fn({"x": x, "config": self.config}, self.state)
        else:
            # Existing fallback only. Do not delete RQ entries or weaken the incumbent.
            self.hold = self.B._v12(x, LAYER, N, self.state)
        return self.hold

    def validator(self, x, dst):
        dst.copy_(self.call(x))  # Untimed oracle only; captured graphs contain no copy node.
        return 0

    def capture(self, x):
        assert self.graph is None
        self.call(x); self.call(x); torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.call(x)
        self.out = self.hold

    def drop_graph(self):
        if self.graph is not None:
            self.graph.reset()
        self.graph = None

    def drop_aliases(self):
        self.hold = self.out = self.state = self.owner = None


def validate(B, reference, case, w):
    gen = torch.Generator(device=w.device).manual_seed(8841)
    ok, message = B._rqval(case.validator, w, M, N, gen, 4e-3)
    assert ok, (case.label, message)
    checks = [{"all_K_onehot": "EXACT", "K": K, "FP64_oracle": message}]
    for distribution in ("normal", "outliers"):
        data = reference.generate_input(M, LAYER, 8843, dist=distribution)
        before = data["x"].clone()
        out = case.call(data["x"])
        ok, message = reference.check_implementation(data, out, {LAYER: w})
        assert ok and torch.equal(data["x"], before), (case.label, distribution, message)
        checks.append({"distribution": distribution, "oracle": message, "input_immutable": True})
    return checks


def graph_checks(reference, cases, x, w):
    original = x.clone()
    probes = torch.zeros_like(x)
    rows = torch.arange(M, device=x.device)
    probes[rows, (rows * 37 + 19) % K] = ((2.0 ** ((rows % 5) + 3)) * (2 * (rows % 2) - 1)).to(torch.bfloat16)
    mutations = [("original", original), ("zero", torch.zeros_like(x)),
                 ("fresh_normal", reference.generate_input(M, LAYER, 8847)["x"]),
                 ("fresh_outliers", reference.generate_input(M, LAYER, 8849, dist="outliers")["x"]),
                 ("all_probes", probes)]
    checks = []
    for name, fresh in mutations:
        x.copy_(fresh)
        for case in cases:
            before = x.clone()
            case.out.fill_(float("nan")); case.graph.replay(); torch.cuda.synchronize()
            assert torch.equal(before, x) and bool(torch.isfinite(case.out).all()), (case.label, name)
            if name == "zero":
                assert not int(torch.count_nonzero(case.out))
                message = "all elements exactly zero"
            else:
                ok, message = reference.check_implementation({"x": x, "config": case.config}, case.out, {LAYER: w})
                assert ok, (case.label, name, message)
            checks.append({"case": case.label, "mutation": name, "oracle": message,
                           "NaN_poison_overwritten": True, "input_immutable": True})
    x.copy_(original)
    for case in cases:
        case.graph.replay()
    torch.cuda.synchronize()
    return checks


def fresh_output_checks(reference, cases, x, w):
    """Additional capture/output allocation, outside timing; never replaces timed graph."""
    checks = []
    for case in cases:
        saved = case.out
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = case.call(x)
            assert out.data_ptr() != saved.data_ptr(), "fresh graph output reused live old output"
            out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
            ok, message = reference.check_implementation({"x": x, "config": case.config}, out, {LAYER: w})
            assert ok and bool(torch.isfinite(out).all()), (case.label, message)
            checks.append({"case": case.label, "oracle": message, "fresh_output": True})
        finally:
            torch.cuda.synchronize()
            graph.reset(); graph = None
            case.hold = saved
    return checks


def profile(case, trace_path):
    from torch.profiler import profile as profiler, ProfilerActivity
    with profiler(activities=[ProfilerActivity.CUDA]) as p:
        case.graph.replay(); torch.cuda.synchronize()
    p.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    kernels = []
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        assert "grid" in args and "block" in args, "profile lacks launch identity"
        kernels.append({"name": event["name"], "grid": args["grid"], "block": args["block"],
                        "duration_us": event["dur"], "registers": args.get("registers per thread"),
                        "shared_bytes": args.get("shared memory")})
    assert kernels, "profile has no kernel identity"
    # CUDA event list catches hidden copies not present in cat=kernel.
    cuda_names = [e.name() for e in p.profiler.kineto_results.events()
                  if e.device_type() == torch.autograd.DeviceType.CUDA]
    assert cuda_names == [k["name"] for k in kernels], ("hidden CUDA event", cuda_names)
    return kernels


def profile_identity(kernels):
    return [{k: row[k] for k in ("name", "grid", "block", "registers", "shared_bytes")} for row in kernels]


def measure(cases):
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.float32)
    sink = torch.empty((), device="cuda", dtype=torch.float32)
    def one(case):
        torch.sum(flush, 0, out=sink)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); case.graph.replay(); end.record(); torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000
    for case in cases:
        for _ in range(5):
            one(case)
    def series(index):
        return [one(cases[index]) for _ in range(21)]
    ordinary_before, incumbent_before = series(0), series(3)
    generic, none = series(1), series(2)
    incumbent_after, ordinary_after = series(3), series(0)
    samples = {"ordinary_before": ordinary_before, "ordinary_after": ordinary_after,
               "incumbent_before": incumbent_before, "incumbent_after": incumbent_after,
               "generic": generic, "none": none}
    f1 = {name: statistics.median(value) for name, value in samples.items()}
    f1["ordinary"] = min(f1["ordinary_before"], f1["ordinary_after"])
    f1["incumbent"] = min(f1["incumbent_before"], f1["incumbent_after"])
    orders = []
    for seq in [(0, 1, 2, 3), (3, 2, 1, 0)]:
        orders.extend([seq[s:] + seq[:s] for s in range(4)])
    pairs = []
    for j in range(21):
        times = [None] * 4
        for index in orders[j % len(orders)]:
            times[index] = one(cases[index])
        pairs.append(dict(zip(LABELS, times)))
    f1_gain = {label: f1[name] - f1["generic"] for label, name in
               [("ordinary_lt", "ordinary"), ("none_lt", "none"), ("incumbent", "incumbent")]}
    paired_gain = {label: statistics.median(p[label] - p["generic_lt"] for p in pairs) for label in f1_gain}
    survives = all(f1_gain[label] >= bar and paired_gain[label] >= bar
                   for label, bar in [("ordinary_lt", .5), ("incumbent", .5), ("none_lt", .3)])
    return {"F1_medians_us": f1, "F1_samples_us": samples, "F1_gain_us": f1_gain,
            "paired_median_gain_us": paired_gain, "paired_samples_us": pairs,
            "provisional_gain_gate": survives}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--helper-dir", type=Path, required=True)
    ap.add_argument("--helper-manifest-sha", required=True)
    args = ap.parse_args()
    own = json.loads((HERE / "manifest.json").read_text())
    for name, sha in own["sha256"].items():
        assert digest(HERE / name) == sha, "native source drift: " + name
    assert digest(PARENT) == PARENT_SHA
    result = {"status": "RUNNING_NO_ADMISSION", "policy": "unpooler128_native_BF16_allocation_only",
              "parent_sha256": PARENT_SHA, "source_sha256": own["sha256"], "cleanup": {},
              "population": "cuMemcpyDtoDAsync_then_stream_synchronize",
              "effective_hardware_compression": "UNMEASURED"}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.json.with_suffix(args.json.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n"); temp.replace(args.json)
    cases, owners = [], []
    failed = False
    try:
        ext, result["fixed_helper"] = load_fixed_helper(args.helper_dir, args.helper_manifest_sha)
        B = load_file(PARENT, "hw_native_pw36")
        sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
        reference = load_file(ROOT / "muse_bf16_proj/reference.py", "hw_native_reference")
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        state = B.prepare(weights)
        torch.cuda.synchronize()
        result["runtime"] = {"torch": torch.__version__, "CUDA": torch.version.cuda,
                             "cublas_wheel": importlib.metadata.version("nvidia-cublas-cu12"),
                             "lt_link_flags": B._ltflags(), "capabilities": dict(ext.capabilities())}
        chosen = selected_source(state)
        if chosen is None:
            result["status"] = "CLOSE_SELECTED_LT_UNSUPPORTED_NO_SUBSTITUTE"; save(); return
        code, layout, key, source = chosen
        result["selected_lt"] = {"code": code, "layout": layout, "key": key,
                                 "cfg": list(B._ext().lt_cfg(M, N, K, 0, code)),
                                 "source_shape": list(source.shape), "source_stride": list(source.stride()),
                                 "source_pointer": source.data_ptr(), "extension_sha256": digest(B._ext().__file__)}
        rp = state.get("rq", {}).get((LAYER, M))
        result["incumbent_rq_pick"] = None if rp is None else {k: rp[k] for k in ("i", "KS", "n")}
        result["ordinary_pointer_properties"] = dict(ext.pointer_properties(source))
        save()
        if result["ordinary_pointer_properties"]["compression"] == 1:
            result["status"] = "CLOSE_NATIVE_ORDINARY_ALREADY_GENERIC"; save(); return
        snapshot = source.view(torch.uint8).cpu().clone()
        raw_snapshot = weights[LAYER].view(torch.uint8).cpu().clone()
        result["source_bytes"] = {"bytes": source.numel() * source.element_size(),
                                  "sha256": hashlib.sha256(snapshot.numpy().tobytes()).hexdigest()}
        stream = torch.cuda.current_stream().cuda_stream
        # No persistent dtype-view local beyond construction; every alias is case-owned.
        owners.append(ext.WeightClone(source.view(torch.uint8), stream, True))
        owners.append(ext.WeightClone(source.view(torch.uint8), stream, False))
        result["allocations"] = []
        for index in range(len(owners)):
            owner = owners[index]
            assert torch.equal(owner.tensor.cpu(), snapshot)
            properties = dict(ext.pointer_properties(owner.tensor))
            assert properties["compression"] == owner.info()["actual_compression"]
            result["allocations"].append({"info": dict(owner.info()), "properties": properties})
        owner = None
        cases = [Case(B, state, "ordinary_lt", code), Case(B, state, "generic_lt", code, key, owners[0]),
                 Case(B, state, "none_lt", code, key, owners[1]), Case(B, state, "incumbent", code)]
        for case in cases[1:3]:
            for name in state:
                if name != key:
                    assert case.state[name] is state[name], "non-weight state changed"
        result["eager_validation"] = {c.label: validate(B, reference, c, weights[LAYER]) for c in cases}
        x = reference.generate_input(M, LAYER, 8851)["x"]
        for case in cases:
            case.capture(x)
        result["timing"] = measure(cases); save()
        result["actual_timed_graph_checks"] = graph_checks(reference, cases, x, weights[LAYER])
        result["fresh_output_checks"] = fresh_output_checks(reference, cases, x, weights[LAYER])
        result["profiles"] = {c.label: profile(c, args.json.with_suffix("." + c.label + ".trace.json")) for c in cases}
        identities = [profile_identity(result["profiles"][label]) for label in LABELS[:3]]
        assert identities[0] == identities[1] == identities[2], "native kernel/launch identity changed"
        assert list(B._ext().lt_cfg(M, N, K, 0, code)) == result["selected_lt"]["cfg"]
        assert torch.equal(source.view(torch.uint8).cpu(), snapshot)
        assert torch.equal(weights[LAYER].view(torch.uint8).cpu(), raw_snapshot), "original native weight mutated"
        for index in range(len(owners)):
            assert torch.equal(owners[index].tensor.cpu(), snapshot)
        if rp is not None:
            assert not int(rp["cnt"].abs().sum()), "incumbent RQ counters not reset"
            result["incumbent_RQ_counters"] = "ZERO"
        result["post_timing_storage"] = "EXACT_ALL_ALLOCATIONS"
        result["status"] = "GAIN_GATE_PENDING_CLEANUP" if result["timing"]["provisional_gain_gate"] else "NO_GAIN_PENDING_CLEANUP"
        save()
    except Exception as error:
        failed = True
        result["status"] = "INVALID_RUNTIME_OR_VALIDATION"; result["error"] = repr(error); save()
        raise
    finally:
        if owners or cases:
            try:
                torch.cuda.synchronize()
                result["cleanup"]["counts_live"] = [dict(o.counts()) for o in owners]
                for case in cases:
                    case.drop_graph()
                for case in cases:
                    case.drop_aliases()
                cases.clear(); case = None
                gc.collect()
                result["cleanup"]["counts_after_alias_drop"] = [dict(o.counts()) for o in owners]
                for index in range(len(owners)):
                    owners[index].close()
                    owners[index].close()  # No-op close after release must be idempotent.
                result["cleanup"]["counts_after_close"] = [dict(o.counts()) for o in owners]
                assert all(c["Region_owners"] == 0 and not c["tensor_defined"]
                           for c in result["cleanup"]["counts_after_close"])
                owners.clear()
                result["cleanup"]["status"] = "CHECKED_CLOSE_PASS"
                if not failed:
                    result["status"] = ("LOCAL_WIN_REVIEW_ONLY" if result.get("timing", {}).get("provisional_gain_gate")
                                        else "CLOSE_NO_NATIVE_ALLOCATION_WIN")
                save()
            except Exception as error:
                result["cleanup"].update({"status": "FAILED", "error": repr(error)})
                result["status"] = "INVALID_CLEANUP_FAILURE"; save()
                raise


if __name__ == "__main__":
    main()
