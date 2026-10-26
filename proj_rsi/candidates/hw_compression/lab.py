"""One root-scheduled immutable-weight allocation experiment; unchanged pw36 RQ kernel."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import adapter as A

sys.path.insert(0, str(A.ROOT / "muse_bf16_proj"))
import reference


class Case:
    """Explicit owner of every external weight, state, callable and captured graph."""
    def __init__(self, B, state, layer, label, clone=None):
        self.B, self.layer, self.label, self.clone = B, layer, label, clone
        self.state = state
        if clone is not None:
            self.state = dict(state)
            self.state["rq"] = dict(state["rq"])
            p = dict(state["rq"][(layer, 128)])
            p["lay"] = clone.tensor
            self.state["rq"][(layer, 128)] = p
        self.hold = None
        self.graph = None
        self.out = None

    def call(self, x, unused):
        self.hold = self.B.kernel_fn({"x": x, "config": self.config}, self.state)
        return self.hold

    def validator(self, x, dst):
        dst.copy_(self.call(x, None))  # Untimed complete-output oracle only.
        return 0

    @property
    def config(self):
        return {"layer": self.layer, "m": 128,
                "n": 4096 if self.layer == "pooler" else 8192,
                "k": 8192 if self.layer == "pooler" else 4096}

    def capture(self, x):
        self.call(x, None); self.call(x, None); torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.call(x, None)
        self.out = self.hold  # Exact timed graph's real output; no timed copy.

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize()
            self.graph.reset()  # Destroy graph exec BEFORE dropping external weight references.
            self.graph = None
        self.out = self.hold = None
        self.state = None
        gc.collect()
        if self.clone is not None:
            self.clone.close()  # C++ rejects any remaining Tensor/Storage alias.
            self.clone = None


def validate(B, case, w):
    gen = torch.Generator(device="cuda").manual_seed(8407)
    ok, message = B._rqval(case.validator, w, 128, case.config["n"], gen, 4e-3)
    assert ok, message
    checks = [{"all_K_onehot": "EXACT", "K": w.shape[1], "FP64_oracle": message}]
    for dist in ("normal", "outliers"):
        data = reference.generate_input(128, case.layer, 8501, dist=dist)
        before = data["x"].clone()
        dst = case.call(data["x"], None)
        ok, message = reference.check_implementation(data, dst, {case.layer: w})
        assert ok and torch.equal(before, data["x"]), message
        checks.append({"dist": dist, "oracle": message, "input_immutable": True})
    return checks


def timed_graph_mutations(cases, x, w):
    layer = cases[0].layer
    original = x.clone()
    probes = torch.zeros_like(x)
    rows = torch.arange(128, device="cuda")
    cols = (rows * 37 + 19) % x.shape[1]
    probes[rows, cols] = ((2.0 ** ((rows % 5) + 3)) * (2 * (rows % 2) - 1)).to(torch.bfloat16)
    mutations = [("original", original), ("zero", torch.zeros_like(x)),
                 ("fresh_normal", reference.generate_input(128, layer, 8609)["x"]),
                 ("fresh_outliers", reference.generate_input(128, layer, 8611, dist="outliers")["x"]),
                 ("all_probes", probes)]
    checks = []
    for name, fresh in mutations:
        x.copy_(fresh)
        for c in cases:
            before = x.clone()
            c.out.fill_(float("nan")); c.graph.replay(); torch.cuda.synchronize()
            assert torch.equal(before, x) and torch.isfinite(c.out).all(), (c.label, name)
            if name == "zero":
                assert torch.count_nonzero(c.out) == 0
                message = "all output elements exactly zero"
            else:
                ok, message = reference.check_implementation({"x": x, "config": c.config}, c.out, {layer: w})
                assert ok, (c.label, name, message)
            checks.append({"allocation": c.label, "input": name, "oracle": message,
                           "NaN_poison_overwritten": True, "input_immutable": True})
    x.copy_(original)
    for c in cases:
        c.graph.replay()
    torch.cuda.synchronize()
    return checks


def profile(c):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        c.graph.replay(); torch.cuda.synchronize()
    return [{"name": e.name(), "duration_ns": e.duration_ns()}
            for e in p.profiler.kineto_results.events() if e.device_type() == torch.autograd.DeviceType.CUDA]


def measure(B, cases, x):
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.float32)
    sink = torch.empty((), device="cuda", dtype=torch.float32)
    for c in cases:
        c.capture(x)
    def time_graph(c):
        samples = []
        for _ in range(21):
            torch.sum(flush, 0, out=sink)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); c.graph.replay(); end.record(); torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        return statistics.median(samples), samples
    for c in cases:
        for _ in range(5):
            torch.sum(flush, 0, out=sink); c.graph.replay()
    torch.cuda.synchronize()
    ordinary, generic, none = cases
    before, before_samples = time_graph(ordinary)
    generic_f1, generic_samples = time_graph(generic)
    none_f1, none_samples = time_graph(none)
    after, after_samples = time_graph(ordinary)
    starts = [torch.cuda.Event(enable_timing=True) for _ in cases]
    ends = [torch.cuda.Event(enable_timing=True) for _ in cases]
    pairs = []
    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1)]
    for j in range(21):
        times = [None] * 3
        for i in orders[j % len(orders)]:
            torch.sum(flush, 0, out=sink)
            starts[i].record(); cases[i].graph.replay(); ends[i].record()
        torch.cuda.synchronize()
        for i in range(3):
            times[i] = starts[i].elapsed_time(ends[i]) * 1000
        pairs.append({"ordinary_us": times[0], "generic_us": times[1], "vmm_none_us": times[2],
                      "ordinary_gain_us": times[0] - times[1], "vmm_none_gain_us": times[2] - times[1]})
    gains = [r["ordinary_gain_us"] for r in pairs]
    control_gains = [r["vmm_none_gain_us"] for r in pairs]
    baseline = min(before, after)
    result = {"ordinary_before_us": before, "ordinary_after_us": after,
              "conservative_ordinary_us": baseline, "generic_F1_us": generic_f1,
              "vmm_none_F1_us": none_f1, "F1_gain_us": baseline - generic_f1,
              "control_F1_gain_us": none_f1 - generic_f1,
              "F1_samples": {"ordinary_before": before_samples, "generic": generic_samples,
                             "vmm_none": none_samples, "ordinary_after": after_samples},
              "paired_median_gain_us": statistics.median(gains),
              "paired_median_control_gain_us": statistics.median(control_gains),
              "paired_wins": sum(g > 0 for g in gains), "pairs": pairs}
    result["survives"] = (result["F1_gain_us"] >= .5 and result["paired_median_gain_us"] >= .5
                           and result["control_F1_gain_us"] >= .3
                           and result["paired_median_control_gain_us"] >= .3)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()
    manifest = json.loads((A.HERE / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((A.HERE / name).read_bytes()).hexdigest() == digest, name + " drift"
    ext = A.extension()  # Build before importing pw36 so its prepare budget clock is unchanged.
    B = A.baseline_module()
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    state = B.prepare(weights)  # Actual frozen selected dispatch; no forced rq or rebake.
    torch.cuda.synchronize()
    result = {"baseline_sha256": A.FROZEN, "policy": "immutable_selected_RQ_weight_VMM_GENERIC",
              "capabilities": dict(ext.capabilities()), "allocator_ABI": dict(ext.abi()),
              "allocator_backend": torch.cuda.memory.get_allocator_backend(),
              "torch_version": torch.__version__, "torch_CUDA": torch.version.cuda,
              "targets": [], "CPU_preflight": "PASS"}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    save()
    if not result["capabilities"]["vmm_supported"] or not result["capabilities"]["generic_compression_supported"]:
        result["status"] = "CLOSE_UNSUPPORTED_FALLBACK_UNCHANGED"; save(); return
    for layer in ("pooler", "unpooler"):
        p = state.get("rq", {}).get((layer, 128))
        if p is None:
            result["targets"].append({"layer": layer, "status": "SKIP_ACTUAL_PREPARED_DISPATCH_IS_NOT_RQ"})
            save(); continue
        row = {"layer": layer, "variant_index": p["i"], "KS": p["KS"], "n": p["n"],
               "ordinary_pointer_properties": dict(ext.pointer_properties(p["lay"])),
               "fallback": "original frozen pw36 kernel_fn state retained"}
        result["targets"].append(row); save()
        if row["ordinary_pointer_properties"]["compression"] == 1:
            row["status"] = "CLOSE_ORDINARY_ALLOCATION_ALREADY_GENERIC"
            save(); continue
        original_bytes = p["lay"].cpu().clone()
        original_digest = hashlib.sha256(original_bytes.numpy().tobytes()).hexdigest()
        clones, cases = [], []
        try:
            stream = torch.cuda.current_stream().cuda_stream
            try:
                clones.append(ext.WeightClone(p["lay"], stream, True))
                clones.append(ext.WeightClone(p["lay"], stream, False))
            except RuntimeError as error:
                row["status"] = "CLOSE_ALLOCATION_NOT_GRANTED_OR_API_FAILED"
                row["allocation_error"] = str(error)
                result["status"] = row["status"]
                save(); return  # No retry or silent ordinary fallback is measured as compression.
            for c in clones:
                assert torch.equal(c.tensor.cpu(), original_bytes), "full packed-weight D2D/D2H bit roundtrip"
                props = dict(ext.pointer_properties(c.tensor))
                assert props["compression"] == c.info()["actual_compression"]
            row["allocations"] = {"generic": dict(clones[0].info()), "vmm_none": dict(clones[1].info())}
            row["full_byte_roundtrip"] = {"status": "EXACT", "sha256": original_digest, "bytes": p["lay"].numel()}
            cases = [Case(B, state, layer, "ordinary"), Case(B, state, layer, "generic", clones[0]),
                     Case(B, state, layer, "vmm_none", clones[1])]
            # State/key and module identity proof: only selected p['lay'] differs.
            for c in cases[1:]:
                cp = c.state["rq"][(layer, 128)]
                assert set(cp) == set(p)
                for key in p:
                    if key != "lay":
                        assert cp[key] is p[key] or (not torch.is_tensor(p[key]) and cp[key] == p[key]), key
            row["validation"] = {c.label: validate(B, c, weights[layer]) for c in cases}
            x = reference.generate_input(128, layer, 8707)["x"]
            row["timing"] = measure(B, cases, x)
            row["actual_timed_graph_mutations"] = timed_graph_mutations(cases, x, weights[layer])
            row["profile"] = {c.label: profile(c) for c in cases}
            names = [[e["name"] for e in row["profile"][c.label]] for c in cases]
            assert names[0] == names[1] == names[2] and len(names[0]) == 1 and "rq<" in names[0][0], names
            assert torch.equal(p["lay"].cpu(), original_bytes), "ordinary weight mutated"
            for c in clones:
                assert torch.equal(c.tensor.cpu(), original_bytes), "external weight mutated"
            assert not int(p["cnt"].abs().sum()), "RQ counters not reset"
            row["post_timing_weight_immutability"] = "EXACT_ALL_THREE"
            row["status"] = "LOCAL_WIN_REVIEW_ONLY" if row["timing"]["survives"] else "CLOSE_NO_ALLOCATION_WIN"
            print("HWCOMP " + json.dumps(row), flush=True)
            save()
        except Exception as error:
            row["status"] = "INVALID_RUNTIME_OR_VALIDATION_FAILURE"; row["error"] = str(error)
            save(); raise
        finally:
            # Captured graph execs first; then state/tensor references; then VMM unmap/release.
            for c in cases:
                if c.graph is not None:
                    torch.cuda.synchronize(); c.graph.reset(); c.graph = None
                c.out = c.hold = None; c.state = None
            if "cp" in locals():
                del cp
            gc.collect()
            try:
                for c in cases:
                    c.close()
                for c in clones:
                    c.close()
            except Exception as error:
                row["status"] = result["status"] = "INVALID_CLEANUP_FAILURE"
                row["cleanup_error"] = str(error)
                save(); raise
            row["cleanup"] = "GRAPHS_DESTROYED_THEN_TENSORS_DROPPED_THEN_VMM_RELEASED"
            save()
    result["status"] = ("LOCAL_WIN_REVIEW_ONLY" if any(r["status"] == "LOCAL_WIN_REVIEW_ONLY" for r in result["targets"])
                        else "CLOSE_NO_WIN_OR_NO_SELECTED_RQ_TARGET")
    save()
    print("HWCOMP_DONE " + result["status"], flush=True)


if __name__ == "__main__":
    main()
