"""Root-only non-timing lifetime gate. No F1/paired/performance rerun or new kernel."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import weakref

import torch
import adapter as Fixed

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "hw_compression"
FIXED_SHA = "7f2dea1ecdcbff20efd1daad1a4250f775bbf9cfb8b9a76a16f257c1268140f8"
V1_LAB_SHA = "a17a4ae1699a1c46284f19a7378861660db6e09ae0f67c0c3d9eb9ea1c2693af"


def original_lab():
    assert hashlib.sha256((V1 / "lab.py").read_bytes()).hexdigest() == V1_LAB_SHA
    spec = importlib.util.spec_from_file_location("hw_lifetime_original_adapter", V1 / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    previous = sys.modules.get("adapter")
    sys.modules["adapter"] = adapter
    try:
        spec = importlib.util.spec_from_file_location("hw_lifetime_original_lab", V1 / "lab.py")
        lab = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = lab
        spec.loader.exec_module(lab)
    finally:
        if previous is None:
            sys.modules.pop("adapter", None)
        else:
            sys.modules["adapter"] = previous
    return lab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    assert hashlib.sha256((HERE / "allocator.cpp").read_bytes()).hexdigest() == FIXED_SHA
    original_manifest = json.loads((V1 / "manifest.json").read_text())
    assert all(hashlib.sha256((V1 / n).read_bytes()).hexdigest() == h for n, h in original_manifest["sha256"].items())
    ext = Fixed.extension()  # Before frozen baseline module's prepare clock.
    lab = original_lab()
    B = lab.A.baseline_module()
    weights = lab.reference.generate_weights(lab.reference.WEIGHTS_SEED)
    state = B.prepare(weights)
    torch.cuda.synchronize()
    result = {"state": "LIFETIME_ONLY_RUNNING", "timing": "NONE",
              "fixed_allocator_sha256": FIXED_SHA, "original_Case_profile_lab_sha256": V1_LAB_SHA,
              "torch": torch.__version__, "traces": [], "oracles": []}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.json.write_text(json.dumps(result, indent=2) + "\n")
    def trace(stage, clones, refs=None):
        # Snapshot C++ counts BEFORE accessing weak Python wrappers, avoiding getter observer effects.
        row = {"stage": stage, "counts": [dict(c.counts()) for c in clones]}
        if refs is not None:
            kinds = []
            for wr in refs:
                t = wr()
                kinds.append(None if t is None else sorted(type(r).__name__ for r in gc.get_referrers(t)))
                del t
            row["Python_referrer_TYPES"] = kinds
        result["traces"].append(row); save()
        print("OWNER_TRACE " + json.dumps(row), flush=True)
    p = state.get("rq", {}).get(("pooler", 128))
    if p is None:
        result["state"] = "CLOSE_NO_ACTUAL_SELECTED_POOLER_RQ"; save(); return
    result["actual_pick"] = {"layer": "pooler", "m": 128, "i": p["i"], "KS": p["KS"], "n": p["n"]}
    source_bytes = p["lay"].cpu().clone()
    stream = torch.cuda.current_stream().cuda_stream
    clones, cases, refs = [], [], []
    cp = None  # Unconditionally clear; do not materialize locals() and leave a stale cp alias.
    control = None
    held = dtype_view = raw_storage = None
    try:
        # One separate lifetime negative control using the same immutable packed W; no graph/performance trial.
        control = ext.WeightClone(p["lay"], stream, True)
        held = control.tensor
        trace("negative_control_real_Tensor_live", [control])
        try:
            control.close()
        except RuntimeError as error:
            assert "storage owners remain" in str(error)
        else:
            raise AssertionError("live Tensor alias was released")
        assert control.counts()["Region_owners"] == 2
        assert torch.equal(held.cpu(), source_bytes), "live alias invalidated by rejected close"
        dtype_view = held.view(torch.bfloat16); raw_storage = held.untyped_storage()
        held = None; gc.collect()
        trace("negative_control_dtype_view_Storage_live", [control])
        try:
            control.close()
        except RuntimeError as error:
            assert "storage owners remain" in str(error)
        else:
            raise AssertionError("live view/Storage alias was released")
        assert torch.equal(dtype_view.view(torch.uint8).cpu(), source_bytes)
        dtype_view = raw_storage = None; gc.collect()
        trace("negative_control_all_aliases_dropped", [control])
        control.close(); control.close()
        assert control.counts()["Region_owners"] == 0
        trace("negative_control_release_idempotent", [control])
        control = None

        for compress in (True, False):
            clone = ext.WeightClone(p["lay"], stream, compress)
            clones.append(clone)
            assert clone.info()["actual_compression"] == int(compress)
            t = clone.tensor
            refs.append(weakref.ref(t))
            trace("getter_live_" + str(compress), clones)
            del t; gc.collect()
            trace("getter_dropped_" + str(compress), clones, refs)
            assert torch.equal(clone.tensor.cpu(), source_bytes)
            props = dict(ext.pointer_properties(clone.tensor))
            assert props["compression"] == int(compress)
            gc.collect(); trace("full_byte_copy_pointer_query_" + str(compress), clones, refs)
        cases = [lab.Case(B, state, "pooler", "generic", clones[0]),
                 lab.Case(B, state, "pooler", "vmm_none", clones[1])]
        trace("original_Case_state_created", clones, refs)
        data = lab.reference.generate_input(128, "pooler", 9117)
        x = data["x"]
        for case in cases:
            before = x.clone()
            out = case.call(x, None)
            ok, message = lab.reference.check_implementation(data, out, {"pooler": weights["pooler"]})
            assert ok and torch.equal(before, x), message
            result["oracles"].append({"case": case.label, "stage": "actual_eager", "oracle": message})
            out = None
            trace("actual_call_" + case.label, clones, refs)
            case.capture(x)  # Exact unchanged frozen v1 Case implementation.
            case.graph.replay(); torch.cuda.synchronize()
            trace("capture_replay_" + case.label, clones, refs)
            # Frozen profile executes a replay; discard durations. This driver records no performance samples.
            result.setdefault("kernel_names", {})[case.label] = [e["name"] for e in lab.profile(case)]
            assert len(result["kernel_names"][case.label]) == 1 and "rq<" in result["kernel_names"][case.label][0]
            gc.collect(); trace("original_profile_dropped_" + case.label, clones, refs)
        assert result["kernel_names"]["generic"] == result["kernel_names"]["vmm_none"]
        original_x = x.clone()
        fresh_outliers = lab.reference.generate_input(128, "pooler", 9121, dist="outliers")["x"]
        for label, fresh in (("normal", original_x), ("zero", torch.zeros_like(x)), ("fresh_outliers", fresh_outliers)):
            x.copy_(fresh)
            for case in cases:
                before = x.clone(); case.out.fill_(float("nan")); case.graph.replay(); torch.cuda.synchronize()
                assert torch.equal(before, x) and torch.isfinite(case.out).all()
                if label == "zero":
                    assert torch.count_nonzero(case.out) == 0
                    message = "all output elements exactly zero"
                else:
                    ok, message = lab.reference.check_implementation({"x": x, "config": case.config},
                                                                     case.out, {"pooler": weights["pooler"]})
                    assert ok, message
                result["oracles"].append({"case": case.label, "stage": "actual_graph_" + label,
                                          "oracle": message, "NaN_poison_overwritten": True, "X_immutable": True})
        for clone in clones:
            assert torch.equal(clone.tensor.cpu(), source_bytes)
        cp = cases[-1].state["rq"][("pooler", 128)]
        for case in cases:
            torch.cuda.synchronize(); case.graph.reset(); case.graph = None
        trace("all_graph_execs_reset", clones, refs)
        for case in cases:
            case.out = case.hold = None; case.state = None
        gc.collect(); trace("state_output_dropped_cp_intentionally_live", clones, refs)
        # cp is a real last-case alias; clearing it must not use `if 'cp' in locals(): del cp`.
        cp = None
        gc.collect(); trace("all_state_output_cp_dropped_gc", clones, refs)
        for case in cases:
            case.close()  # Unchanged frozen v1 Case.close, now calling corrected allocator close.
            trace("checked_Case_close_" + case.label, clones, refs)
        for clone in clones:
            clone.close()
            assert clone.counts()["Region_owners"] == 0
        assert not int(p["cnt"].abs().sum())
        result["state"] = "LIFETIME_GATE_PASS_NO_PERFORMANCE_CLAIM"
        save(); print("OWNER_LIFETIME_DONE " + result["state"], flush=True)
    except Exception as error:
        result["state"] = "INVALID_LIFETIME_GATE"; result["error"] = str(error)
        trace("failure_counts", clones, refs)
        raise
    finally:
        # Best-effort drain on failure, while preserving real aliases; never relax the Region guard.
        for case in cases:
            if case.graph is not None:
                torch.cuda.synchronize(); case.graph.reset(); case.graph = None
            case.out = case.hold = None; case.state = None
        cp = held = dtype_view = raw_storage = None
        gc.collect()
        errors = []
        for clone in clones + ([] if control is None else [control]):
            try:
                clone.close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            result["state"] = "INVALID_CLEANUP_FAILURE"; result["cleanup_errors"] = errors
            save()
            raise RuntimeError("lifetime cleanup rejected remaining owners")


if __name__ == "__main__": main()
