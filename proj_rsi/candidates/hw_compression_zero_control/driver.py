"""Known-zero BF16 Wt allocation control; two fixed-Lt counter nodes, no benchmark."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REAL = HERE.parent / "hw_compression_counters"
REAL_MANIFEST_SHA = "e4f66b6d321a08c8c28eec6c2cb171ee1d731e4f499a2c2f9b2a8a7bc4562596"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_zero(out):
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (128, 8192)
    assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out)) == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert all(sha(HERE / n) == h for n, h in manifest["sha256"].items())
    assert sha(REAL / "manifest.json") == REAL_MANIFEST_SHA
    real_manifest = json.loads((REAL / "manifest.json").read_text())
    assert all(sha(REAL / n) == h for n, h in real_manifest["sha256"].items())
    D = load(REAL / "driver.py", "hw_zero_real_counter_source")
    assert sha(D.NATIVE / "lab.py") == D.NATIVE_SHA
    L = load(D.NATIVE / "lab.py", "hw_zero_native_frozen")
    ext, helper = L.load_fixed_helper(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    assert sha(L.PARENT) == L.PARENT_SHA
    B = load(L.PARENT, "hw_zero_parent_frozen")
    sys.path.insert(0, str(D.ROOT / "muse_bf16_proj"))
    import reference
    result = {"status": "STARTED", "purpose": "KNOWN_ZERO_PATH_OBSERVER_CONTROL_NO_SPEED_CLAIM",
              "control_payload": "KNOWN_ZERO_BF16_WT", "normal_weight_rerun": False,
              "parent_sha256": L.PARENT_SHA, "real_counter_manifest_sha256": REAL_MANIFEST_SHA,
              "fixed_helper": helper, "population": "UNCHANGED_cuMemcpyDtoDAsync_then_stream_sync",
              "profile_order": ["compression_generic", "compression_none"], "checks": [], "cleanup": {},
              "all_K_onehot": "NOT_RUN_ZERO_WEIGHT_UNINFORMATIVE; prior real-native4096basis is separate evidence"}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.json.write_text(json.dumps(result, indent=2) + "\n")
    cases, owners = [], []
    try:
        assert torch.__version__ == "2.8.0+cu128" and importlib.metadata.version("nvidia-cublas-cu12") == "12.8.4.1"
        # Source initialization only. Foreign destinations use the exact reviewed D2D helper unchanged.
        raw = torch.zeros((8192, 4096), dtype=torch.bfloat16, device="cuda")
        wt = raw.t().contiguous()
        snapshot = wt.view(torch.uint8).cpu().clone()
        raw_snapshot = raw.view(torch.uint8).cpu().clone()
        assert snapshot.numel() == 67108864 and not int(torch.count_nonzero(snapshot))
        assert not int(torch.count_nonzero(raw_snapshot))
        lt = B._ext()
        code = int(lt.lt_add(128, 8192, 4096, 0, D.LAYOUT, D.CFG))
        assert code >= 0 and code // 100000 == 2
        assert list(lt.lt_cfg(128, 8192, 4096, 0, code)) == D.CFG
        state = {"lt": {("unpooler", 128): code}, "unpooler_raw": raw, "unpooler_t": wt}
        assert L.selected_source(state)[2] == "unpooler_t"
        result["fixed_lt"] = {"layout": 2, "cfg": D.CFG, "process_local_code": code,
                              "extension_sha256": sha(lt.__file__), "link_flags": B._ltflags(),
                              "cublas_wheel": importlib.metadata.version("nvidia-cublas-cu12"),
                              "Wt_bytes": snapshot.numel(), "Wt_sha256": hashlib.sha256(snapshot.numpy().tobytes()).hexdigest(),
                              "expected_kernel": D.EXPECTED_KERNEL, "expected_grid": [1, 128, 1],
                              "expected_block": [384, 1, 1], "expected_registers": 168,
                              "expected_shared_bytes": 213268}
        result["capabilities"] = dict(ext.capabilities()); save()
        assert result["capabilities"]["generic_compression_supported"] == 1
        stream = torch.cuda.current_stream().cuda_stream
        owners.append(ext.WeightClone(wt.view(torch.uint8), stream, True))
        owners.append(ext.WeightClone(wt.view(torch.uint8), stream, False))
        result["allocations"] = []
        for owner in owners:
            assert torch.equal(owner.tensor.cpu(), snapshot)
            properties = dict(ext.pointer_properties(owner.tensor))
            assert properties["compression"] == owner.info()["actual_compression"]
            result["allocations"].append({"info": dict(owner.info()), "properties": properties})
        assert [a["info"]["actual_compression"] for a in result["allocations"]] == [1, 0]
        owner = None
        cases = [L.Case(B, state, "generic_lt", code, "unpooler_t", owners[0]),
                 L.Case(B, state, "none_lt", code, "unpooler_t", owners[1])]
        data = reference.generate_input(128, "unpooler", 8851)
        x = data["x"]; original_x = x.clone()
        fresh_x = reference.generate_input(128, "unpooler", 8853, dist="outliers")["x"]
        assert int(torch.count_nonzero(x)) > 0 and int(torch.count_nonzero(fresh_x)) > 0
        assert not torch.equal(original_x, fresh_x)
        for case in cases:
            before = x.clone(); out = case.call(x)
            exact_zero(out); assert torch.equal(x, before); out = None
            case.capture(x)
        def mutations(tag):
            rows = []
            for label, payload in (("original_nonzero_X8851", original_x), ("fresh_nonzero_outliers_X8853", fresh_x)):
                x.copy_(payload)
                for case in cases:
                    before = x.clone(); case.out.fill_(float("nan")); case.graph.replay(); torch.cuda.synchronize()
                    exact_zero(case.out); assert torch.equal(x, before)
                    rows.append({"case": case.label, "stage": tag, "input": label,
                                 "NaN_poison_overwritten": True, "complete_output": "EXACT_ZERO", "X_immutable": True})
            x.copy_(original_x)
            return rows
        result["preprofile_actual_graph_checks"] = mutations("preprofile")
        # Additional capture uses the second nonzero X and a different actual output allocation.
        result["fresh_output_checks"] = []
        x.copy_(fresh_x)
        for case in cases:
            saved = case.out; fresh_graph = torch.cuda.CUDAGraph()
            fresh_out = None
            try:
                with torch.cuda.graph(fresh_graph):
                    fresh_out = case.call(x)
                assert fresh_out.data_ptr() != saved.data_ptr()
                fresh_out.fill_(float("nan")); fresh_graph.replay(); torch.cuda.synchronize()
                exact_zero(fresh_out); assert torch.equal(x, fresh_x)
                result["fresh_output_checks"].append({"case": case.label, "input": "fresh_nonzero_X8853",
                                                       "new_output_allocation": True, "NaN_overwritten": True,
                                                       "complete_output": "EXACT_ZERO", "X_immutable": True})
            finally:
                torch.cuda.synchronize(); fresh_graph.reset(); fresh_graph = None; fresh_out = None
                case.hold = saved; saved = None
        x.copy_(original_x)
        flush = torch.empty(64 << 20, device="cuda", dtype=torch.float32)
        sink = torch.empty((), device="cuda", dtype=torch.float32)
        for label, case in zip(result["profile_order"], cases):
            case.out.fill_(float("nan")); torch.sum(flush, 0, out=sink); torch.cuda.synchronize()
            print("ZERO_PROFILE_BEGIN " + label, flush=True)
            torch.cuda.nvtx.range_push(label)
            try:
                case.graph.replay(); torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            # Inspect this profiled output before any subsequent replay can repair it.
            exact_zero(case.out); assert torch.equal(x, original_x)
            result["checks"].append({"case": case.label, "stage": "immediately_after_profiled_graph",
                                     "NaN_poison_overwritten": True, "complete_output": "EXACT_ZERO", "input_immutable": True})
            print("ZERO_PROFILE_END " + label, flush=True); save()
        result["postprofile_actual_graph_checks"] = mutations("postprofile")
        assert list(lt.lt_cfg(128, 8192, 4096, 0, code)) == D.CFG
        assert torch.equal(raw.view(torch.uint8).cpu(), raw_snapshot) and torch.equal(wt.view(torch.uint8).cpu(), snapshot)
        for owner in owners:
            assert torch.equal(owner.tensor.cpu(), snapshot)
        owner = None
        result["immutable_zero_Wt_raw_X"] = "EXACT"
        result["status"] = "COUNTERS_CAPTURE_COMPLETE_PENDING_CLEANUP_REPORT_REVIEW"; save()
    except Exception as error:
        result["status"] = "INVALID_ZERO_CONTROL"; result["error"] = repr(error); save()
        raise
    finally:
        if cases or owners:
            try:
                torch.cuda.synchronize()
                result["cleanup"]["counts_live"] = [dict(o.counts()) for o in owners]
                for case in cases: case.drop_graph()
                for case in cases: case.drop_aliases()
                cases.clear(); case = None
                gc.collect()
                result["cleanup"]["counts_after_alias_drop"] = [dict(o.counts()) for o in owners]
                for owner in owners: owner.close(); owner.close()
                result["cleanup"]["counts_after_close"] = [dict(o.counts()) for o in owners]
                assert all(not c["tensor_defined"] and c["Region_owners"] == 0 for c in result["cleanup"]["counts_after_close"])
                result["cleanup"]["status"] = "CHECKED_CLOSE_PASS"
                if result["status"].startswith("COUNTERS_CAPTURE_COMPLETE"):
                    result["status"] = "DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED"
                save()
            except Exception as error:
                result["status"] = "INVALID_CLEANUP_FAILURE"; result["cleanup"]["error"] = repr(error); save()
                raise
    print("ZERO_CONTROL_DONE NO_LATENCY_OR_WIRE_RATIO_CLAIM", flush=True)


if __name__ == "__main__": main()
