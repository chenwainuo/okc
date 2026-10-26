"""Two profiled fixed-Lt graph nodes; traffic diagnostic only, no prepare/autotune/timing loop."""
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
ROOT = HERE.parents[2]
NATIVE = HERE.parent / "hw_native"
NATIVE_SHA = "0a4db150debff8dfd09d13d33fe97896a27ba39a272e1b7f00b71f7a803c8047"
CFG = [66, 18, 1, 0, 0, 4, 35, 0, 2]
LAYOUT = 2
WT_SHA = "09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120"
EXPECTED_KERNEL = "nvjet_tst_128x64_64x8_1x1_v_bz_coopB_NNN"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(HERE / n) == h for n, h in manifest["sha256"].items())
    assert sha(NATIVE / "lab.py") == NATIVE_SHA
    L = load(NATIVE / "lab.py", "hw_counter_native_frozen")
    ext, helper_identity = L.load_fixed_helper(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    assert sha(L.PARENT) == L.PARENT_SHA
    B = load(L.PARENT, "hw_counter_parent_frozen")
    sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
    import reference
    result = {"status": "STARTED", "purpose": "COUNTERS_ONLY_NO_SPEED_CLAIM",
              "parent_sha256": L.PARENT_SHA, "native_lab_sha256": NATIVE_SHA,
              "fixed_helper": helper_identity, "population": "UNCHANGED_cuMemcpyDtoDAsync_then_stream_sync",
              "profile_order": ["compression_generic", "compression_none"], "checks": [], "cleanup": {}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.json.write_text(json.dumps(result, indent=2) + "\n")
    cases, owners = [], []
    try:
        assert torch.__version__ == "2.8.0+cu128"
        assert importlib.metadata.version("nvidia-cublas-cu12") == "12.8.4.1"
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        raw = weights["unpooler"]
        wt = raw.t().contiguous()
        snapshot = wt.view(torch.uint8).cpu().clone()
        raw_snapshot = raw.view(torch.uint8).cpu().clone()
        assert snapshot.numel() == 67108864
        assert hashlib.sha256(snapshot.numpy().tobytes()).hexdigest() == WT_SHA, "different native-v1 prepared Wt"
        lt = B._ext()
        # Public AlgoCheck/config admission only. No B.prepare/_tune/_ltsearch and no algo grid timed.
        code = int(lt.lt_add(128, 8192, 4096, 0, LAYOUT, CFG))
        assert code >= 0 and code // 100000 == LAYOUT, "exact native-v1 Lt config unsupported"
        assert list(lt.lt_cfg(128, 8192, 4096, 0, code)) == CFG
        state = {"lt": {("unpooler", 128): code}, "unpooler_raw": raw, "unpooler_t": wt}
        assert L.selected_source(state)[2] == "unpooler_t"
        result["fixed_lt"] = {"layout": LAYOUT, "cfg": CFG, "process_local_code": code,
                              "extension_sha256": sha(lt.__file__), "link_flags": B._ltflags(),
                              "cublas_wheel": importlib.metadata.version("nvidia-cublas-cu12"),
                              "Wt_bytes": snapshot.numel(), "Wt_sha256": WT_SHA,
                              "expected_kernel": EXPECTED_KERNEL, "expected_grid": [1, 128, 1],
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
            prop = dict(ext.pointer_properties(owner.tensor))
            assert prop["compression"] == owner.info()["actual_compression"]
            result["allocations"].append({"info": dict(owner.info()), "properties": prop})
        assert [a["info"]["actual_compression"] for a in result["allocations"]] == [1, 0]
        owner = None
        cases = [L.Case(B, state, "generic_lt", code, "unpooler_t", owners[0]),
                 L.Case(B, state, "none_lt", code, "unpooler_t", owners[1])]
        result["eager_validation"] = {c.label: L.validate(B, reference, c, raw) for c in cases}
        data = reference.generate_input(128, "unpooler", 8851)  # Same native-v1 X.
        x = data["x"]
        x_original = x.clone()
        for case in cases:
            case.capture(x)
        result["preprofile_actual_graph_checks"] = L.graph_checks(reference, cases, x, raw)
        result["fresh_output_checks"] = L.fresh_output_checks(reference, cases, x, raw)
        flush = torch.empty(64 << 20, device="cuda", dtype=torch.float32)
        sink = torch.empty((), device="cuda", dtype=torch.float32)
        for label, case in zip(result["profile_order"], cases):
            case.out.fill_(float("nan"))
            torch.sum(flush, 0, out=sink)
            torch.cuda.synchronize()
            print("COUNTER_PROFILE_BEGIN " + label, flush=True)
            torch.cuda.nvtx.range_push(label)
            try:
                case.graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            # Check precisely the replay observed by NCU before any later replay can repair output.
            assert torch.equal(x, x_original) and bool(torch.isfinite(case.out).all())
            ok, message = reference.check_implementation(data, case.out, {"unpooler": raw})
            assert ok, message
            result["checks"].append({"case": case.label, "stage": "immediately_after_profiled_graph",
                                     "oracle": message, "NaN_poison_overwritten": True, "input_immutable": True})
            print("COUNTER_PROFILE_END " + label, flush=True); save()
        result["postprofile_actual_graph_checks"] = L.graph_checks(reference, cases, x, raw)
        assert list(lt.lt_cfg(128, 8192, 4096, 0, code)) == CFG
        assert torch.equal(wt.view(torch.uint8).cpu(), snapshot) and torch.equal(raw.view(torch.uint8).cpu(), raw_snapshot)
        for owner in owners:
            assert torch.equal(owner.tensor.cpu(), snapshot)
        owner = None
        result["immutable_Wt_raw_X"] = "EXACT"
        result["status"] = "COUNTERS_CAPTURE_COMPLETE_PENDING_CLEANUP_REPORT_REVIEW"; save()
    except Exception as error:
        result["status"] = "INVALID_COUNTER_DRIVER"; result["error"] = repr(error); save()
        raise
    finally:
        if cases or owners:
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
                for owner in owners:
                    owner.close(); owner.close()
                result["cleanup"]["counts_after_close"] = [dict(o.counts()) for o in owners]
                assert all(not c["tensor_defined"] and c["Region_owners"] == 0
                           for c in result["cleanup"]["counts_after_close"])
                result["cleanup"]["status"] = "CHECKED_CLOSE_PASS"
                if result["status"].startswith("COUNTERS_CAPTURE_COMPLETE"):
                    result["status"] = "DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED"
                save()
            except Exception as error:
                result["status"] = "INVALID_CLEANUP_FAILURE"; result["cleanup"]["error"] = repr(error); save()
                raise
    print("NCU_COUNTER_DRIVER_DONE NO_SPEED_CLAIM", flush=True)


if __name__ == "__main__": main()
