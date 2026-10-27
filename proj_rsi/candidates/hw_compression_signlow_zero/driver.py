"""One six-node reader-calibrated raw/signlow/known-zero allocation matrix."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
SIGNLOW4 = HERE.parent / "hw_compression_signlow"
SIGNLOW4_MANIFEST_SHA = "11fff851a2ed48c2b2a209047244e10383c9292877c545af653629da02f69db1"
ROOT = HERE.parents[2]
READER = HERE.parent / "hw_compression_byteplanes"
READER_MANIFEST_SHA = "8e3d6016f388d23616fec4fe4055ceefee99b2cba550ef3f437e3f481b52321e"
PATTERNS = HERE.parent / "hw_compression_patterns"
PATTERNS_MANIFEST_SHA = "780fabc881ef06f5f3559742a4ce3bdefed3f93423bc99cb628e0c581e230f9d"
NATIVE = HERE.parent / "hw_native" / "lab.py"
NATIVE_SHA = "0a4db150debff8dfd09d13d33fe97896a27ba39a272e1b7f00b71f7a803c8047"
REFERENCE = ROOT / "muse_bf16_proj/reference.py"
WT_SHA = "09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120"
NBYTES = 67108864
BASE = 0x3C00
LABELS = ("raw_generic", "raw_none", "signlow_generic", "signlow_none", "zero_generic", "zero_none")
KERNEL = "okc_hwc_copy64m"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reader():
    for directory, expected in [(READER, READER_MANIFEST_SHA), (PATTERNS, PATTERNS_MANIFEST_SHA), (SIGNLOW4, SIGNLOW4_MANIFEST_SHA)]:
        assert digest(directory / "manifest.json") == expected, "frozen dependency manifest drift"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected_file in manifest["sha256"].items():
            path = (directory / name).resolve()
            assert path.parent == directory.resolve() and digest(path) == expected_file, "frozen dependency drift"
    return load(READER / "driver.py", "hw_signlow_frozen_copy_reader")


def signlow_encode(rawbytes):
    assert rawbytes.dtype == torch.uint8 and rawbytes.is_contiguous() and rawbytes.numel() % 2 == 0
    u = rawbytes.view(torch.int16).int() & 0xFFFF
    e = (((u & 0x7FFF) ^ BASE) << 1) | (u >> 15)
    return e.to(torch.int16).view(torch.uint8).reshape(-1)


def signlow_decode(encodedbytes):
    assert encodedbytes.dtype == torch.uint8 and encodedbytes.is_contiguous() and encodedbytes.numel() % 2 == 0
    e = encodedbytes.view(torch.int16).int() & 0xFFFF
    u = ((e >> 1) ^ BASE) | ((e & 1) << 15)
    return u.to(torch.int16).view(torch.uint8).reshape(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert all(digest(HERE / n) == h for n, h in manifest["sha256"].items())
    assert digest(NATIVE) == NATIVE_SHA and digest(REFERENCE) == manifest["dependencies"]["reference_sha256"]
    D = load_reader()
    L = load(NATIVE, "hw_signlow_native_binding")
    helper, helper_identity = L.load_fixed_helper(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    ext = D.copy_extension()
    sys.path.insert(0, str(ROOT / "muse_bf16_proj")); import reference
    result = {"status": "STARTED", "purpose": "SIX_NODE_READER_CALIBRATED_REPRESENTATION_ALLOCATION_COUNTERS_NO_SPEED",
              "fixed_helper": helper_identity, "profile_order": list(LABELS),
              "reader_manifest_sha256": READER_MANIFEST_SHA, "CPU_patterns_manifest_sha256": PATTERNS_MANIFEST_SHA,
              "superseded_signlow4_manifest_sha256": SIGNLOW4_MANIFEST_SHA,
              "knownzero": "64MiB_ZERO_BYTES_NOT_ENCODED_BF16_WEIGHT_ZERO",
              "format": {"name": "SIGNLOW_XOR16", "BASE": BASE, "arithmetic_precision_change": False},
              "population": "UNCHANGED_cuMemcpyDtoDAsync_then_stream_sync", "checks": [], "cleanup": {}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.json.write_text(json.dumps(result, indent=2) + "\n")
    owners, cases = [], []
    try:
        assert torch.__version__ == "2.8.0+cu128" and torch.version.cuda == "12.8"
        info = dict(ext.info()); result["copy_resources"] = info
        proof = json.loads((READER / "build-proof.json").read_text())
        assert info["localSizeBytes"] == 0 and info["sharedSizeBytes"] == 0
        assert info["numRegs"] == proof["registers"] and info["occupancy_blocks_per_SM"] >= 1
        assert info["maxThreadsPerBlock"] == 256 and info["binaryVersion"] == 90
        resource_text = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', ext.__file__],
                                       check=True, capture_output=True, text=True).stdout
        resource_match = re.search(r'Function okc_hwc_copy64m:\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', resource_text)
        assert resource_match and list(map(int, resource_match.groups())) == [32, 0, 0, 0]
        result['jit_resources'] = {'registers': 32, 'stack': 0, 'shared': 0, 'local': 0,
                                   'cuobjdump_sha256': hashlib.sha256(resource_text.encode()).hexdigest()}
        result["copy_identity"] = {"kernel": KERNEL, "bytes": NBYTES, "grid": [128, 1, 1], "block": [256, 1, 1],
                                   "source_sha256": digest(READER / "copy_kernel.cu"),
                                   "bindings_sha256": digest(READER / "bindings.cpp"), "jit_sha256": digest(ext.__file__)}
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        raw = weights["unpooler"]; wt = raw.t().contiguous(); weights = None
        native = wt.view(torch.uint8).reshape(-1)
        raw_snapshot = raw.view(torch.uint8).cpu().clone()
        native_snapshot = native.cpu().clone()
        assert native.numel() == NBYTES and hashlib.sha256(native_snapshot.numpy().tobytes()).hexdigest() == WT_SHA
        signlow = signlow_encode(native); signlow_snapshot = signlow.cpu().clone()
        decoded_gpu = signlow_decode(signlow)
        assert torch.equal(decoded_gpu.cpu(), native_snapshot), "full GPU inverse differs from original Wt bytes"
        decoded_gpu = None
        assert torch.equal(signlow_decode(signlow_snapshot), native_snapshot)
        assert torch.equal(signlow_encode(native_snapshot), signlow_snapshot)
        zero = torch.zeros((NBYTES,), dtype=torch.uint8, device=native.device)
        zero_snapshot = zero.cpu().clone()
        assert int(torch.count_nonzero(zero_snapshot)) == 0 and zero_snapshot.numel() == NBYTES
        payloads = {"raw": native, "signlow": signlow, "zero": zero}
        snapshots = {"raw": native_snapshot, "signlow": signlow_snapshot, "zero": zero_snapshot}
        poisons = {name: torch.bitwise_not(p) for name, p in payloads.items()}
        assert bool(torch.all(poisons["zero"] == 255))
        result["payloads"] = {name: {"bytes": NBYTES, "sha256": hashlib.sha256(s.numpy().tobytes()).hexdigest()}
                               for name, s in snapshots.items()}
        result["payloads"]["signlow"]["inverse_to_raw_exact"] = True
        result["payloads"]["signlow"]["GPU_full_payload_inverse_to_raw_exact"] = True
        result["payloads"]["zero"]["all_source_bytes_zero"] = True
        result["payloads"]["zero"]["all_poison_bytes_255"] = True
        result["capabilities"] = dict(helper.capabilities()); save()
        assert result["capabilities"]["generic_compression_supported"] == 1
        stream = torch.cuda.current_stream().cuda_stream
        result["allocations"] = []
        for label in LABELS:
            name, flag = label.split('_'); compressed = flag == "generic"
            owners.append(helper.WeightClone(payloads[name], stream, compressed))
            owner = owners[-1]
            properties = dict(helper.pointer_properties(owner.tensor)); allocation = dict(owner.info())
            assert allocation["actual_compression"] == int(compressed) and properties["compression"] == int(compressed)
            D.exact_bytes(owner.tensor, snapshots[name])
            result["allocations"].append({"label": label, "info": allocation, "properties": properties})
            cases.append(D.Case(label, owner, ext, helper, snapshots[name], poisons[name], stream))
        owner = None
        # Check all source and main output ranges, rather than pointer uniqueness alone.
        ranges = [(c.src.data_ptr(), NBYTES) for c in cases] + [(c.out.data_ptr(), NBYTES) for c in cases]
        assert all(a + n <= b or b + m <= a for i, (a, n) in enumerate(ranges) for b, m in ranges[i + 1:])
        result["allocation_spans_disjoint"] = True
        result["outputs"] = {c.label: {"ptr": c.out.data_ptr(), "properties": c.output_properties} for c in cases}
        result["preprofile_checks"] = []
        result["fresh_output_checks"] = []
        for case in cases:
            case.eager(); case.capture()
            result["preprofile_checks"].append({"case": case.label, "eager_and_actual_graph_bytes_exact": True,
                                                 "complement_poison_overwritten": True, "input_immutable": True})
            result["fresh_output_checks"].append({"case": case.label, **case.fresh_capture(helper)})
        all_outputs = [c.out.data_ptr() for c in cases] + [out.data_ptr() for c in cases for graph, out in c.fresh]
        assert len(set(all_outputs)) == 12
        all_spans = [(c.src.data_ptr(), NBYTES) for c in cases] + [(ptr, NBYTES) for ptr in all_outputs]
        assert len(all_spans) == 18
        assert all(a + n <= b or b + m <= a for i, (a, n) in enumerate(all_spans) for b, m in all_spans[i + 1:])
        result["all18_source_main_fresh_spans_disjoint"] = True
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        sink = torch.empty((), device="cuda", dtype=torch.float32)
        for case in cases:
            case.out.copy_(case.poison)
            torch.sum(flush, 0, out=sink); torch.cuda.synchronize()
            print("SIGNLOW_ZERO_PROFILE_BEGIN " + case.label, flush=True)
            torch.cuda.nvtx.range_push(case.label)
            try:
                case.graph.replay(); torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            D.exact_bytes(case.out, case.snapshot); D.exact_bytes(case.src, case.snapshot)
            result["checks"].append({"case": case.label, "stage": "IMMEDIATELY_AFTER_SELECTED_ACTUAL_GRAPH",
                                     "full64MiB_bytes_exact": True, "complement_poison_overwritten": True,
                                     "input_immutable": True, "repair_replay": False})
            print("SIGNLOW_ZERO_PROFILE_END " + case.label, flush=True); save()
        # No graph replay follows selected profiling. Only full-byte audits then teardown.
        D.exact_bytes(raw.view(torch.uint8).reshape(-1), raw_snapshot.reshape(-1))
        for name, payload in payloads.items(): D.exact_bytes(payload, snapshots[name])
        for case in cases: D.exact_bytes(case.src, case.snapshot)
        result["postprofile_sources_immutable"] = True
        result["status"] = "COUNTERS_COMPLETE_PENDING_CLEANUP"; save()
    except Exception as error:
        result["status"] = "INVALID_SIGNLOW_ZERO_DIAGNOSTIC"; result["error"] = repr(error); save(); raise
    finally:
        if cases or owners:
            try:
                torch.cuda.synchronize()
                result["cleanup"]["counts_live"] = [dict(o.counts()) for o in owners]
                for case in cases: case.drop_graphs()
                for case in cases: case.drop_aliases()
                cases.clear(); case = None
                gc.collect()
                result["cleanup"]["counts_after_alias_drop"] = [dict(o.counts()) for o in owners]
                for owner in owners: owner.close(); owner.close()
                result["cleanup"]["counts_after_close"] = [dict(o.counts()) for o in owners]
                assert all(not c["tensor_defined"] and c["Region_owners"] == 0 for c in result["cleanup"]["counts_after_close"])
                result["cleanup"]["status"] = "CHECKED_CLOSE_PASS"
                if result["status"] == "COUNTERS_COMPLETE_PENDING_CLEANUP":
                    result["status"] = "DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED"
                save()
            except Exception as error:
                result["status"] = "INVALID_CLEANUP_FAILURE"; result["cleanup"]["error"] = repr(error); save(); raise
    print("SIGNLOW_ZERO_DONE NO_LATENCY_OR_WIRE_RATIO_CLAIM", flush=True)


if __name__ == "__main__": main()
