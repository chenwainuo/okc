"""Four-node allocation×representation causal diagnostic; full byte copies, no timing."""
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
ROOT = HERE.parents[2]
NATIVE = HERE.parent / "hw_native" / "lab.py"
NATIVE_SHA = "0a4db150debff8dfd09d13d33fe97896a27ba39a272e1b7f00b71f7a803c8047"
REFERENCE = ROOT / "muse_bf16_proj/reference.py"
WT_SHA = "09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120"
NBYTES = 67108864
LABELS = ("raw_generic", "raw_none", "planes_generic", "planes_none")
KERNEL = "okc_hwc_copy64m"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def byteplanes64(b):
    assert b.dtype == torch.uint8 and b.is_contiguous() and b.numel() % 128 == 0
    return b.reshape(-1, 64, 2).transpose(1, 2).contiguous().reshape(-1)


def unplanes64(b):
    assert b.dtype == torch.uint8 and b.is_contiguous() and b.numel() % 128 == 0
    return b.reshape(-1, 2, 64).transpose(1, 2).contiguous().reshape(-1)


def exact_bytes(actual, snapshot):
    assert actual.dtype == torch.uint8 and actual.is_contiguous() and actual.numel() == snapshot.numel()
    assert torch.equal(actual.cpu(), snapshot), "complete byte oracle mismatch"


def copy_extension():
    from torch.utils.cpp_extension import load as jit_load
    return jit_load(name="okc_hwcopy_" + digest(HERE / "copy_kernel.cu")[:12] + digest(HERE / "bindings.cpp")[:8],
                    sources=[str(HERE / "bindings.cpp"), str(HERE / "copy_kernel.cu")],
                    extra_cflags=["-O2", "-std=c++17"],
                    extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"],
                    with_cuda=True, verbose=True)


class Case:
    def __init__(self, label, owner, ext, helper, snapshot, poison, stream):
        self.label, self.owner, self.ext, self.stream = label, owner, ext, stream
        self.src = owner.tensor
        self.snapshot = snapshot.clone()  # Per-case full CPU snapshot, not a GPU alias.
        self.poison = poison
        self.out = torch.empty((NBYTES,), dtype=torch.uint8, device=self.src.device)
        self.graph = None
        self.fresh = []
        self.output_properties = dict(helper.pointer_properties(self.out))
        assert self.out.data_ptr() % 16 == 0 and self.src.data_ptr() % 16 == 0

    def eager(self):
        self.out.copy_(self.poison)
        self.ext.copy64m(self.src, self.out, self.stream)
        torch.cuda.synchronize()
        exact_bytes(self.out, self.snapshot); exact_bytes(self.src, self.snapshot)

    def capture(self):
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.ext.copy64m(self.src, self.out, torch.cuda.current_stream().cuda_stream)
        self.out.copy_(self.poison)
        self.graph.replay(); torch.cuda.synchronize()
        exact_bytes(self.out, self.snapshot); exact_bytes(self.src, self.snapshot)

    def fresh_capture(self, helper):
        out = torch.empty_like(self.out)
        assert out.data_ptr() != self.out.data_ptr()
        graph = torch.cuda.CUDAGraph()
        self.fresh.append((graph, out))  # Retain both, including on failure.
        with torch.cuda.graph(graph):
            self.ext.copy64m(self.src, out, torch.cuda.current_stream().cuda_stream)
        out.copy_(self.poison)
        graph.replay(); torch.cuda.synchronize()
        exact_bytes(out, self.snapshot); exact_bytes(self.src, self.snapshot)
        return {"new_output_ptr": out.data_ptr(), "output_properties": dict(helper.pointer_properties(out)),
                "full64MiB_bytes_exact": True, "complement_poison_overwritten": True}

    def drop_graphs(self):
        if self.graph is not None: self.graph.reset()
        for graph, out in self.fresh: graph.reset()
        self.graph = None; self.fresh.clear()

    def drop_aliases(self):
        self.src = None; self.owner = None; self.out = None
        self.snapshot = None; self.poison = None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert all(digest(HERE / n) == h for n, h in manifest["sha256"].items())
    assert digest(NATIVE) == NATIVE_SHA and digest(REFERENCE) == manifest["dependencies"]["reference_sha256"]
    L = load(NATIVE, "hw_byteplanes_native_binding")
    helper, helper_identity = L.load_fixed_helper(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    ext = copy_extension()
    sys.path.insert(0, str(ROOT / "muse_bf16_proj")); import reference
    result = {"status": "STARTED", "purpose": "FOUR_NODE_REPRESENTATION_ALLOCATION_COUNTERS_NO_SPEED",
              "fixed_helper": helper_identity, "profile_order": list(LABELS),
              "population": "UNCHANGED_cuMemcpyDtoDAsync_then_stream_sync", "checks": [], "cleanup": {}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.json.write_text(json.dumps(result, indent=2) + "\n")
    owners, cases = [], []
    try:
        assert torch.__version__ == "2.8.0+cu128" and torch.version.cuda == "12.8"
        info = dict(ext.info()); result["copy_resources"] = info
        proof = json.loads((HERE / "build-proof.json").read_text())
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
                                   "source_sha256": digest(HERE / "copy_kernel.cu"),
                                   "bindings_sha256": digest(HERE / "bindings.cpp"), "jit_sha256": digest(ext.__file__)}
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        raw = weights["unpooler"]; wt = raw.t().contiguous(); weights = None
        native = wt.view(torch.uint8).reshape(-1)
        raw_snapshot = raw.view(torch.uint8).cpu().clone()
        native_snapshot = native.cpu().clone()
        assert native.numel() == NBYTES and hashlib.sha256(native_snapshot.numpy().tobytes()).hexdigest() == WT_SHA
        planes = byteplanes64(native); plane_snapshot = planes.cpu().clone()
        assert torch.equal(unplanes64(plane_snapshot), native_snapshot)
        assert torch.equal(byteplanes64(native_snapshot), plane_snapshot)
        payloads = {"raw": native, "planes": planes}
        snapshots = {"raw": native_snapshot, "planes": plane_snapshot}
        poisons = {name: torch.bitwise_not(p) for name, p in payloads.items()}
        result["payloads"] = {name: {"bytes": NBYTES, "sha256": hashlib.sha256(s.numpy().tobytes()).hexdigest()}
                               for name, s in snapshots.items()}
        result["payloads"]["planes"]["inverse_to_raw_exact"] = True
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
            exact_bytes(owner.tensor, snapshots[name])
            result["allocations"].append({"label": label, "info": allocation, "properties": properties})
            cases.append(Case(label, owner, ext, helper, snapshots[name], poisons[name], stream))
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
        assert len(set(all_outputs)) == 8
        all_spans = [(c.src.data_ptr(), NBYTES) for c in cases] + [(ptr, NBYTES) for ptr in all_outputs]
        assert all(a + n <= b or b + m <= a for i, (a, n) in enumerate(all_spans) for b, m in all_spans[i + 1:])
        flush = torch.ones(64 << 20, device="cuda", dtype=torch.float32)
        sink = torch.empty((), device="cuda", dtype=torch.float32)
        for case in cases:
            case.out.copy_(case.poison)
            torch.sum(flush, 0, out=sink); torch.cuda.synchronize()
            print("BYTEPLANE_PROFILE_BEGIN " + case.label, flush=True)
            torch.cuda.nvtx.range_push(case.label)
            try:
                case.graph.replay(); torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            exact_bytes(case.out, case.snapshot); exact_bytes(case.src, case.snapshot)
            result["checks"].append({"case": case.label, "stage": "IMMEDIATELY_AFTER_SELECTED_ACTUAL_GRAPH",
                                     "full64MiB_bytes_exact": True, "complement_poison_overwritten": True,
                                     "input_immutable": True, "repair_replay": False})
            print("BYTEPLANE_PROFILE_END " + case.label, flush=True); save()
        # No graph replay follows selected profiling. Only full-byte audits then teardown.
        exact_bytes(raw.view(torch.uint8).reshape(-1), raw_snapshot.reshape(-1))
        for name, payload in payloads.items(): exact_bytes(payload, snapshots[name])
        for case in cases: exact_bytes(case.src, case.snapshot)
        result["postprofile_sources_immutable"] = True
        result["status"] = "COUNTERS_COMPLETE_PENDING_CLEANUP"; save()
    except Exception as error:
        result["status"] = "INVALID_BYTEPLANE_DIAGNOSTIC"; result["error"] = repr(error); save(); raise
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
    print("BYTEPLANE_DONE NO_LATENCY_OR_WIRE_RATIO_CLAIM", flush=True)


if __name__ == "__main__": main()
