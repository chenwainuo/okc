"""CPU-only routing/bit/identity checks. No helper import, build, CUDA API or GPU."""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import weakref

import torch

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("hw_native_cpu_lab", HERE / "lab.py")
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    assert hashlib.sha256(L.PARENT.read_bytes()).hexdigest() == L.PARENT_SHA
    ast.parse((HERE / "lab.py").read_text())
    layout_checks = []
    for layout in range(4):
        raw = torch.empty((L.N, L.K), dtype=torch.bfloat16, device="meta")
        transposed = torch.empty((L.K, L.N), dtype=torch.bfloat16, device="meta")
        state = {"lt": {(L.LAYER, L.M): layout * 100000 + 17},
                 L.LAYER + "_raw": raw, L.LAYER + "_t": transposed}
        code, got_layout, key, source = L.selected_source(state)
        assert got_layout == layout and code == layout * 100000 + 17
        assert source is (transposed if layout & 2 else raw)
        layout_checks.append({"layout": layout, "key": key, "shape": list(source.shape),
                              "stride": list(source.stride())})
    assert L.selected_source({"lt": {(L.LAYER, L.M): None}}) is None
    assert L.selected_source({"lt": {(L.LAYER, L.M): -1}}) is None
    # Test exactly the dtype/storage operations used around the unchanged CUDA helper.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(128, 512)
    w = bits.view(torch.bfloat16)
    bit_checks = []
    class FakeOwner:
        def __init__(self, source):
            self.tensor = source.clone()
        def info(self):
            return {"ptr": self.tensor.data_ptr()}
    sentinel = object()
    for name, source in [("raw", w), ("prepared_transpose", w.t().contiguous())]:
        owner = FakeOwner(source.view(torch.uint8))
        state = {"lt": {(L.LAYER, L.M): 200017}, "unpooler_t": source,
                 "rq": sentinel, "unpooler_raw": sentinel}
        case = L.Case(None, state, "generic_lt", 200017, "unpooler_t", owner)
        view = case.state["unpooler_t"]
        assert torch.equal(view.view(torch.int16), source.view(torch.int16))
        assert view.shape == source.shape and view.stride() == source.stride()
        assert view.data_ptr() == owner.tensor.data_ptr()
        assert case.state["rq"] is sentinel and state["unpooler_t"] is source
        ref = weakref.ref(view)
        del view
        case.drop_aliases(); gc.collect()
        assert ref() is None, "case retained BF16 Python alias"
        bit_checks.append({"source": name, "all_BF16_patterns": 65536,
                           "dtype_reinterpretation": "EXACT", "pointer_alias": True,
                           "case_reference_drop": "PASS_NOT_VMM_OWNERSHIP_PROOF"})
    p = [{"name": "oldLt", "grid": [1, 2, 3], "block": [128, 1, 1],
          "registers": 64, "shared_bytes": 8192, "duration_us": 15}]
    p2 = [dict(p[0], duration_us=2)]
    assert L.profile_identity(p) == L.profile_identity(p2)
    p2[0]["grid"] = [2, 2, 3]
    assert L.profile_identity(p) != L.profile_identity(p2)
    try:
        L.load_fixed_helper(HERE.parent / "hw_compression", "0" * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid v1 helper was not rejected before import")
    directory, helper_hashes = L.verify_helper_binding(L.HELPER_DIR, L.HELPER_MANIFEST_SHA)
    assert helper_hashes["allocator.cpp"] == L.HELPER_ALLOCATOR_SHA
    binding = {"directory": directory.name, "manifest_sha256": L.HELPER_MANIFEST_SHA,
               "allocator_sha256": L.HELPER_ALLOCATOR_SHA,
               "lifetime_gate": "hw-owner-lifetime01_PASS_INDEPENDENT_TERMINAL_GO"}
    files = ["lab.py", "preflight.py", "README.md"]
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    manifest = {"policy": "ONE_UNPOOLER128_NATIVE_LT_ALLOCATION_ONLY", "parent_sha256": L.PARENT_SHA,
                "helper_binding": binding,
                "GPU_admission": "BOUND_PENDING_FINAL_CRITIC_AND_ROOT_HANDOFF", "sha256": hashes}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {"status": "CPU_ROUTING_BITS_CASE_ALIAS_DROP_PASS_GPU_UNRUN",
              "torch_cpu": torch.__version__, "no_CUDA_or_helper_API": True,
              "layout_checks": layout_checks, "bit_checks": bit_checks,
              "profile_identity_mismatch_guard": "PASS", "reject_invalid_v1_helper": "PASS",
              "helper_runtime_ownership": "HW_OWNER_LIFETIME01_PASS_AND_INDEPENDENT_TERMINAL_GO",
              "helper_binding": binding,
              "sha256": hashes}
    (HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
