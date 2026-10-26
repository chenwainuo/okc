"""No CUDA calls or values printed. Records only ownership counts and referrer type names."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import weakref

import torch

HERE = Path(__file__).resolve().parent


def build():
    name = "okc_owner_cpu"
    base = Path(torch.__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="okc-owner-cpu-"))
    output = tmp / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    command = ["g++", "-shared", "-fPIC", "-O2", "-std=c++17", "-DTORCH_EXTENSION_NAME=" + name,
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
               "-I" + str(base / "include"), "-I" + str(base / "include/torch/csrc/api/include"),
               "-I" + sysconfig.get_path("include"), str(HERE / "owner_cpu.cpp"),
               "-L" + str(base / "lib"), "-Wl,-rpath," + str(base / "lib"),
               "-ltorch_python", "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(output)]
    p = subprocess.run(command, capture_output=True, text=True)
    (tmp / "build.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError("CPU compile failed: " + str(tmp / "build.log") + "\n" + p.stderr)
    spec = importlib.util.spec_from_file_location(name, output)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, command, tmp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, required=True); args = ap.parse_args()
    m, command, tmp = build()
    rows = []
    def snap(name, owner, wrapper=None):
        row = {"stage": name, "counts": dict(owner.counts())}
        if wrapper is not None:
            row["Python_referrer_TYPES"] = sorted(type(r).__name__ for r in gc.get_referrers(wrapper))
        rows.append(row)
    def close(name, owner, method="v1_close", expect=None):
        try:
            getattr(owner, method)(); result = "CLOSED"
        except RuntimeError as error:
            result = str(error).splitlines()[0]
        rows.append({"stage": name, "result": result, "counts": dict(owner.counts())})
        if expect is not None:
            assert (result == "CLOSED") == expect, name
    o = m.Owner(); snap("fresh_no_getter", o); close("fresh_v1_close", o)
    o = m.Owner(); t = o.tensor; wt = weakref.ref(t)
    snap("getter_live", o, t); close("live_same_impl_negative_control", o)
    del t; gc.collect()
    snap("getter_deleted", o)
    rows.append({"stage": "getter_weakref", "wrapper_alive": wt() is not None})
    close("getter_deleted_v1_close", o)
    o = m.Owner(); alias = o.tensor.view(8, 8); snap("view_only_live", o, alias)
    close("live_view_negative_control", o)
    del alias; gc.collect(); snap("view_deleted", o); close("view_deleted_v1_close", o)
    o = m.Owner()
    state = {"rq": {("pooler", 128): {"lay": o.tensor}}}
    p = state["rq"][("pooler", 128)]
    callback = lambda: state["rq"][("pooler", 128)]["lay"]
    snap("state_cp_callback_live", o, p["lay"])
    state = None; callback = None; p = None; gc.collect()
    snap("state_cp_callback_dropped", o); close("state_cleanup_v1_close", o)
    close("fix_cached_wrapper_only", o, "storage_owner_close", expect=True)
    for name, make in [("same_impl", lambda t: t), ("view", lambda t: t.view(8, 8)),
                       ("dtype_view", lambda t: t.view(torch.bfloat16)),
                       ("Storage", lambda t: t.untyped_storage()),
                       ("detached_view", lambda t: t.detach()),
                       ("state_dict", lambda t: {"rq": {"lay": t}}),
                       ("callback_default", lambda t: lambda weight=t: weight)]:
        o = m.Owner(); alias = make(o.tensor); snap("fix_live_" + name, o, alias)
        released_before = o.counts()["released_total"]
        close("fix_reject_live_" + name, o, "storage_owner_close", expect=False)
        assert o.counts()["Region_owners"] == 2 and o.counts()["released_total"] == released_before
        del alias; gc.collect(); snap("fix_deleted_" + name, o)
        close("fix_close_after_" + name, o, "storage_owner_close", expect=True)
        assert o.counts()["released_total"] == released_before + 1
        close("fix_idempotent_after_" + name, o, "storage_owner_close", expect=True)
        assert o.counts()["released_total"] == released_before + 1
    o = m.Owner()
    t = o.tensor; wt = weakref.ref(t); del t; gc.collect()
    close("fix_getter_deleted_wrapper", o, "storage_owner_close", expect=True)
    assert wt() is None
    class GraphPythonReferenceModel:
        # Python reference lifetime only: this does not simulate CUDA graph/native execution.
        def __init__(self, weight): self.callback = lambda t=weight: t
        def reset(self): self.callback = None
        def replay(self): return self.callback()
    o = m.Owner(); graph = GraphPythonReferenceModel(o.tensor); callback = graph.replay
    close("fix_graph_python_callback_live", o, "storage_owner_close", expect=False)
    graph.reset()  # The bound callback still keeps the model object alive, but reset released W.
    gc.collect(); close("fix_graph_model_reset", o, "storage_owner_close", expect=True)
    del callback, graph
    def stale_locals_reproducer():
        owner = m.Owner()
        cp = {"lay": owner.tensor}
        if "cp" in locals():  # EXACT frozen v1 pattern: materializes stale f_locals mapping.
            del cp
        gc.collect()
        before = dict(owner.counts())
        try:
            owner.storage_owner_close()
            return {"unexpected": "closed", "before": before}
        except RuntimeError as error:
            return {"result": str(error).splitlines()[0], "before": before,
                    "after": dict(owner.counts()), "mechanism": "locals() cached cp dictionary survives del fast-local"}
    stale = stale_locals_reproducer()
    assert "rejected" in stale["result"] and stale["after"]["Region_owners"] == 2
    rows.append({"stage": "exact_v1_locals_then_del_reproduces_real_alias", **stale})
    def no_locals_control():
        owner = m.Owner()
        cp = {"lay": owner.tensor}
        cp = None  # No locals() snapshot; initialize/clear this local unconditionally.
        gc.collect()
        owner.storage_owner_close()
        return dict(owner.counts())
    rows.append({"stage": "replace_locals_pattern_with_unconditional_None", "counts": no_locals_control()})
    result = {"state": "CPU_ONLY_OWNERSHIP_FORENSICS", "torch": torch.__version__,
              "CUDA_calls": 0, "source_sha256": hashlib.sha256((HERE / "owner_cpu.cpp").read_bytes()).hexdigest(),
              "build_directory": str(tmp), "build_command": command, "observations": rows}
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
