"""ONE root-owned exclusive trial; actual frozen pw36 prepare/path, complete checks on both.

python -u proj_rsi/candidates/router_small/lab.py --reps 40 --json logs/router-small01.json
No flags for geometry/codec sweeps; source is frozen before a named launch.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import prototype

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference


def graph_of(fn, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(x)
        fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x)
    return graph, out


def event_us(graph, flush, sink):
    torch.sum(flush, 0, out=sink)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000


def check_output(data, out, w, hi_lo):
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, out, {"router": w})
    assert ok, msg
    ref = reference.ref_kernel(data, {"router": w}).double()
    rel = float((out.double() - ref).norm() / ref.norm().clamp(min=1e-30))
    row = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
    assert bool(torch.isfinite(out).all()) and rel <= 2e-5 and row <= 1e-4
    probes = reference.probe_rows(data["x"])
    if len(probes):
        cols = data["x"][probes].abs().argmax(dim=1)
        scale = data["x"][probes, cols].float()
        exact = hi_lo[:, cols].t() * scale[:, None]
        assert torch.equal(out[probes], exact), "Probe rows not exact hi+lo"
    return {"relative_l2": rel, "max_row": row, "task_check": msg, "probes_exact": True}


def eager_checks(fn, w, hi_lo):
    checks = []
    for m in (5, 7, 8):
        for dist in ("normal", "outliers", "zero"):
            data = reference.generate_input(m, "router", 9017 + m, dist="normal" if dist == "zero" else dist)
            if dist == "zero":
                data["x"].zero_()
            out = fn(data["x"])
            checked = check_output(data, out, w, hi_lo)
            if dist == "zero":
                assert bool((out == 0).all()), "zero activation output not zero"
            checks.append({"m": m, "dist": dist, **checked})
    # Literally all8192 basis columns x8, not sampled probes; every expert's hi+lo must be exact.
    x = torch.zeros((8, 8192), dtype=torch.bfloat16, device="cuda")
    for start in range(0, 8192, 8):
        x.zero_()
        x[torch.arange(8, device=x.device), start + torch.arange(8, device=x.device)] = 8
        got = fn(x)
        assert torch.equal(got, hi_lo[:, start:start+8].t() * 8), f"onehot K{start} mismatch"
    return {"normal_outlier_zero_nearby": checks, "all8192_onehot_columns_exact": True}


def replay_checks(graph, out, x, original, w, hi_lo):
    checks = []
    for tag in ("original", "zero", "fresh_normal", "fresh_outlier", "original_again"):
        if tag.startswith("original"):
            x.copy_(original["x"])
            cfg = original["config"]
        else:
            fresh = reference.generate_input(8, "router", 6262, dist="outliers" if tag == "fresh_outlier" else "normal")
            x.copy_(fresh["x"])
            cfg = fresh["config"]
            if tag == "zero":
                x.zero_()
        out.fill_(float("nan"))
        for _ in range(3):
            graph.replay()
        checked = check_output({"x": x, "config": cfg}, out, w, hi_lo)
        if tag == "zero":
            assert bool((out == 0).all())
        checks.append({"case": tag, **checked})
    return {"actual_measured_graph": True, "after_timing": True, "changed_X": True,
            "NaN_poison_before_replay": True, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    result = {"state": "STARTED", "parent_sha256": prototype.PARENT_SHA256,
              "kernel_sha256": hashlib.sha256(prototype.HERE.joinpath("kernel.cu").read_bytes()).hexdigest(),
              "target_m": 8, "gain_bar_us": 0.5}
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["elapsed_s"] = time.monotonic() - started
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        result["device"] = torch.cuda.get_device_name()
        result["torch"] = torch.__version__
        save()
        # Mandatory actual Torch host/device compile; no attempt to time if it fails or spills.
        ext = prototype.extension()
        result["prototype_build_s"] = time.monotonic() - started
        weights = reference.generate_weights(5151)
        w = weights["router"]
        new = prototype.prepare(w)
        result["packing"] = new["packing"]
        result["resources"] = new["resources"]
        print("PROTO resources " + json.dumps(result["resources"]), flush=True)
        print("PROTO packing " + json.dumps(result["packing"]), flush=True)
        base = prototype.load_parent()
        old = base.prepare(weights)  # ACTUAL frozen prepare/bake, no disabled flags/shortcuts.
        result["baseline_prepare_s"] = time.monotonic() - started - result["prototype_build_s"]
        p = old["sel"].get(("router", 8))
        route = "sel"
        if p is None:
            p = old["gv"].get(("router", 8))
            route = "gv"
        # Scalar-only metadata: NEVER stringify the plan tuple containing weight tensors.
        result["baseline_pick"] = {"route": route if p is not None else "v12",
                                   "instance": int(p[0]) if p is not None else None,
                                   "kernel_name": base._name(p[0]) if p is not None else "v12",
                                   "N": int(p[4]) if p is not None else 256}
        print("BASELINE pick " + json.dumps(result["baseline_pick"]), flush=True)
        hi = w.to(torch.bfloat16)
        hi_lo = hi.float() + (w - hi.float()).to(torch.bfloat16).float()

        def oldfn(x):
            return base.kernel_fn({"x": x, "config": {"m": x.shape[0], "n": 256, "k": 8192, "layer": "router"}}, old)

        fns = {"pw36": oldfn, "diagonal": lambda x: prototype.call(new, x)}
        result["eager_validation"] = {}
        for name, fn in fns.items():
            result["eager_validation"][name] = eager_checks(fn, w, hi_lo)
            print("VALID " + name + " all8192 exact + originalFP32 normal/outlier/zero m5/7/8", flush=True)
            save()
        original = reference.generate_input(8, "router", 7777)
        x = original["x"].clone()
        graphs = {name: graph_of(fn, x) for name, fn in fns.items()}
        flush = torch.ones(64 << 20, dtype=torch.float32, device="cuda")
        sink = torch.zeros((), dtype=torch.float32, device="cuda")
        tmp = torch.empty((8, 256), dtype=torch.float32, device="cuda")
        # Use the shipped prepare timer, wrapping actual kernel_fn/call, preserving their graph outputs/allocations.
        pre = base._time_us(lambda xx, oo: oldfn(xx), x, tmp, flush, sink, args.reps)
        cand = base._time_us(lambda xx, oo: fns["diagonal"](xx), x, tmp, flush, sink, args.reps)
        post = base._time_us(lambda xx, oo: oldfn(xx), x, tmp, flush, sink, args.reps)
        result["prepare_style_us"] = {"baseline_pre": pre, "candidate": cand, "baseline_post": post,
                                      "baseline_F1": min(pre, post), "gain_us": min(pre, post) - cand}
        samples = {name: [] for name in fns}
        delta = []
        for rep in range(args.reps):
            order = ("pw36", "diagonal") if rep % 2 == 0 else ("diagonal", "pw36")
            pair = {name: event_us(graphs[name][0], flush, sink) for name in order}
            for name, us in pair.items():
                samples[name].append(us)
            delta.append(pair["pw36"] - pair["diagonal"])
        med = {name: statistics.median(us) for name, us in samples.items()}
        gain = statistics.median(delta)
        result["paired_graph_us"] = {"median": med, "median_paired_gain_us": gain,
                                     "median_difference_gain_us": med["pw36"] - med["diagonal"], "samples": samples}
        result["timed_graph_validation_after_replays"] = {
            name: replay_checks(g, out, x, original, w, hi_lo) for name, (g, out) in graphs.items()}
        # Read scalar resources again: trial compiler must meet the SAME local gate.
        assert int(ext.router_small_regs()) <= 64 and int(ext.router_small_local()) == 0
        passed = gain >= 0.5 and min(pre, post) - cand >= 0.5
        result["state"] = "PASS_REVIEW_REQUIRED" if passed else "CLOSED_GAIN_BAR_FAILED"
        save()
        print("RESULT " + json.dumps({"state": result["state"], "prepare_gain_us": min(pre, post) - cand,
                                      "paired_graph_gain_us": gain, "elapsed_s": result["elapsed_s"]}), flush=True)
    except Exception as exc:
        result["state"] = "FAILED"
        result["error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
