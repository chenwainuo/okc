"""Root-launched, exclusive bounded trial. No submission or automatic tuning.

python -u proj_rsi/candidates/rtk_early/lab.py --reps 40 --stamp-reps 12 --json logs/rtk-early01.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from variant import PLANS, SHIPPED_SHA256, load

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference

TERMS = ("start", "consumer_ready", "first_full", "last_full", "mma_done", "zero_ready", "reds_issued")


def graph_of(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def event_us(graph, flush, sink):
    torch.sum(flush, 0, out=sink)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000


def counter_check(mod, st, v, m):
    cfg = mod.RTV[v]
    kr, _, toks = cfg[:3]
    G, tiles = 8192 // kr, 4 * ((m + toks - 1) // toks)
    slot = 64 + sorted(mod.RTV).index(v)
    counts = st["rw_cnt"][slot * 128:(slot + 1) * 128].cpu()
    assert bool((counts[1::2] == 0).all()), f"poll/sentinel counters changed v{v}"
    assert bool((counts[0:tiles * 2:2] % G == 0).all()), f"incomplete zero group v{v}"
    assert bool((counts[tiles * 2:] == 0).all()), f"unused counters changed v{v}"
    return {"slot": slot, "G": G, "tiles": tiles,
            "counts": counts[:tiles * 2:2].tolist(), "complete_groups": True}


def validate(mod, st, v, m, w):
    n = w.shape[0]
    plan = (-7, v, False, None, n, None, None)
    fn = lambda xx, oo: mod._rscall(st, plan, xx, oo)
    gen = torch.Generator(device="cuda").manual_seed(4242)
    rexact = st["rs_hi"].float() + st["rs_lo"].float()
    ok, msg = mod._validate(fn, w, m, n, True, gen, rexact, {128: 121, 256: 250}[m])
    assert ok, f"m{m} v{v}: {msg}"
    checks = {"all8192_onehot_exact_hi_plus_lo": True, "shipped_validate": msg, "fresh_graph": []}
    for mm in (m, {128: 121, 256: 250}[m]):
        for dist in ("normal", "outliers"):
            data = reference.generate_input(mm, "router", 909 + mm, dist=dist)
            out = torch.full((mm, n), float("nan"), dtype=torch.float32, device="cuda")
            graph = graph_of(lambda: fn(data["x"], out))
            # Fresh activations, persistent counters, same captured pointer; no stale-data shortcut.
            for seed in (1901, 1902, 1903):
                fresh = reference.generate_input(mm, "router", seed + mm, dist=dist)
                data["x"].copy_(fresh["x"])
                graph.replay()
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, out, {"router": w})
                assert ok, f"m{mm} v{v} {dist}: {msg}"
                ref = reference.ref_kernel(data, {"router": w}).double()
                err = float((out.double() - ref).norm() / ref.norm())
                row = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
                assert err <= mod.TOL["f32"] and row <= 5 * mod.TOL["f32"], (v, mm, dist, err, row)
                checks["fresh_graph"].append({"m": mm, "dist": dist, "seed": seed,
                                              "relative_l2": err, "max_row": row, "task_check": msg})
            # Do not require unused counters zero at odd->even shape transition; same bucket tile count.
            checks["counters"] = counter_check(mod, st, v, mm)
    print(f"VALID m={m} v={v} all8192 exact; normal/outlier/fresh graph/zero groups PASS", flush=True)
    return checks


def stamps(mod, st, v, x, out, flush, sink, reps):
    sv = v + 20000
    dbg = st["rw_cnt"][16384:].view(torch.int64).view(-1, 8)
    medians, maxima = [], []
    for _ in range(reps):
        torch.sum(flush, 0, out=sink)
        dbg.zero_()
        torch.cuda.synchronize()
        rc = mod._rwext().rwrun(sv, x, st["rs_hi"], st["rs_lo"], out, st["rw_cnt"])
        assert rc == 0, f"stamp v{sv} rc={rc}"
        torch.cuda.synchronize()
        active = dbg[dbg[:, 0] > 0].clone()
        assert active.shape[0] == 4 * (8192 // mod.RTV[v][0]) * ((x.shape[0] + 127) // 128)
        rel = (active[:, :7] - active[:, 0].min()).double() / 1000
        medians.append(rel.median(dim=0).values)
        maxima.append(rel.max(dim=0).values)
    md = torch.stack(medians).median(dim=0).values.tolist()
    mx = torch.stack(maxima).median(dim=0).values.tolist()
    return {"median_cta_us": dict(zip(TERMS, md)), "max_cta_us": dict(zip(TERMS, mx)),
            "counter_check": counter_check(mod, st, sv, x.shape[0]),
            "ts1_is_consumer_setup_not_producer_issue": True}


def check_timed_graph(mod, st, v, graph, x, out, w):
    """Validate the actual measured graph AFTER replays; replace X in place, poison previous output."""
    checks = []
    m = x.shape[0]
    for dist in ("normal", "outliers"):
        for seed in (7601, 7602):
            fresh = reference.generate_input(m, "router", seed + m, dist=dist)
            x.copy_(fresh["x"])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            data = {"x": x, "config": fresh["config"]}
            ok, msg = reference.check_implementation(data, out, {"router": w})
            assert ok, f"timed graph after replays m{m} v{v}: {msg}"
            ref = reference.ref_kernel(data, {"router": w}).double()
            err = float((out.double() - ref).norm() / ref.norm())
            row = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            assert err <= mod.TOL["f32"] and row <= 5 * mod.TOL["f32"]
            probes = reference.probe_rows(x)
            if len(probes):
                cols = x[probes].abs().argmax(dim=1)
                scale = x[probes, cols].float()
                exact = (st["rs_hi"].float() + st["rs_lo"].float())[:, cols].t() * scale[:, None]
                assert torch.equal(out[probes], exact), f"timed graph probe not exact m{m} v{v}"
            checks.append({"dist": dist, "seed": seed, "relative_l2": err,
                           "max_row": row, "probe_exact_hi_plus_lo": True, "task_check": msg})
    return {"changed_X_in_place": True, "output_poisoned_before_replay": True,
            "checks": checks, "counters": counter_check(mod, st, v, m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--stamp-reps", type=int, default=12)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    result = {"state": "STARTED", "parent_sha256": SHIPPED_SHA256,
              "device": torch.cuda.get_device_name(), "torch": torch.__version__, "shapes": {}}
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["elapsed_s"] = time.monotonic() - started
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        mod = load()
        ext = mod._rwext()
        result["build_s"] = time.monotonic() - started
        weights = reference.generate_weights(5151)
        w = weights["router"].contiguous()
        del weights
        hi = w.to(torch.bfloat16)
        st = {"rs_hi": hi, "rs_lo": (w - hi.float()).to(torch.bfloat16).contiguous(),
              "rw_cnt": torch.zeros(32768, dtype=torch.int32, device="cuda")}
        flush = torch.ones(64 << 20, dtype=torch.float32, device="cuda")
        sink = torch.zeros((), dtype=torch.float32, device="cuda")
        save()
        for m, plan in PLANS.items():
            base, cand = plan["baseline"], plan["candidate"]
            x = reference.generate_input(m, "router", 5252 + m)["x"]
            out = torch.empty((m, 256), dtype=torch.float32, device="cuda")
            versions = (*base, cand)
            row = {"baseline_versions": base, "candidate": cand, "validation": {}, "occupancy": {}}
            result["shapes"][m] = row
            for v in versions:
                occ = int(ext.rwocc(v, m, 256, 8192))
                assert occ > 0, f"v{v} occupancy {occ}"
                assert 4 * (8192 // mod.RTV[v][0]) * ((m + 127) // 128) <= occ * torch.cuda.get_device_properties(0).multi_processor_count
                row["occupancy"][v] = occ
                row["validation"][v] = validate(mod, st, v, m, w)
                save()
            funcs = {}
            graphs = {}
            for v in versions:
                def fn(xx, oo, v=v):
                    rc = ext.rwrun(v, xx, st["rs_hi"], st["rs_lo"], oo, st["rw_cnt"])
                    assert rc == 0, f"v{v} declined rc={rc}"
                    return rc
                funcs[v] = fn
                graphs[v] = graph_of(lambda v=v: funcs[v](x, out))
            pre = {v: mod._time_us(funcs[v], x, out, flush, sink, args.reps) for v in base}
            prep_cand = mod._time_us(funcs[cand], x, out, flush, sink, args.reps)
            post = {v: mod._time_us(funcs[v], x, out, flush, sink, args.reps) for v in reversed(base)}
            f1 = min(*pre.values(), *post.values())
            row["prepare_style_us"] = {"baseline_pre": pre, "candidate": prep_cand,
                                       "baseline_post": post, "baseline_F1": f1, "gain_us": f1 - prep_cand}
            samples = {v: [] for v in versions}
            pair_gains = {v: [] for v in base}
            for rep in range(args.reps):
                for bv in base if rep % 2 == 0 else reversed(base):
                    order = (bv, cand) if rep % 2 == 0 else (cand, bv)
                    pair = {v: event_us(graphs[v], flush, sink) for v in order}
                    for v, us in pair.items():
                        samples[v].append(us)
                    pair_gains[bv].append(pair[bv] - pair[cand])
            med = {v: statistics.median(us) for v, us in samples.items()}
            graph_gain = min(med[v] for v in base) - med[cand]
            row["graph_us"] = {"median": med, "gain_vs_fastest_shipped_us": graph_gain,
                               "paired_gain_median_us": {v: statistics.median(s) for v, s in pair_gains.items()},
                               "samples": samples}
            row["counters_after_timing"] = {v: counter_check(mod, st, v, m) for v in versions}
            row["actual_timed_graph_validation_after_replays"] = {
                v: check_timed_graph(mod, st, v, graphs[v], x, out, w) for v in versions}
            x.copy_(reference.generate_input(m, "router", 5252 + m)["x"])
            row["stamps"] = {v: stamps(mod, st, v, x, out, flush, sink, args.stamp_reps) for v in versions}
            row["passes_gain_bar"] = bool(graph_gain >= 0.5 and f1 - prep_cand >= 0.5)
            print(f"RESULT m={m} graph_gain={graph_gain:+.3f}us prepare_F1_gain={f1-prep_cand:+.3f}us pass={row['passes_gain_bar']}", flush=True)
            save()
        result["state"] = "PASS_REVIEW_REQUIRED" if any(r["passes_gain_bar"] for r in result["shapes"].values()) else "CLOSED_GAIN_BAR_FAILED"
        save()
    except Exception as exc:
        result["state"] = "FAILED"
        result["error"] = repr(exc)
        save()
        raise
    print(json.dumps({"state": result["state"], "elapsed_s": result["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
