"""RS128 lab (H100 box): the m128-plan ladder (S0..S4 + probes) of submission.py's rq kernel, timed side by side with
cuBLAS (torch.mm) on the same shapes.

Usage (inside the container, cwd /okc):
  python proj_rsi/candidates/rs128/lab.py --plan r1 [--build-only] [--reps 40] [--layers pooler] [--ms 128]

Timing: CUPTI kernel durations (kineto), each call after a read pass over a 256 MB buffer (cold L2, the judge's
read flush), variants interleaved round-robin; median and min per variant. Kernel-side only (no replay overhead);
the zero-fill kernel of the full path is reported separately (z=). --graph adds the bake's graph-replay timing
(in-prepare style, includes the ~3.5 us replay floor and the memset node). SM / memory clocks are sampled with
nvidia-smi during the timed section. Every full (PRB 0) variant is validated first: one-hot x8 rows bit-exact against
W over all K columns, random rows (normal and 'outliers') rel L2 vs fp64, and the task's check_implementation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def load_sub(path):
    spec = importlib.util.spec_from_file_location("sub_rs128", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_sub(HERE / "submission.py")
SHAPES = {"pooler": (4096, 8192), "unpooler": (8192, 4096), "o": (8192, 1024), "qkv_gate": (2304, 8192)}


def plan_r1(layers, ms):
    v = []
    for ly in layers:
        for m in ms:
            v.append(("cublas", None, ly, m, 0, 0))
            for nm in ("f12", "f12x", "s0", "s1", "s2", "s3", "s3g1", "s3w0", "s4", "s4g1", "s4h", "ep"):
                v.append((nm, nm, ly, m, S.RQKS[ly], 0))
    return v


def plan_e5(layers, ms):
    """L2-resident period probe: every CTA re-reads row block 0 (walias 1); plus the resident-smem rates (r8 / r9)"""
    v = []
    for ly in layers:
        for m in ms:
            for nm in ("s1", "s2", "s3", "s3g1", "s3w0"):
                v.append((nm + "-L2", nm, ly, m, S.RQKS[ly], 1))
            for nm in ("s3", "r8", "r8g1"):
                v.append((nm, nm, ly, m, S.RQKS[ly], 0))
    return v


def plan_ins(layers, ms):
    v = []
    for ly in layers:
        for m in ms:
            v.append(("s1i", "s1i", ly, m, S.RQKS[ly], 0))
            v.append(("s2i", "s2i", ly, m, S.RQKS[ly], 0))
            v.append(("s3i", "s3i", ly, m, S.RQKS[ly], 0))
            v.append(("s3i-L2", "s3i", ly, m, S.RQKS[ly], 1))
            v.append(("s4i", "s4i", ly, m, S.RQKS[ly], 0))
            v.append(("s3g1i", "s3g1i", ly, m, S.RQKS[ly], 0))
            v.append(("s3g1i-L2", "s3g1i", ly, m, S.RQKS[ly], 1))
    return v


PLANS = {"r1": plan_r1, "e5": plan_e5, "ins": plan_ins}


class Clocks:
    def __init__(self):
        self.samples, self.stop = [], False
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.samples.append((int(o[0]), int(o[1])))
            except Exception:
                return
            time.sleep(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=6)

    def summary(self):
        if not self.samples:
            return "clocks n/a"
        sm = sorted(s[0] for s in self.samples)
        mem = sorted(s[1] for s in self.samples)
        return "SM clock MHz med %d (min %d max %d, n %d), mem %d" % (sm[len(sm) // 2], sm[0], sm[-1], len(sm), mem[len(mem) // 2])


def validate(run, w, lay, m, gen):
    """one-hot x8 rows bit-exact over all K columns; random rows rel L2 (normal, outliers); the task's checker"""
    N, K = w.shape
    dev = w.device
    out = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    wt = w.t()
    bad = 0
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=dev)))
        r = run(x.contiguous(), out)
        if r:
            return False, "launch %d" % r
        got = out[: min(m, K - start)].float()
        want = wt[start:start + got.shape[0]].float() * 8.0
        if not torch.equal(got, want):
            bad += int((got != want).any(1).sum())
    if bad:
        return False, "one-hot: %d inexact rows" % bad
    msg = ["one-hot exact (%d cols)" % K]
    for dist in ("normal", "outliers"):
        data = reference.generate_input(m, lay, 77 + m, dist=dist)
        x = data["x"]
        run(x, out)
        ref = x.double() @ w.double().t()
        err = float((out.double() - ref).norm() / ref.norm())
        row = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        ok, cm = reference.check_implementation(data, out.clone(), {lay: w})
        msg.append("%s relL2 %.2e row %.2e check %s" % (dist, err, row, "ok" if ok else "FAIL " + cm))
        if not ok:
            return False, "; ".join(msg)
    return True, "; ".join(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="r1")
    ap.add_argument("--layers", default="pooler")
    ap.add_argument("--ms", default="128")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ptxv", action="store_true")
    args = ap.parse_args()
    layers = args.layers.split(",")
    ms = [int(a) for a in args.ms.split(",")]
    var = PLANS[args.plan](layers, ms)
    if args.only:
        keys = args.only.split(",")
        var = [x for x in var if x[1] is None or x[0] in keys]
    names = list(dict.fromkeys(x[1] for x in var if x[1]))
    vs = [S.RQV[S.RQI[n]] for n in names]
    t0 = time.time()
    ext = S._rqext(vs, verbose=args.ptxv)
    e = torch.empty(0)
    for i, v in enumerate(vs):
        print("REGS %-6s regs=%d local=%d smem=%d" % (v[0], ext.rqrun(i, e, e, e, e, e, 0, 0, 0, e, e, 1),
                                                     ext.rqrun(i, e, e, e, e, e, 0, 0, 0, e, e, 2), ext.rqrun(i, e, e, e, e, e, 0, 0, 0, e, e, 3)), flush=True)
    print("BUILD %d variants in %.0fs" % (len(vs), time.time() - t0), flush=True)
    if args.build_only:
        return
    dev = "cuda"
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    packs = {}
    for ly in layers:
        w = weights[ly]
        t1 = time.time()
        pk = S._rqpack(w)
        assert pk is not None, "pack not exact"
        lay, tab, esc = pk
        bits = S._rqunpack(lay, tab, esc, *w.shape)
        ok = torch.equal(bits, w.view(torch.int16).int() & 0xffff)
        print("PACK %s %s esc=%d Tb=%#x blocked-unpack %s (%.1fs)" % (ly, tuple(lay.shape), len(esc), tab & 0xffffffff, "EXACT" if ok else "MISMATCH", time.time() - t1), flush=True)
        assert ok
        packs[ly] = pk
    escs = {}
    gen = torch.Generator(device=dev).manual_seed(1281)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    fsink = torch.empty((), dtype=torch.float32, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    prof = torch.zeros(1 << 16, dtype=torch.int64, device=dev)
    calls = []
    for lab, nm, ly, m, KS, wal in var:
        w = weights[ly]
        N, K = w.shape
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
        tag = "%s m=%d %s" % (ly, m, lab)
        if nm is None:
            wt = w.t()
            calls.append((tag, (lambda xx, oo, wt=wt: torch.mm(xx, wt, out=oo)), x, out, "", None))
            continue
        lay, tab, esc = packs[ly]
        if (ly, KS) not in escs:
            escs[(ly, KS)] = S._rqesc(esc, lay.shape[0], lay.shape[1], KS, dev)
        el, eo = escs[(ly, KS)]
        i = names.index(nm)
        v = S.RQV[S.RQI[nm]]
        full = v[4] in (0, 7)
        ins = v[6]

        def run(xx, oo, i=i, lay=lay, el=el, eo=eo, KS=KS, tab=tab, wal=wal, full=full, ins=ins):
            if full:
                oo.zero_()
            return ext.rqrun(i, lay, xx, oo, el, eo, KS, tab, wal, prof if ins else e, sink, 0)
        r = run(x, out)
        torch.cuda.synchronize()
        if r:
            print("SKIP %s launch=%d" % (tag, r), flush=True)
            continue
        info = ""
        if v[4] == 0 and wal == 0 and not args.no_validate:
            t1 = time.time()
            ok, msg = validate(run, w, ly, m, gen)
            print("VALID %s %s %s (%.1fs)" % (tag, "ok" if ok else "INVALID", msg, time.time() - t1), flush=True)
            info = "valid" if ok else "INVALID"
        calls.append((tag, run, x, out, info, v))
    # INS stamps
    for tag, fn, x, out, info, v in calls:
        if v is None or not v[6]:
            continue
        prof.zero_()
        for _ in range(5):
            torch.sum(flush, 0, out=fsink)
            fn(x, out)
        torch.cuda.synchronize()
        ly = tag.split()[0]
        N, K = SHAPES[ly]
        KS = S.RQKS[ly]
        nc = (N // 128) * KS
        pr = prof[: nc * 32].view(nc, 2, 16).double()
        nst = (K // 64) / KS
        for gg in range(2):
            mn = pr[:, gg].mean(0) / nst
            mx = pr[:, gg].max(0).values / nst
            print("INS %s WG%d per-stage clk mean(max): loop %.0f(%.0f) wfull-wait %.0f(%.0f) lds+arrive %.0f(%.0f) decode %.0f(%.0f) "
                  "lds-data-wait %.0f(%.0f) issue %.0f(%.0f) wait_group %.0f(%.0f) xrel %.0f(%.0f) | epilogue %.0f clk" % (
                      tag, gg, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], mn[3], mx[3], mn[4], mx[4], mn[5], mx[5], mn[6], mx[6], mn[7], mx[7], mn[8] * nst), flush=True)
        pw = pr[:, 0, 9:11].mean(0) / nst
        print("INS %s producer empty-wait per stage: W %.0f x %.0f clk" % (tag, pw[0], pw[1]), flush=True)
        gt = prof[: nc * 32].view(nc, 32)[:, 11:16].double()
        t0 = gt[:, 0].min()
        q = lambda c: (float(gt[:, c].sub(t0).min()) / 1e3, float(gt[:, c].sub(t0).median()) / 1e3, float(gt[:, c].sub(t0).max()) / 1e3)
        print("INS %s timeline us (min/med/max from the first CTA entry): entry %.2f/%.2f/%.2f loop-start %.2f/%.2f/%.2f "
              "loop-end %.2f/%.2f/%.2f exit %.2f/%.2f/%.2f W-lane-done %.2f/%.2f/%.2f" % ((tag,) + q(0) + q(1) + q(2) + q(3) + q(4)), flush=True)
    tcalls = [c for c in calls if c[5] is None or not c[5][6]]
    from torch.profiler import profile, ProfilerActivity
    for tag, fn, x, out, _, _ in tcalls:
        fn(x, out)
    torch.cuda.synchronize()
    with Clocks() as clk:
        with profile(activities=[ProfilerActivity.CUDA]) as prof_:
            for r in range(args.reps):
                for tag, fn, x, out, _, _ in tcalls:
                    torch.sum(flush, 0, out=fsink)
                    fn(x, out)
                if r % 8 == 7:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
    evs = sorted([ev for ev in prof_.profiler.kineto_results.events() if ev.device_type() == torch.autograd.DeviceType.CUDA],
                 key=lambda ev: ev.start_ns())
    # group: each flush (reduce kernel over the 256 MB buffer) starts a group = one call
    groups, cur = [], None
    for ev in evs:
        if "reduce_kernel" in ev.name() and ev.duration_ns() > 20000:
            cur = []
            groups.append(cur)
        elif cur is not None:
            cur.append(ev)
    if len(groups) != args.reps * len(tcalls):
        print("WARN groups %d != %d" % (len(groups), args.reps * len(tcalls)), flush=True)
    per = {c[0]: ([], []) for c in tcalls}
    for j, gr in enumerate(groups[: args.reps * len(tcalls)]):
        tag = tcalls[j % len(tcalls)][0]
        while gr and gr[-1].name().startswith("Memset"):
            gr = gr[:-1]  # the flush's own output memset (issued just before the next reduce kernel)
        main = [ev for ev in gr if "rq<" in ev.name() or ("nvjet" in ev.name() or "gemm" in ev.name().lower() or "sm90" in ev.name())]
        zero = [ev for ev in gr if ev not in main]
        per[tag][0].append(sum(ev.duration_ns() for ev in main) / 1e3)
        per[tag][1].append(sum(ev.duration_ns() for ev in zero) / 1e3)
        if j < len(tcalls):
            print("KNAME %s %s" % (tag, " | ".join(ev.name()[:90] for ev in gr)), flush=True)
    print(clk.summary(), flush=True)
    results = []
    ref = {}
    for tag, fn, x, out, info, v in tcalls:
        ts, zs = per[tag]
        if not ts:
            continue
        med, mn, z = statistics.median(ts), min(ts), statistics.median(zs) if zs else 0.0
        key = tag.rsplit(" ", 1)[0]
        if v is None:
            ref[key] = med
        d = "" if key not in ref else " d_cublas=%+.2f" % (med - ref[key])
        print("T %-28s med %6.2f min %6.2f z=%.2f%s  %s" % (tag, med, mn, z, d, info), flush=True)
        results.append({"tag": tag, "med": med, "min": mn, "zero": z, "info": info})
    if args.graph:
        for tag, fn, x, out, info, v in tcalls:
            us = S._time_us(fn, x, out, flush, fsink, it=31)
            print("G %-28s %.2f" % (tag, us), flush=True)
            results.append({"tag": tag, "graph_us": us})
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
