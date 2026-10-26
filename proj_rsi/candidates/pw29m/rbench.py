"""Router-only H100 bench for pw29h (track P3). Run inside the container from /okc:
    python proj_rsi/candidates/pw29h/rbench.py [--ms 128,256] [--only name,name] [--rounds 5] [--reps 20] [--no-gv]
Builds only the rw extension (and pw23's gv extension from the persistent cache for the incumbent rp/rv paths),
validates each path with the submission's own _validate (one-hot x8 exact vs hi+lo, random rows vs FP64, odd m),
then times every path the way h100_bench.py does (CUDA graph of the whole call incl. the memset node, 256 MB L2
flush before each replay, median of reps), interleaved over rounds; prints min and median of the per-round medians
and the replay overhead (empty graph) so kernel-side times can be compared."""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("sub_pw29h", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


SINK = []


def replay_median(g, flush, reps):
    """judge-like cold L2: read (not write) a buffer larger than L2 before each replay (TASK.md)"""
    if not SINK:
        SINK.append(torch.zeros((), dtype=torch.float32, device=flush.device))
    ts = []
    for _ in range(reps):
        torch.sum(flush, 0, out=SINK[0])
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


TERMS = ("start", "issued", "first", "lastdata", "mmadone", "ready", "redsout")


def stamp_report(mod, st, m, n, K, x, out, flush, args):
    """per-CTA stamps (ns from the earliest CTA start): entry, loads issued, first stage landed (= first MMA), last
    stage landed, last MMA done, staged + zero-ready, reds issued. Median and max over CTAs, median over reps."""
    ext = mod._rwext()
    only = set(filter(None, args.only.split(",")))
    dbg = st["rw_cnt"][16384:].view(torch.int64)
    for name, v, zf, ms in mod.RSPALL:
        if (v + 20000 not in mod.RWV and v + 20000 not in mod.RTV) or m not in ms or (only and name not in only):
            continue
        rows = []
        for _ in range(args.reps):
            torch.sum(flush, 0, out=SINK[0]) if SINK else None
            if zf:
                out.zero_()
            dbg.zero_()
            torch.cuda.synchronize()
            rc = ext.rwrun(v + 20000, x, st["rs_hi"], st["rs_lo"], out, st["rw_cnt"])
            torch.cuda.synchronize()
            if rc != 0:
                print("ST m=%d %s rc=%d" % (m, name, rc), flush=True)
                break
            a = dbg.view(-1, 8).clone()
            a = a[a[:, 0] > 0]
            t0 = a[:, 0].min()
            rel = (a[:, :7] - t0).double() / 1000.0  # us
            rows.append(torch.stack((rel.median(0).values, rel.max(0).values)))
        if not rows:
            continue
        r = torch.stack(rows).median(0).values
        ncta = a.shape[0]
        sms = len(set(a[:, 7].tolist()))
        print("ST m=%d %-16s ctas %d sms %d | med: %s | max: %s" % (m, name, ncta, sms,
              " ".join("%s %.2f" % (t, r[0, j]) for j, t in enumerate(TERMS)),
              " ".join("%s %.2f" % (t, r[1, j]) for j, t in enumerate(TERMS))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(HERE / "submission.py"))
    ap.add_argument("--ms", default="128,256")
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-gv", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--stamps", action="store_true", help="MODE-16 instances: per-CTA globaltimer phase report")
    args = ap.parse_args()
    torch.cuda.init()
    dev = "cuda"
    print("device %s torch %s" % (torch.cuda.get_device_name(0), torch.__version__), flush=True)
    mod = load(args.sub)
    mod.RWALL = True
    w = reference.generate_weights(1234)["router"].contiguous()  # fp32 [256, 8192]
    n, K = w.shape
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    st = {"rs_hi": hi.contiguous(), "rs_lo": lo.contiguous(),
          "rs_ws": torch.zeros((256 * 256,), dtype=torch.float32, device=dev),
          "rs_cnt": torch.zeros((64,), dtype=torch.int32, device=dev),
          "rw_cnt": torch.zeros((16384 + 2 * 8 * 1024,), dtype=torch.int32, device=dev)}  # + MODE-16 stamp area
    rexact = hi.float() + lo.float()
    hl_t = torch.cat((hi, lo)).t().contiguous()
    hi_t, lo_t = hi.t().contiguous(), lo.t().contiguous()
    t = time.time()
    mod._rwext()
    print("RW ext ready %.0fs" % (time.time() - t), flush=True)
    gv = not args.no_gv
    if gv:
        t = time.time()
        try:
            mod._gvext()
            print("GV ext ready %.0fs" % (time.time() - t), flush=True)
        except Exception as e:  # noqa: BLE001
            print("GV ext failed %s" % e, flush=True)
            gv = False
    only = set(filter(None, args.only.split(",")))
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)  # 256 MB
    SINK.append(torch.zeros((), dtype=torch.float32, device=dev))
    tiny = torch.zeros(32, device=dev)
    g0 = graph_of(lambda: tiny.add_(1.0))
    gen = torch.Generator(device=dev).manual_seed(4242)
    for m in map(int, args.ms.split(",")):
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.float32, device=dev)
        paths = []
        # cuBLAS references: the harness's torch baseline (2 mm + add) and r6's heuristic m > 32 path (one mm, halves)
        def t2(xx, oo):
            torch.mm(xx, hi_t, out_dtype=torch.float32, out=oo)
            oo.add_(torch.mm(xx, lo_t, out_dtype=torch.float32))
            return 0

        def r6h(xx, oo):
            a = torch.mm(xx, hl_t, out_dtype=torch.float32)
            torch.add(a[:, :n], a[:, n:], out=oo)
            return 0
        paths.append(("torch2mm", t2))
        paths.append(("r6heur", r6h))
        for name, v, zf, ms in mod.RSPALL:
            if m not in ms or (v < 100 and not gv):
                continue
            plan = (-7, v, zf, None, n, None, None)
            paths.append((name, lambda xx, oo, plan=plan: mod._rscall(st, plan, xx, oo)))
        def ms_(xx, oo):
            oo.zero_()
            return 0
        paths.append(("memset", ms_))
        ladder = []
        if args.ladder:
            ext = mod._rwext()
            for name, v, zf, ms in mod.RSPALL:
                if v not in (101, 104) or m not in ms:
                    continue
                for md, tag in ((1, "ld"), (3, "ld-nored"), (2, "ld+mma-nored")):
                    plan = (-7, v + 1000 * md, True, None, n, None, None)
                    ladder.append(("%s:%s" % (name, tag), lambda xx, oo, plan=plan: mod._rscall(st, plan, xx, oo)))
                ladder.append(("%s:nozero" % name, lambda xx, oo, v=v: ext.rwrun(v, xx, st["rs_hi"], st["rs_lo"], oo, st["rw_cnt"])))
        paths += ladder
        ext = mod._rwext()
        for name, v, zf, ms in mod.RSPALL:
            if v >= 100 and m in ms:
                print("RB m=%d %-24s occ %d (clusters if cN, else CTAs/SM)" % (m, name, ext.rwocc(v, m, n, K)), flush=True)
        if args.stamps:
            stamp_report(mod, st, m, n, K, x, out, flush, args)
            continue
        if only:
            paths = [p for p in paths if p[0].split(":")[0] in only or p[0] in ("torch2mm", "memset")]
        good = []
        for name, fn in paths:
            try:
                rc = fn(x, out)
                torch.cuda.synchronize()
                if rc not in (0, None):
                    print("RB m=%d %-24s ERROR rc=%s" % (m, name, rc), flush=True)
                    continue
                if name != "memset" and ":" not in name and not args.no_validate:
                    ok, msg = mod._validate(fn, w, m, n, True, gen, rexact, {128: 121, 256: 250}[m])
                    if not ok:
                        print("RB m=%d %-24s INVALID %s" % (m, name, msg), flush=True)
                        continue
                good.append((name, fn, graph_of(lambda fn=fn: fn(x, out))))
            except Exception as e:  # noqa: BLE001
                print("RB m=%d %-24s ERROR %s" % (m, name, str(e).splitlines()[-1][:200]), flush=True)
        res = {name: [] for name, _, _ in good}
        ovh = []
        for r in range(args.rounds):
            ovh.append(replay_median(g0, flush, args.reps))
            for name, _, g in good:
                res[name].append(replay_median(g, flush, args.reps))
        o = statistics.median(ovh)
        print("RB m=%d OVERHEAD %.2f (empty graph)" % (m, o), flush=True)
        base = statistics.median(res["torch2mm"]) if "torch2mm" in res else float("nan")
        for name, _, _ in good:
            md, mn = statistics.median(res[name]), min(res[name])
            print("RB m=%d %-24s med %6.2f min %6.2f  kernel-side %6.2f  vs torch2mm %+6.2f" % (m, name, md, mn, md - o, md - base), flush=True)


if __name__ == "__main__":
    main()
