"""P2 lab (H100 box): qkv_gate m <= 32 kernel variants timed side by side with r6's own gv candidates.

Usage (inside the container, cwd /okc):
  python proj_rsi/candidates/pw28h/lab.py --plan base [--build-only] [--reps 60] [--ms 1,8,16,32]

Each plan is a list of variants (label, kind, instance, buckets, n): kind "gv" = r6's gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>
(n < 2304 cuts the grid: wrong answer, timing only; V bits 9-11 = pw14 ladder level), "gk" = gvk<MT, ST, NW, KU, XS, V>.
Full-n, ladder-free variants are validated with the submission's _validate (one-hot x8 bit-exact, random rows, odd m).
Timing: CUPTI kernel durations (torch.profiler / kineto), each launch after a read pass over a 256 MB buffer (cold L2),
variants interleaved round-robin; median and min per variant. Kernel-side only: no graph-replay overhead.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
import reference  # noqa: E402


def load_sub(path):
    spec = importlib.util.spec_from_file_location("sub_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_sub(HERE / "submission.py")
N, K = 2304, 8192
T, KC = N // 16, K // 64
LAD = lambda lv: lv << 9  # noqa: E731
MTB = {1: 1, 8: 1, 16: 2, 32: 4}


def r6cands(mt, ms):
    out = []
    for geo, vs in S.GVC[("qkv_gate", mt)]:
        for v in vs:
            inst = (1, mt, S._gvst(geo)) + tuple(geo[:6]) + (v,)
            out.append(("r6:" + ".".join(map(str, geo)) + "v%d" % v, "gv", inst, ms, N))
    return out


R6M1 = (1, 1, 2, 1, 8, 16, 1, 1, 0, 1)  # r6's judge pick at m = 1 / 8


def plan_base():
    v = []
    v += r6cands(1, (1, 8)) + r6cands(2, (16,)) + r6cands(4, (32,))
    # ladder on r6's m=1 pick: grid 144 (real), 132, 128 CTAs; levels W only / + x / + decode / + mma / full
    for n in (N, 132 * 16, 128 * 16):
        for lv in (1, 2, 3, 4, 0):
            if n == N and lv == 0:
                continue
            inst = R6M1[:9] + (R6M1[9] | LAD(lv),)
            v.append(("lad%d-g%d" % (lv, n // 16), "gv", inst, (1,), n))
    for nm, ms, g in S.GKP:
        st, nw, ku, xs, vv = g
        for m in ms:
            v.append(("pw28:" + nm, "gk", (MTB[m], st, nw, ku, xs, vv), (m,), N))
    return v


def plan_deep():
    v = [x for x in r6cands(1, (1, 8)) if x[0] == "r6:1.8.16.1.1.0v1"]
    v += [x for x in r6cands(2, (16,)) if x[0] in ("r6:1.4.8.4.1.2.1v1", "r6:1.4.16.2.1.0v1")]
    v += [x for x in r6cands(4, (32,)) if x[0] in ("r6:1.8.16.1.1.2v1",)]
    I = 1 | 16
    for st in (2, 4, 6, 8):
        v.append(("gk9x16-st%d" % st, "gk", (1, st, 9, 16, 0, I), (1, 8), N))
    for st in (4, 8, 12, 16):
        v.append(("gk8x18-st%d" % st, "gk", (1, st, 8, 18, 0, I), (1, 8), N))
    for st in (3, 4):
        v.append(("gk18x8-st%d" % st, "gk", (1, st, 18, 8, 0, I), (1, 8), N))
    for st in (2, 4, 6, 8):
        v.append(("gk9x16-l1-st%d" % st, "gk", (2, st, 9, 16, 2, I), (16,), N))
        v.append(("gk9x16-l1-st%d" % st, "gk", (4, st, 9, 16, 2, I), (32,), N))
    for st in (6, 10):
        v.append(("gk8x18-l1-st%d" % st, "gk", (2, st, 8, 18, 2, I), (16,), N))
        v.append(("gk8x18-l1-st%d" % st, "gk", (4, st, 8, 18, 2, I), (32,), N))
    for st in (4,):
        v.append(("gk9x16-xr-st%d" % st, "gk", (2, st, 9, 16, 0, I), (16,), N))
    # W-only rung with the ring depth bounded by a register cap (minBlocks 2 / 3) at the real 144-CTA grid
    for mb in (2, 3):
        v.append(("lad1-mb%d-g144" % mb, "gv", R6M1[:9] + (R6M1[9] | LAD(1) | mb << 12,), (1,), N))
    return v


R6M32 = (1, 4, 2, 1, 8, 16, 1, 1, 2, 1)  # r6's H100 lab pick at m = 32 (judge: 1.4.1.1.4.16.2.1.2.1)


def plan_lad32():
    v = [x for x in r6cands(4, (32,)) if x[0] in ("r6:1.8.16.1.1.2v1", "r6:1.4.16.2.1.2.1v1")]
    for n in (N, 132 * 16, 128 * 16):
        for lv in (1, 2, 3, 4, 0):
            if n == N and lv == 0:
                continue
            v.append(("lad%d-g%d" % (lv, n // 16), "gv", R6M32[:9] + (R6M32[9] | LAD(lv),), (32,), n))
    # same ladder at m=1 / 16 on the m=1 pick, for a same-run comparison
    for n in (N, 132 * 16):
        for lv in (1, 3, 4, 0):
            if n == N and lv == 0:
                continue
            v.append(("m1k-lad%d-g%d" % (lv, n // 16), "gv", R6M1[:9] + (R6M1[9] | LAD(lv),), (1,), n))
    v += [x for x in r6cands(1, (1,)) if x[0] == "r6:1.8.16.1.1.0v1"]
    return v


def plan_lad1x():
    """m=1 ladder with every rung held at r6's real occupancy (minBlocks 2: <= 128 regs, 2 CTAs/SM possible)"""
    v = [x for x in r6cands(1, (1, 8)) if x[0] == "r6:1.8.16.1.1.0v1"]
    for n in (N, 132 * 16):
        for lv in (1, 2, 3, 4, 0):
            if n == N and lv == 0:
                continue
            v.append(("mb2-lad%d-g%d" % (lv, n // 16), "gv", R6M1[:9] + (R6M1[9] | LAD(lv) | 2 << 12,), (1, 8), n))
    return v


def plan_rt2():
    """x reuse from registers: RT = 2 row tiles per warp (x fragment loaded once per two W tiles), split-K S over a
    cluster so the grid stays at 144 CTAs; register cap lifted by the minBlocks override (V bits 12-14)"""
    v = [x for x in r6cands(4, (32,)) if x[0] == "r6:1.8.16.1.1.2v1"]
    v += [x for x in r6cands(2, (16,)) if x[0] == "r6:1.4.8.4.1.2.1v1"]
    for mt, ms in ((2, (16,)), (4, (32,))):
        for (wk, ku, s_, xs, st, mb) in ((4, 16, 2, 2, 2, 2), (4, 16, 2, 2, 2, 1), (4, 16, 2, 0, 2, 2), (4, 8, 4, 2, 2, 2),
                                         (8, 8, 2, 2, 2, 1), (2, 16, 4, 2, 3, 3)):
            inst = (1, mt, st, 1, wk, ku, s_, 2, xs, 1 | mb << 12)
            v.append(("rt2-wk%d-ku%d-s%d-xs%d-st%d-mb%d" % (wk, ku, s_, xs, st, mb), "gv", inst, ms, N))
    return v


def plan_wr2():
    """x shared through L1 by WR row-tile warps of one CTA, no split-K (72 CTAs x 16 warps), and WR 3 (48 CTAs)"""
    v = [x for x in r6cands(4, (32,)) if x[0] == "r6:1.8.16.1.1.2v1"]
    v += [x for x in r6cands(2, (16,)) if x[0] == "r6:1.4.8.4.1.2.1v1"]
    for mt, ms in ((2, (16,)), (4, (32,))):
        for (wr, wk, ku, st) in ((2, 8, 16, 2), (2, 16, 8, 2), (2, 16, 8, 1), (3, 8, 16, 2)):
            inst = (1, mt, st, wr, wk, ku, 1, 1, 2, 1)
            v.append(("wr%d-wk%d-ku%d-st%d" % (wr, wk, ku, st), "gv", inst, ms, N))
    return v


def plan_m1x():
    """r6's m=1 geometry knobs on the H100: ring depth under the 2-CTA/SM register cap, decode variant, 16 warps"""
    v = [x for x in r6cands(1, (1, 8)) if x[0] == "r6:1.8.16.1.1.0v1"]
    for st, vv in ((1, 1), (3, 1 | 2 << 12), (4, 1 | 2 << 12), (2, 0), (3, 1)):
        v.append(("m1-st%d-v%x" % (st, vv), "gv", (1, 1, st, 1, 8, 16, 1, 1, 0, vv), (1, 8), N))
    for st in (2, 3):
        v.append(("m1-wk16-st%d" % st, "gv", (1, 1, st, 1, 16, 8, 1, 1, 0, 1 | 1 << 12), (1, 8), N))
    return v


def plan_wr2b():
    """neighbourhood of the m=16 winner (2,16,8,1,1,2) ST1: x shared via L1 by 2 row-tile warps, 72 CTAs x 32 warps"""
    v = [x for x in r6cands(1, (1, 8)) if x[0] == "r6:1.8.16.1.1.0v1"]
    v += [x for x in r6cands(2, (16,)) if x[0] == "r6:1.4.8.4.1.2.1v1"]
    v += [x for x in r6cands(4, (32,)) if x[0] == "r6:1.8.16.1.1.2v1"]
    G = []
    for mt, ms in ((1, (1, 8)), (2, (16,)), (4, (32,))):
        for (wr, wk, ku, st, xs, vv) in ((2, 16, 8, 1, 2, 1), (2, 16, 8, 1, 2, 3), (2, 16, 8, 1, 2, 0), (2, 16, 8, 2, 2, 3),
                                          (2, 16, 8, 1, 0, 1), (2, 16, 8, 2, 0, 1), (2, 8, 16, 1, 2, 1), (2, 8, 16, 3, 2, 1),
                                          (2, 8, 16, 2, 0, 1), (4, 8, 16, 1, 2, 1)):
            if mt == 4 and xs == 0:
                continue
            v.append(("wr%d-wk%d-ku%d-st%d-xs%d-v%d" % (wr, wk, ku, st, xs, vv), "gv", (1, mt, st, wr, wk, ku, 1, 1, xs, vv), ms, N))
    return v


def plan_wr2c():
    v = [x for x in r6cands(1, (1, 8)) if x[0] == "r6:1.8.16.1.1.0v1"]
    v += [x for x in r6cands(2, (16,)) if x[0] == "r6:1.4.8.4.1.2.1v1"]
    v += [x for x in r6cands(4, (32,)) if x[0] == "r6:1.8.16.1.1.2v1"]
    A = lambda lab, inst, ms: v.append((lab, "gv", inst, ms, N))  # noqa: E731
    A("wr2-wk16-ku8-st1-xs2-v1", (1, 1, 1, 2, 16, 8, 1, 1, 2, 1), (1, 8))
    for st in (1, 2):
        A("t1-wk16-ku8-st%d-xs0-mb2" % st, (1, 1, st, 1, 16, 8, 1, 1, 0, 1 | 2 << 12), (1, 8))
    A("t1-wk16-ku8-st1-xs2-mb2", (1, 1, 1, 1, 16, 8, 1, 1, 2, 1 | 2 << 12), (1, 8))
    A("t1-wk8-ku16-st1-xs0-mb3", (1, 1, 1, 1, 8, 16, 1, 1, 0, 1 | 3 << 12), (1, 8))
    A("wr2-wk8-ku16-st2-xs0-v1", (1, 2, 2, 2, 8, 16, 1, 1, 0, 1), (16, 32))
    A("wr2-wk8-ku16-st1-xs0-v1", (1, 2, 1, 2, 8, 16, 1, 1, 0, 1), (16, 32))
    A("wr2-wk16-ku8-st1-xs2-v3", (1, 2, 1, 2, 16, 8, 1, 1, 2, 3), (16, 32))
    A("wr2-wk8-ku16-st1-xs2-v1", (1, 2, 1, 2, 8, 16, 1, 1, 2, 1), (16, 32))
    A("t1-wk16-ku8-st1-xs2-mb2", (1, 2, 1, 1, 16, 8, 1, 1, 2, 1 | 2 << 12), (16,))
    A("wr2-wk8-ku16-st1-xs2-v1-mt4", (1, 4, 1, 2, 8, 16, 1, 1, 2, 1), (32,))
    A("wr2-wk8-ku16-st1-xs2-v3-mt4", (1, 4, 1, 2, 8, 16, 1, 1, 2, 3), (32,))
    return v


PLANS = {"wr2c": plan_wr2c, "wr2b": plan_wr2b, "m1x": plan_m1x, "base": plan_base, "deep": plan_deep, "lad32": plan_lad32, "lad1x": plan_lad1x, "rt2": plan_rt2, "wr2": plan_wr2}

RDCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
template <int U>
__global__ void rd(const uint4* __restrict__ p, long long n16, uint32_t* sink) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  uint32_t a = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int k = 0; k < U; k++) {
      const long long j = i + k * stride;
      if (j < n16) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w) : "l"(p + j));
      else v[k] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < U; k++) a ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
  }
  if (a == 0x9e3779b9u) sink[threadIdx.x] = a;
}
int64_t rdrun(torch::Tensor b, int64_t nbytes, int64_t grid, int64_t nt, int64_t u, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* p = (const uint4*)b.data_ptr(); uint32_t* s = (uint32_t*)sink.data_ptr();
  const long long n = nbytes / 16;
  switch (u) {
    case 4: rd<4><<<grid, nt, 0, st>>>(p, n, s); break;
    case 8: rd<8><<<grid, nt, 0, st>>>(p, n, s); break;
    case 16: rd<16><<<grid, nt, 0, st>>>(p, n, s); break;
    default: return -1;
  }
  return (int64_t)cudaGetLastError();
}
"""


def stream_probe(reps):
    from torch.utils.cpp_extension import load_inline
    from torch.profiler import profile, ProfilerActivity
    ext = load_inline("p2rd", cpp_sources="int64_t rdrun(torch::Tensor, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                      cuda_sources=RDCU, functions=["rdrun"], extra_cuda_cflags=["-O3"], verbose=False)
    buf = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    sink = torch.zeros(1024, dtype=torch.int32, device="cuda")
    flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")
    fs = torch.empty((), dtype=torch.float32, device="cuda")
    cfgs = [(nb, g, nt, u) for nb in (25165824, 28311552, 50331648) for g, nt in ((132, 512), (132, 1024), (264, 512), (528, 256), (1056, 256))
            for u in (4, 8, 16)]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for r in range(reps):
            for nb, g, nt, u in cfgs:
                torch.sum(flush, 0, out=fs)
                ext.rdrun(buf, nb, g, nt, u, sink)
            if r % 4 == 3:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    evs = sorted([e for e in prof.profiler.kineto_results.events() if e.device_type() == torch.autograd.DeviceType.CUDA
                  and "rd<" in e.name()], key=lambda e: e.start_ns())
    per = {}
    for j, e in enumerate(evs):
        per.setdefault(cfgs[j % len(cfgs)], []).append(e.duration_ns() / 1e3)
    best = {}
    for c, ts in per.items():
        med = statistics.median(ts)
        if c[0] not in best or med < best[c[0]][0]:
            best[c[0]] = (med, c)
    for nb, (med, c) in sorted(best.items()):
        print("STREAM %.1f MB best med %.2f us (%.2f TB/s) grid=%d nt=%d u=%d" % (nb / 1e6, med, nb / med / 1e6, c[1], c[2], c[3]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="base")
    ap.add_argument("--ms", default="1,8,16,32")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma list of label substrings")
    ap.add_argument("--stream", action="store_true", help="also run the bare-stream read probe")
    args = ap.parse_args()
    ms = [int(a) for a in args.ms.split(",")]
    var = [x for x in PLANS[args.plan]() if any(m in x[3] for m in ms)]
    if args.only:
        keys = args.only.split(",")
        var = [x for x in var if x[0].startswith("r6:") or any(k in x[0] for k in keys)]
    S.GVI = list(dict.fromkeys(x[2] for x in var if x[1] == "gv"))
    S.GKI = list(dict.fromkeys(x[2] for x in var if x[1] == "gk"))
    t0 = time.time()
    ext = S._gvext()
    print("BUILD %d gv + %d gk instances in %.0fs" % (len(S.GVI), len(S.GKI), time.time() - t0), flush=True)
    for x in var:
        if x[1] == "gv":
            pass
    if args.build_only:
        return
    dev = "cuda"
    if args.stream:
        stream_probe(20)
    w = reference.generate_weights(reference.WEIGHTS_SEED)["qkv_gate"]
    fr = S._pack(w)
    css = [S._c12(fr, dv) for dv in (0, 1)]
    assert all(c is not None for c in css)
    esc = torch.cat((css[1][2], torch.zeros_like(css[1][2][:, :1])), 1)
    ws = torch.full((T * 32 * 128 * 4,), -1, dtype=torch.int32, device=dev)
    gvws = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
    gvcnt = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(2828)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    results = []
    for m in ms:
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        calls = []
        for lab, kind, inst, bms, n in var:
            if m not in bms:
                continue
            if kind == "gv":
                i = S.GVI.index(inst)
                geo = inst[3:9]
                e = S._gvesc(esc, n // 16, geo)
                if e is None:
                    print("SKIP m=%d %s escapes" % (m, lab), flush=True)
                    continue
                cs = css[inst[9] & 1]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                fn = (lambda xx, oo, i=i, cs=cs, e=e: ext.gvrun(i, cs[0], cs[0], xx, oo, e, gvws, gvcnt, cs[1], 0))
                regs = int(ext.gvregs(i, cs[0], cs[0], x, out, e, gvws, gvcnt, cs[1], 0))
                occ = int(ext.gvocc(i, cs[0], cs[0], x, out, e, gvws, gvcnt, cs[1], 0))
                info = "regs=%d occ=%d" % (regs, occ)
            else:
                i = S.GKI.index(inst)
                _, st_, nw, ku, xs, vv = inst
                e = S._gkesc(esc, n // 16, KC, nw, ku, bool(vv & 16))
                if e is None:
                    print("SKIP m=%d %s escapes" % (m, lab), flush=True)
                    continue
                cs = css[vv & 1]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                fn = (lambda xx, oo, i=i, cs=cs, e=e: ext.gkrun(i, cs[0], xx, oo, e, ws, cs[1], 0))
                a = (cs[0], x, out, e, ws, cs[1])
                grid, occ, regs = int(ext.gkrun(i, *a, 3)), int(ext.gkrun(i, *a, 1)), int(ext.gkrun(i, *a, 2))
                spill = int(ext.gkrun(i, *a, 4))
                info = "regs=%d occ=%d grid=%d%s" % (regs, occ, grid, " SPILL=%d" % spill if spill else "")
                if grid < 0 or occ < grid:
                    print("SKIP m=%d %s %s" % (m, lab, info), flush=True)
                    continue
            r = fn(x, out)
            torch.cuda.synchronize()
            if r != 0:
                print("SKIP m=%d %s launch=%d %s" % (m, lab, r, info), flush=True)
                continue
            lad = kind == "gv" and (inst[9] >> 9) & 7
            if n == N and not lad and not args.no_validate:
                ok, msg = S._validate(fn, w, m, n, False, gen, None, {8: 5, 16: 9, 32: 17}.get(m))
                ws.fill_(-1)
                if not ok:
                    print("INVALID m=%d %s %s" % (m, lab, msg), flush=True)
                    continue
                info += " valid"
            calls.append((lab, fn, out, info))
        # timing: interleaved, CUPTI durations
        from torch.profiler import profile, ProfilerActivity
        for lab, fn, out, _ in calls:
            fn(x, out)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for r in range(args.reps):
                for lab, fn, out, _ in calls:
                    torch.sum(flush, 0, out=sink)
                    fn(x, out)
                if r % 8 == 7:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
        evs = [e for e in prof.profiler.kineto_results.events() if e.device_type() == torch.autograd.DeviceType.CUDA
               and ("gv" in e.name())]
        evs.sort(key=lambda e: e.start_ns())
        if len(evs) != args.reps * len(calls):
            print("WARN m=%d events %d != %d" % (m, len(evs), args.reps * len(calls)), flush=True)
        per = {c[0]: [] for c in calls}
        for j, e in enumerate(evs[: args.reps * len(calls)]):
            per[calls[j % len(calls)][0]].append(e.duration_ns() / 1e3)
        ref = None
        rows = []
        for lab, fn, out, info in calls:
            ts = per[lab]
            med, mn = statistics.median(ts), min(ts)
            rows.append((med, mn, lab, info))
        r6best = min((r for r in rows if r[2].startswith("r6:")), default=None)
        for med, mn, lab, info in rows:
            d = "" if r6best is None else " d_r6=%+.2f" % (med - r6best[0])
            print("T m=%-2d %-28s med %6.2f min %6.2f%s  %s" % (m, lab, med, mn, d, info), flush=True)
            results.append({"m": m, "label": lab, "med": med, "min": mn, "info": info})
        if r6best:
            print("R6BEST m=%d %s %.2f" % (m, r6best[2], r6best[0]), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
