"""pw28 stream-K qkv kernel (gvk): CPU emulation of its work split, slot reduction and escape lists.

The kernel's index formulas are transcribed literally (per CTA b, warp w, step u): unit(u) through the u < sw selection,
the switch step sw, pubB / conA, the producer slots [tile][warp][fragment], the consumer's slot predicate and its
fixed summation order, and the final emit loop (h = 0 tile A, h = 1 tile B). Checks per GKP geometry:
  1. coverage: every (tile, chunk) unit is run exactly once; u < sw <=> the CTA-local index q < nB;
  2. slots: the set of slots written by producers equals the set read by consumers, every read finds a written slot,
     the workspace is all-sentinel after the launch, every tile is emitted exactly once;
  3. numbers: the emulated launch (fp32 partials per unit, the kernel's accumulation and reduction order) equals
     x @ W^T within the template tolerance for m = 1, 5, 7, 8, 9, 16, 17, 32 (rows past m clamped as in the kernel), and
     one-hot x8 rows exactly;
  4. escapes: the submission's _c12 pack + _gkesc lists, walked as the kernel walks them (per warp, entry seq == step u,
     lane / register / half fields), restore every weight bit-exactly; the list order lets the walk consume every entry.
python3 -m proj_rsi.tools.gk_emu [submission.py]"""
import importlib.util
import sys

import numpy as np
import torch

SENT = -1


def geometry(T, KC, NW, KU, IL):
    """per CTA: (U0, tA, oA, tB, nB, pubB, conA) and per warp: sw and the unit / chunk of every step"""
    P = NW * KU
    G = T * KC // P
    ctas = []
    for b in range(G):
        U0 = b * P
        tA, tB = U0 // KC, (U0 + P - 1) // KC
        oA = U0 - tA * KC
        nB = U0 + P - tB * KC if tB != tA else 0
        pubB = nB != 0 and U0 + P < (tB + 1) * KC
        conA = oA != 0
        warps = []
        for w in range(NW):
            sw = (min(KU, (nB - w + NW - 1) // NW) if nB > w else 0) if IL else min(KU, max(0, nB - w * KU))
            SU = NW if IL else 1
            q0 = w if IL else w * KU
            uB, uA, cA = tB * KC + q0, U0 - nB + q0, oA - nB + q0
            steps = []
            for u in range(KU):
                inB = u < sw
                c = (q0 if inB else cA) + u * SU
                U = (uB if inB else uA) + u * SU
                q = u * NW + w if IL else w * KU + u
                assert inB == (q < nB), "u < sw must equal q < nB"
                assert U == (tB * KC + q if q < nB else U0 + q - nB) and c == U % KC, "unit / chunk formula"
                steps.append((U, c))
            warps.append((sw, steps))
        ctas.append(dict(U0=U0, tA=tA, oA=oA, tB=tB, nB=nB, pubB=pubB, conA=conA, warps=warps))
    return ctas


def slot_valid(s, oA, KU, IL):
    return s < oA if IL else s * KU < oA


def launch(ctas, Pt, m, T, NW, KU, IL, MT):
    """Pt[U] = fp32 partial [16, 8 MT] of unit U (tile-major). Returns out [T * 16, 8 MT] and the emitted tile list."""
    NI = 16 * 8 * MT  # emulate each warp fragment as a [16, 8 MT] block (layout-free: the kernel's lanes map 1:1)
    ws = {}
    red = {}
    for b, C in enumerate(ctas):  # phase 1: every CTA's main loop (producers store at their switch step)
        for w, (sw, steps) in enumerate(C["warps"]):
            acc = np.zeros((16, 8 * MT), np.float32)

            def flushB(acc):
                if C["pubB"]:
                    key = (C["tB"], w)
                    assert key not in ws, "slot written twice"
                    ws[key] = acc.copy()
                else:
                    red[(b, w, 1)] = acc.copy()
                return np.zeros_like(acc)
            for u, (U, c) in enumerate(steps):
                if u == sw and sw > 0:
                    acc = flushB(acc)
                acc = (acc + Pt[U]).astype(np.float32)
            if sw == KU:
                acc = flushB(acc)
            if sw == 0:
                red[(b, w, 1)] = np.zeros_like(acc)
            red[(b, w, 0)] = acc
    out = np.full((T * 16, 8 * MT), np.nan, np.float32)
    emitted = []
    for b, C in enumerate(ctas):  # phase 2: consumers read (all producers are done: the kernel's spin never sees a race here)
        prod = {}
        if C["conA"]:
            for s in range(NW):
                if slot_valid(s, C["oA"], KU, IL):
                    assert (C["tA"], s) in ws, "consumer reads a slot no producer wrote"
                    prod[s] = ws.pop((C["tA"], s))  # read + restore the sentinel
        for h in (0, 1):
            if h and (C["nB"] == 0 or C["pubB"]):
                continue
            v = red[(b, 0, h)].copy()
            for q in range(1, NW):
                v = (v + red[(b, q, h)]).astype(np.float32)
            if not h and C["conA"]:
                for s in range(NW):
                    if slot_valid(s, C["oA"], KU, IL):
                        v = (v + prod[s]).astype(np.float32)
            t = C["tB"] if h else C["tA"]
            out[t * 16:(t + 1) * 16] = v
            emitted.append(t)
    assert not ws, "slots left non-sentinel after the launch: %s" % list(ws)[:4]
    assert sorted(emitted) == list(range(T)), "every tile emitted exactly once"
    return out


def check_numbers(ctas, T, KC, NW, KU, IL, MT, m, rng, onehot=False):
    K = KC * 64
    W = (rng.standard_normal((T * 16, K)) * 0.02).astype(np.float32)
    W = torch.from_numpy(W).to(torch.bfloat16).float().numpy()
    if onehot:
        x = np.zeros((m, K), np.float32)
        cols = rng.choice(K, m, replace=False)
        x[np.arange(m), cols] = 8.0
    else:
        x = torch.from_numpy(rng.standard_normal((m, K)).astype(np.float32)).to(torch.bfloat16).float().numpy()
    xr = x[np.minimum(np.arange(8 * MT), m - 1)]  # rows past m clamped to m - 1 (xp[n] in the kernel)
    Pt = np.einsum("tikc,jkc->tkij", W.reshape(T, 16, KC, 64), xr.reshape(8 * MT, KC, 64), optimize=True)
    Pt = Pt.reshape(T * KC, 16, 8 * MT).astype(np.float32)
    out = launch(ctas, Pt, m, T, NW, KU, IL, MT)[:, :m]
    ref = (W.astype(np.float64) @ x.T.astype(np.float64))
    if onehot:
        return bool(np.array_equal(out, ref.astype(np.float32)))
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    return err < 1e-5  # fp32 sums of fp32 products (the 4e-3 bf16 tolerance applies after the bf16 store)


def check_escapes(sub, T, KC, NW, KU, IL, ctas):
    """the real pack: _pack -> _c12 (dv 1) -> _gkesc; walk as the kernel; must restore the bf16 bits exactly"""
    g = torch.Generator().manual_seed(7)
    w = (torch.randn((T * 16, KC * 64), generator=g) * 0.02)
    hot = torch.rand(w.shape, generator=g) < 4e-4  # outliers above and below the 8-exponent window
    w = torch.where(hot, w * torch.where(torch.rand(w.shape, generator=g) < 0.5, 300.0, 1 / 300.0), w).to(torch.bfloat16)
    fr = sub._pack(w)
    c = sub._c12(fr, 1)
    assert c is not None, "pack failed"
    data, tab, esc = c
    escs = torch.cat((esc, torch.zeros_like(esc[:, :1])), 1)  # matrix column h = 0
    lists = sub._gkesc(escs, T, KC, NW, KU, IL)
    assert lists is not None, "escape list overflow"
    val = sub._unc12(data, tab & 0xffffffffffffffff, escs[:0, :6], 1).long() & 0xffff  # decode without escapes: [T, KC, 4, 32, 8]
    want = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).long() & 0xffff
    L = lists.long() & 0xffffffff
    n_applied = 0
    for b, C in enumerate(ctas):
        for wi, (sw, steps) in enumerate(C["warps"]):
            e = L[(b * NW + wi) * 32:(b * NW + wi + 1) * 32].tolist()
            ep, nx = 0, e[0]
            for u, (U, c) in enumerate(steps):
                t = U // KC
                while (nx >> 20) == u:
                    lane, r, half, hm, xx = (nx >> 15) & 31, (nx >> 11) & 15, (nx >> 10) & 1, (nx >> 9) & 1, nx & 0x7f
                    assert hm == 0
                    j, v = r // 4, 2 * (r % 4) + half
                    val[t, c, j, lane, v] ^= xx << 8
                    n_applied += 1
                    ep += 1
                    nx = e[ep] if ep < 32 else 0xffffffff
            assert ep == 32 or e[ep] == 0xffffffff, "the walk left entries unconsumed"
    assert n_applied == len(esc), (n_applied, len(esc))
    return bool(torch.equal(val, want)), len(esc)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "proj_rsi/candidates/pw28/submission.py"
    spec = importlib.util.spec_from_file_location("sub", path)
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    T, KC = 2304 // 16, 8192 // 64
    rng = np.random.default_rng(28)
    geos = sorted({(g[1], g[2], bool(g[4] & 16)) for _, _, g in sub.GKP})
    for NW, KU, IL in geos:
        ctas = geometry(T, KC, NW, KU, IL)
        cov = np.zeros(T * KC, np.int64)
        for C in ctas:
            for sw, steps in C["warps"]:
                for U, c in steps:
                    cov[U] += 1
        assert (cov == 1).all(), "coverage"
        shared = sum(C["pubB"] for C in ctas)
        ok = []
        for MT, ms in ((1, (1, 5, 7, 8)), (2, (9, 16)), (4, (17, 32))):
            for m in ms:
                ok.append(check_numbers(ctas, T, KC, NW, KU, IL, MT, m, rng))
            ok.append(check_numbers(ctas, T, KC, NW, KU, IL, MT, ms[-1], rng, onehot=True))
        eok, ne = check_escapes(sub, T, KC, NW, KU, IL, ctas)
        print("gk NW %2d KU %2d %s: %d CTAs, %d shared tiles, coverage ok, slots ok, numbers %s, escapes %s (%d)"
              % (NW, KU, "interleaved" if IL else "contiguous ", len(ctas), shared, "ok" if all(ok) else "FAIL %s" % ok,
                 "ok" if eok else "FAIL", ne))
        assert all(ok) and eok


if __name__ == "__main__":
    main()
