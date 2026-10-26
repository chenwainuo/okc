"""pw29 router wgmma kernel (rwk): CPU emulation of its cp.async staging, smem layout, wgmma descriptors, ring reuse,
accumulator fragments, smem-staged epilogue and the 16-B red.add reduction.

Mirrors rwk<KR, NSTG, NBUF, TOKS, WN, MINB> index for index (variants from the submission's RWV table):
- load(s): thread tid, iteration it -> i = tid + 128 it, row r = i / CPS, 16-B column cc = s CPS + i % CPS; byte address
  base + ((cc >> 3) % NSLOT) SB + 128 r + ((cc & 7) ^ (r & 7)) << 4; rows < TOKS = x (zero past m), then hi, then lo.
- cp.async groups: a group's bytes may land any time between its issue and the wait that retires it; the emulator runs
  both extremes (land at issue / land at retire) and asserts the ring never refills the slot of a stage whose wgmmas
  may still be in flight (after wgmma.wait_group 1 only stage s is).
- wgmma SS m64nNk16: each descriptor is decoded per the PTX matrix-descriptor format (start >> 4, SBO, swizzle mode
  128B, base offset 0); element (row, k) of the canonical K-major SW128 layout sits at L = start + (row / 8) SBO +
  (row % 8) 128 + 2 k, physically at L ^ (((L >> 7) & 7) << 4). D (fp32) = A (W rows) x B (tokens)^T, hi then lo.
- accumulator fragments per PTX (warp w, lane g t): d[4i + e] = D[16w + g + 8 (e >= 2)][8i + 2t + (e & 1)], then
  the kernel's staging formula sd[tok * 68 + n] and the float4 read-back + red.add (fp32, CTA order) into a zeroed out.
Checks: random rows vs x @ (hi + lo)^T, one-hot x8 bit-exact, m with partial second blocks; negative controls.
python3 -m proj_rsi.tools.rw_emu [submission.py]"""
import sys
import torch

SUB = "proj_rsi/candidates/pw29/submission.py"


def rwv(path):
    src = open(path).read()
    i = src.index("RWV = {")
    ns = {}
    exec(src[i: src.index("}", i) + 1], ns)
    return ns["RWV"]


def bf(u):
    v = u.long() << 16
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32).view(torch.float32).double()


def desc(addr):  # rw_desc
    return ((addr & 0x3FFFF) >> 4) | (1 << 16) | (64 << 32) | (1 << 62)


def desc_rows(d, rows):
    """halves index [rows, 16] of a K-major SW128 operand given its 64-bit descriptor"""
    start, sbo = (d & 0x3FFF) << 4, ((d >> 32) & 0x3FFF) << 4
    assert d >> 62 == 1 and (d >> 49) & 7 == 0 and (d >> 16) & 0x3FFF == 1
    if SBO_OVERRIDE[0] is not None:
        sbo = SBO_OVERRIDE[0]
    r = torch.arange(rows).view(-1, 1); k = torch.arange(16).view(1, -1)
    L = start + (r // 8) * sbo + (r % 8) * 128 + 2 * k
    P = L ^ (((L >> 7) & 7) << 4)
    return P // 2


SBO_OVERRIDE = [None]
SWZ_WRITE = [True]
RING_WAIT_BUG = [False]


def cta(xb, hb, lb, m, K, cfg, n0, k0, t0, raw, late):
    KR, NSTG, NBUF, TOKS, WN, MINB = cfg
    RING = NBUF < NSTG
    QT = KR // 16; QS = QT // NSTG; CPS = 2 * QS; NSLOT = NBUF if RING else KR // 64
    ROWS = TOKS + 128; SB = ROWS * 128; NA = TOKS // 2; NH = TOKS // WN; SP = 68; LI = ROWS * CPS // 128
    assert KR % 64 == 0 and QT % NSTG == 0 and (ROWS * CPS) % 128 == 0 and (QS == 4 if RING else NBUF == NSTG)
    SM = NSLOT * SB + 1024
    base = (raw + 1023) & ~1023
    assert base - raw + NSLOT * SB <= SM and TOKS * SP * 4 <= NSLOT * SB
    smem = torch.full(((raw + SM + 1) // 2,), 0x7fc0, dtype=torch.long)  # bf16 NaN: an unwritten half read by the mma poisons the result
    tid = torch.arange(128)

    def load(s):  # -> (halves index [n, 8], values [n, 8])
        idx, val = [], []
        for it in range(LI):
            i = tid + it * 128
            r, cc = i // CPS, s * CPS + i % CPS
            c = cc & 7
            sw = ((c ^ (r & 7)) << 4) if SWZ_WRITE[0] else (c << 4)
            dst = base + ((cc >> 3) % NSLOT) * SB + r * 128 + sw
            kk = (k0 + cc * 8).view(-1, 1) + torch.arange(8).view(1, -1)
            tok = t0 + r
            xv = xb[(tok.clamp(max=m - 1)).view(-1, 1), kk] * (tok < m).long().view(-1, 1)  # src-size 0: zero fill
            hv = hb[(n0 + (r - TOKS).clamp(0, 63)).view(-1, 1), kk]
            lv = lb[(n0 + (r - TOKS - 64).clamp(0, 63)).view(-1, 1), kk]
            v = torch.where((r < TOKS).view(-1, 1), xv, torch.where((r < TOKS + 64).view(-1, 1), hv, lv))
            assert (dst % 16 == 0).all()
            idx.append(dst.view(-1, 1) // 2 + torch.arange(8).view(1, -1)); val.append(v)
        return torch.cat(idx), torch.cat(val)

    groups = []  # committed cp.async groups: [stage or None, (idx, val), landed]

    def commit(s):
        g = [s, load(s) if s is not None else None, False]
        groups.append(g)
        if not late:
            land(g)

    def land(g):
        if not g[2] and g[1] is not None:
            smem[g[1][0].reshape(-1)] = g[1][1].reshape(-1)
        g[2] = True

    def cpwait(n):
        for g in groups[: max(0, len(groups) - n)]:
            land(g)

    D = torch.zeros(64, TOKS, dtype=torch.float64)
    inflight = []  # slots read by wgmma groups not yet retired
    for s in range(NBUF if RING else NSTG):
        commit(s)
    for s in range(NSTG):
        n = (NBUF - 1 if s == 0 else NBUF - (1 if RING_WAIT_BUG[0] else 2)) if RING else NSTG - 1 - s
        cpwait(n)
        slots = set()
        for qq in range(QS):
            q = s * QS + qq
            sb = base + ((q >> 2) % NSLOT) * SB + (q & 3) * 32
            slots.add((q >> 2) % NSLOT)
            dh, dl = desc(sb + TOKS * 128), desc(sb + (TOKS + 64) * 128)
            for h in range(NH):
                db = desc(sb + h * WN * 128)
                Bm = bf(smem[desc_rows(db, WN)])  # [WN tokens, 16]
                for da in (dh, dl):
                    Am = bf(smem[desc_rows(da, 64)])  # [64 rows, 16]
                    D[:, h * WN:(h + 1) * WN] += Am @ Bm.t()
        inflight.append(slots)
        if RING and s >= 1:  # wgmma.wait_group 1: only stage s may still be reading
            inflight = inflight[-1:]
            if s - 1 + NBUF < NSTG:
                tgt = (s - 1 + NBUF) % NSLOT
                assert all(tgt not in sl for sl in inflight), "ring refill into a slot still read by wgmma"
                commit(s - 1 + NBUF)
            else:
                commit(None)
    D32 = D.float().double()  # fp32 accumulators
    # fragments -> registers per thread (PTX D layout), then the kernel's staging formula
    sd = torch.full((TOKS * SP,), float("nan"), dtype=torch.float64)
    for T in range(128):
        w, lane = T >> 5, T & 31
        g, t = lane >> 2, lane & 3
        d = [0.0] * NA
        for h in range(NH):
            for i in range(WN // 8):
                for e in range(4):
                    d[h * (WN // 2) + 4 * i + e] = float(D32[16 * w + g + 8 * (e >= 2), h * WN + 8 * i + 2 * t + (e & 1)])
        for h in range(NH):
            for i in range(WN // 8):
                tok, n = h * WN + 8 * i + 2 * t, 16 * w + g
                e = d[h * (WN // 2) + 4 * i: h * (WN // 2) + 4 * i + 4]
                sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1]
                sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3]
    rows = min(TOKS, m - t0)
    reds = []
    for i in range(rows * 16):
        r, c4 = i >> 4, i & 15
        reds.append(((t0 + r), n0 + 4 * c4, sd[r * SP + 4 * c4: r * SP + 4 * c4 + 4]))
    return reds


def emulate(x, hi, lo, cfg, raw=0, late=False):
    m, K = x.shape
    N = hi.shape[0]
    KR, NSTG, NBUF, TOKS, WN, MINB = cfg
    xb = x.view(torch.int16).long() & 0xffff
    hb, lb = hi.view(torch.int16).long() & 0xffff, lo.view(torch.int16).long() & 0xffff
    out = torch.zeros(m, N, dtype=torch.float32)  # the memset node
    assert N % 64 == 0 and K % KR == 0
    for tb in range((m + TOKS - 1) // TOKS):
        for ks in range(K // KR):
            for nt in range(N // 64):
                for tok, n, v in cta(xb, hb, lb, m, K, cfg, nt * 64, ks * KR, tb * TOKS, raw, late):
                    assert n % 4 == 0 and tok < m
                    out[tok, n:n + 4] += v.float()  # red.global.add.v4.f32 (fp32)
    return out.double()


def check(cfg, ms, hi, lo, ex, gen, K, quick=False):
    ok = True
    for m in ms:
        for late, raw in ((False, 0), (True, 400)) if not quick else ((True, 0),):
            x = torch.randn(m, K, generator=gen).to(torch.bfloat16)
            o = emulate(x, hi, lo, cfg, raw, late)
            ref = x.double() @ ex.t()
            err = float((o - ref).norm() / ref.norm())
            xo = torch.zeros(m, K); xo[torch.arange(m), torch.randint(K, (m,), generator=gen)] = 8.0
            xo = xo.to(torch.bfloat16)
            exact = torch.equal(emulate(xo, hi, lo, cfg, raw, late), (xo.double() @ ex.t()).float().double())
            print("   rwk<%s> m=%3d %s raw+%d: rel err %.1e one-hot exact %s" % (",".join(map(str, cfg)), m, "late " if late else "early", raw, err, exact), flush=True)
            ok &= err < 1e-6 and exact
    return ok


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else SUB
    V = rwv(path)
    gen = torch.Generator().manual_seed(29)
    N = 128
    ok = True
    for v, cfg in sorted(V.items()):
        K = 2 * cfg[0]
        Wf = torch.randn(N, K, generator=gen) / K ** 0.5
        hi = Wf.to(torch.bfloat16); lo = (Wf - hi.float()).to(torch.bfloat16)
        ex = hi.double() + lo.double()
        TOKS = cfg[3]
        ms = (128, 121, 200) if TOKS == 128 else (256, 250, 300)
        print(" variant %d" % v, flush=True)
        ok &= check(cfg, ms, hi, lo, ex, gen, K)
    # negative controls (each must be caught)
    cfg = V[101]; K = 512
    Wf = torch.randn(N, K, generator=gen) / K ** 0.5
    hi = Wf.to(torch.bfloat16); lo = (Wf - hi.float()).to(torch.bfloat16); ex = hi.double() + lo.double()
    for name, flag, val in (("unswizzled smem writes", SWZ_WRITE, False), ("descriptor SBO 512", SBO_OVERRIDE, 512)):
        old = flag[0]; flag[0] = val
        x = torch.randn(128, K, generator=gen).to(torch.bfloat16)
        o = emulate(x, hi, lo, cfg, 0, True)
        flag[0] = old
        ref = x.double() @ ex.t()
        err = float((o - ref).norm() / ref.norm()) if torch.isfinite(o).all() else float("inf")
        print("   negative control (%s): rel err %.2f %s" % (name, err, "caught" if err > 1e-2 else "NOT caught"))
        ok &= err > 1e-2
    cfg = V[103]; K = 1024
    Wf = torch.randn(N, K, generator=gen) / K ** 0.5
    hi = Wf.to(torch.bfloat16); lo = (Wf - hi.float()).to(torch.bfloat16); ex = hi.double() + lo.double()
    RING_WAIT_BUG[0] = True
    x = torch.randn(128, K, generator=gen).to(torch.bfloat16)
    o = emulate(x, hi, lo, cfg, 0, True)
    RING_WAIT_BUG[0] = False
    ref = x.double() @ ex.t()
    err = float((o - ref).norm() / ref.norm()) if torch.isfinite(o).all() else float("inf")
    print("   negative control (ring cp.async wait one group short, late landing): rel err %.2f %s" % (err, "caught" if err > 1e-2 else "NOT caught"))
    ok &= err > 1e-2
    print("RW EMU", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
