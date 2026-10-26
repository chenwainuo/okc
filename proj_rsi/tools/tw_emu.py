"""pw19 "tw" kernel (pwt): CPU emulation of the data path and a randomized simulation of the three-ring protocol.

Data path, per CTA (nb, ks) and stage u, at the kernel's addresses:
  packed W chunk bytes (wring slot) -> per decode warp dw, row tile dw + 4j: 3 x 16 B per lane -> pw_dec -> escape fix
  (pw_fix3, seq = u) -> stmatrix.x4 per k16 step q at lane address tile*2048 + ((L>>3 & 1)*8 + (L&7))*128 +
  (((2q + (L>>4)) ^ (L&7)) << 4) (PTX stmatrix: row i of matrix j comes from lane 8j + i's address; thread T holds row
  T/4, columns 2(T%4), +1 in register j) -> A slot smem image
  x tile smem image as TMA SWIZZLE_128B writes it (row r, 16-B chunk c at r*128 + ((c ^ (r&7)) << 4); rows >= m zero)
  wgmma SS m64nNk16 read through the descriptors (start + 32 q, SBO 1024, 128B swizzle on absolute address bits:
  phys = logical ^ (((logical >> 7) & 7) << 4)), warpgroup wg at A slot + 8192 wg
  D (PTX D layout) -> fragment staging -> cluster rank rectangles summed over the KS peers -> transpose -> out.
Negative controls: the r6 (permuted-k) pack, and stmatrix without the XOR swizzle, must fail.

Protocol: producer lane (polls the W and x empty barriers, issues whichever is free), 4 decode warps (wait W full,
wait A empty, write the A slot, proxy fence, arrive W empty + A full x 32), NWM MMA warps (wait A full + x full, issue
wgmma = asynchronous reads of the A and x slots, wait_group(1) retires chunk u-1 then releases its A and x slots, final
wait_group(0)); copies land at random times. Checks: no deadlock; no W slot refilled before all decode warps read it;
no A slot rewritten / x slot refilled while a wgmma reads it; every wgmma reads complete, fenced, correct-stage A and x
tiles. Mutations that must be caught: "early_release" (release right after issuing the wgmma) and "no_fence" (A full
arrive without fence.proxy.async).

python3 -m proj_rsi.tools.tw_emu <submission.py>"""
import importlib.util, math, random, sys
import torch

M32 = 0xffffffff


def prmt(a, b, sel):
    by = [(a >> (8 * i)) & 255 for i in range(4)] + [(b >> (8 * i)) & 255 for i in range(4)]
    r = torch.zeros_like(a)
    for k in range(4):
        r |= by[(sel >> (4 * k)) & 7] << (8 * k)
    return r


def dec(L0, L1, C, Tb):
    rot = ((C >> 4) | (C << 28)) & M32
    hl = ((C & 0x87878787) + Tb) & M32
    hh = ((rot & 0x87878787) + Tb) & M32
    return [prmt(L0, hl, 0x5140), prmt(L0, hl, 0x7362), prmt(L1, hh, 0x5140), prmt(L1, hh, 0x7362)]


def bf(u16):
    v = u16.long() << 16
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32).view(torch.float32).double()


def words16(by):  # [32 lanes, 16 bytes] -> [32, 4] u32
    wd = by.view(32, 4, 4)
    return wd[..., 0] | (wd[..., 1] << 8) | (wd[..., 2] << 16) | (wd[..., 3] << 24)


def stsm_addr(L, q, swz=True):
    mi, rr = L >> 3, L & 7
    c = 2 * q + (mi >> 1)
    return ((mi & 1) * 8 + rr) * 128 + (((c ^ rr) if swz else c) << 4)


def stmatrix(img, base, regs, q, swz=True):
    """img: int64 tensor of bf16 halves indexed by byte address / 2; regs: [32 lanes, 4] u32 (matrices 0..3)"""
    for j in range(4):
        for i in range(8):  # row i of matrix j: threads 4i..4i+3, register j, 2 elements each
            a = base + stsm_addr(8 * j + i, q, swz)
            for t in range(4):
                v = int(regs[4 * i + t, j])
                img[(a + 4 * t) // 2] = v & 0xffff
                img[(a + 4 * t) // 2 + 1] = v >> 16


def desc_read(img, start, rows):
    """canonical K-major SW128 operand of one k16 step: [rows, 16] bf16 bits from start (= base + 32 q)"""
    out = torch.zeros(rows, 16, dtype=torch.long)
    for r in range(rows):
        for k in range(16):
            logical = start + (r // 8) * 1024 + (r % 8) * 128 + 2 * k
            phys = logical ^ (((logical >> 7) & 7) << 4)
            out[r, k] = img[phys // 2]
    return out


def tma_tile(xb, m, NT):
    """x[:, 64-k chunk] as bf16 bits [m, 64] -> smem image (halves) of the 128B-swizzled box of NT rows"""
    img = torch.zeros(NT * 64, dtype=torch.long)
    for r in range(min(m, NT)):
        for c in range(8):
            a = r * 128 + ((c ^ (r & 7)) << 4)
            img[a // 2: a // 2 + 8] = xb[r, 8 * c: 8 * c + 8]
    return img


def emulate(S, x, n, cfg, KS, lay, tb, esc, eoff, m, swz=True):
    NWG, NT = cfg[:2]
    BN, RTB, TT, NWD = 64 * NWG, 4 * NWG, NT // 8, 4
    NJ = RTB // NWD
    K = x.shape[1]
    KCW, NB = lay.shape[1], n // BN
    FRAG = RTB * TT * 32
    Wf = lay.reshape(-1).long()
    escu = esc.long() & M32
    xb = x.contiguous().view(torch.int16).long() & 0xffff
    lanes = torch.arange(32)
    g, t = lanes // 4, lanes % 4
    out = torch.full((m, n), float("nan"), dtype=torch.float64)
    for nb in range(NB):
        stg = {}
        for ks in range(KS):
            s0, s1 = ks * KCW // KS, (ks + 1) * KCW // KS
            D = torch.zeros(BN, NT, dtype=torch.float64)
            st = {}
            for dw in range(NWD):
                for j in range(NJ):
                    tile = dw + NWD * j
                    ep = int(eoff[(ks * NB + nb) * RTB + tile]); st[tile] = [ep, int(escu[ep])]
            for u in range(s1 - s0):
                c = s0 + u
                aimg = torch.zeros(BN * 64, dtype=torch.long)  # A slot (BN rows x 128 B) as bf16 halves
                for dw in range(NWD):
                    for j in range(NJ):
                        tile = dw + NWD * j
                        base = ((nb * KCW + c) * RTB + tile) * 1536
                        q = [words16(Wf[base + p * 512 + lanes[:, None] * 16 + torch.arange(16)[None]]) for p in range(3)]
                        A = torch.stack(dec(q[0][:, 0], q[0][:, 1], q[2][:, 0], tb) + dec(q[0][:, 2], q[0][:, 3], q[2][:, 1], tb) +
                                        dec(q[1][:, 0], q[1][:, 1], q[2][:, 2], tb) + dec(q[1][:, 2], q[1][:, 3], q[2][:, 3], tb), 1)
                        ep, nx = st[tile]
                        while (nx >> 17) == u:  # pw_fix3 with roff 0
                            ln, r = (nx >> 12) & 31, (nx >> 8) & 15
                            A[ln, r] ^= (nx & 0x7f) << (8 + ((nx >> 7) & 1) * 16)
                            ep += 1; nx = int(escu[ep])
                        st[tile] = [ep, nx]
                        for qq in range(4):
                            stmatrix(aimg, tile * 2048, A[:, 4 * qq: 4 * qq + 4], qq, swz)
                kx = (c * 64) % K
                ximg = tma_tile(xb[:, kx: kx + 64], m, NT)
                for wg in range(NWG):
                    for qq in range(4):
                        Am = bf(desc_read(aimg, wg * 8192 + 32 * qq, 64))   # [64 rows, 16 k]
                        Bm = bf(desc_read(ximg, 32 * qq, NT))               # [NT tokens, 16 k]
                        D[64 * wg: 64 * wg + 64] += Am @ Bm.t()
            # MMA warp w = tile: D layout -> fragment order (kernel: stg + ((tile*TT + j)*32 + lane)*16)
            fr = torch.zeros(FRAG, 4, dtype=torch.float64)
            for tile in range(RTB):
                for jj in range(TT):
                    idx = (tile * TT + jj) * 32 + lanes
                    R = tile * 16
                    fr[idx] = torch.stack((D[R + g, 8 * jj + 2 * t], D[R + g, 8 * jj + 2 * t + 1], D[R + g + 8, 8 * jj + 2 * t], D[R + g + 8, 8 * jj + 2 * t + 1]), 1)
            stg[ks] = fr
        for r in range(KS):  # rank r's rectangle, summed over the peers, transposed, 16-B row stores
            if KS <= RTB:
                ntile, tile0, tt0, ntt = RTB // KS, r * (RTB // KS), 0, TT
            else:
                sp = KS // RTB; ntile, tile0, ntt = 1, r // sp, TT // sp; tt0 = (r % sp) * ntt
            rows = ntile * 16
            tbuf = torch.full((ntt * 8, rows), float("nan"), dtype=torch.float64)
            for e in range(ntile * ntt * 32):
                l, jj, ti = e & 31, (e >> 5) % ntt, (e >> 5) // ntt
                idx = ((tile0 + ti) * TT + tt0 + jj) * 32 + l
                tok = (tt0 + jj) * 8 + 2 * (l & 3)
                if tok < m:
                    a = sum(stg[q][idx] for q in range(KS))
                    rl, tl = ti * 16 + (l >> 2), jj * 8 + 2 * (l & 3)
                    for cc in range(4):
                        if tok + (cc & 1) < m:
                            tbuf[tl + (cc & 1), rl + 8 * (cc >> 1)] = float(a[cc].to(torch.bfloat16).double())
            segs = rows * 2 // 16
            for e in range(ntt * 8 * segs):
                tl, sg = e // segs, e % segs
                tok = tt0 * 8 + tl
                if tok < m:
                    seg = tbuf[tl, sg * 8: sg * 8 + 8]
                    assert not torch.isnan(seg).any(), "row piece with an unwritten element"
                    out[tok, nb * BN + tile0 * 16 + sg * 8: nb * BN + tile0 * 16 + sg * 8 + 8] = seg
    return out


def run(S, iname, n, K, m, ks, seed=0, perm_pack=False, swz=True):
    idx = S.PW_IDX[iname]
    cfg = S.PW_INSTS[idx][2]
    NWG, NT = cfg[:2]
    BN, RTB, TT = 64 * NWG, 4 * NWG, NT // 8
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(n, K, generator=gen) / math.sqrt(K)
    flat = W.view(-1); pos = torch.randperm(flat.numel(), generator=gen)[:40]
    flat[pos[:14]] = 0.0; flat[pos[14:28]] *= 1e-7; flat[pos[28:]] *= 2000.0   # escapes
    W = W.to(torch.bfloat16)
    st = {"raw": {"t": W}}
    if perm_pack:
        st.setdefault("pw_c12nat", {})["t"] = S._pw_c12(st, "t")
    data, tb, esc6 = S._pw_c12(st, "t", nat=True)
    lay = S._pw_layout(st, "t", RTB, nat=True)
    KC, NB = K // 64, n // BN
    esc, eoff = S._pw_esc(esc6, n, (1, RTB, 1, 1), KC, NB, ks, "cpu")
    x = torch.randn(m, K, generator=gen).to(torch.bfloat16)
    o = emulate(S, x, n, cfg, ks, lay, tb, esc, eoff, m, swz)
    ref = (x.double() @ W.double().t()).to(torch.bfloat16).double()
    err = float((o - ref).norm() / ref.norm())
    kk = esc6[:, 1] * 64 + esc6[:, 2] * 16 + (esc6[:, 4] // 4) * 8 + (esc6[:, 3] % 4) * 2 + esc6[:, 4] % 2  # natural k of escapes
    cols = torch.randint(K, (m,), generator=gen)
    cols[: min(m, len(kk))] = kk[: min(m, len(kk))]
    xo = torch.zeros(m, K); xo[torch.arange(m), cols] = 8.0; xo = xo.to(torch.bfloat16)
    o2 = emulate(S, xo, n, cfg, ks, lay, tb, esc, eoff, m, swz)
    exact = torch.equal(o2, (xo.double() @ W.double().t()).to(torch.bfloat16).double())
    tag = (" [r6 pack]" if perm_pack else "") + ("" if swz else " [no stmatrix swizzle]")
    print("   %-6s%s n=%d K=%d m=%d KS=%d: rel err %.1e one-hot exact %s (%d escapes)" % (iname, tag, n, K, m, ks, err, exact, len(esc6)))
    return err < 1e-2 and exact


# ---------------------------------------------------------------- protocol simulation
class Bar:
    def __init__(self, c):
        self.c, self.p, self.done, self.tx = c, c, 0, 0

    def arrive(self, tx=0):
        self.tx += tx; self.p -= 1
        assert self.p >= 0, "arrival overflow"
        self._k()

    def ctx(self, b):
        self.tx -= b; self._k()

    def _k(self):
        if self.p == 0 and self.tx == 0:
            self.done += 1; self.p = self.c

    def ready(self, par):
        return (self.done & 1) != par


def sim(WST, XST, AST, NWM, KCS, seed, bug=None, NWD=4):
    rnd = random.Random(seed)
    wf, we = [Bar(1) for _ in range(WST)], [Bar(NWD) for _ in range(WST)]
    xf, xe = [Bar(1) for _ in range(XST)], [Bar(NWM) for _ in range(XST)]
    af, ae = [Bar(NWD * 32) for _ in range(AST)], [Bar(NWM) for _ in range(AST)]
    wcont, wread = [None] * WST, [set() for _ in range(WST)]
    xcont, xread = [None] * XST, [set() for _ in range(XST)]
    acont = [dict() for _ in range(AST)]          # tile-writer -> (stage, fenced)
    aread = [set() for _ in range(AST)]
    inflight = []
    prod = {"ws": 0, "xs": 0}

    def issue_w(s):
        sl = s % WST
        assert wcont[sl] is None or len(wread[sl]) == NWD, "W slot refilled before every decode warp read it"
        wcont[sl] = None; wread[sl] = set()
        wf[sl].arrive(tx=1); inflight.append(("w", sl, s))

    def issue_x(s):
        sl = s % XST
        assert not xread[sl], "x slot refilled while a wgmma reads it"
        xcont[sl] = None
        xf[sl].arrive(tx=1); inflight.append(("x", sl, s))

    for s in range(min(WST, KCS)):
        issue_w(s)
    for s in range(min(XST, KCS)):
        issue_x(s)
    prod["ws"], prod["xs"] = min(WST, KCS), min(XST, KCS)
    dec = [{"u": 0, "st": "w"} for _ in range(NWD)]
    mma = [{"u": 0, "st": "in", "pend": []} for _ in range(NWM)]
    while True:
        acts = []
        ws, xs = prod["ws"], prod["xs"]
        if ws < KCS and we[ws % WST].ready(((ws // WST) - 1) & 1):
            acts.append(("pw",))
        if xs < KCS and xe[xs % XST].ready(((xs // XST) - 1) & 1):
            acts.append(("px",))
        for i, d in enumerate(dec):
            u = d["u"]
            if u == KCS:
                continue
            if d["st"] == "w" and wf[u % WST].ready((u // WST) & 1):
                acts.append(("dw", i))
            elif d["st"] == "a" and (u < AST or ae[u % AST].ready(((u // AST) - 1) & 1)):
                acts.append(("da", i))
            elif d["st"] == "f":
                acts.append(("df", i))
        for i, k in enumerate(mma):
            u = k["u"]
            if k["st"] == "in" and u < KCS and af[u % AST].ready((u // AST) & 1) and xf[u % XST].ready((u // XST) & 1):
                acts.append(("mi", i))
            elif k["st"] == "wait":
                acts.append(("mw", i))
            elif k["st"] == "in" and u == KCS and k["pend"]:
                acts.append(("mf", i))
        if inflight:
            acts.append(("land",))
        if not acts:
            break
        a = rnd.choice(acts)
        if a[0] == "land":
            ring, sl, s = inflight.pop(rnd.randrange(len(inflight)))
            if ring == "w":
                wcont[sl] = s; wf[sl].ctx(1)
            else:
                xcont[sl] = s; xf[sl].ctx(1)
        elif a[0] == "pw":
            issue_w(prod["ws"]); prod["ws"] += 1
        elif a[0] == "px":
            issue_x(prod["xs"]); prod["xs"] += 1
        elif a[0] == "dw":
            d = dec[a[1]]; u = d["u"]
            assert wcont[u % WST] == u, "decode of a W slot not holding this stage"
            d["st"] = "a"
        elif a[0] == "da":  # A slot free: write this warp's tiles (generic proxy), then fence (or not)
            d = dec[a[1]]; u = d["u"]; sl = u % AST
            assert not aread[sl], "A slot rewritten while a wgmma reads it"
            acont[sl][a[1]] = (u, False)
            wread[u % WST].add(a[1]); we[u % WST].arrive()   # packed chunk dead once decoded
            d["st"] = "f"
        elif a[0] == "df":
            d = dec[a[1]]; u = d["u"]; sl = u % AST
            if bug != "no_fence":
                acont[sl][a[1]] = (u, True)
            for _ in range(32):
                af[sl].arrive()
            d["u"] += 1; d["st"] = "w"
        elif a[0] == "mi":  # issue the wgmma of chunk u: asynchronous reads of A and x
            k = mma[a[1]]; u = k["u"]
            asl, xsl = u % AST, u % XST
            assert len(acont[asl]) == NWD and all(v == (u, True) for v in acont[asl].values()), "wgmma reads an incomplete or unfenced A tile"
            assert xcont[xsl] == u, "wgmma reads an x slot not holding this stage"
            aread[asl].add((a[1], u)); xread[xsl].add((a[1], u))
            k["pend"].append(u)
            if bug == "early_release":
                ae[asl].arrive(); xe[xsl].arrive()
            k["u"] += 1; k["st"] = "wait"
        elif a[0] == "mw":  # wait_group(1): every chunk but the newest retired and released
            k = mma[a[1]]
            while len(k["pend"]) > 1:
                v = k["pend"].pop(0)
                aread[v % AST].discard((a[1], v)); xread[v % XST].discard((a[1], v))
                if bug != "early_release":
                    ae[v % AST].arrive(); xe[v % XST].arrive()
            k["st"] = "in"
        elif a[0] == "mf":  # wait_group(0)
            k = mma[a[1]]
            for v in k["pend"]:
                aread[v % AST].discard((a[1], v)); xread[v % XST].discard((a[1], v))
                if bug != "early_release":
                    ae[v % AST].arrive(); xe[v % XST].arrive()
            k["pend"] = []
    done = all(d["u"] == KCS for d in dec) and all(k["u"] == KCS and not k["pend"] for k in mma) and prod["ws"] == KCS and prod["xs"] == KCS
    assert done, "deadlock"


def protocol(S):
    ok = True
    for iname in ("t64", "t64x2", "t128", "v64"):
        cfg = S.PW_INSTS[S.PW_IDX[iname]][2]
        NWG, NT, WST, XST, AST = cfg[:5]
        good = 0
        for seed in range(150):
            for KCS in (1, 2, 5, 17, 64):
                try:
                    sim(WST, XST, AST, 4 * NWG, KCS, seed); good += 1
                except AssertionError as e:
                    print("   %s seed %d KCS %d: %s" % (iname, seed, KCS, e)); ok = False
        for bug in ("early_release", "no_fence"):
            caught = 0
            for seed in range(150):
                try:
                    sim(WST, XST, AST, 4 * NWG, 17, seed, bug)
                except AssertionError:
                    caught += 1
            print("   protocol %-6s mutation %-14s caught in %d/150 schedules" % (iname, bug, caught))
            ok &= caught > 0
        print("   protocol %-6s %d/750 random schedules clean" % (iname, good))
    return ok


def main():
    S = None
    spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else "proj_rsi/candidates/pw19/submission.py")
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    ok = True
    ok &= run(S, "t64", 128, 512, 128, 2)          # BN 64, KS 2 (cluster pair)
    ok &= run(S, "t64", 64, 512, 121, 4)           # odd m, KS 4 (rectangles of one row tile)
    ok &= run(S, "t64", 64, 256, 128, 1)           # KS 1
    ok &= run(S, "t64x2", 128, 512, 97, 8)         # KS 8 > RTB: tile x token-range rectangles, odd m
    ok &= run(S, "t128", 128, 512, 128, 4)         # BN 128: two MMA warpgroups, decode warps own tiles dw, dw + 4
    ok &= run(S, "t128", 128, 256, 101, 2)
    ok &= run(S, "t128", 256, 512, 128, 2)         # tw128c2: BN 128 at cluster 2 (pooler 32x2 shape class)
    ok &= run(S, "v64", 64, 512, 256, 2)           # m = 256
    ok &= run(S, "v64", 64, 256, 200, 1)
    neg1 = not run(S, "t64", 64, 256, 128, 1, perm_pack=True)
    neg2 = not run(S, "t64", 64, 256, 128, 1, swz=False)
    print("   negative controls (must fail): r6 pack %s, unswizzled stmatrix %s" % ("caught" if neg1 else "NOT caught", "caught" if neg2 else "NOT caught"))
    ok &= neg1 and neg2
    ok &= protocol(S)
    print("TW EMU", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
