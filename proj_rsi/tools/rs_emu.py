"""pw22 router split-K kernel (rsk): CPU emulation of its smem / ldmatrix / mma.sync index chain and both reductions.

Per CTA (n tile, K split, 128-token block): cp.async image of x rows (zero past m) and hi / lo rows, 16-B chunk c of row
r at r*KR*2 + ((c ^ (r & 7)) << 4); ldmatrix.x4 per PTX (lane L addresses row L & 7 of matrix L >> 3; thread T gets
row T / 4, columns 2 (T % 4), +1 of matrix j in register j); mma.sync m16n8k16 per PTX fragment layouts; red: sum over
K splits; last-arriver: fragment slots (w*2 + mi)*NT8 + j, decoded back by the kernel's index formulas. Compared with
x @ (hi + lo)^T (fp64 -> fp32), one-hot rows exactly, odd m. python3 -m proj_rsi.tools.rs_emu"""
import torch


def bf(u):
    v = u.long() << 16
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32).view(torch.float32).double()


def smem_image(rows, KR):
    """rows: [R, KR] int64 bf16 bits -> flat halves image with the XOR swizzle"""
    R = rows.shape[0]
    img = torch.zeros(R * KR, dtype=torch.long)
    for r in range(R):
        for c in range(KR // 8):
            p = r * KR + ((c ^ (r & 7)) << 3)  # in halves: (c ^ (r&7)) * 8
            img[p: p + 8] = rows[r, 8 * c: 8 * c + 8]
    return img


def ldm4(img, addr_halves):
    """addr_halves[L] = row start (in halves) supplied by lane L -> regs[T][j] = (lo, hi) halves of matrix j row T/4"""
    regs = torch.zeros(32, 4, 2, dtype=torch.long)
    for j in range(4):
        for T in range(32):
            a = addr_halves[8 * j + T // 4] + 2 * (T % 4)
            regs[T, j, 0], regs[T, j, 1] = img[a], img[a + 1]
    return regs


def emulate(x, hi, lo, BN, KR, red, TOKS=128):
    m, K = x.shape
    N = hi.shape[0]
    TOK, NT8, WN, MI = TOKS, BN // 16, BN // 2, TOKS // 64
    xb = x.view(torch.int16).long() & 0xffff
    hb, lb = hi.view(torch.int16).long() & 0xffff, lo.view(torch.int16).long() & 0xffff
    lanes = torch.arange(32); g, t = lanes // 4, lanes % 4
    out = torch.zeros(m, N, dtype=torch.float64)
    TB, KS, NTL = (m + TOK - 1) // TOK, K // KR, N // BN
    for tb in range(TB):
        for nt in range(NTL):
            slots = []
            for ks in range(KS):
                rows = torch.zeros(TOK, KR, dtype=torch.long)
                for r in range(TOK):
                    if tb * TOK + r < m:
                        rows[r] = xb[tb * TOK + r, ks * KR: (ks + 1) * KR]
                sx = smem_image(rows, KR)
                sh = smem_image(hb[nt * BN:(nt + 1) * BN, ks * KR:(ks + 1) * KR], KR)
                sl = smem_image(lb[nt * BN:(nt + 1) * BN, ks * KR:(ks + 1) * KR], KR)
                part = {}
                for w in range(8):
                    wt, wn = w & 3, w >> 2
                    acc = torch.zeros(MI, NT8, 32, 4, dtype=torch.float64)
                    for q in range(KR // 16):
                        A = []
                        for mi in range(MI):  # rsk2: mi = (128-token block, m16)
                            r = (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + (lanes & 15); c = 2 * q + (lanes >> 4)
                            A.append(ldm4(sx, (r * KR + ((c ^ (r & 7)) << 3)).tolist()))
                        for img in (sh, sl):
                            for nb in range(NT8 // 2):
                                bj = lanes >> 3
                                r = wn * WN + nb * 16 + (bj >> 1) * 8 + (lanes & 7); c = 2 * q + (bj & 1)
                                Bv = ldm4(img, (r * KR + ((c ^ (r & 7)) << 3)).tolist())
                                for mi in range(MI):
                                    a = A[mi]
                                    Am = torch.zeros(16, 16, dtype=torch.float64)
                                    Am[g, 2 * t], Am[g, 2 * t + 1] = bf(a[:, 0, 0]), bf(a[:, 0, 1])
                                    Am[g + 8, 2 * t], Am[g + 8, 2 * t + 1] = bf(a[:, 1, 0]), bf(a[:, 1, 1])
                                    Am[g, 2 * t + 8], Am[g, 2 * t + 9] = bf(a[:, 2, 0]), bf(a[:, 2, 1])
                                    Am[g + 8, 2 * t + 8], Am[g + 8, 2 * t + 9] = bf(a[:, 3, 0]), bf(a[:, 3, 1])
                                    for half in range(2):
                                        Bm = torch.zeros(16, 8, dtype=torch.float64)
                                        Bm[2 * t, g], Bm[2 * t + 1, g] = bf(Bv[:, 2 * half, 0]), bf(Bv[:, 2 * half, 1])
                                        Bm[2 * t + 8, g], Bm[2 * t + 9, g] = bf(Bv[:, 2 * half + 1, 0]), bf(Bv[:, 2 * half + 1, 1])
                                        D = Am @ Bm
                                        acc[mi, 2 * nb + half] += torch.stack((D[g, 2 * t], D[g, 2 * t + 1], D[g + 8, 2 * t], D[g + 8, 2 * t + 1]), 1)
                    part[w] = acc.float().double()
                slots.append(part)
            if red in (0, 2):  # red.add into a zeroed output (2: into the workspace, then the completing CTA copies rows < m)
                for part in slots:
                    for w, acc in part.items():
                        wt, wn = w & 3, w >> 2
                        for mi in range(MI):
                            for j in range(NT8):
                                tok = tb * TOK + (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + g; n = nt * BN + wn * WN + j * 8 + 2 * t
                                for e, (dt, dn) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
                                    ok = tok + dt < m
                                    out[(tok + dt)[ok], (n + dn)[ok]] += acc[mi, j][ok, e]
            else:  # last arriver: slot index (w*2 + mi)*NT8 + j, lane; decode as the kernel does
                NF = 2 * NT8 * 32 * 8
                tot = torch.zeros(NF, 4, dtype=torch.float64)
                for part in slots:
                    flat = torch.zeros(NF, 4, dtype=torch.float64)
                    for w, acc in part.items():
                        for mi in range(2):
                            for j in range(NT8):
                                flat[((w * 2 + mi) * NT8 + j) * 32 + lanes] = acc[mi, j]
                    tot += flat
                for i in range(NF):
                    l, f = i & 31, i >> 5
                    j, mi, ww = f % NT8, (f // NT8) & 1, f // NT8 // 2
                    tok = tb * TOK + (ww & 3) * 32 + mi * 16 + (l >> 2); n = nt * BN + (ww >> 2) * WN + j * 8 + 2 * (l & 3)
                    for e, (dt, dn) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
                        if tok + dt < m:
                            out[tok + dt, n + dn] = tot[i, e]
    return out.float().double()


def main():
    gen = torch.Generator().manual_seed(11)
    ok = True
    N, K = 128, 512
    Wf = torch.randn(N, K, generator=gen) / K ** 0.5
    hi = Wf.to(torch.bfloat16); lo = (Wf - hi.float()).to(torch.bfloat16)
    ex = hi.double() + lo.double()
    for BN, KR, red, TOKS, ms in ((64, 256, 0, 128, (128, 121, 200)), (64, 128, 0, 128, (128, 121)), (64, 256, 2, 128, (128, 121, 200)),
                                  (64, 256, 0, 256, (256, 250, 130)), (64, 128, 0, 256, (256, 250)), (64, 256, 2, 256, (256, 199)),
                                  (64, 256, 1, 128, (128, 121))):
        for m in ms:
            x = torch.randn(m, K, generator=gen).to(torch.bfloat16)
            o = emulate(x, hi, lo, BN, KR, red, TOKS)
            ref = x.double() @ ex.t()
            err = float((o - ref).norm() / ref.norm())
            xo = torch.zeros(m, K); xo[torch.arange(m), torch.randint(K, (m,), generator=gen)] = 8.0; xo = xo.to(torch.bfloat16)
            o2 = emulate(xo, hi, lo, BN, KR, red, TOKS)
            exact = torch.equal(o2, (xo.double() @ ex.t()).float().double())
            print("   BN %3d KR %3d TOKS %d %s m=%3d: rel err %.1e one-hot exact %s" % (BN, KR, TOKS, ("red ", "last", "self")[red], m, err, exact))
            ok &= err < 1e-6 and exact
    # negative control: no swizzle on the write side
    global smem_image
    good = smem_image
    smem_image = lambda rows, KR: good(rows, KR) if False else torch.cat([rows[r] for r in range(rows.shape[0])])
    x = torch.randn(128, K, generator=gen).to(torch.bfloat16)
    err = float((emulate(x, hi, lo, 64, 256, 0) - x.double() @ ex.t()).norm() / (x.double() @ ex.t()).norm())
    smem_image = good
    print("   negative control (unswizzled smem writes): rel err %.2f %s" % (err, "caught" if err > 1e-2 else "NOT caught"))
    ok &= err > 1e-2
    print("RS EMU", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
