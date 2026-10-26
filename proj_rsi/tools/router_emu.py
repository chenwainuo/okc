"""pw21 router path (gv FMT 2 with the V-bit-25 token loop), CPU emulation of the kernel's index chain.

Per CTA (bz token block, bx row group, by K-group) and warp (wr, wk): W fragments at the kernel's offsets
(wo = ((bx*WR + wr)*RT*KC + c0)*(NL*32) + lane, NL = 4 plain bf16 pieces per unit, hi = W, lo = W2), x of token tile n
loaded at use from row min(n*8 + g, m_blk - 1), k = (c0 + u)*64 + t*16 .. +15 (b0 = first 8, b1 = next 8), mma.sync
m16n8k16 per k16 step q with A = regs 4q..4q+3 and B = (Bq[2q], Bq[2q+1]), hi and lo into the same accumulator,
fragment c0..c3 = (row g, tok 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1); in-CTA + cluster reduction = the sum over wk and
by; emit only tokens < m. Checked against x @ (hi + lo)^T in fp64 (fp32 output) and one-hot rows exactly.

python3 -m proj_rsi.tools.router_emu <submission.py>"""
import importlib.util, sys
import torch


def bf(u16):
    v = u16.long() << 16
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32).view(torch.float32).double()


def emulate(fr_hi, fr_lo, x, n, geo, MT):
    WR, WK, KU, S, RT = geo[:5]
    T, KC = fr_hi.shape[:2]
    m, K = x.shape
    TK = MT * 8
    NL = 4
    Wh = fr_hi.reshape(-1).contiguous().view(torch.int16).long() & 0xffff   # [T][KC][4 pieces][32 lanes][8 values]
    Wl = fr_lo.reshape(-1).contiguous().view(torch.int16).long() & 0xffff
    xb = x.view(torch.int16).long() & 0xffff
    lanes = torch.arange(32)
    g, t = lanes // 4, lanes % 4
    GX = T // (WR * RT)
    out = torch.full((m, n), float("nan"), dtype=torch.float64)
    for bz in range((m + TK - 1) // TK):
        mb = min(TK, m - bz * TK)
        for bx in range(GX):
            acc = torch.zeros(WR, MT, 32, 4, dtype=torch.float64)   # summed over wk and by (in-CTA + cluster reduce)
            for by in range(S):
                for w in range(WR * WK):
                    wr, wk = w % WR, w // WR
                    c0 = by * WK * KU + wk * KU
                    for u in range(KU):
                        base = (((bx * WR + wr) * RT * KC + c0 + u) * NL * 32) * 8   # uint4 index -> 16-bit index
                        A = []
                        for W_ in (Wh, Wl):
                            v = W_[base + (torch.arange(NL)[:, None, None] * 32 + lanes[None, :, None]) * 8 + torch.arange(8)[None, None, :]]
                            A.append(bf(v.permute(1, 0, 2).reshape(32, 32)))   # [lane, piece*8 + v] = step j, value v
                        for nt in range(MT):
                            row = torch.clamp(nt * 8 + g, max=mb - 1) + bz * TK
                            k0 = (c0 + u) * 64 + t * 16
                            Bv = bf(xb[row[:, None], k0[:, None] + torch.arange(16)[None, :]])   # [lane, 16]
                            for q in range(4):
                                # A regs 4q..4q+3 = values 8q .. 8q+7: (g, kk0, kk1), (g+8, kk0, kk1), (g, kk2, kk3), (g+8, kk2, kk3)
                                Am = torch.zeros(16, 16, dtype=torch.float64); Bm = torch.zeros(16, 8, dtype=torch.float64)
                                for h in range(2):
                                    a = A[h][:, 8 * q: 8 * q + 8]
                                    Am.zero_()
                                    Am[g, 2 * t] = a[:, 0]; Am[g, 2 * t + 1] = a[:, 1]; Am[g + 8, 2 * t] = a[:, 2]; Am[g + 8, 2 * t + 1] = a[:, 3]
                                    Am[g, 2 * t + 8] = a[:, 4]; Am[g, 2 * t + 9] = a[:, 5]; Am[g + 8, 2 * t + 8] = a[:, 6]; Am[g + 8, 2 * t + 9] = a[:, 7]
                                    Bm.zero_()
                                    Bm[2 * t, g] = Bv[:, 4 * q]; Bm[2 * t + 1, g] = Bv[:, 4 * q + 1]
                                    Bm[2 * t + 8, g] = Bv[:, 4 * q + 2]; Bm[2 * t + 9, g] = Bv[:, 4 * q + 3]
                                    D = Am @ Bm
                                    acc[wr, nt] += torch.stack((D[g, 2 * t], D[g, 2 * t + 1], D[g + 8, 2 * t], D[g + 8, 2 * t + 1]), 1)
            for wr in range(WR):
                for nt in range(MT):
                    tok = bz * TK + nt * 8 + 2 * t
                    r0 = (bx * WR + wr) * RT * 16 + g
                    f = acc[wr, nt].float().double()
                    for cc in range(4):
                        tk, rr = tok + (cc & 1), r0 + 8 * (cc >> 1)
                        ok = (tk - bz * TK) < mb
                        out[tk[ok], rr[ok]] = f[ok, cc]
    return out


def main():
    spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else "proj_rsi/candidates/pw21/submission.py")
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    gen = torch.Generator().manual_seed(3)
    N, K = 64, 512   # small router-like matrix (the kernel geometry is scaled: KC = 8 chunks)
    Wf = torch.randn(N, K, generator=gen) / K ** 0.5
    hi = Wf.to(torch.bfloat16); lo = (Wf - hi.float()).to(torch.bfloat16)
    fh, fl = S._pack(hi), S._pack(lo)
    exact = hi.double() + lo.double()
    ok = True
    for name, _, ms, g7, v in S.GRP:
        wr, wk, ku, s, rt = g7[1:6]
        # same warp structure, K scaled down: keep WR / WK, choose KU * S = KC / WK
        KC = K // 64
        geo = (wr, wk, max(1, KC // (wk * 2)), 2, rt)
        if geo[1] * geo[2] * geo[3] != KC:
            geo = (wr, wk, KC // wk, 1, rt)
        for m in (128, 121, 256, 250, 9):
            x = torch.randn(m, K, generator=gen).to(torch.bfloat16)
            o = emulate(fh, fl, x, N, geo, 16)
            ref = (x.double() @ exact.t()).float().double()
            err = float((o - ref).norm() / ref.norm())
            xo = torch.zeros(m, K); xo[torch.arange(m), torch.randint(K, (m,), generator=gen)] = 8.0; xo = xo.to(torch.bfloat16)
            o2 = emulate(fh, fl, xo, N, geo, 16)
            ex = torch.equal(o2, (xo.double() @ exact.t()).float().double())
            print("   %-9s geo %s m=%3d: rel err %.1e one-hot exact %s" % (name, geo, m, err, ex))
            ok &= err < 1e-6 and ex and not torch.isnan(o).any()
    print("ROUTER EMU", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
