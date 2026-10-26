"""Instruction-level CPU emulation of pw17's FMT-4 gv data path (one torch lane per device lane, all warps at once):
plane loads at the kernel's wo / js offsets, the per-lane cp.async ring (ce_issue, commit, wait_group) with two landing
models (early: data lands at issue; late: slot is poisoned at issue and filled only when wait_group guarantees it),
ce_refill / ce_unit with generic PRMT and LOP3 (immLut taken from the kernel source), the LUT read from the smem image
the kernel builds (single or lane-replicated), and r6's escape walk with the 8-bit XOR field. Compares every A register
against the bf16 fragments. Usage: python3 -m proj_rsi.tools.codec_e_emu <submission.py> [--small] (every GEP path of a pw17-style submission;
--small uses 256-row Gaussian matrices, default the reference weights of muse_bf16_proj/reference.py, seed 5151)."""
import re, torch
M32 = 0xffffffff


def lop3(a, b, c, imm):
    r = torch.zeros_like(a)
    for idx in range(8):
        if (imm >> idx) & 1:
            ta = a if idx & 4 else (~a) & M32
            tb = b if idx & 2 else (~b) & M32
            tc = c if idx & 1 else (~c) & M32
            r |= ta & tb & tc
    return r


def prmt(a, b, sel):
    bytes_ = [(a >> (8 * k)) & 255 for k in range(4)] + [(b >> (8 * k)) & 255 for k in range(4)]
    r = torch.zeros_like(a)
    for k in range(4):
        r |= bytes_[(sel >> (4 * k)) & 7] << (8 * k)  # no sign-replicate mode in the selectors used
    return r


def fsr(lo, hi, n):  # __funnelshift_r(lo, hi, n): n & 31
    n = n & 31
    return (((hi << 32) | lo) >> n) & M32


def fsl(lo, hi, n):  # __funnelshift_l(lo, hi, n): high word of (hi:lo) << n
    n = n & 31
    return ((((hi << 32) | lo) << n) >> 32) & M32


def emulate(fr, pk, geo, ed, rep, fixed, imm, issue_per_unit, late, pfix=None):
    T, KC = fr.shape[:2]
    WR, WK, KU, S, RT = geo
    NW, GX, NL = WR * WK, T // (WR * RT), 2
    nseg = GX * S * NW
    gw = torch.arange(nseg)
    w_, g2 = gw % NW, gw // NW
    bx, by = g2 % GX, g2 // GX
    wr_, wk_ = w_ % WR, w_ // WR
    lane = torch.arange(32)[None, :]
    c0 = (by * WK * KU + wk_ * KU)[:, None]
    tb = ((bx * WR + wr_) * RT)[:, None]
    plane = pk["plane"].contiguous().view(torch.int32).long().flatten() & M32           # words
    words = pk["stream"].long() & M32
    lut = pk["lut"].long() & M32
    sm_lut = lut.repeat_interleave(32) if rep else lut                                  # the kernel's smem image
    if fixed:
        cP = torch.full((nseg, 1), pfix); s0 = gw[:, None] * pfix * 32
    else:
        so = pk["soff"].long() & M32
        s0 = so[:-1][:, None]; cP = ((so[1:] - so[:-1]) >> 5)[:, None]
    P = cP.expand(nseg, 32)
    ring = torch.randint(0, 2 ** 32, (nseg, 32, ed * 4), dtype=torch.long)
    poison = lambda shape: torch.randint(0, 2 ** 32, shape, dtype=torch.long)
    lo = torch.zeros(nseg, 32, dtype=torch.long); hi = lo.clone(); V = lo.clone(); pb = lo.clone(); iss = lo.clone()
    groups = []

    def issue():
        nonlocal iss
        go = (iss < P) & (iss < (pb >> 4) + ed)
        gi = ((s0 + iss * 32 + lane) * 4).clamp(max=words.numel() - 4)
        data = torch.stack([words[gi + k] for k in range(4)], -1)
        slot = iss % ed
        g = (go, slot, data)
        idx = (slot * 4)[..., None] + torch.arange(4)
        if late:
            cur = ring.gather(2, idx)
            ring.scatter_(2, idx, torch.where(go[..., None], poison(cur.shape), cur))
            groups.append([g, False])
        else:
            cur = ring.gather(2, idx)
            ring.scatter_(2, idx, torch.where(go[..., None], data, cur))
            groups.append([g, True])
        iss = iss + go.long()

    def wait(n):
        for k in range(len(groups) - n):
            if not groups[k][1]:
                go, slot, data = groups[k][0]
                idx = (slot * 4)[..., None] + torch.arange(4)
                cur = ring.gather(2, idx)
                ring.scatter_(2, idx, torch.where(go[..., None], data, cur))
                groups[k][1] = True

    esc = pk["esc"].long() & M32
    E = esc.view(nseg, 32)
    ep = torch.zeros(nseg, dtype=torch.long)
    nx = E[:, 0].clone()
    ok = True
    ref = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).long() & 0xffff
    for _ in range(ed - 1):
        issue()
    for u in range(KU):
        if not issue_per_unit:
            issue(); wait(ed - 2)
        for j in range(RT):
            if issue_per_unit:
                issue(); wait(ed - 2)
            t = tb + j
            c = c0 + u
            wo = ((t * KC + c) * (NL * 32) + lane)                                      # uint4 index (j * js folded in t)
            q0 = torch.stack([plane[wo * 4 + k] for k in range(4)], -1)
            q1 = torch.stack([plane[(wo + 32) * 4 + k] for k in range(4)], -1)
            L = [q0[..., k] for k in range(4)] + [q1[..., k] for k in range(4)]
            A = []
            for jj in range(4):
                need = (V & 0x60) == 0
                wd = ring.gather(2, ((pb & (16 * ed - 1)) >> 2)[..., None])[..., 0]
                wd = torch.where(need, wd, 0)
                at = V & 31
                lo = lo | ((wd << at) & M32)
                hi = hi | fsl(wd, torch.zeros_like(wd), at)
                V = (V + need.long() * 32) & M32
                pb = (pb + need.long() * 4) & M32
                for i in range(4):
                    x = lo & 0xff
                    e = sm_lut[x * 32 + lane] if rep else sm_lut[x]
                    lo, hi = fsr(lo, hi, e), fsr(hi, torch.zeros_like(hi), e)
                    V = (V - e) & M32
                    A.append(lop3(e, torch.full_like(e, 0x807f807f), prmt(L[2 * jj + (i >> 1)], torch.zeros_like(e), 0x3322 if i & 1 else 0x1100), imm))
            A = torch.stack(A, -1)                                                       # [nseg, 32, 16]
            # escape walk (warp-uniform nx per segment)
            seq = u * RT + j
            while True:
                hit = (nx >> 20) == seq
                if not bool(hit.any()):
                    break
                ln = (nx >> 15) & 31
                r = (nx >> 11) & 15
                xr = (nx & 0xff) << (7 + ((nx >> 10) & 1) * 16)
                sel = hit[:, None, None] & (lane[..., None] == ln[:, None, None]) & (torch.arange(16)[None, None, :] == r[:, None, None])
                A = torch.where(sel, A ^ xr[:, None, None], A)
                ep = torch.where(hit, ep + 1, ep)
                nx = torch.where(hit, torch.where(ep < 32, E.gather(1, ep.clamp(max=31)[:, None])[:, 0], M32), nx)
            want = ref[t.expand(nseg, 32), c.expand(nseg, 32)]                           # [nseg, 32(l?)...]
            # ref[t, c] is [4 j, 32 lane, 8 v] -> per lane: A[4 j + i] = v(2i) | v(2i+1) << 16
            want = ref[t[:, 0], c[:, 0]]                                                 # [nseg, 4, 32, 8]
            want = want.permute(0, 2, 1, 3).reshape(nseg, 32, 16, 2)
            want = want[..., 0] | (want[..., 1] << 16)
            if not torch.equal(A, want):
                bad = (A != want).nonzero()
                print("   mismatch at u=%d j=%d: %d regs, first %s A=%08x want=%08x" % (u, j, len(bad), bad[0].tolist(), int(A[tuple(bad[0])]), int(want[tuple(bad[0])])))
                return False
    return ok


def kernel_imm(src):
    m = re.search(r'lop3\.b32 %0, %1, %2, %3, (0x[0-9a-f]+);', src)
    return int(m.group(1), 16)


def main():
    import sys, time, importlib.util
    sys.path.insert(0, "muse_bf16_proj")
    path = sys.argv[1]
    small = "--small" in sys.argv
    spec = importlib.util.spec_from_file_location("sub", path)
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    imm = int(sys.argv[sys.argv.index("--imm") + 1], 16) if "--imm" in sys.argv else kernel_imm(M.GVCU)
    per_unit = "issue_per_unit" in sys.argv or "ce_issue<L::CED>(ciss, ceb.pb, cP, cgs, crl, lane);\n          asm" in M.GVCU
    print("kernel lop3 immLut 0x%x, issue per unit: %s" % (imm, per_unit))
    torch.manual_seed(0)
    if small:
        g = torch.Generator().manual_seed(5151)
        W = {"pooler": (torch.randn(256, 8192, generator=g) / 8192 ** .5).to(torch.bfloat16),
             "qkv_gate": (torch.randn(256, 8192, generator=g) / 8192 ** .5).to(torch.bfloat16),
             "unpooler": (torch.randn(256, 4096, generator=g) / 4096 ** .5).to(torch.bfloat16)}
    else:
        import reference
        W = reference.generate_weights(5151, device="cpu")
    ok_all = True
    seen = set()
    for name, ly, ms, g7, v in M.GEP:
        fmt, wr, wk, ku, s, rt, xs = g7[:7]
        w = W[ly]
        for m in ms:
            n = 1024 if ly == "unpooler" and m <= 8 else w.shape[0]
            if small: n = w.shape[0]
            key = (ly, n, g7[:6], v)
            if key in seen: continue
            seen.add(key)
            inst = M._gxinst(g7, v, m)
            fr = M._pack(w[:n])
            st = {ly: (fr, None)}
            t0 = time.time()
            pk = M._gepack(st, ly, inst, n)
            if pk is None:
                print("FAIL pack", name); ok_all = False; continue
            ed = 8 if (v >> 19) & 1 else 4
            for late in (False, True):
                r = emulate(fr, pk, (wr, wk, ku, s, rt), ed, bool((v >> 20) & 1), bool((v >> 21) & 1), imm, per_unit, late, pk.get("pfix"))
                print("EMU %-10s %-9s n=%d geo %s ed%d %s %s  (%.1fs)" % (name, ly, n, (wr, wk, ku, s, rt), ed, "late " if late else "early", "bit-exact" if r else "MISMATCH", time.time() - t0), flush=True)
                ok_all &= r
                if not r: break
    print("EMU", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
