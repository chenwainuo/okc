"""Instruction-level CPU emulation of the FMT-4 (codec E v3, pw18) gv data path (one torch lane per device lane, all warps at once):
plane loads at the kernel's wo / js offsets, the per-lane cp.async ring (ce_issue, commit, wait_group) with two landing
models (early: data lands at issue; late: slot is poisoned at issue and filled only when wait_group guarantees it),
ce_refill / ce_unit with generic PRMT and LOP3 (immLut taken from the kernel source), the LUT read from the smem image
the kernel builds (single or lane-replicated), and r6's escape walk with the 8-bit XOR field. Compares every A register
against the bf16 fragments. Usage: python3 -m proj_rsi.tools.codec_e_emu <submission.py> [--small] (every GEP path of a pw18-style submission;
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
    """PTX prmt.b32 default mode; sel an int or a tensor (per-lane selectors), sign-replicate nibbles included"""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    shape = torch.broadcast_shapes(a.shape, b.shape, torch.as_tensor(sel).shape)
    by = torch.stack([((a >> (8 * k)) & 255).expand(shape) for k in range(4)] + [((b >> (8 * k)) & 255).expand(shape) for k in range(4)], -1)
    sel = torch.as_tensor(sel).expand(shape)
    r = torch.zeros(shape, dtype=torch.long)
    for k in range(4):
        nib = (sel >> (4 * k)) & 15
        v = by.gather(-1, (nib & 7)[..., None])[..., 0]
        v = torch.where((nib & 8) != 0, ((v >> 7) & 1) * 255, v)
        r |= v << (8 * k)
    return r


def fsr(lo, hi, n):  # __funnelshift_r(lo, hi, n): n & 31
    n = n & 31
    return (((hi << 32) | lo) >> n) & M32


def fsl(lo, hi, n):  # __funnelshift_l(lo, hi, n): high word of (hi:lo) << n
    n = n & 31
    return ((((hi << 32) | lo) << n) >> 32) & M32


def emulate(fr, pk, geo, ed, rep, imm, late, K=1):
    """v3 kernel (pw18 gv FMT 4): K substreams, ring of ed words per lane ([s][slot][lane], ES = ed / K slots), 3-round
    prologue, per unit NI = 4 / K predicated 4 B issues per substream + commit + wait_group(2), refill word of a
    substream's first step read after the wait and the others prefetched one step ahead, len = PRMT(t0, t1, lo & 7),
    value from the LUT smem image, A = LOP3(imm), escape walk. Compares every A register with the fragments."""
    T, KC = fr.shape[:2]
    WR, WK, KU, S, RT = geo
    NW, GX, NL = WR * WK, T // (WR * RT), 2
    ES, NI, NS = ed // K, 4 // K, 4 // K
    nseg = GX * S * NW
    gw = torch.arange(nseg)
    w_, g2 = gw % NW, gw // NW
    bx, by = g2 % GX, g2 // GX
    wr_, wk_ = w_ % WR, w_ // WR
    lane = torch.arange(32)[None, :]
    c0 = (by * WK * KU + wk_ * KU)[:, None]
    tb = ((bx * WR + wr_) * RT)[:, None]
    plane = pk["plane"].contiguous().view(torch.int32).long().flatten() & M32
    words = pk["stream"].long() & M32
    lutf = pk["lut"].long() & M32
    t0, t1 = lutf[256], lutf[257]
    lut = lutf[:256]
    sm_lut = lut.repeat_interleave(32) if rep else lut
    so = pk["soff"].long() & M32
    s0 = so[:-1][:, None]
    P = ((so[1:] - so[:-1]) // (32 * K))[:, None].expand(nseg, 32)
    ring = [torch.randint(0, 2 ** 32, (nseg, 32, ES), dtype=torch.long) for _ in range(K)]
    st = [dict(lo=torch.zeros(nseg, 32, dtype=torch.long), hi=torch.zeros(nseg, 32, dtype=torch.long),
               V=torch.zeros(nseg, 32, dtype=torch.long), pb=torch.zeros(nseg, 32, dtype=torch.long),
               iss=torch.zeros(nseg, 32, dtype=torch.long), nw=torch.zeros(nseg, 32, dtype=torch.long)) for _ in range(K)]
    groups, cur = [], []

    def issue(s):
        b = st[s]
        for _ in range(NI):
            go = (b["iss"] < P) & (b["iss"] < (b["pb"] >> 2) + ES)
            gi = (s0 + (s * P + b["iss"]) * 32 + lane).clamp(max=words.numel() - 1)
            data = words[gi]
            slot = (b["iss"] % ES)[..., None]
            old = ring[s].gather(2, slot)[..., 0]
            if late:  # undefined until wait_group guarantees it
                ring[s].scatter_(2, slot, torch.where(go, torch.randint(0, 2 ** 32, old.shape, dtype=torch.long), old)[..., None])
                cur.append((s, go, slot, data))
            else:
                ring[s].scatter_(2, slot, torch.where(go, data, old)[..., None])
            b["iss"] = b["iss"] + go.long()

    def commit():
        global_ = list(cur); cur.clear(); groups.append([global_, False])

    def wait(n):
        for g in groups[:len(groups) - n]:
            if not g[1]:
                for s, go, slot, data in g[0]:
                    old = ring[s].gather(2, slot)[..., 0]
                    ring[s].scatter_(2, slot, torch.where(go, data, old)[..., None])
                g[1] = True

    def lds_ring(s, pb):
        slot = ((pb << 5) & ((ES - 1) << 7)) >> 7
        return ring[s].gather(2, slot[..., None])[..., 0]

    E = pk["esc"].long() & M32
    E = E.view(nseg, 32)
    ep = torch.zeros(nseg, dtype=torch.long)
    nx = E[:, 0].clone()
    ref = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).long() & 0xffff
    z = lambda: torch.zeros(nseg, 32, dtype=torch.long)
    for _ in range(3):
        for s in range(K):
            issue(s)
        commit()
    for u in range(KU):
        for j in range(RT):
            for s in range(K):
                issue(s)
            commit(); wait(2)
            t = tb + j
            c = c0 + u
            wo = ((t * KC + c) * (NL * 32) + lane)
            q0 = torch.stack([plane[wo * 4 + k] for k in range(4)], -1)
            q1 = torch.stack([plane[(wo + 32) * 4 + k] for k in range(4)], -1)
            L = [q0[..., k] for k in range(4)] + [q1[..., k] for k in range(4)]
            A = [None] * 16
            for tt in range(NS):
                for s in range(K):
                    b = st[s]
                    w = lds_ring(s, b["pb"]) if tt == 0 else b["nw"]
                    need = (b["V"] & 0x60) == 0
                    w = torch.where(need, w, 0)
                    at = b["V"] & 31
                    b["lo"] = b["lo"] | ((w << at) & M32)
                    b["hi"] = b["hi"] | fsl(w, z(), at)
                    b["V"] = (b["V"] + need.long() * 32) & M32
                    b["pb"] = (b["pb"] + need.long() * 4) & M32
                    if tt + 1 < NS:
                        b["nw"] = lds_ring(s, b["pb"])
                for i in range(4):
                    for s in range(K):
                        b = st[s]
                        jj = s * NS + tt
                        ln = prmt(t0.expand(nseg, 32), t1.expand(nseg, 32), b["lo"] & 7)
                        x = b["lo"] & 0xff
                        e = sm_lut[x * 32 + lane] if rep else sm_lut[x]
                        b["lo"], b["hi"] = fsr(b["lo"], b["hi"], ln), fsr(b["hi"], z(), ln)
                        b["V"] = (b["V"] - ln) & M32
                        A[4 * jj + i] = lop3(e, torch.full_like(e, 0x807f807f), prmt(L[2 * jj + (i >> 1)], z(), 0x3322 if i & 1 else 0x1100), imm)
            A = torch.stack(A, -1)
            seq = u * RT + j
            while True:
                hit = (nx >> 20) == seq
                if not bool(hit.any()):
                    break
                ln_ = (nx >> 15) & 31
                r = (nx >> 11) & 15
                xr = (nx & 0xff) << (7 + ((nx >> 10) & 1) * 16)
                sel = hit[:, None, None] & (lane[..., None] == ln_[:, None, None]) & (torch.arange(16)[None, None, :] == r[:, None, None])
                A = torch.where(sel, A ^ xr[:, None, None], A)
                ep = torch.where(hit, ep + 1, ep)
                nx = torch.where(hit, torch.where(ep < 32, E.gather(1, ep.clamp(max=31)[:, None])[:, 0], M32), nx)
            want = ref[t[:, 0], c[:, 0]].permute(0, 2, 1, 3).reshape(nseg, 32, 16, 2)
            want = want[..., 0] | (want[..., 1] << 16)
            if not torch.equal(A, want):
                bad = (A != want).nonzero()
                print("   mismatch at u=%d j=%d: %d regs, first %s A=%08x want=%08x" % (u, j, len(bad), bad[0].tolist(), int(A[tuple(bad[0])]), int(want[tuple(bad[0])])))
                return False
    return True


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
    print("kernel lop3 immLut 0x%x" % imm)
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
            ed = 32 if (v >> 19) & 1 else 16
            K = 1 << ((v >> 22) & 3)
            for late in (False, True):
                r = emulate(fr, pk, (wr, wk, ku, s, rt), ed, bool((v >> 20) & 1), imm, late, K)
                print("EMU %-10s %-9s n=%d geo %s K%d ed%d %s %s  (%.1fs)" % (name, ly, n, (wr, wk, ku, s, rt), K, ed, "late " if late else "early", "bit-exact" if r else "MISMATCH", time.time() - t0), flush=True)
                ok_all &= r
                if not r: break
    print("EMU", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
