"""Codec E for pw17: pair Huffman over the full bf16 exponent, decoded by one 256-entry LUT lookup per A register.

Spec: proj_rsi/tools/CODEC_E_LAYOUT.md. Study: proj_rsi/audits/codec-study.md.

  pack_e(fr, geo, layer)    torch packer (vectorised, runs on fr.device; no Python loops over weights)
  unpack_e(pk, T, KC, geo)  reference unpacker: a lane-parallel torch port of the device routine (buffer, V, refill),
                            used to prove the pack lossless before the kernel is trusted
  CODEC_E_CUH               the device routine (string) for gv
  python3 -m proj_rsi.tools.codec_e --test   bit-exact pack/unpack on 1M weights per distribution and geometry
"""
import argparse
import time

import torch

ESC_MAX = 4e-4          # target escape rate (weights); keeps the 32-entry per-warp list far from overflow
LMAX = 8                # code length limit: 4 pairs per step <= 32 bits (one refill per step suffices)


def _package_merge(cnt, L):
    """optimal prefix-code lengths (max length L) for positive counts (python list)."""
    n = len(cnt)
    if n == 1:
        return [1]
    leaves = sorted((float(q), (i,)) for i, q in enumerate(cnt))
    cur = list(leaves)
    for _ in range(L - 1):
        pk = [(cur[2 * i][0] + cur[2 * i + 1][0], cur[2 * i][1] + cur[2 * i + 1][1]) for i in range(len(cur) // 2)]
        cur = sorted(leaves + pk, key=lambda t: t[0])
    ln = [0] * n
    for _, syms in cur[:2 * n - 2]:
        for s in syms:
            ln[s] += 1
    return ln


def _canonical_lsb(lengths):
    """canonical codes, bit-reversed: the code of length l is the low l bits of the LSB-first peek window."""
    order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
    code, prev, out = 0, None, [0] * len(lengths)
    for s in order:
        l = lengths[s]
        if prev is not None:
            code = (code + 1) << (l - prev)
        prev = l
        out[s] = int(format(code, "0%db" % l)[::-1], 2)
    return out


def _geo(geo):
    wr, wk, ku, s, rt = geo[:5]
    return wr, wk, ku, s, rt


def _to_segments(x, T, KC, geo):
    """[T, KC, 4, 32, 4, ...] (tile, chunk, step j, lane, reg-in-step i) -> [NSEG, 32, KU*RT*16, ...] in segment
    (gw) order and lane-stream order (u, jt, r = 4 j + i)."""
    wr, wk, ku, s, rt = _geo(geo)
    gx = T // (wr * rt)
    rest = x.shape[5:]
    x = x.reshape(gx, wr, rt, s, wk, ku, 4, 32, 4, *rest)
    nd = len(rest)
    x = x.permute(3, 0, 4, 1, 7, 5, 2, 6, 8, *range(9, 9 + nd))  # by, bx, wk, wr, lane, u, jt, j, i
    return x.reshape(s * gx * wk * wr, 32, ku * rt * 16, *rest)


def _gvesc_e(t, c, j, l, v, x, T, geo, dev):
    """r6's _gvesc structure with an 8-bit exponent XOR (bits 0-7; the kernel applies x << (7 + 16 * (v & 1)))."""
    wr, wk, ku, s, rt = _geo(geo)
    gx, nw = T // (wr * rt), wr * wk
    out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=dev)
    if len(t):
        gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
        seq = (c % ku) * rt + t % rt
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


def pack_e(fr, geo, layer="?", verbose=True):
    """r6 fragments [T, KC, 4, 32, 8] bf16 + gv geometry (WR, WK, KU, S, RT, ...) -> dict or None (escape overflow).
    keys: plane uint8 [T, KC, 2, 32, 16]; lut int32 [256]; stream int32 [pieces * 4]; soff int32 [NSEG + 1] (16 B
    units); esc int32 [NSEG * 32]; stats dict."""
    t0 = time.time()
    dev = fr.device
    T, KC = fr.shape[:2]
    wr, wk, ku, s, rt = _geo(geo)
    assert wk * ku * s == KC and T % (wr * rt) == 0, (geo, T, KC)
    b = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff
    e8 = (b >> 7) & 0xff
    plane = ((b >> 8) & 0x80) | (b & 0x7f)
    n = b.numel()

    # exponent ranks and the pair alphabet (from the pair histogram, never per weight in Python)
    cnt = torch.bincount(e8.flatten(), minlength=256)
    val = torch.argsort(cnt, descending=True, stable=True)
    rank = torch.empty(256, dtype=torch.long, device=dev)
    rank[val] = torch.arange(256, device=dev)
    rk = rank[e8].view(T, KC, 4, 32, 4, 2)
    pid = rk[..., 0] * 256 + rk[..., 1]                                  # [T, KC, 4, 32, 4]
    pc = torch.bincount(pid.flatten(), minlength=65536)
    order = torch.argsort(pc, descending=True, stable=True)
    npos = int((pc > 0).sum())
    pcf = pc.view(256, 256)
    N = None
    for cand in list(range(16, 256, 8)) + [256]:
        cand = min(cand, npos)
        C = torch.zeros(65536, dtype=torch.bool, device=dev)
        C[order[:cand]] = True
        C = C.view(256, 256)
        one = C.any(1)[:, None] | C.any(0)[None, :]
        esc = int((pcf * (~C) * (2 - one.long())).sum())
        if esc <= ESC_MAX * n or cand == npos:
            N = cand
            break
    syms = order[:N]
    C = torch.zeros(65536, dtype=torch.bool, device=dev)
    C[syms] = True
    Cm = C.view(256, 256)
    # placeholder for every pair id: itself if covered, else the most frequent covered pair sharing a, else sharing b,
    # else the most frequent pair
    a_i = torch.arange(256, device=dev)
    rowbest = torch.where(Cm, pcf, -1).argmax(1)                         # best b' for row a
    colbest = torch.where(Cm, pcf, -1).argmax(0)                         # best a' for column b
    A, B = torch.meshgrid(a_i, a_i, indexing="ij")
    ph = torch.where(Cm, A * 256 + B,
                     torch.where(Cm.any(1)[:, None], A * 256 + rowbest[:, None],
                                 torch.where(Cm.any(0)[None, :], colbest[None, :] * 256 + B, syms[0].expand(256, 256))))
    ph = ph.flatten()
    eff = torch.bincount(ph[pid.flatten()], minlength=65536)             # counts after mapping
    symlist = syms.tolist()
    ln = _package_merge([max(int(eff[x]), 1) for x in symlist], LMAX)
    code = _canonical_lsb(ln)
    sym_len = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_code = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_len[syms] = torch.tensor(ln, device=dev)
    sym_code[syms] = torch.tensor(code, device=dev)
    # LUT: every 8-bit peek x whose low len bits equal the code
    x = torch.arange(256, device=dev)
    lut = torch.zeros(256, dtype=torch.long, device=dev)
    sv = torch.tensor(symlist, device=dev)
    ent = (val[sv // 256] << 7) | (val[sv % 256] << 23) | torch.tensor(ln, device=dev)
    hit = (x[None, :] & (2 ** torch.tensor(ln, device=dev)[:, None] - 1)) == torch.tensor(code, device=dev)[:, None]
    assert bool((hit.sum(0) == 1).all())
    lut = (hit.long() * ent[:, None]).sum(0)

    # escapes: weights whose rank differs from the placeholder's
    php = ph[pid]                                                        # [T, KC, 4, 32, 4]
    pr = torch.stack((php // 256, php % 256), -1)                        # placeholder ranks [.., 2]
    mis = pr != rk
    ei = mis.nonzero()
    t_, c_, j_, l_, i_, h_ = ei.unbind(1)
    xr = (val[rk[mis]] ^ val[pr[mis]]).long()
    escl = _gvesc_e(t_, c_, j_, l_, 2 * i_ + h_, xr, T, geo, dev)
    if escl is None:
        if verbose:
            print("STATS_E %s escape list overflow (esc=%.5f)" % (layer, len(ei) / n))
        return None

    # code stream: lengths per pair in lane-stream order, exclusive cumsum, bit scatter (int64)
    L = _to_segments(sym_len[php], T, KC, geo)                           # [NSEG, 32, NP]
    V = _to_segments(sym_code[php], T, KC, geo)
    nseg, NP = L.shape[0], L.shape[2]
    off = torch.cumsum(L, 2) - L                                         # bit offset in the lane stream
    tot = L.sum(2)                                                       # [NSEG, 32]
    P = (tot.max(1).values + 127) // 128                                 # pieces per lane per segment
    soff = torch.zeros(nseg + 1, dtype=torch.long, device=dev)
    soff[1:] = torch.cumsum(P * 32, 0)
    nwords = int(soff[-1]) * 4
    wd = off >> 5
    sh = off & 31
    lane = torch.arange(32, device=dev)[None, :, None]
    gword = lambda w: (soff[:-1, None, None] + (w >> 2) * 32 + lane) * 4 + (w & 3)
    words = torch.zeros(nwords + 4, dtype=torch.long, device=dev)
    words.scatter_add_(0, gword(wd).flatten(), ((V << sh) & 0xffffffff).flatten())
    spill = (sh + L) > 32
    words.scatter_add_(0, gword(wd + 1)[spill], (V >> (32 - sh))[spill])
    words = words[:nwords]
    stream = torch.where(words >= 2 ** 31, words - 2 ** 32, words).int()

    pl = plane.to(torch.uint8)
    p0 = pl[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = pl[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    planeb = torch.stack((p0, p1), 2).contiguous()
    lut32 = torch.where(lut >= 2 ** 31, lut - 2 ** 32, lut).int()

    stream_bits = nwords * 32
    pad_bits = stream_bits - int(tot.sum())
    total_bits = 8 * n + stream_bits + 32 * (nseg + 1) + 32 * 256 + 32 * escl.numel()
    st = dict(bits=total_bits / n, esc=len(ei) / n, maxpad=pad_bits / stream_bits, maxpad_bits=pad_bits / n,
              N=N, code_bits=float(tot.sum()) / n, secs=time.time() - t0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        st["secs"] = time.time() - t0
    if verbose:
        print("STATS_E %s bits=%.4f esc=%.5f maxpad=%.4f (code %.4f, pad %.4f bits/w, alphabet %d, %.2f s)"
              % (layer, st["bits"], st["esc"], st["maxpad"], 8 + st["code_bits"], st["maxpad_bits"], N, st["secs"]))
    return dict(plane=planeb, lut=lut32, stream=stream, soff=soff.int(), esc=escl, stats=st)


def unpack_e(pk, T, KC, geo, ED=4):
    """reference decoder: the device routine's exact logic (64-bit buffer, V test (V & 0x60), refill at step start,
    V -= entry, ptr over the lane's interleaved 16 B pieces), run for all lanes of all segments at once.
    Also replays the smem ring protocol of ce_issue (ED slots, prologue ED - 1 pieces, one issue + commit per unit,
    wait_group(ED - 2)) and asserts every refill reads a landed, not yet overwritten piece.
    Returns the bf16 bit patterns [T, KC, 4, 32, 8] (int32) after applying the escape list."""
    dev = pk["stream"].device
    wr, wk, ku, s, rt = _geo(geo)
    M32 = 0xffffffff
    words = pk["stream"].long() & M32
    lut = pk["lut"].long() & M32
    soff = pk["soff"].long()
    nseg = soff.numel() - 1
    P = (soff[1:] - soff[:-1]) // 32
    lane = torch.arange(32, device=dev)[None, :]
    lo = torch.zeros(nseg, 32, dtype=torch.long, device=dev)
    hi = torch.zeros_like(lo)
    Vr = torch.zeros_like(lo)
    ptr = torch.zeros_like(lo)
    NP = ku * rt * 16
    E = torch.zeros(nseg, 32, NP, dtype=torch.long, device=dev)
    Pl = P[:, None].expand(nseg, 32)
    iss = torch.zeros_like(lo)
    groups = []                                                          # pieces issued per commit group (per lane)

    def issue():
        nonlocal iss
        go = (iss < Pl) & (iss < (ptr >> 2) + ED)
        iss = iss + go.long()
        groups.append(iss.clone())                                       # pieces issued once this group lands

    for _ in range(ED - 1):
        issue()
    landed = None
    for st in range(NP // 4):
        if st % 4 == 0:                                                  # unit start: issue, commit, wait_group(ED-2)
            issue()
            landed = groups[len(groups) - 1 - (ED - 2)]
        need = (Vr & 0x60) == 0
        inb = (ptr >> 2) < P[:, None]
        rd = need & inb
        assert bool(((ptr >> 2) < landed)[rd].all()), "ring: piece not landed"
        assert bool(((ptr >> 2) + ED > iss - 1)[rd].all()), "ring: slot overwritten"
        gi = (soff[:-1, None] + (ptr >> 2) * 32 + lane) * 4 + (ptr & 3)
        w = torch.where(need & inb, words[gi.clamp(max=words.numel() - 1)], 0)
        at = Vr & 31
        lo = torch.where(need, lo | ((w << at) & M32), lo)
        hi = torch.where(need, hi | ((w >> (32 - at)) * (at > 0)), hi)
        Vr = torch.where(need, (Vr + 32) & M32, Vr)
        ptr = ptr + need.long()
        for q in range(4):
            e = lut[lo & 0xff]
            n_ = e & 31
            lo, hi = ((lo >> n_) | (hi << (32 - n_))) & M32, hi >> n_        # n_ in 1..8
            Vr = (Vr - e) & M32
            E[:, :, 4 * st + q] = e
    # back to fragment order
    gx = T // (wr * rt)
    E = E.view(s, gx, wk, wr, 32, ku, rt, 4, 4).permute(1, 3, 6, 0, 2, 5, 7, 4, 8).reshape(T, KC, 4, 32, 4)
    pl = pk["plane"].long()                                              # [T, KC, 2, 32, 16]
    pb = pl.view(T, KC, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 4, 32, 8)
    lowh = pb[..., 0::2]
    hih = pb[..., 1::2]
    dup0 = lowh | (lowh << 8)
    dup1 = hih | (hih << 8)
    m = 0x807f
    v0 = (E & ~m & 0xffff) | (dup0 & m)
    v1 = ((E >> 16) & ~m & 0xffff) | (dup1 & m)
    vals = torch.stack((v0, v1), -1).reshape(T, KC, 4, 32, 8)
    esc = pk["esc"].long() & M32
    ent = esc[esc != M32]
    if len(ent):
        # recover the unit of every entry from (warp, seq): rebuild warp id from its index
        gwi = (esc != M32).nonzero().flatten() // 32
        seq, ln_, r, half, x = ent >> 20, (ent >> 15) & 31, (ent >> 11) & 15, (ent >> 10) & 1, ent & 0xff
        nw = wr * wk
        wwr, wwk, g2 = gwi % nw % wr, gwi % nw // wr, gwi // nw
        bx, by = g2 % gx, g2 // gx
        t = (bx * wr + wwr) * rt + seq % rt
        c = (by * wk + wwk) * ku + seq // rt
        v = (r % 4) * 2 + half
        vals[t, c, r // 4, ln_, v] ^= x << 7
    return vals.int()


def _frag_bits(fr):
    return fr.reshape(*fr.shape[:2], 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff


def _pack_r6(w):
    """r6's _pack (bf16 [N, K] -> [N/16, K/64, 4, 32, 8])."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()


# h7 marginals measured on the reference weights (seed 5151): (h7 value, p)
_MARG = {"pooler": [(60, .4757), (59, .3795), (58, .1049), (57, .0264), (56, .0066), (61, .0048), (55, .0016), (54, .0004), (53, .0001)],
         "o": [(60, .4846), (61, .3181), (59, .1475), (58, .0373), (57, .0093), (56, .0023), (55, .0006), (54, .0002), (62, .00002)]}


def _marg_weights(name, N, K, g):
    v, p = zip(*_MARG[name])
    p = torch.tensor(p, dtype=torch.double)
    h7 = torch.tensor(v)[torch.multinomial(p / p.sum(), N * K, replacement=True, generator=g)]
    low = torch.randint(0, 256, (N * K,), generator=g)
    sg = torch.randint(0, 2, (N * K,), generator=g)
    return ((sg << 15) | (h7 << 8) | low).to(torch.int16).view(torch.bfloat16).view(N, K)


def selftest():
    g = torch.Generator().manual_seed(5151)
    cases = [  # (name, N, K, weights, geo (WR, WK, KU, S, RT, XS))
        ("gauss K8192 pooler-geo", 128, 8192, None, (2, 8, 16, 1, 1, 0)),
        ("gauss K8192 rt2", 128, 8192, None, (2, 4, 16, 2, 2, 0)),
        ("gauss K1024 o-geo", 1024, 1024, None, (1, 8, 2, 1, 1, 0)),
        ("gauss K1024 ku4", 1024, 1024, None, (2, 4, 4, 1, 1, 0)),
        ("gauss K3072", 512, 2048, None, (1, 4, 8, 1, 1, 0)),
        ("marg pooler", 128, 8192, "pooler", (1, 8, 16, 1, 1, 0)),
        ("marg o", 1024, 1024, "o", (1, 8, 2, 1, 1, 0)),
    ]
    ok = True
    for name, N, K, mg, geo in cases:
        if mg is None:
            kk = 3072 if "3072" in name else K
            w = (torch.randn(N, K, generator=g, dtype=torch.float64) / kk ** 0.5).to(torch.bfloat16)
        else:
            w = _marg_weights(mg, N, K, g)
        fr = _pack_r6(w)
        pk = pack_e(fr, geo, name)
        if pk is None:
            print("FAIL %s: escape overflow" % name); ok = False; continue
        T, KC = fr.shape[:2]
        t0 = time.time()
        got = unpack_e(pk, T, KC, geo)
        eq = torch.equal(got, _frag_bits(fr))
        print("   %-24s geo %-20s %s  (unpack %.1f s)" % (name, geo, "bit-exact" if eq else "MISMATCH", time.time() - t0))
        ok &= eq
        if name.startswith("gauss K8192 pooler"):                       # negative control: one flipped stream bit
            bad = dict(pk, stream=pk["stream"].clone())
            bad["stream"][bad["stream"].numel() // 3] ^= 1 << 5
            neg = not torch.equal(unpack_e(bad, T, KC, geo), _frag_bits(fr))
            print("   %-24s %s" % ("negative control", "detected" if neg else "NOT DETECTED"))
            ok &= neg
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def bench(n_mb=50):
    """time the packer on a pooler-sized layer (n_mb MB of bf16)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = 8192
    N = n_mb * 2 ** 20 // 2 // K // 16 * 16
    w = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16)
    fr = _pack_r6(w)
    pack_e(fr, (2, 8, 16, 1, 1, 0), "pooler-bench(%s, %d x %d)" % (dev, N, K))


CODEC_E_CUH = r"""
// ---- codec E device decode (proj_rsi/tools/CODEC_E_LAYOUT.md) -------------------------------------------------------
// Per lane state (4 registers): lo, hi = 64-bit bit buffer (next code at bit 0 of lo); V = valid bits mod 128 (bits 7+
// carry garbage from V -= entry; only (V & 0x60) and (V & 31) are ever read); pb = 4 x words taken from the lane stream
// (piece pb >> 4, word (pb >> 2) & 3).
// Ring: per lane ED pieces of 16 B, contiguous per lane: lane l's ring starts at rl = ring + l * ED * 16 (ED a power of
// two, ring aligned to ED * 16), piece p in slot rl + (p % ED) * 16; the word at stream byte pb is rl | (pb & (16 ED - 1)).
// Filled by ce_issue (predicated cp.async 16 B, at most one piece per lane per unit, one commit group per unit).
// LUT: 256 x u32 in smem (1 KB per matrix), smem byte address lut.
// ce_unit: the 16 A registers (r6 order A[4 j + i]) of one 16 x 64 unit from the lane's two plane pieces q0, q1;
// apply r6's escape walk afterwards with A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1)).
// SASS (nvcc 12.8, sm_90a, -O3, 0 spills; probe = ce_issue + commit/wait + ce_unit + 4 HMMA per unit, ED = 4):
//   per unit (32 weights per lane): 194 lane-instructions excluding the 4 HMMA, 2 plane LDG, LDC, BRA = 6.06 / weight
//   (LOP3 54, SHF 40, IMAD 29, LDS 23, PRMT 16, VIADD 9, IADD3 8, SEL 5, LEA 3, ISETP 3, LDGSTS 1, DEPBAR 2, UIADD3 1);
//   per A register: LOP3+IMAD.SHL (LUT address), LDS, 2 SHF, PRMT, LOP3 (+ IADD3 per 2); per step: refill ~13;
//   32 registers in the probe. Static count only (no GPU on the build host).
struct CeBuf { uint32_t lo, hi, V, pb; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f)
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xe4;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// once per step (8 weights): branch-free; the LDS always reads inside the lane's ring (a stale word is masked to 0)
template <int ED>
__device__ __forceinline__ void ce_refill(CeBuf& b, uint32_t rl) {
  const bool need = (b.V & 0x60u) == 0u;
  uint32_t w = ce_lds(rl | (b.pb & (16u * ED - 1u)));
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// once per unit, then cp.async.commit_group and cp.async.wait_group so that pieces up to (pb >> 4) + 1 have landed.
// gsrc: global address of piece 0, lane 0 of this warp's segment (stream bytes + soff[gw] * 16); P: pieces per lane.
template <int ED>
__device__ __forceinline__ void ce_issue(uint32_t& iss, uint32_t pb, uint32_t P, const uint8_t* gsrc, uint32_t rl, int lane) {
  const bool go = iss < P && iss < (pb >> 4) + ED;
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.cg.shared.global [%0], [%1], 16; }\n"
               :: "r"(rl + ((iss % ED) << 4)), "l"(gsrc + ((size_t)iss * 32 + lane) * 16), "r"((uint32_t)go));
  iss += go;
}
template <int ED>
__device__ __forceinline__ void ce_unit(CeBuf& b, uint32_t lut, uint32_t rl, uint4 q0, uint4 q1, uint32_t* A) {
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    ce_refill<ED>(b, rl);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t e = ce_lds(lut + ((b.lo & 0xffu) << 2));
      b.lo = __funnelshift_r(b.lo, b.hi, e); b.hi = __funnelshift_r(b.hi, 0u, e); b.V -= e;
      A[4 * j + i] = ce_sel(e, ce_prmt(L[2 * j + (i >> 1)], 0u, (i & 1) ? 0x3322u : 0x1100u));
    }
  }
}
// escapes (r6 walk, unchanged except the XOR field): A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1))
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="pack a pooler-sized layer of this many MB")
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if selftest() else 1)
    if a.bench:
        bench(a.bench)
