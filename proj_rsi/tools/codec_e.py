"""Codec E (v3, pw18): pair code over the full bf16 exponent whose length is fixed by its first 3 bits (ALU-decoded),
values from one 256-entry LUT lookup per A register; 1, 2 or 4 independent substreams per lane.

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
PK = 3                  # v3: the first PK bits of a code (LSB-first) determine its length (8-entry byte table, one PRMT)


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


def _prefix_code(cnt, k=PK, L=LMAX):
    """optimal prefix code (max length L) in which the first k bits of every code determine its length: codes of length
    <= k, and buckets (one k-bit prefix each) of 2^(Lb - k) codes of length Lb. cnt: python list (any order).
    Returns (lengths, codes LSB-first, tab[2^k] = length per k-bit window)."""
    n, P = len(cnt), 2 ** k
    order = sorted(range(n), key=lambda s: (-cnt[s], s))
    p = [float(cnt[s]) for s in order]
    cum = [0.0]
    for x in p:
        cum.append(cum[-1] + x)
    INF = float("inf")
    D = [[INF] * (P + 1) for _ in range(n + 1)]
    B = [[None] * (P + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n):
        for u in range(P + 1):
            d = D[i][u]
            if d == INF:
                continue
            for l in range(1, k + 1):
                w = 2 ** (k - l)
                if u + w <= P and d + p[i] * l < D[i + 1][u + w]:
                    D[i + 1][u + w], B[i + 1][u + w] = d + p[i] * l, (i, u, "s", l)
            if u + 1 <= P:
                for lb in range(k + 1, L + 1):
                    j = min(n, i + 2 ** (lb - k))
                    v = d + (cum[j] - cum[i]) * lb
                    if v < D[j][u + 1]:
                        D[j][u + 1], B[j][u + 1] = v, (i, u, "b", lb)
    u = min(range(P + 1), key=lambda q: D[n][q])
    assert D[n][u] < INF, "no prefix code fits"
    items, i = [], n
    while i > 0:
        pi, pu, kind, l = B[i][u]
        items.append((kind, l, order[pi:i]))
        i, u = pi, pu
    items.reverse()
    # canonical assignment in the k-bit prefix space (virtual length: l for short codes, k for buckets), then bit-reversed
    items.sort(key=lambda it: it[1] if it[0] == "s" else k)
    lengths, codes, tab = [0] * n, [0] * n, [L] * P
    code, prev = 0, None
    for kind, l, syms in items:
        vl = l if kind == "s" else k
        if prev is not None:
            code = (code + 1) << (vl - prev)
        prev = vl
        pre = int(format(code, "0%db" % vl)[::-1], 2)
        if kind == "s":
            lengths[syms[0]], codes[syms[0]] = l, pre
            for w in range(P):
                if w & (2 ** l - 1) == pre:
                    tab[w] = l
        else:
            tab[pre] = l
            for j, s in enumerate(syms):
                lengths[s], codes[s] = l, pre | (j << k)
    return lengths, codes, tab


def _wpp(ilp):
    """words per ring piece. v3: 4 B pieces at every ilp (padding to a word, not 16 B: ilp 1/2/4 pad 0.12/0.22/0.36
    bits/w against 0.16/0.55/0.36 with 16 B pieces); the code supports 16 B pieces with per-lane word rotation"""
    return 1


def _substreams(x, ilp):
    """[NSEG, 32, KU*RT*16, ...] (unit, register) -> [NSEG, ilp, 32, KU*RT*16/ilp, ...]: substream s holds steps
    j in [s*4/ilp, (s+1)*4/ilp) of every unit, in unit order"""
    nseg, _, n = x.shape[:3]
    rest = x.shape[3:]
    x = x.reshape(nseg, 32, n // 16, ilp, 16 // ilp, *rest)
    x = x.permute(0, 3, 1, 2, 4, *range(5, 5 + len(rest)))
    return x.reshape(nseg, ilp, 32, n // ilp, *rest)


def pack_e(fr, geo, layer="?", verbose=True, ilp=1):
    """v3. r6 fragments [T, KC, 4, 32, 8] bf16 + gv geometry (WR, WK, KU, S, RT, ...) -> dict or None (escape overflow).
    keys: plane uint8 [T, KC, 2, 32, 16]; lut int32 [258] (256 value entries + the 8-byte length table); stream int32;
    soff int32 [NSEG + 1] (in 4 B words; a segment = ilp substream regions of P * 32 pieces of 16 B, 4 B at ilp 4);
    esc int32 [NSEG * 32]; ilp; stats dict."""
    t0 = time.time()
    dev = fr.device
    T, KC = fr.shape[:2]
    wr, wk, ku, s, rt = _geo(geo)
    assert wk * ku * s == KC and T % (wr * rt) == 0 and ilp in (1, 2, 4), (geo, T, KC, ilp)
    b = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff
    e8 = (b >> 7) & 0xff
    plane = ((b >> 8) & 0x80) | (b & 0x7f)
    n = b.numel()

    cnt = torch.bincount(e8.flatten(), minlength=256)
    val = torch.argsort(cnt, descending=True, stable=True)
    rank = torch.empty(256, dtype=torch.long, device=dev)
    rank[val] = torch.arange(256, device=dev)
    rk = rank[e8].view(T, KC, 4, 32, 4, 2)
    pid = rk[..., 0] * 256 + rk[..., 1]
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
    a_i = torch.arange(256, device=dev)
    rowbest = torch.where(Cm, pcf, -1).argmax(1)
    colbest = torch.where(Cm, pcf, -1).argmax(0)
    A, B = torch.meshgrid(a_i, a_i, indexing="ij")
    ph = torch.where(Cm, A * 256 + B,
                     torch.where(Cm.any(1)[:, None], A * 256 + rowbest[:, None],
                                 torch.where(Cm.any(0)[None, :], colbest[None, :] * 256 + B, syms[0].expand(256, 256))))
    ph = ph.flatten()
    eff = torch.bincount(ph[pid.flatten()], minlength=65536)
    symlist = syms.tolist()
    ln, code, tab = _prefix_code([max(int(eff[x]), 1) for x in symlist])
    sym_len = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_code = torch.zeros(65536, dtype=torch.long, device=dev)
    sym_len[syms] = torch.tensor(ln, device=dev)
    sym_code[syms] = torch.tensor(code, device=dev)
    x = torch.arange(256, device=dev)
    sv = torch.tensor(symlist, device=dev)
    ent = (val[sv // 256] << 7) | (val[sv % 256] << 23) | torch.tensor(ln, device=dev)
    hit = (x[None, :] & (2 ** torch.tensor(ln, device=dev)[:, None] - 1)) == torch.tensor(code, device=dev)[:, None]
    assert bool((hit.sum(0) <= 1).all())
    lut = (hit.long() * ent[:, None]).sum(0)
    tabv = sum(t << (8 * i) for i, t in enumerate(tab))
    lut = torch.cat((lut, torch.tensor([tabv & 0xffffffff, tabv >> 32], device=dev)))

    php = ph[pid]
    pr = torch.stack((php // 256, php % 256), -1)
    mis = pr != rk
    ei = mis.nonzero()
    t_, c_, j_, l_, i_, h_ = ei.unbind(1)
    xr = (val[rk[mis]] ^ val[pr[mis]]).long()
    escl = _gvesc_e(t_, c_, j_, l_, 2 * i_ + h_, xr, T, geo, dev)
    if escl is None:
        if verbose:
            print("STATS_E %s escape list overflow (esc=%.5f)" % (layer, len(ei) / n))
        return None

    L = _substreams(_to_segments(sym_len[php], T, KC, geo), ilp)                 # [NSEG, ilp, 32, NP]
    V = _substreams(_to_segments(sym_code[php], T, KC, geo), ilp)
    nseg = L.shape[0]
    wpp = _wpp(ilp)                                                              # words per piece: 16 B, or 4 B at ilp 4
    off = torch.cumsum(L, 3) - L
    tot = L.sum(3)                                                               # [NSEG, ilp, 32]
    P = (tot.flatten(1).max(1).values + 32 * wpp - 1) // (32 * wpp)              # pieces per lane per substream
    soff = torch.zeros(nseg + 1, dtype=torch.long, device=dev)                   # in words (4 B)
    soff[1:] = torch.cumsum(P * 32 * ilp * wpp, 0)
    nwords = int(soff[-1])
    wd, sh = off >> 5, off & 31
    lane = torch.arange(32, device=dev)[None, None, :, None]
    sub = torch.arange(ilp, device=dev)[None, :, None, None]
    rot = ((lane >> 3) & 3) if wpp == 4 else 0
    gword = lambda w: soff[:-1, None, None, None] + ((sub * P[:, None, None, None] + w // wpp) * 32 + lane) * wpp + ((w % wpp) ^ rot)
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
    total_bits = 8 * n + stream_bits + 32 * (nseg + 1) + 32 * 258 + 32 * escl.numel()
    st = dict(bits=total_bits / n, esc=len(ei) / n, maxpad=pad_bits / stream_bits, maxpad_bits=pad_bits / n,
              N=N, code_bits=float(tot.sum()) / n, secs=time.time() - t0)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        st["secs"] = time.time() - t0
    if verbose:
        print("STATS_E %s ilp%d bits=%.4f esc=%.5f maxpad=%.4f (code %.4f, pad %.4f bits/w, alphabet %d, len tab %s, %.2f s)"
              % (layer, ilp, st["bits"], st["esc"], st["maxpad"], 8 + st["code_bits"], st["maxpad_bits"], N,
                 "".join(map(str, tab)), st["secs"]))
    return dict(plane=planeb, lut=lut32, stream=stream, soff=soff.int(), esc=escl, ilp=ilp, stats=st)


def unpack_e(pk, T, KC, geo, ED=16):
    """v3 reference decoder: the device routine's logic per lane and substream (64-bit buffer, refill at step start when
    (V & 0x60) == 0, length = tab[lo & 7] (no memory on the chain), value = lut[lo & 0xff], V -= length), all lanes of
    all segments at once. Replays the per-substream smem ring protocol (ED words per lane in total, ED / K slots per
    substream; prologue: 3 rounds of NI = 4 / K predicated issues per substream + commit; per unit: one round + commit +
    wait_group(2)) and asserts every refill word (incl. the kernel's prefetched ones) is read from a landed, not yet
    overwritten piece. Returns bf16 bit patterns [T, KC, 4, 32, 8]."""
    dev = pk["stream"].device
    wr, wk, ku, s, rt = _geo(geo)
    K = pk["ilp"]
    NS = 4 // K
    M32 = 0xffffffff
    words = pk["stream"].long() & M32
    lut = pk["lut"].long() & M32
    tab = torch.tensor([((int(lut[256 + i // 4]) >> (8 * (i % 4))) & 255) for i in range(8)], device=dev)
    lut = lut[:256]
    soff = pk["soff"].long()
    nseg = soff.numel() - 1
    wpp = _wpp(K)
    P = (soff[1:] - soff[:-1]) // (32 * K * wpp)
    lane = torch.arange(32, device=dev)[None, None, :]
    sub = torch.arange(K, device=dev)[None, :, None]
    rot = ((lane >> 3) & 3) if wpp == 4 else 0
    z = torch.zeros(nseg, K, 32, dtype=torch.long, device=dev)
    lo, hi, Vr, ptr, iss = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    NU = ku * rt
    E = torch.zeros(nseg, K, 32, NU, NS, 4, dtype=torch.long, device=dev)
    Pl = P[:, None, None].expand_as(z)
    groups = []
    ES, NI = ED // K, NS                                                         # slots per substream, issues per unit

    def issue():
        nonlocal iss
        for _ in range(NI):
            go = (iss < Pl) & (iss < ptr // wpp + ES)
            iss = iss + go.long()
        groups.append(iss.clone())

    for _ in range(3):
        issue()
    for un in range(NU):
        issue()
        landed = groups[len(groups) - 1 - 2]
        for t in range(NS):
            need = (Vr & 0x60) == 0
            pc = ptr // wpp                                                    # ptr counts words
            inb = pc < Pl
            rd = need & inb                                                    # words the decode uses (other reads are masked)
            assert bool((pc < landed)[rd].all()), "ring: piece not landed"
            assert bool((pc + ES > iss - 1)[rd].all()), "ring: slot overwritten"
            gi = soff[:-1, None, None] + ((sub * P[:, None, None] + pc) * 32 + lane) * wpp + ((ptr % wpp) ^ rot)
            w = torch.where(need & inb, words[gi.clamp(max=words.numel() - 1)], 0)
            at = Vr & 31
            lo = torch.where(need, lo | ((w << at) & M32), lo)
            hi = torch.where(need, hi | ((w >> (32 - at)) * (at > 0)), hi)
            Vr = torch.where(need, (Vr + 32) & M32, Vr)
            ptr = ptr + need.long()
            for q in range(4):
                n_ = tab[lo & 7]
                E[..., un, t, q] = lut[lo & 0xff]
                lo, hi = ((lo >> n_) | (hi << (32 - n_))) & M32, hi >> n_
                Vr = (Vr - n_) & M32
    # [nseg, K, 32, NU, NS, 4] -> [nseg, 32, NU, 4 j, 4 i] -> fragment order
    E = E.permute(0, 2, 3, 1, 4, 5).reshape(nseg, 32, NU, 4, 4)
    gx = T // (wr * rt)
    E = E.view(s, gx, wk, wr, 32, ku, rt, 4, 4).permute(1, 3, 6, 0, 2, 5, 7, 4, 8).reshape(T, KC, 4, 32, 4)
    pl = pk["plane"].long()
    pb = pl.view(T, KC, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 4, 32, 8)
    lowh, hih = pb[..., 0::2], pb[..., 1::2]
    dup0, dup1 = lowh | (lowh << 8), hih | (hih << 8)
    m = 0x807f
    v0 = (E & ~m & 0xffff) | (dup0 & m)
    v1 = ((E >> 16) & ~m & 0xffff) | (dup1 & m)
    vals = torch.stack((v0, v1), -1).reshape(T, KC, 4, 32, 8)
    esc = pk["esc"].long() & M32
    ent = esc[esc != M32]
    if len(ent):
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
        T, KC = fr.shape[:2]
        for ilp in (1, 2, 4):
            pk = pack_e(fr, geo, name, ilp=ilp)
            if pk is None:
                print("FAIL %s: escape overflow" % name); ok = False; continue
            t0 = time.time()
            got = unpack_e(pk, T, KC, geo)
            eq = torch.equal(got, _frag_bits(fr))
            print("   %-24s geo %-20s ilp%d %s  (unpack %.1f s)" % (name, geo, ilp, "bit-exact" if eq else "MISMATCH", time.time() - t0))
            ok &= eq
        pk = pack_e(fr, geo, name, verbose=False)
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
// ---- codec E v3 device decode (proj_rsi/tools/CODEC_E_LAYOUT.md v3) ------------------------------------------------
// K = 1, 2 or 4 independent substreams per lane; substream s owns steps j in [s*4/K, (s+1)*4/K) of every unit.
// Per lane and substream: lo, hi = 64-bit bit buffer (next code at bit 0 of lo); V = valid bits mod 128 (bits 8+ carry
// garbage from V -= len; only (V & 0x60) and (V & 31) are read); pb = bytes taken from the substream's lane stream;
// iss = words issued into the ring; nw = prefetched refill word.
// Code length: the first 3 bits of every code determine its length, so len = byte (lo & 7) of the 8-byte table (t0, t1):
// LOP3 + PRMT. The bit-position chain (LOP3 -> PRMT -> SHF) has no memory access. The value (both exponents) comes from
// lut[lo & 0xff] (LDS, off the chain); A = lop3(e, 0x807f807f, plane bytes, 0xb8) (bits 0-6 and 15 from the plane).
// Stream: per segment, K substream regions of P * 32 words; word q of lane l of substream s at soff + (s * P + q) * 32 + l
// (4 B pieces: coalesced 128 B per warp, padding to a word). Ring per warp and substream: ES = ED / K word slots x 32
// lanes ([s][slot][lane], 128 B per slot: one bank per lane, conflict-free); word q in slot q % ES.
// Per unit: ce_issue for each substream (NI = 4 / K predicated 4 B cp.async), one commit_group, wait_group(2) (the
// prologue issues 3 rounds); then ce_unit. A substream's first refill word of the unit is read after the wait, the
// others are prefetched one step ahead (same landed set, no issue in between).
struct CeS { uint32_t lo, hi, V, pb, iss, nw; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f): immLut b ? c : a = 0xb8
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// smem address of the word at substream byte pb: rl = ring_s + lane * 4, ring_s aligned to ES * 128
template <int ES>
__device__ __forceinline__ uint32_t ce_addr(uint32_t rl, uint32_t pb) { return rl | ((pb << 5) & ((ES - 1) << 7)); }
// NI predicated 4 B cp.async; gsrc = stream + (soff[gw] + s * P * 32 + lane) * 4 (word 0 of this lane and substream)
template <int ES, int NI>
__device__ __forceinline__ void ce_issue(CeS& b, uint32_t P, const uint8_t* gsrc, uint32_t rl) {
#pragma unroll
  for (int k = 0; k < NI; k++) {
    const bool go = b.iss < P && b.iss < (b.pb >> 2) + ES;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4; }\n"
                 :: "r"(rl | ((b.iss % ES) << 7)), "l"(gsrc + (size_t)b.iss * 128), "r"((uint32_t)go));
    b.iss += go;
  }
}
__device__ __forceinline__ void ce_refill(CeS& b, uint32_t w) {
  const bool need = (b.V & 0x60u) == 0u;
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// one unit: the K substreams' chains interleaved (independent), 16 A registers in r6 order A[4 j + i];
// rl = ring of substream 0 + lane * 4 (substream s at + s * ES * 128); LS = LUT index shift (2 single, 7 lane-replicated)
template <int ES, int LS, int K>
__device__ __forceinline__ void ce_unit(CeS* b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint4 q0, uint4 q1, uint32_t* A) {
  constexpr int NS = 4 / K;
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int t = 0; t < NS; t++) {
#pragma unroll
    for (int s = 0; s < K; s++) {
      const uint32_t rs = rl + s * ES * 128;
      ce_refill(b[s], t == 0 ? ce_lds(ce_addr<ES>(rs, b[s].pb)) : b[s].nw);
      if (t + 1 < NS) b[s].nw = ce_lds(ce_addr<ES>(rs, b[s].pb));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int s = 0; s < K; s++) {
        const int j = s * NS + t;
        const uint32_t len = ce_prmt(t0, t1, b[s].lo & 7u);
        const uint32_t e = ce_lds(lut + ((b[s].lo & 0xffu) << LS));
        b[s].lo = __funnelshift_r(b[s].lo, b[s].hi, len); b[s].hi = __funnelshift_r(b[s].hi, 0u, len); b[s].V -= len;
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
