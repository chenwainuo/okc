_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))

def _c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    (T, KC) = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 65535
    (h7, sg) = (b >> 8 & 127, b >> 15)
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = (d << torch.tensor(_OS[dv], device=dev) | sg.long() << torch.tensor(_SS[dv], device=dev)).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 16843009 if dv else sum((B + i << 8 * i for i in range(8)))
    if not torch.equal(_unc12(data, tab, esc, dv), b):
        return None
    return (data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc)

def _unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    (T, KC) = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 4294967295
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = C[..., None] >> torch.tensor(_OS[dv], device=dev) & 7
    sgn = C[..., None] >> torch.tensor(_SS[dv], device=dev) & 1
    tb = torch.tensor([tab >> 8 * i & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | sgn << 7
    lob = L.permute(0, 1, 2, 4, 3)[..., None] >> 8 * torch.arange(4, device=dev) & 255
    val = hi << 8 | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        (t, c, j, l, vv, x) = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()

def _frag_bits(fr):
    return fr.reshape(*fr.shape[:2], 4, 32, 8).contiguous().view(torch.int16).int() & 65535

def _pack_nat(w):
    """pw10 _pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
    (N, K) = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()

def _rqpack(w):
    """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
    (N, K) = w.shape
    c = _c12(_pack_nat(w), 1)
    if c is None or N % 128:
        return None
    (data, tab, esc) = c
    (T, KC) = data.shape[:2]
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    return (lay, tab, esc)

def _rqunpack(lay, tab, esc, N, K):
    """independent check of the re-blocking: unpack [NB][KC][8][1536] with _unc12 and undo _pack_nat -> bf16 [N, K]"""
    (NB, KC) = lay.shape[:2]
    data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
    v = _unc12(data, tab, esc, 1)
    W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)
    return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)

def _rqesc(esc6, NB, KC, KS, dev):
    """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
    G = KS * NB * 8
    if len(esc6):
        (tile, ch, q, ln, v, xr) = [t.long() for t in esc6.t()]
        (nb, wt) = (tile // 8, tile % 8)
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = u << 17 | ln << 12 | q * 4 + v // 2 << 8 | v % 2 << 7 | xr
        gw = (ks * NB + nb) * 8 + wt
        o = torch.argsort(gw * 65536 + u)
        (gw, ent) = (gw[o], ent[o])
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
    else:
        cap = 32
    arr = torch.full((G * cap + 32,), 4294967295, dtype=torch.long, device=dev)
    if len(esc6):
        arr[gw * cap + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return (arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev))
