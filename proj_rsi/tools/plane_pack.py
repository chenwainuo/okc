"""Two-plane 12-bit lossless packing of a bf16 weight W [N][K] for the mx CUTLASS RS-wgmma mainloop (proj_rsi/candidates/mx).

Code (r6's 12-bit codec, dec1 variant): a bf16 w = s | h7 << 8 | lo (s = sign bit 15, h7 = bits 8-14, lo = low byte =
exponent LSB + 7 mantissa bits). One window base B per layer (8 consecutive h7 codes = 16 exponents, B <= 120);
off = h7 - B in [0, 8). Weights outside the window ("escapes", ~1e-4 of the weights, all tiny on the reference
weights) are coded with off = 0 and corrected exactly in the GEMM: the consumer initialises its fp32 accumulators with
x[tok, k] * (w - w_dec) for every escape of its tile (fp32, exact for one-hot probe rows).

Layout: fragment order of the sm90 wgmma m64nNk16 register-A operand. W is cut into 64-row units u = n // 64 and
64-wide k tiles kt; each (u, kt) is one contiguous 6144-byte chunk of buf [N/64, K/64, 6144]:
  bytes [0, 4096): plane8.  offset (j * 128 + t) * 16 + q, j in {0, 1}, t = thread of the warpgroup (warp w = t / 32,
                   lane l = t % 32), q in [0, 16): k-block kb = 2 j + q / 8, value v = q % 8 of that k-block's 8-value
                   fragment (register i = v / 2, half h = v % 2): row r = 16 w + l / 4 + 8 (i & 1),
                   k = 16 kb + 2 (l % 4) + h + 8 (i >> 1). Byte = lo of W[64 u + r][64 kt + k].
  bytes [4096, 6144): plane4. offset 4096 + t * 16 + 4 kb + b: 32-bit word C(t, kb), dec1 order: byte b: bits 0-2 off of
                   value b, bit 7 sign of value b, bits 4-6 off of value 4 + b, bit 3 sign of value 4 + (b + 3) % 4.
One thread's k-tile is 2 x LDS.128 (plane8) + 1 x LDS.128 (plane4); a warp reads 512 contiguous bytes per LDS.128.

Merge (per k-block, 8 values, 9 ops = dec1): hl = (C & 0x87878787) + B * 0x01010101; hh = (rotr(C, 4) & 0x87878787) +
B * 0x01010101; a0 = prmt(L0, hl, 0x5140), a1 = prmt(L0, hl, 0x7362), a2 = prmt(L1, hh, 0x5140), a3 = prmt(L1, hh, 0x7362)
with L0, L1 the two plane8 words of the k-block.

Escapes: per 64-row unit u (sorted by k), entries esc_off[u] .. esc_off[u + 1] of esc_rk (int32: r << 16 | k, r = row in unit) and
esc_d (float32: w - w_dec, exact in fp64 then rounded to fp32).

  python proj_rsi/tools/plane_pack.py [--device cuda]   # packs the reference weights, proves losslessness, prints stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

UNIT = 64     # rows per unit (one warpgroup's wgmma M)
KT = 64       # k per tile
PLANES = 6144  # plane bytes per (unit, k tile)
HDR = 64       # escape header per (unit, k tile): u32 count | 15 x u32 (row << 22 | (k % 64) << 16 | (w ^ w_dec) 16 bits)
HMAX = 15
CHUNK = PLANES + HDR  # bytes per (unit, k tile)


def _frag_index(dev):
    """(r, k) of plane8 byte offsets 0..4095 and (r, k) of each (t, kb, value) for plane4."""
    o = torch.arange(4096, device=dev)
    j, t, q = o // 2048, (o // 16) % 128, o % 16
    kb, v = 2 * j + q // 8, q % 8
    i, h = v // 2, v % 2
    w, l = t // 32, t % 32
    r = 16 * w + l // 4 + 8 * (i & 1)
    k = 16 * kb + 2 * (l % 4) + h + 8 * (i >> 1)
    return r, k


def _frag4_index(dev):
    """(r, k) for each (t, kb, v): [128, 4, 8]"""
    t = torch.arange(128, device=dev)[:, None, None]
    kb = torch.arange(4, device=dev)[None, :, None]
    v = torch.arange(8, device=dev)[None, None, :]
    i, h = v // 2, v % 2
    w, l = t // 32, t % 32
    r = 16 * w + l // 4 + 8 * (i & 1)
    k = 16 * kb + 2 * (l % 4) + h + 8 * (i >> 1)
    return r.expand(128, 4, 8), k.expand(128, 4, 8)


def choose_base(h7):
    cnt = torch.bincount(h7.flatten(), minlength=128)
    win = cnt.unfold(0, 8, 1).sum(1)  # win[b] = count in [b, b + 8)
    win = win[:121]                   # B <= 120 (dec1's byte add must not carry into the sign bit)
    ba = int(win.argmax())
    bt = min(max(int(h7.max()) - 7, 0), 120)  # window that contains the largest magnitudes: escapes are then all tiny
    tot = int(cnt.sum())
    return bt if tot - int(win[bt]) <= 2 * (tot - int(win[ba])) + 16 else ba


def pack(w, base=None):
    """W bf16 [N, K] -> dict(buf uint8 [N/64, K/64, 6144], base, esc_off int32 [N/64 + 1], esc_rk int32 [E], esc_d f32 [E])"""
    N, K = w.shape
    assert N % UNIT == 0 and K % KT == 0 and w.dtype == torch.bfloat16
    dev = w.device
    b = w.contiguous().view(torch.int16).int() & 0xffff
    h7, sg, lo = (b >> 8) & 127, b >> 15, b & 255
    B = choose_base(h7) if base is None else base
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    off = torch.where(inw, d, torch.zeros_like(d))
    U, T = N // UNIT, K // KT
    # [U, 64, T, 64] -> [U, T, 64(r), 64(k)]
    tile = lambda a: a.view(U, UNIT, T, KT).permute(0, 2, 1, 3)
    lo_t, off_t, sg_t = tile(lo), tile(off), tile(sg)
    r8, k8 = _frag_index(dev)
    p8 = lo_t[:, :, r8, k8].to(torch.uint8)  # [U, T, 4096]
    r4, k4 = _frag4_index(dev)
    o4, s4 = off_t[:, :, r4, k4].long(), sg_t[:, :, r4, k4].long()  # [U, T, 128, 4, 8]
    C = torch.zeros(o4.shape[:-1], dtype=torch.long, device=dev)
    for bb in range(4):
        C |= o4[..., bb] << (8 * bb)
        C |= s4[..., bb] << (8 * bb + 7)
        C |= o4[..., 4 + bb] << (8 * bb + 4)
        C |= s4[..., 4 + (bb + 3) % 4] << (8 * bb + 3)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()  # [U, T, 128, 4]
    p4 = C.contiguous().view(torch.uint8).view(U, T, 2048)
    # escapes
    wdec = (sg << 15) | ((off + B) << 8) | lo
    ei = (~inw).nonzero()
    n_, k_ = ei[:, 0], ei[:, 1]
    wt = b[n_, k_].short().view(torch.bfloat16).double() if len(ei) else torch.zeros(0, dtype=torch.double, device=dev)
    wd = wdec[n_, k_].short().view(torch.bfloat16).double() if len(ei) else torch.zeros(0, dtype=torch.double, device=dev)
    u_ = n_ // UNIT
    order = torch.argsort(u_ * (1 << 40) + k_ * (1 << 8) + (n_ % UNIT))  # per unit, sorted by k (the kernel copies x[:, k] from the streaming B tiles in k order)
    n_, k_, u_ = n_[order], k_[order], u_[order]
    esc_rk = (((n_ % UNIT) << 16) | k_).int()
    esc_d = (wt - wd)[order].float()
    esc_off = torch.zeros(U + 1, dtype=torch.int32, device=dev)
    esc_off[1:] = torch.cumsum(torch.bincount(u_, minlength=U), 0).int()
    # per-chunk escape headers (the kernel reads them from smem with the planes: no global list, no latency)
    hdr = torch.zeros(U, T, HDR, dtype=torch.uint8, device=dev)
    if len(n_):
        kt_ = k_ // KT
        key = u_ * T + kt_
        cnt = torch.bincount(key, minlength=U * T)
        if int(cnt.max()) > HMAX:
            return None  # a chunk has more escapes than its header holds
        first = torch.cumsum(cnt, 0) - cnt
        slot = torch.arange(len(key), device=dev) - first[key]  # key is sorted (u, k)
        hv = hdr.view(U * T, HDR)
        hv[:, 0:4] = cnt.int().contiguous().view(torch.uint8).view(U * T, 4)
        xr = (b[n_, k_] ^ wdec[n_, k_]) & 0xffff
        ent = (((n_ % UNIT) << 22) | ((k_ % KT) << 16) | xr).int().contiguous().view(torch.uint8).view(-1, 4)
        for bb in range(4):
            hv[key, 4 + 4 * slot + bb] = ent[:, bb]
    buf = torch.cat((p8, p4, hdr), 2).contiguous()
    return {"buf": buf, "base": B, "esc_off": esc_off, "esc_rk": esc_rk, "esc_d": esc_d, "N": N, "K": K}


def esc_max(pk, tm, splits):
    """largest number of escapes one CTA (tm rows, one of `splits` k ranges, mx::Kernel's split rule) must correct"""
    T = pk["K"] // KT
    off, rk = pk["esc_off"].long(), pk["esc_rk"].long()
    U = len(off) - 1
    u = torch.repeat_interleave(torch.arange(U, device=rk.device), off[1:] - off[:-1])
    kt = (rk & 0xffff) // KT
    bounds = torch.tensor([s * T // splits for s in range(splits + 1)], device=rk.device)
    sp = torch.searchsorted(bounds, kt, right=True) - 1
    key = (u // (tm // UNIT)) * splits + sp
    return int(torch.bincount(key).max()) if len(key) else 0


def _prmt(a, b, sel):
    """bitwise emulation of prmt.b32 (default mode) on int64 tensors holding u32 values"""
    out = torch.zeros_like(a)
    for i in range(4):
        s = (sel >> (4 * i)) & 7
        src = a if s < 4 else b
        out |= ((src >> (8 * (s % 4))) & 255) << (8 * i)
    return out


def decode(pk):
    """independent decoder mirroring the kernel's dec1 (register-level ops on the packed bytes) -> bf16 W_dec [N, K]"""
    buf, B = pk["buf"], pk["base"]
    U, T, _ = buf.shape
    N, K = U * UNIT, T * KT
    dev = buf.device
    words = buf.view(torch.int32).long() & 0xffffffff  # [U, T, 1552]
    p8 = words[:, :, :1024].view(U, T, 2, 128, 4)      # [j][t][4 words]: words 0,1 = kb 2j (L0, L1), 2,3 = kb 2j+1
    p4 = words[:, :, 1024:1536].view(U, T, 128, 4)     # [t][kb]
    Tb = B * 0x01010101
    vals = torch.zeros(U, T, 128, 4, 8, dtype=torch.long, device=dev)
    for kb in range(4):
        L0 = p8[:, :, kb // 2, :, 2 * (kb % 2)]
        L1 = p8[:, :, kb // 2, :, 2 * (kb % 2) + 1]
        C = p4[:, :, :, kb]
        hl = ((C & 0x87878787) + Tb) & 0xffffffff
        rot = ((C >> 4) | (C << 28)) & 0xffffffff
        hh = ((rot & 0x87878787) + Tb) & 0xffffffff
        a = [_prmt(L0, hl, 0x5140), _prmt(L0, hl, 0x7362), _prmt(L1, hh, 0x5140), _prmt(L1, hh, 0x7362)]
        for i in range(4):
            vals[:, :, :, kb, 2 * i] = a[i] & 0xffff
            vals[:, :, :, kb, 2 * i + 1] = a[i] >> 16
    r4, k4 = _frag4_index(dev)  # [128, 4, 8]
    out = torch.zeros(U, T, UNIT, KT, dtype=torch.long, device=dev)
    out[:, :, r4, k4] = vals
    out = out.permute(0, 2, 1, 3).reshape(N, K)
    out = torch.where(out >= 2 ** 15, out - 2 ** 16, out).short()
    return out.view(torch.bfloat16)


def apply_escapes(wdec, pk):
    """fp64 W reconstructed from W_dec + escapes"""
    w = wdec.double().clone()
    off, rk, d = pk["esc_off"], pk["esc_rk"].long(), pk["esc_d"].double()
    U = len(off) - 1
    u = torch.repeat_interleave(torch.arange(U, device=w.device), (off[1:] - off[:-1]).long())
    n, k = u * UNIT + (rk >> 16), rk & 0xffff
    w[n, k] += d
    return w


def apply_headers(wdec, pk):
    """W from W_dec patched with the per-chunk header XORs (what the kernel does to its A fragments)"""
    buf = pk["buf"]
    U, T, _ = buf.shape
    h = buf[:, :, PLANES:]
    cnt = h[:, :, 0:4].contiguous().view(torch.int32)[..., 0]
    ent = h[:, :, 4:64].contiguous().view(torch.int32).long() & 0xffffffff
    w = wdec.view(torch.int16).int().clone() & 0xffff
    slot = torch.arange(HMAX, device=buf.device)
    m = slot[None, None, :] < cnt[..., None]
    uu, tt, ss = m.nonzero(as_tuple=True)
    ee = ent[uu, tt, ss]
    rr, kk = uu * UNIT + ((ee >> 22) & 63), tt * KT + ((ee >> 16) & 63)
    w[rr, kk] ^= (ee & 0xffff).int()
    return torch.where(w >= 2 ** 15, w - 2 ** 16, w).short().view(torch.bfloat16).double()


def verify(w, pk):
    wdec = decode(pk)
    exact = torch.equal(apply_escapes(wdec, pk), w.double()) and torch.equal(apply_headers(wdec, pk), w.double())
    return exact, wdec


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "muse_bf16_proj"))
    import reference
    ws = reference.generate_weights(reference.WEIGHTS_SEED if hasattr(reference, "WEIGHTS_SEED") else 5151, device=dev)
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        w = ws[layer]
        pk = pack(w)
        ok, wdec = verify(w, pk)
        E = len(pk["esc_rk"])
        h7 = (w.view(torch.int16).int() >> 8) & 127
        above = int((h7 >= pk["base"] + 8).sum())
        nd = int((wdec != w).sum())
        print("%-9s N=%d K=%d base=%d escapes=%d (%.2e, above-window %d, max |d| %.3e, max |w| %.3e) mismatches-before-fix=%d "
              "bytes %.1f MB (bf16 %.1f MB) LOSSLESS=%s" % (
                  layer, pk["N"], pk["K"], pk["base"], E, E / w.numel(), above, float(pk["esc_d"].abs().max()) if E else 0.0,
                  float(w.float().abs().max()), nd, pk["buf"].numel() / 1e6, w.numel() * 2 / 1e6, ok), flush=True)
        per_unit = (pk["esc_off"][1:] - pk["esc_off"][:-1])
        print("          escapes per 64-row unit: mean %.1f max %d" % (per_unit.float().mean(), int(per_unit.max())), flush=True)


if __name__ == "__main__":
    main()
