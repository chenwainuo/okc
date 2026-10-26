"""Exponent-codec study prototype (pure numpy + python ints; no torch, no GPU).

Every design keeps r6's low byte (exponent LSB + 7 mantissa bits, 8 bits, incompressible) and the sign (1 bit, raw,
fixed plane) and codes only h7 = the upper 7 exponent bits, so total bits/weight = 9 + (h7 bits incl. all overheads).

Geometry is r6's: a unit is 16 rows x 64 k = 1024 weights in mma.sync m16n8k16 A-fragment order, 32 weights per
lane, 8 per k16 step (4 A registers = 4 weight pairs). The prototype maps unit weight j to lane j // 32, slot j % 32
(the real permutation is irrelevant: weights are iid). A warp streams U consecutive units (default 16).

Designs (h7 part only):
  R    fixed 2-bit offset in a 4-value contiguous window (r6 style, the analyst's 11-bit route); out-of-window values
       through an XOR escape list, 32 bits/entry (r6's per-warp list format).
  A    two-class flag code: '0'+2 bits for ranks 0-3, '1'+4 bits for ranks 4-19; rarer -> escape list.
  B    block-adaptive: per lane, per 8 weights a 2-bit header picks width 1/2/3 bits (ranks <2/<4/<8) or 0 (all
       rank 0); ranks >= 8 -> escape list.  (The brief's per-16 header over widths 2/3/4 is computed analytically
       as B16 and is > 2 bits by construction.)
  C    single-symbol canonical Huffman over ranks 0..R-1, lengths limited to 4 (16-entry LUT); rarer -> escape list.
  D    pair Huffman: one symbol per A register (2 weights), alphabet R x R ranks, lengths limited to 7 (128-entry
       LUT whose entry is the h7 pair already at bits 8-14 / 24-30 plus the code length in bits 0-4);
       rarer -> escape list.
  E    as D (but lengths <= 8, 256-entry LUT) and the coded field is the full 8-bit exponent (the exponent LSB moves out of the low byte, the sign moves
       in: plane byte = sign | 7 mantissa bits). Costs nothing in bits (the LSB is not uniform given h7) and saves
       the per-pair sign shift on the device: A register = LOP3(LUT entry, PRMT-duplicated plane bytes, 0x807f807f).
Variable-length designs (A-D) use one interleaved stream per warp: each lane holds a <= 64-bit bit buffer; at every
step boundary (S weights) a lane whose buffer holds fewer than M bits (M = max code bits of one step <= 32) takes the
next 32-bit word, lanes served in lane order (ballot + popc on the device). No per-lane lengths are stored; the
encoder replays the decoder's refill schedule. Overheads counted: whole stream words (incl. tail padding), escape
entries, one 32-bit stream offset per warp. Also reported: the per-lane-stream layout (each lane's codes contiguous,
padded to a word, 32 x 16-bit lane starts per warp stream; refill without ballot, but per-lane gathered loads). Rank/LUT tables are per matrix and reported separately (they amortise
over the matrix).

Round trip: pack -> unpack -> compare all 16 bits of every weight, for every design, bit exact.
usage: python3 -m proj_rsi.tools.codec_proto [--n 1048576] [--K 8192 1024 2048 3072] [--seed 5151]
"""
import argparse
import heapq
import time

import numpy as np

UNIT, LANES, WPL = 1024, 32, 32
ESC_BITS = 32


def bf16_gauss(n, K, seed):
    """n bf16 bit patterns of randn / sqrt(K), round to nearest even (the reference weight distribution)."""
    x = (np.random.default_rng(seed).standard_normal(n) / np.sqrt(K)).astype(np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def rank_tables(h7):
    c = np.bincount(h7, minlength=256)
    val = np.argsort(-c, kind="stable")
    rank = np.empty(len(c), np.int64)
    rank[val] = np.arange(len(c))
    return rank, val


def package_merge(p, L):
    """optimal prefix-code lengths with max length L (p > 0)."""
    n = len(p)
    if n == 1:
        return [1]
    assert (1 << L) >= n
    leaves = sorted((float(q), (i,)) for i, q in enumerate(p))
    cur = list(leaves)
    for _ in range(L - 1):
        pk = [(cur[2 * i][0] + cur[2 * i + 1][0], cur[2 * i][1] + cur[2 * i + 1][1]) for i in range(len(cur) // 2)]
        cur = sorted(leaves + pk, key=lambda t: t[0])
    ln = [0] * n
    for _, syms in cur[:2 * n - 2]:
        for s in syms:
            ln[s] += 1
    return ln


def canonical_lsb(lengths):
    """canonical codes, bit-reversed so the stream is read LSB first: code value v of length l occupies the low l
    bits of the peek window."""
    order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
    code, prev, out = 0, None, [0] * len(lengths)
    for s in order:
        l = lengths[s]
        if prev is not None:
            code = (code + 1) << (l - prev)
        prev = l
        out[s] = int(format(code, "0%db" % l)[::-1], 2)
    return out


def lut(codes, lengths, peek):
    t = [-1] * (1 << peek)
    for s, (c, l) in enumerate(zip(codes, lengths)):
        for hi in range(1 << (peek - l)):
            t[c | (hi << l)] = s
    assert min(t) >= 0
    return t


class Design:
    """h7 coder: rank alphabet + per-step code; escapes for ranks >= self.R (coded as rank 0, fixed by XOR)."""
    S, M = 8, 32
    name = "?"

    def encode_step(self, r):          # ranks (len S, all < R) -> list of (value, nbits)
        raise NotImplementedError

    def decode_step(self, buf):        # -> (ranks list, consumed bits)
        raise NotImplementedError


class FlagCode(Design):
    name, S, M = "A two-class flag", 4, 20

    def __init__(self, p):
        self.R = 20

    def encode_step(self, r):
        return [((x << 1), 3) if x < 4 else ((((x - 4) << 1) | 1), 5) for x in r]

    def decode_step(self, buf):
        out, n = [], 0
        for _ in range(self.S):
            if (buf >> n) & 1:
                out.append(4 + ((buf >> (n + 1)) & 15)); n += 5
            else:
                out.append((buf >> (n + 1)) & 3); n += 3
        return out, n


class BlockWidth(Design):
    name, S, M = "B block-adaptive /8", 8, 26

    def __init__(self, p):
        self.R = 8

    def encode_step(self, r):
        m = max(r)
        if m == 0:
            return [(3, 2)]
        w = 1 if m < 2 else 2 if m < 4 else 3
        return [(w - 1, 2)] + [(x, w) for x in r]

    def decode_step(self, buf):
        h = buf & 3
        if h == 3:
            return [0] * self.S, 2
        w = h + 1
        return [(buf >> (2 + w * i)) & ((1 << w) - 1) for i in range(self.S)], 2 + w * self.S


class Huff4(Design):
    name, S, M = "C huffman<=4, 16-LUT", 8, 32

    def __init__(self, p):
        best = None
        for R in range(2, 17):
            ln = package_merge(np.maximum(p[:R], 1e-12), 4)
            cost = float(np.dot(p[:R], ln)) + float(p[R:].sum()) * ESC_BITS
            if best is None or cost < best[0]:
                best = (cost, R, ln)
        _, self.R, self.len = best
        self.code = canonical_lsb(self.len)
        self.t = lut(self.code, self.len, 4)

    def encode_step(self, r):
        return [(self.code[x], self.len[x]) for x in r]

    def decode_step(self, buf):
        out, n = [], 0
        for _ in range(self.S):
            s = self.t[(buf >> n) & 15]
            out.append(s); n += self.len[s]
        return out, n


class PairHuff(Design):
    # M = 28, lengths <= 7: the device keeps the buffer pre-shifted by 2 bits (LUT byte offset = lo & 0x1fc), so a
    # 64-bit buffer holds at most 62 stream bits: refill below 28 valid bits (+32 -> <= 59), 4 pairs <= 28 bits.
    name, S, M, LMAX = "D pair huffman<=7, 128-LUT", 8, 28, 7

    def __init__(self, p):
        best = None
        for R in range(2, 17):
            if R * R > (1 << self.LMAX):
                break
            pp = np.outer(p[:R], p[:R]).ravel()
            ln = package_merge(np.maximum(pp, 1e-15), self.LMAX)
            cost = float(np.dot(pp, ln)) / 2 + float(p[R:].sum()) * ESC_BITS
            if best is None or cost < best[0]:
                best = (cost, R, ln)
        _, self.R, self.len = best
        self.code = canonical_lsb(self.len)
        self.t = lut(self.code, self.len, self.LMAX)

    def encode_step(self, r):
        return [(self.code[r[i] * self.R + r[i + 1]], self.len[r[i] * self.R + r[i + 1]]) for i in range(0, self.S, 2)]

    def decode_step(self, buf):
        out, n = [], 0
        for _ in range(self.S // 2):
            s = self.t[(buf >> n) & ((1 << self.LMAX) - 1)]
            out += [s // self.R, s % self.R]; n += self.len[s]
        return out, n


class PairHuffExp(PairHuff):
    # no pre-shift on the device (LUT address = LEA of the low byte), so the 64-bit buffer holds 63 bits: M = 32
    name, M, LMAX = "E pair huffman<=8 on exp8, 256-LUT", 32, 8
    field = "exp8"


def interleave(design, lane_ranks):
    """lane_ranks [32, W] (all < R) -> list of 32-bit words (one warp stream), replaying the decoder's refill."""
    S, M = design.S, design.M
    steps = lane_ranks.shape[1] // S
    lane_bits, lane_len, cons = [], [], np.zeros((LANES, steps), np.int64)
    for l in range(LANES):
        v, n = 0, 0
        for s in range(steps):
            n0 = n
            for c, b in design.encode_step(lane_ranks[l, s * S:(s + 1) * S].tolist()):
                v |= c << n; n += b
            cons[l, s] = n - n0
            assert n - n0 <= M
        lane_bits.append(v); lane_len.append(n)
    V, fetched, words = [0] * LANES, [0] * LANES, []
    for s in range(steps):
        for l in range(LANES):
            if V[l] < M:
                words.append((lane_bits[l] >> (32 * fetched[l])) & 0xFFFFFFFF)
                fetched[l] += 1; V[l] += 32
        for l in range(LANES):
            V[l] -= int(cons[l, s]); assert V[l] >= 0
    interleave.lane_words = sum((n + 31) // 32 for n in lane_len)
    return words


def deinterleave(design, words, W):
    S, M = design.S, design.M
    out = np.zeros((LANES, W), np.int64)
    buf, V, k = [0] * LANES, [0] * LANES, 0
    for s in range(W // S):
        for l in range(LANES):
            if V[l] < M:
                buf[l] |= words[k] << V[l]; k += 1; V[l] += 32
        for l in range(LANES):
            r, n = design.decode_step(buf[l])
            out[l, s * S:(s + 1) * S] = r
            buf[l] >>= n; V[l] -= n
    assert k == len(words)
    return out


def run(w, U, designs):
    n = len(w)
    assert n % (UNIT * U) == 0
    low, sign, h7 = (w & 0xFF).astype(np.int64), (w >> 15).astype(np.int64), ((w >> 8) & 0x7F).astype(np.int64)
    rank, val = rank_tables(h7)
    rk = rank[h7]
    p = np.bincount(rk, minlength=256) / n
    res = {}
    nw = n // (UNIT * U)

    # R: fixed 2-bit contiguous window + escapes
    cov = [(np.isin(h7, np.arange(b, b + 4)).sum(), b) for b in range(125)]
    base = max(cov)[1]
    off = h7 - base
    esc = (off < 0) | (off > 3)
    code2 = np.where(esc, 0, off)                                       # packed plane: 2 bits per weight
    plane = (code2.reshape(-1, 16) << (2 * np.arange(16))).sum(1)       # 32-bit words
    elist = np.flatnonzero(esc)
    exor = (h7[elist] ^ base)                                           # decoder: h7 = base + code, then ^= xor
    dec = base + ((plane[:, None] >> (2 * np.arange(16))) & 3).ravel()
    dec[elist] ^= exor
    assert np.array_equal(dec, h7)
    bits = 2 * n + ESC_BITS * len(elist)
    res["R fixed 2-bit + esc32"] = (bits / n, len(elist) / n, "window %d..%d" % (base, base + 3))
    res["R' fixed 2-bit + esc16 (hyp.)"] = ((2 * n + 16 * len(elist)) / n, len(elist) / n, "compact list entries")

    # B16 (brief's version, analytic): 2-bit header per 16 weights per lane, width 2/3/4, escape mode = 4 + list
    m16 = rk.reshape(-1, 16).max(1)
    wd = np.where(m16 < 4, 2, np.where(m16 < 8, 3, 4))
    b16 = (2 + 16 * wd).sum() + ESC_BITS * (rk >= 16).sum()
    res["B16 header/16, widths 2/3/4 (analytic)"] = (b16 / n, float((rk >= 16).mean()), "")

    e8 = ((w >> 7) & 0xFF).astype(np.int64)
    rank8, val8 = rank_tables(e8)
    rk8 = rank8[e8]
    p8 = np.bincount(rk8, minlength=256) / n
    for D in designs:
        full = getattr(D, "field", "h7") == "exp8"
        f, fr, fv = (e8, rk8, val8) if full else (h7, rk, val)
        d = D(p8 if full else p)
        t0 = time.time()
        e = fr >= d.R
        r = np.where(e, 0, fr)
        ei = np.flatnonzero(e)
        xr = f[ei] ^ fv[0]                                              # placeholder rank 0 -> fix by XOR
        nwords, lwords, dec = 0, 0, np.empty(n, np.int64)
        for g in range(nw):
            seg = r[g * UNIT * U:(g + 1) * UNIT * U].reshape(U, LANES, WPL).transpose(1, 0, 2).reshape(LANES, U * WPL)
            words = interleave(d, seg)
            nwords += len(words)
            lwords += interleave.lane_words
            back = deinterleave(d, words, U * WPL)
            dec[g * UNIT * U:(g + 1) * UNIT * U] = back.reshape(LANES, U, WPL).transpose(1, 0, 2).ravel()
        h = fv[dec]
        h[ei] ^= xr
        wb = (sign << 15) | ((h << 7) | (low & 0x7F) if full else (h << 8) | low)
        assert np.array_equal(wb.astype(np.uint16), w), d.name
        bits = 32 * nwords + ESC_BITS * len(ei) + 32 * nw
        extra = "R=%d" % d.R
        if hasattr(d, "len"):
            extra += " lengths " + ",".join(map(str, d.len[:12])) + ("..." if len(d.len) > 12 else "")
        lbits = 32 * lwords + ESC_BITS * len(ei) + 32 * 16 * nw        # per-lane streams: 32 x 16-bit starts / warp
        if full:
            bits -= n                                                   # the exponent LSB left the 8-bit plane
            lbits -= n
        res[d.name] = (bits / n, len(ei) / n, extra + "  [per-lane-stream layout: %.3f]  (%.0fs)" % (9 + lbits / n, time.time() - t0))
    Hh = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return res, Hh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--K", type=int, nargs="+", default=[8192, 1024, 2048, 3072])
    ap.add_argument("--U", type=int, default=16, help="units per warp stream")
    ap.add_argument("--seed", type=int, default=5151)
    a = ap.parse_args()
    for K in a.K:
        w = bf16_gauss(a.n, K, a.seed)
        res, Hh = run(w, a.U, [FlagCode, BlockWidth, Huff4, PairHuff, PairHuffExp])
        print("K=%d  n=%d  H(h7)=%.3f -> entropy bound %.3f bits/weight   [round trip bit-exact: all designs]" % (K, a.n, Hh, 9 + Hh))
        for k, (b, er, x) in res.items():
            print("   %-38s beyond 9 %.3f  total %.3f  esc %.4f%%  %s" % (k, b, 9 + b, 100 * er, x))


if __name__ == "__main__":
    main()
