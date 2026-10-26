# Codec E: layout spec for pw17

Status: v1. Codec side by the implementer; the gv integration belongs to the analyst. Open questions are marked **Q**.

**Code:** `proj_rsi/tools/codec_e.py`
- `pack_e(fr, geo)`: the packer.
- `unpack_e(...)`: the reference unpacker.
- `CODEC_E_CUH`: the device routine.

## 1. What is stored per matrix

The input is r6's fragments: `fr = _pack(w)`, shape `[T, KC, 4, 32, 8]` bf16 (tile, chunk, step j, lane, value v). Nothing about r6's K permutation changes.

A unit is (tile t, chunk c). In each unit, lane l owns 32 weights in 16 A registers. Register `r = 4*j + i` (i = 0..3) holds the pair `(v = 2i, v = 2i+1)` of step j. The low half is v even and the high half is v odd, exactly as r6's `A[j][h][r]`.

### 1a. 8-bit plane: `plane uint8 [T, KC, 2, 32, 16]`
- This is r6's c12 low-byte pieces p0 and p1, unchanged in shape and order. Piece 0 holds steps 0–1 and piece 1 holds steps 2–3, 16 B per lane each. Load them exactly as r6 loads `q[0]` and `q[1]`.
- The byte content changes: byte = `sign << 7 | mantissa7`. The exponent LSB is gone from the plane.
- r6's third piece (the nibble plane, `q[2]`) is dropped. NL goes from 3 to 2.
- The plane word for register r is word `2*j + (i >> 1)` of the lane's 8 words: `q[j >> 1]`, component `(j & 1) * 2 + (i >> 1)`. The byte pair is selected by `i & 1`.

### 1b. Decode table: `lut uint32 [256]` (1 KB) per matrix
- `lut[x]` for every 8-bit peek x.
- Bits 0–4: code length (1..8). Bits 5–6: 0.
- Bits 7–14: exponent of the low half (v even).
- Bits 23–30: exponent of the high half (v odd).
- Bits 15–22 and 31: 0.
- The A register is `lop3(e, 0x807f807f, prmt(Lword, 0, i&1 ? 0x3322 : 0x1100), 0xe4)`. That takes bits 0–6 and bit 15 (and the same in the high half) from the duplicated plane bytes, and the rest from e.

### 1c. Code stream
- **Symbols.** One prefix code per A register (pair symbol), at most 8 bits, read LSB first.
- **Lane stream order** within a warp segment: `for u in 0..KU-1: for jt in 0..RT-1: for r in 0..15`. This is r6's walk order (`step(u)` runs j over RT), with registers in A order.
- **Words.** The lane stream is a bit string. Bit b sits in 32-bit word `b >> 5`, at bit `b & 31` (little-endian). Words are grouped in 16 B pieces: piece k = words 4k..4k+3.
- **Segment.** A segment is one warp's range in r6's geometry `(WR, WK, KU, S, RT, XS)`: RT tiles × KU chunks. Its id `gw` is r6's `_gvesc` id:
  `gw = ((c // (WK*KU)) * GX + t // (WR*RT)) * NW + ((c // KU) % WK) * WR + (t // RT) % WR`, with `GX = T / (WR*RT)` and `NW = WR*WK`.
- **Interleave.** Inside a segment, lane l's k-th piece sits at 16 B index `soff[gw] + k*32 + l`. Every lane is padded to `P[gw]` pieces, the longest lane in the segment. Padding bits are 0.
- **Buffers:**
  - `stream uint32 [total_pieces * 4]`: the code stream, 16 B aligned.
  - `soff uint32 [n_segments + 1]`: segment starts in 16 B units. `P[gw] = (soff[gw+1] - soff[gw]) / 32`.
- The stream depends on (KU, RT) and on the segment numbering, so `pack_e` is called once per candidate geometry, the same way `_gvesc` is.

### 1d. Escapes: r6's `_gvesc` structure, 32 int32 entries per warp, -1 padded
- Entry: `(seq << 20) | (lane << 15) | (r << 11) | (half << 10) | (h << 9) | x`, where `seq = (c % KU) * RT + t % RT`, sorted by `(gw, seq)`.
- x is now 8 bits (bits 0–7; bit 8 = 0). The kernel applies `A[r] ^= x << (7 + 16*half)`, where r6 used `(x & 0x7f) << (8 + 16*half)`. **This is the only kernel-side change in the escape walk.**
- An escape occurs when a pair is not in the code alphabet. It is coded as its nearest alphabet pair, and each mismatching weight gets one entry.
- The alphabet size is chosen per matrix so the escape rate is ≤ 0.04% of weights (about 6 per 16-unit warp, against the cap of 32).
- `pack_e` returns None if any list overflows, like `_gvesc`.

## 2. Decode state per lane

| Register | Meaning |
|---|---|
| `lo, hi` | 64-bit bit buffer; the next code is at bit 0 of lo |
| `V` | Valid bits in the buffer, modulo 128. Bits 7 and up carry garbage from `V -= e`. Test with `(V & 0x60) == 0`. |
| `ptr` | Words consumed from the lane's stream: the next word is word `ptr & 3` of piece `ptr >> 2` |

**Refill.** Once per step (4 pairs = 8 weights): if `(V & 0x60) == 0`, then `w = word(ptr++)`, `lo |= w << V`, `hi |= shf.l.wrap(0, w, V)`, `V += 32`. A step never consumes more than 32 bits.

**Pieces.** A lane needs at most 1 new piece per unit. During unit u it may read pieces up to index u+1. The ring holds pieces in smem as `[slot][lane]`, 16 B each. That puts conflicts at about 2-way, because only lanes ≡ mod 8 that read the same word collide. Pieces are filled by predicated `cp.async` 16 B, at most one issue per lane per unit, while `iss < P && iss < (ptr >> 2) + D`. One commit_group per unit. The analyst sets D and the wait depth from r6's loads-in-flight depth.

The exact device code and its SASS count are in `CODEC_E_CUH`, documented in its header comment.

## 3. Cost warnings (measured by `pack_e` STATS)

**Q1.** Padding every lane to the longest lane, in whole 16 B pieces, costs `maxpad` bits. Expect about 0.2–0.3 bits/weight at KU = 16 and RT = 1. At KU = 2–4 (the o layer) it is about 0.6–1 bit/weight, which wipes out most of E's gain. Options:
- 4 B interleave granularity: about 4× less rounding waste.
- Segments spanning all KC chunks of a tile row, with per-lane bit starts.

The STATS line reports `maxpad` so this can be decided per layer.
