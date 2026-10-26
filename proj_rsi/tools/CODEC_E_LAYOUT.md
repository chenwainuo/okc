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

## 2. Decode state per lane (CODEC_E_CUH, v2 — ring layout changed from v1)

| Register | Meaning |
|---|---|
| `lo, hi` | 64-bit bit buffer; the next code is at bit 0 of lo |
| `V` | Valid bits, modulo 128. Bits 7 and up hold garbage from `V -= e`. Only `V & 0x60` (refill test) and `V & 31` (insert position) are read. |
| `pb` | 4 × words consumed from the lane stream. The current piece is `pb >> 4`. |
| `iss` | Pieces issued into the ring |
| `rl` | Lane ring base: `ring + lane * ED * 16`. The ring is aligned to ED·16 and ED is a power of 2. |

**Ring (changed in v2).** Each lane's ED slots are now contiguous, so one LOP3 builds the address: `rl | (pb & (16*ED - 1))`. v1 used `[slot][lane]`. Bank conflicts look random-like in both layouts, about 2–3-way.

**Per unit:**
1. `ce_issue<ED>`: at most one predicated `cp.async.cg` 16 B, issued while `iss < P && iss < (pb >> 4) + ED`.
2. `cp.async.commit_group`.
3. `cp.async.wait_group(ED-2)`, after a prologue of ED-1 issue+commit pairs.
4. `ce_unit<ED>`: 4 steps, each a branch-free refill followed by 4 pairs.

**Protocol check.** The reference unpacker replays this protocol and asserts that every refill reads a landed piece that has not been overwritten. It passes for ED = 4 on every test geometry.

**SASS.** Probe with ED = 4, nvcc 12.8, sm_90a. It uses 32 registers with 0 spills. Per unit it executes 194 lane-instructions, excluding HMMA, plane LDG and loop overhead: **6.06 per weight**.

## 3. Measured (selftest, 1M weights each, bit-exact incl. negative control)

| Case | Geometry (WR, WK, KU, S, RT) | Code bits | Pad bits/w | Total incl. tables, list, offsets |
|---|---|---|---|---|
| Gaussian K = 8192 (pooler phase) | 2, 8, 16, 1, 1 | 10.82 | 0.18 | 11.07 |
| Gaussian K = 8192 | 2, 4, 16, 2, 2 | 10.82 | 0.13 | 10.99 |
| Gaussian K = 1024 (o phase) | 1, 8, 2, 1, 1 | 10.81 | **1.19** | **12.52** |
| Gaussian K = 1024 | 2, 4, 4, 1, 1 | 10.81 | 0.53 | 11.61 |
| Gaussian K = 3072 | 1, 4, 8, 1, 1 | 10.86 | 0.26 | 11.26 |

- The pack for a 50 MB pooler-sized layer takes 0.55 s on CPU (48 threads), and is faster on GPU.
- The per-warp escape list costs a fixed 32 × 32 bits per warp, which is 0.06 bits/weight at 16 units per warp.

**Q1 (decide before building o and unpooler_s).** Padding to the longest lane in 16 B pieces dominates at small KU. Codec E loses to r6 at KU = 2, and is only marginal at KU = 4. Two options:
- Use E only on layers or geometries with KU·RT ≥ 8.
- Change the interleave granularity to 4 B. Lane l's k-th word would sit at `(soff + k*32 + l)*4`, with the ring filled by 4 B cp.async. That needs more issue instructions.
