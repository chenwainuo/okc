# Codec E: layout spec (v3, pw18)

Status: v3 (pw18). v2 was pw17's layout; the v3 changes are in 1b, 1c and 2. Older status line: v1. Codec side by the implementer; the gv integration belongs to the analyst. Open questions are marked **Q**.

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
- The plane word for register r is word `2*j + (i >> 1)` of the lane's 8 words### 1b. Code and decode tables: `lut uint32 [258]` per matrix
- **Code (v3).** One prefix code per A register (pair symbol), at most 8 bits, read LSB first, in which **the first 3 bits
  of every code determine its length**: codes of length <= 3, plus "buckets" (one 3-bit prefix each) holding 2^(L-3) codes
  of length L. The lengths are DP-optimal under that constraint (`_prefix_code`). On the reference weights this costs
  +0.03 bits/weight against an unconstrained length-limited Huffman; a 4-bit prefix would cost 0 but needs about 3 more ops per pair.
- `lut[x]` for every 8-bit peek x (value lookup, off the bit-position chain):
  - Bits 0–4: code length (informational; the kernel does not use it). Bits 5–6: 0.
  - Bits 7–14: exponent of the low half (v even). Bits 23–30: exponent of the high half (v odd).
- `lut[256..257]`: the 8-byte length table, byte w = length of every code whose first 3 bits are w. The kernel computes
  `len = prmt(lut[256], lut[257], lo & 7)`. The upper bytes of len are garbage, which is harmless because shifts use len & 31 and V's
  garbage sits above bit 7.
- The A register is `lop3(e, 0x807f807f, prmt(Lword, 0, i&1 ? 0x3322 : 0x1100), 0xb8)` (immLut `b ? c : a`; v1 said
  0xe4, which is `c ? a : b` and wrong).

### 1c. Code stream (v3)
- **Substreams.** K = 1, 2 or 4 independent substreams per lane. Substream s holds steps j in [s·4/K, (s+1)·4/K) of every
  unit, so each has its own bit buffer and the K decode chains interleave.
- **Order** inside a substream: `for u in 0..KU-1: for jt in 0..RT-1: for its steps j: for i in 0..3`.
- **Words.** Bit b of a substream's lane stream sits in its 32-bit word `b >> 5`, at bit `b & 31`.
- **Segment** (one warp's range, id `gw` as in r6's `_gvesc`): K regions of P·32 words. Word q of lane l of
  substream s is at word `soff[gw] + (s·P + q)·32 + l`. Each warp load is 128 contiguous bytes, padded to a word.
  P is the longest lane-substream in the segment, in words.
- **Buffers:** `stream uint32`; `soff uint32 [NSEG + 1]` in words; `P = (soff[gw+1] - soff[gw]) / (32 K)`.
- 16 B pieces with per-lane word rotation are still supported by the packer (`_wpp`), but are not used: they pad
  ILP 1/2 to 0.16/0.55 bits/weight against 0.12/0.22 with 4 B pieces.

 numbering, so `pack_e` is called once per candidate geometry, the same way `_gvesc` is.

### 1d. Escapes: r6's `_gvesc` structure, 32 int32 entries per warp, -1 padded
- Entry: `(seq << 20) | (lane << 15) | (r << 11) | (half << 10) | (h << 9) | x`, where `seq = (c % KU) * RT + t % RT`, sorted by `(gw, seq)`.
- x is now 8 bits (bits 0–7; bit 8 = 0). The kernel applies `A[r] ^= x << (7 + 16*half)`, where r6 used `(x & 0x7f) << (8 + 16*half)`. **This is the only kernel-side change in the escape walk.**
- An escape occurs when a pair is not in the code alphabet. It is coded as its nearest alphabet pair, and each mismatching weight gets one entry.
- The alphabet size is chosen per matrix so the escape rate is ≤ 0.04% of weights (about 6 per 16-unit warp, against the cap of 32).
- `pack_e` returns None if any list overflows, like `_gvesc`.

## 2. Decode state per lane and substream (CODEC_E_CUH v3)

| Register | Meaning |
|---|---|
| `lo, hi` | 64-bit bit buffer; the next code is at bit 0 of lo |
| `V` | Valid bits mod 128. Bits 8 and up hold garbage from `V -= len`. Only `V & 0x60` (refill test) and `V & 31` (insert position) are read. |
| `pb` | Bytes taken from the substream's lane stream (word `pb >> 2`) |
| `iss` | Words issued into the ring |
| `nw` | Prefetched refill word |

**Ring (v3).** Per warp and substream, the ring has ES = 16/K word slots × 32 lanes, laid out `[s][slot][lane]` (128 B per
slot). Word q goes to slot q % ES at `rl | ((pb << 5) & ((ES-1) << 7))`. Every lane reads its own bank, so the ring is
conflict-free; v2's lane-contiguous ring cost 12–22 wavefronts per refill (codec-study post-mortem). The total is
64 B per lane, as in v2 ED 4.

**Per unit:**
1. For each substream, NI = 4/K predicated 4 B `cp.async.ca` while `iss < P && iss < (pb >> 2) + ES`.
2. One `commit_group`.
3. `wait_group 2`. The prologue issues 3 rounds.
4. `ce_unit`: for each of its 4/K steps, every substream refills (branch-free; the first step's word is read after the wait,
   later steps use the word prefetched right after the previous refill) and then decodes 4 pairs. Per pair:
   `len = PRMT(t0, t1, lo & 7)`, `lo, hi = funnel-shift by len`, `V -= len`, `e = lut[lo & 0xff]`, `A = LOP3(e, plane)`.
   The bit-position chain is LOP3 → PRMT → SHF.R.W, with no memory access.

**Protocol check.** `unpack_e` replays the per-substream protocol and asserts that every word the decode uses is landed and
not overwritten. `proj_rsi/tools/codec_e_emu.py` emulates the pw18 kernel instruction by instruction, with early and late
landing. It catches immLut 0xe4 and a `wait_group 4`.

**SASS** (nvcc 12.8, sm_90a, pw18 pooler m=1 2.8.16). Per unit (1,024 weights, 32 per lane), for K1 single / K1 replicated /
K2 replicated / K4 replicated: 256 / 272 / 279 / 286 instructions. 0 spills.

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
