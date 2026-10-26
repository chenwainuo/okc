# pw18 critic review (codec E v3) — 2026-10-26

**Verdict: APPROVE.** I found no blockers. The open questions are performance only: issue pressure, and whether the gain
reaches the kill bar.

What I ran:
- `codec_e --test`: SELFTEST PASS at ilp 1/2/4, and the negative control is detected.
- `codec_e_emu.py pw18`: EMU PASS, 48 runs bit-exact (24 path/geometry/K combinations, each under early and late
  landing).
- A rebuild of the gv block with nvcc 12.8, sm_90a, -O3, -Xptxas -v: 99 instances.
- A property test of `_prefix_code` on 3,000 alphabets (1-256 symbols; uniform, geometric, flat and one-dominant counts).
- An instruction-level def-use scan of the pooler m=1 E SASS.

The packer and unpacker functions in submission.py are identical to proj_rsi/tools/codec_e.py.

**Correction to my pw17 review.** I priced the v2 ring at about 3.5-way by assuming random per-lane word offsets.
Lanes advance in lockstep, so that was wrong; the mapping is worked through in section 1.

## 1. Bank mapping of a refill LDS (32 lanes, all at the same stream word w)

| Layout | Address (bytes) | Bank of lane l | Lockstep | Drifted lanes |
|---|---|---|---|---|
| v2, ED 4 | l·64 + 4·(w mod 16) | (16l + w) mod 32: even lanes on one bank, odd on another | **16-way** | measured 12 |
| v2, ED 8 | l·128 + 4·(w mod 32) | w mod 32 for every lane | **32-way** | measured 22 |
| v3, ring 16 words, ES = 16/K | ring_s + 128·(w mod ES) + 4l | **l** | 1 wavefront | 1 wavefront |
| v3, ring 32 words (bit 19) | the same with ES = 32/K | **l** | 1 | 1 |

- In v3 the bank does not depend on the slot, on w, or on drift between lanes, so the claim holds for every K and depth.
- The 4 B LDGSTS writes to `rl | slot<<7` also land one per bank.
- The replicated value LUT (`lut + x·128 + 4l`) has bank l and is conflict-free.
- The single LUT has bank x mod 32: about 3.2-way under uniform peeks, 3.7-way measured. That is why the K1-single path
  should lose to the r variants.

The alignment holds: rl = cs0 (aligned to CAL ≥ ES·128) + w·CED·128 + s·ES·128 + 4l, so `rl | ((pb<<5) & ((ES-1)<<7))`
never collides with set bits (submission.py:596, 737-748).

## 2. Chain in the SASS (pooler m=1, 2.8.16, K1 single and K4 replicated)

- **Shift amounts.** All 505 (K1) and 484 (K4) `SHF.R.W` shift amounts come from a PRMT. Every PRMT selector register
  comes from a LOP3 (`lo & 7`). The chain is LOP3 → PRMT → SHF.R.W, and **no LDS sits between consecutive position
  updates**.
- **Value lookups.** The LUT LDS results feed only the `LOP3 …0x807f807f…0xb8` value select (256 per kernel), plus
  epilogue and reduction reads. They never feed back into lo, hi or V.
- **Ring LDS.** Ring words feed only the refill (SEL, SHF.L / IMAD.SHL, LOP3). There are 4 ring LDS per unit at K1: the
  word for t = 0 is read after DEPBAR, and the later steps use words prefetched one step ahead. So one chained LDS per
  substream per unit remains.
- **Counts.** There are 320 LDS in total (16 units × (16 LUT + 4 ring)), 16 × `DEPBAR.LE SB0,2`, and 76 LDGSTS
  (3 + 16 issue rounds × 4). No instruction uses local memory (LDL/STL).

## 3. Length table vs the kernel's PRMT

- **Packer side.** `_prefix_code` returns tab[8], which pack_e packs little-endian into lut[256] (bytes 0-3) and
  lut[257] (bytes 4-7).
- **Kernel side.** The kernel loads `ct0 = lut[256]` and `ct1 = lut[257]` from **the same per-matrix lut pointer (T)**
  as the value LUT (submission.py:747). `prmt(t0, t1, lo & 7)` selects byte lo&7 of {t1:t0}, and bit 3 of the selector
  is 0, so there is no sign-replicate.
- **Upper bytes.** The upper bytes of len are copies of byte 0. Shifts use len & 31. The only effect on V is a borrow
  into bits 8 and up, and only V & 0x7f is read.
- **Property test.** For every symbol and every 8-bit extension of its code, `tab[x & 7]` equals the length. The codes
  are prefix-free, Kraft ≤ 1, and every length is ≤ 8. **3,000 of 3,000 alphabets pass.**
- **Unused peeks.** A partly filled last bucket leaves some peeks unused. Their lut entry is 0 and they are unreachable.

## 4. Substream regions and bounds

- **Addressing.** Word q of lane l in substream s is at `(s0 + (s·P + q)·32 + l)·4`. `cgs + s·P·128 + iss·128` is
  computed from soff, s, lane and iss only. There is no data-dependent offset.
- **Bounds.** Loads are predicated on `iss < P`, so they stay inside `[s0, s0 + K·P·32)`, which is the segment.
- **P.** `P = (soff[gw+1] - soff[gw]) / (32K)` divides exactly, because pack_e steps soff by P·32·ilp. `_gepack` derives
  ilp and ed from the same V bits as the kernel's CK and CED.
- **Ring invariant.** I checked it by hand, with ES = 4·NI words per substream and at most NI consumed per unit.
  - *Steady state:* `iss_n = min(P, c_n + ES)`.
  - *What wait_group 2 guarantees:* the words landed are at least `min(P, c_n + 2NI)`, while the reads reach at most
    `c_n + NI - 1`, including the prefetch.
  - *Overwrites:* a slot is reissued only for a word below c_n.
  - *Units 0-2:* the prologue covers them.
- MINOR, carried over from pw17: some cp.async can still be in flight at kernel exit. A free `cp.async.wait_all` after
  the K loop would settle it.

## 5. r6, validation and prepare

- **r6.** All 78 r6 functions are instruction-identical to r6-c12.
- **Unchanged code.** `_gxbake`, `_validate` and `prepare` are textually identical to pw17's. That covers the r6
  baseline, the 0.99 margin, the per-path try, one-hot ×8 with odd m 5/9, and the PROBE format.
- **Pack proof.** The proof still runs before any plan exists, and an AssertionError from the ring check now declines
  the path (submission.py:2286-2293).
- **Registers.** The 20 E instances have 0 spills and use 119-128 registers. The exception is unpooler m≤8 K4 at 148,
  with 128 threads and 64 CTAs. qkv runs with minBlocks 2 → 128 registers → 2 CTAs/SM.
- **Prepare time.** On the judge pw17 logged `PREP r6 done in 111s` and `PREP done in 120s`, so 29 E paths took 9 s.
  pw18 compiles 99 instances in about 27 s locally, the same as pw17. With 32 paths I expect about **125-135 s**
  against the 240 s guard.
- **Log.** pw17's log was 8.4 KB; pw18's should be about 9 KB.

## 6. Will the decode hide at pooler m=1 (16 warps/SM)?

- **Static size.** Measured between the first and last DEPBAR, a unit is 323 (K1 single) to 347 (K4 r) static
  instructions. That includes the escape-walk code, which rarely runs. Dynamic counts are about 256-286, which agrees
  with the implementer.
- **Issue.** About 270 × 16 warps / 4 schedulers ≈ 1,080 issue cycles per unit window. The window is about 1,420-1,600
  clk at the 11.1-bit stream, so the schedulers are about 70% busy.
- **Chain.** Per pair the chain is about 3 ALU ops (about 13-15 clk), and each refill adds an LDS at unit start. That
  gives about 300-330 clk at K1, about 170 at K2 and about 100 at K4, comfortably inside the window.

**MAJOR (performance, unproven).** Issue, not latency, is now the binding resource. 4 warps per scheduler at about 70%
utilization is plausible but has no slack. Expect K2 or K4 replicated to be the best path. A gain of 0.8 µs needs the
decode fully hidden: the bytes save about 0.95 µs, minus about 0.2 for the 512-thread geometry, minus the tail. Reaching
0.8 is possible but unlikely; I expect 0.3-0.7.

## Minor

- CODEC_E_LAYOUT.md is garbled where §1a runs into §1b ("…lane's 8 words### 1b"). A fragment of the old §1c is left
  under §1b ("numbering, so pack_e is called once…"). The §3 table still shows the v2 16 B padding numbers.
- The kernel comment says V bits 12-14 are "pw14 probes only"; pw18 now uses them for paths (the qkv minBlocks 2 cap).
