# pw17 critic review (codec E on r6-c12) — 2026-10-26

**Verdict: APPROVE.** No blockers. Two MAJOR items are about expected payoff and residual risk, not correctness.

What I ran:
- I rebuilt the pure-CUDA gv block with nvcc 12.8, sm_90a, -O3 -lineinfo, -Xptxas -v. It has 98 instances and took 26 s.
  The r6-c12 block has 78 instances and took 14 s. pw16 has 122 instances and took 27 s.
- I ran `codec_e --test` (SELFTEST PASS, negative control detected).
- I ran `codec_e_emu.py` on the reference weights. All 15 GEP keys were bit-exact under both early and late landing, at
  immLut 0xb8 with issue per unit.
- I ran lint (ok).

All scratch files are in /tmp/claude-1000/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/scratchpad/b/.

## 1. Decode data path vs the packer: OK

The SASS of pooler m=1 E (`gv<4,1,2,2,8,16,1,1,0,1>`, 118 registers, 0 spills) shows the following:
- **Select.** `LOP3.LUT Rd, Re, 0x807f807f, Rprmt, 0xb8` appears 256 times. With a=e, b=mask, c=d this is `mask ? d : e`,
  as in the spec (submission.py:591-594).
- **Plane permutes.** The PRMT selectors are only 0x1100 and 0x3322 against RZ.
- **LUT address.** It is `IMAD.SHL x4; LOP3 &0x3fc; LDS [R+UR4]` (single LUT) and `x80, &0x7f80` (replicated LUT, so the
  bank equals the lane).
- **LUT fields.** pack_e (submission.py:2019) builds the entry as `val[low] << 7 | val[high] << 23 | len`:
  - the length is in bits 0-4;
  - the v-even exponent is in bits 7-14 and the v-odd exponent in bits 23-30;
  - bit 15 and bit 31 are 0.

  This matches the selection mask and the bf16 layout.
- **Bit buffer.** It is advanced with `SHF.R.W` on the LUT word, so the shift is `e & 31`. `V -= e` borrows only upward,
  so `V & 127` stays exact.
- **Refill.** The refill runs when V < 32, so V stays in [32, 64) after a refill. A step consumes at most 4 × 8 = 32
  bits, so V never underflows and the 64-bit buffer never overflows.
- **Piece boundaries.** Refills are word-granular (pb += 4 bytes), so 16 B piece boundaries play no role. Nothing is
  lost or duplicated.

## 2. Ring protocol: OK

- **What the ring is.** It is a per-lane **shared-memory ring filled by cp.async.cg 16 B**, not a register ring as the
  brief says. Its addresses depend only on `iss`, `P`, `soff` and `lane`, never on decoded data.
- **Issue counts.** There is one issue + commit per unit (u, j) at submission.py:799-803. The SASS has 3 prologue LDGSTS
  plus 16 LDGSTS + LDGDEPBAR + `DEPBAR.LE SB0,2` for KU = 16, and every ring LDS sits after its unit's DEPBAR.
- **Bounds.** `go = iss < P && iss < (pb>>4) + ED`, so global reads never leave the segment.
- **Landing invariant.** Each unit consumes at most one piece (4 refills of at most 4 B), which gives
  `iss_n = min(P, (pb_n>>4) + ED)`. wait_group(ED-2) then guarantees that pieces up to min(P, (pb_n>>4)+2)-1 have
  landed, which covers the reads (at most `(pb_n>>4)+1`). Before the first decode, pieces 0 and 1 have landed.
- **Overwrites.** A slot is reissued only for a piece numbered below pb>>4, which is fully consumed.
- **Past the end.** A read past P hits a stale slot that is never in flight, and only fills bits beyond the lane's last
  code.
- **Hangs.** Nothing can hang: there is no mbarrier on the E path.
- MINOR: up to ED-2 cp.async (6 for d8) can still be in flight at kernel exit. They target the CTA's own ring and do not
  overlap `red`. Adding a free `cp.async.wait_all` after the K loop would remove the question.

## 3. Escapes: OK

- `_gvesc_e` (submission.py:1940) builds `r = 4j + v//2` and `half = v&1`, using r6's gw and seq formulas, which are
  identical to the kernel's `(by*GX+bx)*NW + wk*WR + wr`.
- The kernel's XOR is `(nx & 0xff) << (7 + 16*half)` (submission.py:858).
- The walk is otherwise r6's.
- Capacity comes from the actual weights: a list overflow returns None, and the path is reported as declined-pack.

## 4. r6 untouched: OK

- **SASS.** All 78 r6 functions are instruction-identical to r6-c12's build.
- **Order.** prepare runs r6's bake first. `_gxbake` starts each (layer, m) from `best = r6` and replaces it only on
  `us < r6 * 0.99` (submission.py:2304).
- **Exceptions.** A Python exception per path is caught, and a failure of the whole bake resets `sel` (submission.py:1813-1819).
- **Residual risk (MAJOR, unavoidable).** A device fault, such as an illegal address, is sticky and would also kill r6's
  dispatch. I found no out-of-bounds access, checking:
  - cp.async source within P and destination 16 B aligned;
  - soff[gw+1] ≤ nseg;
  - smem size includes CLUT + CRING + 128 of alignment slack, and is set via cudaFuncSetAttribute (max about 76 KB);
  - plane, esc and LUT indexing.

  This is still the first GPU execution of this code.
- **Memory.** Packs are cached until the end of `_gxbake` and then popped. About 10 packs at roughly 50-75 MB each plus
  pack_e temporaries come to about 2 GB peak, which is fine on an H200.

## 5. Validation: OK

- `_gepack` checks `unpack_e == _frag_bits(fr)` on the whole matrix, with a ring-protocol replay at the instance's ED,
  before any plan is returned (submission.py:2210).
- Every path then goes through the template `_validate`: one-hot ×8 exact, random rows, and odd m 5/9/17 for buckets
  8/16/32 (submission.py:2294).
- `_bk` keys sel by bucket:
  - judge m = 5 and 7 → bucket 8, validated at 8 and 5, same MT 1 code;
  - m = 9 → bucket 16, validated at 9;
  - m = 33 and above go to r6 or cuBLAS.

## 6. Prepare budget and log: OK

- **Compile cost on the judge.** The judge's r6-phase cost scales at about 1.8× local compile time: pw16 had 122
  instances, 27 s locally and 133 s on the judge, against r6's 78 instances, 14 s and 109 s. pw17 compiles in 26 s, so
  its r6 phase is about **130 s**.
- **E bake.** The E bake adds about 15-30 s:
  - 7 packs, each under 1.5 s with its proof;
  - about 10 m=1 one-hot sweeps at 4-8 K launches, about 0.5 s each;
  - 29 timings;
  - the PROBE (6 × 21 replays).
- **Total.** About **150-165 s** against the 240 s guard (checked per path) and the 300 s hard limit.
- **Log.** pw16's log was 7.2 KB. pw17 drops pw16's gc lines and adds about 29 BAKE lines plus about 25 PREP/STATS_E
  lines, for an estimated **9 KB**.
- **PROBE format.** `PROBE ecod pooler m=1 ge-p512r 13.10 r6=13.95 delta=-0.85` parses with okc_result's regex as
  engine ecod, config "pooler m=1 ge-p512r", 13.10 µs. The none and skipped-budget forms are ignored.

## 7. Registers, occupancy and whether the decode hides at m=1

**Registers and residency.**
- Pooler m=1 E 2.8.16 uses 118 registers × 512 threads, which allows 1 CTA/SM = **16 warps/SM**. r6's judge pick
  (2.16.8, 1024 threads, 64 registers) runs 32 warps/SM.
- qkv m=1 E 1.8.16 uses 118 registers × 256 threads, giving 2 CTAs/SM, the same as r6.
- The m=16/32 E instances use 114-128 registers at r6's own residency.

**Decode budget at pooler m=1.** Each warp owns 16 units over a kernel of about 12.5 µs (≈24.7 k clk), so it has about
**1,500 clk per unit**. Against that:
- **Serial chain.** 16 dependent LUT steps × ~30 clk plus 4 refills is about 550 clk, roughly 35% of the window.
- **Issue.** 196 instructions × 16 warps over 4 schedulers is about 784 of 1,500 slots (~52%).
- **Shared-memory pipe, single LUT.** Uniform-peek Monte Carlo puts conflicts at 3.15-way for the LUT and 3.45-way for
  the ring. That is 16×3.15 + 4×3.45 ≈ 64 wavefronts per warp-unit, or about 1,020 per 1,500 clk (~68%).
- **Shared-memory pipe, replicated LUT.** About 30 wavefronts per warp-unit (~32%).
- **In flight.** The ST=2 plane prefetch keeps about 32 KB/SM in flight, which also covers the dependent
  soff→first-piece round trip.

**Expectation.** The decode should hide in steady state.

**MAJOR (payoff).** The kill bar looks unlikely to be met. The bytes saved are 50.3 → 46.4 MB, about −0.95 µs. Against
that:
- the 512-thread geometry costs +0.2 µs;
- the exposed last-unit chain costs about +0.25 µs;
- the startup costs +0.1-0.3 µs.

The expected net is **−0.2 to −0.5 µs**, short of the 0.8 µs kill bar. If the single LUT is used it sits near the
shared-memory throughput limit, so expect the r or f variants to win. This is a prediction only; the PROBE settles it.

## Minor

- submission.py:21 and the hypothesis call the ring "cp.async". That is accurate. The brief's "register ring" is not.
- The fixed-stride variant costs 13 pieces/lane, which is +0.25 bits (11.31 bits total). It only pays if soff latency is
  exposed.
