# pw24 critic review (plain bf16 GEMM "bg" at m=128/256) — 2026-10-26

**Verdict: APPROVE.** I found no correctness, protocol or alignment defect. Two MINOR items.

What I ran:
- I built the gv block with nvcc 12.8, sm_90a, -O3, -Xptxas -v: 78 gv, 8 rs*, 7 bg instances, 15.6 s locally.
- r6's 78 functions are SASS-identical to r6-c12, and pw23's 8 router kernels are identical.
- Registers: bg 58 for all variants and the full probe, 19-21 for the other probes. All have 0 spills and no LDL/STL.
- `bg_emu.py` passes (BG EMU PASS):
  - bit-exact and one-hot exact at m = 128, 121, 256 and 250, with and without the cluster;
  - protocol 800/800 clean for both;
  - the mutations early_release (186/200), no_remote (200/200) and local_count (200/200) are caught.
- pw23's `_rsbake` re-zero of `rs_ws`/`rs_cnt` is still present (submission.py:3139).

## 1. Multicast: OK

**The two half-tile loads.**
- Each CTA issues one `UTMALDG.2D.MULTICAST`:
  - box 64 k × 32 rows, at row `n0 + 32·rank`;
  - mask 3, so it lands in both CTAs;
  - destination `wring + slot·8192 + rank·4096`, the same CTA-relative offset in both;
  - it completes the transaction on `wfull+8·slot` in both CTAs.
- Each CTA's wfull is initialised with count 1 and given `expect_tx(8192)` by its own producer, which covers both
  halves.
- If the peer's 4 KB lands before the local expect_tx, the barrier's transaction count goes negative within the phase,
  which is legal, and the phase cannot complete early: an arrival is still pending.

**Release counting.**
- wempty is initialised with count 8: 4 local MMA-warp arrives plus 4 peer arrives, sent with `mapa` and a plain
  `mbarrier.arrive.shared::cluster` (SASS `SYNCS.ARRIVE.TRANS64.RED.A1T0`), as pw13 found works.
- Neither producer refills a slot before both CTAs have released it.
- Both barrier sets are initialised and fenced before the opening `barrier.cluster`, so no multicast can hit an
  uninitialised barrier.

**Swizzle.**
- The 128B swizzle XORs address bits 4-6 with bits 7-9, and the slot base is 1024-aligned. The second half starts at
  +4096 = row 32 × 128 B, also 1024-aligned. So two 32-row boxes give exactly the image of one 64-row box.
- The descriptor is SBO 1024, swizzle 128B, LBO 1, +2 per k16 step. That is the same descriptor pw10 and pw19 use and
  the emulator confirms.

## 2. x ring: OK

- The map is `{K, m}` with box `{64, 64}`, SWIZZLE_128B and OOB fill NONE (zeros), so rows ≥ m are zero.
- TT = ceil(m/128)·2 is always even, as the cluster pair requires.
- The B descriptor is for N = 64, K-major.

## 3. wgmma and exit: OK

- The SASS is `HGMMA.64x64x16.F32.BF16 R, gdesc[UR], R`: SS, both transposes 0, 32 accumulators.
- The loop runs `wait_group 1`, releases slot s-1 (local W-empty, remote W-empty, x-empty), then `wait_group 0` and
  releases the last slot.
- All 192 threads (the producer warps after their loops) reach `barrier.cluster` before exit, so no multicast or remote
  arrive can target a CTA that has exited.

## 4. Epilogue: OK

- The [64 tok][64 n] tile reuses x slot 0, only after `wait_group 0`. By then every x tile has been waited on, and
  every W tile (including the peer's multicast halves) has landed and been consumed. The producers issue exactly KC
  of each.
- The W ring is untouched.
- The stores are 16 B, with tok < m guarded.

## 5. Two CTAs/SM: OK

| Variant | Dynamic smem | Static smem | Registers | Fits 2/SM? |
|---|---|---|---|---|
| bg6, bg6c | 1024 + 98,304 + 192 = 99,520 B (97.2 KB) | 0 | 58 × 192 threads | yes: 2 × (99,520 + 1 KB) = 201 KB |
| bg12, bg12c | 198,016 B (193.4 KB) | 0 | 58 | 1 CTA/SM, within 227 KB |

## 6. Dispatch, prepare, log

- Only `sel[(layer, 128|256)]` changes, on `us < r6 × 0.99`. The bake runs inside a try that pops only those keys.
- The odd m 121 and 250 are validated, though not served.
- **Prepare time.** About 120 s (pw23), plus about 3 s of compile, plus the bg bake: 32 paths with 32-64 one-hot
  launches each, plus 4 ladder timings, about 10-15 s. Total: **about 135-140 s** against 240.
- **Log.** About 7-8 KB.
- **MINOR (format).** `PROBE bghbm pooler m=128 W-only tbs=… tbs_minus4us=…` contains no bare number, so okc_result's
  PROBE regex ignores it. Print `PROBE bghbm pooler m=128 W-only <tbs> <tbs_minus4us>`. The `PROBE ladder <label> 2x64
  <us>` lines parse correctly.
- **MINOR.** nvcc warns #1886: `extern __shared__ smem` is declared `__align__(16)` in bg and `(128)` elsewhere. It is
  harmless, because bg rounds up to 1024 itself and r6's SASS is unchanged, but declaring 128 everywhere would silence it.
