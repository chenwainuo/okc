# P4d: per-stage anatomy of the RS128 12-bit kernel (rs128p, a copy of rs128 e8d7ab8)

**Setup.**
- Box: H100.
- Lab: `proj_rsi/candidates/rs128p/lab.py --plan anat`, one locked batched run, `/root/okc/logs/rs128p/batch.log`.
- Shape: pooler m=128, KS 4, 128 CTAs, 32 k64 stages per CTA; qkv m=128 (KS 7) for the production variants.
- **T** = CUPTI kernel µs; **G** = graph replay after a read flush (the in-prepare method); medians.
- Stamps: clock64 per math warp (WG0 mean, the stamped twin of each variant), per k64 stage.
- The escape walk is now stamped separately (prof slot 32+WG). It is part of the decode stamp, or of the wait_group stamp for the pre-walk variant.
- Validation: every valid variant is one-hot exact over all K, and the checker passes (rel L2 3.0e-3 pooler, 3.6e-3 qkv). The zero-XOR probe is INVALID by construction.

**Clock caveat.** Stamps count SM cycles at the *effective* clock. Under tensor-heavy load the box runs below the 1980 MHz nvidia-smi reports. The resident constant-A rung reads 521 clk/stage while its kernel time implies 0.343 µs/stage, i.e. ~1.5 GHz. So compare stamps within a variant; across variants, compare µs.

## Results

**In-prepare (G), µs.** r6 is pw23's tuned cuBLASLt from the pw32 box bake (pooler 27.10, qkv 19.49). torch.mm is in the same run.

| variant | pooler m=128 G / T | qkv m=128 G / T | vs shipped (G) | valid |
|---|---|---|---|---|
| torch.mm | 28.58 / 26.24 | 20.42 / 18.34 | | |
| **shipped rq s4s5hz** | **24.06 / 21.97** | **16.80 / 14.88** | 0 (−3.0 / −2.7 vs r6) | yes |
| x: escape XOR by switch (FL 32) | 23.94 / 21.82 | 16.90 / 15.18 | −0.12 / +0.10 | yes |
| p: escape walk moved into the wait_group window (FL 64) | 24.61 / 22.53 | 17.34 / 15.76 | +0.55 / +0.54 | yes |
| w5: x release before the next LDS (WM 5) | 24.13 / 22.02 | 17.12 / 15.07 | +0.07 / +0.32 | yes |
| z: walk + XOR by a runtime 0 (probe) | 23.97 / 21.94 | 17.02 / 15.12 | | no |
| no escape lists (bound, probe) | 23.42 / 21.57 | 16.35 / 14.72 | −0.64 / −0.45 | no |
| earlier: w3 interleaved decode/issue | T 22.40–22.59 | | +0.4–0.6 | yes |
| earlier: w4 + next-stage LDS inside the issue | T 23.26 | | +1.2 | yes |

**Resident rungs (SM side only: no TMA, no barriers).** This is the term that binds on the H200, whose HBM period ≈ 0.375 µs/stage.

| rung | T µs | G µs | per stage (T) | reading |
|---|---|---|---|---|
| r10: constant A, register-A wgmma | 11.49 | 13.92 | 0.343 µs | tensor floor |
| r11: same, SS wgmma | 11.30 | 13.89 | 0.337 µs | register-A is not slower than SS |
| r10d2: tiny-value A probe | 11.39 | 13.89 | | no data-dependent HGMMA speed |
| r8 without escapes | 11.98 | 14.56 | 0.359 µs | LDS + decode add only 0.5 µs over the floor |
| **r8: LDS + decode + escapes (as shipped)** | **14.14** | **16.61** | **0.426 µs** | escapes +2.16 µs |
| r8x / r8z / r8p / r8w5 | 14.18 / 14.27 / 15.23 / 14.42 | | | |

## Per-stage anatomy (clk, WG0 mean, stamped twins)

decode = decode stamp minus escape walk; epi share = epilogue clk / stages per CTA.

| variant | loop | full wait | LDS + W arrive | LDS data wait | decode | escape walk + apply | fence + 4 wgmma + commit | wait_group | x release | epi share |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped, pooler (streamed) | 941 | 350 | 25 | 30 | 150 | 79 | 177 | 34 | 95 | 73 |
| shipped, no escapes (probe) | 945 | 469 | 27 | 30 | 157 | 34 | 95 | 30 | 103 | 69 |
| x (switch) | 957 | 381 | 26 | 29 | 148 | 79 | 169 | 32 | 93 | 73 |
| p (pre-walk) | 991 | 277 | 28 | 33 | 201 | 200 (in wait_group) | 216 | 161 | 73 | 78 |
| w5 | 957 | 304 | 26 | 57 | 159 | 80 | 183 | 48 | 99 | 73 |
| shipped, qkv | 1014 | 419 | 24 | 30 | 150 | 79 | 180 | 34 | 95 | 134 |
| r8 resident | 749 | 66 | 17 | 166 | 187 | 75 | 188 | 27 | 23 | – |
| r8 resident, no escapes | 646 | 64 | 20 | 169 | 211 | 32 | 92 | 32 | 26 | – |
| r10 resident, constant A | 521 | 48 | 35 | 86 | 14 | 0 | 287 | 25 | 25 | – |

## Mechanism

1. **On the box the shipped kernel is stream-bound.**
   - The full-barrier wait is 350 clk/stage. The bare W+x stream alone is 19.0 µs (T) against 22.0 for the full kernel; the rest is fixed terms (first wgmma at ~2 µs, epilogue ~2.3k clk).
   - So SM-side levers barely move box numbers. The H200, with 1.43× the HBM bandwidth, is where the SM side binds.
2. **The SM-side stage is the tensor floor plus the escape walk.**
   - r8 without escapes is only 0.5 µs above the constant-A floor (decode + LDS overlap the other warpgroup's wgmmas).
   - Escapes add **2.16 µs resident** (0.067 µs/stage), with ~36 hits per CTA (max 54).
   - Loop time correlates with hits per CTA (r ≈ 0.5), so the escape-heaviest CTAs set the kernel time.
   - Stamps: walk 75 clk vs 32 empty. The fence+issue stamp doubles, 92 → 188, because `wgmma.fence` waits for the warpgroup's slowest warp, the one walking its list (SHFL → compare → branch per hit).
3. **What the escape cost is not.**
   - Not the 16-way register select: the switch apply changes nothing.
   - Not the data: XOR by a runtime 0 costs the same, and tiny-value A shows no slowdown.
   - Not the walk's position: moving it into the wait_group window is worse, because it then delays the next issue and the x release.
4. **The coordinator's levers.**
   - **(a) Descriptors.** No R2UR rebuilds: x descriptors live in uniform registers (UMOV ×2 + ULOP3 per HGMMA, slot mod 5 by UIMAD.WIDE), so there is nothing to hoist.
   - **(b) Batched release.** The 95-clk x release is mostly the release-semantics arrive waiting for the LDS just issued. Moving it before the LDS (w5) moves the wait into the LDS data wait: +0.07.
   - **(c) Polling.** Not retried. P4 showed that polling steals issue slots.
   - **(d) Escape walk.** This is the lever: ≈ 2.2 µs SM-side, 0.45–0.64 µs on the box. The three placements tried (switch, pre-walk, per-step) do not capture it.
   - **(e) Elected-lane release.** Releases are 20–25 clk resident, so there is nothing to gain.
5. **The ≤ 650 clk/stage goal** is below what this box sustains for 8 × m64n128k16 per stage (r10 ≈ 0.343 µs/stage). The attainable SM-side target is the no-escape resident rate, 0.359 µs/stage, about −16% vs shipped.

## Next lever (proposed, not built)

**Design.** Take escapes off the math warps entirely.
- One idle producer warp (warp 10 or 11, after zeroing) walks the lists.
- It writes, per ring slot, a per-(math warp, lane) 16-bit patch word (reg, xor7, half; 0xffff = none) into a 256 × 2 B smem table per slot.
- It arrives as a third arrival on the slot's full barrier, and waits the W empty barrier before reusing a slot's table.
- Math warps per stage: one `LDS.U16` alongside the W LDS, plus a predicated XOR. No SHFL chain, no warp-divergent loop before the fence.
- Lanes with 2 escapes in one stage (the max is 2 for pooler and qkv) need 2 words: 512 B per slot.

**Expected.** Resident → r8-noesc (−2.0 µs, about the H200's SM-side margin); box production −0.4 to −0.6.
