# pw10 PROBE ladder: where the 17 µs over the stream go (pooler m=128)

Check `7715c850ff89450a`. All figures are in-prepare µs (graph replay ≈ 5 µs included). The kernel is pw9's w1n128:
- BN 64 W rows per CTA, one consumer warpgroup plus a producer warpgroup.
- Ring of 4 stages; each stage is the 16 KB x tile (TMA) plus the 6 KB W chunk (bulk).
- Grid 64 × 2 = 128 CTAs, 1 CTA/SM.

| line | µs | Δ |
|---|---:|---:|
| bare bulk stream of the same W bytes (pw2 probe: 132 CTAs, 16 KB × 4 stages, one consumer warp) | 17.2 | |
| L1 no-x: skeleton streaming W only (6 KB stages) | 20.9 | +3.7 (i) |
| L1: W + x tiles | 25.4 | +4.5 (ii) |
| L2: + decode | 24.6 | −0.8 (noise; decode is hidden) |
| L3: + wgmma | 30.1 | +5.5 (iii) |
| L4: full (+ cluster-2 epilogue) | 34.6 | +4.5 (iv) |

**Correction to the head's reading of (i).** The probe kernels are launched without the cluster attribute (`cfg.numAttrs = (KS > 1 && PRB == 0) ? 1 : 0` in `pw_wlaunch`). So L1 and L1 no-x are plain 128-CTA launches, and cost (i) is not cluster launch or placement. Cluster cost, if any, sits entirely inside (iv).

## (i) +3.7: bytes in flight on the W stream, not the handshake

- **Mechanism.**
  - With x skipped, each stage carries only the 6 KB W chunk (BN 64 = 4 tiles × 1536 B). A 4-deep ring therefore has at most about 24 KB in flight per SM.
  - The bare stream probe holds 64 KB per SM (16 KB × 4).
  - pw2 measured the same bulk engine at 16 KB × 2 (32 KB in flight, 1 CTA/SM) at 18.3 µs (3.48 TB/s), against 17.0 at 64 KB. At 24 KB, Little's law gives roughly 3.1–3.2 TB/s, i.e. about 21 µs. That is the measured 20.9.
- **Not the handshake.** pw7's skeleton probes (pw4: `free`, no empty waits, ≈ base) showed the per-stage mbarrier round trip costs ≤ 0.5 µs.
- **Single fix: a deep W-only ring.** Keep ≥ 48–64 KB of W in flight per SM: W stages of 4 chunks (24 KB) × 3, or 6 KB × 10.
- **Why it can't be done in the shared ring.** In pw9/pw10 the stage depth is bounded by the 16 KB x tile riding in every slot (8 × 22 KB = 177 KB already). So the W stream must have its own ring; see the combined fix under (iii).
- **Expected:** −3 µs.

## (ii) +4.5: x tiles re-read from L2 (128 MB on top of 50 MB of W)

- **Mechanism.**
  - At BN 64 every n-block CTA pulls its half of x: 64 n-blocks × 2 MB = 128 MB, through TMA from L2.
  - Total L1-line traffic is 178 MB in 20.4 kernel-µs, about 8.7 TB/s into the SMs. Of that, about 6.3 TB/s is L2 hits, competing with the 2.4 TB/s of HBM-fill traffic for the W stream.
  - The extra 4.5 µs is 128 MB at about 28 TB/s marginal, which is not a bandwidth wall. The more likely reading is per-SM TMA/L2 delivery: one SM must ingest 22 KB per stage (73% x) at the stream's stage rate. That puts x on the critical path of every stage, behind the W-limited period.
- **Supporting data.**
  - w2c2 (BN 128, half the x traffic) was not faster (38.8), but it carries pooler 32 × 2 = 64 CTAs and two warpgroups.
  - pw9's w2n128a (32 × 4, cluster 4) was 46.1. Neither isolates x.
- **Single fix: share each x tile between two n-block CTAs** with a (2, KS) cluster and `cp.async.bulk.tensor … .multicast::cluster` (ctaMask = the n-pair).
  - x traffic halves to 64 MB, each SM ingests 8 KB less per stage, and the grid stays 128 CTAs.
- **Why not x via `ld.global.nc` (L2 → registers)?**
  - It would turn the 16 KB x tile into 128 ld.v4 per warp per stage, for each of the 4 warps: the same bytes, now through the LSU.
  - wgmma's B operand must come from shared memory, so the data would also have to be stored to smem. That is strictly worse.
- **Expected:** −2 µs (multicast), −2.5 to −3 if combined with BN 128 at 128 CTAs (KS 4) where the cluster cost of (iv) is removed.

## (iii) +5.5: wgmma exposed because the slot release waits for it, starving the ring

- **Mechanism.**
  - Per stage the warpgroup runs: wait full(u) → decode(u) → issue mma(u) → wgmma.wait_group 1 (blocks until mma(u-1) has drained) → release slot(u-1).
  - The ring slot therefore stays held for: the decode, plus the whole MMA, plus the wait that follows it. About 200 + 256 + issue cycles beyond what the stream needs.
  - With 4 slots, about 2 stages of lookahead remain for the producer. That is less than the 22-KB-stage TMA latency (~1 µs from HBM + L2), so decode(u+1) stalls on full(u+1).
  - The tensor time itself (256 clk of 830) is not the cost; the delayed release is.
- **Evidence.**
  - Depth 8 (w1d8: 32.3 vs 34.6, −2.3) recovers about half.
  - The k-split (wk2: 31.7, two warpgroups each holding the slot for half the MMA) recovers −2.9.
  - The wait_group(0)-per-stage control was no better, consistent with release timing being the lever rather than MMA overlap.
- **Is it a warpgroup stalled on wait_group before it can arrive?** Yes, by construction. Two warpgroups ping-ponging on alternate stages, each 64 rows over full K, would halve the hold per slot. But it would not decouple W from wgmma; it would still need the larger ring.
- **Single fix: split the rings.**
  - **W ring:** W chunks in their own deep ring (cost (i) fix). Slots released right after the decode, as in pw7: W smem is dead once it is in registers.
  - **x ring:** x tiles in a second ring released after the covering wait_group. x is L2-resident with short latency, so 4–6 x slots (64–96 KB) hide its wait easily.
  - The producer issues the two streams independently (two full/empty barrier sets).
  - This removes the coupling between the HBM-bound W stream and the MMA, and lets (i) be fixed in the same edit.
- **Expected:** −4 µs (together with (i): −6 to −7).

## (iv) +4.5: cluster-2 epilogue (launch, DSMEM reduce, smem transpose, 1 MB store)

- **Mechanism.**
  - The full path is the only rung launched as a cluster. pw7 measured cluster vs global epilogues on the same kernel family, and cluster was 5–10 µs slower on pooler even at m=1, where the reduce moves almost nothing. So most of (iv) is the cluster launch/placement plus two cluster barriers, not the DSMEM bytes.
  - The transpose through smem plus 16-B row stores (1 MB at m=128) is a smaller share: about 64 KB staged per CTA, read once.
- **Single fix: no cluster.** With KS = 2, each CTA writes its fp32 partial (64 × 128 × 4 = 32 KB) with plain coalesced stores in fragment order, then fences and counts. The last-arriving CTA of the n-block adds the peer partial from L2 (32 KB read, one round trip) to its own registers. It then writes bf16 out directly from the D layout:
  - Each thread holds rows g/g+8 × tokens 8j+2t, +1.
  - `stmatrix.sync.aligned.m8n8.x4.trans` to smem, then 16-B row stores; or bf16x2 stores along n, pairing g with g+1 via one shuffle.
  - Traffic: 4 MB of partial writes + 4 MB of reads over 128 CTAs, both L2.
  - This differs from pw5's failed finalize: that one read back KS-1 partials per n-block at the tail of a long stream with many n-blocks per CTA. Here each CTA owns one n-block and the read-back is a single 32 KB L2 read.
- **Expected:** −3 µs.

## Ranked plan and expected pooler m=128 time (in-prepare, torch 28.9)

| step | change | removes | expected after |
|---|---|---|---:|
| 0 | pw10 w1d4 (measured) | | 34.6 |
| 1 | split rings: deep W ring (≥ 48 KB in flight, released after decode) + x ring released after wait_group | (i) + most of (iii) | 28 |
| 2 | no-cluster epilogue: plain-store partials + last-arriver, register D-layout bf16 store | most of (iv) | 25 |
| 3 | x multicast over an n-pair cluster (or BN 128 at KS 4 once clusters are out of the epilogue) | about half of (ii) | 23 |
| 4 | two warpgroups ping-ponging stages (only if a gap between L3 and L1 remains after 1) | the rest of (iii) | 22 |

- **Steps 1 and 2:** independent, and both testable in one check as separate PATHS on the pw10 kernel.
- **Probe for step 1:** the L1 no-x rung with the separate W ring should land at about 17.5 (the bare stream).
- **Step 3:** reintroduces a cluster, but for the producer only (TMA multicast); the epilogue stays cluster-free.
- **If steps 1 and 2 land as estimated:** pooler m=128 beats torch (≈ 25 vs 28.9) and unpooler likely follows (torch 23.3, w2c2 already 26.8). o (K=1024, only 16 stages) stays dominated by fixed cost, and qkv by its 36-n-block grid.

## pw13 post-mortem (check `79ca764ed3ca465c`) and the CUTLASS option

Ladder on h1, with pw10 in brackets: W-only 19.2 (20.9) | W+x 25.3 (25.4) | +decode 25.4 (24.6) | +wgmma 33.9 (30.1) | full 42.1 (34.6).

**1. Why each fix regressed.**

- **W ring (−1.7 of 3.7).** Bytes in flight were the right cause, but not the only one. The remaining ~2 µs over the bare stream is 128 CTAs on 132 SMs, KS=2 halving each CTA's stream (384 KB), and the ramp/tail per CTA.
- **wgmma step (+8.5 vs +5.5).** Yes: 4 x slots, each freed only after `wait_group<1>`, leave at most 2 x stages in flight. Worse, the single producer lane uses a merged in-order schedule. Whenever it blocks on an x empty slot (held by a slow MMA), it also stops issuing W, so the MMA latency leaks into the HBM stream (head-of-line blocking). The right experiment was:
  - one producer lane per ring, so W never waits on x;
  - a deeper x ring (6–8 × 16 KB).
  
  A single ring with an early W release is not possible: a slot holds both.
- **Epilogue (+8.2 vs +4.5).** In a one-wave grid every CTA reaches the epilogue together. Each CTA's tail is a dependent chain: store 32 KB of partial, fence (waits for L2 acknowledgement), atomic, then on the last CTA a 32 KB L2 read, the add, the transpose and the store. That is several serialized L2 round trips with nothing behind them. pw11 measured ≥ 2 µs for that chain at a tiny partial; at 1 MB per n-block pair it costs more. cuBLAS avoids it for pooler: 64×64 tiles over tokens and rows, 2 CTAs/SM, enough tiles without split-K. It re-reads W from L2 instead of reducing partials. The lesson: at m=128, split-K tails cost 4–8 µs in every variant. Tiling the tokens (wgmma N=64, two token tiles) is the cuBLAS structure we never tried.

**2. CUTLASS sm90 mixed-input (example 55) on the judge.**

- **Headers.** The minimal set (CuTe plus the cutlass gemm/collective/kernel/epilogue/arch/numeric headers that example 55 pulls in) is about 4–6 MB of source. xz plus base64 brings that to about 1–1.5 MB embedded in submission.py, written to a temp include directory in prepare and passed via `-I`. That is mechanically possible; whether a 1.5 MB submission passes the judge's checks is unknown.
- **Compile.** 30–90 s per collective instantiation on top of our ~45 s. That leaves room for two or three configurations under the 240 s guard.
- **Codec fit.** This is the blocker. The mixed-input collective converts a narrow A type through `NumericArrayConverter` with scale/zero modes. A 12-bit element is not a CuTe sub-byte type (4/8 bit), and the escape fixup needs a per-warp side table walked inside the mainloop. Both mean forking the ~1500-line collective, i.e. our own kernel on CUTLASS's skeleton, with new layouts to validate offline.
- **Expected if it worked.** Bound by the 12-bit stream (13.2 µs) plus CUTLASS-level overhead: about 21–24 in-prepare against torch 29.4, and only with a no-split tiling.

**Recommendation: no-go.** Main risk: compile time plus the forked collective for a non-power-of-two element. Payoff: about 0.15–0.2 ln across verify16 and verify32. If the m=128 line is revisited, test token tiling without split-K (64×64-style tiles, 2 CTAs/SM) on the existing pw9 pipeline first: one check, no new dependencies.
