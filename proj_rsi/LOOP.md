# muse_bf16_proj loop — head log

Head: Fable 5.1 (high). Implementers/critics: Opus 5.5 (medium), one candidate directory each.
Goal: board under 3000 µs (speedup ≥ 1.6103×). Incumbent on board: r6-c12, job `23cecdef1bb54c9e`, 3428.0 µs (1.4092×).
Tools: `python3 -m proj_rsi {board,check,wait,result,status,lint,cpu-check,quota}`. Protocol: `proj_rsi/protocol.json`.
History before this loop: `proj_rsi/archive/fable-LOG-2026-10-25.md` and `proj_rsi/submission/NOTE.md` (another session's summary).

## Iteration 0 — 2026-10-25 18:00 UTC: board arithmetic and the plan

Per-load sums from the r6 scored profile (yours / SGLang, µs): decode1 38.2/59.8, decode16 51.2/76.0, decode32 60.4/76.2,
verify1 38.9/75.7, verify16 72.6/85.5, verify32 86.8/101.3. Summed ln(ratio) = 2.058; 3000 µs needs 2.859 (+0.80);
the published floors allow 4.236 (2384 µs). Leverage in score-µs per µs cut on one shape: decode1 14.9, verify1 14.7,
decode16 11.2, decode32 9.5, verify16 7.9, verify32 6.6.

What r6 leaves on the table, by mechanism:

| mechanism | shapes | r6 yours → plausible | score µs |
|---|---|---|---|
| packed-weight (12-bit lossless) tensor-core GEMM at m=128; r6 runs cuBLAS on 67 MB uncompressed at 3.5 TB/s | verify16 qkv/o/pooler/unpooler | 15.4/7.2/20.3/19.3 → ~9.5/5/15/15 | ~140 |
| same kernel at m=32; r6's SIMT GEMV is compute-bound there (16.9 vs 14.3 at m=16) | decode32 qkv/o/pooler/unpooler | 14.2/6.6/16.9/16.7 → ~8.5/4.5/13.8/13.8 | ~130 |
| same kernel at m=16 | decode16 | 51.2 → ~44 | ~80 |
| streaming efficiency + launch/tail overhead on tiny shapes (unpooler sliced 4.9 vs 1.75 floor, router 4.1 vs 1.75, qkv 10.0 vs 7.9) | decode1, verify1 | 38.2 → ~32.5 each | ~165 |
| router m=128/256 as one K=2K GEMM on packed hi/lo | verify16/32 router | 10.3/11.1 → ~4.5 | ~90 |
| wgmma at m=256 (compute-bound, cuBLAS at 71% of peak) | verify32 | 86.8 → ~78 | ~70 |

Everything together projects to roughly 2850–2950 µs; the goal needs about 430 µs of the ~675 listed. The previous
loop's failures (direct-global mma.sync at 0.4–0.7 TB/s; hand wgmma with a 35–60 µs epilogue; cuBLASLt sweeps that
changed nothing) were execution failures of specific designs, not evidence that the floor is unreachable; the leader
sits at 3000.75 with the same rules.

Design chosen for the rewrite ("pw"): one kernel family, packed weights in mma-fragment order streamed once from HBM
with deep bytes-in-flight, dequantized in registers, tensor-core MMA against x held in shared memory, K split across
CTAs to fill 132 SMs, partials reduced through fp32 atomics into a workspace zeroed inside kernel_fn, converted by the
last arriving CTA. Served shapes grow outward from m∈{16,32,128}; every unserved shape stays on the torch fallback.

Execution constraints: no local GPU. Measurement is `okc check` only (5 left today, 20/day from 00:00 UTC, 10 min a
run, prepare ≤ 300 s). Every candidate therefore carries its own bake-off (template: `proj_rsi/template/submission.py`)
so one check times every variant and never dies on a kernel bug.

## Iteration 1 — pw1: packed tensor-core streaming GEMM for m ∈ {16, 32, 128}, four BF16 layers

Brief (to Opus implementer, medium): start from the template; reuse r6's judge-proven 12-bit codec and
m16n8k16 fragment order; pack W in fragment order per n-block so each CTA streams one contiguous range; two stream
engines as separate paths (register ring of ld.global.nc.v4, and cp.async.bulk 1D stages with mbarriers); x staged
in smem; K split across CTAs with fp32 red.global.add partials and a self-cleaning last-arriver epilogue; serve the
four BF16 layers at m ∈ {1, 8, 16, 32, 128}; router and m=256 stay on the fallback. Pass bars (yours µs): pooler
m=128 ≤ 17, unpooler m=128 ≤ 17, qkv m=128 ≤ 11, pooler m=32 ≤ 15, qkv m=32 ≤ 10, pooler m=16 ≤ 14.0,
pooler m=1 ≤ 13.5, unpooler m=1 ≤ 3.5.

Delivered: `candidates/pw1/` — 13 instantiations behind 22 paths, 0 spills (nvcc 12.8 local, now installed at
/usr/local/cuda-12.8), CPU emulator bit-exact on 7 geometries. Critic (separate Opus agent): APPROVE, no blockers;
applied its fix (launch error in kernel_fn now falls back instead of raising) and moved the prepare budget guard
after serves() so a timeout cannot flood the log. Spent check #1 (5 left today before the 00:00 UTC reset).

### Result — check `c14d12e8d3f847a9`: 9/9, 4731.7 µs (1.021×). Correct everywhere, fast nowhere.

Prepare: compiled in 33 s, done in 53 s. Every one of the 120 pw builds passed the one-hot and random validation on the
judge (0 INVALID, 0 ERROR) — codec, fragment order, escapes, split-K reduction and the self-cleaning epilogue are all
right on the first run. Bake-off (in-prepare µs, which run ≈ 3–4 µs above judge "yours"; torch column is cuBLAS):

| shape | torch | best pw | r6 "yours" | note |
|---|---:|---:|---:|---|
| pooler m=1 | 24.8 | 22.3 (bulk BN64) | 13.95 | stream ≈ 2.7 TB/s; ring variants 27–29 |
| qkv m=1 | 17.9 | 16.5 (bulk) | 9.97 | |
| unpooler m=1 (n=1024) | 11.1 | 11.2 | 4.88 | 6 MB read costs 7 µs: fixed overhead dominates |
| pooler m=16 | 25.2 | 24.2 | 14.30 | |
| pooler m=32 | 25.6 | 33.6 | 16.94 | 2× slower than m=1: per-stage cost scales with m |
| pooler m=128 | 30.5 | 158.9 (BN128) | 20.32 | 128 bulk copies of 128 B per stage |
| qkv m=128 | 21.3 | 99.7 | 15.43 | |

Judge picked pw only on qkv/pooler at m ∈ {1, 8, 16} and unpooler m=8, each still slower than r6; score fell to 4731.7.

Diagnosis: x is staged through the shared-memory ring with one `cp.async.bulk` per token row per 64-k stage, and
every stage needs a producer→consumer→producer mbarrier round trip. Cost scales with m (128 TMA issues per stage at
m=128) and even at m=1 the handshake bubbles cap the W stream below r6's register ring. The ring variants inherited the
same x handshake, so they were no better. Nothing here is a verdict on the packed tensor-core idea; it is a verdict
on the data flow.

Decision: refine → pw2, "x-stationary CTAs": each CTA loads its x K-slice into shared memory once (one bulk copy per
row, total ≤ 128 KB), then streams W for several n-blocks with no further x traffic or per-chunk handshakes; W through
either a deep register ring (ld.global.nc, no barriers) or a bulk ring whose only barriers are W's own. Plus a pure
streaming PROBE in prepare that reports achieved TB/s per engine/config, so the next check calibrates the engine even
if the GEMM paths lose again.

## Iteration 2 — pw2: x-stationary CTAs + stream probe. Check `66c8bc426e1748c1`: 9/9, 4766.6 µs.

GEMM paths still lose everywhere (in-prepare µs; torch=cuBLAS; graph replay itself is 5.1): pooler m=1 b8t1 24.2 vs
torch 25.4 vs r6-yours 13.95; m=32 b8t4 29.2 vs 26.0; m=128 bm24 64.6 vs 30.1. All paths valid again.

PROBE (the point of this check), 50.3 MB packed pooler, cold L2, µs minus 4 for the replay:
- register ring (`ld.global.nc.L1::no_allocate.v4`): 0.6 TB/s at 4 warps×1 CTA/SM … 2.3 TB/s at 8×2 … 3.3 TB/s only at
  8 warps×4 CTAs/SM (D=8). A register ring needs ~32 warps/SM to stream; as an engine it is dead.
- bulk (`cp.async.bulk` 1D into an smem ring): 3.70–3.83 TB/s for every stage size (16/32 KB), depth (2/4/6) and
  1–2 CTAs/SM. **3.8 TB/s is the HBM ceiling on this judge**: 50 MB = 12.5 kernel-µs ≈ 17.5 in-prepare ≈ 14.0 judge-yours.
  r6's pooler m=1 (13.95 yours) is already at that ceiling with 12-bit packing.
- 6 MB read: 3.4 kernel-µs (fixed latency floor ≈ 2.5 µs); so unpooler-sliced/router/o can gain ≤ ~1 µs each.
- `red.global.add.v4.f32`, 16 MB: 6.5 µs = 2.5 TB/s. Split-K partials at m=128 (2 MB per K-split for pooler) cost
  ~0.8 µs per split: KS must be ≤ 4, which rules out x-stationary smem at m=128 (x slice would be 512 KB).

What this changes in the plan: the headroom at m ≤ 8 is ~1–2 µs per big shape, not 3–5. The gains that reach 3000 are
m=32 and m=16 brought to the stream ceiling (r6: 16.9/14.8 at m=32, 14.3/14.8 at m=16 vs ~14.0 ceiling; qkv 14.2/11.2
vs ~8.4) and m=128 (cuBLAS 20.3 vs a ~15 µs design), plus the router. The consumer path (decode + mma + barriers)
in pw2 costs ~6.5 µs on top of the stream even at m=1, i.e. it is not overlapped; that must be decomposed before
another design round.

Decision: refine → pw3: (1) probes that decompose the consumer cost (stream-only / decode-only / full; 8 vs 16
consumer warps; 1 vs 2 CTAs/SM; stage sizes) on pooler m=1 and m=32; (2) m=128 path with x streamed by one 2D TMA
tensor copy per stage (r7's proven `make_xmaps` / `cp.async.bulk.tensor.2d`), BN=128, KS ≤ 4; (3) drop the register
ring paths.

## Iteration 3 — pw3: consumer-cost decomposition + 2D-TMA m=128. Check `5d0ab1df4e61480e`: 9/9, 4861.0 µs.

PROBE cons (pooler, in-prepare µs; replay alone 4.8; pure bulk probe of the same 50 MB from pw2: 16.5–17.0):

| config | L1 wait/arrive | L2 +lds | L3 +decode | L4 +mma | L5 full | AG2 |
|---|---:|---:|---:|---:|---:|---:|
| m=1, 8 warps, 1 CTA/SM (grid 126) | 22.2 | 20.4 | 21.4 | 21.3 | 23.5 | 24.3 |
| m=1, 8 warps, 2 CTAs/SM (grid 260) | 22.4 | 22.8 | 22.8 | 23.6 | 26.9 | |
| m=1, 16 warps, 1 CTA/SM (grid 128) | 20.4 | 20.3 | 20.7 | 21.0 | 22.7 | |
| m=32, 8 warps, 1 CTA/SM | 26.0 | 26.3 | 26.5 | 27.8 | 29.3 | 31.3 |
| m=32, 16 warps, 1 CTA/SM | 25.2 | 25.4 | 25.6 | 25.8 | 26.6 | |

Reading: the compute (lds+decode+mma+escapes) costs 1.3–2.3 µs over the skeleton and is nearly hidden; 16 warps hide
it best. The **skeleton itself** (producer/consumer barriers, prologue, per-n-block flush and end-of-CTA finalize) is
4–5 µs slower than the pure bulk stream, and at m=32 the skeleton is 5 µs slower again than at m=1 although the W
bytes are identical (only x prologue, x smem footprint, workspace/flush sizes change). That is the target: a skeleton
at ~17 in-prepare gives a full GEMM at ~18.5 → judge yours ≈ 13.6 at m=1/16/32 vs r6 13.95/14.30/16.94.
m=128 TMA x-ring: tm 41.7 (pooler) vs cuBLAS 29.8; halving KS (tmh) 62 — fewer CTAs hurt more than reductions.
Parked until the m ≤ 32 skeleton is solved; the same fix likely applies.

Decision: refine → pw4: skeleton-isolation probes on the L1 kernel (no epilogue; no prologue; free-running producer
with no empty waits; n-blocks per CTA 1/2/4; KS 3/7/14; grid 126/132/264; stage 16/24/32 KB), each one PROBE line, so
check #4 names the microseconds. Keep the GEMM paths unchanged.

## Iteration 4 — pw4: skeleton isolation. Check `6267697e34674a82`: 9/9, 4765.0 µs.

PROBE skel (pooler, in-prepare µs, same run; the judge drifts ~10% between runs — `pure` was 16.5 in pw2, 18.7 here —
so only within-run deltas count):

| line | m=1 | m=32 | | line | m=1 | m=32 |
|---|---:|---:|---|---|---:|---:|
| pure stream (reference) | 18.7 | – | | free (no empty waits) | 20.8 | 25.1 |
| base (L1 skeleton) | 20.4 | 25.1 | | latecopy | 21.2 | 25.8 |
| **noepi** | **18.1** | **19.9** | | nb1 / nb2 / nb4 | 20.8 / 21.1 / 22.1 | 25.1 / 26.9 / 31.8 |
| nopro | 20.6 | 23.4 | | ks3(5) / ks7 / ks14 | 22.9 / 20.5 / 21.3 | 25.4 / 24.9 / 26.9 |
| noepi+nopro | 18.2 | 18.0 | | g132 / g264 | 21.9 / 22.1 | 27.9 / 29.0 |
| xreg (m=1 kernel, m=32 prologue) | – | 22.1 | | st16 / st24 / st32 | 21.1 / 20.6 / 20.6 | 25.3 / 25.4 / 25.3 |
| smallws | – | 23.5 | | | | |

The epilogue (per-n-block `red.global.add.v4` flush → `__threadfence` → counter atomic → last-arriver read/convert/
store/zero) is the whole gap: 2.3 µs at m=1, 5.2 µs at m=32; without it the skeleton equals the pure stream. The
handshake, prologue order, n-block range, K-split count, grid and stage size do not matter (≤ 0.5 µs, or worse).
GEMM paths: 16-warp (BN=256) beats 8-warp on every shape except unpooler n=1024 (b16t1 qkv m=1 17.0 vs b8 19.7;
pooler m=32 26.4 vs 28.8); the judge picked pw on qkv/pooler at m ≤ 16, still behind r6 by the epilogue margin.

Decision: refine → pw5: (1) epilogue decomposition probes on the L1 skeleton at m=1/32: red-only, +threadfence,
+counter, +finalize (full), finalize with all 512 threads reading, and an overlapped variant; (2) real-path change:
consumers dump each finished n-block's accumulators to shared memory and continue; two dedicated epilogue warps do
red.global + fence + counter + finalize from that buffer while the stream runs on, so only the last n-block's finalize
is exposed; (3) keep b16 paths, drop b8 except for unpooler n=1024.

## Iteration 5 — pw5: epilogue decomposition + overlapped epilogue. Check `eff33a46f7394a11`: 9/9, 4828.6 µs.

PROBE epi (pooler, in-prepare µs, same run; pure 18.6):

| line | m=1 | m=32 | | line | m=1 | m=32 |
|---|---:|---:|---|---|---:|---:|
| noepi | 17.8 | 19.3 | | fin512 (wide finalize) | 20.6 | 23.7 |
| redonly | 18.1 | 20.3 | | redv1 (scalar atomics) | 21.1 | 31.3 |
| red+fence | 18.6 | 20.7 | | fencecta | 20.5 | 24.5 |
| red+fence+cnt | 18.9 | 21.3 | | overlap (2 epilogue warps) | 21.7 | 31.3 |
| full (+finalize) | 20.5 | 25.1 | | overlap-nb4 / nored | 25.4 / 21.9 | 44.9 / 31.3 |

The split-K **finalize** — the last-arriving CTA reading the fp32 partials back from L2, converting, storing and
re-zeroing — is the cost: 3.8 of 5.8 µs at m=32, 1.5 of 2.6 at m=1. It is dependent L2 round trips at the tail of a
one-wave grid, so nothing in the CTA overlaps it. The reds, fence and counter together are ≤ 2 µs. The overlapped
epilogue (pw5's real paths) was worse everywhere: it needed more K-splits (grid 116–120 with ≥ 2 n-blocks per CTA)
and still serialized the last finalize; b16t4 pooler m=32 went 26.4 → 34.1. Rejected.

Second observation, from `yours` vs profiled kernel time across pw3/pw4/r6: r6's zero-smem `gv` kernels run at
their profiled time (gap −0.3…+0.9 µs); every large-smem kernel — ours (+2.5…3.5 at m=1), cuBLAS nvjet (+1.5…4.9),
the FP32 xmma (+6.8) — shows `yours` above its profiled time. That is what an L1/shared-memory carveout
reconfiguration at launch would look like under the judge's alternating replay (flush kernel → ours → SGLang's), and
it would tax every TMA/smem-heavy design in this repo's history by ~2 µs per shape. Must be measured, not assumed.

Decision: reject pw5 → pw6: (1) carveout probes: empty kernel with 0 / 64 / 200 KB dynamic smem after the same
flush; pure bulk stream with 8/16/32/64/100 KB rings; flushes performed by our own 0-smem and 200-KB-smem kernels;
(2) real paths: **cluster split-K with DSMEM reduction** — KS CTAs of one n-block form a cluster, each parks its fp32
accumulators in its own smem, one cluster barrier, each CTA reduces a 1/KS slice across the peers' smem
(`ld.shared::cluster`), converts and stores bf16; no workspace, no counters, no atomics, no read-back. Tail ≈ 0.3 µs.

### SASS audit (between checks, 2026-10-25 20:40 UTC) — `proj_rsi/audits/pw6-sass-audit.md`

Per 64-k chunk per warp the consumer loop issues 111 instructions at m=32 (ideal ~74) and 90 at m=1 (ideal ~56); decode is at
its floor (36), the ~45 extra are bookkeeping (runtime ring depth and x row stride, rematerialized addresses, escape seq
recompute, dead n-block test, late arrive). No bank conflicts. Issue and tensor pipe have ~3× slack; **shared-memory read
bandwidth is the closest-to-binding resource at m=32 (54 %, 64 % at 4.5 TB/s)** because every warp re-reads the same x
fragments — the fix is 2 row tiles per warp sharing one x fragment set. Six edits filed for pw7 (five numerics-preserving).

## Queued for the 00:00 UTC reset (built while the quota was exhausted)

- **pw7** = pw6 (carveout probes + cluster-DSMEM split-K) + the SASS-audit edits: 111 → 84 instr/chunk at m=32, 90 → 69 at
  m=1 (verified by cuobjdump), early arrive, chunk-major x prefetched before the wait, RT=2 warp-shape paths. Critic: APPROVE.
  Runs first (check #7); pw6 is the fallback.
- **pw8** = pw7 + the fused router: hi/lo 12-bit packs laid out as one K' = 2K stream per n-block, x read with K-wrap,
  FP32 output, BN=32 paths with cluster (8×8) and global (8×16, 8×32) epilogues for m ≤ 32 and a global 8×32 path at
  m=128. Critic caught a **blocker** — the FP32 store helper recursed into itself and nvcc emitted an infinite loop in
  every BF16 epilogue (prepare would have hung and killed the check); fixed to call `pw_put`, PTX re-verified
  (0 self-loops, 352 bf16 stores, 0 spills). Runs as check #8 after pw7's result is read.


## Parallel branch opened 23:40 UTC (before check #7): pw9 = wgmma m=128/256 kernel

Earlier rule ("design m=128 after the carveout answer") dropped: both answers go into the bake-off, and building does not
consume quota. Reason the branch is needed: verify16 72.6 µs and verify32 86.8 µs are cuBLAS on 67 MB uncompressed;
a packed stream gives ~15 (pooler/unpooler) and ~9.5 (qkv) → ~17 + ~20 µs → ~0.3 of the 0.8 ln-gain. Why pw3's m=128
path lost (41.7 vs 29.8): mma.sync tops at ~600 TFLOP/s, so 8.6 GFLOP (pooler m=128) is ~14 µs of issue on top of
decode — not hideable under a 13 µs stream. pw9 therefore uses wgmma m64nNk16 with A (decoded W) in registers and B
(x tile, 2D TMA, 128B swizzle) via smem descriptor; BN=128, KS from {2,4,7,8}, cluster-DSMEM split-K as pw7.
Implementer: build on pw8 (m≤32 + router unchanged). Critic: independent PTX-ISA layout/descriptor study
(audits/wgmma-layout.md) to cross-check fragment order before a check is spent.

## Step-back: where 3000 can come from (computed 23:40 UTC on the r6 table)

Ceiling model per shape: packed 12-bit W bytes / 3.8 TB/s + 1.0 µs (6 MB shapes floor 3.4), or the wgmma compute
floor 2mNK / 790 TFLOP/s + 1 when larger (binds only at m=256). Judge-`yours` µs:

| load | r6 sum | all-at-ceiling | ln available |
|---|---:|---:|---:|
| decode 1 | 38.2 | 34.4 | 0.105 |
| decode 16 | 51.2 | 44.7 | 0.137 |
| decode 32 | 60.4 | 44.7 | 0.302 |
| verify 1 | 38.9 | 34.4 | 0.123 |
| verify 16 | 72.6 | 44.7 | 0.486 |
| verify 32 | 86.8 | 68.6 | 0.236 |

Score if everything hit ceiling: **2720 µs**. Need 0.800 ln of the 1.389 available (58%).
- m≤32 + router only (pw7+pw8 perfect): **3067** — cannot reach 3000 on its own. The m=128 kernel (pw9) is required.
- m≤128 at ceiling: 2829. So the plan is pw7 (decode16/32, verify1) + pw8 (router, all loads) + pw9 (verify16, with
  verify32 as a bonus; m=256 is near its compute floor already so its gain is ~0.1 at best).
- Realistic (each branch captures ~70% of its pool): 0.45 + 0.22 + 0.30 ≈ 0.97 > 0.80. Feasible, but all three must land.

The one lever that moves the floor itself: bits per weight. 11-bit lossless would cut every big shape 8% (~0.1 ln
across the board). pw9's prepare prints STATS (exponent window coverage, mantissa low-bit zeros, entropies) so check #9
answers whether a 3-bit-exponent-with-escapes or trailing-zero-mantissa codec is viable at decode cost ≈ 12-bit's.

Dead ends, not to be revisited: register-ring W loads (≤3.3 TB/s); red.global split-K partials at m≥32; overlapped
finalize epilogue; x-stationary smem at m=128; mma.sync at m≥128 (issue-bound); full entropy coding (decode rate).
Kill criteria: pw7 — decode32 sum not below 52 ⇒ the skeleton design is wrong, go register-direct (gv lineage);
pw9 — pooler m=128 not below 17 ⇒ wgmma pipeline not overlapping, read ptxas serialization and stage release.

## Iteration 6 — pw7: carveout probes + cluster split-K + SASS edits. Check `ce07e92197154629`: 9/9, 4721.3 µs. REJECTED.

- Carveout tax: **disproven**. Empty 132×256 launch 5.02 / 4.90 / 4.90 µs at 0 / 64K / 200K smem (x2 CTAs 6.59).
- Bare bulk stream of 50 MB, in-prepare: 17.2 µs (ring ≥ 64K); register ring D=8: 19.1. Flush method irrelevant.
- Cluster DSMEM epilogue (c) vs global (g): pooler m=1 28.8 vs 21.8, m=32 36.0 vs 26.2 — **c is 5–10 µs slower**;
  qkv ≈ equal; unpooler b8t1c 11.0 vs 11.3. RT=2 slower everywhere (+3–4 µs).
- Judge: pw-b16t1g pooler m=1 17.86 vs r6 13.95; everything else fell to cuBLAS in the bake. decode32 sum 77 (kill
  criterion was 52).
- The decisive comparison: pw skeleton 21.8 in-prepare vs bare stream 17.2 vs r6 ≈ 18 (judge 13.95 + ~4 offset).
  The TMA-ring + mbarrier + split-K-finalize skeleton has been 4.6 µs over the stream in every variant pw3–pw7.
  The pw4/pw5 attribution to the "finalize read-back" was wrong in kind: removing it via cluster reduction cost more.

Decision: **kill the pw skeleton for m ≤ 32**. Pivot to r6's register-direct lineage (zero smem, at the ceiling at
m=1) extended to m=16/32 and qkv/o/router — analysis commissioned (audits/r6-lineage.md). pw8 (router on the pw7
skeleton) is skipped: its m ≤ 32 router paths would carry the same 4.6 µs and r6's router is at 4.1–5.9. Its router
paths are included in pw9 anyway. Check #8 = pw9 (wgmma m=128/256 + pw8 router + STATS), job `520ad1dce4144c76`:
kept because the wgmma compute question at m=128 is independent of the skeleton result; caveat that pw9 uses the
cluster epilogue that lost here.

## Iteration 7 — pw9: wgmma register-A m=128/256 + router + STATS. Check `520ad1dce4144c76`: 9/9, 4405.5 µs. REFINE.

In-prepare bake (µs), m=128: pooler torch 30.2 | w1n128a 64x2 **34.4** | w1n128b 64x4 52.3 | w2n128a 32x4 46.1 |
w2n128b 32x8 44.4; o torch 11.0 | w2n128a 64x2 16.3 | w1n128a 128x1 17.3; unpooler torch 23.2 | w2n128a 64x2 29.8;
qkv torch 22.4 | w2n128a 18x4 26.7. m=256: pooler torch 34.0 | w1n256a 43.6. Router m=128: rtg128 8x32 28.0 vs
torch 23.4 (dead); router m ≤ 32 on the pw skeleton 9–12 vs r6 4–6 (dead, as predicted).
Reading: best = fewest consumer warps (1 warpgroup) + smallest cluster (2), still 2× the stream (34.4−5 = 29 kernel-µs
vs 13.2 of bytes). Decode throughput per SM is the limit: 4 consumer warps × ~1100 issue slots per 64-k stage × 64
stages ≈ 38 µs; wgmma is 4× cheaper (245 clk/stage). Two warpgroups were only tried with clusters 4/8, and each
cluster-size step costs 10+ µs (as in pw7). Cluster launch cost is now a measured fact on this judge.
STATS (weights are synthetic; entropy 10.46 bits/weight): best 8-code exponent window leaves 1.25–1.36% escapes
(router lo 4.0%); 16-code window 0.01%; mantissa low bits uniform (no trailing-zero gain). An 11-bit codec would need
~7 escapes per 512 — dense fixups, maybe 5% stream saving (0.05 ln); parked.
Decision: refine → **pw10**: same kernel, consumer warpgroups split each stage by k16 slices (W=4 at m=128, W=2 at
m=256) on the same 64 W rows, accumulators summed through smem at the end; cluster 2 only. Expected pooler m=128 ≈
21 in-prepare vs torch 30.2. r6-lineage analysis for m ≤ 32 still pending.

## r6 lineage audit (audits/r6-lineage.md) — the m ≤ 32 pivot

Why r6's gv kernel sits at the stream: each warp loads W straight into registers (ld.global.nc.L1::no_allocate.v4,
3–4 chunks in flight at launch), decodes once (9 ops per 8 weights) and feeds mma.sync as the A operand; each CTA owns
16–32 rows and splits K across its own 4 warps, reduced through one smem round — **no cross-CTA split-K at all**, no
atomics, no workspace, no CTA waits on another. pw's 256-row tiles forced a cross-CTA split-K, and that tail is the 4.6 µs.
What limits r6: x is re-fetched once per row tile, so L2→SM traffic is 0.08× W at m=1, 1.33× at m=16, 2.67× at m=32;
fitted L2→SM bandwidth ≈ 7 TB/s (qkv32 moves 103 MB in 14.1 µs). qkv m=1 is geometry: 144 CTAs on 132 SMs.
Correction: in r6's own run only decode32 o falls to cuBLAS at m ≤ 32; all m=128/256 are cuBLASLt (GVRB=[]); the
router does not use the 12-bit codec (streams 8.4 MB instead of 6.3).
Extension (pw11 "gvx", built by the analyst on a copy of r6): 128–144-row CTAs with K split across KS CTAs, reduction
either (A) cluster DSMEM or (B) global last-arriving-CTA with ~4 MB of traffic and no cluster; o with x in smem;
router 12-bit hi+lo. Model: qkv ~8.7–9.0 (9.97–14.21 now), pooler m32 ~14.3, unpooler m16/32 ~14.3, o ~4.6, router
~3.7–4.0 → ≈ +0.53 ln. Kill if pooler m=32 (B) is not ≥ 1.5 µs under r6's own in the same bake.

## Note for pw12 (after pw11's result): the stream itself has ~15% headroom over r6

Judge device is an H200 (PREP log), HBM3e. From the pw9 probes (in-prepare minus the 4.9 µs empty launch): bulk TMA
ring ≥ 64K streams 47–50 MB in 11.4–12.0 µs = **4.15–4.4 TB/s**; the register ring D=8 takes 14.0 µs = 3.55 TB/s; r6
pooler m=1 at 13.95 judge-µs = 3.6 TB/s. So r6's register loads leave ~2 µs per 50 MB shape on the table, and the pw
lineage captured the faster stream but lost 4.6 µs to its producer/consumer skeleton. Candidate design for pw12:
r6's structure (per-CTA row tile, K split across the CTA's own warps, in-CTA smem reduction, no cross-CTA split) with
each warp streaming its own contiguous K range through a private cp.async.bulk ring and a private mbarrier — no
producer warp, no cross-warp handshake — i.e. only the load instruction changes. Ceiling model would move from
bytes/3.8 + 1 to bytes/4.3 + 1 (pooler 12.7 instead of 14.25), worth ≈ 0.1 ln across the m ≤ 32 loads.

## Iteration 8 — pw10: m=128 latency variants + probe ladder. Check `7715c850ff89450a`: 9/9, 4386.8 µs. PARKED.

Bake, pooler m=128 in-prepare: torch 28.9 | wk2 (k-split) 31.7 | w1d8 32.3 | w1d4 (=pw9) 34.6 | w1d8z 38.7 | w2c2 38.8.
unpooler: torch 23.3 | w2c2 26.8; o: torch 11.6 | w2c2 16.5; qkv: torch 21.6 | wk2 30.1. m=256 all worse.
Ladder (pooler m=128, 64x2): L1 W+x 25.41 | L1 no-x 20.90 | +decode 24.58 | +wgmma 30.14 | full 34.62; bare bulk
stream of the same bytes 17.2. Four separate costs of 4–5 µs each: cluster-2 skeleton streaming W alone +3.7;
x tiles +4.5; wgmma exposed +5.5 (not hidden despite 256 clk/stage of tensor time); epilogue +4.5. Depth 8 and the
k-split each recover 2–3 of the 18. Analysis-only follow-up commissioned (audits/pw10-ladder.md); no build.

Re-derived target math with the stream headroom (bulk 4.3 TB/s vs r6's 3.6–3.8):
- ceiling bytes/3.8+1: m ≤ 32 + router alone → 3067; plus m=128 big four at cuBLAS−3 → 2976.
- ceiling bytes/4.3+1: m ≤ 32 + router alone → **2879**; plus m=128 → 2794.
So if pw12 (r6 structure + per-warp bulk ring) delivers the faster stream, the m=128 kernel is no longer required
for 3000. Main line is now pw11 → pw12 on the r6 lineage; m=128 is the reserve.

## Iteration 9 — pw11 (gvx): r6 + wide-row cross-CTA split-K. Check `824f32ce203541c4`: 9/9, 3385.7 µs. REJECTED (= r6).

Score 3385.7 is r6 re-measured: not one gx path won a bake, so the kernel set that ran is r6's; the 1.2% over 3428 is
judge drift (CI 1.405–1.448 covers 1.409). Per-shape deltas vs the incumbent run are all within ±0.6 µs except
verify16/32 unpooler (−0.85/−1.0, cuBLAS noise).
Bake (in-prepare, r6 vs best new): qkv m=1 14.24 vs q128a 15.26 (cluster 8) / q144b 16.19 (last-arriver); m=32 18.69
vs q144b 20.80; pooler m=32 21.28 vs p128b 24.51 (kill rule was ≤ 19.8); unpooler m=32 21.15 vs u128a-s 21.47;
o m=16/32 10.21/10.91 vs o64-s 10.56/11.30; router m=1 8.42 vs r12c16 10.72. Cluster variants for pooler/unpooler
m=16 and qkv (a) were mostly declined by the occupancy guard.
Reading: any cross-CTA split-K costs ≥ 1–2 µs on this judge even inside r6's structure (cluster DSMEM +1.0 at best,
last-arriver +2.0), which exceeds what the x-traffic saving returns at m=32. The wide-row idea is dead in both
lineages. The 12-bit router is slower than r6's bf16 hi/lo pair (escape handling on the lo half). o in smem: no gain.
What remains for m ≤ 32: the stream itself (pw12, per-warp bulk ring, no split-K). For 3000, m=128/256 (pw13) is
back to being required unless pw12 lifts the whole m ≤ 32 table by ~8%.

## Iteration 10 — pw12: r6 + per-warp cp.async.bulk W ring. Check `f1e6a18689484451`: 9/9, 3374.4 µs. REJECTED (= r6).

No gb path won a bake. In-prepare: pooler m=1 r6 17.79 | gb-p16w-n4 18.11 | -n4-f 18.14 (proxy fence costs nothing)
| -n8 19.01 | gb-p-n4 19.58; pooler m=32 r6 21.06 | gb-p-n4 21.86 | gb-p-n8 29.66 (1 CTA/SM); qkv m=1 r6 14.05 |
gb-q-n4 14.11; unpooler m=32 r6 20.83 | gb-u-n4 22.18; o m=1 r6 9.50 | gb-o-n4 9.79.
Correction to the pw12 lead: the "4.3 vs 3.6 TB/s" headroom compared a judge-µs figure for r6 with an in-prepare
probe. In the same units r6 pooler m=1 (17.8 in-prepare) is within ~0.6 µs of the bare bulk stream (16.3–17.2). W
delivery is not a limiter at m ≤ 8; r6's register loads already saturate HBM. The m ≤ 32 pool is therefore only the
m=16/32 excess (pooler/unpooler ~2.7, qkv ~3.5–5.8, o ~1–2 µs over m=1), whose cause pw14's ladder measures.
Ceiling model restated with bw = 3.8 (r6-calibrated): m ≤ 32 + router alone → 3067; so m=128/256 (pw13) is required.

## Iteration 11 — pw13: two-ring m=128 kernel + no-cluster epilogue + multicast. Check `79ca764ed3ca465c`: 9/9, 4395.5 µs. REJECTED; m=128 branch CLOSED.

Pooler m=128 in-prepare: torch 29.44 | h1 42.05 | hk2 39.36 | hmc 44.61 | pw10's wk2 31.90 / w1d8 32.74. Same order
on every layer and at m=256. Ladder on h1: W-only 19.23 (bare stream 17.2) | W+x 25.31 | +decode 25.38 | +wgmma
33.86 | full 42.08. Versus pw10's ladder: the deep W ring recovered 1.7 of the 3.7; the x ring released after
wait_group made the wgmma step worse (+8.5 vs +5.5); the no-cluster last-arriver epilogue is +8.2 vs the cluster-2
DSMEM epilogue's +4.5 (consistent with pw11's finding that the last-arriver chain costs ≥ 2 µs on a one-wave grid, and
here the partial is 1 MB per CTA). Multicast lost further.
Five checks (pw3-tm, pw9, pw10, pw13 + ladder) never beat cuBLAS at m ≥ 128; the best is 2.5 µs behind on pooler and
10 behind on qkv. The in-house wgmma pipeline is closed. Ceiling arithmetic without m ≥ 128 gains: m ≤ 32 + router at
ceiling → 3067; realistic (pw14 explains and recovers half of the m=16/32 excess) → ~3300. 3000 therefore needs a
different m=128 approach (CUTLASS-grade mixed-input pipeline, or none) — decision recorded for the user.

## Iteration 12 — pw14: probe ladder on r6's gv. Check `066cf9bd503040cf`: passed, 1.417× (= r6). REFINE → pw16.

In-prepare (regs / CTAs per SM): pooler m=1 L0 18.24 (2) | L3 18.40 | L4 18.05 — flat, at the stream.
pooler m=32: L0 16.93 (126 regs, 2) | +x 19.74 | +decode 20.51 | +mma 20.74 | full 21.02; cap255 (1/SM) 25.09;
cap64 45.89 (spills); bulk-ring L0 18.05. qkv m=1: L0 13.31 | L3 13.95 | L4 14.21. qkv m=32: L0 12.51 (4/SM) | +x
15.01 | +decode 18.53 (74 regs, 6/SM) | +mma 17.22 (4/SM) | cap255 17.66 | full 18.56. gvbulk n4 17.86 vs r6 17.89.
Reading: the m=32 excess is the x loads (2.5–2.8 µs on both layers) plus decode/compute on qkv (2–3 µs, occupancy-
confounded across steps). Occupancy matters (1 CTA/SM costs +4). Max recoverable ≈ 2.5 µs per big shape at m=32 and
≈ 1.3 at m=16 → ≈ 0.2 ln → ~3310. Next: pw16 = one x-traffic experiment (L1-allocating x loads so co-resident CTAs
share lines / smem x slab / cluster-2 multicast without split-K) on r6's own geometry. m=128: pw15 (token tiling,
no split-K) is the last check on that branch.

## Iteration 13 — pw15: token-tiled no-split-K m=128/256. Check `ce1163b9280c47a2`: 9/9, 4423.1 µs. REJECTED; m ≥ 128 CLOSED.

Pooler m=128 in-prepare: torch 30.46 | q2 (ping-pong) 37.50 | q1a 43.74 | q1b 44.48 | q1c 50.91 | pw10's wk2 31.46.
m=256 far worse (56–93). Ladder (q1b, 2 CTAs/SM): W+x 26.30 | +decode 32.96 (+6.7) | +wgmma 43.26 (+10.3) | full
44.26 (+1.0: the epilogue is finally cheap). Without split-K each CTA decodes its 64 rows over the full K, and with
one consumer warpgroup per CTA the decode is no longer hidden; the wgmma step is worse still. cuBLAS's structure does
not transfer to a decode-bearing kernel at one warpgroup per CTA.
Tally for m ≥ 128: pw3-tm, pw9, pw10, pw13, pw15 — best ever 31.5 vs cuBLAS 29.4–30.5 (pw10 wk2). Closed.
Consequence (ceiling at r6's calibration): m ≤ 32 + router alone reach 3067 at best, ~3300 realistically. 3000 is not
reachable from the current evidence without an m=128 kernel that beats cuBLAS by ~5 µs per big shape, and no design
tried here has come within 2 µs of it.

## Iteration 14 — pw16: x cache policy on r6's m=16/32 geometries. Check `f03e2e7785224172`: 9/9, 3365.0 µs. REJECTED (= r6).

In-prepare: pooler m=32 r6 20.93 | -xel 20.77 | -xcg 29.06 | -L1 28.38; pooler m=16 r6 18.14 | -xel 18.14 | -xcg 22.53;
qkv m=32 r6 18.50 | -xel 18.46 | -xcg 23.20; qkv m=16 r6 16.61 | -xel 16.13; unpooler m=32 r6 20.93 | -xel 20.77 |
-xcg 29.47 | -L1 42.72. Bypassing L1 for x costs 2–9 µs, so r6 already reuses x lines in L1 between co-resident CTAs;
evict_last is a tie; a maximal L1 carveout is a disaster (reduction smem/occupancy). The x-traffic lever is exhausted.

## State of the loop after 16 checks (2026-10-26 ~07:00 UTC)

Board unchanged: r6-c12 at 3428 (re-measured 3365–3386 in today's drift). Every lineage and lever has been measured:
- m ≤ 8: at the HBM stream (two independent lineages within 0.6 µs of the bare bulk probe). Nothing left.
- m = 16/32: excess over m=1 is x loads (2.5–2.8 µs) + qkv decode/occupancy (2–3). Wide tiles + cross-CTA split-K
  (pw11), bulk W ring (pw12), x cache policy (pw16) all tie or lose; L1 reuse already happens. Remaining ideas
  (qkv m=32 geometry with more bytes in flight; cluster-2 x multicast) are each worth ≤ 1 µs net.
- m ≥ 128: five designs, best 31.5 vs cuBLAS 29.4–30.5; closed. Structural lesson: decode must leave the MMA warpgroup.
- router: latency-bound at 4.1–5.9 vs 3.4 floor (≈ 50–70 score-µs if brought to the floor at all m) — the largest
  remaining identifiable pool; 12-bit codec made it slower.
- codec: 11-bit needs 1.3% escapes; ≤ 5% stream saving at best.
Conclusion: 3000 is not reachable from this evidence (ceiling 3067 without m ≥ 128 gains; realistic ≤ 3300–3350).
Options for the user: (A) stop, r6 stands; (B) continue cheap m ≤ 32 / router iterations for ≤ 2–3%; (C) a long
project on an m=128 transform-warpgroup kernel with uncertain payoff (≈ 0.2 ln if it beats cuBLAS by 3 µs).

## Correction (user: "leaderboard is at 3ms"): the codec lever was mis-sized

A 3000.75 exists on the board, so the ceiling model above is missing a lever. Recomputed with r6's table (stream part
= packed bytes/3.8 TB/s scaled by bits/12; everything else as r6 pays it today):

| bits/weight | m ≤ 32 only | + half the m=32 excess | + m=128 big four −3 µs | all three |
|---|---:|---:|---:|---:|
| 12.0 | 3437 | 3327 | 3253 | 3149 |
| 11.4 | 3353 | 3243 | 3174 | 3069 |
| 11.0 | 3297 | 3186 | 3121 | 3015 |
| 10.6 | 3240 | 3128 | 3067 | 2961 |

The earlier "≈ 0.1 ln" for an 11-bit codec counted it as a single-load effect; a byte cut applies to every load at
once, so it is worth ≈ 0.06 ln per load per 8% of bytes. Weights have 10.46 bits of entropy (exponent ≈ 2.46 bits vs
r6's 4-bit field). Main line now: pw17 = fixed 11-bit + escape lists inside r6's gv (analyst); in parallel a
variable-length exponent codec study targeting ≈ 10.6–11.0 bits at ≤ 6 lane-instr/weight (implementer). The decode
budget: 11 weights/clk/SM needed at m=1; the pw3/pw14 ladders show decode is hidden at m ≤ 8 today.

## Iteration 15 — pw17: codec E (pair-Huffman exponent, 11.07 bits) in r6's gv. Check `9f6de54218a447d5`: 9/9, 3389.6 µs. REFINE.

Correctness: the codec and its kernel are bit-exact on the judge (first run on hardware). Speed: every E path lost by
8–12 µs in-prepare — pooler m=1 r6 18.30 | ge-p512r (replicated LUT) 26.24 | ge-p512 29.44 | ge-p512f 29.18 |
ge-p512d8 36.32; qkv m=1 r6 14.37 | ge-qr 26.34; unpooler m=1 r6 8.77 | ge-u16 18.75; m=16 pooler r6 18.37 | ge-p512r
28.99. The critic's steady-state estimate (decode 35% of the per-unit budget, hidden) was wrong by ~3x: the kernel
runs at 1.45–2.1x the stream time, i.e. it is bound by the serial per-lane decode chain (16 dependent LDS lookups per
unit, each gating the next bit position), not by issue slots or bandwidth. Byte saving realised: 7.8% (11.07 bits with
padding), worth ~1 µs per 50 MB shape if hidden.
Decision: refine → pw18: (a) canonical Huffman so the code length comes from the leading bits by ALU (clz/lop, ~4 ops,
no memory access on the chain) while the exponent LUT lookup moves off the chain; (b) two independent sub-streams per
lane (ILP 2) as a variant; (c) same geometry set. Honest yield if hidden: ~0.5 µs net per m ≤ 16 big shape → ~0.12–0.15
ln → ~3330–3350; the codec lever as built is an ~3% lever, not the 8–12% the bits suggest, because E cannot serve
m=32 or o and the 512-thread geometry costs 0.2 µs.

## Iteration 16 — pw18: codec E v3 (ALU lengths, conflict-free ring, 1/2/4 sub-streams). Check `b15f30a832ab4169`: 9/9, 3390.6 µs. REJECTED; codec lever CLOSED.

In-prepare: pooler m=1 r6 18.02 | ge-p512 27.81 | ge-p512r 29.09 | -k2 29.92 | -k4 29.47 (pw17: 26.24); qkv m=1 r6
14.21 | ge-q 25.18 | -k4 25.92; unpooler m=1 r6 8.48 | ge-u16 19.46; m=16 pooler r6 18.43 | ge-p512r 29.18.
Removing the LDS from the chain, making the ring conflict-free and adding ILP changed nothing (±1 µs): two
latency models in a row (chained LDS; bank conflicts) failed to predict. The constant is the instruction count:
256–286 warp-instructions per 1024-weight unit vs r6's 49. Pooler m=1 = 32.8k units → 8.8 M warp-instr → 67k per SM
→ ≥ 9.3 µs at a perfect 4 IPC, realistically 15–18 µs: the kernel is issue-bound at ~2x the 13 µs stream, exactly as
measured. A variable-length decode needs ≥ 5–6 lane-instr/weight (peek, length, shift, value, plane merge, select);
hiding under the stream alongside r6's own 1.5/weight needs ≤ ~2. No scheme considered (pair/quad symbols,
bit-planes with compaction) gets there. The codec's realised payoff was 7.8% of bytes for 10 µs of issue.
Closed. Together with Iterations 10–14 this means r6's 12-bit fixed-width codec is at the decode sweet spot for this
kernel structure on this judge.

## Where this leaves the goal (2026-10-26, after 18 checks)

Board: r6 at 3428 (drift-measured 3365–3391 today). Every lever measured: m ≤ 8 at the stream; m=16/32 excess is x
loads + decode with no cheap fix found; m ≥ 128 five designs behind cuBLAS; codec at the decode budget; router
latency-bound (~50 score-µs, unbuilt). The 3000.75 on the board therefore comes from something this loop has not
found: most likely an m=128/256 kernel that beats cuBLAS (the transform-warpgroup design in audits/pw10-ladder.md is
the one untested structure) combined with a cheaper-than-expected decode, or a judge-side effect we have not
identified. Recommendation: stop spending checks on refinements; any further attempt should be a from-scratch
m ≥ 128 kernel program with its own multi-day budget, or a study of the leader's likely approach before building.

## pw19 (user request): transform-warpgroup m ≥ 128 kernel

Build opened at the user's request: decode in a dedicated warpgroup writing bf16 W tiles into a swizzled smem ring,
MMA warpgroup doing only wait → wgmma SS (both operands from smem) → release, producer warp on separate packed-W and x
rings; BN=64 (and BN=128 variant), cluster-2 DSMEM split-K, 1 vs 2 CTAs/SM, probe ladder. Per-stage budget at
1 CTA/SM: ~350 cycles (6 KB packed W at 31 GB/s per SM) against 245 tensor cycles, ~170 decode warp-instr, ~300
cycles of smem traffic — tight; 2 CTAs/SM or BN=128 is where a 3–4 µs win would have to come from. Expected: 27–29
in-prepare vs cuBLAS 29.4 at 1 CTA/SM. Kill: pooler m=128 not ≥ 1.0 µs under torch. Payoff if it wins by 3 µs on the
four big layers at m=128 and m=256: ≈ 0.2 ln (~3320 alone); necessary but not sufficient for 3000.

## Iteration 17 — pw19: transform-warpgroup m ≥ 128 kernel (user request). Check `09e3c6f672a24c71`: 9/9, 4430.5 µs. REJECTED.

Pooler m=128 in-prepare: torch 29.54 | pw10 wk2 31.30 | tw64 43.74 | tw64x2 (2 CTAs/SM, confirmed by the occupancy
log, cluster 4) 55.01 | tw128 (cluster 4) 60.61 | tw128c2 (64 CTAs) 55.07; o: torch 11.4 | wk2 16.7 | tw64x2 18.7;
unpooler: torch 23.6 | wk2 30.7 | tw64x2 35.1; qkv: torch 21.1 | wk2 30.1 | tw64x2 32.7; m=256 tv64 54 vs vk2 41.
Ladder (tw64 vs pw10 w1d4): W-only 19.9 vs 20.9 | W+x **30.4 vs 25.4** | +decode 32.5 vs 24.6 | +wgmma **39.5 vs 30.1**
| full 43.8 vs 34.6. The decode is now hidden (+2 µs, in separate warps) — the structural claim from the closing note
held for the decode — but two other costs grew: x delivery lost 5 µs (3–4 x slots of 16 KB = fewer bytes in flight
than pw10's 8-stage single ring; aggregate L2→SM 5.9 TB/s vs cuBLAS's 10.6 with a 13-stage pipeline) and SS wgmma
added +7 µs (≈ 60 KB of smem traffic per stage with the bf16 A tile written and read back, against a 420-cycle stage).
Cluster-4 and 64-CTA variants were far worse again.
Conclusion: on this judge the m ≥ 128 problem is L2→SM delivery (bytes in flight, x re-read volume) and smem bandwidth,
not decode placement. Six designs; the best remains pw10's register-A wk2 at 31.3 vs cuBLAS 29.5. Closed. The only
untried lever consistent with the data is cuBLAS-like pipeline depth (≥ 12 stages of W+x in flight per SM) with
register-A wgmma and 64-token tiles, and the ring-depth trend in pw10 (4→8 stages: −1.9 µs) suggests ≤ 1–2 µs more.

## Iteration 18 — pw20: codec-E lite paths at 64 registers / 1024 threads per SM. Check `7edbd6544f034326`: 9/9, 3324.8 µs. REJECTED (= r6).

Every pick is r6. In-prepare: pooler m=1 r6 17.95 | ge-lp1k (128x1024) 27.81 | ge-lp16 (256x512) 27.74 — identical to
pw18's 1-CTA/SM paths (27.8), so doubling the warps per SM changed nothing: the codec decode is issue-bound, not
occupancy-bound; qkv m=1 r6 13.76 | ge-lq16 25.38; unpooler m=1 r6 8.45 | ge-lu8 14.46. Codec lever closed for good
(four checks: pw17, pw18, pw20 on hardware + the pw19 structure for decode placement).
Note on the score: 3324.8 with r6's kernels is 3% under r6's board 3428 and the CI (1.431–1.475) excludes r6's 1.409:
the judge drifts by that much between runs (SGLang baselines moved too: decode16 76.5 vs 74.9). The board number is
one draw; a scored re-submission of the same kernel would land anywhere in ~3325–3430. That is noise, not progress,
and is left to the user as a choice.

## pw21 opened: router at m=128/256 on r6's gv (+ L2-scratch probe)

Remaining pool with judge arithmetic: verify16/32 router at 10.3/11.1 (cuBLASLt with 4–8 CTAs on K=8192 plus a
separate add kernel) vs a 2.2–2.7 floor; leverage 7.9/6.6 → ≈ 90–100 score-µs if a dedicated path reaches ~4 µs.
N=256 means the x re-read factor is only 256/rows, so the big-layer L2 problem does not apply. Build: r6 base +
router paths (12-bit hi+lo, K split across many CTAs, token tiles looped in-CTA, reduction folded into the launch).
Probe only: decode-to-L2-scratch + chunked cuBLAS pipeline on pooler m=128. Rejected as rule violations: persisting-L2
carve-out, persistent kernels holding weights, RNG regeneration.

## Iteration 19 — pw21: router m=128/256 on r6's gv + L2-scratch probe. Check `15cb5198043e436d` (run by the implementer at the user's request): 9/9, 1.432× (= r6). REFINE.

Router m=128 in-prepare: r6 (cuBLASLt) 14.72 | rt-c16 15.71 | rt-c8 15.90 | rt-c16s2 18.88 | rt-g16s2 26.82; m=256:
15.65 | cluster variants declined (occupancy) | c16s2 21.12 | g16s2 27.90. Latency-bound: 4 warps per SM, each chunk's x
waits a full L2 round trip; no better than cuBLAS. PROBE l2scratch: two-stream decode→L2 scratch + chunked cuBLAS
110.5 vs plain cuBLAS 23.8 (decode alone 45.4, chunked cuBLAS alone 75.7) — chunking makes cuBLAS 3x slower; closed.
Process note: the user is also driving the implementer directly; it ran this check itself and I submitted a duplicate
(`801c044fd5ad48d6`) before learning of it. Coordination rule added: job ids are posted to the head on submission.
Decision: refine → pw22, a standard split-K router for m=128/256 (4 output tiles x 32 K-splits = 128 CTAs, all x and
hi/lo loads issued up front, mma from smem, fp32 reduction by red.global into a zeroed output (memset node in the
graph) or last-arriver). Expected 4–5 µs vs 10.3/11.1 → ≈ 2–3% of score; kill at < 3 µs under torch.
Duplicate check `801c044fd5ad48d6` (same pw21 file + finally-guard): 9/9, 3374.3 µs, 1.4316× — reproduces the
implementer's run within 0.1%; all router picks r6; probe 110.5 again. Two same-file runs 20 minutes apart agree to
0.1%, so the ~3% drift seen earlier is between judge sessions, not between adjacent runs.

## Iteration 20 — pw22: split-K router m=128/256. Check `cdb3e04eda1c4d13`: 9/9, 3379.6 µs. REFINE (first within-run win).

Router m=128 in-prepare: r6 (cuBLASLt) 15.01 | **rs64k32 14.66 (picked)** | rs64k64 15.52 | rs128k64 16.83 | last-arriver
26.0/32.6; judge verify16 router 9.82 vs 10.58 (−0.76). Profile: rsk kernel 7.5 µs + the memset node 0.8 µs. m=256:
r6 15.94 | rs64k32 21.44 (two sequential token blocks) — lost. PROBE rszero 5.44 in-prepare (memset alone).
Reading: the kernel is latency-bound with a single wait on 128 KB per CTA at 1 CTA/SM (16 MB total: floor ≈ 2.5–3 µs);
7.5 µs leaves ~4 µs on the table at m=128 and the m=256 structure doubles the chain. Last-arriver loses because the
partials are 16 MB. Gain so far ≈ 9 score-µs (noise-level), but it is the first path to beat r6 inside a run.
Decision: refine → pw23: 2–3-stage pipelined loads per CTA (mma on stage s while s+1 lands), m=256 with both token
blocks in flight (or a 256-token tile), 2 CTAs/SM where registers allow, and a cheaper zeroing (self-cleaning fp32
workspace with the last CTA copying to out, vs the 0.8 µs memset node) baked against the memset. Target router
m=128 ≤ 6 and m=256 ≤ 7 judge-µs → ≈ 70 score-µs.

## Iteration 21 — pw23: pipelined split-K router. Check `9e45ff0e1e78459d`: 9/9, 3377.2 µs. PROMOTE → scored submission.

Router m=128 in-prepare: r6 15.10 | rs64k32 14.94 | **rp64k32s4 13.89 (picked)** | rp64k32s2 13.95 | rp64k64x2 15.81 |
rp64k32sc 15.71 (self-cleaning loses to the 0.9 µs memset node); judge verify16 router 8.84 vs 10.58 in the r6 run
(−1.7 µs, ≈ 13 score-µs). m=256: r6 15.62 | rv64k32 18.21 | rv64k64x2 19.58 | rv64k32sc 22.53 — r6 kept. Profile:
rsk2 7.5 µs + memset 0.9; floor for 16 MB ≈ 3, so ~4 µs remain at m=128 (suspects: scalar fp32 atomics burst at the
end of a one-wave grid, 4 MB of partials; first-stage latency), and m=256 needs the kernel under ~8 µs to beat cuBLAS.
pw23 is r6 plus a strictly better router at m=128 and is the first candidate that dominates r6 within a run → scored
submission (10 slots unused). Expected board: today's r6 drift band (3365–3390) minus ~13 µs.
Router post-mortem (audits/router.md): rsk2's 7.5 µs ≈ 1 first-data latency + 2.5 load (x re-read by the 4 tiles) +
0.5 uncovered last stage + ~1 of red.global.add.v2 partials; 16 B adds + cluster-2 pre-reduction ≈ −0.8 µs at m=128,
below the 1.5 µs bar; m=256 needs ≤ 9.5 kernel-µs against a measured 13.5 and is not winnable without a wgmma rewrite.
Router line closed after pw23's +1.7 µs at m=128.

## Scored submission of pw23: job `17f4484b97644673` — 9/9, **3399.3 µs**, 1.4211× (CI 1.400–1.442). New incumbent.

Board moved from 3428.0 (r6-c12) to 3399.3: −0.8%, of which ~13 µs is the router kernel at m=128 and the rest is the
draw. Per-load vs the pw23 check run: decode1 37.58 (37.33), decode16 50.52 (50.78), decode32 61.38 (60.70), verify1
38.55 (39.09), verify16 71.66 (70.52) — this scored draw sat ~0.6% worse than the check. Leaderboard at the time:
#1 1.637× (≈ 2951 µs), #2–#8 at 1.547–1.570× (≈ 3077–3122 µs); we are at 1.421×.

## Calibration against the judge's own speed-of-light (scored run 17f4484b97644673) and the board

All 30 shapes at the judge's SOL → 2376; only m ≥ 128 at SOL → 2875; the big four at m ≥ 128 at SOL → 3022; m ≥ 128
halfway to SOL → 3159. Largest gaps (judge µs over SOL): router m=256 8.5, qkv m=256 7.7, qkv m=128 7.5, router m=128
7.0, pooler/unpooler m=256 6.9/6.5, qkv m=32 6.4, pooler m=128 5.2. Board: #1 1.637× (≈ 2951), #2–#8 1.547–1.570×
(≈ 3077–3122), us #108 at 1.421×. Seven teams near 3100 is consistent with a competent packed-weight GEMM at
m=128/256 roughly halfway to SOL (pooler m=128 ≈ 17 judge-µs vs cuBLAS 20.3), i.e. exactly the kernel six of our
designs failed to build (best ≈ 27 judge-µs). Public write-ups found (ozgureyilmaz, lee101, PostTrainLLM) are all at
≤ 1.15× and say nothing about the top tier. Kernel track closes 2026-11-04 23:59 PT.

## Program to 2026-11-04 (audits/m128-study.md) — day 1 opened: pw24

Study conclusion: the binding limit at m ≥ 128 is L2→SM delivery (cuBLAS 10.7 TB/s on 262 MB; ours 7.0–8.7), split-K
cost 4.5–8.2 µs in every variant and must go. Recommended design (b): cuBLAS's geometry (64 rows x 64 tokens, full K,
no split-K, 128 CTAs) with cluster-2 W multicast over the token-tile pair, two producer lanes, ~168 KB in flight; the
seven teams near 1.55x need no codec at all if a bf16 GEMM streams at 4.0–4.2 TB/s (16–17 µs vs cuBLAS 20.3).
Plan: days 1–2 bf16 version with a ladder (kill if W+x > 20 in-prepare); days 3–4 add the 12-bit decode; days 5–6
x multicast if delivery still binds; days 7–9 harden + final scored run. (a) CUTLASS mixed-input (8-bit + 4-bit
planes) is the fallback if (b) stays below 9 TB/s. pw24 = day 1.

## Iteration 22 — pw24 (bf16 cuBLAS-geometry GEMM). Check `5c994bf40c144571`: FAILED at setup. REFINE.

ptxas: wgmma.* "not supported on .target 'sm_90'": the new bg kernel was compiled inside r6's gv extension, which uses
plain sm_90 flags (the pw lineage always built with compute_90a/sm_90a). Plus advisory warnings on
'.multicast::cluster' for cp.async.bulk.tensor. The whole run was lost (no fallback when an extension fails to build).
Fixes: bg in its own load_inline extension with sm_90a flags; prepare continues with r6 + router if any extra
extension fails to build; loop lint now flags wgmma/setmaxnreg sources without a compute_90a flag in the same build.
Process lesson: the local nvcc checks compiled the kernel block with -arch=sm_90a by hand, never with the flags the
submission actually passes to load_inline; the critic's build must use the submission's own flag list per extension.

## Iteration 23 — pw24 resubmitted (bf16 cuBLAS-geometry GEMM). Check `715ddc0765664971`: 9/9, 3392.7 µs (= pw23). REJECTED; day-1 kill rule hit.

Ladder on bg12c (pooler m=128, in-prepare): W-only 22.43 (67 MB → 3.6 TB/s effective, fine) | W+x 25.57 (> the 20
kill line) | +wgmma 36.45 (**+10.9**) | full 38.24. Bake pooler m=128: cuBLAS 24.19 | bg12 (no cluster) 28.90 | bg6
33.02 | bg6c 37.89 | bg12c 38.34 — the cluster-2 W multicast costs ~10 µs, 2 CTAs/SM costs 4. Same order on qkv
(19.4 vs 28.7), o (11.0 vs 11.7 bg6), unpooler; m=256 far worse (bg6 39 vs 27.8).
Reading: even with no decode at all, the hand-written pipeline at cuBLAS's geometry is 20% behind cuBLAS, and the
wgmma rung (+11 µs ≈ 150 cycles per 64-k stage) shows the SS wgmma is not overlapped with the stream — tensor time
per stage (4 x m64n64k16 ≈ 124 clk) should hide under the ~250-clk stage; suspects: smem bandwidth (TMA writes +
wgmma reads = 32 KB per stage ≈ 128 B/clk, the smem limit) or wgmma issue serialisation. Multicast is a net loss here.
Program status: design (b) failed its day-1 test. Options: (a) CUTLASS sm90 warp-specialised mainloop (proven to hide
wgmma; feasibility = header bundle size + compile time + the 12-bit transform), or stop at 3399. Seven checks since
the m ≥ 128 branch was "closed" have confirmed the same conclusion from a different direction.

## CUTLASS program opened (audits/m128-study.md, post-mortem + feasibility)

pw24 post-mortem: the SS m64n64k16 wgmma reads 4 KB per 32 clk, the whole 128 B/clk smem port, and the single MMA
warpgroup serialises fence→wgmma→commit→wait→release; the cluster pair runs at its slower CTA's pace (+9.4 µs) and
W-only is HBM-bound so multicast saves nothing. CUTLASS 3.9.2 feasibility: 458 headers, 13.8 MB raw → 0.90 MB xz+b64
embedded; one 128x64x64 cooperative warp-specialised instantiation compiles in 23.8 s locally (judge est. 35–45 s),
168 regs, 0 spills, expected SASS structure. Step A = pw25: bf16 CUTLASS path for the big four at m=128/256 (parity
test with cuBLAS + judge-side mechanics). Step B (only if A reaches parity): the 12-bit decode as the A-transform
(8-bit + 4-bit planes, escape fixup) — the actual gain. Track closes 2026-11-04.

## pw25 (CUTLASS bf16 path): upload rejected at 1,074,299 bytes (limit 1,000,000; lint rule added), shrunk to 816,800 with identical SASS, check `04dc0a41413f45f3` submitted.

Step-B plan (audits/cutlass-mixed-input.md): stock mixed-input collective compiles at 168 regs / 0 spills on uint8 and
uint4 A planes (36.8 s for both); route (i) = forked collective with two TMA streams (8-bit low byte + 4-bit sign/
exponent-offset plane) merged in the convert step with r6's dec1 arithmetic, escapes (0.01%) by a 1–1.5 µs post-GEMM
fixup kernel; expected pooler m=128 ≈ 16.5 judge-µs vs cuBLAS 20.3; ~5 days; gated on pw25 reaching cuBLAS parity.

## Iteration 24 — pw25: CUTLASS bf16 path. Check `04dc0a41413f45f3`: 9/9, 3377.7 µs (= pw23). REJECTED; kill rule hit.

Mechanics all worked: bundle extracted in the sandbox, `PREP cut compiled in 52s`, `PREP done in 170s`, 9/9.
Speed: pooler m=128 in-prepare cuBLAS 23.81 | cutsk (stream-K) 37.92 | cut128 45.98 | cut128c2 47.84 (probe: 2.03
TB/s on W bytes); unpooler cut128 28.10 vs 22.43; o cut128c2 13.79 vs 10.75; qkv cutsk 31.14 vs 18.78; m=256 46–56 vs
27.7. The stock 128x64x64 cooperative warp-specialised kernel is a poor config for N=128 tokens (64 tiles on 132
SMs; stream-K helps but is still +14 µs); cuBLAS's nvjet picks 64x64x64 with 13 stages, 1x2 cluster, coopB.
Program status: step A (bf16 parity) failed on the first config. A CUTLASS tile/stage sweep could close part of the
gap (1–2 checks, compile-time bound: 52 s for 3 instantiations), but even at parity the mixed-input fork is ~5 days
with the deadline 9 days out, and no pipeline of ours — hand-written or CUTLASS — has yet reached cuBLAS on this
shape. 27 checks in total; the board stands at 3399.3 (pw23).
CUTLASS sweep plan (audits/cutlass-mixed-input.md): the 128x64 tile count (64 on 132 SMs) explains cut128; a sweep
of 64x64x64 ping-pong (13 fixed stages), + 1x2 cluster W multicast (nvjet's shape), single-warpgroup 2 CTAs/SM,
64x64x128, and a non-swapped control, expected 24–29 in-prepare vs cuBLAS 23.8; four instantiations per check fit
the budget (~220 s). Implementer's odds: bf16 within 10% of cuBLAS ≈ 50% (m=128) / 40% (m=256); mixed-input fork
beating cuBLAS by ≥ 3 µs by 2026-11-04 ≈ 20–25%; overall ≈ 10–12%. Decision left to the user; loop paused.
pw26 ready (approved): four 64x64 basic warp-specialised configs (ping-pong spills in 3.9.2). Critic finding that
also indicts pw25: without -DNDEBUG, CUTLASS's device asserts stay in the main loop and ptxas reports the wgmma
pipeline serialised in every instantiation — part of pw25's 2x loss may be that, not the config. Fixed in pw26.
Not submitted: awaiting the user's decision; quota resets 00:00 UTC.

## Drawing board v2 (user: "3399 is still not good enough")

Ranking every shape by (gap to the judge's SOL) x leverage on the scored table, top 16 (score-µs): verify16 qkv
59, decode32 qkv 59, verify32 router 56, verify16 router 55, verify32 qkv 51, verify1 router 46, verify32 pooler 45,
verify32 unpooler 43, verify1 unpooler 42, verify16 pooler 41, decode32 router 41, decode1 unpooler 41, decode1
router 39, decode16 router 39, verify16 unpooler 36, verify32 o 34. Two pools we never attacked:
- **Small shapes** (router m ≤ 32, unpooler/o m ≤ 8): 6–12 MB reads at 4.4–6.3 judge-µs vs SOL 1.75–1.86, at 15x
  leverage ≈ 150–200 score-µs. Our "3.4 µs floor" was pw2's bulk probe, never a latency-optimised design. Track P1.
- **qkv at m ≤ 32**: 144 row tiles on 132 SMs → a 12-tile second wave; balancing without a general split-K
  (stream-K-style straddled tiles with a tiny last-arriver) ≈ 1.3–2 µs per shape ≈ 77 score-µs. Track P2.
- Router m=128/256 wgmma rewrite ≈ 90 score-µs (P3, after P1). CUTLASS sweep pw26 (ready) at the reset; mixed-input
  fork if parity (P4).
Sum of P1–P3 at realistic capture ≈ 250–300 score-µs → ~3100–3150 without beating cuBLAS at m ≥ 128.
pw27 (track P1) ready and approved: PROBE rd8 read-floor ladder (8 MB; 128–1056 CTAs; 16 B loads vs bulk chunks;
+ fp32 atomics / + cluster-8 reduce / memset alone) and atomic-reduce gv paths (router m ≤ 32 fp32 reds, unpooler/o
m ≤ 8 packed bf16x2 reds; no cluster launch). Accuracy simulated 3.2e-3 relative L2 at a 4-way bf16 split vs the 1%
rule; one-hot exact. Queue order at the 00:00 UTC reset: pw26 (CUTLASS sweep), pw27 (P1), then pw28 (P2) and pw29
(P3) as they clear the critic.
Reset queue (all approved, waiters chained): pw26 (CUTLASS 64x64 sweep, NDEBUG) → pw27 (P1 small-shape floor ladder
+ atomic-reduce gv paths) → pw29 (P3 router wgmma split-K) → pw28 (P2 qkv stream-K balancing, sentinel handoff).
Each is measured against the pw25 run (04dc0a41413f45f3, r6 + router = the current incumbent's kernels).

## 2026-10-26 15:20 UTC — H100 offered; ultracode workflows begin

The user provides an H100 over ssh (root@162.243.109.198; key authorisation pending) and asks to keep it busy.
Consequence: local measurement replaces the 20-checks-a-day bottleneck (H100: 3.35 TB/s HBM vs the H200 judge's
4.8; same 132 SMs and 50 MB L2, so stream-bound times scale by ~0.7x and structural results transfer). Rules for its
use: timing runs are serialised through /root/gpu_lock.sh (flock); the task's own harness muse_bf16_proj/checks.py
calibrates each candidate against the judge; the judge remains the only board-relevant number.
Workflow wf_d35c8afb-f3d (ultracode): Understand (DeepGEMM sm90 bf16 kernel + host/JIT/embeddability; our
pipelines' ranked structural defects; 12-bit decode insertion points) → Design (three independent designs, three
judges, synthesis into audits/m128-plan.md with an H100 experiment order). Reset queue unchanged (pw26, pw27, pw29,
pw28 at 00:00 UTC).

## H100 box online (root@107.170.6.60): first calibration, pw23

Container torch 2.8.0+cu128; local harness proj_rsi/tools/h100_bench.py. pw23 passes every official correctness
case (m = 1, 5, 7, 8, 9, 33, 130, 512, 1000 on all five layers). Graph-replay times on the box (µs, overhead
included): pooler m=1 28.0 (judge 13.95), unpooler m=1 13.3 (judge 4.44), router m=1 12.6 (judge 4.35), pooler
m=128 39.9 vs torch 40.5 (judge 19.9 vs 20.3), qkv m=128 30.7 vs torch 31.0, verify32 pooler 44.6 vs 44.7. So:
(1) the box's fixed replay cost is ~9 µs vs the judge's ~4 (now measured by a REPLAY_OVERHEAD probe in the
harness); (2) big-shape kernel-side times scale ~0.7x to the judge as the HBM ratio predicts; (3) within-box
comparisons (candidate vs r6 vs torch) are the evidence, absolute numbers are not. Queue on the box: pw26, pw27,
pw29, pw28, pw24, pw25, pw10 (serialised by /root/gpu_lock.sh).
Agents on the H100 (2026-10-26 ~16:30 UTC): CUTLASS bf16 config sweep at our eight (layer, m) shapes (parity hunt);
P1 small-shape iteration (pw27h); P2 qkv balancing iteration (pw28h); P3 router m=128/256 iteration (pw29h). Each
edits a local copy here, rsyncs to the box, and measures under the GPU lock; judge candidates stay untouched until a
measured win is ported. The understand/design workflow for the m>=128 packed kernel runs in parallel.
H100 local result, pw26 (CUTLASS 64x64 sweep, in-prepare bake on the box, µs): pooler m=128 cuBLAS 27.65 | cw7c2
31.97 | cw7 32.10 | cwk128 34.21 | cw13 34.46; qkv m=128 19.46 | cw13 23.97 | cw7 26.37; unpooler m=128 27.07 |
cw7c2 28.48; o m=128 11.23 | cw7c2 11.01 (picked); m=256 all 40–250% behind. So with -DNDEBUG and 64x64 tiles the
stock CUTLASS warp-specialised kernel is 16–23% behind cuBLAS at m=128 on this box and far behind at m=256; the
cluster-2 (x multicast) config is the best of the four at pooler/unpooler. The wide sweep agent continues from here.
pw23's and pw26's harness tables agree within noise (pooler m=128 39.5–39.9 vs torch 40.5), confirming the box's
replay overhead (~9 µs) and repeatability. First harness runs of pw27/pw28/pw29/pw24 failed on a probe bug (fixed,
re-queued).
H100 local results (in-prepare µs on the box; empty graph 3.46): pw27 — 8 MB read floor 6.72 (bulk) / 6.78 (264+
CTAs x 16 B), +fp32 atomics +1.5, +cluster-8 +1.6; r6 router 7.26/8.74/8.83/9.50 at m=1/8/16/32 (m=1 is 0.5 over
the pure read), rk atomic variants picked at m=8/16/32 by 0.3–0.6 µs, unpooler/o r6 unbeaten (7.4/8.8 vs 8.0–10.7).
pw28 — balancing ties r6 at m ≤ 16 (delta 0.00) and loses at m=32 (+1.7): the 144-tile tail hypothesis is refuted on
hardware; P2 now runs a composition ladder. pw29 — only pw23's rp paths appear (rwk variants absent: investigating);
m=256 cuBLASLt 14.88 beats rv64k32 18.08. The harness's REPLAY_OVERHEAD (11 µs, idle-GPU) is not the in-prepare
replay cost (3.46): small-shape comparisons use in-prepare BAKE/PROBE numbers only.
Box pitfall found and fixed: torch 2.8 in the container appends -gencode compute_90/sm_90 to every load_inline build
(TORCH_CUDA_ARCH_LIST unset), which breaks hand-written wgmma extensions (pw24's bg, pw29's rwk "build failed" on the
box; CUTLASS survives via its __CUDA_ARCH_FEAT_SM90_ALL guards). /root/okc_run.sh now sets TORCH_CUDA_ARCH_LIST=9.0a;
pw24 and pw29 re-queued. Candidates should guard wgmma/setmaxnreg/multicast code with __CUDA_ARCH_FEAT_SM90_ALL so a
stray sm_90 pass compiles to a stub instead of failing the extension.
H100 local results after the arch fix (in-prepare µs on the box): pw29 — router m=128 cuBLASLt 16.10 | incumbent
rp64k32s4 13.60 | **rw64k32s8 12.38** (wgmma split-K, −1.2 vs incumbent), m=256 cuBLASLt 14.75 | best rwv64k64s4x2
17.34 (loses); harness verify16 router 16.40 vs torch 24.26. pw24 — ladder pooler m=128: W-only 27.33 | W+x 28.22 |
+wgmma 36.29 | full 37.44; bake cuBLAS 27.46 | bg12 29.02 | bg6 34.78 | bg12c 37.47. The W-only stream equals
cuBLAS's whole GEMM on this box (2.88 TB/s(−4) of 3.35 peak), x tiles cost only +0.9 here (vs +3.2 on the judge), and
the wgmma rung costs +8.1 on both machines: the overlap failure is reproducible locally. Track P4 opened: an agent
runs a variant ladder on pw24h (groups in flight, commit granularity, ping-pong / 128x64 two-warpgroup tiles,
register-A, release placement, smem-port test) to find and fix the mechanism.

## H100: CUTLASS bf16 sweep (233 configs) — parity reached on the box

audits/cutlass_sweep.md: within 1.5% of the best torch path on 6 of 8 shapes (ratios 0.945–1.036 incl. ~13 µs of
replay overhead on both sides): qkv m=128 0.979 (ws_64x64x64_c1x2_sA_nosmem), o m=128 0.993 (…c1x2_s7_tma), pooler
m=128 0.984 (…c1x2_sA_tma), pooler m=256 0.945 (non-swapped ns_ws_64x128x64_c1x1_sA_tma), o/unpooler m=256 1.01
(ws_128x128x64_c1x2_sA_tma), unpooler m=128 1.032, qkv m=256 1.036 (ns 64x96). Basic warp-specialised beats
ping-pong/cooperative; a 1x2 cluster helps in swap-AB; stream-K never wins; ping-pong no longer spills once the
epilogue has no C input. Compile 17–31 s per instantiation. Consequences: (1) pw30 = the four parity configs as a
judge candidate (confirms transfer to the H200; little score by itself except pooler m=256 −5.5%); (2) the gain must
come from the 12-bit two-plane A operand inside this pipeline — agent 'mx' is building it (M1 packer/merge, M2
two-plane collective, M3 timing, M4 judge-shaped candidate).
P3 result (audits/p3-h100.md; box in-prepare µs, read-flushed): router m=128 **rw64k32s8z 11.74** | rw64k32s8
12.61 | incumbent rp64k32s4 13.28 | cuBLASLt 16.19; m=256 cuBLASLt 15.01 | rw64k32r3x2z 15.14 (tie, 2 CTAs/SM).
Wins: dropping the memset node (each K-split CTA zeroes its slice of the output and bumps a per-tile counter checked
before the final adds; −1.4 µs, mostly the graph node); 2 CTAs/SM at m=256. Losers: cluster-2 pre-reduction (+1.4),
red reordering (0), 512-wide K slices (24+), 256-token CTAs. H200 estimate −1.0 to −1.5 µs at m=128. Harness lesson:
a write flush (zero_) leaves L2 dirty and changes m=256 rankings; h100_bench now read-flushes like the judge.
pw29h proposed; critic review next; it replaces pw29 in the reset queue if approved.
pw30 (four parity configs as a judge candidate) built; on the box r6 won all eight shapes: the sweep's "parity" was
against untuned torch with a write flush, while r6 runs a per-shape cuBLASLt algorithm search and the bake read-
flushes — against that, the best CUTLASS config is 0.3–2.3 µs behind (pooler m=128 c64n 28.26 vs r6 27.42, +3%).
So the bf16 CUTLASS pipeline is ~1–8% behind r6's tuned cuBLASLt on the box; the two-plane build must win on bytes
(−25%) despite that. pw30 goes last in the reset chain as a transfer test only.

## Workflow wf_d35c8afb-f3d complete: understanding + design for m ≥ 128 (audits/m128-plan.md)

Understand: DeepGEMM's sm90 bf16 kernel keeps ONE wgmma group in flight (wait_group 0 per stage) and hides latency
with ring depth and merged BLOCK_K=128 stages (8–12 wgmmas per sync round); clusters disabled on one-wave grids; TMA
store epilogue; it routes bf16 to cuBLASLt by default on a full GPU. Pipeline diff: our wait/release protocol equals
CUTLASS's; the difference is a single math warpgroup whose per-batch bookkeeping rivals the batch's tensor time at
64x64; a 64x64 bf16 SS pipeline is capped at cuBLAS's level by the smem port (32 KB/stage ≈ 256 clk) and the L2
model — bf16 SS can never beat cuBLAS; headroom exists only with register-A + 12-bit bytes + fewer x bytes. pw24's
"+10.9 wgmma rung" was measured on the cluster variant that alone costs +9.4; bg12 (no cluster) is 72 clk/stage
behind cuBLAS. Decode insertion ranking: two math warpgroups with register-A (−4.5 to −5.5 µs vs bf16-at-parity) ≈
register-A with interleaved decode and k128 stages ≈ two-plane TMA+ldmatrix ≫ transform warpgroup → SS (+0 to +2) >
consumer decode → stmatrix → SS (+2 to +4). DeepGEMM embedding: +29 KB over pw25's bundle, 1.6 s/instantiation.
Design (RS128): swap-AB, 128 W rows x all tokens per CTA, two math warpgroups, register-A from the 12-bit pack decoded
in the math warps, x as smem B, K split on a one-wave grid (pooler 32x4), producer WG with separate W and x lanes,
setmaxnreg 40/232, cross-CTA reduce by cp.reduce.async.bulk.tensor bf16 add, no cluster; H200 estimate pooler m=128
≈ 17.0 (16.5–17.8) vs cuBLAS 20.3; m=256 parity to −1. An agent now builds it on the H100 per the plan's ladder
S0–S5 (milestones R1–R4), in parallel with the CUTLASS two-plane route (mx). P4 re-pointed at the no-cluster ladder.
mx M2 (box, pooler m=128, read flush incl. 3.6 µs replay): torch 28.32 | bf16 CUTLASS 64x64 c1x2 27.97 | mx two-plane
e0 27.52 — output bit-identical to x @ W_dec^T, 74 regs, 0 spills; only −0.45 µs so far because the kernel is not
stream-bound (HBM ≈ 2.1 TB/s): x L2→SM re-reads (128 MB at 64-row tiles) bind, as the workflow predicted. c2x2 49,
128x64 single-WG 38 (64 CTAs), escape prologue 60 µs (serialised gathers; to be replaced by an epilogue correction).
Next for mx: A-only/B-only probes, then the 128-row cooperative tile with split-K to 128 CTAs and a TMA reduce-add
epilogue (= RS128's geometry).
P1 result (audits/p1-h100.md; box in-prepare, alternated with r6): router m=1 rr6 6.91 vs 7.20 (−0.3); m=8
rk-s16w8z 7.5 vs 8.7 (−1.15); m=16 rk-s8z 7.7–7.9 vs 8.8–9.0 (−1.1); m=32 rk-s8z 7.9–8.1 vs 9.2–9.4 (−1.3);
unpooler m=8 uk-s4z 7.0 vs 7.3–7.6 (−0.3 to −0.5); o stays r6. Mechanism: the reduction's whole cost is zeroing the
output — memset node +1.5, cluster reduce +1.8, PDL zero kernel +1.2; "flag zeroing" (the tile's first split-K CTA
zeroes and raises a flag, others acquire-wait once) costs ~0.8 with no memset node and no cluster. Read floor: 8 MB
in 6.7 (3.3 GPU-side) at ≥ 2 CTAs/SM. A no-reduction whole-row router (rr2) lost (7.33 vs 5.18 GPU-side). Expected
judge gain ≈ −0.3/−0.5/−1.3 µs at router m=8/16/32 (latency, not bandwidth) ≈ 20–25 score-µs. pw27h proposed;
critic review next (flag ordering/forward progress). Box fix: bench.sh now holds the lock for the whole run again —
prepare() times while it compiles, so an unlocked warm step perturbed other agents' timings.

## P4 result: the wgmma-overlap mechanism found and fixed (audits/p4-h100.md)

With clock64 stamps on the box: without the cluster the +wgmma rung is +4.2 µs (W+x 27.36 → 31.49), and it is the
single consumer warpgroup's SERIAL per-stage bookkeeping (two barrier waits + wake-ups, fence, 4 wgmma, commit, wait,
2–3 release arrives ≈ 230 clk) against a ~371 clk stream period per 64-k stage — it adds nearly 1:1 to the period;
on the H200 the period is 0.7x, hence +11 there. Smem traffic is not it (full stream + an independent wgmma loop costs
nothing); a 240-clk spin per stage costs +6.5. Not helping: wait_group depth 1/2/3, commit per k16, early release,
polling. Fix: fewer sync rounds per byte — 128-k slots (7 x 32 KB), one full barrier per slot shared by both producer
lanes, one empty barrier, blocking try_wait, one wgmma.fence per slot, wait_group 1 then release the previous slot, no
cluster (+0.9, saves nothing). Result (bf16, box in-prepare): pooler m=128 k2x 28.58 vs cuBLAS 28.90 (ladder 25.60 |
26.78 | 27.84 | 28.58); qkv m=128 20.67 vs 20.54; unpooler m=128 28.70 vs 28.26 and o 11.10 vs 11.36 (64x128
tiles); pooler m=256 31.74 vs 34.11 (−2.4). Pitfalls: __float2bfloat16 on accumulators serialises register-A wgmma
(use cvt.rn.bf16x2.f32 inline); the serial consumer keeps only ~110 clk of slack per 64-k stage on the H200 for a
decode. Forwarded to RS128 and mx. pw24h proposed as a judge candidate (transfer test; pooler m=256 win).
pw27h approved (critic: ordering correct, 39-case 3000-replay stress clean; fixes: rk-s16w8x dropped at m=32 — 272
CTAs exceed 264 resident slots — and the flag wait bounded with trap). Reset chain now: pw26 → pw27h → pw29h → pw28 →
pw30 (pw24h to follow its review). pw31 = pw27h + pw29h's router section is being merged as the scored candidate.
pw31 (= pw27h + pw29h over pw23) built: 65/65 official checks on the box, every merged path wins its own bake
(router m=128 rw64k32s8z 11.55 vs 13.63; router m=8/16/32 −1.0/−1.0/−1.2; unpooler m=8 −0.26; m=256 tie); critic
diff-review running, then a SCORED submission (scored quota is separate from checks). A soak job now keeps the GPU
loaded between agent bursts (proj_rsi/tools/soak.py: pw31 vs pw23 alternated in short locked bursts; running table in
/root/okc/logs/soak_pw31_vs_pw23.md).
pw32 (k2x/n128x ported onto pw23): on the box the bake picks nothing — r6's tuned cuBLASLt beats the bf16 128-k-slot
pipeline by 0.1–2.5 µs (pooler m=128 27.10 vs 28.38; pooler m=256 29.70 vs 32.16); P4's "parity" was vs torch.mm,
which the tuned path beats by 1.6–4 µs. Not submitted. Correct targets for the 12-bit kernels (box in-prepare, r6):
pooler m=128 27.10, qkv 19.49, unpooler 27.04, o 11.04, pooler m=256 29.70.

## Scored submission of pw31 (= pw23 + pw27h small shapes + pw29h router): job `bbb732a78955425b` (16:42 UTC)

Critic approved the merge (diff3 byte-exact, both fixes present, 65/65 box checks, picks as expected). Expected judge
effect from the box deltas (in-prepare): router m=128 −2.1, router m=8/16/32 −1.0/−1.0/−1.2, unpooler m=8 −0.3 →
roughly 35–45 score-µs (≈ 1–1.3%) plus the draw. Reset chain unchanged (its pw27h/pw29h checks become confirmatory).
P4 asked for a decode-slack curve in the k2x pipeline (instructions per thread per stage a single consumer absorbs).
