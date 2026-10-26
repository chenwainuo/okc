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
