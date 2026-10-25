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
