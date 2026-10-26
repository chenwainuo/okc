# Where the lossless 12-bit decode fits in a DeepGEMM-style warp-specialised pipeline

Analysis only (2026-10-26). Sources: `LOOP.md` (it. 7-8, 11, 13, 17, 22-24, drawing board v2), `audits/pw10-ladder.md`,
`audits/m128-study.md`, `audits/cutlass-mixed-input.md`, `audits/wgmma-layout.md`, `audits/pw19-critic.md`,
`audits/codec-study.md`; kernels `candidates/pw10/submission.py` (pwg, register-A) and `candidates/pw19/submission.py`
(pwt, transform warpgroup); DeepGEMM `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`,
`csrc/jit_kernels/heuristics/sm90.hpp`, `deep_gemm/ptx/wgmma.cuh`, `deep_gemm/mma/sm90.cuh`. No kernel was run.

## 0. Shared model (pooler m=128, swap-AB: wgmma M = 64 W rows, N = tokens, K = 16 per instruction)

| Quantity | Value | Source |
|---|---|---|
| Bytes: bf16 W / 12-bit W / x | 67.1 MB / 50.3 MB / 2 MB | N=4096, K=8192 |
| HBM floor H100 (3.35 TB/s peak, ~3.0 effective) | bf16 22.4 us, 12-bit 16.8 us, +1.5 ramp/tail | pw2/pw9 probes reach 93-95% of peak on H200 |
| HBM floor H200 (4.2 TB/s bulk measured) | bf16 16.0, 12-bit 12.0 | LOOP "Note for pw12" |
| Tensor time (8.6 GFLOP, 4096 dense bf16 FLOP/clk/SM, 1.98 GHz) | 8.0 us (= 49% of the H100 floor, 67% of the H200 floor) | m64n64k16 = 32 clk, m64n128k16 = 64 clk |
| Geometry taken as the parity skeleton | 64 W rows x 64 tokens, full K, 128 CTAs at 1/SM, no split-K (nvjet's shape) | m128-study (b), pw24/pw25 |
| Stage period at the 12-bit floor, k64 stage | 260 clk (H100) / 186 clk (H200) | 128 stages |
| Tensor per k64 stage at 64x64 | 128 clk | 4 x m64n64k16 |
| smem port | 128 B/clk/SM | |
| L2->SM traffic at 64x64 | x 128 MB + W 2 reads: 96 MB (12-bit) / 134 MB (bf16) = 224 / 262 MB | each W chunk is read by the 2 token-tile CTAs |
| Decode (r6 dec1, `pw_dec` pw10:190) | 9 instr per 8 weights = 1.125 lane-instr/weight; 3 x LDS.128 per 1024-weight tile per warp | SASS counts, codec-study |
| Escapes (16-value exponent window) | 0.01% = ~3.4 K on pooler = ~52 per CTA (64 rows) = ~13 per warp over 128 stages = 0.1 hits per warp-stage | LOOP it. 7 STATS; `_c12` pw10:1164 |

**The gating item is not the decode.** With 224 MB of L2->SM traffic, the skeleton needs 13.3 TB/s to sit at the H100
12-bit floor; our skeletons measured 7.0-8.7 TB/s (pw10 W+x: 178 MB in 20.4 kernel-us) and nvjet 10.7 (or ~7.9 if its
1x2 cluster multicasts W). Every insertion point below inherits this. The two ways to get under it are x multicast over
an n-pair cluster (x 128 -> 64 MB, total 160 MB = 18.4 us at 8.7 TB/s, = the floor) or 128x128 tiles with a 4-way
split-K (114 MB, but a 64 KB fp32 reduction per CTA). The costs in section 7 are conditional on the bf16 skeleton
reaching parity; the decode placement is second-order until the W+x rung of the skeleton is at <= 18 us on the H100.

### What DeepGEMM's sm90 bf16 kernel does that pw10/pw19 did not

- One TMA warpgroup (128 threads, `warpgroup_reg_dealloc<48>` cuh:127,149), one elected thread issues; one or two math
  warpgroups at 248/224 registers (cuh:128,212). pw10 used setmaxnreg only at NWG=2; at NWG=1 the consumer sat at the
  launch-bounds default.
- Stage = A + B in one slot, full barrier `init(1)` + `arrive_and_expect_tx`, empty barrier `init(math warps)`
  (cuh:116), arrive by lane 0 of each math warp after `warpgroup_wait<0>()` (cuh:278-281). Same release rule as pw10
  (slot held until the wgmma retires), so DeepGEMM pays the same "slot pinned by the MMA" cost pw10-ladder (iii) found;
  it compensates with depth (up to 16 stages, sm90.hpp:149) and with **stage merging**: with >= 10 stages and one math
  warpgroup it multiplies BLOCK_K by stages/5 (cuh:46-56, "for reducing the warpgroup_wait<0>() overhead"). That is the
  knob none of our kernels had: k128 or k192 per barrier round trip, so the fixed per-stage sync cost (try_wait,
  fence, commit, wait, arrive: ~60-100 clk of latency) is amortised over 256-384 tensor clk instead of 128.
- With one math warpgroup there is no wgmma overlap across stages at all (wait<0>); with two (BLOCK_M=128) the
  warpgroups interleave on alternate halves of the same stage. Neither is a ping-pong over stages.
- Heuristics worth copying: swizzle >= 64 B (sm90.hpp:101), >= 3-4 stages (107), multicast disabled for one-wave grids
  (234, i.e. for exactly our 128-CTA case: consistent with pw24's +9.4 us for W multicast).
- Its tile picker has no split-K and no swap-AB; for M=4096 W rows, N=128 tokens it would choose BLOCK_M 64 x BLOCK_N
  128 = 64 CTAs, the pw25 failure. BLOCK_N = 64 tokens (128 CTAs) must be imposed.
- Epilogue: STSM into a swizzled smem D tile, then one TMA store per 64-column block (cuh:300-370). For swap-AB the
  stmatrix must be `.trans` so that the smem tile is [token][n] and out rows are contiguous; pw15 showed this epilogue
  costs ~1 us without any transpose buffer, and a TMA store should make it ~0.5.

## 1. Insertion point (1): register-A wgmma in the consumer warpgroup, decode interleaved between issue and wait

pw10's structure (`decode(u+1); mma(u+1); wait_group<1>; release(u)`, pw10:743) inside DeepGEMM's skeleton, at 64x64
tiles with merged k128 stages.

**Per-stage consumer timeline (k128, NT = 64).** `wait full(u+1)` (already landed) -> 6 x LDS.128 for u+1 (issued at the
end of iteration u, so they are in registers) -> 8 x `pw_dec` (72 ALU, 8 independent chains of depth 3 + PRMT: ~75-90 clk
issue-bound at ~1 IPC) -> escape walk (2 uniform instr) -> `wgmma.fence` -> 8 x `wgmma m64n64k16` RS (~40 clk to issue)
-> commit -> `wait_group<1>` (returns when wgmma(u) drains) -> arrive empty(u) -> wait full(u+2) + 6 LDS for u+2.
The tensor pipe runs wgmma(u) for 256 clk; wgmma(u+1) is queued ~130 clk after wgmma(u) started, so the pipe never
idles as long as decode chain + issue (~130 clk) < tensor per stage (256). Margin ~2x at k128/NT 64; ~1x at k64/NT 64
once the ~60-100 clk of per-stage sync latency is added, which is where pw15 q1b lost (+6.7 decode, +10.3 wgmma at
k64, 2 CTAs/SM, 109 regs, no prefetch).

| Item | Value |
|---|---|
| Registers / thread (consumer) | acc NT/2 + A double buffer (2 x 4 regs x k16 steps per stage) + packed prefetch (3 uint4 per k64) + escape cursor 4 + descriptors/loop/addresses ~28. NT 64, k128: 32 + 64 + 24 + 4 + 28 = **~150**; k64: ~110 (pw15 q1b measured 109); NT 128 k128: ~180; NT 256 k64: ~200. All under setmaxnreg 232/240. |
| smem bytes / k64 stage | bulk W 6 KB + TMA x 8 KB = **14 KB** (k128: 28 KB -> 7 stages = 196 KB). Port traffic per k64 stage: fills 14 + LDS 6 + wgmma B reads 8 = 28 KB = 219 clk (84% of the H100 260-clk period; port-bound at 219 clk on the H200, still 14.2 us < floor+ramp). |
| Extra consumer instr / weight | decode 1.125 + LDS 0.094 + escape check 0.06 + address/loop ~0.1 = **~1.4 lane-instr/weight**; per warp per k64 stage ~47 on top of the ~18 of the bf16 SS loop. Issue utilisation ~25% of one scheduler. |
| Escape fixup without a side table | No (see section 6): the in-register XOR walk needs the per-warp list (2 uniform instr/stage + ~22 per hit, 0.1 hits/warp-stage). The epilogue-correction scheme moves the list to the idle producer warps + epilogue, 0 consumer instructions in the main loop. |
| Expected vs bf16 parity (pooler m=128) | **-4 to -5 us** (H100: 19-20 vs ~24; H200: 15-16.5 judge-us vs 20.3). Risk case (decode exposed as in pw15): +1. |

Notes.
- RS halves the wgmma's port demand versus SS at N=64 (A 2 KB + B 2 KB per 32 clk = 128 B/clk SS; B only = 64 B/clk
  RS). That is the reason CUTLASS's mixed-input collective is RS and the reason (2)/(3) lose below.
- A registers are read asynchronously: double buffer + `"+r"` fences after the covering wait (wgmma-layout.md trap 1);
  pw10 did this with 0 serialisation warnings. The packed layout `[n-block][chunk][tile][1536]` makes a k128 stage one
  contiguous 12 KB bulk copy; x needs two SW128 boxes of {64 k, 64 tok} per k128 stage (DeepGEMM's BLOCK_ATOM_K = 64
  descriptor advance handles the second atom).
- Rings: a combined (W + x) 7 x 28 KB ring keeps 5 stages = 80 KB of x in flight (x needs ~60 KB per SM at 1 us loaded
  L2 latency) and 60 KB of W (needs ~40 KB at 25 GB/s per SM). If the W+x rung shows x starvation, split the rings
  (W released after the LDS, x after wait_group<1>) with one producer lane per ring in different warps (pw13's
  head-of-line lesson; DeepGEMM's TMA warpgroup has 4 warps, 3 idle).
- H100 vs H200: the stage period is 1.4x longer on the H100, so "decode hidden on the H100" needs >= 1.4x margin to
  transfer. Measure the margin (L3 - L2 of the ladder) rather than the verdict.

## 2. Insertion point (2): transform warpgroup -> bf16 A ring -> SS wgmma (pw19 with a DeepGEMM-grade pipeline)

pw19 `pwt` (decode warps pw19:991-1035: 3 LDS.128 + `pw_unit` + `pw_fix3` + 4 `stmatrix.x4` + `fence.proxy.async`;
MMA warps: wait A full + x full, 4 SS wgmma, wait_group<1>, release). The DeepGEMM-grade fix would be: one combined
x+A+W ring depth >= 8, two producer lanes, setmaxnreg (transform 80-96, MMA 232), merged k128 stages.

| Item | Value |
|---|---|
| Registers / thread | MMA warpgroup: acc 32 + desc + misc = ~60 (pw24's SS bg: 58 measured); transform warpgroup: 16 A + 12 packed + 4 escape + 4 stsm addresses + ~20 = ~60 (pw19 t64: 90 for the whole kernel). 3 warpgroups = 384 threads. |
| smem bytes / k64 stage | W 6 + x 8 + **A 8** = 22 KB (k128: 44; 4-5 stages in 227 KB, or three rings of unequal depth as pw19). Port traffic: fills 14 + LDS 6 + STSM 8 + wgmma A 8 + B 8 = **44 KB = 344 clk per k64 stage** > the 260-clk H100 period: port-bound at 44 K clk = 22.2 us kernel (pw19 measured 60 KB/stage at NT 128 and +7 us on the wgmma rung; pw24's bf16 SS at 32 KB/stage already showed +10.9). |
| Extra consumer instr / weight | **~0.06** (one extra mbarrier wait and arrive per stage). Transform warps: ~1.6 lane-instr/weight incl. STSM, proxy fence, membar (pw19 SASS: 82 warp-instr per tile-stage). |
| Escape fixup without a side table | The list walk lives in the transform warps: **0 instructions on the MMA critical path by construction**, bit-exact, but it is still a side table. |
| Expected vs bf16 parity | **+0 to +2 us** at 64x64 (port-bound at 22.2 + ramp + three-ring latency chain, ~24-26 vs ~24). At 64x128 split-K 2 the port ratio improves (60 KB per 520-clk stage) but the split-K reduction costs +4.5. No configuration beats (1). |

The decode is hidden (pw19 proved that: +2 us in separate warps) but every decoded byte crosses the smem port twice more
(STSM write + SS read) and the SS wgmma at N=64 saturates the port by itself. Dominated.

## 3. Insertion point (3): consumer decodes the NEXT stage into registers while the current stage's SS wgmma runs; stmatrix before the fence

Single warpgroup; A tile written by the consumer itself into a 2-slot smem A buffer (no A-ring barriers: the slot is
reused after the wait_group that covers the wgmma that read it).

| Item | Value |
|---|---|
| Registers / thread | acc 32 + A 16 (one buffer suffices: registers are dead after the STSM) + packed 12 + escape 4 + stsm addr 4 + ~28 = **~96** (NT 64). The cheapest in registers. |
| smem bytes / k64 stage | W 6 + x 8 + A 2 slots x 8 = 14 KB per stage + 16 KB fixed. Port traffic **44 KB per k64 stage = 344 clk**, identical to (2). |
| Extra consumer instr / weight | decode 1.125 + LDS 0.094 + 4 STSM.x4 (0.125) + `fence.proxy.async` + `membar` + escape 0.06 + addresses ~0.1 = **~1.6**, all on the critical path, plus the proxy fence's latency (pw19 SASS: MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S) before every wgmma.fence. |
| Escape fixup without a side table | Same as (1): in-register XOR before the STSM (list) or the epilogue correction. |
| Expected vs bf16 parity | **+2 to +4 us**: (2)'s port bound plus the decode and proxy fence back on the single warpgroup's critical path. |

What (3) buys over (1) is only the removal of the RS hazards (A-register liveness across the async read, double
buffering) and 50 registers; it pays 16 KB per k64 stage of port traffic and a proxy fence per stage for that. Dominated
by (1) in every column; worth building only as a control if RS wgmma at N=64 turns out to issue-serialise on the H100
(check `ptxas -v` for C7510 and the SASS for WARPGROUP.DEPBAR after every HGMMA).

## 4. Insertion point (4): two consumer warpgroups ping-pong over stages (one decodes while the other issues)

Warpgroup A takes stages 0, 2, 4, ...; B takes 1, 3, 5, ...; both own the same 64 rows x NT tokens with separate
accumulators, summed once through smem at the end (pw15 q2's structure; m128-study step 4). Not DeepGEMM's BLOCK_M=128
cooperative split (that needs 128-row tiles: 64 CTAs at pooler, or 32-token tiles with 264 MB of L2->SM traffic).

| Item | Value |
|---|---|
| Registers / thread | per warpgroup: acc 32 + A 16 (single buffer: decode(u) -> fence -> wgmma(u) -> wait<0>; the other warpgroup fills the gap) + packed 12 + escape 4 + ~28 = **~95** (NT 64); NT 256: ~190. 2 x 128 consumer threads at setmaxnreg 224 (DeepGEMM's 2-WG value) + 128 producer threads at 40. |
| smem bytes / k64 stage | **14 KB**, as (1); plus NT/2 x 128 x 4 B = 16 KB (NT 64) for the final cross-warpgroup sum, reused from the drained ring. |
| Extra consumer instr / weight | per warpgroup ~1.4 lane-instr/weight on its own stages, i.e. **~0.7 per weight of wall time per warpgroup**; the SM's total issue slots equal (1). Each warpgroup's period is decode ~100 + wgmma 128 + sync ~80 = ~310 clk per 2 stages against a 520-clk stream period: 1.7x margin at k64 without any software pipelining, ~3x at k128. |
| Escape fixup without a side table | As (1); the list must be split by stage parity (pw15 `pw_fix4` did this). |
| Expected vs bf16 parity | **-4.5 to -5.5 us**: the same floor as (1) with the decode exposure risk removed; -0.2 us for the final sum and the second accumulator's zero-fill. |

Cost: 384 threads, two escape cursors, the stage-parity ownership of ring slots (an even ring depth is required: pw15
found the parity aliasing at odd ST), and at m=256 the second accumulator set (128 regs per warpgroup is fine at 224).
This is the lowest-risk way to hide the decode and the smallest change from (1): swap the single consumer's loop for a
stage-parity loop and add the smem sum.

## 5. Insertion point (5): CUTLASS mixed-input style, two narrow A planes (8-bit + 4-bit) converted in registers

Two K-major planes, `P8[n][k]` (exponent LSB + mant7, 4 KB per 64x64 k64 stage, SW64) and `P4[n][k]` (sign + 3-bit
offset, 2 KB, SW32 at k64 / SW64 at k128), each a 2D TMA box per stage, loaded into registers with `ldmatrix` and
converted with dec1's arithmetic; RS wgmma. In the real CUTLASS collective this is the forked
`sm90_mma_tma_gmma_rs_warpspecialized_mixed_input` (cutlass-mixed-input.md option (i)); in a hand DeepGEMM-style
pipeline it is (1) with a different load path.

| Item | Value |
|---|---|
| Registers / thread | = (1) + 2 descriptors/addresses: **~155** (NT 64, k128). CUTLASS's stock RS mixed-input instantiations: 168, 0 spills. |
| smem bytes / k64 stage | P8 4 + P4 2 + x 8 = **14 KB** (same bytes as (1); 2 TMA issues + 1 x box per stage instead of 1 bulk + 1 box). Port traffic 28 KB per k64 stage: as (1). |
| Extra consumer instr / weight | ldmatrix: 3 LDSM.x4 per tile replace 3 LDS.128 (equal); conversion = dec1 only if the planes are stored k-permuted so that the ldmatrix fragment lands in dec1's word positions (lane t of an 8x8 b16 matrix gets 4 consecutive k of one row; the bf16 A fragment wants k {2t, 2t+1, 8+2t, 8+2t+1}: a static per-row K permutation at pack time fixes this, which TMA does not care about). Without that, +4-8 PRMT/LOP per 32 weights: **1.4-1.65 lane-instr/weight**. |
| Escape fixup without a side table | No. The planes have no spare code (a 16-value window leaves none; a 7-code window raises escapes to ~0.5-1%). In a hand pipeline the in-register walk works exactly as in (1); in the CUTLASS collective the converter is a pure functor, so the list means forking the collective, which is why cutlass-mixed-input.md proposed the post-GEMM correction (1-1.5 us, and it double-rounds unless applied in the fp32 epilogue). |
| Expected vs bf16 parity | **= (1) +/- 0.5 us** as a hand pipeline (the only gains are TMA OOB zero-fill and multicast-ability of both planes; the losses are two tensor maps and a 32-B swizzle at k64, which DeepGEMM's heuristic rejects as slow). As the CUTLASS fork: the same floor minus 0.5-1 us of collective overhead, plus 5 days and a 0.9 MB header bundle. |

The bespoke pack (`_pack_nat` + `_c12`, pw10:1149-1186) is already a lane-contiguous fragment image (48 B per lane per
tile, 3 conflict-free LDS.128, bit-exact packer and decoder proven on the judge). Two planes + ldmatrix reproduce it
at best. Build (5) only if the CUTLASS collective is the chosen skeleton; in a DeepGEMM-style hand pipeline it is (1).

## 6. The escape fixup: three mechanisms

Facts: entry = `seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7` (pw10:231), one sorted sentinel-terminated list
per consumer warp, cursor `nx` prefetched one entry ahead, `nxs == seq` compared once per stage (`pw_fix3`
pw10:622-633). ~0.1 hits per warp per k64 stage on pooler.

| Mechanism | Main-loop cost on the consumer | Side table | Numerics | Fits |
|---|---|---|---|---|
| A. In-register XOR walk (current) | 2 uniform instr/stage + ~22 on a hit (one lane: compare, 16-way predicated XOR, reload) = ~4 per warp-stage, 0.1 lane-instr/weight | yes (global list, 4 B/entry, 13 KB total) | bit-exact | (1) (3) (4) (5-hand); in (2) it runs in the transform warps |
| B. Epilogue correction with x columns captured in-loop | **0 on the consumer in the main loop.** The 3 idle warps of the TMA warpgroup (40 regs) walk the per-CTA list (~52 entries); at stage s they copy `x[tok][k]` for the entry's k (NT x 2 B from the swizzled x tile, 128 B at NT 64) into a 52 x 128 B side buffer, and count on the stage's empty barrier (+1). After `wait_group<0>` and before the bf16 convert, the 4 lanes of group g = n mod 8 do NT/4 LDS.u16 + NT/4 FFMA per entry: ~52 x ~40 clk spread over 8 groups = ~0.15-0.3 us tail. | yes, but off the consumer's loop | fp32-exact for one-hot rows (W_dec + (W - W_dec) is exact in fp32); random rows within tolerance (order) | all five; the natural choice for the CUTLASS fork |
| C. In-band | impossible in 12 bits: an escaped weight needs 16 bits; reserving a window code costs 1.3% escapes (STATS: 8-code window) or +1 bit/weight | none | | none |

Do not apply the correction to the accumulators inside the main loop: touching D in the generic proxy requires a
`wait_group<0>` and drains the pipe (wgmma-layout.md section 5). The answer to "without a side table in the main loop"
is therefore B: the list is consumed by the producer warpgroup and the epilogue, never by the MMA warpgroup's loop.

## 7. Summary table (pooler m=128, 64x64 tiles, 128 CTAs, H100 SXM unless stated; parity = DeepGEMM-grade bf16 SS at ~24 us kernel on the H100, 20.3 judge-us on the H200)

| # | Insertion point | Regs/thread (NT 64) | smem/k64 stage | Port traffic/k64 stage | Extra consumer lane-instr/weight | Escape w/o side table in loop | Expected vs parity |
|---|---|---|---|---|---|---|---|
| 1 | RS wgmma, decode interleaved, k128 merged stages, A double-buffered | ~150 (k128) / ~110 (k64) | 14 KB | 28 KB = 219 clk | ~1.4 | B (0 instr) or A (0.1) | **-4 to -5 us** (risk +1) |
| 2 | Transform warpgroup -> bf16 A ring -> SS | MMA ~60, transform ~60 | 22 KB | 44 KB = 344 clk (port-bound 22.2 us) | ~0.06 | walk in transform warps (0 instr) | +0 to +2 |
| 3 | Consumer decode -> stmatrix -> SS, next stage in registers | ~96 | 14 KB + 16 KB | 44 KB = 344 clk | ~1.6 + proxy fence | A or B | +2 to +4 |
| 4 | Two consumer warpgroups ping-pong over stages, RS | ~95 per WG, 384 threads | 14 KB (+16 KB final sum) | 28 KB = 219 clk | ~0.7 per WG of wall time | A (parity-split lists) or B | **-4.5 to -5.5 us** |
| 5 | 8-bit + 4-bit planes via TMA + ldmatrix, RS | ~155 | 14 KB | 28 KB = 219 clk | 1.4-1.65 | A or B (B in the CUTLASS fork) | = (1) +/- 0.5 |

On the H200 judge (12-bit floor 12 us, port floor 14.2 us at 219 clk x 128 stages): (1)/(4)/(5) land at 15-16.5
judge-us if delivery is solved (= m128-study's 16-18 estimate); (2)/(3) are port-bound at 22.2 us, above cuBLAS.

Ranking: (4) >= (1) ~ (5-hand) >> (2) > (3). Recommended: build (1) on the DeepGEMM skeleton (single math warpgroup,
k128 merged stages, 7 x 28 KB, setmaxnreg 40/232, TMA-store epilogue with stmatrix.trans) and keep (4) as the one-line
fallback if the H100 ladder's L3 - L2 step exceeds ~1 us; the two share the producer, the packer, the decode, the
escape walk and the epilogue.

## 8. What to measure on the H100 before choosing (local, no judge)

Ladder on the skeleton, pooler m=128, each rung in kernel-us (cudaEvent, 20 reps, L2 flushed between reps):
1. S0: bulk stream of the 50 MB pack alone, 128 CTAs, 7 x 12 KB in flight: the floor (expect ~17-18).
2. S1: + x TMA tiles (64 x 64, SW128): the delivery rung. **Kill line: > 19 us**, in which case x multicast (cluster 2
   along n) or 128x128/split-K 4 comes before any decode work.
3. S2: bf16 SS control (TMA'd bf16 A, 8 KB, 13 x 16 KB stages or 6 x 32 KB merged): defines parity on this GPU; compare
   with cuBLAS/torch on the same box.
4. S3: (1) with the decode into a register sink (no wgmma): decode exposure.
5. S4: (1) full RS wgmma; S4 - S3 is the wgmma step, S3 - S1 the decode step. Both should be < 1 us at k128.
6. S5: + epilogue (stmatrix.trans + TMA store). Expect < 0.7.
7. Microbenchmarks that settle (2)/(3) without building them: SS vs RS m64n64k16 throughput under a concurrent 55 B/clk
   TMA fill into the same SM (port contention), and the stmatrix + fence.proxy.async latency per stage.

Scale every H100 margin by 1/1.4 before reading it as an H200 prediction (the stage period on the judge is 186 clk
against 260 here).
