# m=128 plan: RS128 — 128-row swap-AB register-A split-K kernel with a two-lane producer and a TMA reduce-add epilogue

Synthesis of three designs (DG-RS128, pwc, SK-128) and three judge verdicts (winners: SK-128, pwc, pwc), written
2026-10-26. Ground truth used: LOOP.md (it. 7-8, 11, 13, 17, 22-24, drawing board v2, H100 sections), audits/pw10-ladder.md,
m128-study.md, cutlass-mixed-input.md, wgmma-layout.md, pipeline-diff.md, deepgemm-sm90-bf16.md, decode-insertion.md,
p2-h100.md (box stream calibration), cutlass_sweep.md (box cuBLAS kernels and parity configs), DeepGEMM
`sm90_bf16_gemm.cuh` / `tma_copy.cuh` / `heuristics/sm90.hpp` (re-read for the cited lines). No kernel was run here.

Deadline: kernel track closes 2026-11-04 23:59 PT (9 days). Judge checks: 20/day; the box (H100, 132 SMs, 3.35 TB/s
peak, 50 MB L2) is the measurement instrument, the judge (H200) the only board number.

---

## 0. Decision in one paragraph

All three designs are the same physics: swap-AB, 128 W rows x all m tokens per CTA, two cooperative math warpgroups
feeding register-A `wgmma.m64n{m}k16` from the lossless 12-bit pack decoded in the math warps, x as the smem B operand,
K split across CTAs, cross-CTA reduction by fire-and-forget `cp.reduce.async.bulk.tensor` bf16 add, no cluster. They
differ in (a) the producer/ring structure (one combined 28 KB slot and one lane in DG-RS128/pwc; split W/x rings with two
lanes and W released at the LDS in SK-128), (b) the grid (KS split-K grid of 126-128 CTAs vs a 132-CTA stream-K unit
walk), (c) how `out` is zeroed (memset node vs in-kernel), and (d) the experiment order. The verdicts agree on the
resolution: **pwc's kernel body and experiment order, SK-128's producer (two lanes, split rings, W released at the LDS),
DG-RS128's L2-resident clk/stage probe and judge-mechanics day, memset node first with in-kernel zeroing as a measured
refinement, and the 132-CTA stream-K walk as the second-iteration grid for qkv/o once the KS grid passes.** The name
below is RS128. Its central H200 expectation at pooler m=128 is ~17.0 judge-us (range 16.5-17.8) vs cuBLAS 20.3, i.e.
-3.3 us; the >= 3 us bar is at the centre of the distribution, not comfortably cleared, and the two levers that add
~1.2 us of headroom (no memset node; 132 instead of 128 streaming SMs) are the first refinements after the core passes.
m=256 is parity to -1 us for every design (tensor/port co-limited) and ships only if it beats cuBLAS on the box by >= 1.5 us.

The parallel CUTLASS route ('mx': two-plane 12-bit A operand inside the parity-reached `ws_64x64x64_c1x2` collective)
is not replaced by this plan; section 7 states the comparison point between the two.

---

## 1. Chosen design: RS128 (full spec)

### 1.1 Orientation, tile, grid

- Swap-AB as pw9/pw10: wgmma M = 64 W rows per math warpgroup (A = decoded W in registers, K-major from the pack),
  N = tokens (B = x, K-major SW128 in smem), K = 16 per instruction. D[row][tok] -> `stmatrix.trans` -> out[tok][row].
- CTA tile = 128 W rows x NT tokens x (K / KS), NT = m in {128, 256}; every other m stays on the cuBLAS/router fallback.
- Grid (first iteration, "KS grid"): `(N/128, KS)`, KS = floor(132 / (N/128)) capped so the grid is one wave:
  pooler (N 4096, K 8192): 32 x 4 = 128 CTAs, 32 k64 stages each; unpooler (8192, 4096): 64 x 2 = 128, 32 stages;
  o (8192, 1024): 64 x 2 = 128, 8 stages; qkv (2304, 8192): 18 x 7 = 126, 18/19 stages (uneven ranges allowed; the
  reduce-add does not care). 1 CTA/SM, non-persistent, no cluster attribute anywhere.
- Grid (second iteration, "SK grid", after the KS grid passes S4): 132 persistent CTAs each owning a contiguous range of
  12 KB units `u = rb*(K/64) + s` (pooler/unpooler 4096 units -> 31.03 per CTA, qkv 2304 -> 17.45, o 1024 -> 7.76); a
  range crosses at most one row-block boundary, so at most two reduce-adds per CTA; accumulator reset by `scale-d = 0`
  on the first k16 after a boundary. Same kernel body; only the range assignment, the escape-list keying and the
  mid-range epilogue differ. It removes the 126/128-CTA under-fill, qkv's 18/19 unevenness and (with in-kernel zeroing)
  the memset node. Its known cost: a second segment can be as short as one stage (SK-128's ">= 7 stages to the next
  partial" is false), so the D slab is reused only after `cp.async.bulk.wait_group.read 0` and the tail is bounded, not
  hidden.

### 1.2 Warp roles, registers

- 384 threads, `__launch_bounds__(384, 1)`: warps 0-3 = math WG0 (rows 0-63 of the block), warps 4-7 = math WG1
  (rows 64-127), warps 8-11 = producer WG. `setmaxnreg`: producer `dec 40`, math `inc 232` (128x40 + 256x232 = 64,512).
  This is CUTLASS's split (DeepGEMM uses 48/248 for one math WG, 48/224 for two); pw10 wk2 ran 40/232 on the judge.
- Producer WG: warp 8 lane 0 (`elect_one_sync`) = W lane (`cp.async.bulk` 12,288 B per stage from the contiguous pack,
  `complete_tx` on `wfull[s]`); warp 9 lane 0 = x lane (`cp.async.bulk.tensor.2d` box {64 k, NT tok}, SW128,
  `complete_tx` on `xfull[s]`); warp 10 inits barriers and prefetches the tensor maps; warp 11 idle (reserved for the
  escape-capture mechanism B of decode-insertion.md s6 if the in-register walk ever shows on the stamps). Two lanes in
  different warps = pw24 bg's producer (12.1 TB/s SM ingest measured on the judge; the box reproduced W-only at the
  cuBLAS-GEMM rate), so W never waits behind a pinned x slot (pw13's head-of-line lesson).
- Math warp w (0..7) decodes rows `16*(w%4) .. +15` of its WG's 64-row half for all NT tokens.
- Registers per math thread: NT 128: acc 64 + A double buffer 2x16 + packed prefetch 12 + escape cursor 4 + descriptors,
  stage/phase, addresses, loop ~30 = ~140. NT 256: acc 128 -> ~205; must be proven by `ptxas -v` under
  `__launch_bounds__(384,1)` + setmaxnreg (the lineage's "168 at 2 WGs" is the launch-bounds cap, not evidence); if it
  spills, single-buffer A with `wait_group 0` (-16 regs) is the first fix.

### 1.3 Shared memory: split rings, m=128

| region | size | notes |
|---|---|---|
| W ring | 9 x 12,288 B = 110,592 B | slot released by lane 0 of each math warp right after its 3 x LDS.128 (W smem is dead once in registers: RS removes wgmma-layout trap 2 for W) -> 8-9 of 9 slots always loading, ~100 KB of W in flight vs ~58 KB Little's-law need at 1.5 us loaded latency |
| x ring | 7 x 16,384 B = 114,688 B | 1024-aligned SW128 tiles; released after the `wait_group 1` that covers the stage's wgmmas (trap 2 applies to x) -> 5 free slots = 80 KB effective; pw19's 4 x 16 KB ring (32-48 KB effective) cost +5 us, SK-128's 5 x 16 KB was judged at the knee, 7 is the minimum here |
| D slabs | carved from drained x slots 0-1 after the last stage (KS grid); a dedicated 16 KB slab per WG in the SK grid (then W ring 8 x 12 KB) | [NT tok][64 n] bf16, SW128 XOR by hand as DeepGEMM K:311-340 |
| barriers | wfull/wempty/xfull/xempty, 32 x 8 B = 256 B | wfull/xfull init 1 + `expect_tx`; wempty/xempty init 8 (lane 0 of each math warp) |
| total | 225,536 B (KS grid) / 229,632 B (SK grid) | <= 232,448 opt-in; `cudaFuncSetAttribute` max dynamic smem |

m=256: x tile 32 KB; W 7 x 12 KB + x 4 x 32 KB = 212 KB (2 free x slots = 64 KB, ~1 us of lookahead at a ~1,100-clk
tensor-bound period: enough only because m=256 is not HBM-bound); D carved from the drained x ring.

Combined-slot fallback kept as a PRB flag (pwc's 8 x 28 KB ring, one lane, one full barrier per slot, slot released after
`wait_group 1`): ~168 KB effective in flight (two slots pinned), 72 KB of W. It is measured side by side with the split
rings at rung S1 (section 5, E4), not assumed.

### 1.4 Packed W layout (prepare)

- pw10's `_pack_nat` + `_c12` fragments (judge-proven bit-exact packer/decoder: 16-row x 64-k tiles of 1536 B, 48 B
  contiguous per lane, 3 conflict-free LDS.128 per lane; low-byte planes p0/p1 + code word p2 with the 16-value exponent
  window, table `Tb = B*0x01010101`), re-blocked to 128-row blocks: `[row-block][k64 chunk][8 tiles x 1536 B]` =
  12,288 B per (block, chunk) = one `cp.async.bulk`. Warp w of WG g reads its 1536 B at `stage_base + (4g+w)*1536`.
- KS grid: CTA (nb, ks) streams chunks `[ks*S/KS, (ks+1)*S/KS)` of block nb: one contiguous byte range. SK grid: unit
  order is the same layout read as a flat array.
- Escapes (~0.01%, ~0.1 hits per warp-stage): pw10's sorted sentinel-terminated per-warp lists (`seq<<17 | lane<<12 |
  reg<<8 | half<<7 | xor7`), partitioned per (row-block, KS range, in-block tile index) for the KS grid; keyed by global
  unit index with a binary-searched cursor for the SK grid. `pw_fix3`: 2 uniform instructions per stage, ~22 on a hit.
- Independent `_unc12`-style decoder check of the new blocking in prepare before any timing; a bug here fails every
  shape at once.

### 1.5 Math warpgroup loop (per WG, per k64 stage s), RS wgmma

```
[packed words for s already in 3 x uint4 from the LDS issued last iteration]
pw_unit: 4 x pw_dec (per 8 weights: 2 LOP3, 1 SHF, 2 IADD, 4 PRMT = 9 instr; 36 ALU)  -> A[s&1][16]
pw_fix3 escape walk (2 uniform instr; ~22 on a hit)
wait xfull[s](phase)
wgmma.fence
4 x wgmma.mma_async.sync.aligned.m64n{NT}k16.f32.bf16.bf16  {A[s&1][4q..4q+3]}, bdesc(s) + 2q, scale-d (0 on the CTA's / segment's first k16, else 1), 1, 1, 0
wgmma.commit_group
wait wfull[s+1](phase'); 3 x LDS.128 of stage s+1's packed words; lane-0 arrive wempty[s+1]
wgmma.wait_group 1            // group s-1 retired
"+r" fences on A[(s-1)&1], "+f" fences on acc (wgmma-layout.md traps 1-2)
__syncwarp; lane-0 arrive xempty[s-1]
```

- One group in flight per WG, two per SM; the two WGs are not phase-locked, so WG1's ~200-300 clk poll/LDS/decode/issue
  chain runs under WG0's 256-clk batch and vice versa. This is CUTLASS's K_PIPE_MMAS=1 protocol with a lagged release,
  not DeepGEMM's (`warpgroup_wait<0>` then same-stage arrive, K:278-281); DeepGEMM's variant (`wait_group 0`, single A
  buffer, -16 regs) is kept as a flag and is the first fix if ptxas reports C7510 or the stamps show a drain bubble.
- B descriptor built once per CTA (`make_gmma_desc` form: start>>4, LBO 1, SBO 1024 -> 64, base offset 0, SW128 = 1<<62),
  `__shfl_sync`-broadcast, low word advanced by `stage_bytes/16` per stage and +2 per k16 (DeepGEMM K:219-222, 250-251,
  266-270).
- Accumulators never zero-filled by hand (scale-d 0 on the first k16).
- Variants kept as flags: k128 merged stages (8 wgmma per fence/commit/wait, two W units per round), `wait_group 2` with a
  triple A buffer, stage-parity ping-pong (one WG primed one stage ahead) if the stamps show the two WGs in lockstep.

### 1.6 Epilogue (per WG; no cross-WG sync, no cluster)

`wait_group 0` -> acc fences -> named barrier (128) -> `cvt` pairs to bf16x2 -> 16 x `stmatrix.sync.aligned.m8n8.x4.trans`
into the WG's slab `[NT tok][64 n]` (128-B rows, XOR `(n/8) ^ (tok & 7)`) -> `fence.proxy.async.shared::cta` -> named
barrier -> one elected thread: `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [out_map,
{n0 + 64*wg, 0}], [slab]` -> `cp.async.bulk.commit_group` -> `cp.async.bulk.wait_group.read 0` before the slab is reused
or the CTA exits. `out` tensor map: dims {N, m}, box {64, NT}, SW128, bf16.

- `out` zeroing, first iteration: a memset node captured in the same graph before the kernel (pw22/pw23's judge-proven
  pattern, 0.8-0.9 us at 1-4 MB). Second iteration (E8): the KS=0 CTA (or each CTA's 1/132 slice in the SK grid) zeroes
  with `st.global.v4` during its prologue, `fence.proxy.async.global` + `red.release.gpu` on a per-row-block flag; every
  CTA `ld.acquire`-polls the flag once before its reduce-add (set >= 10 us earlier, no wait in practice); the last
  exiting CTA resets flags/counters for the next replay. Promoted only after 100+ graph replays with checksums show
  -0.6 us or better; pw23's only prior self-cleaning variant lost (15.71 vs memset 13.89).
- Numerics: KS partials each rounded once to bf16, KS-1 bf16 adds in the TMA unit; one-hot probe rows exact (each
  partial is one exact W element or exactly 0); random rows ~3-4e-3 relative L2 at KS 4, ~4e-3 at KS 7 (torch
  simulation lineage, pw27; never measured on hardware) vs the judge's REL_L2_TOL 1e-2. Fallback: fp32 workspace
  reduce-add (`cd_dtype_t = float`) + a convert node (+1-1.5 us). The candidate's own bake tolerance must be set per KS
  (the lineage's TOL bf16 = 4e-3 would silently drop the KS 7 qkv path in prepare).

### 1.7 bf16 control (same kernel, one template switch)

W as a bf16 TMA box {64 k, 128 rows} SW128 (16 KB per stage) into the W ring (7 x 16 KB), SS `m64n{NT}k16` with the
A descriptor per WG, no LDS/decode; stage = 32 KB (not 48). It isolates the codec's byte cut from the pipeline. Its pass
criterion is parity within 1.5 us of cuBLAS on the box (not beating it): at this geometry only HBM binds, so a bf16 rung
can land 0-1 us under cuBLAS if the stream reaches the bulk-ring rate; a bf16 rung >= cuBLAS + 1.5 is a real warning.

### 1.8 Host and judge mechanics

- Own `load_inline` extension with the submission's exact flags: `-gencode arch=compute_90a,code=sm_90a -O3 -DNDEBUG
  --expt-relaxed-constexpr`; every wgmma/setmaxnreg/bulk-tensor block guarded by `__CUDA_ARCH_FEAT_SM90_ALL` so a stray
  sm_90 pass compiles to a stub (box pitfall: torch 2.8 appends compute_90/sm_90 unless TORCH_CUDA_ARCH_LIST=9.0a;
  `/root/okc_run.sh` sets it). Prepare continues with r6 + router if the extension fails (pw24 lesson, lint rule).
- Tensor maps via `cuTensorMapEncodeTiled` through `cudaGetDriverEntryPoint` (pw10:1044-1046 / pw24 BgTmap), two per
  shape (x, out); packed W a plain device buffer; `cudaFuncSetAttribute` max dynamic smem.
- Cache hints: W bulk `L2::cache_hint` EVICT_FIRST, x map EVICT_LAST, as a flag (DeepGEMM uses EVICT_NORMAL on both;
  the split is ours and unmeasured; zero cost to toggle).
- Router unchanged (r6 + pw23/pw29 router serve every other shape); upload < 1,000,000 B (no header bundle); prepare
  <= 200 s including r6's 118 s and the 128-row repack (~2-4 s on 160 MB).

---

## 2. Grafted ideas (and where each came from)

| graft | source | where it lands |
|---|---|---|
| Two producer lanes in separate warps, split W/x rings, W slot released at the LDS | SK-128 (bg's producer, 12.1 TB/s judge; pw13 head-of-line) | s1.2-1.3; flag variant vs the combined slot at S1 |
| x ring >= 7 slots (>= 80 KB effective) | verdicts 1-3 vs SK-128's 5 x 16 KB; pw19 tw64 +5 us at 32-48 KB | s1.3 |
| Offline torch emulation before any CUDA: KS-way bf16 reduce on the real weights (random, 'outliers', one-hot), stmatrix.trans slab XOR, 128-row repack bit-exactness | pwc E0 | E0 |
| Resident-smem RS/SS wgmma rate microbenchmark under a concurrent 28/44 KB-per-stage TMA fill + 12 KB LDS | pwc E2, pipeline-diff s4(c) | E2 |
| Isolated reduce-add experiment: 4-/7-way bf16 `cp.reduce.async.bulk.tensor`, one-hot exact, 100 graph replays | SK-128 E1 | E3 |
| Ladder as PRB-flag rungs of ONE kernel (S0..S4 + bf16 control) | pwc, pw24h's `bh<..., PRB>` structure | E4-E6 |
| L2-resident SM-side period probe with per-stage clock64 stamps and the <= 650 clk/stage bar | DG-RS128 E3 / SK-128 E4 | E5 (the only rung that transfers H100 -> H200) |
| Memset node first; in-kernel zeroing + release flag as a measured refinement | DG-RS128 E4b, SK-128, pw22/23 | s1.6, E8 |
| 132-CTA stream-K contiguous-unit walk as the second grid (qkv/o balance, no memset, no 18/19 split) | SK-128 | s1.1, E8 |
| EVICT_FIRST (W) / EVICT_LAST (x) hints as a flag | SK-128 | s1.8 |
| scale-d 0 accumulator reset; descriptor built once, shfl-broadcast, low word advanced per stage; uneven K ranges | DG-RS128 / DeepGEMM | s1.5 |
| `wait_group 0` + single A buffer (-16 regs) as the first NT 256 / C7510 fix | DG-RS128, DeepGEMM | s1.2, E2, E8 |
| Judge-mechanics day: exact flags, FEAT guards, driver entry point, memset node in graph, prepare budget, SASS pass (HGMMA RS count, UTMALDG/UTMASTG/UTMAREDG, no C7510, USETMAXREG pair) | DG-RS128 E7 | E9 |
| GO line: pooler m=128 <= 0.87 x cuBLAS in-prepare on the box, no served shape > cuBLAS + 0.5 (else routed) | pwc E9 | E7 |
| Per-KS bake tolerance (judge 1e-2 with the 5x-per-row rule; lineage bake 4e-3 would drop KS 7) | verdict 3 | s1.6, E0 |
| Reproduce the lineage's own 128-row 2-WG numbers (pw9 w2n128a 46.1, pw10 w2c2 38.8) rung by rung on the new skeleton so "lost to confounders" is tested | verdict 3 | E7 (optional) |
| Every clk figure recorded with `nvidia-smi -q -d CLOCK` sampled during the run | DG-RS128 E0, pipeline-diff s8 | E1 and all rungs |

Dropped: DG-RS128's E1 (DeepGEMM verbatim, 1 day, off the critical path); any W multicast; cluster DSMEM or last-arriver
reductions; the transform-warpgroup (pw19) structure; the token-tiled no-split (pw15) structure; CUTLASS stock configs as
the vehicle for the 12-bit path (they are the parallel 'mx' track).

---

## 3. Refuted claims to avoid (consolidated from the three verdicts; the corrected value is what the plan uses)

1. "cp.reduce.async.bulk.tensor bf16 add is the pattern pw22/pw23 proved on the judge." pw22/23 proved the memset node +
   `red.global.add` fp32 scalar atomics. The TMA reduce-add has never run on the judge or the box; DeepGEMM's accumulate
   path adds one tile once from one CTA. -> E3 isolates it before integration.
2. "cuBLAS pooler m=256 ~29-30 judge-us; m=256 win 6-8 us." Measured: 27.7-27.8 in-prepare = ~24 judge-us (cuBLAS at ~71%
   of tensor peak). A 1,024-clk-per-stage tensor-bound kernel at 85% is parity to -2 us. The >= 2 us m=256 goal is
   unsupported by any design; m=256 ships only on a measured box win.
3. "Box in-prepare = kernel + 4.9; cuBLAS kernel 22.7 us on the box (2.95 TB/s); F12 16.5-17." Measured: the box's empty
   graph is 3.46 in-prepare, so cuBLAS pooler m=128 is ~24.1 kernel-us (2.78 TB/s); the bare read probe is 17.41
   kernel-us for 50.3 MB (2.89 TB/s, ~1 us fixed + 3.2 TB/s marginal; p2-h100.md). F12 on the box is ~17.4-18; every
   H100 projection that used 4.9 is ~1.4 us optimistic.
4. "Keep DeepGEMM's skeleton exactly / protocol verbatim" with `wait_group 1` + lagged release + 40/232. DeepGEMM is
   `wait<0>` + same-stage release (K:274-281), 48/248 or 48/224 (K:127-128), producer warp `math/32 + 2`. Harmless, but
   label it CUTLASS's protocol and keep the DeepGEMM variant as a flag.
5. "Epilogue ~0.5 us; memset/zero ~0.5." Measured: memset node 0.8-0.9 us; ~1 us for 4 MB of partials at a one-wave tail
   (pw23 post-mortem); pw15's no-split epilogue +1.0; bg's +1.8 incl. drain. Fixed tail + zeroing is ~2 us, which alone
   consumes the ~1 us of slack DG-RS128/pwc had over the 3 us bar.
6. "224 KB in flight (96 KB W, 128 KB x), margins 2-3x" for the combined slot. Two slots are always pinned until
   wgmma(u-1) retires: ~168 KB effective, 72 KB of W, 1.25x over the ~58 KB need. The split ring with W released at the
   LDS is how the W margin becomes ~1.7x.
7. "Two cooperative warpgroups hide the decode/issue chain for free (w1d8z/wk2 evidence)." Unmeasured: wk2 was -0.6 vs
   w1d8 (-2.9 vs w1d4) and still ~2 us behind cuBLAS; the only 128-row 2-WG RS measurements (pw9 w2n128a 46.1 at
   cluster 4 / depth 4; pw10 w2c2 38.8 at 64 CTAs) were the slowest of their bakes; no ladder ever had a wgmma rung
   under +5 us. E5's stamps are the first measurement; the design is a hypothesis until then.
8. "x ring 5 x 16 KB, needs ~36 KB at 1 us latency." pw19's 4 x 16 KB ring (32-48 KB effective) cost +5 us; pipeline-diff
   fits ~1.5 us loaded latency (~100 KB for 34 B/clk/SM). x gets >= 7 slots here.
9. "pw27 measured 2.4e-3 / 3.2e-3 for 2-/4-way bf16 reduce." It is a torch simulation on m <= 32 router geometries; no bf16
   TMA reduce has been measured on any GPU (E0 re-simulates on the real weights; E3 measures).
10. "DeepGEMM uses EVICT hints; runs 64x256 at 224 regs." DeepGEMM issues EVICT_NORMAL on both operands (tma_copy.cuh:43);
    224 is the two-WG value, 248 for any 64xN tile. The EVICT_FIRST/LAST split is ours and unmeasured.
11. "judge-us = kernel + 0.5." The only calibration (cuBLAS 20.3 judge vs 24.2 in-prepare vs 4.9 empty launch; r6 m=1 shows
    the same ~4 us gap) gives judge = kernel + ~1.0. SK-128's 14.5-15.5 is 15.6-16.6 after normalisation.
12. "+9-10 us per cluster step is a cluster-launch cost; settle it with an empty-cluster probe." The box already settled
    it: CUTLASS cw7c2 (cluster 1x2) 31.97 vs cw7 32.10 (parity), while bg12c 37.47 vs bg12 29.02 (+8.5) is W-multicast
    pair lockstep. Cluster launch is ~free; sharing the HBM-bound operand across a pair is what costs. x multicast
    (L2-resident operand) stays an option only if S1 is ingest-bound.
13. "cuBLAS's 64x64 SS sits at 97% of the smem port (256/265)." 305 clk/stage at the judge's reported 1.98 GHz = 84%. The
    clock assumption must travel with every clk figure (this plan quotes both 1.755 and 1.98 GHz).
14. "m64n128k16 = 61 clk; stage tensor 488 clk; m=256 976." At 2048 MAC/clk/SM it is 64 clk: 512 clk per stage at m=128,
    1,024 at m=256 (compute floor 18.1 us at 100% on the H200 for pooler m=256).
15. "bf16 control stage 48 KB (port 96 KB)." A 128-row x 64-k bf16 W tile is 16 KB; the control stage is 32 KB, port
    80 KB = 640 clk.
16. "168 regs at 2 WGs (wk2) shows ~140-200 fits." 168 = 65536/384 rounded down to 8, the launch-bounds cap; only
    `ptxas -v` on the setmaxnreg build says whether NT 256's 128 accumulators fit without spills.
17. "4-way reduce 3e-3 vs tol 1e-2" without noting the lineage's own bake TOL (4e-3 bf16, pw27 submission): a KS 7 qkv
    path at ~4.2e-3 would be dropped in prepare, not by the judge. Set the bake tolerance per KS before the first bake.
18. "bf16 cannot beat cuBLAS by construction (three coincident ceilings)." Those ceilings are the 64x64 SS geometry's;
    at 128x128 RS/SS only HBM binds and the judge's bulk ring measured 4.15-4.4 TB/s vs cuBLAS's ~3.3-3.5 effective, so
    the bf16 control can land 0-1 us under cuBLAS. Gate stays "within 1.5 us"; >= cuBLAS + 1.5 is a warning.
19. "H100 margin is 1.6x the H200's." Period ratio 955/746 = 1.28 (1,077/841 at 1.98 GHz); SM-side slack ratio
    (955-576)/(746-576) = 2.2. Transfer rule: compare clk/stage; divide measured SM-side slack on the box by ~2.2.
20. "Ramp 1-1.5 us on top of the bare stream" double-counts (the probe includes ramp), but every single-lane W-only rung
    sat 2.0-3.7 us above the probe and only bg's two-lane two-ring producer matched it; "31 equal stages" is 31.03 (four
    CTAs carry 32 units, +3%).
21. "In-kernel zeroing + counters is the primary path." pw23's self-cleaning variant lost to the memset node; cross-proxy
    ordering and replay-safe reset are silent-failure risks with ~2 judge checks of slack. Memset first.
22. "Build days: 5" for DG-RS128 with 6.0 days of listed experiments and E1 off the critical path; "6 build days" for
    SK-128 = the whole window with no judge-mechanics day. This plan is 4 build days to the first judge check.

---

## 4. Per-stage budget (unit: one k64 stage of 128 W rows x NT tokens per CTA; smem port 128 B/clk/SM; 2048 MAC/clk/SM)

### 4.1 pooler m=128 (KS 4: 128 CTAs x 32 stages; W 50.3 MB 12-bit, x 2 MB)

| resource | per stage | clk | share of the H200 period |
|---|---|---|---|
| tensor | 2 WGs x 4 x m64n128k16 @ 64 clk | 512 | 69% of 746 |
| smem port | TMA fills 28 KB (W 12 + x 16) + LDS 12 KB + RS B reads 2 x 4 x 4 KB = 32 KB -> 72 KB | 576 | 77% (cuBLAS 64x64 SS: 84% of its 305) |
| L2 -> SM ingest | 28 KB per stage per SM = 38 B/clk/SM at 746 clk; aggregate 114 MB / 13.6 us = 8.4 TB/s | - | bg measured 12.1 TB/s, cuBLAS 13.0, pw10 single lane 8.7 |
| issue | 8 math warps x ~55 instr (36 decode + 3 LDS + 2 escape + 4 wgmma + ~10 pipeline) / 4 schedulers | ~110 slots | 15% |
| per-WG serial chain | full poll ~50 + decode ~80-130 + fence/4 issues ~40 + commit/wait/arrive ~40 | 210-300 | hidden while < the other WG's 256-clk batch + own queued group (margin 1.6x, derived, not measured) |
| producer | 2 empty waits + 2 bulk ops + 2 expect_tx | ~10 | - |
| bytes in flight | W ~100 KB (needs ~58), x ~80 KB (needs ~60-75 at 1.5 us) | - | 1.7x / 1.1-1.3x |
| mbarrier ops | 2 expect_tx + 16 arrives per stage | - | <= 0.5 us total (pw4/pw7) |

HBM-set period, H200: 50.3 MB / 3.7 TB/s = 13.6 us -> 746 clk at 1.755 GHz (841 at 1.98); at the 4.2 TB/s bulk-ring rate:
662 (747 at 1.98). Max(576, 512, 300+256) = 576 < 662: HBM-bound with 13-23% port margin. Non-HBM floor 576 x 32 /
1.755 GHz = 10.5 us.

HBM-set period, H100 box: 50.3 MB / 2.89 TB/s = 17.4 us -> 955 clk at 1.755 GHz (1,077 at 1.98): the box hides SM-side
overruns up to ~950 clk, so a kernel that beats cuBLAS on the H100 proves nothing about the H200 unless E5's L2-resident
probe reads <= 650 clk/stage.

Kernel estimate, H200: 13.6 (stream at 3.7) + ~1.2 (tail: reduce-add drain + last-CTA skew) + 0.85 (memset node) =
15.6 us; at 4.2 TB/s: 14.0. Judge-us = kernel + ~1.0 -> 16.6 (central 17.0 with the stream between the two rates, range
16.5-17.8) vs cuBLAS 20.3: -2.5 to -3.8, central -3.3. In-kernel zeroing -0.8; SK grid (132 SMs) -0.4.
Box: 17.4 + 1.2 + 0.85 = ~19.5 kernel-us -> ~23.0 in-prepare vs cuBLAS 27.5 (-4.5, of which ~3.5 transfers).

### 4.2 bf16 control (SS, W 16 KB/stage; stage 32 KB)

Port: fills 32 + A reads 16 + B reads 32 = 80 KB = 640 clk; tensor 512; HBM 67.1 MB / 3.7 = 18.1 us -> 994 clk: HBM-bound.
Kernel ~18.1 + 1.2 + 0.85 = ~20.2 (H200) / 67.1 / 2.78 = 24.1 + 2 = ~26 box kernel-us vs cuBLAS ~24.1: parity to +2 on the
box is the expected outcome (its 4-way split-K costs ~2 us of fixed tail that cuBLAS's no-split kernel does not pay).

### 4.3 Other m=128 shapes

- unpooler (KS 2, 64 x 2 CTAs, 32 stages, 50.3 MB): identical per-CTA budget to pooler; ~17 judge-us vs cuBLAS ~19.3.
- qkv (KS 7, 126 CTAs, 18/19 stages, 28.3 MB): H200 7.6 us / 19 = ~700 clk at 1.755 GHz > 576: HBM-bound; kernel ~7.6 x
  19/18.3 + 2 = ~9.9 -> ~11 judge-us vs cuBLAS 15.4. Highest-leverage shape (verify16 qkv 59 score-us); the SK grid
  removes both the 126-CTA under-fill and the 18/19 unevenness.
- o (KS 2, 8 stages, 12.6 MB): ramp/tail-dominated, ~3.4 + 2 = ~5.5-6.5 vs cuBLAS 7.2: parity to -1; routed to cuBLAS
  if the bake says so.

### 4.4 m=256 (NT 256, m64n256k16 @ 128 clk, acc 128 regs)

Tensor 2 x 4 x 128 = 1,024 clk; port fills 44 KB + LDS 12 + B reads 64 = 120 KB = 960 clk; HBM period 746: tensor/port
co-limited at ~1,100-1,200 clk -> 32 x 1,150 / 1.755 GHz = 21 + 2 = ~23 kernel-us vs cuBLAS ~23-24 kernel (27.7-27.8
in-prepare, ~24 judge-us). Parity to -1 us for any RS tiling (B reads are 1/32 B per MAC = 64 B/clk at full tensor
rate). The CUTLASS sweep's `ns_ws_64x128x64_c1x1` already has pooler m=256 at 0.945 on the box (pw30). m=256 ships from
this kernel only if it beats cuBLAS on the box by >= 1.5 us at pooler/unpooler; otherwise cuBLAS/pw30.

### 4.5 Fixed costs that every projection must carry

memset node 0.8-0.9 us (1-4 MB); reduce-add drain at a one-wave tail ~0.5-1 us (4 MB at KS 4); ramp + last-CTA skew
~1 us (inside the bare-stream probe already; counted once); judge replay floor ~1.0 over kernel; box in-prepare floor
3.46, box harness graph-replay floor ~11-13 (idle GPU; compare within a run only); box CUPTI kernel durations (pw28h
lab method) are the cleanest box number.

---

## 5. Ordered experiment plan for the H100 (cheapest decisive measurement first, judge check last)

Conventions: every timing under `/root/gpu_lock.sh`; `nvidia-smi -q -d CLOCK` sampled during each run and written next
to the numbers; rungs reported as kernel-us (CUPTI/kineto durations, L2 flushed, median of >= 30) and as clk/stage at
the recorded SM clock; box deltas transfer to the judge only as clk/stage or as HBM-bound us x 0.78; the pw28h lab style
(`lab.py`: compile only what the plan needs, validate every full path, interleave variants) is the template.
Work goes in a new candidate directory (e.g. `candidates/pw31h/`), never in a judge candidate.

| # | when | what to build | what to measure | kill criterion | what it decides |
|---|---|---|---|---|---|
| E0 | day 0, offline, 2-3 h, no GPU | (a) torch emulation of the KS-way bf16 TMA reduce rounding (KS = 4/2/7/2 for the KS grid; 5/3/9 partials for the SK grid) on the task's real weights (`muse_bf16_proj/reference.py`) for all 8 (layer, m) shapes with random, 'outliers' and one-hot x; (b) the 128-row repack + per-(block, range, tile) escape lists + an `_unc12`-style bit-exact decode of the new blocking; (c) stmatrix.trans slab XOR addressing emulation (pw15 emu style); (d) per-KS bake tolerance table | relative L2 per shape, one-hot exactness, packer round-trip | any shape > 5e-3 or any one-hot row inexact -> fp32 workspace + convert node (+1-1.5 us) from the start; packer round-trip not bit-exact -> stop until fixed | whether bf16 reduce-add is the epilogue, and the bake TOL per KS |
| E1 | day 0, box, 2 h | calibration kit: SM/mem clock under a 2 s load; kineto capture of cuBLAS's kernel name and duration for the 8 shapes (cutlass_sweep.md has F.linear names: pooler-128 `nvjet_tst_64x64_64x13_2x1`, unpooler-128 `128x64_64x8_1x2`, pooler-256 `128x64_64x8_1x2_h`, unpooler-256 `128x128_64x6_2x1`; torch.mm's came back empty and must be recaptured); bare `cp.async.bulk` stream of the 50.3 MB pack in the design's exact pattern (128 CTAs x 9 x 12 KB contiguous K/4 ranges; and 132 CTAs x unit ranges) and of 67.1 MB (F16); memset node at 1/2/4 MB inside a graph; empty-kernel launch with/without a cluster attribute | F12, F16 (kernel-us and clk/stage), memset cost, cluster-launch cost, cuBLAS kernel-us per shape | none (calibration); F12 > 20 us means the box is off spec: fix the box first. Expect F12 ~17.4-18 (p2-h100.md: 17.41 with ld.global), F16 ~23-24, cuBLAS pooler-128 ~24 | the bars for every later rung |
| E2 | day 0-1, box, 4 h | resident-smem microbenchmark (no TMA): 2 WGs each looping fence -> 4 x RS m64n128k16 -> commit -> wait_group 1, clk per batch; then + a concurrent producer writing 28 KB/stage into an unread ring + 12 KB/stage of consumer LDS (the real port load); repeat for SS m64n128k16, RS m64n256k16 with 44 KB/stage; compile the NT 256 consumer under `__launch_bounds__(384,1)` + setmaxnreg 232 and read `ptxas -v` | clk per RS instruction alone (expect ~64), clk per 2-WG stage pair loaded (expect ~550-600), spills/C7510 | RS n128 alone > 70 clk/instr or SASS shows DEPBAR after every HGMMA -> C7510/serialisation: fix the A-register fences before building; loaded stage > 700 clk -> the port model is wrong: switch to k128 merged stages and re-measure, and if still > 746 the 128x128 RS geometry cannot close on the H200 (fallback G1: 64x64 full-K RS, parity+); NT 256 spills -> single-buffer A, else m=256 routed to cuBLAS | the 576-vs-746 port question that the H100 cannot expose in a real kernel; the m=256 register question |
| E3 | day 1, box, 3 h | reduce-add epilogue in isolation: 128/132 CTAs each STSM.trans + `cp.reduce.async.bulk.tensor.add.bf16` one or two random 128x64 slabs into a memset-zeroed `out` (1 MB pooler, 2 MB o/unpooler), 4-way and 7-way (qkv) and 9-way (SK qkv) overlap; one-hot rows; 'outliers' x; 100 graph replays with checksums; standalone time and inside a dummy 32-stage loop; also the in-kernel zeroing + flag + last-exiter reset variant (for E8) | rel L2 vs E0's emulation, exactness, added kernel time, replay stability | rel L2 > 5e-3 at 7/9-way or any protocol failure -> fp32 workspace + convert node; added time > 1.5 us -> KS 2 geometries only (64-row CTAs) or fp32 scratch | whether the instruction that has never run on the judge is the epilogue; the fixed-cost number for s4 |
| E4 | day 1-2, box | the real kernel skeleton as PRB flags: producer WG (two lanes, split rings W 9 x 12 KB / x 7 x 16 KB) + two math WGs that only wait/arrive; rungs S0 W-only, S1 W+x; variants: combined 8 x 28 KB single-lane slot; x depth 5/7; W depth 7/9; EVICT hints on/off; x box {64,64} x 2 | kernel-us, clk/stage, B/clk/SM; clock64 stamps of empty-wait vs full-wait per CTA | S0 > F12 + 1.0 -> the barrier/ring/issue structure loses bytes in flight: fix before adding anything; S1 > S0 + 1.0 (box x cost was +0.9 in pw24h) -> delivery binds: depth, lanes, hints, box split in that order; S1 > F12 + 3.0 after all variants -> the geometry is delivery-limited: run E9's x-multicast variant before any decode work, and if that fails too fall back to m128-study (b) | which producer structure carries the kernel; whether 114 MB of ingest sits at the stream floor |
| E5 | day 2, box, 0.5 day (needs the consumer from T3 step 2) | L2-resident SM-side period probe: the full 12-bit consumer (3 LDS.128 + pw_unit + pw_fix3 + 4 RS wgmma per warp-stage, 2 WGs, wait_group 1, real producer) with the W pointer aliased so every CTA re-reads a 1.5 MB packed block (all L2 hits) plus the real x tiles; 32 stages; per-stage per-WG clock64 stamps (full-wait / LDS / decode / issue / wait_group / release) written per CTA; variants: wait_group 0 single-A, k128 merged, stage-parity offset, 1 WG control | clk/stage independent of HBM; the stamp breakdown; whether the two WGs run in lockstep | pass <= 650 clk/stage (model 576; H200 budget 662-746); 650-746 marginal: build on but expect only -2 to -3 us on the judge and keep the variants; > 746 after the variants -> this geometry cannot close on the H200 regardless of H100 microseconds: stamps say whether the port (TMA share) or the chain binds; fallback G1 (64x64 full-K RS, ~18 judge-us) or 256-row tiles with 4 math WGs sharing one x fill | the only H100 -> H200 transfer evidence; the decode-hiding hypothesis (claim 7) |
| E6 | day 2-3, box | HBM-fed rungs on all 8 shapes: S2 + decode into a register sink (no wgmma); S3 + RS wgmma (full mainloop, accumulator checksum); S4 + epilogue (memset node + reduce-add); S3b bf16 SS control; `check_implementation` on the 9 official m cases + one-hot probes + per-KS rel-L2 | kernel-us and clk/stage per rung; ptxas 0 spills, no C7510 | S2 > S1 + 0.5 -> chain exposed: LDS one more stage ahead, check the escape walk is uniform in SASS; S3 > S1 + 1.0 (box) after E5's best variant -> inspect DEPBAR placement, try k128 / wait_group 2 triple-A / ping-pong, and if still > S1 + 2.5 the 2-WG hiding claim is recorded as refuted and the design parks at its measured delta; S4 > S3 + 1.2 with both zeroing variants -> fp32 scratch or KS 2 only; S4 pooler-128 > F12 + 4.5 -> stop; bf16 control > cuBLAS + 2.5 on the box -> the skeleton is behind DeepGEMM grade: diff stage stamps against a direct DeepGEMM instantiation before continuing | the decode and wgmma rungs that every earlier ladder failed (+5..11); whether S4 reaches ~F12 + 3 (~20.5 kernel-us, ~-3.5 vs cuBLAS on the box) |
| E7 | day 3, box | full bake through `h100_bench.py --official` with the router picking per shape and the per-KS TOL; optional: reproduce pw9 w2n128a / pw10 w2c2 on this skeleton to account for 46.1 / 38.8 -> ~23 rung by rung | in-prepare and harness tables vs cuBLAS/torch per shape | GO when pooler m=128 <= 0.87 x cuBLAS in-prepare (<= ~24.0 vs 27.5) and no served shape > cuBLAS + 0.5 (route it otherwise); KILL the track if pooler m=128 > 0.95 x cuBLAS after E6 passes | go/no-go for the judge port |
| E8 | day 3-4, box, parallel to E9 | refinements, each gated: (a) in-kernel zeroing + release flag (from E3's variant) replacing the memset node: pass -0.6 us or better with 100+ replays clean; (b) SK grid (132-CTA unit walk, mid-range epilogue, unit-keyed escape cursors) at qkv/o first, then pooler: pass if qkv-128 improves >= 0.5 us and pooler-128 does not regress; (c) m=256 NT 256 (acc 128, W 7 x 12 / x 4 x 32) vs NT 128 x 2 tiles: ship only if >= 1.5 us under cuBLAS on the box at pooler/unpooler; (d) x multicast over a (2,1) row-tile pair only if E4 showed S1 ingest-bound: pass if S1 improves >= 1 us | box deltas, replay stability | each refinement drops on its own kill; none blocks E9 | the ~1.2 us of headroom over the 3 us bar; m=256 and qkv balance |
| E9 | day 4 | judge mechanics on the box before the first check: own load_inline extension with the submission's exact flag list, `__CUDA_ARCH_FEAT_SM90_ALL` guards, `cuTensorMapEncodeTiled` via the driver entry point, memset node captured in the graph, prepare <= 200 s incl. r6 118 s + router, upload < 1,000,000 B, lint, critic SASS pass (HGMMA RS count, UTMALDG/UTMASTG/UTMAREDG, no C7510, USETMAXREG pair); then ONE judge check (`okc check`) | judge table, 9/9 | pass: pooler m=128 <= 17.5 judge-us and 9/9; kill: judge delta vs cuBLAS < 1.5 us while the box showed >= 3 -> the H200's stream/port ratio differs from the model: re-read E5's stamps against the judge's clock and stop at the measured gain (ship it if it is a within-run win) | the board |

Days 5-6: second judge check with E8's refinements folded in; days 7-8: harden, final scored run. If E2 or E5 kills the
geometry by day 2, the remaining days go to the 'mx' CUTLASS two-plane track and pw30 (section 7).

---

## 6. First three build tasks (start now, in parallel, separate agents)

**T1 — Packer, lists and offline emulation (no GPU; owns E0).**
Deliverables in the new candidate dir: `pack128(w)` producing `[row-block][chunk][8 tiles x 1536 B]` from pw10's
`_pack_nat` + `_c12` (same lane fragment order, so `pw_dec`/`pw_unit` are unchanged), escape lists partitioned per
(row-block, KS range, in-block tile) with pw10's entry format and sentinel, plus the unit-keyed variant for the SK grid;
an `_unc12`-style bit-exact round-trip over all four layers; `emu_rs128.py`: KS-way bf16 TMA reduce rounding on the real
weights for the 8 shapes (random, 'outliers', one-hot) and the stmatrix.trans slab XOR addressing; a per-KS tolerance
table for the bake. Output: the pack/list functions, the emu script, the tolerance table, and the E0 verdict (bf16 reduce
or fp32 workspace).

**T2 — Calibration, microbenchmarks and the isolated epilogue on the box (owns E1, E2, E3).**
A `lab.py` in the pw28h style under `/root/gpu_lock.sh`: clock sampling; kineto capture of cuBLAS kernels for the 8
shapes; bare bulk stream F12/F16 in the design's exact 128 x 9 x 12 KB and 132 x unit patterns; memset node cost; empty
cluster-launch probe; the resident-smem RS/SS wgmma rate kernel with and without the concurrent 28/44 KB TMA fill + LDS;
`ptxas -v` of the NT 256 consumer stub under setmaxnreg 232; the reduce-add isolation kernel (STSM.trans +
`cp.reduce.async.bulk.tensor.add.bf16`, 4/7/9-way, one-hot, outliers, 100 replays) plus the in-kernel zeroing + flag
variant. Output: the bar table (F12, F16, cuBLAS kernel-us, memset, cluster launch), clk per RS instruction alone and
loaded, the reduce-add precision and cost, and the E2/E3 verdicts.

**T3 — The kernel skeleton with PRB rungs (owns E4, then E5/E6 as T1/T2 land).**
One templated kernel `rs128<NT, KS, RING(split|combined), WDEPTH, XDEPTH, PRB, ...>` in its own load_inline extension
with the exact submission flags and FEAT guards: producer WG with two elected lanes and split rings (combined-slot flag),
two math WGs, setmaxnreg 40/232, mbarrier protocol of s1.3, tensor maps via the driver entry point, clock64 stamp
buffer per CTA, host launch without a cluster attribute, h100_bench/lab integration. Step 1 (day 0-1): PRB 1 = S0 W-only,
PRB 2 = S1 W+x (waits/arrives only) and the E4 variant sweep. Step 2 (day 1-2): the math loop of s1.5 (pw10's `pw_unit`
+ `pw_fix3` + RS wgmma, A double-buffered with the "+r"/"+f" fences), PRB 3 = decode sink, PRB 4 = +wgmma, the W-alias
mode for E5, the variant flags (wait_group 0 single-A, k128, parity offset, 1 WG control). Step 3 (day 2-3): epilogue of
s1.6 with the memset node (bf16 reduce-add or fp32 per E0/E3), bf16 SS control switch, 8-shape bake table, per-KS TOL.

Shared interfaces to fix on day 0 so the three can proceed independently: the 12 KB unit layout and the per-warp LDS
address formula (T1 -> T3); the escape entry format and list partition (T1 -> T3); the out tensor-map box {64, NT} and
slab layout (T2 -> T3); the stamp buffer format (T3 -> T2's analysis scripts).

---

## 7. Relationship to the parallel CUTLASS track and to pw30

- pw30 (the four parity CUTLASS bf16 configs) goes to the judge independently to confirm H100 -> H200 transfer; it
  carries little score on its own (pooler m=256 -5.5%) and does not compete with this plan.
- 'mx' (two-plane 12-bit A operand inside `ws_64x64x64_c1x2_sA`): same byte cut, CUTLASS's proven pipeline, 64x64 SS
  geometry whose port (256 clk/stage) and L2 traffic (224 MB) are the ceilings pipeline-diff s3 identified; its win is
  bounded by those (parity-ish on the port, -2 to -4 on bytes). Decision point: at the end of day 3 both tracks report
  pooler m=128 in-prepare on the box against the same cuBLAS number; whichever is lower and passes its own correctness
  goes to the judge first on day 4, the other follows on days 5-6 if it is within 1 us. They do not share a packer (plane
  layouts differ) but share the bake, the router, the tolerance table and the judge-mechanics checklist.
