# router-small01 independent preflight

Reviewer `/root/proj_evidence`, 2026-10-26 UTC. Local source/CPU/device compilation only; no GPU launch, remote build/job, source edit or commit. Full target remains official<3000us; this screen concerns one m8 router mechanism.

## Decision: GO exactly one bounded frozen trial

Kernel `042bfbbd2d0916f3bc33a3ff11c3ec1aac64bc97efc27dc2194b21269bf78f5a`, lab `069d214589a43177db4c633f1759961c56aa19927d984d482635a48e85fe8ba4`, prototype `bcbd63b4f6256461610bf0336ad6bc24a4664b0338cb485090b89ab0c79b9ac0`, packing `bcd7c27092bb121a67fd1e1be6a1809e8b21041361973e7364f53f0c359ef77c`; parent pw36 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Final SOURCE_MANIFEST fixes these bytes. Earlier kernelb974... was observed before host dtype/device/alignment guards and occupancy helper were finalized; no launch approved for that provisional hash.

No device math/packing/barrier/bounds/DCE blocker found. Full CUDA-Torch host build on the actual image and runtime<=64register/local0/occupancy>=1 are mandatory before any timing. Root exclusively owns named `h100_trial.sh` handoff: inner300s/outer360s, after the CPU-only lock test finishes. No geometry/codec/cohort sweep, m16/32 expansion, fallback surrogate, automatic port or judge slot. Survive only >=0.5us in **both** prepare-F1 and median paired graph gain, with all eager/actual-timed-graph checks complete on both sides. Failure closes this fixed policy.

## Independent math and all-K coverage

Checked [NVIDIA PTX BF16 m16n8k16 fragment definition](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float), CPU emulator and C++ loads/selectors independently.

For lane=(g<<2)+t, B column=g encodes group=floor(g/2), expert=g%2; its two packed registers cover k=2t+{0,1,8,9}. Packed index uses expert `(lane>>2)&1` and group `lane>>3`, matching those coordinates. A registers0/2 carry rowg and registers1/3 rowg+8; their X loads select neighboring K groups within cohort0/2. Thus d0 contains A groups0/1 and d1 groups2/3 while B columns span all4groups/two experts.

After synchronous tensor accumulation, C columns2t+expert belong to group t. Choosing d0/d1 by t<2 and C rowg/g+8 by t parity keeps the matching K diagonal. XOR1 then XOR2 sum exactly the four t groups within each fixed-g quartet, without crossing tokens. One t0 writer/warp/token/expert produces32 warp partials; final sum covers8192K=32warps*4groups*4steps*16 exactly once per expert. Both hi and residual-lo BF16 products enter FP32 accumulators before the same reduction. Off-diagonal products are discarded intentionally; no needed product is omitted.

For m5/7, `min(g,m-1)` duplicates only inactive token rows; quartet shuffles keep g fixed and final tok<m guard discards duplicates. No inactive value leaks into active rows. Shape1..8 pointer arithmetic remains in bounds, though this trial/proposed dispatch is m8 with nearby5/7 validation.

Independent rerun CPU proof passes:4194304 BF16 hi/lo values bit-exact,2097152 direct indices bijective per plane,65536 possible BF16 encodings through wide fallback, five adversarial narrow base windows, symbolic all8192K basis coverage. Dense PTX-coordinate matrix emulation relativeL2=2.4088523e-18 versus X@(hi+lo)^T; deliberately wrong cohort/diagonal/B-group controls give errors1.40/1.37/1.38. This proves mapping/reconstruction, not GPU rounding/performance. Runtime strict reference is original FP32 W, not merely reconstructed hi+lo.

## Lossless codec, bounds and shared reduction

Each warp-plane has512 BF16 values and fixed1024byte allocation. Narrow format retains low8, high7 offset3 and sign1; base+offset<=127 avoids cross-byte carry. It reads first512+256bytes. If high7 range>7, entire warp-plane is wide and preserves raw16bits across512/256/256byte segments. Decode's PRMT/wide pairs match the independent physical decoder. Wide flag/header is shared by a whole warp; no per-lane stream, approximation, escape walk or output-derived cache exists.

Header4096*4bytes and two-plane4096*2048bytes bound every fixed address. Largest extra load ends at its plane boundary. X loads stay within selected token's8192K; 32bit BF16-pair loads are4byte aligned. Host checks CUDA/device/dtype/shape/contiguity, packed16byte and X4byte alignment. Separate output is float32[M,256], each CTA stores only its own two experts; maximum output index isM*256-1. No global atomic/zero flag/workspace or cross-CTA ordering.

Only t0 writes red[warp][g][expert], a disjoint2KiB shared allocation fully populated before the unconditional1024-thread barrier. All threads execute every MMA and reach that barrier; format branches are warp-uniform. Only first16 threads reduce32 initialized partials, then overwrite distinct outputs. No shared read races a writer, and no CTA waits for another CTA's scheduling. Source X is const and only load instructions reference it; output bounds/independent allocation exclude input aliasing in this lab.

## Compile/resource and DCE gates

Reviewer independently compiled current042b kernel with local NVCC12.8 sm90a/O3:62registers,0stack/spills,2048B shared,1barrier, no warnings/C75/LDL/STL. SASS retains16HMMA.16816.F32.BF16,1CTA barrier,1output store site; both cohorts and both planes remain in observable outputs. Full accumulator reduction is the sink, not D[0].

1024threads*62registers=63488 before allocation granularity; rounding toward64registers/thread reaches the64KiB SM register budget, so one CTA/SM is expected. Grid128 can occupy128 of132 H100 SMs. This is resource feasibility, not bandwidth proof; actual Torch macro compile may differ and prototype.prepare checks actual regs/local/occupancy before launch. Dynamic shared opt-in is unnecessary for this2KiB static shared allocation. No register redistribution/async A lifetime applies to synchronous mma.sync.

CPU pilot logical transport6617344bytes includes narrow/wide rates; runtime must round-trip its own actual GPU-generated weights and record its own packing rates. Logical transported payload is not a measured DRAM-byte counter, and allocated8404992bytes differ from requested payload. No H200 bandwidth/score prediction follows.

## Final lab review

Actual frozen pw36 is loaded by SHA and fully prepared without disabling flags/bakes. Scalar-only selected route/instance metadata avoids the earlier Lt lab's tensor-string logging. Comparator calls actual `kernel_fn`; prototype calls actual allocating wrapper. Both call structures are graph-captured with output tensors retained.

Before timing, **both** sides get strict original FP32/FP64 L2<=2e-5/row<=1e-4, exact hi+lo probe rows, normal/outlier/zero at m5/7/8, and literal all8192 basis columns x8 (1024 launches/side). Zero outputs exact. Source/CPU reconstruction is not substituted for these CUDA gates.

After timing, **both actual measured graphs** have output poisoned NaN and X changed to original/zero/fresh-normal/fresh-outlier/original-again, three replays each, then task/strict FP64/probe oracle. This repairs earlier finite-only baseline evidence. Concrete limitation: lab does not explicitly assert X unchanged after these calls; source audit proves this prototype only loads X, and frozen parent is already validated, so it is not a bounded-run blocker. Do not claim an executed input-immutability assertion that the script does not contain.

Prepare timer is inherited `_time_us` on actual calls; baseline pre/post min is F1.40 paired cold256MiB read-flush samples alternate AB/BA, preserve both aligned arrays, and decide by median paired absolute gain. Median-difference is diagnostic only. Flush/event/replay are stream-ordered; CPU packing/build/validation are outside timing. No profiler time or approximate model chooses a winner.

`baseline_prepare_s` subtracts build elapsed from start-to-prepare completion and therefore includes CPU packing/device transfer as well as baseline prepare; treat it as setup context rather than pure baseline prepare duration. Actual PREP stdout is the phase timing authority. Resource recheck after timing retains the gate. Frozen file identity comes from root-staged manifest/recorded kernel SHA; the lab itself does not compare every file hash to the manifest.

Admit only terminal JSON `PASS_REVIEW_REQUIRED` or `CLOSED_GAIN_BAR_FAILED` with both complete eager groups, both5-case actual-graph checks,40finite positive aligned samples/side, correct resource fields, real parent identity and exit0. A `STARTED`/`FAILED`/partial result, launch refusal, skipped validation, spill, source mismatch, timeout or CUDA error cannot be performance evidence. Root confirms named container/compute-PID disappearance on every exit. No automatic retry; any survivor requires independent raw review and transfer assessment before an isolated production integration.

## Runtime review

COMPLETE: independent review below closes this fixed prototype. No score gain or promotion.

## router-small01 terminal independent review: CLOSED_GAIN_BAR_FAILED

Root PID354126, container `okc-proj-router-small01`, inner300s/outer360s, terminal2026-10-26T21:30:01Z. Host log has `TRIAL_EXIT okc-proj-router-small01 0`; JSON state CLOSED_GAIN_BAR_FAILED,elapsed80.843s. Root reports container absent/GPU0MiB. Reviewer independently parsed copied full JSON/log, verified identities, all numerical/replay groups,40finite positive aligned samples/side, and recomputed both gates. No error/CUDA/C75/skipped-budget lines. Real Torch build passes62register/local0/occupancy1/2KiB; all parent phases complete in46s.

Actual frozen baseline is sel instance85, kernel descriptor`2.1.1.1.8.1.16.1.0.134217728`,N256. This is the real prepared pw36 small-router pick, not rr/r6/torch substitution.

|metric|baseline|diagonal prototype|gain (baseline-candidate)|admit>=0.5us|
|---|---:|---:|---:|---|
|prepare F1 us|7.872 (pre7.872/post7.904)|10.528|-2.656|NO|
|paired graph marginal medians us|7.840|10.496|median paired difference-2.592|NO|

All40 paired calls lose. Median paired difference differs from difference of marginal medians; the independent recomputation correctly uses aligned per-pair deltas and agrees with JSON to<1e-9us. No extra profiler or score-derived time is mixed into the comparison.

Both sides pass literal all8192onehot columns exactly against hi+lo, plus9 eager cases each (m5/7/8 x normal/outlier/zero). Both actual measured outputs pass5 mutated-input cases after timing with NaN poisoning and three replays/case; oracle checks occur after each three-replay sequence (not independently after every replay). Thus18 eager and10 actual-graph oracle records are present, plus both all-K checks. Max recorded L2/row: baseline2.541607e-6/2.891380e-6, prototype2.537191e-6/2.868036e-6, all below2e-5/1e-4. Original task probes and zero exactness pass. The preflight X-immutability-assertion limitation is retained; no nonexistent runtime assertion is claimed.

Actual logical transport6619904bytes (hi269wide/4096,lo950wide/4096), allocated8404992bytes; both hi/lo bit roundtrips exact. Reduced logical payload did not overcome this fixed policy's other costs. It does not prove a particular stall cause or authorize geometry/format/cohort tuning.

**Close this one full-K diagonal-folding/12-or16bit warp transport/m8 policy.** No port, sweep, retry, wider-M expansion, official slot or incumbent change. Preserve numerical/index proof and negative performance evidence. This measured loss does not prove every possible folding or lossless small-router design is impossible. Source/results README closure is documentation only; kernel/lab/packing/prototype/manifest remain frozen.

Raw local artifacts `results/proj-rsi/takeover/20261026/router-small01.{json,host.log}`.

- JSON SHA256 `b64bb818f4f61319b16bcd4707b144a55254f3d0e989214057769f4e9f96c7f5`.
- Host log SHA256 `66bdc73c2b216ca776fe3760fc62abfe7cdae3d2dbca006ab5a3cc526b797539`.
- Kernel042bfbbd... and lab069d2145... match the preflight freeze above; parent remains2a067059....
