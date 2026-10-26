# Next ROI after takeover — preregistered counter decisions, no new prototype

State: `GPU_BY_THIS_REVIEWER=NONE; NEW_KERNELS=NONE; m16_IMPLEMENTATION=NO_GO_PENDING_COUNTER_EVIDENCE; operand_reverse=CLOSED_POLICY_DIAGNOSTIC_ONLY`.

Scope: independent source/profile interpretation. Root is collecting one separate baseline/candidate2kernel NVTX diagnostic with Ncu2025.1. No additional launch, kernel variation, profiler sweep or reopening is requested here. No local/remote implementation edits. Aligned-epilogue NO-GO remains recorded and its artifacts preserved.

## Same official job, actual dispatch and score arithmetic

Use ONLY pw36 official job`e22a4b9d693d4902`, source`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`, score3212.415452073451us,9/9passed, CI1.4812..1.5263. Raw input: `candidates/pw36/profiles/submission.e22a4b9d693d4902.json`. Best rs32 score3200.615 is a different job and is NOT spliced into this job's projection.

Exact benchmark load sums from `benchmarks[].median_ns` (not profiled kernel times):

|load|ours us|
|---|---:|
|decode1|36.425574000922|
|decode16|49.178875270804|
|decode32|53.037878608383|
|verify1|37.840604464743|
|verify16|65.510400061208|
|verify32|85.080609634965|

Holding THIS job's six SGLang load totals fixed, `S_new=S_old*exp(sum_load(log((T_load-delta_load)/T_load))/6)`. Deltas must be genuine complete benchmark-shape gains on this job's device/dispatch; GPU profile durations, local rq timings and changes in a different job are not interchangeable.

|conditional genuine change, all else fixed|new score us|score reduction us|
|---|---:|---:|
|decode16 total−0.3 (one m16 layer)|3209.141067|3.274385|
|decode16 total−0.6 (both target layers−0.3)|3205.849892|6.565560|
|decode16 total−2.0 (both−1.0)|3190.263375|22.152077|
|decode16 total−3.0|3178.892434|33.523018|
|verify16 genuine total−0.5|3208.316002|4.099450|
|verify16 genuine total−2.0|3195.857986|16.557466|
|decode16−3 ANDverify16−3 jointly|3154.153589|58.261863|

These are arithmetic scenarios, NOT predictions or addable wins. Reaching3000 from this job using decode16 alone requires16.556477us of genuine load reduction; verify16 alone22.054621us. No "all floors at once" plan is credible from current evidence. Reported SOLs assume the task's reference byte model; they are not strict floors for a12bit packed stream.

Official m16 pooler is `gv<1,2,2,2,8,16,1,1,0,1>` (CUPTI14.6us); unpooler `gv<1,2,2,1,4,16,1,1,2,1>` (14.5us). Both are one node, FMT1 lossless12bit, S1/no crossCTA reduction. Small-m rq bake kept r6/GV for these shapes. Official m128 unpooler is one Nvjet kernel,18.6us/profile and19.103us/benchmark, NOT the local operand-reverse trial's rq comparator. There is no removable extra graph node on any of these three official paths.

## 1. m16 X delivery: no admitted material novel implementation

Read frozen pw36 GVCU, `audits/{r6-lineage,p2-h100,p4-decode-pipe}`, handover10 closures and actual selected SASS from aligned_epilogue's never-launched local baseline compile. Preserved baseline sources only.

Current pooler: WR2/WK8/KU16/XS0,128CTA×512threads,128regs. Two row-warps share X through L1 and use a register ring. Current unpooler: WR1/WK4/KU16/XS2,512CTA×128threads,110regs; loads X at use through L1. W bypasses L1 allocation. Each layer streams48MiB packedW; nominal X payload delivered to row tiles is32MiB pooler /64MiB unpooler at m16, with cache/sector requests needing measurement. Raw fresh X is only256KiB/128KiB; a high L2 hit rate is expected and does NOT establish cheap L2->SM delivery.

Already closed: XS/cache policy/L1carveout changes as H200 lever; generic shared-X slab and TMA/bulk staging; wide N tiles requiring crossCTA/cluster reduction; deeper W rings and occupancy/geometry families; codecE; request-priority rtk gate; aligned epilogue's2add removal without bottleneck evidence. Do not relabel these as new X-delivery fixes.

One source ordering variation was checked for novelty: prioritize first X requests over initial W prefetch. It is NOT admitted or queued. Source `ld(s,u)` puts W before X in XS0, but actual SASS already hoists multiple X chunks before the first HMMA. XS2 steady-state issues X at use before the future W refill; only its initial prologue has nine W vector loads before first X vectors. C++ reorder alone need not change SASS. This cannot currently be tied to>=0.3us complete-shape savings; it risks reducing useful W outstanding work, as the closed rtk priority gate demonstrated in a different kernel. This is an assessed/unadmitted minor variation, not a proposed new kernel or revived priority sweep.

**Decision: NO-GO for a new m16 prototype.** No credible materially new implementation clears the source/history/ROI gate. If later measurement shows a material exposure, return to source with that evidence; do not guess a pipeline now.

Required evidence before that decision can change (not a request for another GPU run):

- Source/PC-correlated `long_scoreboard` on instructions consuming X LDG registers, with low eligible-warps/issue rate and substantial exposure. Attribute W `.NA.128` loads, escape prefix and X `.128` separately. Attributing a generic LOP3/HMMA stall to decode is invalid when its input originated in an outstanding LDG.
- Global request/sector efficiency, L1 hit rate, L2->SM traffic and DRAM bytes/bandwidth. High long-scoreboard with low bandwidth indicates latency/dependency; high L2/LSU utilization and excessive sectors indicates delivery/issue. DRAM saturation with expected packed bytes and healthy issue means no source-only X fix established.
- `lg_throttle`/LSU issue and L1TEX active wavefronts distinguish a full request queue from a waiting operand. Shared-memory stalls on the reduction tail are a different mechanism. PC sampling must separate hot mainloop from the final WK reduction.
- Baseline W/X payloads, actual registers/occupancy, no local spills; role of compiler-hoisted future X loads. A useful lever must remove>=0.3us exposed work without increasing bytes/register spills or sacrificing the W stream. Counter fractions alone are not time savings.

The known X+decode cost is not divisible into clean independent gains: P2's qkv m32 x-only rung changed with grid, and its real-grid decode increment was only0.22us. These are sibling-shape evidence, not direct m16 bounds. Do not apply their larger pathological-grid numbers to pooler/unpooler.

## 2. Operand_reverse: preregistered independent falsifiers

The policy is CLOSED. Complete graph F1:47.648001us candidate vs24.127999us baseline, loss23.520002;21/21paired losses. Separate profiler calls: candidate45.090us vs actual prepared rq21.472us, gap23.618us. Correctness and actual timed-graph mutations passed. The current two-kernel Ncu capture diagnoses failure; it does not grant tuning/retry/KS/m256 changes.

The candidate must remove about24us of its present overhead just to beat that local rq by0.5us. Recovering45->21.47 is merely parity, not a20+us official score win. The official job selected a different m128 unpooler kernel, so even local rq parity would not prove official improvement.

### Compare against actual rq, not only pw19

Both128CTAs and same useful M128×N8192×K4096 MACs; packedW48MiB in both. Baseline rq:128channels×K2048/CTA,32k64stages, eight math warps decode; candidate64channels×K4096/CTA,64stages, four producer warps decode. Each candidate producer warp traverses2×the W stages; the candidate also doubles X's global/L2->SM payload demand64->128MiB. DRAM need not double because the same1MiB X can be served repeatedly from L2.

Nominal logical shared traffic, including fills, generic LDS/STS and tensor operand reads:

|perCTA k64 stage|actual rq128channel/KS2|reverse64channel/KS1|
|---|---:|---:|
|packedW fill|12KiB|6KiB|
|packedW LDS|12KiB|6KiB|
|X shared fill|16KiB|0|
|decodedW generic STS|0|8KiB|
|tensor shared operand reads|32KiB X|16KiB W|
|total/stage|72KiB|36KiB|
|stages/CTA|32|64|
|total/CTA|2304KiB|2304KiB|
|total all128CTAs|288MiB|288MiB|

Thus60->36KiB is a comparison with closed pw19 at a common64channel tile; it is NOT a40% aggregate shared-byte reduction versus the measured rq baseline. Candidate moves64MiB of shared writes from asynchronous TMA to generic producer STS while keeping nominal aggregate reads/writes equal. Different engines, granularity, port arbitration and overlap still matter; the byte model is not a hardware port ceiling.

Additional source costs: producer executes16scalar STS/thread/stage plus proxy fence and128thread barrier before publishing B; W refill occurs only after this decode/store/publish path. Candidate has64producer barriers versus no matching baseline producer-WG decode barrier. Math executes four coalesced X vector loads/thread/stage and32SHFL+32SELP to form register-A; it also doubles fence/commit/wait rounds perCTA (64 vs32), with half-width n64WGMMA instead ofn128. No spills/occupancy failure was measured. These distinctions must be tested rather than summing every source operation into a time claim.

### Hypothesis matrix: evidence needed and rejection patterns

PC groups MUST distinguish producer(warp8..11) from math(warp0..7), and producer W-full from B-empty wait and math B-full wait. Use generated frozen source + SASS CFG/call-site addresses: inlined `rq_wait` source names alone can merge semantically different barriers. Normalize counts per useful MAC or decoded W byte, not perstage, since stage count differs2×.

|hypothesis|predicted pattern|pattern rejecting it as DOMINANT cause|
|---|---|---|
|H_P: producer production throughput (decode/escapes/STS/publish) underfeeds math|Math waits on B-full; producer has little B-empty backpressure and spends its nonwaiting PCs in decode/escape, STS or proxy/bar3; tensor gaps; W-full usually ready|Producer mostly waits B-empty while math is active on X LDG/transpose; producer nonwait instruction/STS pressure small. This rejects producer production as leader even if pooled barrier% is high|
|H_D: specifically decode/escape work inside H_P|Producer integer/decode/escape PCs carry issue/dependency pressure; producer barrier tails coincide with slower escape warp; STS/MIO not dominant|Producer waiting on W-full or spending dominant productive pressure on scalar STS/proxy fence/bar3, with decode PCs minor. Do not rename these costs "decode"|
|H_XL: X global/L2 delivery latency/issue|Math long-scoreboard at consumers of X LDG and/or LG queue throttle; L2->SM/global request demand rises as model predicts; producer B-empty backpressure; low math eligible issue/tensor gaps|Math X result consumers have little exposed long-scoreboard/LG pressure, math predominantly waits B-full, producer does not wait B-empty. High L2 hit rate alone does not reject H_XL|
|H_XT: register transpose/dependency issue after X lands|Math hot SHFL/SELP/fence chain has short-scoreboard/MIO or math-pipe issue pressure and low eligible issue; X load wait relatively small; producer B-empty wait|Transpose PCs have little exposure while math waits B-full or X global results. A large SHFL instruction count alone does not prove it is critical|
|H_W: W ring/refill starvation or coupling|Producer W-full wait dominates, W refill serialized after production; DRAM/TMA request behavior/inflight below sustained stream expectation; math B-full waits|W-full mostly ready while decode/STS or consumer backpressure dominates; sustained W payload rate healthy. This is distinct from H_D|
|H_C: perround synchronization/tensor scheduling|Math waits at WGMMA fence/wait/dependency or producer publish-barrier without dominant X load/transpose exposure; actual tensor issue/active cycles and doubled round count explain gaps|Tensor/round PCs not exposed and H_P/H_X gives consistent opposing backpressure. No C75 at compile time does not prove optimal runtime grouping|

Do not turn warp sample fractions into microseconds by multiplying wall duration: stalls overlap, faster roles legitimately wait, and samples are attributed to the waiting CONSUMER PC. Aggregate low integer throughput does not by itself reject H_D when one warp's escape tail holds a128thread barrier. Mixed B-full/B-empty waits can be pipeline startup/tail or alternating backpressure; without enough PC samples/temporal evidence, classify INCONCLUSIVE.

### Required Ncu data and interpretation limits

Use available Nsight sections/metrics; root must query exact spellings supported by Ncu2025.1/H100. Examples are a schema, not permission to add profiler passes blindly:

- Launch/resources + duration: `gpu__time_duration.sum`, registers/shared/occupancy/local-spill stats; SM/memory clock settings and cache/replay policy. Compare the TWO captured kernels in the same collection context. Ncu replay duration is diagnostic, never new graph selection timing.
- Scheduler/tensor: eligible and active warps, issue-active, tensor-pipe active cycles, integer/other execution mix. Tensor utilization low alone cannot choose H_P vs H_X.
- Memory: DRAM read bytes/throughput, L2 read sectors/throughput and hitrate; L1TEX global requests/sectors/sector efficiency; shared load/store wavefronts and bank conflicts. Generic-LSU shared counters do NOT include all TMA/tensor shared traffic; no128B/clk port claim follows from them alone.
- SourceCounters/PC samples with actual sample counts: `smsp__pcsamp_warps_issue_stalled_{long_scoreboard,short_scoreboard,lg_throttle,mio_throttle,math_pipe_throttle,barrier,membar,wait,not_selected}` where supported; correlate with role-specific source/SASS. `not_selected` is a scheduler state, not wasted-stall time. Missing/insufficient PC data means no attribution, not zero cost.

Primary interpretation source: [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons). Its replay/cache/clock controls mean collection mode must be preserved in interpretation; a warm profiled call is not the cold paired graph benchmark. The pending separate NVTX diagnostic must not include tuning loops/full prepare in the profiled range or count a fallback kernel as the candidate.

## Ranked actions / admission state

1. **Already authorized, highest information ROI:** independently harvest root's ONE rq/reverse Ncu capture and evaluate the matrix above. No parameter change. It can close a false architecture premise or identify the real cost category; it earns NO speedup by itself. Keep reverse policy closed regardless of interesting counters.
2. **Conditional measurement topic, no new kernel queue:** m16 pooler/unpooler X delivery only if suitable actual-kernel PC/memory evidence becomes available. Potential material score slope is~10.9score-us per genuine decode16 load-us on this job, but attainable delta unknown. No admitted untested mechanism meets the0.3us bar today.
3. **Not queued:** footer, aligned scale, router diagonal folding, runtime patch table, streamK, request-priority gates, generic rewrite, all-floor/m256 promises. Preserve their closed findings; do not occupy an agent/GPU with an unfalsified version of the same policy.

Next decision is counter interpretation, not implementation. If the pending capture cannot distinguish causes, record what is missing and stop; root decides whether further evidence has enough expected value. This reviewer remains available for independent interpretation when raw counters arrive.
