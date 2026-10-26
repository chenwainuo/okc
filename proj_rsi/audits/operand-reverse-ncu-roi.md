# Operand reverse: independent Ncu interpretation and ROI disposition

`state=DIAGNOSTIC_CLOSED; policy=CLOSED_NO_RETRY; new_GPU_runs=0; new_kernels=0; official_best_us=3200.615008; promotion=NONE`.

**Finding:** moving decoded W to shared memory did not remove the actual rq comparator's aggregate shared-byte workload. It added exactly **4,194,304 warp-level SHFL/SEL executions** for X register transposition, plus substantial generic loads/stores, address/control and synchronization work. Generic LSU/MIO pressure rises sharply; HBM and tensor utilization fall. This invalidates a simple “less shared traffic therefore faster” argument. The capture does **not** identify producer decode, X delivery, transpose dependency, or W refill as the dominant latency source: there are no PC stall samples, and barrier retry counts do not measure blocking time.

## Identity and comparability

Inputs, under `results/proj-rsi/takeover/20261026/`:

|artifact|SHA256|
|---|---|
|`operand-reverse01-ncu.raw.csv`|`1dc954fd57dd68c7911333ef712fcf29d961af6e7758adabad0e27adeb775408`|
|`operand-reverse01-ncu.source.csv`|`2db4f88564506894a054d1373fdd339db7fb5cc6ec8d9375c17fc4528f73a54d`|
|`operand-reverse01-ncu-driver.json`|`70241b3cb81527769fa0eb17796e4a43d527a12bb8bd08eada6bcdb620e1cf91`|

Supporting preserved artifacts: `.ncu-rep`, `.host.log`, earlier `-setup-invalid.log`. The invalid setup log is not a kernel measurement. Preregistration: [next-roi-after-takeover.md](next-roi-after-takeover.md), SHA256 `72f739d47e6e5f7e7e703b236fad5af0f7160ccc0f0a096aef34fb341bb6f0dc`.

Driver: `DIAGNOSTIC_COMPLETE`, `COUNTERS_ONLY_NO_SPEED_CLAIM`; source `ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; frozen pw36 parent `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Baseline then candidate, context1/stream7/process34, H10080GB/CC9.0. Both post-profiled-graph oracle checks and input immutability pass; baseline relative L2/worst row `2.686e-3/2.986e-3`, candidate `1.421e-4/2.633e-4`, both12probe rows exact.

|field|Ncu ID0 baseline|Ncu ID1 candidate|
|---|---|---|
|kernel|`rq<128,5,1,0,0,0,2>`|`rq<128,8,1,0,0,0,20>`|
|actual baseline pick|`rq/s4s5z/KS2`|reverse KS1|
|grid / block|`64x2 / 384`|`128x1 / 384`|
|useful shape|M128/N8192/K4096|same|
|perCTA channel/K slice|128/2048|64/4096|
|k64 stages/CTA|32|64|
|hardware duration, diagnostic only|25.088us|45.856us|
|registers/thread|168|168|
|dynamic shared B/CTA|144520|103600|
|allocated shared B/CTA|145664|104704|
|register / shared resident-CTA limits|1 / 1|1 / 1|
|replay passes / warmup passes|18 / 0|18 / 0|

These are **not** the original paired-graph baseline variant/times. Original `operand-reverse01` selected `s4s5hz/KS2`: complete-graph F1 `24.127999 -> 47.648001us`,21/21paired losses; its separate profiler calls were `21.472 -> 45.090us`. Do not use this Ncu report to decompose that21.472us. The Ncu baseline was freshly prepared under the profiler and picked `s4s5z`; setup bake timings include invalid negative values and are not performance evidence.

Clocks were not fixed by this diagnostic. Average GPC clock `1.77649 -> 1.81192GHz`; DRAM clock `2.61684 -> 2.61619GHz`. Replay, SASS patching, caches, different passes and blocking/polling schedules limit cross-metric exact arithmetic. The report is useful for work/pressure categories, not a benchmark replacement.

## Source execution accounting: measured fixed addition

The source CSV is two tables: `Kernel Name`, then `Address/Source/Instructions Executed`. Count1792 baseline PCs and1024 candidate PCs. Strip any `@predicate` before grouping opcode; retain all addresses, including zero counts. Source `inst_executed` counts warp-level execution and ignores instruction predicates; it does not count active lanes or predicated-on bytes. The export contains no PC stall samples. See [NVIDIA Source Metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#source-metrics).

|opcode group|baseline source count|candidate source count|delta|
|---|---:|---:|---:|
|SHFL, all|4323|2100963|2096640|
|SHFL.BFLY, transpose only|0|2097152|2097152|
|SEL, all|59696|2149936|2090240|
|SEL, candidate128transpose PCs|0|2097152|2097152|
|PRMT|524288|524288|0|
|LDS|98560|98304|-256|
|LDG|2390|280064|277674|
|STS|256|540672|540416|
|HGMMA|131072|262144|131072|
|WARPGROUP|66560|132096|65536|
|BAR|3840|36352|32512|
|FENCE|1792|33920|32128|
|MEMBAR|1408|33792|32384|
|YIELD|185599|0|-185599|
|SYNCS, all|302639|227584|-75055|
|BRA|342278|355438|13160|
|**sum of all source counts**|**4410325**|**11685448**|**7275123**|
|**raw HW `smsp__inst_executed.sum`**|**3846732**|**11646140**|**7799408**|

The subset rows overlap their parent SHFL/SEL rows and must not be summed twice. Candidate transpose is independently matched by both source algebra and the measured PCs:

```
128 CTAs * 8 math warps * 64 stages * (32 SHFL + 32 SEL)
= 4,194,304 warp-level executions.
candidate SHFL.BFLY: 2,097,152
candidate SEL: 128 PCs * 16,384 = 2,097,152
candidate other SEL: 16 escape PCs * 3,299 = 52,784
```

That fixed transpose work is **57.6527% of the source-table instruction delta**, or35.8934% of candidate source executions. It is not57.65% of time. Some issue/dependency work can overlap memory or producer waits. A large count establishes an implementation cost, not exposed critical-path cost.

The remaining source delta is3,080,819 executions and is not “all producer decode.” Uniform/address/control groups also grow (`ULOP3 +592896`, `USHF +409984`, `IMAD +393179`, `UMOV +211840`, `UIADD3 +192512`, `UISETP +155648`). Some producer/barrier polling work shrinks. PRMT's unchanged524288 executions and the unchanged lossless weight volume establish that operand reversal has not reduced the decode core's total work. Distribution changes from eight math warps to four producer warps; each producer warp handles twice the stages, which can still create a production bottleneck. These counts cannot resolve that bottleneck.

Twice as many HGMMA instructions is consistent with half-width n64 versus n128 and the same useful MACs, not twice the useful compute. The raw software metric `inst_executed` exactly matches the source-table totals for both kernels, confirming that this source export reconciles internally. Raw HW `smsp__inst_executed.sum` and summed source counters differ by563593 baseline /39308 candidate. Keep hardware and software measurements separate: source metrics are SASS-patched and the collection spans replay passes. Polling/blocking behavior can change between passes; the exact HW/software discrepancy is not established here. Do not divide the4.194M source addition by the7.799M HW delta or force an exact reconciliation.

### Wait-PC accounting does not establish time or backpressure direction

|SYNCS TRYWAIT group, source pass|baseline|candidate|
|---|---:|---:|
|math first full checks|32768|65536 (B-full)|
|producer first checks|6912 (W-ring empty)|32768 W-full +28672 B-empty|
|math slow-path retry checks|1712|15872|
|producer slow-path retry checks|185599|0 W-full +0 B-empty|
|all TRYWAIT counts|226991|142848|

Candidate math retry PCs `0x7d2a577b0f00/0f30/0f60/0f90`: `1320/2218/6478/5856`. Candidate first producer W-full/B-empty PCs `0x7d2a577b0240/02f0`: `32768/28672`; their slow retry PCs `0x7d2a577b0fc0/0ff0`: `0/0`. Baseline producer retry PCs `0x7d2919ec7140/7180`: `97842/87757`, matching its185599 YIELD executions.

**Do not infer “producer never waits” from zero retry counts.** The inherited `rq_wait` uses `mbarrier.try_wait.parity`, whose first execution can suspend until phase completion or a timeout. One execution can hide substantial wait time, and repeated executions measure incomplete observations rather than elapsed waiting. [NVIDIA PTX mbarrier wait semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait). Candidate math retries establish that some B-full observations were incomplete in that source pass; they do not quantify exposed delay or disprove simultaneous B-empty/W-full waiting elsewhere.

## Aggregate pressure: consistent signals, limited attribution

All percent values below use `% of peak sustained elapsed` unless a different denominator is named. Stall entries are the raw `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`, unit `inst`; they are not walltime percentages or PC samples.

|metric|baseline|candidate|
|---|---:|---:|
|DRAM throughput GB/s, decimal|2289.908|1194.038|
|DRAM active %|68.36|35.66|
|L2 throughput %|65.62|44.11|
|L2 sector hitrate %|26.88|61.48|
|L1TEX LSU wavefront activity %|25.79|59.74|
|L1TEX LSU input-request activity %|4.30|30.57|
|L1TEX LSU writeback activity %|24.63|41.89|
|SM LSU instruction activity %|2.07|28.18|
|SM MIO instruction issue activity %|2.31|15.40|
|SM MIO read/write queue activity %, each|2.53|25.46|
|SM tensor pipe activity %|35.65|19.12|
|SM ALU pipe activity %|13.15|17.17|
|SM issue active %|16.39|26.56|
|SMSP issue active % of active cycles|20.71|29.97|
|eligible warps/SMSP/active cycle|0.26|0.39|
|active warps/SM/active cycle|9.89|11.81|
|long-scoreboard / issue-active ratio|6.51|3.00|
|short-scoreboard / issue-active ratio|0.91|2.13|
|MIO throttle / issue-active ratio|0.01|0.95|
|LG throttle / issue-active ratio|0|0.36|
|barrier / issue-active ratio|0.17|0.37|
|GMMA / issue-active ratio|0.73|0.11|

1. **Confirmed:** much more generic LSU/MIO work and pressure, plus a large measured X-transpose instruction addition. Candidate direct X LDG/transpose and decoded-W scalar STS are both relevant source categories. The aggregate counters do not distinguish their causal shares. MIO activity cannot automatically be labeled “decode.”
2. **Unsupported as immediate capacity explanations:** HBM bandwidth saturation and tensor throughput saturation. Both fall substantially in the slower candidate. This does not rule out W-memory latency, coupled refill, a late producer warp or X dependency at low bandwidth. ALU percent17.17 also cannot rule out a serial producer escape tail.
3. **No occupancy recovery:** both remain one resident CTA/SM,168regs/thread. Candidate's smaller shared allocation does not buy another CTA. Active warps rise toward the12warps/CTA limit; longer-lived producer warps and more issued work are not useful throughput gains.
4. **Do not invert denominator-dependent ratios:** lower long-scoreboard or GMMA ratio is not evidence that X loads or WGMMA waits are cheaper. Candidate issues far more instructions, changes role lifetimes and runs longer. Higher short-scoreboard/MIO is consistent with transpose/shared/dependency pressure, but has no PC attribution. [NVIDIA warp stall reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).
5. **No complete memory-byte/PC dataset:** CSV has604columns but no PC-sampling metrics and no raw DRAM-byte/L2-sector/L1-sector sum counters. It cannot provide exact per-role byte counts, sector efficiency, bank-conflict causality or PC-local stall fractions. Source export is execution counts only.

Two approximate consistency checks, **not exact decompositions**:

```
dram rate * gpu duration:
  baseline 57,449,216 B; candidate 54,753,792 B
gpc elapsed cycles * tensor-active fraction:
  baseline ~15,888.7; candidate ~15,886.3 cycle-equivalents
```

The first is consistent with broadly similar HBM volume despite doubled X delivery demand; it is not a substitute for raw read-byte counters and inherits replay/domain mismatch. The second is consistent with the same useful tensor work spread over more elapsed cycles; it combines GPC and SM metrics and is only a proxy. Neither turns45.856us into an exact additive latency budget.

## Actual traffic model versus the discarded premise

Both profiles execute128CTAs and decode48MiB packedW. Baseline X delivery is64MiB; candidate128MiB, although the underlying input X is only1MiB and repeated loads can hit L2. Higher L2 hitrate therefore does not prove lower L2-to-SM delivery cost.

PerCTA-stage logical shared traffic is72KiB baseline /36KiB candidate, but stages are32/64: both total2304KiB/CTA, **288MiB across the launch**. Candidate shifts64MiB of shared writes from asynchronous TMA to generic decoded-W STS. The old60->36KiB stage comparison used pw19 at a common64channel tile, not this measured rq baseline. Generic LSU counters omit other shared clients; this model does not assert a shared-port bandwidth ceiling.

The reversed policy additionally moves W refill after decode/store/publish, adds a128thread producer barrier per stage, doubles math rounds, and introduces four vector X loads plus32SHFL/32SEL per math warp/stage. Each is real work; their exposed latency is coupled, so their hypothetical savings cannot be added.

## Preregistered hypothesis disposition

|hypothesis|what current data supports|disposition|
|---|---|---|
|H_P producer production underfeeds math|Math has B-full retries; generic STS/control/producer work rises|**Unresolved dominance.** First-call blocking hides both W-full and B-empty wait time; no role-specific stall evidence|
|H_D specifically decode/escape|Core PRMT total unchanged, half as many decoder warps and2xstages/warp|**No decode-specific attribution.** Could be serialized production, but STS/publish/refill and X remain confounders|
|H_XL X global/L2 delivery|X source demand doubles; LDG/LSU/LG activity rises and L2 hitrate rises|**Credible added cost, unresolved exposure.** No X-consumer PC samples or raw sectors|
|H_XT X register transpose|Exactly4.194M new fixed SHFL/SEL; short-scoreboard/MIO rises|**Strongest established added-work category.** Critical-path dominance unproved without PC/dependency exposure|
|H_W W-ring/refill coupling|Refill after production; lower HBM/TMA rate|**Unresolved.** Zero W-full retries cannot reject first-call blocking; low rates cannot distinguish starvation from backpressure|
|H_C round synchronization/scheduling|HGMMA/WARPGROUP count2x, generic barrier/fence additions|**Confirmed extra round/control work, unresolved exposure.** No temporal/PC attribution|

This capture falsifies the strongest simple shared-byte/HBM/tensor-capacity explanation. It does not choose one latency hypothesis by treating absent PC data as zero stalls. The source count substantially strengthens H_XT as a cost to avoid; it does not license another reverse-policy variant.

## ROI and next admissible action

1. **Close this diagnostic and retain the reverse policy closure.** No retuning, geometry/KS sweep or follow-up GPU profile is requested. Correctness passed; the complete benchmark loss remains decisive. There is no measured positive speedup to project.
2. **CPU design constraint for any independently proposed mechanism:** preserve the actual comparator's X reuse and useful tensor work, and count newly introduced transpose/LDG/STS/barrier/control instructions against that comparator. A new W-multicast idea cannot claim an inferred20us benefit from these data or treat global HBM saturation as established. It needs its own byte/issue/convergence/fallback proof and source novelty review; these counters alone do not admit it to GPU.
3. **m16 remains NO-GO pending evidence.** This M128 reverse capture says nothing direct about the actual m16 GV X-consumer PCs. Do not transplant its stall ratios or launch geometry to GV. The preregistered actual-GV evidence requirements remain unchanged, and no new m16 prototype is admitted.

Same-official-job accounting stays in the preregistration: pw36 job`e22a4b9d693d4902`, score3212.415452, m128 unpooler dispatch **Nvjet**,19.103us/benchmark. This Ncu rq is not that official dispatch. Conditional genuine verify16 load gain0.5us on that job would project3208.316002us (reduction4.099450); genuine decode16 total gain0.6us would project3205.849892us (reduction6.565560). Neither gain has been demonstrated. Returning the reverse candidate to rq parity would be zero gain versus rq, and still no win versus official Nvjet. Best separate rs32 job remains**3200.615008us**; no `<3000us` result or policy promotion follows from this diagnostic.

Reviewer actions: local read/CSV arithmetic, official-doc checks, this audit only; no source changes, compile, remote jobs, GPU launches or profiler reruns.
