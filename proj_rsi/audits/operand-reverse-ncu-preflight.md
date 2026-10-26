# operand-reverse NCU diagnostic preflight — 2026-10-26

`decision=VALID_COMPLETED_DIAGNOSTIC; candidate_v1=FAILED_GAIN_BAR; performance_admission=NO; counter_evidence=VALID_FOR_EXPORTED_TWO_KERNELS_ONLY; family=ITERATE_FROM_DIAGNOSED_MECHANISM`

Later user policy clarification supersedes any family-wide reading of the historical stop language below: preserve the failed v1 and diagnostic evidence, then review material successors. Direct native-A loads in v2 address the observed X shuffle/select exposure; they require independent source proof and unprofiled validation/measurement. No additional old-v1 GPU profiles are needed.

Scope: diagnose the already closed unpooler128 operand-reversal source against one freshly prepared frozen pw36 actual callable. No kernel/lab/proof/manifest changes, timing admission, parameter variants, implementation port, official submission or incumbent change. Reviewer launched no GPU job and edited only this audit.

## Frozen artifacts and ownership

- Diagnostic driver `proj_rsi/tools/operand_reverse_ncu.py`, SHA256 `89f76f0bd49d9a3ac9a30804476a3488f97f71f351351b6b79a9bae953ad7709`.
- Container runner `proj_rsi/tools/h100_container.sh`, SHA256 `8b3529b5fa873e61e4e236e0cbcdaa0a94ad426a9a21b6aa6469af4bf6e29d67`.
- Host deadline `proj_rsi/tools/h100_trial.sh`, SHA256 `674036ba22edcc3c64dda2677271c58f27fe822339cf132c4f36ce58837ee9e5`.
- Candidate generated source `ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; pw36 baseline `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Driver asserts both identities before computation. Reviewer separately verified all executable/proof entries in the candidate manifest remain unchanged; README closure is intentional metadata history.

Reviewer AST and `bash -n` checks pass. Parent owns one named container `okc-proj-operand-reverse-ncu01`, inner240s/outer300s under existing container-held flock and host cleanup guard. Only this invocation sets `PROJ_GPU_PROFILE=1`; ordinary jobs leave it unset. The runner's sole permission change is conditional `--cap-add SYS_ADMIN`, with no `--privileged`, driver-policy edit, module reload or clock mutation. Root reports image `/usr/local/cuda/bin/ncu` is Nsight Compute2025.1.1; host has no installed ncu. [NVIDIA documents the container capability for restricted-counter access](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters). This is temporary diagnostic-container privilege, not a global permission grant.

## Exact planned command

```bash
/usr/local/cuda/bin/ncu \
  --target-processes application-only --kill no \
  --replay-mode kernel --graph-profiling node \
  --cache-control all --clock-control none \
  --nvtx --nvtx-include profile/ --launch-count 2 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --section LaunchStats --section Occupancy \
  --export /okc/logs/operand-reverse01-ncu -f \
  python -u proj_rsi/tools/operand_reverse_ncu.py \
  --json /okc/logs/operand-reverse01-ncu-driver.json
```

Root must use a fresh export target or prove a newly written report; `-f` is not evidence that an old file was replaced. The report, stdout/stderr, driver JSON, exact command/version, identities and terminal owner state must be archived. No auto-retry if collection fails.

## NVTX / graph selection

Driver prepares weights and the actual baseline once, packs W exactly, asserts runtime regs168/local0 and opt-in-before-occupancy>=1, then warms/captures two retained graphs. Baseline captured output stays alive in the graph dictionary; candidate output is separate. Role order and actual same-run rq/Lt selection are persisted. No source flags or lineinfo are added to the frozen extension. No torch.profiler/CUPTI collector is nested inside ncu.

**Fresh-selection limitation:** Nsight injection can alter even excluded preparation/graph timings and consequently the baseline bake's dispatch choice. Original operand-reverse01 used rq`s4s5hz`,KS2; the new diagnostic must report its own actual route. Counters describe those exported kernels only. If baseline route/config changes, comparing its counters cannot decompose the original baseline's timing loss. The candidate stays source-identical, so its valid counters can still diagnose the closed implementation under this profiler setup. All profiled-process PREP/F1/event numbers are excluded from benchmark evidence; no baseline-selection correction or frozen-kernel retuning is authorized here.

Only each selected `graph.replay()` plus its synchronization is inside `torch.cuda.nvtx.range_push("profile")`/pop. NaN fill,256MiB flush read, preparation, warmup, capture and task oracles are outside. [The CLI's push/pop syntax requires the trailing slash in `profile/`, and `--nvtx` enables that filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering). The two ranges occur on the launching Python thread/default NVTX domain; no regular-expression or domain ambiguity is introduced. `--kill no` allows both immediate postprofile oracles to execute after the launch cap.

The expected workload is two independent, single-kernel host-launched graphs; each kernel's necessary CTA/WG cooperation is inside the same launch. Node/kernel replay is appropriate for that scope. **The filter's syntax is not proof of actual selection:** exported report must contain exactly two intended roles. If fresh baseline dispatch has multiple nodes and consumes the two-launch cap, or NVTX/graph handling selects zero/wrong kernels, the diagnostic is INVALID. Verify actual demangled names, launch grids/blocks, shared/register metadata and sequence against each role; do not simply trust result count2. Driver's JSON role order supports interpretation; exported kernel identities decide it.

## Replay semantics and direct output checks

[NVIDIA describes kernel replay as save/restore of accessible/written device memory between metric passes and cache flushing between replay passes](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay). These kernels have no host-response or mandatory interkernel-concurrency dependency. Their output/counter allocations remain alive. `--launch-count2` caps selected application launches, **not** profiler-internal passes. Record pass count; the seven sections can require multiple executions and software instrumentation.

The app flushes256MiB before selection; `--cache-control all` independently requests cold caches for every profiler pass. These are isolated-cache counters, not a preservation of the app's precise cache state. `--clock-control none` avoids profiler clock changes; it does not make dynamic clocks invariant. Nsight durations, event/graph timing and official calibrated benchmark times are distinct observations. This run cannot supersede operand-reverse01's measured loss or certify a new speedup.

After **each selected profiled replay**, before another replay, driver synchronizes, compares X to the original seeded input and task-oracles that graph's retained output. Output had been filled with NaN before selection; task `compare` rejects nonfinite output, so incomplete writes fail. There is no unprofiled replay before these oracles that could hide a replay/restore problem. Failures persist JSON`FAILED` and rethrow. These checks complement the earlier exact4096-column and10mutated-graph suites on identical source; the diagnostic does not claim to rerun every earlier suite.

## Report admission and inference limits

GO is for execution only. Require `DIAGNOSTIC_COMPLETE`, both immediate output/immutability oracles, successful profiler/export exit, two correct role identities/configurations, a newly produced nonempty importable report and finite relevant hardware counters. A JSON success without a valid report is insufficient. `ERR_NVGPUCTRPERM`, unsupported flags/sections/graph nodes, timeout, zero/wrong launches, missing export, `N/A`/NaN/inf counters or mismatched identity invalidate the associated claim. Do not convert a missing metric into zero; uncollected optional metrics remain unavailable.

Use raw counts/bytes, issue/eligible-warp activity, tensor/other-pipe throughput, launch occupancy and stall distributions together. [NVIDIA cautions that stall categories should be interpreted with scheduler issue activity](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Producer and math warps share aggregate measurements; a barrier/long-scoreboard percentage alone does not identify a specific Bfull/X-load critical path. Instruction-site/source attribution requires actual available report detail; frozen source lacks added lineinfo. No physical shared-port ceiling or absolute speedup is derivable from this single diagnostic. A failed collection or suggestive counter does not automatically reopen the closed implementation or authorize another GPU run.

## Setup-only failure and reviewed host metadata correction

Root reported first setup stopped with exec126 **before Docker**, because rsync preserved the repository runner's non-executable mode664. It produced no GPU sample/report and is SETUP_INVALID, not an optimization or profiler result. Root archived `results/proj-rsi/takeover/20261026/operand-reverse01-ncu-setup-invalid.log`, corrected runner executable mode locally/remotely to755 without content/SHA changes, and started fresh named `okc-proj-operand-reverse-ncu02` (reported outer PID357830; actual wrapper/container identity to verify at terminal). Frozen driver, candidate, CLI/cache/clock/sections and fresh export target are unchanged. This is a host executable-bit repair before the first GPU diagnostic, not a parameter retry. Initial named ncu01 remains failed history; only a valid terminal ncu02 report can supply counter evidence.

Reviewer read the setup-invalid log: both timestamps21:59:25Z, exact failure`env: '/root/okc_run.sh': Permission denied`, exit126; local runner mode755 independently verified. Root then identified **actual guardPID357831**, with357830 merely its shell and-list parent; terminal ownership checks must target357831 and the named ncu02 container. Existing host/container deadlines still bound the detached wrapper.

## Terminal collection and pending import validation

Actual ncu02 host log spans22:00:08Z..22:01:21Z (73s), PREP58s, two selected`rq` launches/18passes each, both`PROFILE_END`, `NCU_DRIVER_DONE`, fresh report path and`TRIAL_EXIT0`. Driver JSON is`DIAGNOSTIC_COMPLETE`; both immediate profiled-output oracles pass, input immutable,12exact probe rows each. Candidate L2/row errors1.421e-4/2.633e-4; baseline2.686e-3/2.986e-3. Reviewer independently verified actual guardPID357831 absent and named ncu02 container absent after terminal.

Fresh baseline reports **rq`s4s5z`,KS2**, different from original rq`s4s5hz`,KS2. Expected source configs are FL2 versus original FL3 (cache-hint flag differs); exported demangled names/configurations still need verification. Neither profiler-process bake times nor profiler duration is an updated benchmark.

Archived under `results/proj-rsi/takeover/20261026/`:

|artifact|SHA256|
|---|---|
|operand-reverse01-ncu.ncu-rep (457432B)|61a148ceae4be699d2ea5d634d4b51d1a13f4f91ad1965f46bab2c2e71419ae4|
|operand-reverse01-ncu-driver.json|70241b3cb81527769fa0eb17796e4a43d527a12bb8bd08eada6bcdb620e1cf91|
|operand-reverse01-ncu.host.log|ce9f62a6e4c44933e123ebb50ccf78c12df0d42e6151b7e7a875139ff66ac19c|
|operand-reverse01-ncu-setup-invalid.log|51f4499fd5330bcf06736897b3cb0a06bcfa9a9580331f147320f5cced0bcf76|

The first73B `.raw.csv` is **not counter data**: it contains only`unrecognised option '--units'`. This is a CPU-side import syntax failure, not failed GPU collection or evidence that metrics are zero. Parent notified to import the intact report with installed CLI's supported display option (`--print-units base`, or retain native units). Until successful import proves exactly both roles and finite counters, no bottleneck inference is admitted. No GPU rerun is needed to repair the import.

## Corrected CPU imports: report VALID

Parent repaired only the CPU import command, without another GPU run. Corrected raw CSV SHA256 `1dc954fd57dd68c7911333ef712fcf29d961af6e7758adabad0e27adeb775408`,40880B. This is a **wide CSV**,604columns: header, units row with blank ID, then exactly two data rows IDs0/1. Reviewer parsed this format directly rather than assuming one metric per row. There are591metric columns;1170numeric values across the two kernels are finite, with no NaN/inf. Remaining entries are named categorical attributes (device/policy/etc.) or blank optional `launch__kernel_name`; actual kernel identities are present in the `Kernel Name` column.

|role/ID|actual demangled configuration|grid|block|registers/thread|dynamic SMEM B|NVTX|passes|
|---|---|---|---|---:|---:|---|---:|
|baseline0|rq<128,5,1,0,0,0,2>|(64,2,1)|(384,1,1)|168|144520|default/profile|18|
|candidate1|rq<128,8,1,0,0,0,20>|(128,1,1)|(384,1,1)|168|103600|default/profile|18|

Both are in process34/context1/stream7, H10080GB HBM3, CC9.0. Their identities, roles, outputs and source metadata match the driver. Each grid has128CTAs; occupancy is limited to one CTA/SM by both registers and shared memory. Parent's permission/capability setup successfully collected finite hardware counters. Two correct nodes and both immediate output oracles establish a valid diagnostic for these two exported paths. Baseline is **FL2 s4s5z**, not original FL3 s4s5hz; the fresh-selection limitation remains.

|counter (reported normalization)|baseline|candidate|
|---|---:|---:|
|gpu duration, us (diagnostic only)|25.088|45.856|
|smsp hardware instructions executed, sum|3846732|11646140|
|software `inst_executed`|4410325|11685448|
|tensor cycles active, %peak sustained elapsed|35.65|19.12|
|SMSP issue active, %peak sustained active|20.71|29.97|
|eligible warps/SMSP active cycle|0.26|0.39|
|SM LSU instructions, %peak sustained elapsed|2.07|28.18|
|L1TEX throughput, %peak sustained active|31.08|67.02|
|DRAM throughput, %peak sustained elapsed|68.36|35.66|
|DRAM total byte rate, TB/s (read+write)|2.289908|1.194038|
|L2 sector hit rate, %|26.88|61.48|
|long-scoreboard stalled warps /issue-active (reported ratio)|6.51|3.00|
|short-scoreboard stalled warps /issue-active (reported ratio)|0.91|2.13|
|MIO throttle /issue-active (reported ratio)|0.01|0.95|
|LG throttle /issue-active (reported ratio)|0|0.36|

Raw metric keys are respectively `gpu__time_duration.sum`, `smsp__inst_executed.sum`, `inst_executed`, `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.pct_of_peak_sustained_active`, `smsp__warps_eligible.avg.per_cycle_active`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`, `l1tex__throughput.avg.pct_of_peak_sustained_active`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum.per_second`, `lts__t_sector_hit_rate.pct`, and `smsp__average_warps_issue_stalled_{long_scoreboard,short_scoreboard,mio_throttle,lg_throttle}_per_issue_active.ratio`. Retain the different active/elapsed denominators; percentages are not interchangeable occupancy or wall-time fractions.

Interpretation: this closed candidate performs more instruction/delivery work while feeding tensor operations less effectively. Hardware instruction count is3.027541×, diagnostic duration1.827806×, and issue/eligible activity rises despite the longer kernel. Short-scoreboard/MIO/LG and LSU/L1TEX activity are consistent with the newly introduced loads/stores/transpose path. Eligible warps remain<1/SMSP; higher issue activity alone is not an improvement. Reduced long-scoreboard **per-issue** ratio does not prove less absolute waiting when the instruction denominator changes this much.

There is no evidence of whole-HBM bandwidth saturation in the candidate's35.66% DRAM throughput; memory-delivery latency or a local queue/port can still limit a kernel below aggregate peak. Higher L2 hit rate also prevents equating doubled logical X loads with doubled HBM bytes. `dram__bytes_read.sum` was **not collected**. The available total byte-rate metric includes reads+writes; rate×a separately reported duration is not an exact measured X-specific/read-byte sum and is not used as one. No128B/cycle shared-port ceiling follows.

## Existing-report source import: measured fixed work versus polling

Parent also performed CPU-only `--page source --print-source sass --csv` on the same report, no GPU rerun. Archive `operand-reverse01-ncu.source.csv`,170017B/2820lines, SHA256 `2db4f88564506894a054d1373fdd339db7fb5cc6ec8d9375c17fc4528f73a54d`. Its per-PC `Instructions Executed` values are **dynamic instruction execution counts**, not PC stall samples or latency attribution.

Reviewer summed every exported PC: baseline4410325, candidate11685448, **exactly reconciling raw software `inst_executed` for both kernels**. These are not the hardware `smsp__inst_executed.sum` values. Different definitions/collection passes and timing-sensitive polling preclude interchanging these totals; the report contains18passes and software instrumentation can alter poll iterations. Do not describe a3.0275×hardware count as3.0275×fixed useful work.

Candidate's exported math branch contains exactly2097152dynamic SHFL and2097152SEL executions. Each matches32instructions×8math warps×64stages×128CTAs, precisely the source's X lane-transpose work. Together4194304fixed transpose warp instructions account for**57.6527%** of the software instruction increase7275123. Producer-branch SHFL3811 and SEL52784 are separate escape/decode/control work. Candidate decoded-B stores contribute524288producer STS warp-instruction executions (math output stores add16384); HGMMA executions131072→262144 reflect the narrower n64 instruction shape/more k64 stages, not doubled mathematical FLOPs.

Tight PHASECHK/(optional YIELD)/backward-branch poll loops, identified by their actual exported instruction windows, total560221instruction executions in baseline versus31744in candidate. Candidate's math Bfull retry loops account for the latter; its Wfull/Bempty tight retry loops show0executions in this software pass. These counts show that **the added instruction work is not merely more polling**. They do not measure time spent waiting: loop instruction latency, instrumentation, warp scheduling and producer/consumer overlap remain unmeasured at PC level. In particular, zero retry-loop instructions is not proof that a wait/check instruction consumed zero cycles. No producer-versus-math stall-cycle percentage is claimed.

## Accounting correction against the actual comparator

The design's60→36KiB/k64 comparison was explicitly against old pw19 SS, not this actual rq baseline. For actual full-width rq128, the nominal per-CTA/k64 shared model is Wfill12+W-LDS12+Xfill16+X-WGMMAreads32=72KiB. That CTA covers128channels and K/2; the candidate covers64channels and full K at36KiB/stage. With128CTAs, baseline32stages versus candidate64 gives **288MiB nominal main-loop shared traffic in both**. This is source accounting excluding escape/control/output traffic, not a physical transaction or port measurement. The candidate still removes split/output-reduction traffic, but cannot claim a40% main-loop shared-byte reduction over the actual comparator. This supports the independent queue correction and explains why the pw19-only comparison was insufficient for ROI.

**Disposition:** valid closed-source diagnostic; fixed X-transpose and delivery work is established, tensor underfeeding is observed, a unique bottleneck/quantified removable latency is not established. No speedup, automatic implementation retry, new flags/geometry, transfer or official slot is admitted. Preserve the closed candidate and all original timing evidence. A future hypothesis needs a separately reviewed mechanism and falsifier; this report alone is not that authorization.

## Addendum: one final SourceCounters PC-stall diagnostic preflight

`decision=GO_ONE_FINAL_PC_DIAGNOSTIC; question=H_P_vs_H_XL/H_XT_associations; optimization_retry=NO; further_sampling_tuning=NO`

The seven-section capture above collected per-PC instruction counts, not PC stall samples. A hardware try-wait can occupy cycles without many software polling iterations; those counts cannot resolve whether producer B publication or math X delivery/transpose supplies the critical path. One bounded sampling capture is justified to test the already preregistered opposing-backpressure observations in `audits/next-roi-after-takeover.md`. It need not produce a definitive cause; missing/sparse/mixed evidence is a STOP result rather than permission to keep profiling.

Installed primary artifact archived `results/proj-rsi/takeover/20261026/SourceCounters.section`, SHA256 `5307d8bb457909afbf3757d2f1964fbc458aa55a729c97284332cb245177c189`. Reviewer read its actual collection stanzas: lines154/175 `group:smsp__pcsamp_warp_stall_reasons`,180 not-issued group,190 `smsp__pcsamp_sample_count`; per-PC long/short-scoreboard, LG/MIO, barrier, wait and other reason metrics follow. Its `MaxArch:CC_90` entries filter the IMC metrics, include H100CC9.0, and do not exclude the main sampling groups. This is the installed2025.1.1 section, not an inference from a modern section name.

No source/compile flags/lineinfo changes. SASS addresses and disassembly already export correctly without lineinfo; GPU sampling can associate a warp state with an instruction PC without a C++ source-line map. Remap against the **new actual binary's** SASS/CFG before role classification. Candidate math and producer are distinct branches after the USETMAXREG sites; out-of-line math wait loops follow the producer EXIT in lexical address order, so address order alone is invalid. Baseline may freshly select another path under Nsight and must again be identified independently.

Exactly the same driver89f76f0b…7709, sourceca92a4fb…85f8, runner8b3529b5…9d67 and replay/cache/clock/NVTX/permission options apply, with the **seven sections replaced by only `--section SourceCounters`**, fresh export `/okc/logs/operand-reverse01-ncu-pc`, fresh driver JSON `/okc/logs/operand-reverse01-ncu-pc-driver.json`, name`okc-proj-operand-reverse-ncu-pc01`, inner240s/outer300s. Driver immediately checks each profiled output before another replay as before. No performance admission or new variants; parent owns the exclusive bounded launch and cleanup. Reviewer sent GO after reading the installed section; no GPU execution by reviewer.

Admission and interpretation are separate:

- **Collection validity:** exact two intended kernels/role identities and both immediate oracles, successful fresh report, per-PC sample counts and stall-reason columns actually present. `Instructions Executed` alone does not meet this requirement. Capture sample interval, sampling/replay-pass count and drop/overflow metadata where available; report unavailable metadata as unavailable. Failed/zero/unsupported PC sampling is INVALID for the question, not zero stall cost.
- **Denominators:** preserve raw all-state sample totals per kernel and candidate role (math/producer), plus setup/epilogue/drain/unmapped counts. Normalize each phase to its own role's all-state samples and retain raw counts. Producer has4warps versus8math warps, with different active lifetimes. Do not compare only global percentages. Not-issued categories are subsets of the all-sample categories; never add both groups into one denominator. Selected/not-selected are scheduler states, not lost time.
- **Qualitative sufficiency preregistration:** candidate math and producer each need>=200all-state samples for role comparison; a proposed exposed leading phase needs>=30samples, plus consistent opposite-role backpressure. These are conservative minimums for a coarse diagnostic, not independent-sample confidence guarantees. Periodic sampling can alias phases and samples can be correlated. Missing/ambiguous PC mapping, substantial dropped data, sparse roles or contradictory/mixed backpressure =>INCONCLUSIVE and STOP. A mapped region with0samples is not missing data and does not prove zero cycle cost.
- **H_P support:** math Bfull PCs exposed while producer Bempty backpressure is weak; producer resides in decode/escape/STS/proxy/bar3 rather than Wfull. Exposed producer Wfull instead favors packed-W delivery/refill, not decode ALU specifically.
- **H_XL/H_XT support:** producer Bempty exposed while math resides at X-LDG result consumers with long-scoreboard/LG, or at mapped transpose/dependency PCs with short-scoreboard/MIO/issue pressure; math Bfull relatively weak. Attribute dependencies from actual operand/SASS context; the sampled consumer PC need not be the initiating LDG or producer instruction.
- **Limit:** samples are time-biased warp-state observations, not a temporal overlap trace, stage-period measurement or causal microsecond decomposition. A clear association can rank the existing hypotheses; it cannot establish removable latency or authorize an implementation. Aggregate old-run scheduler counters cannot be silently treated as simultaneous with these new sampled PCs.

No sampling-interval/pass tuning, source rewrite, additional profile after an insufficient result, port or official submission follows automatically. The closed candidate remains closed regardless of diagnostic outcome.

## Final PC diagnostic: valid, sufficient; stop old-reverse profiling

Root's actual guardPID360871/named`okc-proj-operand-reverse-ncu-pc01` ran22:20:00Z..22:21:12Z; two selected launches,5replay passes each, both immediate output/immutability oracles pass and`TRIAL_EXIT0`. Candidate source/driver remain frozen. New baseline is FL3`s4s5hz`,KS2, matching original operand-reverse01 and differing from the first seven-section capture's FL2. Reports remain distinct. Raw report has2sampling aggregation passes, interval index6/2048cycles,32MiB sampling buffer,0dropped bytes; explicit overflow flag was not exported. No further profile follows this result.

Reviewer independently found actual guardPID360871 and named pc01 container absent after terminal.

CPU source CSV explicitly contains all-state/not-issued totals, `# Samples`, dynamic instruction counts and per-reason sample columns. Source PC counts sum exactly to raw `smsp__pcsamp_sample_count`: baseline2405, candidate4844. Visible source reason columns sum to those totals minus raw warpgroup-arrive171/47 respectively; that category is present in raw aggregate metrics but lacks its own displayed source column. This residual is preserved rather than silently discarded. Not-issued subsets are never added into all-state totals.

Candidate SASS CFG yields3173math-body samples,1532producer-body samples and139common setup samples, summing4844; no sampled PC remains unmapped. Both role denominators exceed200. Common setup remains separate rather than guessing its mixed warp attribution (21samples are identifiable initial math X vector loads). All rows below use the stated body denominator, include every scheduler state and are not fractions of wall time.

|candidate mapped phase|all-state samples|% of its role body|selected reason counts|
|---|---:|---:|---|
|math Bfull checks/conditional branches/cold retry blocks|2|0.063%|1dispatch,1selected|
|math transpose SHFL/SEL sites|2388|75.260%|940long-SB,696short-SB,292MIO,201wait|
|math main-loop X vector LDGs|244|7.690%|155LG,41MIO|
|math epilogue/drain|86|2.710%|50long-SB,22barrier|
|producer Wfull check/retry|153|9.987%|138long-SB|
|producer Bempty check/retry|80|5.222%|72long-SB|
|producer decoded-B publish fence/bar3/next branch|273|17.820%|113barrier,136long-SB|

Actual candidate Bfull PC set includes initial try-waits0xfe0/0x1580/0x1df0/0x2500, their conditional branches0x1320/0x19a0/0x1fb0/0x26a0, plus0x3e00..0x3eb0 cold blocks. Only0x1df0 has2samples. Producer Wfull uses0x3140/0x3150/cold0x3ec0; Bempty0x31f0/0x3200/cold0x3ef0; publish0x3bc0..0x3bf0. Remapping is relative to the actual kernel's exported base, not a remembered global GPU address.

A concrete X dependency is visible: candidate0xe00 `SHFL.BFLY ... R106 ...` has307samples,305long-scoreboard. R106 belongs to the raw X vector loaded into R104..R107 by LDG.128 at0x2a0/0x2060. Another transpose consumer0x16d0 SEL reads R66 from X LDG.128 into R64..R67 and has212long-scoreboard samples. A hot SHFL/SEL PC is therefore not automatically transpose ALU cost: many observations wait on the upstream global X load. Other transpose sites carry short-scoreboard/MIO exposure, so X delivery and register-routing dependencies both occur.

**Candidate hypothesis decision:** pure H_P publication-starves-math is not supported as the leading explanation: math Bfull exposure is2samples, below the preregistered30positive-phase threshold, while mapped X/transpose exposure is large. H_XL/H_XT association is strongly supported and meets the sample thresholds. Their causal split remains mixed; producer Bempty80samples coexists with Wfull153 and publish273. Do not turn those counts into a removable-microsecond budget or ignore producer work. Producer publication/decode can still contribute even though it is not the observed dominant math wait. Stop profiling the old reverse source; no automatic retuning/port.

## Current baseline FL3: sampled phases and legitimate target

Actual baseline source is rq<128,5,1,0,0,0,3>,grid64×2,384threads. Its corrected CFG must parse dual-predicate instructions such as `@!P0 BRA P1,target`; a parser accepting only `BRA target` misses the producer branches. Including direct helper call targets gives2039math-body samples,256producer-body samples,110common setup samples; helper bodies themselves have0samples. All2405samples are accounted for. This baseline has only two active TMA issuer warps; its zeroing warps exit early, so its role proportions should not be equated to the candidate's four active decoder warps.

|baseline mapped phase|all-state samples|interpretation|
|---|---:|---|
|combined W+X full-barrier waits|974 (47.768%math;40.499%all)|954long-scoreboard; actual `full` address0x23000|
|X producer xempty backpressure|119/256producer|114long-scoreboard; address0x23050|
|W producer wempty backpressure|110/256producer|109long-scoreboard; address0x23028|
|one packed-W LDS→decode consumer at0x1460|107|106short-scoreboard on LOP3 inputR23|
|initial escape-window SHFL at0x8f0|21|21long-scoreboard; not dominant|
|math EXIT0x38b0 after TMA reduce/read-drain sequence|106|long-scoreboard at tail; do not relabel as zero/flag polling|

The four major math full-barrier conditional branches are0xa70(150samples),0x1370(39),0x1dc0(623),0x27f0(154); their preceding SYNCS try-waits and cold blocks complete974. Source full barriers count **both** packed-W bulk and X TMA arrivals. These samples locate the combined ingress/dependency pipeline; they cannot decide W versus X from that barrier alone. Simultaneously,119Xempty+110Wempty samples occupy229/256producer observations. This opposing backpressure can arise from ring reuse, stage ordering, startup/tail and alternating phases; it is not proof that a pure HBM supply limit explains974math samples.

At0x13a0, `LDS.128 R20,...` supplies R20..R23 from packed W; later0x1460 `LOP3 ...R23...` has107samples/106short-scoreboard. That is shared-load readiness on a decode consumer, not evidence that decode ALU throughput itself dominates. The initial cached-escape SHFL's21long-scoreboard samples are much smaller. Raw warpgroup-arrive171samples (7.11%all) mostly reconcile to HGMMA source rows through the omitted-category residual; explicit final `WARPGROUP.DEPBAR0` at0x3280 has21samples/20barrier. WGMMA scheduling/retirement is present but does not replace the larger combined-full observation.

The source's zero/flag protocol is separate: producer zero stores/global-counter poll then shared zflag publication, acquired by math before TMA reduce-add. Its sampled PCs do not appear as a dominant hotspot. EXIT0x38b0 follows UTMAREDG/UTMACMDFLUSH/DEPBAR;106long-scoreboard samples there are tail/retirement context, with no precise zero-store versus TMA-source-read causal attribution. Do not call them a removable global-zero or flag-poll tax.

**Next topics for new proposals:** current-rq staged ingress/dependency/reuse, including the known LDS→decode exposure, and operand-reversal v2's direct native-A loads removing observed X shuffle/select work. These observations justify explicitly different mechanisms with source accounting and a bounded falsifier. They do not establish that an arbitrary stage-depth/cache-hint/barrier change or decoded-W/prepack/zero-protocol rewrite is profitable. A numerical speedup still requires an unprofiled same-run complete-output comparison.

PC archives, all under `results/proj-rsi/takeover/20261026/`:

|artifact|SHA256|
|---|---|
|operand-reverse01-ncu-pc.ncu-rep|1e1e28d4f654cae9335cbe4f9272fddf1a3cf2f6b3bb99f7a961d569d7013951|
|operand-reverse01-ncu-pc.raw.csv|cd3e90d46208b6c7e27ef939ae8e65028e66be9206fe8794189a3d47744c6cc8|
|operand-reverse01-ncu-pc.source.csv|349e8145313ea1271da6857b25c915af977400e38be057f1a45fc992e47111d1|
|operand-reverse01-ncu-pc-driver.json|23ba6dd80090ad5bee2c04bc07bf368412f3e4b23dd89c9d9ff1d2ff8eab55e2|
|operand-reverse01-ncu-pc.host.log|f5a3936b32ce8cee469a8fc6c9cba860b246f6d7f73b350e1bf5fc0f8d84b4ca|
|operand-reverse01-ncu-pc.session.txt|c998d4ba421bb9dcaa41b103cb208721dfbb137155ec55c3646d8eff006a7225|
