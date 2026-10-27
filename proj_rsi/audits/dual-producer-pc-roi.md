# Fresh DP PC/counter audit: publication improves; residual math fence exposure

`state=VALID_FRESH_DP_COUNTER_PC_ASSOCIATIONS; no_time_decomposition; backend_incumbent_win=NO; next=BOUNDED_CPU_FENCE_PAIR_PROOF`

Reproduce: `python3 results/proj-rsi/takeover/20261027/dual-producer-ncu/analysis.py`. [Program](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/analysis.py), [JSON](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/analysis.json), [raw](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/dual-producer-ncu01.raw.csv), [source](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/dual-producer-ncu01.source.csv), [actual JIT](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/dual-producer-ncu01-driver.jit.sass), [driver](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/dual-producer-ncu01-driver.json), [host](../../results/proj-rsi/takeover/20261027/dual-producer-ncu/dual-producer-ncu01.host.log). All input hashes and analysis-program hash are retained in JSON.

## Binding / validity / measurement boundary

Root383540, terminal2026-10-27 01:10:47UTC exit0,2nodes×34metric passes; root confirms no process/0MiB. Exact driver65e8, DP281ff/source97dd, actual JITbebe4726 matches the valid unprofiled DP binary. The checker independently relocates/normalizes **all848main source rows** against the actual JIT, including absolute branch targets and omitted `.reuse`, then derives disjoint role reachability from the actual post-CTA-barrier role branch0xa90→0x1de0. Math starts0xaa0; producer starts0x1de0, with each role's outlined paths included. Common/startup remains separate.

Actual512threads,122regs→allocated128×512=65536,0local,103600dynamic shared/104624NCU total, opt-in-before-occupancy1, max512. Pack is256×256/22regs. Actual-JIT A4 physical gate passes before math. Immediate profiled graph output oracle passes relativeL2=1.421e−4/worst row2.633e−4 with12probe rows exact; XP is exact all1MiB after poison, W/pack/esc/X immutable, counters0, graph reset after drain. No parent `prepare` or weaker reference was introduced.

Actual command uses LaunchStats/SourceCounters/MemoryWorkloadAnalysis_Tables/SchedulerStats/WarpStateStats/SpeedOfLight, NVTXprofile,2nodes, kernel/node replay, cache-controlall/clock-controlnone; session records application-only process targeting. This is **per-node cold profiling**, unlike normal packed-X→main consumption where the first kernel has just written XP. No Nsight duration is used as a whole-pipeline benchmark. Six unavailable ctc RX/TX metrics stay unavailable. Unmodified-clock warning remains;34passes are replay passes, not34independent performance observations.

## Current leading locations, with complete accounting

3395main samples =1750math+1556producer+89common/startup. Producer now has8warps across two owners, math remains8; common producer PCs cannot distinguish owners. Only source-proved bar3/bar4 predicates separate even/odd owners. Both role totals exceed the preregistered200sample gate; relevant phases exceed30.

|current DP location|samples|fraction of its role|interpretation|
|---|---:|---:|---|
|math4×WARPGROUP.ARRIVE|833 (832long-SB)|47.60% math|Large exposed dependency association; not intrinsic fence latency/current-X attribution|
|math4fast TRYWAIT+4failed-entry branch PCs|118 (109long-SB)|6.74% math|B publication remains visible, reduced leading share|
|all math Bfull slow/test paths|119|6.80% math|Includes1outlined-path sample beyond the8exact PCs|
|math16HGMMA sites|299|17.09% math|Tensor issue/body association, not a throughput ceiling|
|math16loop X LDG128 sites|34|1.94% math|Load issue locations exclude stalls later seen at consumers/fences|
|producer Bempty TRYWAIT+predicate branch|626 (branch622/620long-SB)|40.23% producer|Consumer retirement/backpressure exposed; initial TRYWAIT can block|
|producer complete Bempty region|635|40.81% producer|Also includes BSYNC/control; keep distinct from exact626|
|producer Wfull complete wait region|115|7.39% producer|Current W-fill dependency association|
|producer LDS/decode/skip/escape|260|16.71% producer|Includes data waits and duplicated CSR traversal, not pure codec throughput|
|producer scalar-store address+STS|162|10.41% producer|Original scalar store mechanism retained|
|producer proxy/bar3+bar4/convergence|153|9.83% producer|Shared publication/order/convergence, owner-specific barriers recorded|

The JSON partitions every PC once, retaining remaining setup, addresses, release, output, publish/refill and loop-control phases. Source reason columns sum3178vs3395samples, residual217; raw warpgroup-arrive sample category is also217. This is compatible with an unexported per-PC reason, **not permission to fabricate its per-PC distribution**. Source instruction sum8,481,243 and raw8,176,336 stay separate; replay/metric/polling differences do not justify forcing equality.

## Failed predicates: actual lane evidence, not unpredicated branch counts

|failed-entry branch→outlined block|predicated-on failed lanes / tested lanes|failed-lane fraction|slow-test lanes / failed-retry lanes|
|---|---:|---:|---:|
|0xcc0→0x3110 (@!P0)|137696/524288|26.2634%|424544/286848|
|0x1030→0x3150 (@!P0)|25824/524288|4.9255%|33312/7488|
|0x1290→0x3190 (@!P1)|453376/524288|86.4746%|1907712/1454336|
|0x1570→0x31d0 (@!P1)|196832/524288|37.5427%|613856/417024|

Total failed-entry lanes813728/2097152=38.8016%; failed-retry lanes2165696. For every site, slow-test lane count exactly equals entry-failure+retry-failure lanes. This validates the actual predicate/CFG interpretation. `Instructions Executed` is16384per fast branch **regardless of predicate**; it is not a taken-branch count. `/32` would be only a warp-equivalent normalization unless per-instance predicate uniformity is independently proved.

Producer Bempty branch0x2280 failed608/917504lanes=0.06627%, yet its622samples are mostly long-SB. [PTX8.7 try_wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait) may block until completion/time limit; no explicit timeHint is supplied here. A successful first test can therefore have waited. Neither small failure fraction nor zero outlined samples proves initially-ready data or only intrinsic instruction cost. Source execution counters and samples from replay passes are not a single causal timing trace.

## Freshness versus PX / next ROI

Pinned [PX analysis](../../results/proj-rsi/takeover/20261027/packed-x-ncu/analysis.json), source2c28: math2387samples, Bfull1027(~43.0%), ARRIVE471(~19.7%); producer1111samples, Bempty70(~6.3%). Fresh DP composition changes to Bfull6.74%/ARRIVE47.60% math and Bempty40.23% producer. This supports the intended shift: parallel producers reduced publication's leading exposure while consumer-side dependencies/backpressure now deserve attention. Different captures and4→8producer warps preclude subtracting raw samples as elapsed savings.

LDG requests263168 and sectors4198400 exactly match math262144requests/4194304sectors plus1024producer initial-CSR requests/4096sectors. X vector coverage stays intact; doubled initial CSR warps explain the metadata delta from PX, with no half-sector regression. Main observed DRAM reads51,467,008B, eligible warps0.41/cycle-active, issue29.68%; no HBM transfer explosion follows. Aggregate DRAM bytes do not isolate W/XP/cache metadata or prove bandwidth saturation.

Separate valid unprofiled DP: F134.496us versus PX39.520/STMX38.176/incumbent24.096; paired gains+5.152/+3.904/−10.592us. Thus meaningful version progress, still no incumbent win. The later combined runtime's F1+.832us versus DP but paired−.256us/10of21wins is **not** a new best; its order analysis belongs to the other track.

Root now authorizes one CPU paired-fence prototype on scalar-store DP, supported by [ISA/A4 feasibility](dp-fence-batching-feasibility.md). It must test the exact hypothesis with32instead of64fences/WG and future A loads after the retained fence, with explicit3ahead→2ahead overlap confound. Hardware wait masks remain unproven; today's hot ARRIVE samples may move rather than disappear. No K128/new backend or lower precision is inferred from this profile. Source/resource/actual-JIT lifetime+fence checks and all incumbent/DP/STMX/fresh-graph oracles precede any bounded root GPU trial.
