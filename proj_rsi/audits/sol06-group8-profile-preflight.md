# DP versus canonical G8 counter preflight

State: CPU_BINDING_MAP_PASS_GPU_UNRUN; root and independent critic GO required before root GPU execution. Worker06 made no CUDA API call, extension build, GPU context, capture or prepare call. Only new driver and this audit are owned here.

Frozen driver: `tools/group8_compare_ncu.py`, SHA256 `245b78898a16c2087b00924cd809cd33fb74bfd7c19c9012bd7d040005485702`.

CPU proof command (run from `/home/ubuntu/okc/proj_rsi`):

```sh
python3 tools/group8_compare_ncu.py --cpu-check --json /tmp/group8-compare-cpu.json
```

Observed exit0, state `CPU_BINDING_MAP_PASS_GPU_UNRUN`; proof SHA256 `cafc09b48b76dd4fc90227da7154c084a761f87293e46dbccf52a1cb43ca8aa5`. CPU path binds every full-manifest file, regenerated source, pinned parent drivers, physical checker and actual saved SASS map. Predicate inversion at a discovered failed-entry branch is rejected for both variants. AST checks exclude CUDA/extension/prepare/build access from CPU proof functions; imports and source construction do not invoke CUDA. CPU SASS evidence is preflight; GPU path archives/checks actual JIT binaries before any math launch.

Pins: DP source `97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b`, manifest `281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241`; G8 source `e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`, manifest `2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7`. Parent drivers remain unchanged: DP `65e8d9fda91c2b281f7a8b72a693b0f94f37ef1fd4d61fb2160b991a2dc65525`; packed-X `a3e81cc0adece02ffbf7f28250bb2b798642b94ccdfc63762ea90054b9b74d4d`.

## Question and predictions

G8 halves static math fence/commit/wait1 boundaries but its valid runtime has not won. This diagnostic asks whether reduced X lookahead exposes dependencies at retained fences, coarser two-B retirement increases producer Bempty backpressure, or wait exposure simply moves. Higher math X/load-consumer/fence exposure with intact request/sector coverage supports the first association. Increased producer Bempty exposure/failed-lane evidence with comparable Wfull/store behavior supports the second. Similar total role composition with redistributed wait PCs supports movement. These observations remain associations across replay passes, not causal time decomposition. No implementation or optimization claim follows from the capture.

## Physical preflight

The generalized map uses G8 `outlined_entry_PCs` and DP `outlined_wait_paths_no_A_writes`, discovers entry predicate/fast test/slow test/retry/return from each binary, exports every outlined instruction, and discovers the actual post-CTA-barrier role branch and disjoint complete reachable sets. No old hardcoded PC is reused.

|Saved-SASS property|DP|G8|
|---|---|---|
|Role branch|`@P1 BRA 0x1de0`|`@P1 BRA 0x1e90`|
|math / producer / common PCs|324 / 339 / 185|352 / 344 / 184|
|failed-entry PCs|cc0,1030,1290,1570|ca0,d50,1370,1410|
|outlined entry PCs|3110,3150,3190,31d0|3210,3250,3380,33c0|
|fence PCs|d10,1070,12d0,1600|d90,1450|
|wait1 PCs|e80,11f0,14c0,17b0|1060,1760|
|loop / total LDG128 sites|16 / 32|16 / 32|

Both retain scalar STS before producer proxy fence and complementary owner-specific bar3/bar4. Actual GPU resource gate requires512 threads,122regs,0local,103600SMEM,maxThreads512,65536rounded register pool and occupancy exactly1 after opt-in; pack22regs/0local/maxThreads>=256. Actual JIT liveness uses frozen DP A4 and G8 A2x32/fence coverage checkers.

## Root-only proposed capture

One NVTX range encloses DPpack→DPmain then G8pack→G8main, four selected nodes across two graphs. Fresh disjoint XP/output/counter/input workspaces; no parent prepare/autotune. Both actual JIT SASS files are saved before math. Immediately after selected replays, both exact output oracles, finite/poison checks, full1MiB XP exactness, W/pack/esc/X immutability and cnt0 precede any replay/workspace mutation. Both graphs reset after drain in `finally`; failure records remain in JSON. Inner240 / outer300 seconds are proposed explicit caps, including compilation and cleanup; timeout/signal termination can prevent Python cleanup and must remain an invalid/incomplete attempt until root confirms process/GPU drain.

```sh
timeout --signal=TERM --kill-after=10s 300s ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none --nvtx --nvtx-include profile/ \
  --launch-count 4 --kill no \
  --section LaunchStats --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables --section SchedulerStats \
  --section WarpStateStats --section SpeedOfLight \
  --export /okc/logs/group8-compare-ncu01 -f \
  timeout --signal=TERM --kill-after=10s 240s \
  python3 /okc/proj_rsi/tools/group8_compare_ncu.py \
  --json /okc/logs/group8-compare-ncu01-driver.json
```

Root must use its established exclusive GPU/NCU executor and profiling capability; the command is not worker authorization. Require terminal exit0, both selected-oracle records, two cleanup records, expected4 nodes and actual source/JIT/report binding before accepting counter data. Preserve failed attempt artifacts.

Analysis preregistration retains each parent capture's denominators, all math/prod/common samples, all phase residuals and raw/source execution totals separately. Require >=200 samples per math/producer role and >=30 per leading phase, otherwise report insufficiency. Predicated-On Thread counts failed lanes; TRYWAIT may block even when its first predicate succeeds, and per-thread nonuniformity precludes claiming exact warp counts from `/32`. SourceInstructionsExecuted can differ across hardware passes. Per-node cache-controlall replay makes this a cold counter experiment, unlike normal freshly-packed XP consumption. PC fractions are not microseconds; NCU duration is not a pipeline benchmark. On an inconclusive capture, identify the missing evidence before proposing another bounded diagnostic; retain G8 family rather than discarding a valid losing version.

## Completed fresh capture: exposure moved to the loop head

Root reported terminal2026-10-27 02:45:50UTC exit0. Archived raw/source/session/JIT/host/driver artifacts are in `diagnostics/group8-compare-ncu/` and `results/proj-rsi/takeover/20261027/group8-compare-ncu/`. Both actual selected output/XP/immutability/cnt gates pass and both graphs reset after drain. CPU analysis ran locally only, with no remote export during later timing.

Reproduce:

```sh
python3 tools/group8_compare_analysis.py --directory diagnostics/group8-compare-ncu
```

Analysis program SHA256 `c793dda022c0f5f0f88fd49dffa5e621fdc03621659e2c140c1731c6aaaa2aec`; resulting `analysis.json` SHA256 `cde04cfc14987ec0802c9a599778ae6ee0d6ba8fa6f5bba0b2fc82402a736039`. All848 DP main rows,880 G8 main rows and56 rows for each pack bind exactly to their respective actual JIT, normalizing relocated branch targets and omitted reuse annotations. Every main PC belongs to exactly one phase. Parent DP capture retains its own complete totals/roles/phases and hash in JSON.

|Fresh capture measure|DP|G8|
|---|---:|---:|
|All main / math / producer / common samples|3367 /1719 /1548 /100|3604 /1840 /1639 /125|
|Math fence samples (% math)|770 (44.79%)|426 (23.15%)|
|Math loop-head `MOV R0,0x400` (% math)|1 (0.06%)|466 (25.33%)|
|Math Bfull complete fast+outlined (% math)|104 (6.05%)|156 (8.48%)|
|Math HGMMA / loop LDG128 samples|271 /48|277 /54|
|Math wait1 / tail wait0 samples|13 /3|5 /1|
|Producer Bempty complete region (% producer)|543 (35.08%)|628 (38.32%)|
|Producer Wfull complete region|79|90|
|Producer proxy/barrier/convergence|200|203|

The new G8 hotspot is actual PC`0xba0`, an ordinary loop-head `MOV`, with465/466 samples classified long scoreboard. The same instruction in DP at`0xbc0` has1sample and no long scoreboard. G8's loop backedge`0x18e0→0xb90` follows XP reloads at the loop bottom; this hotspot occurs before the next iteration's Bfull tests/fence. Fence-only accounting therefore misses the moved dependence exposure. This is evidence for a scheduling/dependency problem around the reduced A lookahead and loop boundary, not evidence of costly intrinsic MOV execution or a decoded proprietary scoreboard wait mask. The combined fence+loop-head association is44.85% DP math versus48.48% G8 math, each using its own denominator; those fractions are not elapsed-time estimates.

Math failed-entry lanes rise from824608/2097152 (39.3204%) to1180288/2097152 (56.2805%). DP's four site fractions are27.14%,5.12%,87.08%,37.94%; G8's are66.36%,45.91%,65.56%,47.29%. All eight sites satisfy exact slow-test-lanes=failed-entry+failed-retry conservation. Producer Bempty branch failures rise672/917504 (0.07324%)→1376/917504 (0.14997%), with exact conservation; its branch samples537→626 are overwhelmingly long scoreboard532→624. Wfull failed lanes remain0 for both despite79/90 regional samples: a successful first TRYWAIT can have blocked. The Bempty shift supports somewhat greater producer retirement/backpressure exposure, while increased Bfull failures also show downstream publication scheduling changed. Neither establishes initial readiness or an exact warp count.

Memory coverage stays exactly263168 global-load requests and4198400 sectors for each main (262144/4194304 math plus1024/4096 initial CSR). Main DRAM reads51465984→51487488B differ only21504B (~0.042%); read-hit rate31.10%→33.51% does not support an L2-miss explosion. Eligible warps0.42→0.40 and issue-active30.46%→30.19% show no throughput improvement. Global byte totals cannot isolate XP from W/metadata, so this capture does not prove XP-specific L2 latency. Both pack nodes have identical8192load requests/32768sectors and1051904DRAM-read bytes; pack samples291/286 are reported separately. Zero observed pack DRAM-write bytes is not zero pack stores: writes may remain cached under these metrics.

Source instruction totals8482629/8614707 stay separate from raw8176340/8267088. Per-PC reason residual201/187 matches each raw warpgroup-arrive category but is not redistributed to PCs. Both math/prod roles and each leading fence/Bfull/Bempty/G8-loop-head phase exceed200/30 sample requirements. DP loop-head1 and both wait1/tail phases are too sparse for detailed comparative claims. Historical parent DP math1750/prod1556/common89 remains separate from this fresh math1719/prod1548/common100.

The supported diagnosis is that halving boundaries removed fence-site exposure without removing the underlying dependency chain: exposure moved prominently to the G8 loop head, while B publication/retirement pressure increased. This explains why boundary count was an inadequate performance predictor, and is consistent with the separate unprofiled losing result; this single cold capture cannot uniquely apportion that loss. The next useful mechanism targets A delivery/lookahead or loop-head dependency scheduling while retaining exact math order and physical lifetime proofs. Root's already queued native-X trial asks a related delivery question, but its larger shared-memory footprint and altered input path remain confounds to measure independently. No kernel optimization or performance gain is claimed from this diagnostic.
