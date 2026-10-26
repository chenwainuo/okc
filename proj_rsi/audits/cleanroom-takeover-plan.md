# Cleanroom takeover: verifier admission and bounded measurement plan

Snapshot: 2026-10-26 UTC. Independent reviewer `/root/proj_evidence`. Readonly local source/history and SSH log inspection; only reviewer audit documents written. No source edits, GPU launches, remote mutation or commits.

## Decision

**GO for one quiet, bounded full-submission comparison of the four serving policies versus frozen pw36.** Tensorcore admission is **MODE0 only, default serving policy, fresh process/state**. Its unchanged diagnostic ladder is **BLOCKED by a definite completion-counter bug**. Latency admission requires b8 real-CUDA checks to pass on the selected frozen default; see `latency-b8-preflight.md`. A failing school can be marked invalid without blocking other schools.

This completes the previously unfinished verifier stage. It is not an approval to port a school, reopen closed variants, consume judge slots, or claim an H200 speedup. The old four-school workflow is dead; its existing source and logs are inputs to a new root-owned scheduled comparison.

## Tensorcore hang: source explains the first router MODE3 deadlock

Frozen file: `proj_rsi/cleanroom/tensorcore/submission.py`, SHA256 `b92f012bf96a168fda296073e2e1fb3030289caa258a54a09a0ff6b07295b522`. Local equals remote `/root/okc/...` and committed `d07ee6f`.

Recorded ladder9 environment is `TC_MAX_M=256`, no grid override. Saved command/container evidence is `results/proj-rsi/takeover/20261026/ladder9-before-stop.txt`; stopped log ends `EXIT 137`. Sequence: qkv_gate m16/128/256, then router m16, modes `3,8,4,5,0`.

Control-flow proof, using current file lines:

1. `GRID=132`, every `ROUTER_GRID[mp]=128` (42,61). For router m16: 2 output n-blocks x128 k-chunks =256 units, so partition returns G=128. Python passes `Gc=max(0,132-G)=4` (827).
2. Launch includes **G+Gc=132** blocks and cooperative attribute when Gc>0 (554-561).
3. Converter blocks 128..131 enter `while (*counters < p.G)` unconditionally, with no mode guard (222-226).
4. GEMM completion counter increments **only if `p.mode==0`** (503-506). MODE3 leaves it zero forever. No later mode or second convert kernel can rescue the already stalled GEMM.
5. `ladder.py` first times torch, then MODE3 via `time_graph`; `time_graph` launches two warmups and synchronizes before printing the row. Consequently the first router MODE3 warmup can stall with no partial router row. This exactly matches the observed qkv-only progress. Qkv default G=132/Gc=0 explains its successful diagnostics.

This is stronger than the earlier log-only conclusion “somewhere in modes3/8/4/5/0”. Actual PC was not sampled, but the recorded source/arguments deterministically contain this deadlock. Modes4/5/6/7/8 with Gc>0 suffer the same waiting-converter defect. Raw diagnostic modes also intentionally omit output/cleanup; they cannot substitute for full-kernel performance or correctness. Do not rerun ladder9, cmp8 variants, or a debug mode on shared mutable state.

MODE0 has a live counter-completion path, cooperative launch keeps its converter/GEMM blocks co-resident, and converters reset workspace/counters. A fresh MODE0-only harness does not enter the diagnosed mode mismatch. **Conditional bounded admission is appropriate; unconditional certification is not.** Future diagnostic repair would need to omit converter CTAs for nonzero modes (and correctly retain separate convert semantics for mode8) or otherwise provide mode-consistent completion/cleanup. That is a separate source change; no patch was made here.

## What the existing TC logs establish

| artifact on H100 `/root/okc/logs/` | verified content | limitation |
|---|---|---|
| `tensorcore_bench6.log` | 65 CHECK ok, 30 timing rows, EXIT0; replay floor13.31us | completed18:58:06, predates v25; not current-source all65 proof |
| `tensorcore_bench7.log` | 65 CHECK ok, 30 timing rows, EXIT0; replay floor14.85us | completed19:01:13, predates v25/19:02:38 source mtime; transcript queued it at18:58:13 |
| `tensorcore_cmp7.log` | current-v25 vs v22, MODE0, qkv16/128,o32,router16/128; no correctness `!`, EXIT0 | only five normal-input shapes; source not hashed in log |
| `tensorcore_ladder9.log` | qkv16/128/256 full rows correct; stalls before router16, later stopped EXIT137 | diagnostic hang; no router results |
| `tensorcore_cmp8.log` | EXIT143 only | no measurements to harvest |

The inherited Claude tensorcore transcript explicitly labels cmp7 v25-vs-v22 and sets `TC_MAX_M=256`; current source is unchanged since d07ee6f. This gives credible limited MODE0 provenance without retroactively assigning v25 to bench6/7. Current cmp7 router16=19.62us vs v22 15.17us; router128=24.77 vs20.69. These are negative v25 evidence under old measurement conditions, not pw36 comparisons.

Log hashes: bench6 `11e749f503de2e73b37bc425837017bcaa68229046e15b4c03ccd820da0c170c`; bench7 `290807dbe5c4e7337894bf5b09fbdbc1145cb0e6f0f721090183bdba0c9c616a`; cmp7 `164ed135aef941b139cc9d246b9166759f073a7d77d1e4ec5fdc587111749360`; stopped ladder9 `07300e63ff922dd3d0fd08795e40f9c577a5d37bd9c44500f4e4c00cf1295e4e`.

## Source freeze and school admission

All four schools pass local `proj_rsi lint`; sizes are below1MB. Source hashes must be recorded again before/after the scheduled run, including remote correspondence.

| submission | bytes | SHA256 | allowed policy |
|---|---:|---|---|
| `candidates/pw36/submission.py` |272704|`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`| frozen incumbent comparator, default prep/bakes |
| `cleanroom/bandwidth/submission.py` |54770|`9ac20ff1a8711db694a7ab1cc70066f33dee402bc2328da520221e119f641fe6`| BW_BIG=0; current small/fp32-router paths, torch above32 |
| `cleanroom/tensorcore/submission.py` |42244|`b92f012bf96a168fda296073e2e1fb3030289caa258a54a09a0ff6b07295b522`| MODE0; bf16 layers torch, router packed through256; TC_CLUSTER=1, TC_PDL=1; unset TC_MAX_M and TC_GEMM_G |
| `cleanroom/contrarian/submission.py` |51987|`47027d0dbeff674d95f4ee80a37f0e6f1e8c8919f112e7528a26ce10d031c094`| CONTRARIAN_BAKE=1, BAKE_TORCH=1, WG=1, MODE=0; unset CFG |
| `cleanroom/latency/submission.py` |59309|`e4c68752a7734d3a4644a21bef3df649435ece3b8f126959a755bf3f9678cb1b`| b8-selected frozen default, LOADER3, DBG512, empty PLAN, SMAX16; hash if changed |

Bandwidth default preparation calls GPU warmups and validates function attributes before capture. `bandwidth/batch5.log` completed65/65 and has30 rows; `v9.json` is terminal. It ran immediately after the hang cleanup, so exclusivity is unproven. Hash batch5 `24c1fc6450ed12a029c7b3ee82521cbd0befaf0d66e454712f9f86bb39799b00`. Do not rerun its diagnostic sweep as part of verification.

Contrarian `bench5.log` contains65/65 and30 rows; `check_all.log` contains65/65 including outliers, `FAILS0`. Hashes `8b06548584a1da498febc0bd94667734fbf17f363c46807bdd6cb13a203f7e6f` and `d66152a141f5cad7b9f2f03f18c63a8f5c78218188473378667cdc15108b1c71`. Its split-K epilogue busy-waits, so physical GPU exclusivity is required. `_cfg_ok()` bounds total grid by measured occupancy x actual SM count; the standard launch is not a CUDA cooperative launch. Existing completed evidence supports a bounded exclusive rerun, not using this implementation concurrently with another GPU job.

## Bounded scheduler protocol

1. Root owns a single exclusive window using the repaired in-container `/gpu.lock` flock. Root reports ordered same-inode C/D monotonic test passed: C_EXIT15871561587989 < D_ENTER15871581462511; initial coarse wallclock test was inconclusive. Verify no CUDA owner/container bypasses the lock before starting. No soak, profiler or CUDA compiler warmup process runs beside the measurements.
2. Freeze and hash the five source files. Explicitly clear inherited school environment overrides; print effective policy. **All GPU prepare work stays inside the lock.** `h100_bench.py --prepare-only` generates GPU weights and calls CUDA packing/bakes/warmups, so it is not CPU-only compilation safe to run outside this ownership window.
3. Run separate fresh Python processes for each full submission, in one reserved window. Suggested coarse order: pw36 -> bandwidth -> contrarian -> latency -> tensorcore -> pw36. Put TC last among schools so a fault cannot erase completed other-school artifacts. Separate processes isolate extension/global picks/workspaces; no diagnostic MODE transitions. Each school runs `h100_bench.py <frozen_file> --official --reps 20 --json <unique_path>` on all five layers and all six loads.
4. Give each process/container a240s deadline, total window cap12minutes. Use root's named-container cleanup monitor. Preserve partial output immediately; stop on no progress/fault/timeout and confirm compute PID/container termination before another CUDA process starts. A Docker-client timeout does not prove CUDA termination. Do not duplicate a slow or hanging queue entry.
5. `h100_bench --official` currently prints failures but **does not fail the process**. Result admission must require exactly65 CHECK lines, all ok,30 timing records, finite positive times, terminal JSON, process exit0, and source/policy match. Mark a partial/failed result invalid even if the process returns0. Its correctness loop uses normal distribution for every m: add the exact nine official distributions (including router9/qkv512/unpooler1000 outliers) before a mechanism is eligible to port; a “65/65” claim alone does not cover these outliers.
6. Save device/driver, clocks/power/temperature, actual CUDA process ownership at transitions, lock/container names, source hashes, PREPARE duration, full logs, effective dispatch/bake picks and per-process replay/read6MB probes. A stable end baseline limits coarse drift; it is not a substitute for paired confirmation of a marginal win.

The six-process fixed-order table suffices for screening clear losers and closing the cleanroom verifier inventory. For any apparent >=0.3us per-shape win, use one small targeted same-process AB/BA paired graph comparison against pw36's **actual prepared pick**, same inputs/read flush, fresh state, correctness after repeated graph replay. Do not repeat every school or retune every config. If the end baseline drifts materially, keep only clear qualitative results and remeasure the affected candidate/baseline pair.

## Outputs and decision gates

Root/verifier writes `proj_rsi/audits/cleanroom-measure.md`: source-bound65/65 status plus exact-nine status, preparation seconds, replay probes,30-shape table vs pw36, dispatch picks, six load sums, raw artifact hashes, invalid/excluded schools and reasons. Three independent judge roles may review the one result set: numerical/synchronization correctness, measurement validity, and implementation cost/ROI. No additional GPU runs per judge.

Synthesis `cleanroom-plan.md` ranks **mechanisms**, not whole-file replacements. Each item records source shape, measured delta versus actual pw36 pick, paired repeat evidence, transfer mechanism (fixed delay vs overlap), integration cost/risk, predicted score contribution as an explicitly labeled H100->H200 hypothesis, and stop bar. Gate: >=0.3us relevant shape improvement in the same quiet comparison, required correctness, prepare<=240s, then minimal isolated integration and official check at reset. Aggregate by six load sums; an unweighted mean/geomean of30 shape ratios is not the official score. Do not apply a precise H200 speedup from H100 replay-floor subtraction or diagnostic mainloop time.

If all full schools lose to pw36, the verifier succeeded: close unchanged schools and retain only profiled bottlenecks (e.g. TC atomic tail, contrarian instruction issue, latency producer/consumer fill) as future hypotheses. Keep unchanged TC diagnostics blocked. Continue the previously ranked P4/latency/stream-K queue using measured promotion/kill gates, not speculative H200 byte-ratio predictions.
