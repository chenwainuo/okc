# 11 — Briefs to re-spawn the tracks (Agent tool, subagent_type general-purpose, model opus, effort medium; one per track)

Common preamble for every brief: "Repo /home/ubuntu/okc. Read handover_folder/README.md, 02 and 03 first, then the track file
named below and the last 150 lines of proj_rsi/LOOP.md. Box: ssh root@107.170.6.60; run everything through
/root/gpu_lock.sh /root/okc_run.sh <cmd>; batch variants into one locked run; compile outside the lock with --prepare-only;
rsync your own candidate dir before each run; logs under /root/okc/logs/<track>/. Rules: lossless, no host sync, no
input-derived caching, file ≤ 1,000,000 bytes, prepare ≤ 240 s with budget guards, wgmma guarded by
__CUDA_ARCH_FEAT_SM90_ALL with -gencode arch=compute_90a,code=sm_90a. Validate with proj_rsi/tools/h100_bench.py --official
(all CHECK ok), python3 -m proj_rsi lint, cpu-check --official. Commit only your own directory. Report raw data: one within-run
table vs the reference, in-prepare bake picks with both baselines, registers/spills, prepare seconds, file size, commit hash."

- **RS128 author (04)**: "Track file 04-track-rs128-m128.md. Own proj_rsi/candidates/rs128/. Levers in order: 132-SM stream-K grid;
  2-CTA-cluster TMA multicast of x; epilogue staggering. Measure each alone vs the shipped rq paths (F1 rule); stop at the first
  that does not beat them by ≥ 0.3 µs; port a winner into pw3x's RS128 section too." 
- **P4 (07)**: "Track file 07-track-p4-anatomy.md. Own proj_rsi/candidates/rs128p/. Build the producer-written per-slot escape
  patch table (math warps: one LDS + predicated XOR, no list walk, no fence waiting on a walker). Kill at < 0.3 µs in-prepare.
  Report stage clocks next to the shipped row."
- **rs32 (05)**: "Track file 05-track-rs32.md. Own proj_rsi/candidates/rs32/. NT=32/16 instantiations of rq for qkv/o (+ pooler/
  unpooler if the bake says so), short-K decomposition for o, dispatch by exact m, bake vs pw34's actual m=32/16 picks.
  Kill: qkv m=32 not ≥ 2 µs under pw34's pick after two iterations."
- **Router (06)**: "Track file 06-track-router-pw35.md. Own proj_rsi/candidates/pw29m/. Cut the 2.2 µs to first data in rtk;
  router m ≤ 32 one-node latency (gap +0.9…1.2 on gv paths). Measure vs pw35's picks in one run."
- **Latency loader (08)**: "Track file 08-track-cleanroom.md, latency row. Own proj_rsi/cleanroom/latency/ only (clean-room: do
  not read proj_rsi/candidates/* or other schools; pw34 may be run as a black box). Fix the loader to ≥ 23 GB/s per SM
  (mb_load.py first), port it behind LATENCY_LOADER, measure vs pw34 in one run with prof.py; targets in the track file."
- **Clean-room verifier/judges (08)**: "Read each school's DESIGN.md and submission; run all four plus pw34 in ONE locked harness
  run (--official, quiet GPU); write proj_rsi/audits/cleanroom-measure.md (table) and cleanroom-plan.md (mechanisms worth
  porting, ranked by score-µs using the leverage table in 01)."
- **Integrator (09)**: "Build pw36 = pw35 + the winning sections (rs32, rs128 fixes, latency kernels) one at a time, each
  box-validated vs the previous pw3x in one run (65/65), then check at the reset, then scored."
- **Critic (any candidate before a judge slot)**: "Review <candidate> for: correctness fallbacks on every new path, bake baselines
  (F1 rule, warm-up discard, budget guards with timestamps, judge r6 phase 105–120 s), dispatch by exact m, file size, lint,
  the transfer rule (fixed-delay vs overlap gains); approve or list blocking fixes." (Pattern: proj_rsi/audits/*-critic.md.)
