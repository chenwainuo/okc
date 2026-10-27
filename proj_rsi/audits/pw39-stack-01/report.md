# PW39 full-scope author analysis

**VALID local execution; paired aggregate improves versus PW38, while conservative F1 regresses.** No official H200 score or isolated confirmation-policy gain is established. This result provides a measured local improvement eligible for root review without a significance threshold; the mixed F1 evidence remains material.

Exact candidate source `8a13389137add37b6d363685bfec98a7c83d48704cab2a72444129efc0c5520a`; driver `2f4afe883c269ccef64831aa88b52c1b395294601756032bf7fed5adc04be0cf`. Collected job `pw39.stack.fullscope.01`, guard `424082`, terminal exit0. All 48 collected artifact hashes and 61 dispatched source/review pins independently match.

Validity includes all222 correctness cases (74per arm, including9exact official distributions),4350raw timings,30shapes,180immediate terminal output oracles,360retained-graph mutation checks with output and float-scratch poisoning,30graph-drain/reset records and an empty final graph list. Physical arena counter/alias protocols and dormant/immutable storage pass the frozen reviewed driver; tensors cannot be reconstructed after exit. Fifteen JIT archives were captured before preparation math; all6RQ resource/PC/predicate gates rerun successfully. EH/control resources satisfy168registers/LOCAL0; all actual PW38/PW39 RQ instruction streams match. Cleanup independently verifies no container, guard, compute PID, residual containers or GPU memory and cleanup_failed=false. Exit0 alone was not used as validation.

| Control | Paired six-load GM control/PW39 | Conservative F1 GM | Paired shapes win/tie/loss |
|---|---:|---:|---:|
| rs32 | 1.009565229 | 1.005619699 | 19/3/8 |
| pw38 | 1.007044460 | 0.997206585 | 23/2/5 |

Ratios above1 favor PW39. Six-load ratios use sums of five marginal shape medians, then an equal-load geometric mean; these are descriptive local H100 summaries, not the official score formula. F1 uses the lower before/after control median against the middle candidate median. All30paired samples, deltas, IQR and within-batch bootstrap intervals are retained in analysis.json; these intervals do not establish cross-run repeatability.

Preparation and confirmation

- rs32: prepare wall 58.405215s, original import-clock 58.664696s; no clock reset, two build threads joined, no serving CPU fallback.
- pw38: prepare wall 59.828036s, original import-clock 59.828358s; no clock reset, two build threads joined, no serving CPU fallback.
- pw39: prepare wall 48.339844s, original import-clock 48.366792s; no clock reset, two build threads joined, no serving CPU fallback.

The new helper was actually invoked17times:6confirmed,11skipped because no screened candidate was admitted,0errors and0allowance/prepare-budget partials. All42five-replay blocks/210unsorted raw replay values and every invocation/returned choice/status are copied into analysis.json. Used allowance 1.307574608s; the in-memory observer overhead was charged to the original clocks. Recorded raw medians, bracket order, conservative ratios, unchanged0.99qualification, selected winner, returned latency and zero host launch statuses independently recompute.

| Shape | Screen → confirmed | Candidate worst bracket ratios |
|---|---|---|
| pooler/M128 | rq:s4s5hz → rq:s4s5hzEH (confirmed) | s4s5hz 0.911271006, s4s5hzEH 0.902877657 |
| pooler/M256 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| qkv_gate/M128 | rq:s4s5hzEH → rq:s4s5hzEH (confirmed) | s4s5hzEH 0.863414641, s4s5hz 0.863414641 |
| qkv_gate/M256 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| unpooler/M128 | rq:s4s5hz → rq:s4s5hzEH (confirmed) | s4s5hz 0.914868086, s4s5hzEH 0.913669036 |
| unpooler/M256 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| o/M128 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| o/M256 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| pooler/M32 | rq:t32s5hze-k4 → rq:t32s5hze-k4 (confirmed) | t32s5hze-k4 0.960839187, t32s5hz-k4 0.973426614 |
| pooler/M16 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| qkv_gate/M32 | rq:t32s5hze-k6 → rq:t32s5hze-k6 (confirmed) | t32s5hze-k6 0.831578985, t32s5hz-k6 0.840350931 |
| qkv_gate/M16 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| unpooler/M32 | rq:t32s5hze-k2 → rq:t32s5hze-k2 (confirmed) | t32s5hze-k2 0.958041963, t32s6hze-k2 0.960948423 |
| unpooler/M16 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| o/M32 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| o/M16 | r6 → r6 (skipped) | no-screened-candidate-admitted |
| o/M128 | r6 → r6 (skipped) | no-screened-candidate-admitted |

Confirmation changed the original screen only for pooler128 and unpooler128, choosing EH instead of nonEH; qkv128 retained EH. All three M128 EH paths are selected. The three M32 confirmations retained screened choices. This is not an observed reduction in selection regret: one fresh prepare per arm also changes unmodified GV/LT/router choices, and unselected alternatives were not benchmarked through the full serving schedule. Within-confirmation candidate ratios justify the observed local decision only. No partial fallback was exercised; CPU tests, not this GPU run, cover that branch.

All30 shapes

Positive deltas favor PW39. Picks are RS32 / PW38 / PW39. Every tiny positive, tie and regression is retained.

| Shape | Actual picks | Paired Δ vs RS32 µs | Paired Δ vs PW38 µs | F1 Δ vs RS32 µs | F1 Δ vs PW38 µs |
|---|---|---:|---:|---:|---:|
| decode1/qkv_gate | GV5 / GV5 / GV5 | -0.144 | -0.128 | -0.288 | -0.160 |
| decode1/o | GV29 / GV29 / GV29 | +0.064 | +0.016 | +0.096 | +0.128 |
| decode1/pooler | GV53 / GV51 / GV51 | -0.048 | +0.048 | -0.224 | -0.064 |
| decode1/unpooler | GV92 / GV92 / GV93 | +0.000 | +0.128 | +0.032 | +0.160 |
| decode1/router | RR1006 / RR1007 / RR1006 | -0.032 | +0.000 | +0.000 | +0.000 |
| decode16/qkv_gate | GV15 / GV15 / GV15 | +0.192 | +0.048 | +0.000 | -0.096 |
| decode16/o | GV34 / GV34 / GV34 | -0.032 | +0.032 | -0.032 | +0.000 |
| decode16/pooler | GV15 / GV15 / GV15 | +0.144 | +0.144 | -0.032 | -0.064 |
| decode16/unpooler | GV64 / GV64 / GV64 | +0.128 | +0.160 | -0.064 | +0.032 |
| decode16/router | GV86 / GV86 / GV86 | +0.000 | +0.000 | +0.032 | +0.000 |
| decode32/qkv_gate | t32s6hze / t32s6hze / t32s5hze | +0.032 | +0.144 | +0.064 | +0.160 |
| decode32/o | t32s5hze / GV47 / GV46 | -0.064 | -0.144 | -0.064 | -0.064 |
| decode32/pooler | t32s5hze / t32s5hz / t32s5hze | +0.064 | +0.304 | -0.256 | -0.032 |
| decode32/unpooler | t32s5hze / t32s6hze / t32s5hze | -0.032 | +0.176 | -0.192 | +0.032 |
| decode32/router | GV84 / GV87 / GV84 | +0.000 | -0.064 | -0.096 | -0.128 |
| verify1/qkv_gate | GV6 / GV5 / GV6 | +0.160 | +0.128 | +0.000 | +0.032 |
| verify1/o | GV29 / GV29 / GV29 | -0.032 | -0.032 | +0.000 | -0.096 |
| verify1/pooler | GV0 / GV50 / GV51 | +0.176 | +0.016 | +0.096 | +0.128 |
| verify1/unpooler | GV93 / GV93 / GV93 | +0.144 | +0.032 | +0.032 | -0.096 |
| verify1/router | GV85 / GV88 / GV88 | +0.160 | +0.016 | +0.256 | +0.032 |
| verify16/qkv_gate | s4s5hz / s4s5hz / s4s5hzEH | +0.256 | +0.240 | -0.096 | -0.448 |
| verify16/o | LT:200008 / LT:200018 / LT:200034 | -0.032 | +0.064 | +0.224 | +0.064 |
| verify16/pooler | s4s5hz / s4s5hz / s4s5hzEH | +0.096 | +0.304 | -0.160 | -0.992 |
| verify16/unpooler | s4s5hz / s4s5hz / s4s5hzEH | +0.240 | +0.304 | +0.224 | +0.000 |
| verify16/router | router126 / router201 / router201 | +1.248 | +0.176 | +1.184 | +0.192 |
| verify32/qkv_gate | LT:11 / LT:51 / LT:91 | +0.064 | -0.288 | -0.768 | -1.184 |
| verify32/o | LT:300008 / LT:300012 / LT:300016 | +0.160 | +0.080 | +0.000 | +0.064 |
| verify32/pooler | LT:300011 / LT:200024 / LT:200028 | +0.832 | +1.424 | +1.440 | +0.992 |
| verify32/unpooler | LT:300008 / LT:300012 / LT:300016 | +0.064 | +0.144 | -0.096 | -0.032 |
| verify32/router | router107 / router200 / router200 | +1.792 | +0.032 | +2.208 | -0.064 |

Whole-load sums (µs)

| Load | Paired RS32 / PW38 / PW39 | F1 RS32 / PW38 / PW39 |
|---|---|---|
| decode1 | 59.248 / 59.536 / 59.408 | 58.848 / 59.296 / 59.232 |
| decode16 | 75.856 / 75.616 / 75.424 | 75.136 / 75.104 / 75.232 |
| decode32 | 78.016 / 78.608 / 78.224 | 76.736 / 77.248 / 77.280 |
| verify1 | 60.736 / 60.448 / 60.256 | 60.416 / 60.032 / 60.032 |
| verify16 | 89.280 / 88.544 / 87.408 | 89.152 / 86.592 / 87.776 |
| verify32 | 106.224 / 105.040 / 103.328 | 106.304 / 103.296 / 103.520 |

Largest paired gain versus PW38 is verify32/pooler +1.424µs with different LT selections; largest loss is verify32/qkv_gate −0.288µs with different LT selections. Confirmed M128 EH gains in paired serving are +0.240/+0.304/+0.304µs for qkv/pooler/unpooler, but their F1 deltas are −0.448/−0.992/0.000µs. Decode32 pooler +0.304µs paired becomes −0.032µs F1. These disagreements and unrelated selector changes prevent attributing the aggregate gain solely to confirmation or EH.

Reproduce CPU-only audit: `python3 proj_rsi/audits/pw39-stack-01/analyze.py`. The source, artifacts, raw timing journal, exact schedule, JIT gates and cleanup must all remain pinned. No GPU/API/queue action was performed by this analysis.
