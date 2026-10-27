# PW40 full-scope author analysis

**VALID local execution with positive aggregate point estimates in paired and conservative F1 timing. W-ring was exercised successfully but selected nowhere; the measured full-policy difference cannot be attributed to serving W-ring.** No significance barrier is imposed, and no official H200 score is inferred.

Candidate source `3bc0df52d96f62479c3d5c19ab2717c7a9f4d18c1dd087fb786e2f9f080f0278`; driver `9559a9ded1c4b3f60db4ef68decddba857cb13cd20dac048be1640a2dd53c1dc`. Job `pw40.stack.fullscope.01`, guard `427126`, terminal exit0. All63 artifact hashes and127 dispatched source/review pins independently match.

Validity covers222correctness cases,4350raw timings in the exact145-call schedule per shape,30shapes,180actual terminal output oracles and360retained-graph mutations with output and float-scratch poisoning. All30graphs drain/reset and the final graph list is empty. Full-storage source-bound counter/alias, dormant storage, immutable current-X/weight/pack and status guards pass. Actual tensor bytes are not recoverable after process exit; validation is supported by complete frozen-driver runtime records. Physical cleanup separately verifies no container, guard, compute PID or GPU memory; cleanup_failed=false. Exit0 alone was not used.

Fifteen loaded-module JIT archives include17RS32/19PW38/20PW40 RQ main functions; these counts are functions, not20separate compilations. Every RQ function matches its exact generated source dispatch and canonical PC/predicate/opcode/operand stream with168REG/LOCAL0. Mandatory EH control/physical gates rerun; the new split lifetime/publication/finalwait0/STSM/reduction/TMA-drain/TRAP/232-40register gates pass. Original and diagnostic-host-augmented generated CUDA are retained and have an exact append/remove inverse.

Device: NVIDIA H100 80GB HBM3, UUID `54a7daab-fe3e-c7fc-7f7c-6fbc40145a22`, SM90,132SMs. This is recorded before candidate import/prepare. Before the split extension returns to any preparation math, actual attributes are168registers,0local,148656shared,occupancy1,maxdynamic148656,maxthreads384. The126CTA grid fits132SMs. Shared bytes use separate X6/W4 rings:1024+6×16384+4×12288+8×(2×6+2×4)+16=148656, rather than the ordinary common-ring ST6 formula. Uniform and dynamic register pools are64512≤65536.

| Control | Paired six-load GM control/PW40 | Conservative F1 GM | Paired shapes win/tie/loss |
|---|---:|---:|---:|
| rs32 | 1.006555240 | 1.008638844 | 15/1/14 |
| pw38 | 1.003152408 | 1.001110967 | 18/5/7 |

Ratios above1 favor PW40. Six-load sums combine five marginal shape medians and use an equal-load geometric mean; these are descriptive local H100 measurements, not the official H200 score. F1 uses the lower bracketed before/after control median. Every30paired differences/ratios, IQR and within-batch bootstrap interval is retained in analysis.json; cross-run reproducibility is unmeasured.

Preparation and actual contender outcome

- rs32: wall 60.087206s / original import-clock 60.296401s, no clock reset or serving CPU fallback, two build threads joined.
- pw38: wall 63.186411s / original import-clock 63.186844s, no clock reset or serving CPU fallback, two build threads joined.
- pw40: wall 63.370953s / original import-clock 63.371280s, no clock reset or serving CPU fallback, two build threads joined.

The new W-ring contender received70successful host calls in the admitted QKV128/N2304/KS7 cell: one initial launch,64full-K basis calls,2random checks,2warmups and1graph capture. These are host-call records;15timed graph replays do not re-enter the Python host observer and must not be added as host calls. No decline/exception or budget skip occurred. The source preparation BAKE log reports numerical pass(ok3.8e-03), W-ring17.09µs, existing nonEH17.15µs and EH16.83µs; EH wins. The0.06µs screen contrast against nonEH is retained, but W-ring loses0.26µs against actual EH in that sequential screen and has no selected-serving sample. This run does not establish a W-ring benefit, additive EH/W-ring gain or reduced selection regret.

PW40 selects existing EH for qkv128/pooler128/unpooler128; PW38 selects EH only for pooler128 in this fresh preparation. Source-identical GV/LT and small-M paths also select differently, and source-identical plans can time differently. Whole-policy point gains therefore include preparation/context drift. All BAKE/PICK/PREP lines are retained verbatim in analysis.json, including positive, tied and negative alternatives. No PW39 confirmation policy is present.

All30 shape results

Positive deltas favor PW40. Picks are RS32 / PW38 / PW40.

| Shape | Actual picks | Paired Δ RS32 µs | Paired Δ PW38 µs | F1 Δ RS32 µs | F1 Δ PW38 µs |
|---|---|---:|---:|---:|---:|
| decode1/qkv_gate | GV6 / GV6 / GV6 | -0.080 | +0.000 | -0.032 | -0.160 |
| decode1/o | GV29 / GV28 / GV29 | +0.160 | +0.096 | +0.128 | +0.096 |
| decode1/pooler | GV51 / GV50 / GV51 | -0.000 | +0.160 | +1.024 | +0.512 |
| decode1/unpooler | GV58 / GV93 / GV93 | +0.352 | -0.032 | +0.480 | +0.160 |
| decode1/router | RR1006 / RR1006 / RR1006 | +0.000 | +0.032 | -0.096 | +0.000 |
| decode16/qkv_gate | GV15 / GV15 / GV15 | -0.080 | -0.160 | -0.096 | -0.032 |
| decode16/o | GV34 / GV34 / GV34 | -0.032 | +0.048 | -0.096 | +0.000 |
| decode16/pooler | GV15 / GV54 / GV15 | -0.128 | +0.144 | +0.096 | +0.064 |
| decode16/unpooler | GV64 / GV64 / GV64 | -0.304 | -0.080 | -0.096 | +0.064 |
| decode16/router | GV86 / GV86 / GV86 | +0.032 | +0.048 | -0.064 | -0.160 |
| decode32/qkv_gate | t32s5hze / t32s6hze / t32s5hze | +0.272 | +0.336 | +0.288 | +0.192 |
| decode32/o | t32s6hze / GV45 / GV45 | -0.032 | +0.000 | +0.032 | +0.032 |
| decode32/pooler | t32s5hze / t32s5hze / t32s5hz | -0.176 | -0.160 | -0.512 | -0.544 |
| decode32/unpooler | t32s5hze / t32s5hze / t32s5hze | +0.032 | +0.032 | +0.096 | +0.064 |
| decode32/router | GV84 / GV84 / GV84 | -0.016 | +0.000 | -0.032 | -0.224 |
| verify1/qkv_gate | GV5 / GV5 / GV6 | -0.064 | -0.096 | -0.096 | -0.192 |
| verify1/o | GV28 / GV29 / GV29 | +0.128 | +0.000 | +0.000 | +0.064 |
| verify1/pooler | GV51 / GV51 / GV51 | -0.064 | +0.176 | +0.224 | +0.032 |
| verify1/unpooler | GV93 / GV93 / GV93 | -0.160 | +0.000 | +0.096 | +0.032 |
| verify1/router | GV85 / GV85 / GV85 | -0.064 | -0.064 | +0.000 | -0.064 |
| verify16/qkv_gate | s4s5hz / s4s5hz / s4s5hzEH | +0.032 | -0.016 | -0.352 | +0.064 |
| verify16/o | LT:300009 / LT:200018 / h128s8e | +0.096 | +0.192 | -0.160 | +0.032 |
| verify16/pooler | s4s5hz / s4s5hzEH / s4s5hzEH | +0.048 | +0.080 | -0.352 | -0.160 |
| verify16/unpooler | s4s5hz / s4s5hz / s4s5hzEH | +0.288 | +0.448 | -0.096 | -0.320 |
| verify16/router | router126 / router201 / router201 | +1.168 | +0.064 | +0.992 | +0.032 |
| verify32/qkv_gate | LT:11 / LT:40 / n256s4hz | +0.176 | +0.288 | -0.064 | -0.096 |
| verify32/o | LT:300008 / LT:300012 / LT:300016 | -0.016 | +0.176 | +0.000 | +0.128 |
| verify32/pooler | LT:200014 / LT:200018 / LT:200028 | +0.384 | +0.256 | +0.576 | +0.320 |
| verify32/unpooler | LT:300008 / LT:300012 / LT:300016 | +0.032 | +0.176 | +0.000 | +0.256 |
| verify32/router | router107 / router201 / router200 | +1.712 | +0.016 | +2.336 | +0.384 |

Whole-load sums (µs)

| Load | Paired RS32 / PW38 / PW40 | F1 RS32 / PW38 / PW40 |
|---|---|---|
| decode1 | 59.472 / 59.232 / 59.024 | 60.128 / 59.232 / 58.624 |
| decode16 | 76.752 / 77.168 / 77.248 | 75.456 / 75.648 / 75.712 |
| decode32 | 78.960 / 79.104 / 79.072 | 77.312 / 76.960 / 77.440 |
| verify1 | 60.656 / 60.880 / 60.720 | 60.416 / 60.064 / 60.192 |
| verify16 | 89.312 / 88.240 / 87.696 | 87.424 / 87.040 / 87.392 |
| verify32 | 106.672 / 105.072 / 104.320 | 105.984 / 104.128 / 103.136 |

Versus PW38, the largest paired gain is verify16/unpooler +0.448µs (existing EH instead of nonEH), followed by decode32/qkv_gate +0.336µs (existing t32ST5 instead of ST6). The largest paired losses tie decode16/qkv_gate and decode32/pooler at−0.160µs; pooler32 loses−0.544µs F1 with the nonEARLY plan. QKV128 selects EH but measures−0.016µs paired/+0.064µs F1 against PW38 nonEH. No small loss is discarded.

CPU reproduction: `python3 proj_rsi/audits/pw40-stack-01/analyze.py` regenerates analysis.json and this report, independently rechecking source/artifact pins, raw schedule/statistics, actual JIT/resources and cleanup. No GPU/API/queue action was performed.
