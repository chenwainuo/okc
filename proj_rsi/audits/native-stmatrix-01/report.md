# Native-X STMatrix scheduler attempt01 independent result

**VALID_REGRESSION against actual local incumbent, with small observed progress
against native parent.** New paired31.440us versus incumbent24.048us: median
paired difference is **7.456us slower**,0/32 wins. No shipping or official-score
gain; no forecast. This valid losing version remains retained for diagnosis.

Authoritative collection:
`results/proj-rsi/scheduler/artifacts/native.stmatrix.v1.01/collection-1793075669605575962/`.
Result SHA256 `e99e1e79cd80b5e5d8c9d940cb128a161fca35fb7458dace7748fd59f760f1a0`.
Verified5 inventory file hashes and68 frozen dispatch source pins. Driver
87ec3afa2add4ce14e53f924bdc3d8310960aeee22e7947df6e9d4ec89b94f22,
trial7677ff2bfbac57b50a739dea72f3ea3ed1a28784d4d82fd7cf02de1c4b0a71e3,
kernel2c132b82/sourcef6a765f0 match independently admitted package. Current
original-X descriptor source identity verified. No archive/queue/source changes,
GPU action or remote work performed by reviewer.

|Control|F1 lower bracket control µs|NewF1 µs|F1control−new µs|Paired control median µs|Pairedcontrol−new median µs|Newwin/tie/loss|
|---|---:|---:|---:|---:|---:|---|
|Actual LOCAL pw36 selected rq|24.095999|31.136001|-7.040001|24.048001|-7.455999|0/0/32|
|DS fresh globalXP pack+main|34.784000|31.136001|+3.648000|34.655999|+3.103999|32/0/0|
|Native-X scalar-store parent|31.328000|31.136001|+0.192000|31.679999|+0.175999|21/0/11|

Paired new median31.440001us. Difference-of-paired medians is computed from
32 actual pairs; it need not equal subtraction of separate policy medians.
Native-parent paired-delta range[-1.247998,+1.408000]us, MAD0.559999us around
median0.175999us. This is a small noisy positive observation, retained without
a minimum-gain cutoff; no significance or transferable fixed gain claimed.
DS comparison includes its pack node; parent comparison isolates new source
store composition more closely but still permits codegen/scheduling effects.

Recomputed all295 raw calls:20warm+147F1+128paired, exact fixture seeds/block
order and8balanced paired orders repeated4times. Every finite positive sample
retained. Recomputed7F1 medians, paired32×4 tuples, conservative gains, median
ratios and wins equal result exactly. Independent analyzer negative controls
reject a changed firstF1 fixture seed and a changed declared parent gain.

Reexecuted all3 actual-JIT gates from archived loaded-SO SASS, verified saved
SASS hashes and exact canonical instruction streams. Kernel-specific A4,
LDSM/TMA/bar5, producer4STSM/proxy/complementary-owner and source checks pass.
Native/new122registers,0local,137232B shared,512threads,65536rounded pool and
actual opt-in occupancy1. DS same register/thread/pool,103600B shared;
packer resource gates pass. Native/new actual graph1correct main each; DS
actual graph2nodes (rx_pack+correctmain). Profile durations establish node
identity only and are not used as timing samples or phase-cost attribution.

All4 policies'4096basis exact/FP64 and task normal/outlier records pass.
Three candidate fresh-workspace/output checks pass; DS-only all65536 BF16
storage patterns/1MiB XP identity pass. BOTH native raw hardware mover-bit
roundtrips remain explicitly UNPROBED. Twenty retained actual graph mutations
cover4policies×original/zero/freshnormal/freshoutlier/allprobes, NaN output,
input immutability and DS pack poison overwrite. Eight strict terminal actual
outputs at F1/paired boundary are checked against EACHpolicy's own recorded
lastfixture before any replay/mutation/profile; currentX, W/pack/escape,
counters and no-output-repair gates all pass. Graphs finallyRESET_ALL_FOUR.

Inventory records terminal exit0 at2026-10-27T04:34:16Z and post-exit cleanup:
no compute/guard PIDs or containers, named container absent,0MiB GPU memory,
cleanup_failedfalse. Reviewer verifies archived cleanup evidence, not a fresh
live observation while another job runs. Result is local H100 research on
unpooler128; actual comparator is freshly prepared pw36, not official rs32
H200 timing. No promotion or official submission follows. Next causal question
can use retained small parent observation without treating family as closed.

Machine result: `analysis.json`; base read-only reproducer: `analysis.py`.
Scheduler validation: `proj_rsi/scheduling/validations/native-stmatrix-01.json`.
