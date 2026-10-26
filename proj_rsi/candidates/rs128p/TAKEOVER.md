# P4 takeover: producer-written escape patch table

State: CLOSED — producer-written patch table fails the preregistered gain bar in a clean exclusive run. No kernel edits or GPU launches by takeover agent.

Source hash: 24c517b809d50fb82b59e35c90990d3a87ef7a623219f37dede50a377a005af4
Lab hash: bf50a0442999dcc9d77413b9ea112f48333f4b8485177572622a532d1ba1ede6

## Decision to make

Compare production `s4s5hz` with `s4s5hzt` at m128 for pooler and qkv, with instrumentation counterparts only for causal attribution. The patch table must improve the actual graph/in-prepare comparison by at least0.3us without correctness/counter-reset failures. If it does not, close this mechanism; do not tune new variants automatically.

The inherited contended `pt-lab.log` showed table +1–2us, but orphan kernels were sharing the GPU so that is not decisive. A single clean bounded replay is justified. The resident controls and the m256 branch are unnecessary for this kill decision and excluded to save runtime.

## Root execution request

Inside the root-owned container that itself holds the device flock:

```
timeout --signal=TERM --kill-after=10s 240s python -u proj_rsi/candidates/rs128p/lab.py \
 --plan anat --layers pooler,qkv_gate --ms 128 \
 --only s4s5hz,s4s5hzt,s4s5hzi,s4s5hzti,s4s5hz-noesc \
 --reps 40 --graph --json logs/rs128p/takeover-pt01-lab.json \
 > logs/rs128p/takeover-pt01-lab.log 2>&1
```

Root supplies the outer container lifecycle/terminal checks. Keep source bytes stable until completion. Existing lab validates full kernels on one-hot K coverage, random and outlier inputs; `noesc` is timing-only and must not be treated as correct. INS variants are excluded from headline timing by the lab.

If a production table path clears the kill bar, run a new bounded official targeted harness and compare its in-prepare bake against the shipped rq (F1 baselines before packing/after variants), then port the mechanism into the current merged incumbent. Otherwise no additional expensive prepare or judge slot is warranted.

## Integration caveat

This standalone rs128p still has its old `PREPARE_BUDGET_S-100` initial rq gate. It is a diagnostic copy, not a judge-ready successor to pw36. Any winner must be merged into the current parent's compile/bake order and budget guards. The prior m256 rq closure stands.

## Results

Root launched the exact requested command under outer300s / inner240s bounds. Host PID342205 and container`okc-proj-p4-takeover-pt01` were terminal/absent after completion. Root supplied the repaired container-held device lock; this agent only observed and archived results.

Raw files: `takeover/takeover-pt01-lab.log` and `.json`.

| layer m128 | shipped graph us | patch-table graph us | regression | shipped CUPTI median us | table CUPTI median us |
|---|---:|---:|---:|---:|---:|
| pooler |23.968|25.728|+1.760|21.952|23.632|
| qkv_gate |17.056|18.720|+1.664|15.007|16.656|

Both production paths validate: complete8192-column one-hot coverage is exact, normal/outlier relativeL2 ~0.003–0.004 within task tolerance, counters reset. Weight pack/unpack is exact. Baseline and patch use168registers/no local memory; shared allocation rises144520->156984bytes. INS variants are not included in headline timing.

The mechanism loses decisively against the required >=0.3us gain; no official harness, judge slot, or port is justified. Preserve the existing experimental source for evidence, but do not ship its patch paths.

### Why it loses

Instrumented median first-WGMMA time rises2.05->3.58us for both layers, and loop start rises from~0.51us/~0.51us to2.05us. In contrast, steady-state WG0 loop falls955->937clocks on pooler and1124->994 on qkv; issue falls176->146 and180->153. The producer-side escape-list/table preparation adds ~1.5us of startup delay that erases the smaller steady-state benefit. This is measured attribution from instrumented variants; their clocks are not clean latency totals.

A materially different future mechanism could prepare static weight-derived stage patch metadata once in `prepare`, eliminating the runtime list prefill. That would require a new hypothesis, memory/instruction accounting and independent tests; it is not implemented, approved, or grounds to extend this failed producer-side sweep.
