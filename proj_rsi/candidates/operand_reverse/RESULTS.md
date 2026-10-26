# operand-reverse01 — CLOSED, complete-output performance failed

`decision=CLOSE_POLICY_NO_SWEEP; shape=unpooler M128 N8192 K4096 KS1; correctness=PASS; F1_gain_us=-23.520002142; paired_gain_us=-23.519998416; promotion=NO`

One root-scheduled exclusive H100 trial, named container`okc-proj-operand-reverse01`, actual rootPID355459, inner240s/outer300s under the container-held lock/host deadline. Host log2026-10-26T21:47:49Z..21:49:17Z (88s), `LAB_DONE`, `TRIAL_EXIT0`. Reviewer independently found actual PID and named container absent; parent additionally reported GPU0MiB. No automatic retry.

Frozen source SHA256 `ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; device `59f3b775d28410c23616a749ad58661112f9c6a1a959fec05333068ca9f3800a`; lab `ea1d4597418618619438f805fb3bebeeac666bb9ec84f8910dda8d1001d6c827`; baseline pw36 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`. Pre-closure manifest SHA `e04584c9054a1e345d1b6def6798bb9bd0fbff19aec8cefa53c218cdd1f7dcc5`; all8files verified before documentation changes. Executable/proof/manifest files are unchanged. README's manifest entry records its pre-closure content.

## Resource and correctness gates

Real Torch/CUDA extension compile+link succeeded with the frozen source. Runtime168regs/local0/shared103600B; maxDynamic49152→103600 before occupancy query; occupancy1; producer64/math216 requests63488 registers from the64512-register initial CTA pool. The originally invalid64/224 proposal was caught and corrected before any GPU execution; its65536-register request was never tested on GPU.

Both candidate and actual baseline pass all4096one-hot columns bit-exact and full-output FP64 random/outlier validation. Candidate official normal/outlier relative-L2 errors1.369e-4/1.494e-4, worst row3.234e-4/3.328e-4. Baseline2.558e-3/2.836e-3, worst row2.975e-3/3.009e-3. All task probes exact. Candidate's separate original/zero/fresh-outlier retained replay also passes.

Actual baseline callable/output has a separate immediate task oracle and input-immutability check before timing. Both **actual timed** ga/gb graph objects pass original, zero, fresh normal, fresh outlier and all-probe mutation checks with real captured outputs poisoned to NaN before each replay. All10outputs are fully overwritten/finite; both zero cases are exactly zero; all128probe rows are exact in both all-probe cases; X is unchanged. Source identity and all pre-closure manifest digests match. Candidate's smaller numeric error does not satisfy the performance gate.

## Timing and actual dispatch

|quantity|us|
|---|---:|
|F1 baseline before|24.224000052|
|F1 baseline after|24.127999321|
|F1 candidate|47.648001462|
|F1 gain=min(before,after)−candidate|−23.520002142|
|paired baseline marginal median|24.288000539|
|paired candidate marginal median|47.775998712|
|median paired gain|−23.519998416|

All21paired differences are losses (gain range−24.191999808..−22.240001708us). Candidate F1 is1.974801× conservative baseline. Independent arithmetic agrees with recorded gains to<1e-10us. Both preregistered>=0.5us gain bars fail.

Comparator is the same-run prepared frozen pw36 `kernel_fn`, selected rq`s4s5hz`,KS2. Profile shows one `rq<128,5,1,0,0,0,3>`21472ns. Candidate is one `rq<128,8,1,0,0,0,20>`45090ns. Historical official pw36 used a different selected unpooler128 path; its19.103us figure is not this trial's baseline. Profiling happened outside selection timing, and its single-call durations are not substitutes for cold-L2 event/graph timing.

The fixed complete-output design fails despite its nominal60→36KiB shared-traffic reduction. The trial does not isolate X ingress/transpose, producer decode, synchronization or shared-memory throughput as the dominant cause. No hardware-counter or physical-port ceiling claim follows.

## Artifacts and closure

- `results/proj-rsi/takeover/20261026/operand-reverse01.json`, SHA256 `aecc7ee25d045d3924cb3969938e6658808049e4b70355cdc6a6fd0696debb65`.
- `results/proj-rsi/takeover/20261026/operand-reverse01.host.log`, SHA256 `1ce385db93f8499afda41ec64a03c0e78589fd73269b718eb34b12020bbf270e`.
- Independent preflight/runtime audit: `proj_rsi/audits/operand-reverse-architecture-preflight.md`.

**No integration, automatic rerun, flag/geometry/prefetch sweep, pooler/qkv/m256 port or official slot.** Incumbent stays rs32; no score improvement or<3000us result. Later diagnostic profiling requires its own reviewed scope and does not automatically reopen this implementation.
