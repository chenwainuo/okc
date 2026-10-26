# o_direct_w — one fixed-policy short-K experiment

`state=CLOSED_FIXED_POLICY_NO_SWEEP; GPU=o-direct-w01_COMPLETE; incumbent_edits=NONE; submission=NONE`

Final result: both numerically valid shapes lose every21 timing pair; paired gains m32=-2.208us,m128=-3.936us. No port/retry/sweep. See [RESULTS.md](RESULTS.md). The pretrial hypothesis/gates below are preserved as historical context.

Hypothesis: for o (N8192,K1024), replace HALF's W global→bulk-TMA→smem→LDS path with four stages of packed-W GPR prefetch. Retain rs32's lossless12bit natural-K layout, dec1, sorted exact escape fixups, register-A WGMMA,64-channel tile/one math WG/full K, X-only TMA, and plain TMA output store. Two instances only: NT32 and NT128. W prefetch depth4; X ring8 inherited from HALF; no tuning flags or geometry sweep.

Prior-work audit: handover05 and rs32 NOTES show HALF KS1 m32 lost (7.97µs kernel versus rq splitK2 7.58), and HALF m1288.54µs narrowly beat torch/cuBLAS8.90 but did not beat r6's tunedLt in bake. `cleanroom/contrarian` is closed by `audits/cleanroom-measure.md`; its packed W uses per-lane cp.async into smem, and its larger WGMMA path writes decoded BF16 into smem and uses cooperative split-K. Its apparent o32 gain was torch fallback plus drift. That measured loser is not the proposed mechanism. `audits/r6-lineage.md` explains direct-GPR W feeding WGMMA as a design possibility; no existing implementation of this HALF/full-K/plain-store/Q4 policy was found. We port only the loader mechanism into rs32's proven arithmetic/epilogue.

Static ROI: each call still reads12.58MB packed W; it removes12.58MB W shared fills and12.58MB W LDS, W full/empty barriers, and the W dependency from the first full barrier. First4 chunks request3MiB across128 CTAs. Core register budget=Q48+A32+D16/64=96/144 regs before escape/address/control state. Compiled resource allocation is232regs,0stack/spills for both instances. The benefit is conditional: direct-load issue/latency and GPR lifetime may exceed saved staging; the first CTA barrier remains. Expected candidate range is a hypothesis of roughly0.5–1µs local graph improvement, not a prediction or H200 score claim. At0.5µs on both shapes the old leverage coefficients imply only about8–9 score-µs; this is a bounded small opportunity.

Exact generated extension source SHA256: `2f571e925469ea90476aa1ec87d72d86997d460876a7250603113c1653d3feb1`. Device-only source SHA256: `8466df061cca77cb7fc378a20771ddff679712e708da804d6f86c36037314aa0`. Helpers are pinned to rs32 source SHA `a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`; there is no full submission copy. `prototype.py` replaces only RQ's kernel/class and host launch size/shape guards inside its extracted CUDA section. The codec, escape format/fix, maps, descriptor, WGMMA wrappers, and store helper are inherited. Host rejects anything except N8192,K1024,M32/M128,KS1. No odd-M/fallback dispatch is introduced.

CPU/local gates passed:

- Exhaustive direct-pointer decode/escape comparison:131072 arbitrary BF16 values, including zero/sign/subnormal/Inf/NaN bit patterns;122691 escapes, cap15392, so >32-entry reload windows are exercised. Every decoded pair is compared independently against natural row/K coordinates. This is a bit proof, not GPU arithmetic validation.
- Q4 static-slot tags, A2 retirement before overwrite, and X8 release/refill ordering pass the CPU schedule model. No global synchronization/counter/reduce path exists.
- Actual inherited RQ flags (`compute_90a/sm_90a`,O3,DNDEBUG,relaxed-constexpr) compile the exact kernel into a cubin. Both instances:232regs,0stack,0spills,no C75 diagnostics. Each SASS has16 HGMMA/4 WARPGROUP.ARRIVE across4 unrolled rounds,24 LDG.128,0 LDS.128,0 W UBLKCP,16 X UTMALDG,1 plain UTMASTG output. All accumulator fragments feed the output store.
- Python modules pass syntax checks. Full Torch host-extension compilation is not locally certified: this workspace's CPU-only torch/CUDA installation lacks `cusparse.h` required by the inherited ATen CUDAContext header. The device compile is valid; the real GPU environment must still build the extension. No header stubs or incumbent changes were used.

Reproduce CPU/device gate without GPU:

```bash
python3 proj_rsi/candidates/o_direct_w/preflight.py
```

After independent review, root can schedule exactly one bounded lab under its exclusive GPU container/lock:

```bash
python3 proj_rsi/candidates/o_direct_w/lab.py --json /okc/logs/o-direct-w01.json
```

The lab builds exactly two fixed instances, prepares frozen pw36 once (SHA `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`), and compares `pw36.kernel_fn`'s actual prepared exact-M pick. It records rq variant/KS or r6 Lt code/gv choice; torch is not the independent baseline. Per shape: all1024 one-hot columns must be exact, random normal/outlier FP64 checks must pass, official normal/outlier input checks must pass, captured pointers must observe original/zero/fresh-outlier X mutations without input writes. A failed check aborts before performance admission.

Measurement admission/kill rule: warmups discarded; F1 incumbent timed before/after candidate with the lower baseline retained; also21 rotated AB/BA cold-L2 graph pairs using the same prepared dispatch. Survive for a shape only if the prototype gains≥0.5µs in BOTH the F1 comparison and median paired graph difference. Preserve all raw pairs. Failure closes this fixed policy; no retry, parameter sweep, or full submission copy under this task. Even a survivor requires root review/H200 validation before promotion. No GPU or remote compile/launch has been performed by the author.
