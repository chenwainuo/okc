# router_small — frozen one-prototype trial

State: CLOSED_GAIN_BAR_FAILED after router-small01. Both numerical sides pass; F1gain-2.656us,median paired gain-2.592us,40/40pairs lose. No port/retry/sweep. See [RESULTS.md](RESULTS.md). Frozen executable sources/manifest unchanged; pretrial design and gates below are preserved as history. Design/cost/closed-work novelty: `../../audits/router-small-plan.md`.

Mechanism:128CTA x1024threads, each owns2expert rows over fullK; tensor-core diagonal K folding + lossless12/16 warp-adaptive transport, one CTA-only reduction. Target bucketm8 (nearby5/7 validated), no prototype expansion to m16/32. No global reduction/flags/zeroing/partial workspace/TMA/cluster, no escape records. FP32 hi+lo accumulation, allhi/loBF16 bits preserved.

CPU proof (`cpu_proof.py`): all4,194,304BF16 weights bit-exact, 2,097,152 direct indices bijective perplane, exhaustive65,536BF16 bitpatterns throughwidefallback and5narrowwindows, symbolicall8192K basis coverage. Independent PTX register->dense matrix emulation matches X@(hi+lo)^T at2.41e-18 FP64 relativeL2; three deliberate layout bugs produce~1.4error. This does NOT prove actual GPU rounding/convergence/speed.

Local CUDA12.8.93 compile: `nvcc -arch=sm_90a -cubin -O3 -lineinfo -Xptxas -v kernel.cu`. Actual ptxas62registers,0stack/spill stores/spill loads,2048Bsmem,1barrier. SASS16HMMA.16816.F32.BF16,oneCTAbarrier,0local loads/stores/atomics/global reductions; allstores overwrite. Artifacts: `local-compile.log`, `kernel.sm90a.{cubin,sass}`, `local-build.json`. Local Python torch2.13+cpu cannot serve as the Torch CUDA host-build environment; actual Torch2.8/cu128 host+device compilation and resource recheck are mandatory gates at the root's named trial. Reject >64regs OR anylocalbytes/spills before timing.

Proposed ONLY named inner command (root executes under exclusive container-held flock):

```sh
timeout --signal=TERM --kill-after=10s 300s python -u proj_rsi/candidates/router_small/lab.py --reps 40 --json logs/router-small01.json > logs/router-small01.log 2>&1
```

Outer360s. Estimated60–160s if frozenpw36's cached dependencies remain available; full source unmodified actualpw36 prepare can consume~80s and prototype hostcompile~20–40s. No retry if timeout/fault/correctness/resource failure. No source patch duringtrial. Root owns launch/termination/harvest policy.

`lab.py` loads SHA-pinned actualpw36, calls full `prepare(weights)` without changingflags/bake lists, records actual selected router path with scalar metadata only. Its timed baseline graph captures actual `pw36.kernel_fn`; prototype graph captures its actualsinglecall. Both eager sides pass originalFP32/FP64 strict2e-5 (stronger than task1e-4), row1e-4, exacthi+lo probes, all8192 onehot columns, normal/outliers/zero andnearby5/7. After timing, BOTH actual measured graph outputs are poisoned toNaN and checked after original, zero, freshnormal, freshoutlier, originalagain inputs, three replays each. No finite-only baseline comparison.

Prepare timer is actual shipped `_time_us` on actual calls; baselineF1=min(pre,post). Graph40AB/BA paired cold256MiB samples, median paired absolutegain. Need >=0.5us BOTH; reportmedian-difference too. Incomplete JSON, skipped correctness or wrong parent hash closes. No officialsubmission or autoport even if local win; independent review of H200 transfer required.
