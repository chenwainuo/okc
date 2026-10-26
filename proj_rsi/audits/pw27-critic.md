# pw27 critic review (red.add split-K gv paths, PROBE rd8) — 2026-10-26

**Verdict: APPROVE.** I found no correctness blocker. There is one MAJOR performance finding on the bf16 atomics.

What I ran:
- I built GVCU with r6's real arch (`-gencode=arch=compute_90,code=sm_90 -O3 -lineinfo`): 110 functions, 0 errors,
  16.5 s locally.
- r6's 78 instances are SASS-identical to r6-c12 at sm_90, and pw23's 8 rsk/rsk2 kernels are identical.
- The 16 new gv instances use 54-156 registers and the 8 probe kernels 10-40, all with 0 spills.
- The file is 169.5 KB.

## 1. bf16 atomic epilogue (unpooler / o, m ≤ 8)

**What the code does.**
- `reda(bf16*)` is the scalar `red.global.add.noftz.bf16`, one element at a time, not `__nv_bfloat162`.
- Each thread's two values per row pair are 8 rows apart (o and o+8), so they are not adjacent and cannot pair as
  written.

**MAJOR (performance).**
- On sm_90, ptxas lowers the scalar bf16 red to a **CAS loop** (`ATOMG.E.CAS.STRONG.GPU`) in all five unpooler/o
  instances. The fp32 router path is a native `REDG.E.ADD.F32`.
- The result is still atomic and correct, but each output element costs a load + CAS round trip, retried under S-way
  contention.
- The packed `red.global.add.noftz.bf16x2` is native on sm_90. It would need the (row, row+1) pair from the neighbouring
  lane (lane ^ 4) via a shuffle, with the even row doing the add.
- Expect these paths to lose to r6 for this reason. The bake will decide.

**Accuracy.**

| | Analysis | Simulation (bf16 W, N(0,1) x, fp64 reference) |
|---|---|---|
| Single fp32→bf16 rounding (r6) | | rel L2 1.66e-3 |
| S = 2 | S partial roundings + S-1 running-sum roundings ≈ u·sqrt(1 + Σ_{k=2..S} k/S): ×1.41 | 2.42e-3 (o: 2.48e-3) |
| S = 4 | ×1.8 | 3.2e-3, worst row 3.4e-3 |

- Against the judge (TASK.md: "at most 1%", any row ≤ 5×), there is a ≥ 3× margin.
- Against the bake's bf16 TOL of 4e-3, the S = 4 path (uk-s4) passes, but with only about 20% margin.
- The outlier test cases (m = 9, 512, 1000) are not served by the bf16 paths.

**One-hot.** A one-hot row has its single nonzero in exactly one K split. That partial is 8·w (a power of two times a
bf16, exact), and every other split adds bf16 +0 into a zeroed output. Exact.

## 2. Memset: OK

- `_gvcall` runs `out.zero_()` whenever the instance has V bit 26, and then `gvrun` on the same current stream into
  the same tensor that kernel_fn returns. Under capture that is a memset node before the kernel, for every served
  red-add path.
- The zero is 0x0000 for bf16 and 0.0 for fp32.
- The cluster occupancy guard is correctly skipped for bit 26, which launches no cluster: gvl's `clu` excludes it.

## 3. Index math: OK

- `redemit` is r6's `emit` line for line, with `sto` replaced by `reda`:
  - `row = row0 + (r/MT)·16 + (l>>2)`;
  - `tok = 8n + 2(l&3)`;
  - stores at (tok, row), (tok, row+8), and when tok+1 < m at (tok+1, row) and (tok+1, row+8).
- It is reached after the same WK smem sum and `live()` filter as r6's S == 1 emit (submission.py:1044), and the
  kernel returns before the cluster and global epilogues (:1051).

## 4. Probes: OK

- `_pr27` only times; no path is registered.
- **Cluster-8 probe.** G = 264 is divisible by 8. The 8 CTAs of a cluster are co-scheduled by hardware, so the two
  barrier.cluster calls cannot deadlock. A launch failure returns -3 and prints `launch-error`.
- **Bulk probe.** per = 64 KB / 32 KB, divisible by CHK and under 200 KB. The bulk copies and expect_tx use exactly
  `per` bytes.
- **PROBE format.** `PROBE rd8 <name> <us> <tbs> tbs` parses as engine rd8, config name, then µs and TB/s.
  `empty-memset <us>` parses too, and the launch-error lines carry "-N", which does not match.
- **Guard.** It runs only when the elapsed time is under 210 s.

## 5. Prepare budget: OK

- Compile is +2 s local over pw23, about +4 s on the judge.
- The bake adds about 15 router paths over buckets 1/8/16/32, each with a memset per launch. The m = 1 one-hot sweeps
  are 8,192 launches each, about 1 s each. Add the unpooler/o paths and about 2 s of probes.
- Estimate **about 140-150 s** against 240.
