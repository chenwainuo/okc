# rs128 critic review (pw23 + RS128 swap-AB register-A split-K kernel at m = 128 / 256) — 2026-10-26

**Verdict: APPROVE for a judge check, after one measurement fix (F1).**

I found no wrong-answer, hang or build-failure defect. F1 is not a correctness bug. The bake compares RS128 against an r6
baseline that is inflated by about 4 µs on exactly the shapes it picks. The check's kill criterion ("pooler m=128 not
under r6 in the judge bake") is decided by that comparison, so as it stands the check cannot fail that criterion
honestly.

## What I ran

**Diff vs pw23.**
- The only changes are:
  - the docstring (:23-30);
  - `_rqstart()` (:2181);
  - the rq bake call (:2211-2219);
  - the kernel_fn dispatch (:2909-2913);
  - the appended RS128 block (:2956-3776).
- pw23's CUDA sources are byte-identical, so the r6 and router kernels are SASS-identical under the same flags.
- The box copy differs from this file only in the docstring.

**Build.** The 6 production instances (RQPROD), RQFLAGS, nvcc 12.8, `-Xptxas -v`, torch glue stubbed.
- Every instance: 168 registers (the launch-bounds cap; setmaxnreg 40/232 moves the rest to the math warps), 0 spills, 0 stack.
- There is no C75xx serialisation or other warning. The device code took 3.5 s locally.
- A build with an added `compute_90/sm_90` gencode compiles cleanly, so the `__CUDA_ARCH_FEAT_SM90_ALL` guard works.

**SASS per instance.**
- 16 HGMMA.64x{128|256}x16 register-A.
- 4 WARPGROUP.ARRIVE, 5 DEPBAR.LE.
- UBLKCP and UTMALDG producers.
- 8 or 16 STSM.MT88.4, 1 UTMAREDG.2D.ADD.
- USETMAXREG DEALLOC / TRY_ALLOC.
- Register pool: 128 × (168 − 40) freed = 256 × (232 − 168) needed = 16,384. Exact fit, so TRY_ALLOC cannot starve.

**smem and residency.**
- s4s5hz: 144,520 B. s4hz: 230,576 B. n256s5: 226,440 B. All are ≤ 232,448.
- 1 CTA/SM by registers (384 × 168).
- Grids are one wave: 128, 128, 128 and 126 CTAs on 132 SMs.

**Static checks.**
- `python3 -m proj_rsi lint`: ok. Its one warning, "several load_inline builds", is pre-existing: pw23's gv build.
- `cpu-check --official`: 9/9 ok. Size 203,133 B.

**Box rerun** (`/root/okc/logs/critic_r6probe.log`): the full prepare, then re-timing after prepare. The details are in F1.

## 1. bf16 reduce-add epilogue: OK
- `out` tensor map (:3490-3504): dims {N, m}, box {64, NT}, SW128, OOB fill none.
  - Reduce-adds outside the map bounds are not written. Serving is at m == NT exactly (RQBAKE is keyed by m; `p.m > NT` returns −4).
- Slab: `xring + g·XB` is 1024-aligned and [NT tok][64 n] with 128-B rows. The XOR is `(ch ^ (tok&7))`, which matches SW128.
  - The stmatrix.trans fragment mapping matches out[tok][n].
  - `rq_pk` puts lo in the low half.
- Ordering: `fence.proxy.async.shared::cta` comes before the named barrier, then one elected lane issues `UTMAREDG.ADD`, commit, `wait_group.read 0`.
  - `.read` only guarantees the smem source was consumed. Global completion is ordered by kernel completion, which is the CUTLASS convention and is fine for a stream-ordered consumer.
- Numerics: each partial is rounded RN once to bf16 by inline cvt, then KS−1 bf16 adds land in the TMA unit.
  - A one-hot row has one exact partial plus ±0.0 from the other splits, so the result is exact (a −0.0 result can only occur where W = 0, and it compares equal).
  - The order of the adds across CTAs is nondeterministic, so replay outputs can differ in the last ulp. That is within tolerance.
- Prepare validates one-hot x8 rows over every K column, and random normal + 1%-outlier rows at RQTOL[KS] (4e-3 to 8e-3, rows 5×), for every variant at each m.
  - Box err: 2.4e-3 to 3.9e-3, against the judge's 1e-2.

## 2. In-kernel zeroing: OK
- The CTA zeroes rows [ks·m/KS, (ks+1)·m/KS) × 128 columns. The union over ks is [0, m).
- Chain: `st.global` zeros → per-thread `fence.proxy.async.global` → `bar.sync 4,64` → `red.release.gpu` cnt[nb] (cumulative through the bar).
  - Poller: `ld.acquire.gpu` until ≥ KS → `fence.proxy.async.global` → `st.release.cta` zflag.
  - Epilogue lane: `ld.acquire.cta` zflag → `fence.proxy.async.global` → reduce.
  - So no reduce-add into a block can precede any split's zeroing.
- Reset: each poller `atomicAdd`s cnt[NB+nb] only after its own wait. The KS-th one stores 0 to both counters, when no poller of the block is still reading.
  - The next replay is ordered by the kernel boundary.
  - Prepare asserts the counters are zero after every validation. Each layer has its own `cnt`, shared by its m=128 and m=256 paths, which run stream-serial.
- Residency: one wave at 1 CTA/SM, no PDL, stream-ordered after the flush. All KS CTAs of a block are therefore co-resident.
- MINOR: both waits are bounded (2^22 × nanosleep(128) in warp 10, 2^24 smem polls in the epilogue lane), but on timeout they proceed silently and give a wrong answer. pw27h's critic asked for a trap; add `asm("trap;")` on timeout so a failure is loud.

## 3. Escape lists: OK
- `_rqesc` (:3626-3650):
  - KS is the layer's RQKS for both packing and launch.
  - `s0 = ks·KC//KS` is the same floor as the kernel's `(long long)ks·KC/KS`.
  - `u = ch − s0[ks]` is the CTA-local stage, which `rq_fix` compares against `GS·G+j`.
  - Tile → (nb, warp) is `tile//8, tile%8`, matching `wl = wring + warp·1536` and the `[NB][KC][8][1536]` re-blocking.
  - The register is q·4 + v//2 = 4q + 2kh + half; the half bit is e.
- Capacity is computed from the real weights: max list + 1, rounded up to 32. So every list ends in a sentinel (seq 0x7fff, never a stage).
- The 32-entry window refills read at most index cap+31, and there are 32 trailing sentinels.
- Any mismatch would fail the one-hot-every-column validation in prepare, which yields INVALID, not a wrong served answer.

## 4. Background compile and the r6 baseline
- CPU contention is not what inflates r6. The box rerun reproduces it with the thread already finished ("rq extension ready at 37s"; the bake joins first):

  | shape | r6 in the rq bake | same r6 path after prepare (_time_us ×3) | lt_mm, prealloc | RS128 after prepare |
  |---|---|---|---|---|
  | pooler m=128 | 31.71 | 27.58 / 27.74 / 27.49 | 27.5 | 23.6–23.9 |
  | unpooler m=128 | 30.72 | 26.98 / 27.01 / 27.01 | 26.9–27.4 | 23.9–24.1 |
  | qkv m=128 | 19.07 (normal) | 18.9–19.2 | | 16.6–17.4 |

  - pw30 and pw24 bakes timed the same lt config (2:66.15…0.5) at 27.42 and 27.46. The inflation is specific to the rq bake's first r6 timing after the 32M-weight `_rqpack`/`_rqunpack` work. I did not isolate the cause.
  - The true box gains are −3.9 (pooler), −3.0 (unpooler) and −2.1 (qkv), so the box picks are right.
  - On the H200 the expected margin is ~1.5, so a +4 µs bias would let a losing RS128 path be picked at pooler/unpooler m=128, and the kill criterion would read it as a win.
- **F1 (required):** re-time r6 after the RS128 variants for each (layer, m) and compare against min(r6 before, r6 after). Alternatively, time r6 before `_rqpack`. Print both, so the judge log shows the bias.
- Budget: on the judge, pw23's r6 phase ends at ~101 s (job 17f4484b). The rq bake took ~2–5 s on the box, and the skip guard is at budget−100 = 140 s, so it fits.
  - A serial compile (judge ~35–50 s, by analogy with "rw compiled in 33–39 s") would land at ~140–150 s and trip the guard, so keep the background thread.
  - CPU overlap: the thread overlaps r6's own extension compile and possibly the start of the lt search. It is two nvcc jobs and ninja releases the GIL, so the risk is low.
  - Add a timestamp when the thread finishes, so the judge log shows whether it overlapped the lt search.

## 5. Mechanics
- Every kernel body is guarded with `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`, and RQFLAGS carries `-gencode=arch=compute_90a,code=sm_90a`.
- A build failure, an exception or a missed budget leaves `st["rq"] = {}`, so r6 and the router serve everything. A daemon thread that is never joined cannot block.
- Dispatch is at m exactly 128 / 256 with n == the layer's N (:2909-2913).
- MINOR: kernel_fn ignores `_rqcall`'s return code (:2912). A nonzero return (e.g. −21 for a misaligned x) would serve an uninitialised `out`. It is unreachable for the judge's freshly allocated x. Fall through to r6 when it is nonzero (`if _rqcall(...) == 0: return out`).
- NOTE: the base is pw23, not the incumbent pw33. That is fine for a check, but port the block onto pw33 before any scored run.
