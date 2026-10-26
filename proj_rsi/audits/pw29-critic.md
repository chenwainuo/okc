# pw29 critic review (wgmma router rwk at m=128/256) — 2026-10-26

**Verdict: APPROVE.** I found no blockers.

What I ran:
- I built `_rwsrc()` with the file's RWFLAGS (`-O3 -lineinfo -gencode=arch=compute_90a,code=sm_90a -DNDEBUG`) plus
  torch's COMMON_NVCC_FLAGS. The torch binding is replaced by a plain-pointer switch over the same RWV cases. Result:
  - 0 errors and no C7510 wgmma-serialisation warning;
  - all 9 instances have 0 spills;
  - 7.6 s locally.
- GVCU, GVI and CUDA are byte-identical to pw23, so r6's 78 instances and pw23's rsk/rsk2 kernels are unchanged.
- pw23's `rs_ws`/`rs_cnt` re-zero is still at the end of `_rsbake` (submission.py:3050).
- `rw_emu.py` passes (RW EMU PASS): every variant is one-hot exact with relative error ≤ 3.8e-8 at m = 128/121/256/250/300,
  under early and late landing, with x at raw+0 and raw+400. Three negative controls are caught: unswizzled stores,
  SBO 512, and a ring wait one group short.
- The file is 170.7 KB.

## 1. cp.async → wgmma handoff: OK

- Every stage runs `cp.async.wait_group N` → `fence.proxy.async.shared::cta` → `__syncthreads` → wgmma.
- The SASS shows, for every stage of both the up-front and the ring variants: `DEPBAR.LE SB0,N` → `MEMBAR.ALL.CTA` →
  `FENCE.VIEW.ASYNC.S` → `BAR.SYNC 0` → HGMMA.
- **Wait counts, up-front variants.** Stage s waits NSTG-1-s.
- **Wait counts, ring variants.** Stage 0 waits NBUF-1 and later stages wait NBUF-2. One group is committed per
  iteration s ≥ 1, empty once the K slice is exhausted, so group j is always stage j and the wait leaves stages 0..s
  landed.
- **Ring refill.** It happens only after `wgmma.wait_group 1` (stage s-1 retired in this warp) plus `__syncthreads`
  (all four warps), and it targets slot (s-1) % NBUF. That is never the slot stage s is reading. `wgmma.fence`
  precedes the next batch.

## 2. Descriptors and layout: OK

- The descriptor is start>>4, LBO 1, SBO 64 (= 1024 B), mode 1<<62.
- Slot size SB = (TOKS+128)·128 (32 or 48 KB), base rounded to 1024. The x, hi (+TOKS·128) and lo tiles all start
  1024-aligned, because TOKS % 8 == 0.
- k16 step q adds (q&3)·32 B.
- The store `r·128 + ((c ^ (r&7)) << 4)` within each 64-k chunk is the hardware 128B-swizzle image. It matches
  wgmma-layout.md and the emulator, and the SBO-512 and unswizzled controls fail.
- The h2 variant puts the second n128 set at +128·128.

## 3. hi + lo into one accumulator: OK

- D is zeroed in registers, then `wgmma.fence`. Per k16 step there are two m64nN wgmmas (hi, then lo) with scale-d = 1
  on the same registers, in one commit group per stage, which is legal and ordered.
- ptxas turned the very first wgmma into scale-d = false (`RZ, !UPT`) because D is known to be zero.

## 4. Epilogue: OK

- After `wait_group 0` + `__syncthreads`, D is staged as [tok][68] floats over the slot area, which the static_assert
  checks fits.
- No cp.async is in flight there. In the up-front variants the last wait is 0. In the ring variants, every group still
  pending at the end is empty.
- 16-B `red.global.add.v4.f32` (SASS `REDG.E.ADD.F32x4`) runs for rows < min(TOKS, m − t0) only.
- `rwrun` rejects a misaligned out (-11).
- `_rscall` zeroes `out` (the returned tensor) on the same stream first.

## 5. Registers and launch: OK

| Variant | Registers | Threads | CTAs/SM | smem |
|---|---|---|---|---|
| 114 (KR 512, ring 4, n256) | 255 | 128 | 1 (`__launch_bounds__(128,1)`) | 193 KB |
| others | 90-154 | 128 | | |
| 102, 112 (MINB 2) | 90 / 154 | 128 | 2 (39k registers) | 65 / 97 KB |

- All smem sizes are ≤ 193 KB, within the 227 KB opt-in, and the attribute is set once per instance.

## 6. Binding, budget, misaligned x

**Binding.** `rwrun` checks the shapes, contiguity and 16-B alignment, uses `at::cuda::getCurrentCUDAStream()`, and
returns the launch error codes.

**Build failure.** A failure prints `PREP rw build failed` and drops only the rw paths. The outer try around
`_rsbake` pops only the router keys.

**Budget.**
- The rw build starts only if elapsed < 180 s.
- The .cu is 7.6 s locally, plus the torch-glue compile, so about 30-40 s on the judge.
- Total **about 155-165 s** against 240.

**Misaligned x.**
- `x.clone()` inside `_rscall` depends only on the host pointer, not on values. Under capture it records an
  allocation from the graph pool plus a 2 MB (m=128) or 4 MB (m=256) D2D copy node. That costs about 1-2 µs per call,
  and only when x is misaligned.
- The judge's fresh inputs come from torch's allocator, which aligns to at least 256 B, so expect it never to trigger.
