# pw28 critic review (gvk: stream-K-balanced qkv_gate, sentinel slots) — 2026-10-26

**Verdict: APPROVE.** The liveness and memory-model questions resolve in the design's favour. There are no blockers.

What I ran:
- I built GVCU with r6's real flags (compute_90/sm_90, -O3 -lineinfo) plus explicit gvk instantiations for all 10
  GKI tuples: 0 errors.
- r6's 78 instances and all 86 of pw23's functions are SASS-identical.
- The gvk instances use 94-132 registers with 0 spills.
- `gk_emu.py` passes for NW 9×KU 16 (contiguous and interleaved) and NW 18×KU 8: 128 CTAs, 112 shared tiles; coverage,
  slots, numbers and escapes (5,223) all ok.
- The file is 178 KB.

## 1. Forward progress

**Why it cannot deadlock.**
- CTA b runs its tile-B units first and publishes the shared tile's partial at its switch step (`flushB`). It waits
  (on b−1's slots) only at its very end.
- Publishing therefore never waits on anything, the dependency chain only points from b+1 to b, and no cycle is
  possible.

**Residency.**
- Each launch is 128 CTAs with `__launch_bounds__(NW·32, 1)`.
- Registers: 94-132 × 288 threads, or 94-96 × 576 threads, gives 1 CTA/SM. smem is 3·NW·NI·16 B, at most 55 KB.
- The bake declines any path with `occ (= maxActiveBlocks × SMs) < grid` (`declined-occ`), so every served path has its
  whole grid co-resident on a sole-tenant GPU (128 ≤ 132).
- In the judge's sequence (flush kernel, then the graph, on one stream) nothing else shares the SMs.

**When the cap is hit.**
- If 2^20 re-reads (about 0.15-0.3 s of L2 polling) all still see the sentinel, the code:
  1. adds the sentinel word 0xffffffff, a NaN, into the output;
  2. writes the sentinel back;
  3. continues.
- The output then has NaN rows, which the judge's per-call check rejects ("No NaN or Inf"). So a capped spin is a
  **visible failure** of the run, not a silently wrong number.
- With a co-resident grid, the producer publishes within microseconds of starting. The cap is reachable only if CTA b
  is not running for over 0.15 s, which needs another tenant or preemption. I consider it unreachable on the judge.
- I recommend keeping the cap: a hang would cost the whole 10-minute run as a timeout instead.

## 2. Memory model: OK

- The producer uses `st.relaxed.gpu.global.v4` and the consumer `ld.relaxed.gpu.global.v4` (SASS:
  `STG/LDG.E.128.STRONG.GPU`), so both bypass non-coherent L1.
- A v4 access is not single-copy atomic as a whole, but each aligned 32-bit word is. The consumer re-reads while
  **any** of the 4 words is the sentinel. Each word changes exactly once per launch (sentinel → final value), so a
  vector with no sentinel word is entirely final. No torn read is possible.
- **Sentinel.**
  - 0xffffffff is a negative NaN with a full payload.
  - The partials are fp32 sums of finite bf16 products. Hardware arithmetic produces only the canonical NaN 0x7fffffff,
    and only from inf−inf, which cannot occur.
  - `gk_nan` additionally remaps 0xffffffff to 0x7fc00000.
- **Write-back.** Each consumer thread writes the sentinel back only to the vector it has just read in full. The
  producer has already finished that store, and the next replay is stream-ordered after this kernel.
- **The all-sentinel invariant.**
  - Every published slot (pubB of b) is consumed exactly once (conA of b+1 reads the slots of warps
    `IL ? s < oA : s·KU < oA`, which are exactly the producer warps with sw > 0).
  - `gkws` is filled with −1 before each path, after any failure, at the end of `_gkbake`, and in prepare's `finally`.

## 3. Escapes and determinism: OK

- `_gkesc` builds the per-(CTA, warp) lists with seq = the warp's step u, in the kernel's unit order (tile-B units
  first, interleaved or contiguous). r6's walk runs inside `step` before the HMMAs, so producer and consumer partials
  both include the fixes.
- **Fixed sum order.** Own warps 0..NW−1, then producer slots s ascending, so the result is deterministic.
- **One-hot.** Each row has a single product, the other terms are +0, so it is exact.

## 4. Build, PROBE and budget: OK

- **PROBE line.** `PROBE skq qkv_gate m=1 sk9i 9.80 r6=9.97 delta=-0.17` parses as config "qkv_gate m=1 sk9i"
  followed by 9.80. "r6=…" and "delta=…" are non-numeric tokens, which the trailing group accepts.
- **Pick rule.** A path is picked only if its PROBE minimum is under r6 × 0.99.
- **Prepare time.** The gk bake starts only if elapsed < 200 s. Compile is about +5 s on the judge for 10 instances.
  The bake is 8 paths × buckets, with 8,192-launch one-hot sweeps at m=1, plus 6 alternated timings per bucket: about
  15-20 s. Total **about 140-150 s** against 240.
- **MINOR.** The contiguous-order instances contain a `STG.STRONG …, RZ` (a zero partial) at the u = 0 flush site.
  If it ever executes, it writes zeros to a slot that no consumer reads (the warp has sw = 0), so it is harmless.
