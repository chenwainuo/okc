# pw21 critic review (router m=128/256 on r6's gv, PROBE l2scratch) — 2026-10-26

**Verdict: APPROVE.** I found no blockers. One MAJOR item is about residual risk in the probe; the rest are minor.

What I ran:
- An nvcc 12.8 build (sm_90a, -O3, -Xptxas -v) of the gv block plus gvdec: 82 gv instances, 14.3 s locally against
  r6's 13.8 s.
- r6's 78 functions are SASS-identical to r6-c12.
- `router_emu.py`: ROUTER EMU PASS. All 4 paths are one-hot exact with relative error 0 at m = 128, 121, 256, 250 and 9.
- I diffed pw21 against pw20. The changes are:
  - the V bit 25 token loop (submission.py:889-921) and the x-at-use guard (:847);
  - gvdec / gvdecrun;
  - GE = False and the bulk-ring paths off;
  - GRP;
  - `GVB.get(bm, 16)`;
  - buckets 128/256 in `_gxbake` with odd m 121/250;
  - `_l2probe`, run after the bake.

## 1. hi+lo accumulation and validation: OK

**Single pass.**
- The paths use FMT 2: NA = 2, with W = hi fragments and W2 = lo, so there is no decode and no escape list.
- `hmma(acc[j][n][0], A[j][h]+4q, …)` runs for h = 0 and 1 into one fp32 accumulator, so hi and lo are summed in one
  pass.

**One-hot ×8 is exact.**
- |lo| ≤ ulp(hi)/2, so 8·hi + 8·lo needs at most about 17 significant bits. That is exact in fp32, and the zeros from
  the other K slices add nothing.
- `rexact` = hi.float() + lo.float(), so the comparison is bit-exact.

**Random rows.**
- They are checked at TOL 2e-5, tighter than the judge's 1e-4. This is the same representation r6 already serves at
  m ≤ 32.

**Coverage.**
- Validation runs at m = 128 and 256 plus random rows at 121 and 250 (submission.py:2510).
- Serving is exact-m only: `_bk(m)` is 128 or 256 only for exactly those m, and m = 130 goes to r6's dispatch.

## 2. Token loop and registers: OK

**Per unit.**
- `ld(s, u+ST)` is issued once.
- Each of the 16 token tiles gets 2 × `ld.global.nc` of x at `min(8n+g, m-1)`, which is the same address formula as
  r6's xp.
- Each tile has its own accumulator `acc[0][n][0]`: 64 fp32 values (NC = 1 at MT 16).
- The function then returns before r6's generic mma and escape walk.

**Registers.** All four instances have 0 spills, a 0 B stack frame and no LDL/STL.

| Path | Registers | Threads | CTAs/SM | Grid at m=128 / 256 |
|---|---|---|---|---|
| rt-c16 | 255 | 128 | 2 (8,192 registers/warp × 4 warps = 32,768) | 128 / 256 CTAs |
| rt-c16s2, rt-g16s2 | 168 | 128 | 3 | |
| rt-c8 | 255 | 256 | exactly 1 (65,536) | 64 / 128 CTAs |

## 3. In-launch reduction: OK

- **Cluster paths.** These are r6's own DSMEM epilogue, which already runs the m ≤ 32 router with S = 16 on the judge.
  - gvl sets NonPortableClusterSizeAllowed for S > 8.
  - The bake declines a cluster path when `gvocc` (max active clusters × S) is smaller than the grid.
  - rt-c16 at m=256 needs 16 clusters of 16 CTAs at 2/SM. It may be declined; c16s2 and g16s2 cover that case.
- **Last-arriver (V bit 2).** This is pw11's chain, unchanged:
  1. store the partial to `ws`;
  2. `__threadfence`;
  3. `__syncthreads`;
  4. thread 0 does `atomicAdd(cnt)`;
  5. the last CTA does `__threadfence` and sums the S partials with `__ldcg`;
  6. `emit` writes the fp32 output once (live rows only);
  7. the counter is reset.
- **Sizes.** `ws` needs 2 × 8 × 16 × 1024 × 16 B = 4 MB, out of the 64 MB of `gvws`. The counter index is
  `bz·GX + bx` ≤ 16.
- **REDB.** It is 48 KB per CTA (WK 2) or 80 KB (rt-c8).

## 4. r6 untouched: OK

- The 78 instances are SASS-identical.
- r6's bake, `_r6` and the m ≤ 32 picks are unchanged.
- For router 128/256 the baseline is `_r6` → `_v12` → cuBLASLt / mm. sel takes a router path only on
  `us < r6 × 0.99`.

## 5. PROBE l2scratch

**OK.**
- It runs after `_gxbake`, inside its own try. A failure prints `PROBE l2scratch declined <msg_with_underscores>`.
- The fork and join are legal inside capture:
  - fork: `sa`/`sb.wait_stream(cur)`;
  - inside the capture: `wait_event` on events recorded earlier in the same capture;
  - join: `cur.wait_stream(sa)` and `(sb)` before the end.

  The temporary from the final permuted copy comes from the graph pool.
- The streams are pooled torch streams, and the events and graph go out of scope.
- gvdec stays in bounds: t0 + tl < T, and it uses the dec1 (V1) pack with its matching table.

**MAJOR (residual).** An exception raised while capture is open (inside `torch.cuda.graph`) depends on torch's
`capture_end` to leave capture mode cleanly. If it did not, the judge's later capture of kernel_fn would fail and take
every shape with it.

The code inside the capture is straightforward and can only fail on an argument or allocation error. A cheap hardening:
- check that `torch.cuda.is_current_stream_capturing()` is false after the probe;
- or run `torch.cuda.synchronize()` in a `finally` and print the state.

## 6. Prepare and log: OK

- **Time.** The compile is r6-sized (82 instances). The r6 phase is about 110 s, as in pw17's judge log (111 s). The
  second bake has 2 buckets × 4 paths at 32-64 one-hot launches each, plus about 2 s for the probe. Total: **about
  115-125 s** against the 240 s guard, and the probe is skipped after 220 s.
- **Log.** About 5-6 KB (r6-c12's log is 4.4 KB).
- **PROBE format.** `PROBE l2scratch pooler m=128 chunks=8 <us> torch=… decode=… mmchunks=…` parses as config
  `pooler m=128 chunks=8` followed by a number. The declined line has no number and is ignored.

## Minor

- The probe's pipeline does not apply escapes. It is timing-only and its output feeds no path, so this is fine, but the
  line should not be read as a correctness signal.
- rt-c8 at 255 registers × 256 threads leaves no register headroom for a second CTA. That is intended.
