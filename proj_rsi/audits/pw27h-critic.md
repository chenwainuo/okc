# pw27h critic review: self-zeroing split-K (gv V bits 27/29) and the rr 1024-thread router (2026-10-26)

**Verdict: APPROVE.** I found nothing that gives a wrong answer or breaks the build on the judge.

There is one robustness gap. The flag wait has no bound, so a broken assumption would hang the run instead of failing
visibly. Only one baked path, rk-s16w8x at router m=32, depends on such an assumption, and that assumption holds on
current hardware. The fix is cheap and recommended, but it does not block (finding 1).

What I ran:
- I built GVCU at the judge's flags (`-gencode=arch=compute_90,code=sm_90 -O3 -lineinfo`, with a torch stub header).
  The build has 0 errors. The only warning is #1886, which was already there.
- r6-c12's 82 functions (78 gv + rr) are **SASS-identical**, and so are pw23's 90 (rsk/rsk2/rr, including rr<M,R> →
  rr<M,R,512>).
- The 14 new gv instances use 56–150 registers with 0 spills. rr<1,1,1024> uses 24 registers and rr<1,2,1024> uses 32,
  both with 0 spills.
- `lint` passes (174,871 B), and `cpu-check --official` passes 9/9.
- **H100 stress** (logs/critic/stress_pw27h.py on the box, under the lock, after a full prepare): every GRD variant
  at its served m and at m−3, plus the 8 kernel_fn shapes, as CUDA-graph replays. Each case got 3,000 replays
  interleaved with eager launches on the same counters and periodic L2 flushes, and every output was compared with a
  validated eager result.
  - Result: 39 cases, 0 failures.
  - Router differs by at most 4e-7 of max|out| from run to run (fp32 atomic order).
  - Unpooler differs by at most 9e-3 of max|out|, which is bf16 atomic-order nondeterminism. Each call stays at
    2.3–3.6e-3 rel L2 against the judge's 1e-2, and probe rows are exact.

## 1. Flag protocol (submission.py:719-730, 779-791, 1070-1075)

The ordering is correct, and the SASS shows it.
- The zeroer does `STG ×n`, then `MEMBAR.SC.GPU`, then `ATOMG` on the flag.
- The waiter does `LDG.E.STRONG.GPU`, then `CCTL.IVALL` (acquire) in a YIELD loop, then `BAR.SYNC`, then the reds.

Each of the S CTAs of a tile takes exactly one ticket, and the tile's zeroer raises the flag exactly once per launch.
So `flag ≥ tk/S + 1` names this launch's zero. The counters are monotone and nothing resets them, so there is no reset
race.

The counters live per instance (`st["p1cnt"]`, :1964). They are first allocated eagerly in the bake (:2943), never
during a capture. The tile index is at most 64 at the served m, well inside the 1024-tile limit.

The z zeroer runs after `p.out` and `p.m` are rebased per bz (:731). The XZ zeroer uses `bz*TK` and
`min(TK, m−bz*TK)`. Both cover exactly redemit's (tok, row) set.

**MAJOR (robustness): the spin is unbounded (:1073).** It cannot produce silent garbage, but any failure hangs the
judge run until it times out.

Forward progress on 132 SMs, with registers from ptxas:

| path | CTAs | resident capacity | co-resident? |
|---|---|---|---|
| rk-s8z m=32 | 128 | 132 (1 per SM) | yes |
| rk-s16w8z | 256 | 264 | yes |
| uk-s4z/x | 256/320 | 396 | yes |
| uk-s4w4x | 320 | 528 | yes |
| rk-s32x | 528 | ≥1056 | yes |
| **rk-s16w8x m=32** | **272** | **264 (92 regs)** | **no: 2 waves** |

rk-s16w8x at m=32 is safe only because blocks dispatch in linear order and the zeroers are z-plane 0. Every NVIDIA GPU
does this, but it is not architecturally guaranteed.

Fix (cheap; either one):
- (a) Count the polls and `trap;` after about 2^24 iterations. The run then fails at once and visibly. Writing NaN is
  not safe here, because a late zeroer could overwrite it with 0.
- (b) Drop "x" for rk-s16w8 at m=32. It lost there on the H100: 8.42 vs rk-s8z 8.19.

## 2. rr<1,1,1024> / rr<1,2,1024> (:1202-1244, :1602-1603)

- Each thread reads NI = 2 float4 per row, so 1024 × 2 = 2,048 cover K/4. The 32 warps reduce in smem, and threads
  t < R write row0 + t.
- The kernel returns −1 for m > 1, and bucket 1 is m = 1 only.
- `_gvok` now uses mm = 1 for rr6/rr7 (:1938). pw27's expression would have given 8, so the launch would have
  returned −1.

## 3. Unpooler bf16x2 one-hot exactness (:576-592)

The argument still holds:
- The tile is zeroed (+0) before any red.
- Only one split carries the single nonzero product, rounded to bf16 once, as in the reference.
- Every other split adds +0.

The H100 checks show the probe rows exact.

LOW: the shuffle mask is `__activemask()` after a `continue`. Lanes l and l^4 always share liveness, and ptxas emits
WARPSYNC.COLLECTIVE, so this works. A `__ballot_sync` taken before the branch would be cleaner.

## 4. `_gvcall` per variant (:1961-1968)

| V bits | what `_gvcall` does |
|---|---|
| 26 (memset) | `out.zero_()` |
| 27/29 | no zero; passes the per-instance counters |
| neither | r6's call |

No bit-26 or bit-28 instance is in GVI. The memset path and p1zero are dead code: p1zero is compiled but never
launched, and PR27 is False.

## 5. Dispatch and prepare

- A pick replaces r6 only when it wins the timed bake at MARGIN 0.99 (:2957).
- Router m=1 now chooses among rr0/rr6/rr7 inside r6's own bake against v12. With no margin against rr0, that pick is
  noise-level (rr6 or rr7 across runs), but every option is validated.
- Picks moved between runs (unpooler m=1 uk-s4w4x → uk-s4z; router m=32 rk-s8z → rk-s16w8z), and all were validated.
- The m=128/256 router bake is unchanged.
- Prepare took 37–42 s on the box, against the 240 s budget plus guards.

## 6. Cosmetic

- `hypothesis.json` in pw27h is pw27's: id "pw27", with the old memset-path claims.
- `_name(-7)` prints GVI[-7] for the rs plan in SEL dumps.
