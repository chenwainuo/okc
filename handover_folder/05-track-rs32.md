# 05 — Track rs32: RS128 at m=32 and m=16 (qkv, o; pooler/unpooler if the rate allows) + short-K o

Files: `proj_rsi/candidates/rs32/{submission.py,lab.py}` (seeded from `git show 49b22d8:proj_rsi/candidates/rs128/submission.py`),
notes to go in `rs32/NOTES.md`; box logs `/root/okc/logs/rs32/`. Agent spawned 18:40 UTC (Opus medium); at 19:00 it had a
first timing run queued (tag t3) and was splitting the small-m instantiations into a second extension compiled on its own
thread so the m=128 extension's build time (judge readiness at ~140 s) is unchanged. No results yet.

## Why
Judge: qkv m=32 = 14.27 (cuBLAS splitK: 11.8 kernel + 2.5 gap), o m=32 = 6.73 (cuBLAS 4.7 + 2.0 gap); gv at m=32 costs
+2.3…+4.4 over m=16 (issue-bound), so the bake leaves them on cuBLAS. rq at NT=32 does 4× less MMA than at NT=128, so even at
the m=128 W-stream rate (2.7 TB/s on the H200) the targets are qkv ≈ 11.0 (−3.3), o ≈ 5.1 (−1.6) → ≈ −45 score-µs at
leverage 9.2; m=16 (leverage 11.1) only wins if NT=16 beats gv's 2.8 TB/s. pooler/unpooler m=32 (gv 16.5/16.3 kernel at
3.05 TB/s) win only if NT=32 streams faster than gv — the bake decides.

## Brief given (keep it)
1. Instantiate `rq<NT,…>` at NT=32 and NT=16 (x TMA box NT×64, B descriptor, NT/2 accumulators/thread, cp.reduce box of NT
   columns, escape/one-hot lanes); deeper ring is cheap at small NT (x box 4/2 KB per stage).
2. o layer (N=8192, K=1024 → 8 rounds per CTA at KS2): try 64 W rows per CTA with one math warpgroup and full K (128 CTAs ×
   16 rounds), KS1 with a deep ring, or 2 CTAs/SM; report whether it also helps o at m=128 (cuBLAS 7.71 judge incl. 2.2 gap).
3. Register (layer,32)/(layer,16) paths in the rq bake (≤ 4 timed paths each, budget guard, F1 rule, warm-up discard), dispatch by
   exact m, bake against pw34's actual m=32/16 pick (gv or cuBLAS), never torch.mm.
4. Box: one harness run with --official (65 checks), lint, cpu-check; commit rs32/ only. Kill: NT=32 qkv not ≥ 2 µs under
   pw34's pick after two iterations → stop and report the limiter.

## When it wins
Port the rs32 section into the next scored candidate (pw36 = pw35 + rs32 paths), box-validate vs pw35 in one run, check
first at the reset, then scored.
