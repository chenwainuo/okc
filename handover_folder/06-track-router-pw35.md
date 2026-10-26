# 06 — Track router (P3) and pw35

Files: `proj_rsi/candidates/pw29m/submission.py` (rtk router development), `proj_rsi/candidates/pw35/submission.py`
(= pw34 @ 49b22d8 + rtk router section; 260,514 bytes; sha256 7069218d…; commits 9683aa4 (content) + d938890 (record)),
`proj_rsi/audits/p3-m256.md`, `p3-h100.md`, box logs `/root/okc/logs/pw35ref.log`, `pw35ref2.log`, `pw29m*.log`.
Agent: P3 (Opus) — finished; "waiting on its own background work" only (a box waiter). Nothing more assigned.

## Result
Old router kernels (rwk) lost because threads stalled 3.7–11 µs just issuing cp.asyncs, so the first MMA waited for nearly all
data. rtk = one TMA-issuing warp, one math warpgroup, one zeroing warp (self-zeroing output, no memset node), globaltimer
stamps, co-residency guard (grids that cannot co-reside in one wave are refused → r6 fallback; a cluster-of-4 x-sharing variant
deadlocked with self-zeroing, so that path is closed).
Box, in-prepare: m=256 rt64k16b6z 12.42–12.80 vs r6 14.85–15.01 (F1 baseline); m=128 rt64k32b3x2z 10.43 vs pw34's
rw64k32s8z 11.74. Fixed terms at m=256 (µs from kernel start, median CTA): first MMA 2.21, last data 7.17, last MMA 7.78,
reduces issued 8.74 (worst CTA 9.50). The 2.2 µs to first data is the biggest remaining fixed term.
pw35 vs pw34, same locked session (harness µs): verify32 router 17.90 → 15.49 (−2.42), verify16 router 14.10 → 13.62 (−0.48),
other 26 shapes within ±0.4; decode1 router/unpooler ±1.2 depending on run order (position, not code). 65/65 official,
lint ok, cpu-check ok. Prepare 76.7 s on the box (router build 33 s, finishing at 76 s, after rq bake and gx; judge estimate
160–180 s vs the 240 s budget; if it overruns the guard only the router paths are skipped and r6 serves them).
Bake list: m=128 rw64k32s8z, rt64k32b4z, rt64k32b3x2bz, rt64k32b3x2z; m=256 rw64k32r3x2z, rt64k16b6z, rt64k32b3x2z; r6 is the
F1 baseline at both sizes.

## Status
Scored run launched 19:08 UTC by the detached script `submit_pw35.sh` → `results/proj-rsi/jobs/pw35.scored.txt` (job id on its
first line once submitted). Expected judge effect: −1…−2 µs at m=256 (×6.3) and −0.5 at m=128 (×8.2) ≈ −10…−17 score-µs
→ ~3280. If the report shows "skipped-budget" for the router build, move the router build earlier or trim its bake list.
Next for the router: cut the 2.2 µs to first data (prefetch the tensormaps earlier, smaller first boxes, issue the first
loads before the zeroing warp's work), and router m ≤ 32 fixed latency (gap +0.9…+1.2 µs on gv paths; the rr 1024-thread
kernel at m=1 has only +0.6).
