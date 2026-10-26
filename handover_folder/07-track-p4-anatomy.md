# 07 — Track P4: decode cost and the per-stage anatomy of rq (rs128p)

Files: `proj_rsi/candidates/rs128p/{submission.py,lab.py,hypothesis.json}` (instrumented copy of rs128 with clock64 stamps and
ALU/probe modes), audits `p4-h100.md` (wgmma-overlap mechanism), `p4-decode-slack.md` (WRONG first verdict; carries a
correction note), `p4-decode-pipe.md` (the correct one), `pw24h-critic.md`. Box logs `/root/okc/logs/rs128p*`, `p4*`.
Agent: P4 (Opus) — running; a batched probe run (switch-fix, pre-walk, zero-XOR, runtime zero-XOR, data-pattern variants,
baselines re-timed, pooler and qkv m=128) is queued behind the lock; it was asked to deliver ONE anatomy table.

## Established
- bf16 SS pipelines: a single consumer warpgroup's serial per-stage bookkeeping (~230–286 clk) vs a ~371 clk stream period on the
  H100 (~260 on the H200) caps a 64×64 tile at cuBLAS level; fix = 128-k slots with one barrier pair per slot (bf16 hand
  pipeline k2x reached cuBLAS parity on the box, pooler m=256 −2.4 µs). 128-row two-warpgroup tiles amortise it (RS128).
- Decode cost in the REAL register-A path (LDS → decode → wgmma): dec1 (0.875 ALU + 0.25 FMA-pipe ops/weight) costs only
  +0.03…0.32 µs at pooler m=128 on the box; PRMT-merge floor 0.5 ops/weight (+0.1); aligned-window codec +0.07; OR + bf16
  rescale +0.35…0.55. LOP3 and HFMA2 are half rate (2.2–2.4 clk/op with ILP), FFMA full rate (1.1). Headroom ≈ 2× dec1 per stage
  on the H100, at the edge on the H200. The earlier "2.2 clk/op INT-bound, needs ≤ 16 ops/stage" verdict was a latency-chain
  artefact of synthetic probes.
- Escapes (weights outside the 12-bit window) are tiny normal weights (< 2^-19), 5 exact zeros, no subnormals → no
  data-dependent tensor slowdown expected (zero-XOR probe pending to confirm).
- The rq per-stage flow (decode + wgmma issue/commit/wait + releases) costs ~+2 µs over the bare W+x stream at m=128 (see 04);
  goal written in the brief: ≤ 650 clk per stage via precomputed descriptors, batched releases, blocking waits, escape-walk
  placement (pre-walk vs in-loop), switch-based escape XOR.

## What to do with P4's table when it lands
Take any variant that beats the shipped rq path by ≥ 0.3 µs in-prepare in the same run, port it into `rs128/submission.py`
(and the pw3x RS128 section), re-run --official, and queue it for a check at the reset. Combine with the author's stream-K /
multicast results (04) only after each is measured alone.

## P4's anatomy table (landed 19:20 UTC; commit 5b86c0f, `proj_rsi/audits/p4-rs128-anatomy.md`)
Clocks per k64 stage, warpgroup 0 (resident = no TMA, no barriers):

| variant | loop | full wait | LDS wait | decode | escape walk | fence+issue | wait_group | release | epilogue share |
|---|---|---|---|---|---|---|---|---|---|
| shipped, streamed | 941 | 350 | 30 | 150 | 79 | 177 | 34 | 95+25 | 73 |
| resident | 749 | 66 | 166 | 187 | 75 | 188 | 27 | 40 | – |
| resident, no escapes | 646 | 64 | 169 | 211 | 32 | 92 | 32 | 46 | – |

In-prepare m=128 (one batched locked run): shipped rq 24.06 pooler / 16.80 qkv vs torch.mm 28.58 / 20.42 and r6 27.10 / 19.49.
Findings: the box is stream-bound (bare W+x = 19.0 of the kernel's 22.0 µs) so per-stage fixes barely show there, but the SM
side binds on the H200; register-A runs as fast as SS (11.5 µs resident), LDS + decode add only 0.5 µs; the ESCAPE WALK adds
2.16 µs resident (~36 escapes per CTA, escape-heaviest CTAs set kernel time) because the four-warp `wgmma.fence` waits for the
warp walking its list (issue 92 → 188 clk). Descriptors already in uniform registers; x release before LDS +0.07; polling
worse; escape variants switch −0.12 (noise) / pre-walk +0.55 / per-step +0.4…0.6. ≤ 650 clk/stage is below what the box
sustains for 8 m64n128k16 per stage under power throttling (~0.343 µs/stage); attainable SM-side ≈ 0.359 µs (−16%).
NEXT (assigned 19:25 UTC): a producer-warp-written per-slot escape patch table so math warps do one LDS + predicated XOR
(expected −2 µs SM-side, −0.4…−0.6 on the box, more on the H200); kill at < 0.3 µs in-prepare; port into rs128 if it wins.
