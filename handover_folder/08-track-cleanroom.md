# 08 — Track clean-room rewrite (workflow wf_1d859ebc-409) and the latency-loader follow-up

User's ask (2026-10-26 ~17:00 UTC): "121 is not good enough… send a subagent to rewrite the kernel from scratch without looking at
the current implementation but with understanding of the model and performance requirement." Implemented as a Workflow:
four schools build full submissions in `proj_rsi/cleanroom/<school>/` from the task spec alone (forbidden to read
`proj_rsi/candidates/*`, audits, LOOP.md; allowed: muse_bf16_proj/, tools/h100_bench.py, running pw34 as a black box), each
with a DESIGN.md; then a verifier (apples-to-apples table vs pw34 → `proj_rsi/audits/cleanroom-measure.md`), three judges and a
synthesiser (`proj_rsi/audits/cleanroom-plan.md`: mechanisms worth porting). Journal of the run:
`/home/ubuntu/.claude/projects/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/subagents/workflows/wf_1d859ebc-409/journal.jsonl`
(agent transcripts `agent-*.jsonl` beside it). If the session died, the workflow died with it: resume by reading each school's
DESIGN.md and re-running the verify/judge stages as plain agents (briefs in 11).

| school | thesis | status at handover |
|---|---|---|
| latency (`cleanroom/latency/`) | one node per call, 128 CTAs one wave, everything in flight at the first instruction, 12-bit planes (L byte + H nibble, per-layer 16-exponent window, per-CTA escape lists, probe rows exact), mma.sync for m ≤ 32, same template for all layers; m > 32 torch fallback | DONE as a report: correct 20/20, but its 1-D cp.async.bulk loader serialises (~1 µs + ~16 GB/s per SM) → 1.2–3× slower than cuBLAS. Prepare 37 s. Follow-up agent spawned 18:40 UTC to fix the loader (mb_load.py microbenchmark, LATENCY_LOADER=1 256-thread cp.async ring, ld.global.v4 streams, 2-D TMA boxes); targets ≤ bytes/3.0 TB/s + 1 µs on the H100 (router 3.1, slice 3.1, o 5.2, qkv 10.4, pooler 17.8 at m = 1/8/16/32). Its projected H200 gains if the loader streams: router 2.5 vs 3.2–5.0, unpooler slice 2.6 vs 4.0–4.2, o 4.1 vs 5.1–7.0, qkv 8.3 vs 9.9–14.6 → ≈ −3 µs on decode1/verify1 (×15) and −13 on decode32 (×9). No result yet. |
| bandwidth (`cleanroom/bandwidth/`) | bytes/weight is the only free variable: lossless ~12-bit re-encode laid out in tensor-core fragment order, one register-direct mma.sync kernel family for every m incl. 128/256 (12-bit stream + mma.sync small/rfma/big kernels, all judge cases correct per commit 67f5786; launcher smem-attr fix, one-wave configs in 3f5c93a/9683aa4) | building/measuring (harness v2 run queued at 18:35) |
| tensorcore (`cleanroom/tensorcore/`) | TMA + mbarrier + wgmma GEMM over a 12-bit pack, W as the register-A operand (same orientation as RS128), v10 restructured mainloop (outer n-block loop, per-group W loads, 256-thread integrated producer above MP=64), v20 reserved DSMEM slot + batched escape prefetch, v21 programmatic dependent launch for the convert kernel | ladders running (bcb1c5e, 1257f24) |
| contrarian (`cleanroom/contrarian/`) | one m-adaptive packed-weight GEMM: 12-bit layout decoding directly into A fragments (4 ALU ops per weight pair), x the only smem operand, cooperative split-K reduction, router one kernel over hi|lo (K=16384); PTX-level experiments | correctness logs + official harness runs committed (59c3479, a80dea9); bench5 official run at 18:37 |

How to judge a school: only a within-run harness table vs pw34 on a quiet GPU counts (REPLAY_OVERHEAD printed first; the
schools' early numbers were taken under the soak's 12.6 µs replay floor). Port mechanisms, not files: pw34's lineage is
battle-tested on the judge; a school's kernel replaces a shape only after it wins that shape's in-prepare bake.
