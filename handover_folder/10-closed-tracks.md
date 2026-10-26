# 10 — Closed tracks (do not re-drill; evidence in proj_rsi/audits and LOOP.md)

| idea | verdict | evidence |
|---|---|---|
| Variable-length codecs (codec E pair-Huffman 10.8 bits, 11-bit fixed) | CLOSED: bit-exact but issue-bound (256–286 warp-instr per 1024 weights vs r6's 49 → 1.5–2× the stream); 11-bit needs 1.3% escapes | pw17/pw18/pw20, audits/codec-study.md, pw17-/pw18-critic.md |
| Decode in dedicated warps → bf16 smem → SS wgmma ("transform warpgroup", pw19, user-requested) | CLOSED: hid the decode but lost 10+ µs to x delivery (bytes in flight) and smem bandwidth; m ≥ 128 is L2→SM-delivery/issue-bound, not decode-bound | pw19-critic.md |
| Plain bf16 hand pipelines at cuBLAS geometry (pw24, pw24h k2x/n128x, pw30/pw32) | CLOSED as candidates: 0.1–2.5 µs behind r6's TUNED cuBLASLt (parity was only vs torch.mm); taught the 128-k-slot barrier fix | p4-h100.md, pw24h-critic.md |
| CUTLASS 3.9.2 bf16 (pw25, 817 KB bundle, 233-config sweep) and CUTLASS mixed-input two-plane 12-bit (mx) | CLOSED: stock configs 60% slower than nvjet; best sweep at parity with torch.mm but behind tuned cuBLASLt; mx −0.6/−1.0 at m=128, loses at m=256; reusable pieces (in-chunk escape header, self-cleaning zeroing, one-lane reduce-add store) in audits/mx-h100.md | cutlass_sweep.md, mx-h100.md, pw25-critic.md |
| DeepGEMM port | not pursued: routes bf16 to cuBLASLt itself; its sm90 pipeline structure documented (one wgmma group in flight, merged k128 stages, no clusters on one-wave grids) | deepgemm-sm90-bf16.md, deepgemm-host-embed.md |
| TMA-ring + mbarrier + split-K skeleton for m ≤ 32 (pw3–pw7) | KILLED: +4.6 µs over the bare bulk stream at every m; r6's register-direct loads are within 0.6–0.75 µs of the stream at m ≤ 8 | LOOP.md iterations 3–7, pw6-sass-audit.md, r6-lineage.md |
| Bulk W ring for small m (pw12), wide-row tiles with cross-CTA split-K in r6's lineage (pw11), clusters of 4/8 (DSMEM reduce) | lose 1–3 µs; any cross-CTA reduction costs ≥ 1–2 µs; clusters co-schedule only 120 of 128 CTAs; each cluster-size step 4→8 costs 10+ µs; cluster launches ~1–2 µs fixed | LOOP.md, pw11/pw12 |
| qkv tail balancing at m ≤ 32 (stream-K straddled tiles, pw28/pw28h) | CLOSED: the m ≤ 32 qkv gap is spread over x loads / decode / mma, not the tail; P2's shorter-chain geometries at m=1/8/16 did ship (pw33) | p2-h100.md, pw28-critic.md |
| x cache policy / L2 hints as a judge lever | neutral on the H200 (box-only gain) | pw34 judge bake |
| Deeper separate W ring in RS128 (rs128b), epilogue half-slab early issue | equal / monotonically worse; −0.13…−0.16 | LOOP.md rs128b |
| RS128 at m=256 (n256 and two-pass n128×2) | CLOSED: tensor rate 100% resident but loads +3.4 and epilogue +2.4 → 0–3 µs behind cuBLAS at every layer | 04, rs128s/ |
| rq per-stage levers: descriptor precompute (already uniform regs), x release before LDS (+0.07), elected-lane release, polling, escape pre-walk (+0.55), per-step (+0.4…0.6), switch XOR (−0.12 noise) | no gain; the escape walk itself (2.16 µs resident) is the remaining SM-side lever → patch table (07) | p4-rs128-anatomy.md |
| Router x sharing across a cluster of 4 (pw29m) | deadlocked with self-zeroing (two waves); grids that cannot co-reside are now refused | p3-m256.md |
| "Decode is INT-pipe-bound at 2.2 clk/op, ≤ 16 ops per stage" | WRONG (latency-chain artefact); dec1 costs +0.03…0.32 µs in the real path | p4-decode-pipe.md |
| Soak job to keep the GPU busy | harmful to measurement (12.6 µs replay floor); stopped | 03 |
