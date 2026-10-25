# SASS audit of the pw6 consumer loop (sm_90a, nvcc 12.8 -O3) — 2026-10-25

Instances: `pwk<1,16,1,4,1,5,1,1,0,1>` (m=32, cluster) and `pwk<1,16,1,1,1,5,1,1,0,1>` (m=1, cluster).
Disassembly: scratchpad a6/m32c.sass, a6/m1c.sass (cuobjdump -sass). Report only; no files edited.

## Instructions per 64-k chunk per warp (hot path: first poll succeeds, no escape hit, no n-block boundary)

| class | m=32 | m=1 |
|---|---:|---:|
| LDS.128 (3 W + 2·TTW x) | 11 | 5 |
| HMMA.16816.F32.BF16 | 16 | 4 |
| decode (8 AND, 8 ADD, 4 SHF, 16 PRMT) — at the codec floor | 36 | 36 |
| integer / address arithmetic | 20 | 16 |
| predicate / select | 7 | 7 |
| branch / convergence (BRA, BSSY, BSYNC, WARPSYNC) | 11 | 11 |
| S2R / LDC / ULDC reloaded every chunk | 5 | 4 |
| mbarrier try_wait + arrive | 2 | 2 |
| NOP | 3 | 5 |
| **total** | **111** | **90** |
| core work | 63 | 45 |
| realistic ideal (core + ~11 loop) | 74 | 56 |

## Waste, by cause
1. W address rebuilt from SR_TID every chunk (6 instr): `lane*16 + wn*1536` rematerialized (85 regs used, no pressure).
2. x address rebuilt every chunk (6 at m=32): `xrow` is a runtime value so `tt*8*xrow` is a dependent IMAD chain + LDC reload.
3. x loads pinned after wait + W loads + decode + escape region by `asm volatile` ordering and the escape BSSY/BSYNC; ~30 cycles of
   LDS latency + the address chain sit between decode and the first HMMA with nothing to fill them.
4. Escape check 7 instr/chunk (recomputes seq = ((cnb-nb0)*KCS+u)*RT; equals the stage counter when PRB=0). Hit path ~60 instr + a
   dependent LDG; negligible at 4–20 hits per warp per launch.
5. n-block boundary test 5 instr/chunk; can never fire in the cluster path (NG == NB).
6. Slot/phase bookkeeping 8 instr/chunk because ST is a runtime value.
7. Arrive 5 instr (S2R + lane==0 predicate rematerialized + addr + NOP + arrive), issued after all HMMAs although W smem is dead after
   the decode: each ring slot is held 150–300 cycles longer than needed.
8. Spin loop fine (TRYWAIT + BRA; 3 instr per retry).

## Bank conflicts: none. x loads: group (g + 2t) mod 8 distinct per quarter-warp (xrow ≡ 16 mod 128). W loads: 16-B lane stride.

## Budget per SMSP per chunk-round (16 consumer warps/SM, 1650 cycles per round at 3.8 TB/s, 1400 at 4.5)
| | m=32 | m=1 |
|---|---|---|
| issue | 444 cyc (27% / 32%) | 360 (22% / 26%) |
| tensor pipe | ~410 (25% / 29%) | ~100 |
| **shared-memory reads (per SM, 128 B/clk)** | **704 cyc (54% / 64%)** | 320 (31% / 37%) |
Issue and tensor have ~3× slack; **shared-memory read bandwidth is the closest to binding at m=32** because 8 of 11 LDS per
warp are x fragments identical across all 16 warps (16× redundant). The measured 1.3–2.3 µs compute cost is most plausibly
edge latency (per-chunk critical path at stream start/end, slot hold), not steady-state issue.

## Recommended edits (1–4, 6 numerics-preserving; 5 is a bake-off item)
1. ST as a template parameter, stage loop unrolled by ST, `wl = sbase + wn*RT*1536 + lane*16` hoisted: W/barrier addresses become
   immediates; phase flips once per outer iteration. Saves ~12–14 instr/chunk.
2. Issue the x-fragment loads before `pw_mbwait` (x was written before the CTA barrier) with a chunk-major x layout
   `[chunk][row][144 B]` so `tt*8*144` is an immediate. Saves ~5 instr and ~50–60 exposed cycles/chunk. Watch regs at m=32
   (~117 of ~120 cap); prefetch only tt=0 if tight.
3. Escape compare against the stage counter via a loop-carried `nxs = nx >> 17`; drop the n-block test when CLU && KS>1 (and make
   the launcher require NG == NB for CLU && KS == 1 too). Saves 6–8.
4. Arrive on the empty barrier right after `pw_unit` (before pw_fix/pw_mmas): shortens slot hold by 150–300 cycles/warp.
5. Warp shape RT=2 (e.g. <RT=2, WN=8, WTOK=2, TTW=2>): two W units share one x fragment set → 11 → 5 LDS per W unit,
   ~111 → ~75 instr per W unit; bit-identical results (same per-chunk k order). Addresses the smem-bandwidth limit.
6. Pin `lane` (`asm volatile("" : "+r"(lane))`) and hoist `p.Tb` into a register: ~3 instr.
Codec note: window base restricted to multiples of 8 would fuse AND+ADD into LOP3 (−8/chunk) but likely adds escapes; not recommended
without measuring escape counts.
Expected: m=32 111 → ~80 instr/chunk, m=1 90 → ~62, all numerics unchanged; edit 5 is the one that moves the smem ceiling.
