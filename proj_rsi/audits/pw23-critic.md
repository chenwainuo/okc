# pw23 critic review (rsk2: staged loads, dual token block, self-cleaning reduction) — 2026-10-26

**Verdict: APPROVE.** I found no blockers. The one MAJOR item is a cheap guard on the self-cleaning workspace, worth
adding before the check.

What I ran:
- I built the gv block with nvcc 12.8, sm_90a, -O3, -Xptxas -v: 78 gv + rsk + 7 rsk2 instances, 14.3 s.
- r6's 78 functions are SASS-identical to r6-c12, and rsk<64,256,0> is identical to pw22's.
- All rsk2 instances have 0 spills, a 0 B stack frame and no LDL/STL.
- `rs_emu.py` passes (RS EMU PASS) with relative error ≤ 3.8e-8 and exact one-hot. It covers:
  - memset paths at m = 130, 200 and 250;
  - self-cleaning paths at m = 121, 199 and 200;
  - the unswizzled negative control, which is caught.
- The diff against pw22 adds rsk2 and rs2_go, cases 5-11, the new RSP with per-path m, and a 256×256 workspace. It
  removes rsk cases 1-4.

## 1. Stage protocol: OK

- All NSTG commit groups are issued up front.
- Stage s then runs `cp.async.wait_group(NSTG-1-s)` followed by `__syncthreads`. The SASS shows:
  - for NSTG 4: `DEPBAR.LE SB0,0x3 / 0x2 / 0x1 / 0x0`;
  - for NSTG 2: `DEPBAR.LE SB0,0x1 / 0x0`.

  None of them is a wait_all. Each is followed by `BAR.SYNC 0`, and no LDSM of a stage comes before its DEPBAR and BAR.
- Stage s holds chunks [s·CS, (s+1)·CS) of every x, hi and lo row. Its k16 steps q = s·QS … read only those chunks.

## 2. Dual block at m=256: OK

- With MI = 4, `mi>>1` selects the token block and `mi&1` the m16 tile. The x rows are `(mi>>1)·128 + wt·32 + (mi&1)·16`.
- Rows with tok ≥ m are zero-filled through src-size 0 and never stored (`tok < m`, `tok+8 < m`).
- The emulator exercises a partial second block in both paths:
  - memset: TOKS 256 at m = 130 and 250, TOKS 128 at m = 200;
  - self-cleaning: TOKS 256 at m = 199, TOKS 128 at m = 200.

## 3. Self-cleaning reduction (RED 2)

**OK.** The chain is:
1. red.add into ws;
2. `__threadfence` (fence.sc.gpu orders the earlier reds);
3. `__syncthreads`;
4. thread 0 does `atomicAdd(cnt[tile])`;
5. the last CTA does `__threadfence`, then for rows < m does `__ldcg` → `__stcg` zero → store to out;
6. it resets the counter.

This is safe because:
- No other CTA of the launch touches the tile after its KS-th arrival.
- The next replay is stream-ordered after kernel completion, so it cannot see a half-zeroed tile or a stale counter.
- Only rows < m are ever written, and exactly those are re-zeroed. ws starts as `torch.zeros` and is checked to be at
  least TB·TOKS·N.

**MAJOR (guard).**
- **The risk.** Both self-cleaning paths share `rs_ws` and `rs_cnt`, and nothing re-zeroes them after the bake. Suppose
  rp64k32sc is picked at m=128 and then rv64k32sc (at m=256) leaves garbage, through a bug or an aborted run. It would
  fail its own validation and be declined, but rp64k32sc would then serve from a dirty workspace: a silent INVALID on
  the judge.
- **The emulator does not cover it.** It does not model the re-zeroing or the cross-call invariant.
- **Fix.** Run `st["rs_ws"].zero_(); st["rs_cnt"].zero_()` at the end of `_rsbake`, or after every RED-2 path, outside
  any capture. Alternatively, drop the self-cleaning picks if any self-cleaning path errored or was INVALID.

## 4. smem and registers: OK

| Variant | smem | Registers | CTAs/SM | Limit |
|---|---|---|---|---|
| rv64k32 | 196,608 B (192 KB) | 122 | 1 | smem; within the 227 KB opt-in |
| rv64k64x2 | 96 KB | 118 | 2 | 120 × 256 × 2 = 61k regs; 2 × 97 KB smem |
| rp64k64x2 | 64 KB | 83 | 2 | |
| other 128 KB variants | 128 KB | 83-124 | 1 | |

- The dynamic smem attribute is set once per instance.
- The 2-CTA variants carry `__launch_bounds__(256, 2)`, so they are capped at 128 registers.

## 5. r6, dispatch, budget: OK

- r6 is identical. Dispatch changes only for router m = 128 and 256, and only on a timed 1% win over r6.
- **Prepare time.** pw22 on the judge logged `PREP r6 done in 118s` and `PREP done in 119s`; its rs bake took about 1 s.
  pw23 has 7 more small instances (+0.7 s local compile) and 8 bake paths, so I expect about **120-125 s** against 240.
- **Log.** About 5 KB (pw22's was 5.0 KB).
- **PROBE format.** `PROBE rszero router m=128 <us>` is unchanged.
- **For the coordinator.** pw22 showed rs64k32 losing at m=256 (21.4 µs vs r6 15.9), so the m=256 hopes rest on the
  rv variants.
