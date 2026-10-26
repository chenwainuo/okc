# pw22 critic review (router m=128/256 split-K GEMM "rs") — 2026-10-26

**Verdict: APPROVE.** I found no blockers. There is one MINOR occupancy correction.

What I ran:
- I built `GVCU` up to the torch glue with nvcc 12.8, sm_90a, -O3, -Xptxas -v: 78 gv instances plus 5 rsk, 13.6 s.
- r6's 78 functions are SASS-identical to r6-c12.
- The rsk instances have 0 spills, a 0 B stack frame and no LDL/STL.
- `rs_emu.py` passes (RS EMU PASS). All 5 variants have relative error ≤ 3.8e-8 at m = 128, 121 and 200, and the
  one-hot check is exact. The unswizzled-write negative control is caught.
- The diff against pw21:
  - adds rsk, rs_go, rs_launch and rsrun;
  - routes `_gvcall` to `_rscall` for p[0] == -7;
  - runs `_rsbake` in place of `_l2probe`, inside a try that pops only the router 128/256 selections;
  - removes GRP and gvdec.

## 1. The memset node: OK

- kernel_fn allocates `out` and passes it to `_gvcall` → `_rscall`, which runs `out.zero_()` and then `rsrun` into the
  same tensor on the current stream (submission.py:2763-2766, `_rscall`).
- Under the judge's capture this becomes a memset (or fill) node in program order on the same stream, ahead of the
  kernel node.
- It zeroes the very tensor that kernel_fn returns. No private buffer or copy is involved.
- `_time_us` captures the zero_ together with the kernel, so the bake times include the memset.

## 2. Atomic order: OK

- **Random rows.** fp32 reassociation over at most 64 splits gives about 1e-7 relative error (emulator: 3.6e-8). That
  is well inside the bake's 2e-5 and the judge's 1e-4.
- **One-hot ×8 is exact under any order.**
  - Only the split that contains k contributes a non-zero value: 8·hi + 8·lo, which is exact in fp32 (|lo| ≤ ulp(hi)/2).
  - Every other split adds ±0.0 to a zeroed output.
  - `torch.equal` treats -0.0 as equal to +0.0.

## 3. Swizzle, ldmatrix and smem: OK

**Store and load patterns.**
- The store puts 16-B chunk c of row r at `r·RB + ((c ^ (r&7)) << 4)`. RB is 256 or 512 B, a multiple of 128, and
  there are 16 or 32 chunks per row (at least 8).
- Each 8×8 ldmatrix phase reads 8 consecutive rows at one logical chunk. Those rows land in 8 distinct 16-B bank
  groups, so there are no bank conflicts.

**Fragment mapping.**
- A (x, token rows): lanes 0-15 read rows 0-15 at chunk 2q, and lanes 16-31 at chunk 2q+1. That gives
  {rows 0-7 k-lo, rows 8-15 k-lo, rows 0-7 k-hi, rows 8-15 k-hi}, which is the a0..a3 order.
- B (W rows, K-major, `.col`): the mapping `br = (j>>1)·8 + (lane&7)`, `bc = j&1` gives b0/b1 for n-tile 0 and b2/b3
  for n-tile 1.
- hi and lo go into the same accumulator.

**Occupancy (MINOR correction).**

| Variant | smem | Registers | CTAs/SM | Limited by |
|---|---|---|---|---|
| rs64k32, rs64k32l | 128 KB | 84 / 88 | 1 | smem |
| rs128k64 | 96 KB | 133 | 1 | registers |
| rs64k64, rs64k64l | 64 KB | 84 / 88 | **2, not 3** | registers (3 × 256 × 88 > 64k) |

- rs64k64 still fits m=128's 256 CTAs in one wave (264 slots).
- The static smem is only the 4 B `last`.

## 4. Up-front loads: OK

- Every x, hi and lo chunk is requested by a 16-B `cp.async.cg`.
- Rows with tok ≥ m use src-size 0 with a clamped valid address.
- One `commit_group; wait_all` is followed by `__syncthreads`, and no ldmatrix comes before them.
- The dynamic smem attribute is set once per instance (64-128 KB).

## 5. Last-arriver: OK

- The chain is the same as pw11's:
  1. `__stcg` the partial into the CTA's slot;
  2. `__threadfence`;
  3. `__syncthreads`;
  4. thread 0 does `atomicAdd(cnt[tb·NTL+nt])`;
  5. the last CTA does `__threadfence` and sums the KS slots with `__ldcg`;
  6. it writes each output element once;
  7. it resets the counter.
- `ws` is sized exactly for the largest case (BN64 KR128: 2·4·64 slots × 2,048 float4), and `rs_go` checks the size.
- `cnt` has 64 entries; at most 8 are needed.

## 6. r6, dispatch, budget, log: OK

- r6's instances are identical, and `_gxbake` no longer has any 128/256 paths.
- kernel_fn serves `sel[(router, _bk(m))]` only for exactly m = 128 or 256, and only after `us < r6 × 0.99` in
  `_rsbake`. The odd m 121 and 250 are validated.
- **Prepare time.** The r6 phase is about 110 s, the compile matches r6's size, and `_rsbake` is 10 paths with 32-64
  one-hot launches each, a few seconds. Total: **about 115-120 s** against 240.
- **Log.** About 5 KB.
- **PROBE format.** `PROBE rszero router m=128 <us>` parses as config "router m=128" followed by the time.
