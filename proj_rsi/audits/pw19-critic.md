# pw19 critic review ("tw": decode warpgroup → bf16 A ring → SS wgmma) — 2026-10-26

**Verdict: APPROVE.** I found no correctness or deadlock defect. The two MAJOR items are performance risks:
- tw64x2 and tw128 are real cluster-4 launches;
- tw64x2's 2 CTAs/SM is never checked at runtime.

What I ran:
- An nvcc 12.8 build (sm_90a, -O3, --use_fast_math, -Xptxas -v) of KERNELS_SRC for pw19 and pw10.
- The 23 kernels the two builds share are SASS-identical. pw19 drops pw10's pwg probe instances.
- No wgmma-serialization or other ptxas warning.
- The pwt instances use 90 registers (t64, t64x2, t128, pT3), 155 (v64) and 28-41 (other probes), all with 0 spills.
- `tw_emu.py`: TW EMU PASS. That is:
  - bit-exact at KS 1/2/4/8, m = 97/101/121/128/200/256 and BN 128;
  - the r6 pack and an unswizzled stmatrix both fail, as they should;
  - 750/750 protocol schedules clean per instance;
  - early_release and no_fence each caught in 150/150 schedules.

## 1. A-operand layout and descriptor: OK

**Write side.**
- The A slot is BN rows × 128 B. Row r, 16 B chunk c sits at `r·128 + ((c ^ (r&7)) << 4)`.
- stmatrix.x4 lane L addresses row `(L>>3 & 1)·8 + (L&7)` of the 16-row tile and chunk `2q + (L>>4)` (submission.py:1000-1004).
- The tile base (tile·2048), the 8-row half (+1024), every A slot (AOFF = XST·XB, AB = BN·128) and the warpgroup offset
  (+8192) are all multiples of 1024. So address bits 7-9 equal r&7, and the XOR is exactly the hardware 128B swizzle
  (bits 4-6 ^= bits 7-9).
- The matrix order {rows 0-7 k-lo, rows 8-15 k-lo, rows 0-7 k-hi, rows 8-15 k-hi} matches the mma A fragment
  a0..a3.

**Descriptor (submission.py:639).**
- start>>4 in bits 0-13.
- LBO = 1. It is ignored for swizzled K-major layouts.
- SBO = 1024 >> 4: one 8-row × 128 B core-matrix group.
- base_offset = 0, which is valid because every start is 1024-aligned.
- Layout type 1 = SWIZZLE_128B in bits 62-63.
- k16 step q adds 2 to the descriptor, i.e. +32 B inside the 128 B row, which is the CUTLASS/DeepGEMM idiom.

**The wgmma.**
- It is `m64n{128,256}k16.f32.bf16.bf16 D, descA, descB, p, 1, 1, 0, 0`. A is M=64 × K=16 K-major (W rows × k), B is
  N × K K-major (the x tile), and both are untransposed.
- The transpose rule: imm-trans-a/b may be 1 only for .f16/.bf16 and only for operands read from shared memory through
  a descriptor. Every other type is K-major only, and an A taken from registers cannot be transposed.
- The SASS shows `HGMMA.64x128x16.F32.BF16 R24, gdesc[UR16], R24`: both operands come from descriptors, with no
  register A.

## 2. Proxy fence: OK

- The decode SASS per stage is `STSM ×4 → MEMBAR.ALL.CTA → FENCE.VIEW.ASYNC.S → SYNCS.ARRIVE` (W-empty from lane 0,
  A-full from every thread). afull is initialised with count 128, so every writer fences before arriving.
- The MMA side waits with `mbarrier.try_wait.parity` (`SYNCS.PHASECHK.TRANS64.TRYWAIT`), whose default semantics are
  .acquire.cta in the generic proxy. That is followed by `WARPGROUP.ARRIVE` (wgmma.fence) and then HGMMA, which is
  CUTLASS's producer-side-fence pattern.
- On reuse, the A slot is released only after `WARPGROUP.DEPBAR.LE gsb0,1`. The A-empty and x-empty arrives come after
  that wait (SASS 0x23e0-0x2410).

## 3. Three-ring handshake: deadlock-free

**Barrier counts and parities.**
- Counts:
  - xfull and wfull: 1 + tx;
  - xempty and aempty: NWM (lane 0 of every MMA warp, since each warp's wait_group only covers its own groups);
  - afull: 128;
  - wempty: 4.
- Every parity is `(u/S)&1` for full barriers and `((u/S)-1)&1` for empty ones.

**Why it cannot deadlock.**
- The producer never blocks: it calls test_wait on W-empty first, then on x-empty, and loops.
- MMA(u) needs A(u) and x(u).
  - x(u) needs chunk u-XST released. MMA releases u-1 right after issuing u, so this holds when XST ≥ 2.
  - A(u) needs decode(u), which needs W(u) and the release of u-AST. W(u) needs decode(u-WST) to have run.
- There is no cycle when XST ≥ 2 and AST ≥ 2. Every instance has XST ≥ 3 and AST ≥ 3. With AST = 1 it would deadlock;
  a static_assert would guard against that.
- W-first polling cannot starve x. The W ring fills, its test fails, and the producer issues x.

## 4. tw128 (NWG 2, 416 threads): OK

- There is one A ring holding both halves: AB = 128·128, and warpgroup wg reads at +8192·wg.
- xempty and aempty have count NWM = 8, so both warpgroups release before the shared x or A slot is reused.
- Registers: 90 × 416 threads is about 40k (32-register-per-lane granularity × 13 warps). Shared memory is 189,648 B.
  That gives 1 CTA/SM.

## 5. Occupancy and clusters

- **tw64x2 (MINB 2).** Each CTA has 111,808 B of dynamic shared memory plus 0 static. With the 1 KB reserve,
  2 × 112,832 = 225,664 B, which fits under 233,472. Registers are 90 → 18 warps × 3,072 = 55k, under 64k. Two CTAs/SM
  are therefore feasible.
- **MAJOR.** Nothing verifies 2/SM at runtime. The launcher only checks `cudaOccupancyMaxActiveClusters > 0`, and the
  carveout is left to the driver. I suggest setting PreferredSharedMemoryCarveout = 100 and logging the
  max-active-clusters count. For pooler it should be ≥ 64, i.e. 2 × 132 / 4.
- **MAJOR (cost).** These are real **cluster-4 launches**, not pairs. In `_pw_tks`:
  - tw64x2 ("x2", KS ≤ 8, NB·KS ≤ 264) runs pooler as 64 × 4 with cluster (1,4,1), and qkv as 36 × 4;
  - tw128 ("c4") runs pooler as 32 × 4 with cluster 4;
  - x2 can reach KS 8 (cluster 8) when NB ≤ 33.
  pw9's cluster-4 w2n128a ran 46.1 µs, against 32-39 µs for the cluster-2 paths. Only tw64 and tv64 are cluster 2.

## 6. Epilogue reuse of the ring: OK

- MMA reaches `bar.sync 1, NC` only after `wait_group 0`. That is after the A-full wait of the last chunk, which is
  after the decode's last STSM, fence and arrive.
- Every W chunk was waited on by the decode, and every x tile by the MMA, so no bulk or TMA write is in flight.
- Staging, from sbase up to BOFF (checked to be -8), never touches the barriers.
- The cluster barrier is "all non-exited threads", so the decode and producer warps that have exited are fine. The
  SASS has the BAR.SYNC 0 fallback only for the cluster-size-1 branch, which is never taken when KS > 1.

## 7. Probes, shared kernels, prepare

- **Probes.** The probes (pT1 / pT1w / pT2 / pT3) have `ms = ()`, so they are never selectable paths. They launch
  without a cluster and stop before the epilogue.
- **Probe deadlock.** They follow the same deadlock-free protocol. With PRB 5 there is no x at all: the producer starts
  with `xs = KCS` and the MMA neither waits on nor releases x.
- **Selection.** kernel_fn selects by exact (layer, m), so odd m go to the fallback. A launch error code also falls back.
- **Prepare time.** pw10 on the judge logged `PREP compiled in 45s` and `PREP done in 57s`. pw19 builds 26 instances
  (14 b, 4 w, 8 t), against pw10's 33 kernels, and compiles locally in 5.6 s against pw10's 7.2 s. I expect about
  **55-70 s**, and a log of about 9-10 KB (pw10: 9.4 KB).

## Minor

- `_pw_tks` is defined twice (submission.py:1692 and 1875). The later 5-argument version shadows the old one, so the
  `_pw_tks(NB, half)` call at :1803 would raise a TypeError. It is dead code today, because only kind "b" reaches
  `_pw_runner`. Rename one of them.
- The producer lane busy-polls test_wait on two barriers and steals issue slots from the MMA and decode warps on its
  scheduler. A `nanosleep` back-off, or try_wait with a suspend hint on the ring that is due next, would fix it.
- pw19 is pw10-lineage, with no r6 gv paths. Its check measures the tw kernel; its board score is not comparable to
  r6's 3428.
