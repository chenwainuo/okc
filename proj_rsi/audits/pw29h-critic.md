# pw29h critic review (self-zeroing wgmma router, sm_90a build fix) — 2026-10-26

**Verdict: APPROVE.** I found no blockers. Two low-priority hardening notes are listed at the end.

The candidate file is byte-identical to the box copy (md5 87d38b21…). It is 179,665 B.

## What I ran
- **Diff vs pw29.** Every change is at submission.py:2820–3142 (RWCU, RWV, `_rwwg`, `_rwsrc`, `_rwext`, RSPALL/RSP,
  `_rscall`, `_rsbake`). GVCU, GVI and CUDA are untouched, so r6's gv instances and pw23's rsk/rsk2 kernels compile from
  identical source.
- **Local build.** nvcc 12.8 with RWFLAGS, torch's COMMON_NVCC_FLAGS and the 9.0a gencode. The torch binding is swapped
  for a plain-pointer switch over the same 6 shipped instances (101, 104, 107, 112, 120, 126).
  - 0 spills; no C7510 / "serialized" warning.
  - Registers: 90 (s4z, s8z, s8), 142 (r3x2), 148 (r3x2z), 154 (rwv64k64s4x2).
  - For the 2/SM instances, 148 rounds to 152, giving 2 × 19.5k = 39k ≤ 64k.
  - smem: 132,096 B at 1/SM; 99,328 B at 2/SM, which is 2 × (99,328 + 1 KB) ≤ 228 KB.
  - A dual sm_90 + sm_90a build also succeeds; the sm_90 pass compiles to the stub.
- **Box (H100, 132 SMs).**
  - `bench.sh pw29h-critic --official --shapes verify16,verify32 --layers router`: every CHECK passes (m = 1…1000).
  - PICK m=128 is rw64k32s8z; PICK m=256 is r6.
  - verify16 14.66 vs torch 23.47; verify32 18.46 vs 30.00.
  - `rwocc` reports 1 CTA/SM for 120/126 and 2 for 104/107/112.
- **Box stress (critic script, since deleted).** About 64.7k graph replays for each of 126@128, 120@128, 107@128, 107@256,
  107@250 and 126@121.
  - `out` is NaN-filled before every replay, so any unzeroed slice shows up as NaN.
  - Half the iterations replay twice back-to-back; the result must equal a single call, not double it.
  - Periodic 256 MB read flush.
  - Checks per replay: one-hot rows exact, random rows within 2e-5 relative to fp64.
  - Result: 0 bad, no hang. All counters ended as multiples of 32.
- **lint:** ok (expected multi-build sm_90a warning). **cpu-check --official:** passed, 9/9.

## 1. Self-zeroing ordering: OK
**Producer side (submission.py:2876-2883).** Each CTA writes its slice [e0, e1) of rows*16 float4. The slices partition
the tile, rows < min(TOKS, m − t0) only, so rows ≥ m are never touched.
- SASS: `STG.E.128 RZ` → `BAR.SYNC` → thread 0 `MEMBAR.ALL.GPU` → `ATOMG.E.ADD.S32.STRONG.GPU`.

**Consumer side (:2918-2924).** Thread 0 spins on `LDG.E.STRONG.GPU` + `CCTL.IVALL`, then `WARPSYNC.ALL`,
`WARPGROUP.DEPBAR 0`, and `BAR.SYNC` twice (staging) before any `REDG.E.ADD.F32x4`.
- This is the same pattern as cooperative-groups grid.sync and CUTLASS's GenericBarrier.

**Second token block.** At m=256 the tile index is x + 4z, and t0 = 128z.

## 2. Deadlock: none on the judge's geometry
- A CTA arrives before it can wait, so a waiter needs only that its 31 tile-mates get scheduled.
- **m=128, s8z/s4z:** 128 CTAs at 1/SM on 132 SMs.
- **m=128, r3x2z:** 128 CTAs at 2/SM.
- **m=256, r3x2z:** 256 CTAs at 2/SM, 264 slots. Even at 1/SM the z=0 slab (128 CTAs) holds complete tiles.
- No PDL is used, so no previous-kernel CTAs hold SMs.

## 3. Counter reset: not needed
- Counters are monotonic. target = (arrived/G + 1)·G holds because every launch of a variant adds exactly G = K/256 = 32
  per tile it touches, and launches are stream-serialised. That covers judge replays, bake/validate and m=121/250.
- Other paths never touch `rw_cnt`. `rw_cnt` lives in state for the life of the run (:3142).
- Slot·64 + tile ≤ 1087, well under 4096 counters.
- The counter wraps after about 67M calls, which is unreachable.

## 4. sm_90a: OK
- `_rwext` forces `TORCH_CUDA_ARCH_LIST=9.0a` and restores it afterwards (:3084-3092).
- The box's build.ninja for the shipped hash rw29_f9643a4d contains only `compute_90a,code=sm_90a`, twice. It has no
  sm_90 cubin and no PTX, so the `__trap` stub (:2838) is never in the fatbin.
- Even if both cubins were present, a box test shows the driver picks sm_90a over sm_90 in either gencode order and
  with compute_90 PTX.
- If an old torch rejected "9.0a", the result would be "PREP rw build failed" and the pw23 paths would run. That is
  safe.
- SASS still has HGMMA 64x128x16 (32 per instance).
- The cp.async → wgmma handoff runs on every stage: DEPBAR.LE → MEMBAR.ALL.CTA → FENCE.VIEW.ASYNC.S → BAR.SYNC, with 4 or
  8 fences as expected.
- No extra WARPGROUP.ARRIVE is injected. There is 1 for up-front instances and 4 for rings, matching pw29.

## 5. Dispatch and budget: OK
- `_bk` serves only m == 128 and 256.
- z paths have zf=False, so there is no memset node.
- rw64k32s8 and s8z are restricted to m=128 (:3125).
- Box warm prepare is 76 s, including a 37 s rw build of 6 instances. pw29 built 9.
- Judge estimate: about 105 s (pw23) + 30–40 s + bake ≈ 145–155 s. The build is gated at 180 s (:3150).

## Low-priority notes (not blocking)
- **No spin cap and no co-residency guard.** `rwocc` (:3017) is built but `_rsbake` never consults it. On a device with
  fewer than 128 resident slots, s8z would hang inside prepare. That cannot happen on a 132-SM H200. A guard
  `sms * rwocc(v) >= grid` before baking z paths would make this explicit.
- **Stale comment.** The comment at :2823-2825 still describes v1 ("the first CTA … zeroes the tile").
