# pw24h critic review (k2x / n128x bf16 kernels, `bh<...>`) — 2026-10-26

**Verdict: REJECT as a scored judge candidate; the kernels are APPROVED for porting.**

The file says of itself that it is "H100 research copy ... NOT a judge candidate" (submission.py:2). It is a standalone
harness and does not contain any of the integration the brief assumes:
- no r6 gv kernels and no pw23 router kernels;
- every shape it does not pick goes to torch (submission.py:642-644).

Submitting it would return correct answers, but at m <= 32 and for the router it would fall back from r6 + pw23 to
torch, a large board regression against pw23 at 3399. The only safe use of it on the judge is a measurement-only check.
I found no wrong-answer, hang or OOB defect in the k2x, n128x or rsk2x kernels themselves.

What I ran:
- I built `_src()` with the exact FLAGS (submission.py:473) plus `-Xptxas -v` on nvcc 12.8. The torch headers were stubbed, because local torch has no cusparse.
  - It contains 20 instances, all with 0 spills and 0 stack.
  - Registers: k2x/k2x6/k2s1 58, n128x/n128x8 90, rsk2x 154, n128rsx 138.
  - There is no C7514/C7515 serialization. ptxas reports info-level C7519 ("warpgroup.arrive injected") on the SS KS=2 and n128 instances.
- k2x SASS (cuobjdump), per 128-k slot: 1 TRYWAIT, 1 WARPGROUP.ARRIVE, 8 HGMMA.64x64x16 with a single gsb0, `DEPBAR.LE gsb0,1`, then a lane-0 `SYNCS.ARRIVE` on the empty barrier. After the loop comes `DEPBAR.LE 0`.
  - The WARPGROUP.ARRIVE is the injected one (C7519), so FN=1 does not remove the per-slot arrive in SASS. This has no correctness impact, and the measured times already include it.
  - Epilogue: 16 F2FP.PACK_AB with the low half = D[4i], then 4 STSM.16.MT88.4 at xring (+0x1c000), then BAR.SYNC 1,128.
- Box log pw24h-r6 (prepare 39.6 s standalone):
  - all 8 CHECK lines are ok, with probe rows exact;
  - picks: pooler128 k2x, o128 n128x, pooler256 n128x, and cuBLAS everywhere else.
- `python3 -m proj_rsi lint`: ok. `cpu-check --official`: 9/9 ok (CPU path only). Size 32.6 KB.

## 1. Shared full barrier: OK
- With MB, the full barrier is initialised with count 2 (`MB && XON && TMAON`, :139). Each producer does `arrive.expect_tx` with its own constexpr bytes: WTB = KS·BM·128 (:161) and XTB = KS·BN·128 (:190).
  - These equal the bytes the TMA boxes actually issue: KS·(BM/64) boxes of 64×64 for W, and KS boxes of 64×BN for x.
  - There is no partial stage: K is 8192, 4096 or 1024, so KC = K/128 is 64, 32 or 8, and the n128 path (KS=1) has KC = 128, 64 or 16.
- Rows of the x box beyond m are zero-filled by the tensor map, which uses dims {K, m} and OOB fill NONE (= zeros), and the full box bytes are still counted. So expect_tx must be the full XTB, as it is. The implementer validated m−7 for this.
- Parity: the consumer waits on (s/ST)&1, and each producer waits on the empty barrier at ((s/ST)−1)&1 for s ≥ ST (:155-158, :184-187).
  - Neither producer can get more than ST slots ahead.
  - A fast producer cannot arrive into the previous phase of the full barrier: its use k+1 needs empty phase k, which needs full phase k to have completed.
  - The empty barrier cannot advance two phases past a slow producer, because use k+1 of that slot needs the slow producer's own arrive.
- If one producer's complete_tx arrives before the other's expect_tx, the tx count goes transiently negative with one arrival still pending. That is legal, and the phase cannot complete early.

## 2. wait_group 1 + release s−1: OK
- After unit u is committed, `wait_group<1>` leaves only unit u in flight, and then u−1 is released (:253-261).
- The tail does `wait_group<0>` and then releases the last WN units (:263-268).
- RS path: one fence per slot. A of stage s−1 is pinned until its wait (:301-305). The KC-parity tail is correct for even and odd KC (:309-312).

## 3. Epilogue: OK
- The smem tile at the xring base is written only after `wait_group 0`. All KC full barriers have been waited, so every TMA has landed, and producers issue exactly KC stages.
- The tile is 8 KB (64×64) or 16 KB (128×64), which is ≤ XTB·ST.
- stmatrix.trans mapping: matrix j has tokens 8(i + j/2) + rr and n chunk 16w + 8(j&1). That matches q0..q3 = D[4i..4i+7], with the low half = even token.
- The swizzle XORs the chunk with tk&7, and the read-back (:383) uses the same XOR.
- Rows with t0+tk ≥ m are not stored (:381). The uint4 stores are 16 B-aligned because N%64 == 0.
- `cvt.rn.bf16x2.f32` is RN-even and propagates NaN. One-hot inputs are exact.

## 4. 64×128 paths: OK
- m64n128 has 64 fp32 accumulators per thread; n128x uses 90 registers with no spills.
- smem:
  - n128x: 1024 + 9·24 KB + 288 = 222,496 B;
  - k2x: 1024 + 7·32 KB + 224 = 230,624 B.
  - Both are ≤ 232,448.

## 5. Build robustness: findings
1. **MAJOR: no `__CUDA_ARCH_FEAT_SM90_ALL` guard** around the wgmma, stmatrix and TMA asm (:76-396).
   - If the judge's torch appends a compute_90/sm_90 pass (the box pitfall in LOOP.md), the extension fails to build.
   - prepare catches the exception (:546-550), so the result is all-torch rather than a failed check, but the measurement is lost.
2. **MINOR:** `cudaGetDriverEntryPointByVersion` is called unconditionally (:407). pw24 has a `#if CUDART_VERSION >= 12050` fallback; port that too.
3. **MINOR:** `bh_go` only checks `K % 64` (:429). For KS=2 it should reject `K % 128`, or the last 64-k block is silently dropped. This is latent: no layer hits it.

## 6. Integration items, all missing in this file
- There are no r6 gv or router kernels to compare SASS against.
- There is no PREPARE_BUDGET_S guard. The bake (:600-628) and the ladder probes (:565-599) run unconditionally on the judge. Drop the ladder.
- Bake only k2x, n128x, n128x8 and rsk2x. That cuts the 20 instances down, and the 35 s box compile with them, before the module sits behind r6's ~110 s judge phase.
- Dispatch is exact-m only (:637); keep that.
- The 1% pick margin (:624) is thin: o128 n128x won by 1.7% on the box.
