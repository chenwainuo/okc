# Conditional fixed-width byte-plane W consumer

`state=CPU_LANE_AND_BIT_PROOF_PLUS_INVERSE_SASS; GEMM_IMPLEMENTATION=NONE; GPU_BY_AUTHOR=NONE; serving_gain=UNMEASURED`

2026-10-27 result update: [byteplanes01 is valid and flat](hw-compression-byteplanes-result.md): raw G/N67,112,960/67,114,496B, planes67,122,176/67,114,240B, ratio-of-ratios1.000141. This consumer design remains historical/conditional and **is not admitted for a matmul implementation**. Root's next authorized version tests sign-low XOR with the same frozen copy reader; it does not implement this backend.

Admission still depends on root's actual **four-case new-reader matrix** and independent review. The earlier two-node proposal is superseded. A flat/invalid matrix does not automatically authorize a matmul; a positive result establishes only this payload/reader's response. Previous versions and their evidence stay preserved.

## Exact mapping into existing RQ A registers

Actual frozen [pack](../candidates/pw36/submission.py:4610) maps channel tileT16, k64stagec, q∈[0,4), lane=4g+t and v=4kh+2half+e to:

```text
n = 16*T + 8*half + g
k = 64*c + 16*q + 8*kh + 2*t + e
g=lane//4; t=lane%4; kh=v//4; half=(v//2)%2; e=v%2
```

One math warp owns16channels×64K=1024weights,32BF16weights per lane. Candidate per-warp2048B layout has four512B planes, each32lanes×16bytes:

```text
plane0: low bytes of q0's8values, then q1's8values, per lane
plane1: low bytes of q2's8values, then q3's8values
plane2: high bytes of q0's8values, then q1's8values
plane3: high bytes of q2's8values, then q3's8values
layout [NB][KC][8warps][4planes][32lanes][16bytes]
```

For one q, let L0/L1 contain low bytes of v0..3/v4..7 and H0/H1 their original high bytes:

```text
A[4*q+0] = PRMT(L0,H0,0x5140)  // W[v0] | W[v1]<<16
A[4*q+1] = PRMT(L0,H0,0x7362)  // W[v2] | W[v3]<<16
A[4*q+2] = PRMT(L1,H1,0x5140)
A[4*q+3] = PRMT(L1,H1,0x7362)
```

For example0x5140 emits bytes[L0.byte0,H0.byte0,L0.byte1,H0.byte1]. No shift/mask/table/variable boundary is needed. Fourq produce16packed-u32 A registers: **16PRMT for32BF16**, matching the requested4PRMT/8 denominator.

The new CPU-only [proof](../candidates/hw_compression_byteplanes_inverse/proof.py) checks every bit pattern exactly once over128×512W positions, the full warp/q/lane/value bijection, equality to the actual `_pack_nat` function extracted from frozen pw36, every packed-A halfword, inverse W scatter, both64channel halves and all16B lane-load offsets. NaN payloads/zeros/infinities remain integer bits; no floating operation canonicalizes them. [Microcompile SASS](../candidates/hw_compression_byteplanes_inverse/inverse.sass) has exactly4LDG128+16PRMT+4STG128, all A outputs stored,0stack/spills. It has no host launcher or WGMMA. Its32registers **do not establish full-kernel fit or launchability**.

## Exact delta from current source, not a matmul port

Current [rq_unit](../candidates/pw36/submission.py:4056) consumes3uint4/48B and produces16u32/64B per lane. Proposed Q[GS][4] consumes4uint4/64B; per-warp shared base changes1536→2048, plane spacing remains512B, per128channel stage12288→16384B. HALF offsets6144→8192; full packed-stage stride12288→16384 globally. All W bulk-copy byte counts and full-barrier expected TX bytes must change together, including EARLY and normal producer paths. X tensor map/layout/delivery, W-empty-after-LDS, X-empty-after-covering-wait, WGMMA fence/waits/A-buffer liveness and epilogue/reduction stay semantically identical to the admitted baseline.

Store original high bytes, so remove `Tb`, escape allocation/list addressing, initial list prefetch, cached nx/nxs/ei/ec state, per-stage `rq_fix` and its SHFL/window reload loop. Escapes are not rounded away; every original high bit is in plane2/3. This permits exact reconstruction without an escape path for **all65536patterns**, not merely normal weights. Router hi/lo and unrelated paths are outside this target.

Current source decode model is36integer ops/lane/k64 (4×9) plus escape handling; candidate16PRMT, saving20source ops, at the cost of one additional LDS128/lane/stage. For NB64,KC64,8math warps, that is655360fewer warp-level decode instructions and32768extra LDS128 instructions before escape savings/compiler effects. This is source accounting, not executed SASS counters or a latency forecast. Microcompile verifies only the candidate PRMT denominator.

## Traffic and resource floors, m128 unpooler first

For N8192,K4096,NT128 and unchanged KS partition, W occurs once across all channel/K CTAs: payload48→64MiB, logical W TMA/LDS+33.33%. X is1MiB repeated across64channel blocks:64MiB total logical X TMA in either case. W+X TMA fill112→128MiB (+14.29%). Per math stage both channel WGs consume the same shared X tile; conceptual X operand reads128MiB plus W LDS48→64MiB gives176→192MiB math shared reads (+9.09%). These are logical bytes, not a proved aggregate shared-port/cycle ceiling; physical broadcast/transaction behavior and producer/read paths need measurement. Hardware reducing HBM bytes does **not** shrink TMA destination/shared storage, LDS payload or uncompressed register values.

`SMEM=1024+ST*(XB+WB)+24*ST+16`, XB=128*NT. For NT128/ST5:144520→165000B (+20480), below Hopper227KiB block limit. For NT256/ST5:226440→246920B, **over** that limit; a same-ST m256 port fails before launch. Reducing ST would be a separate pipeline decision, not an implied follow-up. Do not generalize the m128 fit to all shapes.

Raw Q registers grow12*GS→16*GS; A buffers and FP32 accumulators stay. Removing list state/pointer/Tb may recover roughly7persistent u32 slots, but compiler liveness decides actual pressure. Existing3WG RQ uses producer40/math232 setmaxnreg with static168regs:128*40+256*232=64512=384*168. **A new smaller static allocation is not automatically safe** with unchanged role limits: static160 provides61440, allowing at most220math regs before granularity;232would oversubscribe. First full-kernel compile, if later authorized, must verify actual static allocation, emitted role limits/CFG maxima,0spill/local and `128*producer_limit+256*math_limit <=384*static_allocated_regs <=65536`, accounting for hardware allocation granularity. Actual opt-in SMEM→occupancy→launch remain gates. No flag sweep or fit claim follows from inverse-only32regs.

## Decision after root's matrix

The active diagnostic packs **raw contiguous Wt** in64value groups with64B low/high spans, consumed by volatile global copy. A future RQ representation instead groups native WGMMA fragments with512B plane spans and uses bulk TMA/shared LDS. Byte ordering, alignment/window boundaries and reader differ; a good copy response is a prior, **not a transferable ratio**. Retain raw controls under the new reader and compare within-payload GEN/NONE first, then allocation ratios across payloads. Do not extrapolate the zero control's73.8% aggregate read saving.

If the response is credible, root may authorize one exact-fragment representation/resource gate, retaining ordinary/VMM-NONE controls as needed. Any subsequent full prototype is one m128unpooler policy, with actual frozen-pw36 `prepare`/`kernel_fn` winner as comparator. Fresh bake may pick RQ or Lt; do not force a weaker RQ or compare only with Torch. Require all weight bits/maps, real all4096onehots/FP64 normal/outlier oracles, actual timed graph/fresh-output/NaN-poison/input-immutability checks and checked owner/graph lifetime. Current numerical precision, X delivery and reduction order remain binding. Acceptance would need a real paired graph **and** F1 gain against that actual incumbent; the source-op saving alone is insufficient. No official/full-submission copy, new backend or GPU launch is authorized by this conditional design.
