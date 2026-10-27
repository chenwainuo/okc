# GV16 actual-binary dependency supplement

Projection only. CPU extraction from the exact previously measured native SO; no GPU, JIT, queue or source changes. This supplements frozen robustness DONE evidence without changing its disposition.

## Finding

The 2080→1992 static instruction reduction is real, but does not prove a shorter executed critical path. Twelve of the 88 removed instructions are NOPs (non-NOP 2050→1974, −76). The new kernel also introduces repeated lane-base reconstruction. Both kernels retain the cursor/SHFL recurrence, all sixteen predicated XOR destinations per repair, predicate save/restore, weight loads and tensor instructions. No GV16 hardware-counter capture exists here, so bandwidth or scoreboard bottlenecks remain UNKNOWN.

The robustness primary equal-weight diagnostic point remains control−new **−0.063999556 us** across48 pairs. Per-weight points +0.015999190/−0.128000043/0 us are preserved. This evidence identifies code-generation tradeoffs, not their causal timing fractions.

## Exact native evidence

Input SO: results/proj-rsi/scheduler/artifacts/research.gv.unpool16.deltakey.01/collection-1793137803622279277/logs/scheduler/gv-unpool16-deltakey-01/run/actual.so, SHA256 d1c06fd80d1f5391986c5257765851e95eb749ded49129601a5183b0871f3c49. The robustness job loads these exact bytes. CUDA12.8 cuobjdump CPU extraction is retained in actual.sass/actual.resources; instructions.json contains parsed instruction rows and opcode counts.

| Fact | Control | Delta key |
|---|---:|---:|
| Static instructions including NOP |2080|1992|
| NOP |30|18|
| Registers |110|108|
| Stack/local bytes |0/0|0/0|
| Static/dynamic shared bytes |1024/5120|1024/5120|
| Runtime queried active CTA limit |4|4|
| SHFL.IDX / P2R / HMMA |17/16/128|17/16/128|
| LDG / LDS.128 / PRMT |113/12/262|113/12/262|
| S2R |5|20|
| IMAD.SHL.U32 |6|22|
| IMAD.IADD |10|25|
| ISETP.EQ.U32.AND |272|0|
| ISETP.NE.U32.AND |16|256|

These are whole native function static counts, not executed repair counts, stall percentages or time attribution. Runtime occupancy comes from the existing result receipts, not an estimate from register arithmetic.

## Dependency observations

First repair: control 0x0730→0x0790 extracts target and hm, compares target to lane then merges hm into P5. Per-index comparisons depend on that gate. New 0x0760 SHF→0x0780 mask→0x07b0 **IMAD R7,R0,-0x40,R3**→0x07e0 comparison replaces those gates. The source subtraction is fused into IMAD; there is no standalone SUB opcode. The new first XOR occurs at0x0840 versus old0x0870. This first-site sequence does not support the simplistic claim that SUB necessarily lengthens the path.

At fifteen later unrolled repair sites the new binary rematerializes lane: for example0x0db0 **S2R R0,SR_TID.X**→0x0dc0 IMAD.SHL x64→0x0dd0 mask0x7c0→0x0e80 **IMAD.IADD R0,R0,1,-R7**→comparisons. Old code keeps lane in R100 and compares against it at0x0e60, without the corresponding three-instruction rematerialization. This is PROVEN extra work and an additional input dependency; its exposed latency is UNKNOWN. Register allocation may prefer shorter live ranges; its motive is not established by the dump.

Both arms retain P2R16 and seventeen SHFLs, and still serialize cursor advancement through next-record SHFL before that record can be decoded. New next-SHFL firstsite0x0990 versus old0x09e0; moving source text earlier alone previously failed because ptxas reordered it. These PCs show layout/order, not cycle savings. The record is initially fetched in LDG and distributed by SHFL; packed weight LDGs and PRMT/decode feed repaired registers, which feed unchanged HMMA. Which input waits dominate the critical path is unmeasured.

## Cause audit and smallest discriminator

PROVEN: unchanged queried4CTA capacity; no spill; unchanged load/HMMA/SHFL/P2R static counts; repeated new S2R lane rematerialization; robust fixture/oracle/cleanup checks passed. UNKNOWN: fraction of runtime in repair, bandwidth saturation, issue pressure, exposed IMAD/S2R latency, or why timing varies by order/weight. Prior PW38 GV53 M1 counters belong to a different kernel/geometry and cannot establish GV16 bottlenecks.

Cheapest next implementation discriminator is **CPU-only**, one exact unpooler template: compute the lane base once through an opaque PTX value and retain it across the unrolled repair sites, leaving the packed-key arithmetic, cursor order and all math unchanged. Inspect whether the resulting SASS actually removes the fifteen repeated S2R chains without spills or a lower queried occupancy limit; a source-level hoist alone proves nothing. If codegen remains unchanged or resources worsen, preserve the result and park this version. A GPU timing replay without a changed mechanism adds little information. No candidate has been built or queued by this supplement.

If root later prioritizes causal profiling, the minimal informative capture uses the existing exact SO, one matching fixture, one control and one new launch, querying actual cluster attributes and supported NCU metrics first. Request executed instructions, issue/eligible-warps, long/short-scoreboard and math-pipe-throttle samples, DRAM/L2 throughput, plus sampled stall PCs if supported. Separate profiled counters from unprofiled timing; aggregate counters alone cannot locate the repair fraction. This diagnostic remains optional and lower priority than larger actual-selected opportunities; no large framework or repeat proof is warranted.

Reopening condition: codegen demonstrably removes rematerialization at unchanged legal resources, or same-GV16 sampled PCs identify a repair dependency worth changing. Preserve the mixed delta-key version and all losses; do not discard the family or integrate the negative aggregate point.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize→repeat. [Contract](../../ITERATION.md).
