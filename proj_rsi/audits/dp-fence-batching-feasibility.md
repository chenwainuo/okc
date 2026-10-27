# DP paired register fence: ISA/lifetime feasibility

`state=ISA_FEASIBLE_BUT_CPU_CODEGEN_FAILED; hardware_waitmask=UNDECODED; speedup=UNMEASURED; GPU_CALLS=0`

Bind current DP281ff/source97dd, actual JITbebe4726; see [fresh PC audit](dual-producer-pc-roi.md). Root subsequently authorized a new isolated paired-fence CPU prototype; this document itself contains no new kernel/GPU action. A6 is excluded: DP512threads already allocates128×512=65536registers and A4 occupies64input registers/thread.

[PTX8.7 §9.7.15.7.1](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence) orders earlier accesses to the **same** accumulator/A-fragment registers before WGMMA; it does not require a new fence merely because unrelated registers change. First WGMMA needs a fence; same-shape asynchronous accumulator chains are ordered by default. Generic access to pending A/D still requires a covering wait (§9.7.15.7.3). All warpgroup participants must execute aligned operations consistently. Shared-B proxy ordering remains separate. These rules permit one fence covering two preloaded A sets, provided neither is generically accessed afterward before its WGMMA and D has no intervening generic access.

NVIDIA [CUTLASS3.8 wrappers](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cute/arch/mma_sm90_gmma.hpp) distinguish actual `wgmma.fence`/wait/commit from empty-asm operand pins. Existing DP uses the same kind of empty pins. They are compiler lifetime constraints, not automatically extra hardware moves; every future binary must verify that assumption.

## Exact A4 schedule, per math WG

Four physical input banks each hold16u32 for32BF16 across k64. Current mapping is q0R104..119,q1R88..103,q2R72..87,q3R56..71, with D R24..55. New allocation may differ and must be rediscovered.

```text
startup: load A0→q0, A1→q1 ONLY; initialize D before initial fence
for b=0,2,...62:
    await decoded B_b; fence A_b and A_(b+1), already loaded
    if b+2<64: load A_(b+2) into q[(b-2)%4] AFTER fence
    issue/commit group b; wait1 retires b-1 if present; release B_(b-1)
    if b+3<64: load A_(b+3) into q[(b-1)%4] after that retirement
    await B_(b+1); issue/commit group b+1 WITHOUT another fence
    wait1 retires b; release B_b
tail: wait0 retires63; pin all banks/D; release B63; unchanged output
```

At batch entry, pending old group is at most b−1. Bank b−2 has already retired in prior round, so the first future load cannot overwrite a pending operand. The next future bank b−1 becomes available only after group b's wait1. The b+1 input bank remains unchanged since the batch fence, even though the two other banks receive future loads. Next batch fences those future values before use. Startup0/1 has no old pending groups; batch62 skips64/65 loads and final wait0 covers63. A read-only symbolic64stage model confirmed64issues,64distinct stage loads,32fences, no pending-bank overwrite and exact final pending63.

This retains64groups/commits and stage-by-stage B releases; it is **two groups per register fence**, not merging two stages into one longer commit group. Bfull/Wfull/Bempty, two parity producers/CSR, B proxy publication, output and fresh packed-X API stay unchanged. Generic descriptor/index work is not matrix A/D data. No D generic read/write may appear between the paired groups; source `rq_facc`/`rq_fa` pins require actual codegen review rather than semantic assumptions about C++ variable references.

## Actual current SASS evidence / hardware limit

Current DP math loop contains no actual generic destination in D R24..55; D changes via same-shape HGMMA only. Existing A4 lifetime checker passes. Initial16LDG128 sites load A0..A3 before first ARRIVE. Source refills A(s+3) after wait1(s), before ARRIVE(s+1), so a fence may encounter outstanding future-bank loads.

|current ARRIVE PC|upper64bits shown by cuobjdump|fresh samples / long-SB|
|---|---|---:|
|0xd10|0x020fe20000000000|463/462|
|0x1070|0x000fe20000000000|0/0|
|0x12d0|0x000fe20000000000|0/0|
|0x1600|0x020fe20000000000|370/370|

All initial16vector loads share upper encoding0x000768000c5e9d00. These are **raw scheduling/control encodings**, not a proven wait-mask decode. Installed nvdisasm12.8 `-json` reports ARRIVE opcode without dependency-mask fields; `-raw/-hex` and [NVIDIA Binary Utilities docs](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html) do not provide the Hopper bit-layout semantics needed to label a particular mask or registers. No community encoding is promoted into primary evidence here. Hardware may impose broader waits on prior LDGs than the minimum PTX per-register contract. ARRIVE long-SB samples therefore do not establish current-X latency, intrinsic fence latency or a removable time fraction.

## Benefit remains falsifiable

Paired scheduling halves register-fence count64→32/WG, keeps256HGMMA/WG,64stage loads and64commits, but moves some X refills from3ahead to2ahead. That changed overlap is an explicit confound. Removing today's hot0x1600 may merely move waiting to a retained fence; no833sample→microsecond saving is claimed. Fresh DP still exhibits real failed Bfull lanes and producer backpressure, so fence samples cannot all be assigned to X.

First allowed source/compile gate must prove per-register fence coverage, pending-A immutability, no generic D accesses, same input/output maps, zero spill/local/C75,≤128registers,512thread launch/occupancy and physical64K pool. Actual SASS must contain half as many ARRIVEs and future loads **after the intended batch fence**; hoisting or slot aliasing defeats the hypothesis and is a HOLD/diagnose result. Generalized actual-JIT checks run before math, with old invalid-lifetime negative control. Runtime admission then requires the complete four-way frozen-incumbent/DP/STMX/new candidate validation and whole-pipeline F1+paired evidence. No gain or implementation admission follows from ISA legality alone.

## Actual CPU implementation outcome

The isolated [candidate](../candidates/operand_reverse_fence_pair/README.md)
retains exact parent source outside startup/math scheduling and scratch. Its
root-requested revision moves **both** Bfull waits before the paired fence and
defers releases of b−1 and b until after the second group. Steady future loads
are unconditional; startup and tail use compile-time policies. The text schedule
above describes the earlier ISA proposal, not this revised release timing.

The revised symbolic model checks64issues/64loads/32fences and the wait-for DAG.
R(even r) now follows I(r+1); R(odd r<63) follows I(r+2). Thus producer stage s
waiting for decoded-slot release(s−8) depends only on issue(s−7) for even s or
issue(s−6) for odd s. Requiring publication of b+1 before issuing b remains
acyclic at B8; no claim is made about the added latency/backpressure.

NVCC/ptxas12.8 did **not** realize the intended physical A4 schedule. Initial
and control-flow repairs emitted C7520/C7519 and48ARRIVE/48staticHGMMA instead
of6ARRIVE/48HGMMA across startup/steady/tail. v3 SASS loads futureA2 chunks
into currentA0 registers after each compiler-inserted wait0/fence, e.g current
R80 at0xb40 is reused by LDG R80 at0xba0. Empty C++ asm pins leave no actual
PTX operation; MOV-self and identity-PRMT pins were eliminated too. This
establishes actual reuse/serialization, not the optimizer's full decision cause.

One further root-authorized test used real predicated volatile shared stores of
all16 retired A words after covering wait1 (plus final wait0). The512B CTA scratch
raises requested SMEM103600→104112 and adds4MiB logical shared writes across
the grid. The stores survived, but source`ad08afe6…` emits C7512+C7519,
126registers,48B stack/48B spill stores/144B spill loads and still48/48fences/MMAs.
It therefore fails independent spill and fence-count gates. No GPU, full timing
lab or candidate-JIT admission followed; failed sources/logs remain preserved.

This is a failed compiler-expression version with measured source/device
evidence. It neither quantifies removable ARRIVE time nor proves paired-fence
hardware infeasibility. Reopening needs a materially justified liveness/compiler
mechanism, not more identity instructions, warning suppression or flag guesses.
