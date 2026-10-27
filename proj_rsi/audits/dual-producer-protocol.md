# Alternating dual producers: independent source/protocol review

`state=CPU_SOURCE_PROTOCOL_NO_BLOCKER_FOUND; GPU_ADMISSION=HOLD_FOR_FROZEN_LAB_AND_INDEPENDENT_REVIEW; GPU_BY_REVIEWER=NONE; speed_claim=NONE`

Snapshot: generated extensionSHA`4552e4e898d75da240be91c946de82ea12cecf870989bd9fcf40de9e8616d95b`, kernel body`05c009c93c22404273ec6bb761aee8cfbd0dafc6f787d921e76724352a2645b0`, device cubin`af3d18367328876cfca4babe2d2232363f49c4347ba1911ba00f810c11e1b9c7`, SASS`556039ac78367c19f692204f2d7f26257aa544432e529b9cbb60149f73870934`. Root-owned [source](../candidates/operand_reverse_dual_producer/kernel_body.cuh), [prototype](../candidates/operand_reverse_dual_producer/prototype.py), [CPU evidence](../candidates/operand_reverse_dual_producer/preflight.json) were read only. No reviewer root-directory edits, extension load, remote action or GPU work. Changes after this snapshot require recheck.

## Thread roles / ownership

|threads|role|stages / named barrier|
|---|---|---|
|0..127|math WG0, token0..63|all64stages; math bars1/2count256 together with WG1|
|128..255|math WG1, token64..127|all64stages; unchanged native A4 pipeline|
|256..383|producer0, four warps w0..3|s=0,2,..62; bar3count128; leader256|
|384..511|producer1, four warps w0..3|s=1,3,..63; bar4count128; leader384|

Initial CTA `__syncthreads` includes all512threads. Only tid256 initializes all W/B barriers, fences initialization and issues all6initial W copies. Neither producer returns before that CTA barrier. `w=warp&3` maps both producers to the same64channel decode tile at different stage parities; their immutable CSR lists are intentionally duplicated. X prefetch remains guarded by warp<8, so producer warps do not read the math-only packed-X addresses.

W6 and B8 are both even: `owner(s)=s%2=owner(s+6)=owner(s+8)`. Each raw W slot and decoded B slot therefore has one producer owner over its entire lifetime. Producer0/1 have3Wslots/4Bslots each; no cross-owner publisher/refill race. Separate bar3/bar4 prevents128-arrival barriers from mixing the two producers. Actual SASS has predicated bar4@0x2e70 and bar3@0x2eb0, each0x80threads, while initial bar0 and math bars1/2 remain separate.

## W / B phase induction

For global stage s:

```text
Wslot=s%6; Wphase=(s//6)&1
Bslot=s%8; Bfull_phase=(s//8)&1
Bempty_wait_phase=((s//8)&1)^1, used only for s>=8
```

Base s<6: initialization expects6144TX bytes then queues exactly the existing compressed half-tile. Owner warps wait Wfull phase0 before LDS. Base s<8: decoded B slots have no prior readers, so no Bempty wait is needed. Each decoded tile uses unchanged rq_unit/rq_fix and all16scalar stores/lane, followed by **every writer's** proxy fence and its own128thread barrier. Only leader256/384 publishes one Bfull arrival for that slot.

Inductive W reuse: all128owner threads have finished their W LDS and decoded shared stores before their local barrier releases. After the barrier the owner leader may overwrite that same raw slot with stage s+6, using exact expected6144bytes. No other owner reads that W slot. The next same-slot read waits opposite Wfull parity; it cannot miss two completions because the only refiller must first complete the intervening decode/barrier. Initial/refill source address `((nb*64+s)*12288)+half*6144` is unchanged and partitions original packed weights exactly.

Inductive B reuse: each of8math warps contributes one lane0 arrival to Bempty only after its covering WGMMA wait retires that stage. Producer stage s>=8 waits for all8arrivals retiring stage s−8 before overwriting the slot. Bfull count1 then publishes the opposite phase. A fast math WG cannot permit reuse while the other WG still reads: Bempty count8 includes both. No W/B phase ABA is allowed by these handshakes.

[NVIDIA PTX mbarrier semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive) default arrive to release and successful try_wait to acquire at CTA scope; parity waits must track the current/immediately preceding phase. The existing writer proxy-fence→local barrier→leader publish→consumer acquire chain is retained, now using separate producer barriers. This is source ordering analysis, not a substitute for actual binary/runtime checks.

## No logical dependency cycle

Let P(s)=decoded publication, I_g(s)=math WGg issue/commit, R_g(s)=retirement/release. Necessary edges:

```text
P(s) <- P(s-2)                           same producer order
P(s) <- P(s-6)                           refill, if s>=6
P(s) <- R_0(s-8), R_1(s-8)               B reuse, if s>=8
I_g(s) <- P(s), I_g(s-1)
R_g(s) <- I_g(s+1) + wait_group1         s<63
R_g(63) <- I_g(63) + final wait_group0
```

The important correction to shorthand is **R(s−8) is emitted in round s−7**, not merely when stage s−8 is issued. s−7 is still strictly earlier than s; B8 gives ample ordering distance. A read-only independent320node model (64publications +2×64issues +2×64retirements) topologically removes every node, including independent consumer WG progress. Stage0/1startup uses initial W copies; final drains need no future producer stage. No cross-CTA counter or full-grid co-residency assumption exists. Fair warp scheduling and finite memory/TMA completion are ordinary execution assumptions, not measured performance claims.

## CSR skip / cached32entry windows

Source correctly adds `while(nxs<s)` **before** inherited rq_fix. Each producer's cursor starts0 on the same sorted immutable list. It skips earlier stages without XOR, applies every record with its assigned s using old rq_fix, and keeps the first later record for the next assigned stage. Without skip, the other parity's first record would block all later fixes; the root negative control catches that defect.

New skip uses exactly inherited advancement: increment ei; at `(ei&31)==0` reload `el+ei+lane`; broadcast entry `ei&31`; derive unsigned nxs. nxs/ei/s are warp-uniform, so full-mask SHFL and reload execute with all32lanes; lane-targeted XOR finishes before the next uniform advancement. Four warps may traverse different list lengths safely before their128thread producer barrier. Producer1's w0..3 still addresses lists `(nb*8+4*half+w)`, within the existing8lists/nb allocation.

Sentinel0xffffffff gives nxs32767>all s0..63. `_rqesc` reserves at least one sentinel and CAP is a multiple32, so fetching the cached window containing that sentinel remains in-bounds; neither skip nor fix advances past it. The root tuple-list model proves parity application but does **not itself emulate** lane-cached reload/sentinel encoding. Reviewer additionally ran a read-only encoded-entry cache model at record counts0,1,31,32,33,63,64,65,512,4096, including33same-stage edits crossing a window and both parity skip/apply boundaries. Full multiset application matched the sequential XOR oracle exactly; no cached-window OOB. CAPs were32,32,32,64,64,64,96,96,544,4128. This supplements the unchanged inherited helper/source proof; runtime dense/probe validation is still required.

Metadata traversal is approximately doubled, not free: both owners walk earlier opposite-parity records and load their own initial/cache windows; owner0 need not consume final stage63records. Each actual edit is XOR-applied once overall. Packed W bytes, decode operations per stage and decoded B stores remain unchanged; only their producer scheduling changes.

## Tail / output reuse: precise sufficient invariant

There is no explicit producer-exit join. It is therefore too strong to claim every producer's literal EXIT has executed before output stores. The sufficient invariant **does hold**: consuming stages62/63 implies both producers passed their last store/fence/barrier and published their last Bfull. Neither last stage issues a future W copy (`s+6>=64`), and no further shared write follows its publication. All queued W copies have been waited before their corresponding last decode. The producer tail then only branches/increments/returns.

Math final wait_group0 retires all remaining WGMMA, release63 executes, then math bar1count256 permits the unchanged output slab reuse; output writes cannot race any future producer scratch write. Literal thread return is unnecessary for that property, and CTA lifetime still waits for all threads. No extra join/barrier is requested by this review.

## Concrete compile evidence / remaining gates

Independent cuobjdump shows main122regs/pack22,0stack/local, and no LDL/STL/USETMAXREG. Kernel launch bounds are512, prototype occupancy query and actual launch use512, rx_pack remains256×256. Main allocation rounds122→128per thread, **128×512=65536 exactly**; no dynamic-register pooling is used. Static resource report includes1024B shared reserve; requested dynamic scratch remains103600B and runtime occupancy must include both. There is no register headroom after allocation rounding.

Read-only physical A4 checker passes this actual SASS: four16register sets104..119,88..103,72..87,56..71; math loop PCs0xbb0..0x1890, four wait1s, final wait0@0x18d0. Outlined math wait paths write no A registers. The math/Xpack/output source restores to exact frozen packed-X after undoing only the reviewed producer/block-size changes. Reusing those numerical/index proofs is justified by that restoration; it does not certify new GPU launch/runtime.

No source-level safety correction is required from this review. Admission remains **HOLD** until root freezes a complete lab and independent critic verifies it: actual Torch2.8 JIT/resource/SASS/512thread pool, SMEM opt-in before occupancy, occupancy≥1 and real launch; pack freshness/poison and every-byte immutability; all4096onehots/FP64 normal/outlier and actual timed graph/fresh-output oracles; exact incumbent and packed-X predecessor comparison with whole two-kernel pipeline; cleanup/process ownership. A faster producer is an unmeasured hypothesis, not a2×promise or authorization to port a submission.
