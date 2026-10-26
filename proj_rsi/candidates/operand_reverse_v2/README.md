# Operand reverse v2: direct native-A X loader

`state=CPU_PROOF_SM90A_COMPILE_PASS_GPU_UNRUN; policy=ONE_LOADER_REVISION; v1=FROZEN; full_submission=NONE`.

Hypothesis from completed v1 diagnostics: X-consuming SHFL/SEL PCs have940long-scoreboard samples; transpose also696short-scoreboard/292MIO, plus155LG at direct load sites. v1 performs4,194,304fixed transpose warp instructions. Replace only that loader with native WGMMA-A fragment loads; measure this version versus v1 as well as the actual incumbent. A failure is evidence about this version and its next bottleneck, not a terminal rejection of the architecture.

One shape/geometry: unpooler M128/N8192/K4096,128CTAs,384threads, KS1,64channels/CTA, plain native output. Producer still loads6packed-W slots by bulk TMA, decodes/repairs exactly into8shared-B slots, preserves the128thread fence/publish barrier, and math retires sharedB with the same8warp arrivals. Output slab, fullK, precision, pack and escape repair are unchanged. CPU preflight verifies producer body/output source identity except removal of the dynamic register declaration; frozen v1 files are never rewritten.

Direct uint32 register at kstep0..3/i0..3/lane g*4+t:

```
row = wg*64 + warpWithinWG*16 + g + 8*(i%2)
k   = stage*64 + 16*kstep + 8*(i/2) + 2*t
A[4*kstep+i] = bitwise LDG32(X[row,k:k+2])
```

Each thread issues16scalar32bit `ld.global.ca` loads per stage, replacing4vector128bit loads plus32SHFL/32SEL. Logical X bytes remain128MiB per launch. Scalar instructions are4x as many and touch12832Bsectors per warp-stage before cache reuse versus64unique sectors/2048useful bytes. The two16B halves of each32Bsector occur in different scalar instructions; `.ca` is fixed to permit local reuse. Effective traffic/hit rates are unmeasured. This version removes4.194Mtranspose warp instructions while adding~0.786Mscalar-load warp instructions; the expected next risk is LG/L1 issue/sector pressure, not a promise of bandwidth improvement. No cache-policy sweep.

**Async register lifetime:** initial direct-A buffers hold stages0..3. After commit(s), wait_group1 retires all older groups throughs−1; pin retiredA, release sharedB(s−1), and only then load stage s+3 into that same four-slot buffer. Since `(s+3)%4==(s−1)%4`, the current group's A is never overwritten. Stage0 has no refill. Final wait_group0 then pins all4buffers before B63 release/output. Copying v1's current-q refill(s+4) into a direct-A buffer would be illegal; CPU negative control catches60such hazards.

Register safety: remove both64/216setmaxnreg instructions because reduced static allocation could otherwise create a pool deadlock. Actual sm90a compile uses162static registers,62208logical registers/384thread CTA, no dynamic requests,0stack/spill/local and noC75. Runtime opt-in-before-occupancy>=1 and register/local checks remain mandatory. SASS has noUSETMAXREG and noSHFL.BFLY; only inherited producer escape SHFL.IDX remains. Four unrolled rounds contain16HGMMA,4WG-arrive,one output TMA; no X TMA.

CPU preflight proves all524288X indices symbolically against the independent v1 transpose map, all65536BF16 payload patterns are compatible with the unchanged bitwise route,524288arbitrary W values/escapes through shared-B SW128,8192output scalars, and the four-slot async commit-order lifetime. It is not a GPU arithmetic, arbitrary barrier-interleaving or throughput proof.

```bash
python3 proj_rsi/candidates/operand_reverse_v2/preflight.py
```

That command launches no CUDA; it performs a real local nvcc sm90a cubin compile and archives generated source, resource log and SASS. Runtime must additionally build the actual full Torch extension; the local device build is not that gate.

Lab freezes one actual pw36 prepare (SHA2a067059…bccf), v1 generated sourceca92a4fb…85f8 and this v2 source. Both JIT extensions build **before** pw36 import so its prepare budget clock stays normal. Three comparisons use actual `kernel_fn`, frozen v1 and v2, sharing immutable X/W/packed/escape data but separate outputs/counters. Incumbent metadata contains scalar fields only. Every path validates all4096K one-hots, FP64 normal/outliers, task oracles and input immutability. The exact3timed graphs are retained and checked after F1+paired timing with NaN-poisoned original/zero/fresh-normal/fresh-outlier/all-probe inputs (15graph checks). Weight bytes/escape data stay exact and counters zero. Profiles run outside timing and distinguish `rq<...>` v1 from `rq_v2<...>` v2.

Timing:256MiB flush,5discarded cold replays,21sample F1 medians with lower before/after actual-incumbent **and v1** baselines,21rotated3-way paired rounds. Report raw gains against BOTH. A>=0.5us win against actual incumbent in both methods is review-only survival; a smaller/no incumbent win still records improvement or regression versus v1 and the remaining data-delivery question. No automatic family closure, retry, hidden variant, new profile, precision/geometry or full-submission change follows. Root decides the next single evidence-based iteration.

After independent critic review, root alone may issue one bounded exclusive command:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v2/lab.py --json /okc/logs/operand-reverse-v201.json
```

No GPU or remote compile by this author. Current state remains GPU_UNRUN; first runtime/API/numerical fault invalidates this version's timings and calls for diagnosis.
