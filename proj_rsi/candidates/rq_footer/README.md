# rq_footer — NO-GO after bounded CPU/source feasibility

Decision: do not implement/compile/queue this128B footer policy. The current evidence does not credibly support >=0.3us net gain. No CUDA candidate, GPU preparer, lab, remote compile, GPU launch, full submission or official slot. This closes the present policy, not a claim of physical impossibility for every prepared sparse representation.

Parent is frozen `rs32/submission.py` SHA256 `a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`. Only its helpers/kernel were inspected/used. No rs128p dirty kernel/lab imported or copied; only its explicitly requested pureCPU footer model and audit results were read. `router_small` remains frozen/untouched.

## Actual metadata eligibility

`feasibility.py` runs the challenge reference generator on CPU, frozen `_rqpack`/`_rqunpack` and `_rqesc`, then exact128B footer encoder/independent stock-list oracle. All6 full BF16 matrices roundtrip every original bit. All eligible stage footers reproduce their complete stock XOR maps. CPU and CUDA reference generators have different sequences: these are actual CPU reference matrices, NOT an assertion about cached judge/H100 tensors.

|seed/layer|global128row-k64 stages|escapes|max affected lanes|max registers/lane|overflow stages|whole layer eligible|
|---|---:|---:|---:|---:|---:|---|
|5151 pooler|4096|4668|6|2|0|yes|
|5151 qkv_gate|2304|2605|6|2|0|yes|
|5151 unpooler|4096|3240|6|2|0|yes|
|1234 pooler|4096|4645|6|2|0|yes|
|1234 qkv_gate|2304|2658|9|2|1|NO|
|1234 unpooler|4096|3179|6|2|0|yes|

Overflow witness: seed1234 qkv nb17/chunk41 has9 distinct affected(warp,lane) records, all with1 register. Eight records cannot encode it. Required exact fallback: if ANY stage exceeds8 lanes, ANY lane needs>2distinct registers, any duplicate/invalid escape occurs, or pack/oracle check fails, disable footer selection for that ENTIRE layer and keep its original frozen s4s5z path/stock escaped pack. Never truncate, drop an escape, partially zero a footer, or run a mixed-policy stage. Capacity expansion is outside this fixed policy.

The inherited pureCPU model also passes24,576 single-escape cases,4mixed cases,3overflow/duplicate negative cases and20KS ownership cases. One actual overflow now confirms that fallback is necessary on ordinary held-out Gaussian weights.

## Static cost, fixed m128 / s4s5z

NT128,ST5,GS1,FL2; KS pooler4/qkv7/unpooler2. No geometry changes.

- W payload12288 ->12416B/stage (+1.04167%); packed W remains byte-identical beforefooter. Same1-D `cp.async.bulk` request copies footer with payload. X alone uses a2-D tensor map. One request perstage remains, but128extraB =four32B sectors perstage.
- Dynamic SMEM144520 ->145160B (+640B). W ring slot/footer remain128B aligned; bars remain8B aligned; X descriptors/swizzle unchanged. Fits existing one-CTA/SM shared capacity. No exact register/spill result is claimed because NO CUDA prototype was admitted.
- Shipped parent has only W+X producers and full count2, W-empty/x-empty count8. There is NO runtime patch producer/thirdarrival in this parent to remove. Existing W/X handshakes are required. Removing the failed P4 table's extra barrier is not a gain relative to shipped s4s5z.
- Stock escape-list capacity is32 in all6 matrices, max populated list8–13. Every math warp issues just ONE128B lane-spread prefix at entry; NO mid-loop global list-window refills occur. Prefix volume131072B pooler/unpooler,129024B qkv. The list pointer is already affine; no dependent offset fetch exists.
- Footer mask/base/records must be captured into registers BEFORE `wempty` arrival. Otherwise the W producer may refill/reuse the shared slot before decoding reads the footer. This lifetime constraint is absent from a naive "read footer during decode" sketch.
- Direct sparse read adds mask-LDS -> active-lane/rank -> record-LDS -> XOR dependence before slot release. Alternative whole-footer warp-window read avoids dependent LDS but loads128B per math warp (8x globalfooter bytes inshared, ~4MiB pooler/unpooler) and adds mask/base/record SHFLs+POPC. Empty stages add an LDS/header operation where the old sentinel walk uses cached registers. These are real added instructions/transactions, not free sparse fixes.
- Direct form needs roughly5–7 temporary32bit values (mask,base,rank/index,two words/pointer) while removing ec/nx/nxs/ei and list pointer. Exact net liveness requires compilation; no basis to promise168registers/no spills merely from footer size.

## ROI decision

Clean P4 source evidence is full streamed INVALID-noescape controls, NOT resident compute-only kernels:

|layer|observed graph removal|observed CUPTI removal|gross footer bytes H100 /H200 ideal us|optimistic NETbytes H100 /H200 ideal us|
|---|---:|---:|---:|---:|
|pooler|0.672|0.384|0.187/0.138|0.140/0.103|
|qkv_gate|0.800|0.223|0.105/0.078|0.059/0.044|
|unpooler|unmeasured|unmeasured|0.187/0.138|0.140/0.103|

Netbytes gives the proposal maximal credit for deleting ALL initial prefix bytes as HBM misses, which may over-credit cached prefix loads. At2.8TB/s H100, subtracting only those netbytes from the clean CUPTI removal leaves0.244us pooler /0.164us qkv BEFORE any metadata/POPC/SHFL/LDS cost. Even keeping those same measured removals on3.8TB/s H200 gives0.281/0.179us. These are optimistic budgeting comparisons, not transferable hard hardware upper bounds.

The graph controls suggest a larger hypothetical budget (0.532/0.741us afternetbytes), but disagree materially with direct kernel timings and contain deliberately missing corrections. They are not proof this metadata layout can recover that time. Unpooler has fewer escapes and no measured noescape removal here. Eliminating the initial prefix could alter fixed startup, but its isolated latency is unmeasured; the source already requests it before the W flood. No additional measured0.3us fixed term can be identified.

The~2.16us resident-noescape difference cannot be added to streamed gains: those rungs remove TMA/barrier/stream overlap. H200 may expose more SM work, but that does not establish the present footer's net behavior or clear the required same-run bar. Likewise the failed runtime-producer table's1.5us startup penalty is NOT a shipped cost to remove.

Therefore metadata eligibility alone does not justify another GPU prototype. No tuning, capacity/geometry variant or speculative measured-speedup claim is kept alive. Reopening requires new independent counterevidence that changes this cost decision, not the old resident number or graph control repeated as an attainable gain.

## Reproduction / evidence

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 proj_rsi/candidates/rq_footer/feasibility.py
```

Outputs `feasibility.json`; full source/evidence hashes in `SOURCE_MANIFEST.json`. No GPU APIs are invoked by this script. Parent import only defines helpers; no extension builder/prepare is called. Source hash guard rejects any changed parent. CPU footer objects are feasibility checks and are never handed to a GPU kernel.
