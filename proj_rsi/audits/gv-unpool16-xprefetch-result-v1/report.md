# Unpooler M16 X-prefetch: actual fixed-cell gain

Recommend **DONE / VALID_GAIN** for `research.gv.unpool16.xprefetch.fixed.01`. Positive means control minus prefetch; this is a same-window H100 implementation comparison against exact selected501d template, not a fresh policy/stack or official H200 gain.

| Estimate | Control−new, us |
|---|---:|
|24 paired median|+0.576000661|
|24 paired mean|+0.522666766|
|Control-first12 paired median|+0.543999486|
|New-first12 paired median|+0.783999451|
|F1 control-before−new|−0.127999112|
|F1 control-after−new|+0.544000417|
|F1 endpoint-average−new|+0.208000652|

Paired21 wins/0 ties/3 losses; individual deltas−0.992000103..+1.343999058us retained. Median control/new ratio1.026588846. F1 medians22.592000663/22.719999775/23.264000192us, control drift+0.671999529us. Same-seed F1 endpoint medians−0.128000975/+0.544000417us. F1 and paired use different X banks; blocked drift and negative early endpoint remain explicit. Do not combine estimators or erase three losing pairs.

All11 collected artifact hashes and18 launch source pins pass. ActualCUDA equals frozen pilot+bridge; actualSO cuobjdump PCs/predicates/128-bit encodings equal recorded actualSASS and frozen thinSASS for both functions. Current control matches retained actual501d unpooler binary. Thus intended motion is realized: unit1 X loads0x4a0–0x4d0 precede first decode PRMT0x580, versus control0xab0–0xae0 after first repair; candidate after-repair loads now target unit2. No claim of a measured scoreboard reduction: this run has no counters.

Actual queried resources: control110 versus candidate126 registers; local0, user-static0, dynamic5120B, maxthreads128, device-reserved1024B, active-CTA capacity4 for both. This is queried capacity, not measured achieved residency. Correction to earlier CPU brief's shared-memory description: cuobjdump's1024B SHARED includes reserved accounting here; CUDA function attributes correctly separate user-static0/device-reserved1024. The frozen driver recorded raw fields before checking four-CTA capacity and did not assert user-static1024; **no execution/setup failure or source repair occurred**. Original CPU brief remains preserved, this result records the clarification.

Validity:91 raw journal rows equal receipt (81 clean/10 warm); prospective Wseed5344, shared original packed W/escape bank, no repack;6 initial task+FP64 checks,16 basis columns per arm,4 own terminals with byte-exact current X before replay/mutation,10 retained graph mutations with NaN overwrite, immutable W/payload/escape/input and S1 full scratch/counter backing guards pass. Thin ABI passes the current capture stream, retains strong tensor owners and retires them only after graphs drain/reset. Internal allocated/reserved0 and BLAS map empty; external exit0, GPUmemory0/no PIDs/container/guards. Passing these focused checks does not prove absence of every possible bug.

Cost: nativecompile/link2.180052s, taskweight/pack0.268624s, whole run5.819773s. No Torch C++ compile, full prepare or profiler. Host submission median22.750us, enqueue→sync103.315us; F1 start→lastsync6.571774ms. These host spans are not GPU kernel time and cannot establish queue-underflow/clock causes. No device-clock telemetry collected.

Cause confidence: **PROVEN** correct intended load motion, unchanged mathematical/repair order and current output contract, equal queried capacity, primary paired gain including both order groups. **UNKNOWN** exact cycle-level mechanism and H200/policy transfer. Earlier X issue is consistent with increased overlap; no measured dependency-wait attribution. Prior lane-base tie/quarter losses remain distinct counterevidence against assuming earlier work always wins, not reasons to discard this positive version.

Next: preserve the candidate and explicitly stack only against the actual official selected source after the17d44d/40d856e085274f51 receipt; check whether this tuple remains selected or expose a legitimate optional contender. Independent review applies to combined routing/source changes. Root can price a CHECK without waiting for scored quota reset; no automatic unchanged rerun/profiler gate. Actual official score forecast UNKNOWN; the prior conditional0.2us→2.1518scoreus scenario is not this result's official gain.

Loop: profile → measure → hypothesis → optimize → test → validate → stack → submit → learn → reprioritize → repeat; idle≤10s. [Contract](../../ITERATION.md).
