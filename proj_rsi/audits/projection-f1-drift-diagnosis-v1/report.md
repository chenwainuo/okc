# Projection F1 drift: source and archived evidence

The large control-bracket movements are real, but their clock/thermal cause is **UNKNOWN**. No clock, pstate, power, temperature or throttle telemetry was found in the three archived JSON/log collections. No clocks were queried or changed by this CPU audit. Existing dispatch cleanup probes query only compute PIDs and memory; those are not performance-state evidence.

## Observed

| Attempt | Same-arm last−first block median | Runtime / compile+load |
|---|---:|---:|
| lane-base M16 |+1.216001809us|45.8790 /42.6144s|
| quarter-stream M8 |+1.152001321us|45.1801 /41.8381s|
| robust6203 pre/post |+.032000244/−.095999800us|3.6531s /native-load.00218s|
| robust6257 pre/post |−.159999356/−.048000365us|same job|
| robust6287 pre/post |−.144000165/+.128000975us|same job|

These are schedule-associated contrasts, not isolated causal experiments: robust reuses a different binary, different weights and a different schedule/check placement. Compile accounts for~42s of the45s jobs, so calling the bracket difference a45s timed drift is unsupported. Raw timing rows contain no per-sample walltimestamps; the actual inter-block wallspan cannot be recovered. Lane-base thirdcontrolblock is elevated acrossall11samples; quarter thirdcontrolblock varies21.024–22.720us rather than a uniform constant. No claim of linear drift follows.

Both oldstyle F1 controlblocks repeat identicalseed91100..91110 from the same storedfixtureobjects. Thus a different Xseedbank **cannot explain their own before/after difference**. F1versuspaired compares differentXbanks(91100..10 vs91200..23), plus differentorders/times; that remains a distinct confound. New robust shares all8Xfixtures acrossstyles/weights, preserves bothbracketorientations, but does not isolate elapsedtime/thermal effects. Difference-of-medians and median-of-differences also differ nonlinearly; preserve both.

## Source-proven measurement structure

Lane driver lines151–168 andquarter driver169–181 enqueuecopy→256MiBfloatread-sumflush→new CUDAevents→start.record→singlegraph.replay→end.record→devicesynchronize→elapsed_time. Flush is outside the eventinterval, but its cache/power/workloadstateeffects persist. Each rowopens/writes/flushes/closes raw.jsonl **after** devicesynchronize. No perrow fsync; an OSflushstall is not demonstrated. The same journaling pattern applies to botharms/styles, so no arm-specific I/Ocost is proven. Timingevents are constructed eachsample; noeventreuse and nohostsubmissiontimestamps.

CUDAevent boundaries exclude work beforestart, but this source does not ensure theGPUhas pendingwork throughout start→replay→end. Hostgraphsubmission/eventrecord gaps **can** enter elapsedtime if thedevice catches up. Existence of the host-side sequence is proven; actualexposedgapduration and contribution to~1us changes are UNKNOWN. A precedingflush may hide thosegaps, but its completion timing is unrecorded. There is no CPUclock/hostpower/deviceclock evidence sufficient to distinguish these possibilities.

Initialnumeric/task/FP64/basis checks andgraphcapture occur beforewarmups. Warmups are fivecontrol thenfivenew, notcontinuoussteady-state saturation. In lane/quarter the timedown-terminal FP64/reference/integrity checks run **after all33F1rows at i43**, thenpairedsampling starts. Those checks cannot explain the drift **between** thetwoF1controlblocks because neither runs between them. They can alter the state beforepairedsampling, a different transition. Journaling runs betweenallrows; save() followsphase/validation/terminal boundaries, notbetweenF1blocks. Robust insteadchecks theterminalarm plusfullintegrity aftereveryblockedbank andaftereachinterleaved8bank/order; this is a real schedule difference, not a failingvaliditycheck.

## Disposition and smallest discriminator

No setup/correctness failure is observed. Do not discard valid pairedsigns or positiveF1endpoints, and do not promote an averageF1positive as an unqualifiedstablewin when its endpoints conflict. No new admissiongate or standalone unchangedtimingrerun is warranted.

On the **next already-useful projection candidate screen**, piggyback a tiny diagnostic change: preallocate/reuse thetwo timingevents outside samples; record monotonic_ns immediatelybeforestartrecord/afterendrecord/aftersync andbefore/afterjournalwrite, plusblockboundarytimestamps. Keep the same copy/flush/kernel/order/fixtures/oracles. This costs hosttimestampcalls andbytes in the existingjournal, with no newkernellaunch. It distinguishes submission/I/Ogap association and provides the missing timespan; it does **not** itself measure deviceclocks or establish thermalcausality. Preserve the oldtimingmethod provenance when comparing runs.

If an existing hostmonitor can supply read-only GPUclock/pstate/temp samples cheaply, retain onlyblock-boundarysamples outside eventtiming; do not add a perlaunch nvidia-smi process. A clockchange correlated withslowerblocks supports an association, not solecause. If hosttimestamps show large exposedqueue gaps while clocktelemetry is stable, a subsequent boundedmeasurement-mechanism comparison (prequeuedGPUevent/graphinterval versuscurrenthostsequence ontheexactSO) is the smallest causaltest. That is conditional, not an immediateGPUrequest.

Allprior DONEevidence/proposals are frozen unchanged. Scope muse_bf16_proj only. Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit→learn→reprioritize. [Contract](../../ITERATION.md).
