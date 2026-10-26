# Operand reverse v2 / direct native-A loader independent preflight

`decision=VALID_VERSION_REGRESSION_DIAGNOSE_NEXT_LIMIT; GPU=TERMINAL_EXIT0; source=FROZEN; v1=PRESERVED; promotion=NO; failure_policy=DIAGNOSE_AND_ITERATE`

Reviewer changed no candidate code/manifest, launched no GPU/remote job, and owns only this audit. Root owns execution, resource cleanup and result/queue decisions. This is a concrete successor to the diagnosed v1 loader cost, not an unchanged retry or a full-submission change.

## Exact reviewed package

|artifact|SHA256|
|---|---|
|v2 manifest|3a64fc10b1b649240ee96f22d2630386f90481ecbd7af77dcab0722b90fa6deb|
|generated Torch extension source|97664bdee7593d46e0493b9599062c37af150262d597002a722016f060548faf|
|generated device source|dcbdc2bbc4feccf1e2b4115f86261fe0934eea7c4155019a2564ea044676ba36|
|kernel body|443762da78ce3b72927887c5af49273cafa5c8d58c93191fc5118ffdc0ef297a|
|lab|63aaf365d223139a3b95c0335f9488dc17a38891dadf23ec5d7e4d61842a63ee|
|preflight result|35a523a10fe79a16cd5d45c2c61b13839983b548a573256a018e5a48fb8842d9|
|SASS|4a4bbb82f269ca2b41004a9d806698dfaad269c6e24f4ec82e97a0eed26bd2ba|

Reviewer independently verified all10 manifest file hashes, regenerated source/device text equality without writing files, frozen v1 six source/proof/lab files against its manifest, and actual v1 generated sourceca92a4fb…85f8. Baseline pw36 source is pinned2a067059…bccf. The v2 prototype uses hash-pinned rs32 pack/decoder/host helpers. Both Torch extensions build before baseline import starts its prepare clock. Full actual Torch-host build/runtime attrs remain GPU gates; the local device-only cubin does not certify those.

## Mechanism and lifetime

One shape: unpooler M128/N8192/K4096,128CTAs×384threads,64output channels/CTA, KS1/fullK/plain output. Math warps directly load the native WGMMA-A fragments through16bitwise scalar uint32 `.ca` loads/thread/stage. The source map `row=wg*64+w*16+g+8*(i%2)`, `k=s*64+16*kstep+8*(i/2)+2*t` independently matches the v1 four-lane transpose output. No BF16 conversion or lossy codec is introduced. Reviewer independently ran X524288index/all65536BF16-payload patterns, output8192scalar-bijection, and A4 lifetime/negative-control functions, plus source identity of the producer and complete output epilogue.

Four distinct A arrays begin with stages0..3. Issue/commit stage s, wait_group1, then pin/release retired stage s−1 and refill its slot with s+3. `(s+3)%4==(s−1)%4`; current A is never overwritten while its newest group may remain asynchronous. For s0 no retirement/refill occurs. At final s63, wait_group0 retires the last group, all4 A arrays and D are pinned before output reuse. Compiler fences use volatile read/write register constraints and memory clobbers. CPU negative control catches60illegal current-slot refills from naively copying v1's refill schedule. The source order also preserves group completion before shared-B empty arrival.

Producer W6/B8 decoding/escape repair and the full output epilogue are byte-identical to v1 except removal of dynamic setmaxnreg. Its128producer-thread fence/bar.sync precedes the full publication; each decoded-B slot gets8math-warp empty arrivals only after its group retires. Producer waits for empty before wrapping slot8. W-ring next bulk load starts after the sole producer finishes using the prior slot, just as frozen v1. Math and producer branch by uniform warp roles; aligned WGMMA/barriers remain convergent within their required groups. Final math barrier follows both WGs' final completion before decoded-B ring memory becomes the output slab. These are source lifetime checks, not an arbitrary GPU-interleaving proof; runtime full-output tests remain required.

## Resources and interpretation limits

Archived actual sm90a compile:162registers,0stack/spill/local,4barriers, noC75warning. SASS contains16HGMMA/4WARPGROUP.ARRIVE/oneUTMASTG/noX-TMA, noUSETMAXREG and noSHFL.BFLY. Four static inherited producer SHFL instructions remain; “zero shuffle” applies only to the removed X transpose. Every output D scalar is observed, and volatile global loads feed the fully used A operands, preventing a partial-work/DCE comparison.

Uniform static allocation removes both prior64/216dynamic pool operations, avoiding their possible pool request deadlock after static register reduction. Logical162×384=62208registers; even168/thread rounding totals64512<65536. Actual CUDA runtime registers/local bytes and opt-in-before-occupancy>=1 are mandatory, with103600B shared memory; no reliance on nominal sums substitutes for those runtime checks.

The old X transpose contributes4,194,304fixed warp SHFL/SEL instructions. Direct loads add786432scalar-load warp instructions relative to the old vector load count. X logical traffic stays128MiB/launch; each warp-stage touches128sector instances but only64unique32B sectors. `.ca` is the one fixed policy permitting half-sector reuse; actual cache hits/transport/issue pressure require measurement. Removing instruction count is not a speedup guarantee. W shared-byte/producer geometry did not change, so a result cannot be claimed as proof that all prior shared-port/producer bottlenecks disappeared.

## Frozen runtime validation and measurement

Three paths: actual freshly prepared `kernel_fn`, frozen v1 and v2. Shared immutable X/W/packed/escape data, distinct candidate counters and all3retained output buffers. Baseline validator copies only outside timed capture. Every path must pass all4096one-hot columns/FP64 and normal/outlier task oracles with immutable input. Actual timed baseline output is kept alive independently of the closure hold; all3real outputs are disjoint. After timing, actual graphs are NaN-poisoned and checked on original/zero/fresh-normal/fresh-outlier/all-probe X,15complete-output checks, then original X is restored. Final W/packed/escape exact comparisons and all candidate/incumbent counters0 are mandatory. No silent fallback after launch/API error.

F1 uses21cold replays/series, before/after actual-incumbent AND v1 medians, taking the faster median for each comparator. Paired timing uses21rotated three-way rounds. Every replay has a256MiB flush outside its event region;5discarded cold replays/path. Timed capture of actual incumbent has no extra copy. Report raw deltas against both comparators; >=.5us vs actual incumbent in BOTH conservative F1 and paired median is only local survival pending independent terminal review. A speedup against v1 alone is useful mechanism evidence, not an incumbent improvement.

Profiles occur after timing and after actual graph mutation checks; candidate v1/v2 must each be exactly one actual kernel with distinct `rq<...>`/`rq_v2<...>` names. Baseline actual kernel metadata/profile must be interpreted as selected in this run, without attributing a cross-run prior winner. All real graphs synchronize/reset in finally. Timing/API/numerical failure invalidates version timings; any local winner still requires integration and official validation. A nonwin informs the next bottleneck/hypothesis and does not close the whole architecture family.

Root-only reviewed command, after the prior named job/container drains:

```bash
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v2/lab.py \
  --json /okc/logs/operand-reverse-v201.json
```

GO for one normal-capability named exclusive trial with inner240s/outer300s bounds and existing host/container lock/cleanup guard. No reviewer GPU launch, hidden variant or automatic profiling action. Await raw JSON/log, runtime build/attrs, all validation/counter/graph identities, exact gate arithmetic and clean terminal ownership before any promotion claim.

## Independent terminal evidence: correct v2, slower delivery policy

Actual guard366966/container`okc-proj-operand-reverse-v201`, launched23:09:22Z and terminal2026-10-26T23:11:13Z, exit0. Reviewer independently inspected the remote hostlog/JSON and verified guard/container absent by read-only SSH. Raw remote prefix`/root/okc/logs/operand-reverse-v201`; root owns local archive copy. JSON13174B SHA256`1a111196dba2860138ca04fc6cb767bce8054f9ba7363c0e46a49f801a15c84a`; hostlog13693B SHA256`c3faf801543e9e1712bb487cb49393ab9de2154d7d63bd7d7d9aebb60da5f329`. Status`V2_LOADER_DIAGNOSIS_REQUIRED`, frozen source hashes match above, no Traceback/invalid/CUDA-error marker. Independent CPU parsing recomputed every F1 median/conservative baseline, all21paired differences and win counts from the actual remote JSON.

Runtime full Torch build/resource gates pass: v1 regs168/local0, v2 regs162/local0,103600Bshared each; both maxDynamic49152→103600 after opt-in, occupancy1. Actual incumbent chooses`s4s5hz`/KS2 and profiles one`rq<128,5,1,0,0,0,3>`. V1 profiles one`rq<128,8,1,0,0,0,20>`, v2 one`rq_v2<128,8,1,0,0,0,20>`; profiles occur outside admitted timing and are not interchangeable timing samples.

|same-run comparator|before F1 median us|after F1 median us|conservative us|v2 F1 gain us|v2 paired gain us|v2 wins/ties/losses|
|---|---:|---:|---:|---:|---:|---:|
|actual incumbent|24.256000295|24.224000052|24.224000052|−32.799998298|−33.151999116|0/0/21|
|frozen v1|47.359999269|47.231998295|47.231998295|−9.792000055|−9.567998350|0/0/21|

V2 F1 median57.023998350us, approximately20.7%slower than conservative v1. Every v1/v2/incumbent4096one-hot/FP64/normal/outlier check passes; supplementary candidate original/zero/outlier replay checks pass. All15actual timed-graph poison/mutation/oracle/input checks pass. Full W/packed/escape bytes stay exact, all candidate/incumbent counters reset, and all3actual graph execs reset in finally. This is a valid version regression, not a correctness/resource invalidation.

**Interpretation / next falsifier:** removing4.194Mtranspose warp instructions while increasing scalar-LDG count/sector instances did not help. A simple instruction-count-only explanation is contradicted. Load issue, sector reuse/cache locality and producer/math LSU contention changed together; timing alone cannot assign the regression to one. It does not show decoded-W architecture universally unprofitable. Preserve v1/v2 and either diagnose this changed loader with counters that decide a concrete next design, recover vector delivery with cheaper fragment conversion, or review a material shared-A/SS delivery mechanism. Any successor needs its own source/lifetime proof, bounded run and actual-incumbent comparison; no promotion/official slot for this v2. No additional old-v1 GPU profile is required to preserve its existing evidence.
