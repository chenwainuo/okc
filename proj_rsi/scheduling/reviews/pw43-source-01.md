# PW43 source review

**GO_SOURCE_CPU_ONLY_RUNTIME_UNPROVEN**.

Exact candidate source: `fd8e9f73c74a03cb7b979094b6c24997a27c217cc23989d0af82ae03847f2d81`.
Exact manifest: `f05e27c4136b4d9771d16bd015df18e2f92c0f1c55f61b7e98bcc64cf70fb67f`.

Worker08 independently checked all frozen hashes, complete-module AST inverse, unchanged GPU/prepare/serving/timer code and precise BF16 M256 scope. Reexecuted23 frozen author cases with writes captured in memory; added four independent estimator cases and six actual old/new non-target search equivalence cases. Tiny1e-12 positive deltas remain selectable; incomplete/invalid/loss observations are retained and declined. No1%/absolute target threshold is introduced.

Full audit and reproduction are `proj_rsi/audits/pw43-source-independent.{md,json,py}`. Scope limits: the original timer exposes medians rather than underlying replay samples; no blocking-operation preemption or hard prepare-wall-time proof; postdeadline host configuration queries remain; exact non-target decision logic does not guarantee identical later picks under shared timing. These are explicit runtime measurement obligations, not inferred evidence of failure or gain.

GO is for a separately reviewed bounded actual-policy experiment only. Require unchanged PW38 versus PW43 full fresh prepares, actual configuration and decision logs, repeated selection/prepare timing, disjoint balanced raw/F1 evaluation, all four target outputs and26 other timed shapes' correctness/counter/state/graph/terminal/cleanup gates. PW39 mixed local signs and official regression remain negative transfer evidence. No official or production admission follows this source review. Root owns GPU execution and scheduler finalization. No GPU/API/prepare/queue/frozen-file operation occurred.
