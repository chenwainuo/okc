# Coordinator startup instructions

Current scope (user correction 2026-10-27): optimize **muse_bf16_proj only**,
target official score below 3000 microseconds. Do not optimize serving or MoE,
launch their experiments, or treat their local gains as projection progress.
Retain their historical evidence; active ROI work and GPU admissions are projection only.

Before continuing optimization, read [proj_rsi/ITERATION.md](proj_rsi/ITERATION.md),
especially its Continuous coordinator loop section. This records the user's
explicit persistent instructions and applies to takeovers and compacted sessions.

Every user-facing answer/update must restate the optimization loop briefly and
link that contract. During active authorized work, status questions do not pause
coordination; deliberate idle waits are capped at 10 seconds. Report actual
activity accurately. Preserve these requirements in handovers and summaries.

Use the existing ROI queue and exclusive GPU worker; implementation agents do
CPU work and return source-pinned proposals to the root coordinator. Do not
launch competing GPU processes or rewrite frozen experiment artifacts.

A source-pinned proposal, GO review, or DONE evidence becomes frozen when sent
to the coordinator. Later wording corrections, analysis, or additional evidence
go in separately named supplements; do not overwrite a delivered hash, even
before receiving the coordinator's admission or completion acknowledgment.
