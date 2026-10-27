# Independent DP versus G8 counter review

Worker07: LOCAL CPU/read-only artifacts only while native-X393324 runs. No
remote import/export, GPU context, benchmark, profile or frozen-artifact edit.
Verdict: **GO for the reported counter associations and next causal diagnosis;
HOLD any claim of intrinsic fence/MOV cost, uniquely identified X cause, or
performance gain/loss derived from this capture.**

Reproduced `tools/group8_compare_analysis.py` into
`/tmp/sol07-group8-analysis.json`; parsed result equals recorded analysis.json
exactly. Analysis script SHA256
`c793dda022c0f5f0f88fd49dffa5e621fdc03621659e2c140c1731c6aaaa2aec`;
recorded driver SHA256
`245b78898a16c2087b00924cd809cd33fb74bfd7c19c9012bd7d040005485702`
matches current frozen source. All recorded CSV/log/JIT/driver artifact hashes
verify. Session records H10080GB, kernel replay, graph node profiling,
cache-control all, clock-control none, four selected nodes. Host log confirms
DP pack/main then G8 pack/main,34 passes each, exit0 at02:45:50UTC.

Driver JSON records both selected outputs passing relative L2/worst-row/probe
oracles (1.421e-4/2.633e-4), full1MiB XP byte identity, W/pack/escape/X
immutability, counter zero and overwritten poison, followed by two graph resets
after drain. These are valid diagnostic selected-output checks, not this
profile independently repeating a full4096basis validation. Root separately
reports guard/container absence and GPU0 before the next job; worker07 does
not claim to independently re-observe cleanup during the active next trial.

## Binding, roles and predicate accounting

All four sections (both packers and mains) have exact PC coverage and
instruction/predicate/operand equality to their actual loaded JIT SASS after
relocation of absolute in-module addresses. Whitespace and `.reuse` are the
only textual normalization. Main source rows DP848 and G8880 complete; no
subset/top-PC-only binding. Local independent negative controls replace an
instruction with NOP and shift one address by8 in EACH of all four sections;
all8 corruptions reject. Source/manifest identities match recorded DP/G8 pins.
Actual JIT artifacts are retained, not a guessed source-line attribution.

Recomputed role CFGs from actual JIT and exact loop bounds match archived maps.
DP role split0xa90 `@P1 BRA 0x1de0`; G8 split0xa70 `@P1 BRA 0x1e90`.
Fallthrough is math warps0..7, target producer warps8..15. Reachability includes
outlined polling paths; math/producer disjoint and common/setup/unreachable
remainder retained. Partitions and phases conserve all rows and samples.

Both math and producer exceed preregistered200 samples: DP1719/1548,
G81840/1639; common100/125 retained. Leading fence/Bfull/Bempty phases exceed30.
G8 MOV466 exceeds30; DP MOV1 does not. Therefore describe a new G8 hotspot,
without treating DP MOV as a statistically useful baseline ratio.

For every four math Bfull sites and both producer wait classes, slow-test lane
instructions equal failed-entry plus failed-retry predicated-on lane
instructions exactly (all flow residuals0). Failure means @!P branch after
TRYWAIT, verified with fast/slow matching predicates and return targets.
Math Bfull failed lane totals / tested lane totals:
DP824608/2097152=39.3204%; G81180288/2097152=56.2805%.
These are failed branch lane fractions after potentially blocking TRYWAIT,
not initial-readiness probabilities or exact warp retry counts. No division
by32 establishes uniform predicates. Producer Wfull has0 failed entry lanes
while still carrying79/90 samples: another reason not to equate branch failure
with total wait exposure.

## What changed in observed exposure

|Observable|DP|G8|
|---|---:|---:|
|Math fence samples / math total|770/1719 (44.7935%)|426/1840 (23.1522%)|
|Fence long-scoreboard samples|767|422|
|Loop-head `MOV R0,0x400` samples|1 at0xbc0|466 at0xba0|
|MOV long-scoreboard samples|0|465|
|Math Bfull samples|104/1719 (6.0500%)|156/1840 (8.4783%)|
|Producer Bempty samples|543/1548 (35.0775%)|628/1639 (38.3160%)|
|Producer Bempty long-scoreboard samples|532|624|
|Main global load requests|263168|263168|
|Main global load sectors|4198400|4198400|

G8 source groups two original K64 stages into each async group, preserving K
order; actual fence/wait1 static sites4→2 and fence warp-instruction counts
65536→32768 match that grouping. HGMMA warp-instruction counts remain262144
and predicated-on thread counts8388608 in both. Both mains122registers,
512threads,103600B dynamic shared,0local,1CTA queried occupancy after opt-in;
packers22registers/0local. The grouping does not reduce the measured main
logical global-load requests/sectors. Sector count is a count of32B sectors,
not DRAM bytes or HBM traffic; requests metric has blank exported unit and is
reported as its explicit request-count metric. No µs/cycle conversion made.

The fence reduction does not demonstrate removed pipeline waiting. In G8,
465 long-scoreboard samples now associate with loop scalar setup MOV before
Bfull/fence. That MOV writes a scalar address/setup register and is not an A
vector load. Samples identify the exposed dependency endpoint, not MOV's
intrinsic execution latency or the unique producer. No undocumented SASS
control/wait-mask decoding is used. Descriptively fence+MOV totals771/1719
(44.8517%) versus892/1840 (48.4783%); this is not conservation of elapsed time
or an argument that one site directly transferred those particular samples.

NVIDIA documents that long scoreboard is a dependency on an L1TEX operation,
and that samples attach to a waiting consumer rather than the producer.
That supports an outstanding memory-dependency hypothesis, with source context
needed to identify its origin; it does not identify packed X uniquely.
[Primary profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

All-samples versus not-issued samples are separate collected quantities;
not-issued need not be a subset in these exports/passes. Exported reason sums
leave residual201 DP /187 G8, retained explicitly rather than set to zero.
Role/phase fractions use total all-samples with each variant's own denominator.
Raw/source execution totals may differ across replay passes and variable
polling; neither is rewritten to force agreement. Historical DP denominators
remain separate. The six unavailable CTC metrics in host log do not affect
these requested LSU/source/sample metrics.

## Smallest useful next question

Evidence supports two nonexclusive hypotheses: less X lookahead or changed
register scheduling exposes an outstanding memory dependency at the G8 loop
head; paired B lifetime/retirement increases producer backpressure, visible at
Bempty and post-TRYWAIT Bfull failures. The source change couples grouping,
lookahead, physical scheduling and paired release, so neither hypothesis is
isolated. Keep native-X experiment's own evidence separate. A future retained
version should vary one of lookahead/release granularity while holding the
other fixed if feasible, rerun correctness/lifetime/resource gates, and compare
whole fresh-pack pipelines with ordinary timing. This cold per-node profile
alone supplies no performance forecast, causal verdict, or family rejection.
