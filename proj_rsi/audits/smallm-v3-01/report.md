# Smallm cursor v3 all-five local result

The attempt is a valid local comparison of the actual freshly prepared policies.
It does not establish an incremental cursor gain. There are small measured wins
against some controls, ties and regressions; the strongest matched selected-path
comparison, pooler32 against rs32, regresses. This is not an official H200 result
and does not measure the PW37/PW36+cursor/EHint stack.

Job `smallm.cursor.v3.all5.01`, guard PID398864, completed 2026-10-27 04:11:07 UTC
with exit0. The collected cleanup observation proves no guard PID, named or other
running container, compute PID or GPU memory remains. Frozen v3 result status is
`VALID_FOCUSED_DEVELOPMENT_ONLY_FULL_SCOPE_REQUIRED`; despite that inherited
status wording, its scope and records cover all five requested shapes. Root
validator owns disposition; this audit does not edit the queue or admit promotion.

Signed deltas below are **control minus increment** in microseconds: positive
means increment faster, negative means increment slower. Paired delta is median
of the 30 per-seed paired differences. Ratio is median(control/increment), not
ratio of marginal medians. F1 delta uses the lower of control-before/after
medians minus increment median, preserving the conservative bracket rule.

| Shape | Paired ratio pw36 / rs32 | Paired delta pw36 / rs32, us | F1 delta pw36 / rs32, us |
|---|---:|---:|---:|
| qkv32 | 0.997977 / 1.000000 | -0.032 / 0.000 | +0.064 / +0.128 |
| qkv16 | 1.002181 / 1.002169 | +0.032 / +0.032 | +0.032 / +0.032 |
| pooler32 | 1.004370 / 0.993492 | +0.096 / -0.144 | +0.064 / -0.096 |
| unpooler32 | 0.983485 / 0.984240 | -0.384 / -0.352 | -0.384 / -0.256 |
| o32 | 1.000000 / 0.988546 | 0.000 / -0.112 | -0.064 / -0.160 |

| Shape | Paired marginal median pw36, us | rs32, us | increment, us |
|---|---:|---:|---:|
| qkv32 | 15.296 | 15.328 | 15.328 |
| qkv16 | 14.832 | 14.848 | 14.768 |
| pooler32 | 22.256 | 22.000 | 22.144 |
| unpooler32 | 21.952 | 22.224 | 22.640 |
| o32 | 9.888 | 9.792 | 9.856 |

These marginal medians differ from medians of paired differences. For example,
o32 pw36 has a +0.032us difference of marginal medians, but paired median delta
is exactly0.000us; the paired result is the declared comparison statistic.
No official score is inferred from sums of these focused local measurements.

Actual picks explain what these numbers can establish:

| Shape | pw36 | rs32 | increment |
|---|---|---|---|
| qkv32 | t32s6hze, KS6 | t32s6hze, KS6 | t32s5hze, KS6 |
| qkv16 | GV15 | GV15 | GV15 |
| pooler32 | t32s5hz, KS4 | t32s5hze, KS4 | t32s5hze, KS4 |
| unpooler32 | t32s5hze, KS2 | t32s5hze, KS2 | t32s6hze, KS2 |
| o32 | GV47 | GV47 | GV46 |

qkv16 records a small +0.032us paired and F1 win against both controls, but all
three serve the unchanged GV15 kernel, so it supplies no cursor-effect evidence.
Its paired win/tie/loss counts are16/0/14 versus pw36 and15/2/13 versus rs32;
the central half of paired deltas crosses zero against both. Retain the tiny
measured win without presenting it as a reproducible optimization gain.

qkv32's F1 bracket favors increment by0.064/0.128us. Its balanced paired result
instead slightly favors pw36 by0.032us and ties rs32. Pairwise counts are
11/3/16 and14/4/12. The candidate selects ST5 while both controls select ST6;
the run establishes complete prepared-policy behavior, not isolated recurrence
latency. Both observations remain in the record.

pooler32 beats development pw36 by0.096us paired and0.064us F1. However pw36
selects the different non-EARLY hz body. Against the official-source rs32
comparator, both select t32s5hze KS4 and the candidate loses0.144us paired
(about0.651% by the paired ratio) and0.096us F1. It wins7 and loses23 pairs
against rs32; the paired-delta interquartile range is[-0.304,-0.032]us. This
is the clearest same-selected-body evidence here and it favors the parent.

unpooler32 regresses against both controls:0.384us versus pw36 and0.352us versus
rs32 paired, and0.384/0.256us F1. It wins4 and loses26 pairs versus pw36;
versus rs32 it wins3, ties2 and loses25. Increment selects ST6 while controls
select ST5, so this is a concrete regression of the freshly prepared policy,
with bake selection as a confound for attributing it solely to cursor code.

o32 ties pw36 paired but loses0.064us F1; it loses0.112us paired and0.160us F1
to rs32. None serves the modified RQ cursor: increment selects GV46 and controls
GV47. rs32 is the faster measured comparator; the path-selection difference
prevents attributing the result to the dormant cursor change.

The local best control is pw36 at qkv32/unpooler32 and rs32 at pooler32/o32;
qkv16's tiny increment advantage occurs on an unchanged path. No shape gives
consistent isolated cursor-positive evidence against the same selected rs32
body. Preserve the family and causal diagnosis under ITERATION.md. These results
support testing the concrete PW36+cursor stack only with its actual selections
and an unchanged matched comparator; they do not justify claiming its gain in
advance or discarding every possible cursor version.

The audit recomputed all values from raw.jsonl and required byte-for-byte equality
with result.raw. Per shape there are9 discarded warmups,55 F1 calls and90 paired
calls:154 calls each,770 total. It checked every label/mode/seed and exact F1
before/increment/after ordering. Each of the six label permutations occurs five
times per shape, yielding30 balanced paired rounds with seed20040..20069.
Every recorded duration is finite and positive. Both stored paired ratios and
all F1 medians/control ratios match independent recomputation.

All15 full-K basis/FP64 checks pass. Frozen _rqval enumerates every K column with
one-hot x8 inputs and requires bit-exact W responses; it then tests normal and
outlier rows against FP64 matmul at4e-3 tolerance. All75 graph mutation checks
pass: normal, outliers, zero, all-probes and fresh-normal for every shape/label,
with NaN-poisoned output and eligible workspace. The successful frozen control
flow also checks finite/current-input outputs, zero output for zero input,
input immutability and immutable weight/pack/escape views. Both F1 and paired
terminal oracles occur before any subsequent replay/mutation. There are30
label-specific strict terminal output oracles: last seed20011 for F1 and20069
for paired, all PASS.

The archive inventory's12 files all match their SHA256 entries. Every frozen
job file and independent review hash matches locally. The audit extracted only
literal RQCU/RQV/RQSBAKE data and pure _rqwg/_rqss/_rqsrc functions from each
pinned candidate AST, executed no candidate import, imported no torch, and
regenerated each archived CUDA source exactly. pw36 and rs32 have identical
rq32 source, recorded SO identity and SASS; increment has its distinct cursor
source/SO identity. All three actual SASS artifacts match the admitted selected
body opcode counts. Selected reference symbols retain168 registers, stack0,
local0; selected runtime RQ attributes record168/0 and expected dynamic shared
bytes83080(ST5) or99488(ST6). The first-math JIT gate precedes each full prepare.
This run resets only the documented prepare clock after gate compilation and
uses warmed local infrastructure; it is not an official cold-prepare feasibility
or official score measurement.

Counter semantics are also consistent with the narrow repair. The archive has73
counter-view prepare baselines, all int32 with full allocation identity/digest
and small values. All selected RQ counters start zero. qkv32/pooler32/unpooler32
serve RQ and require strict reset0 by physical storage identity, including
shared plan aliases. qkv16/o32 serve ordinary no-counter GV routes; the audit
independently extracted the base GVI tuples and confirmed the source's S1/no
counter classification. P1 and router monotonic counters remain dormant.

There are39 nonzero P1 baseline views (13 in each prepared module), plus three
nonzero rw_cnt baselines. Reapplying the old v2 all-cnt-zero condition to this
new actual preparation would first fail at pw36/p1cnt/92:128 nonzero words,
first values18864,4716 repeated. This directly reproduces the old harness
overconstraint on the current preparation. Exact first failing tensor from the
earlier invalid attempt remains UNKNOWN because that assertion stored no name.
No new result rewrites that old evidence.

Frozen v3 executes125 all-state invariant evaluations:5 before-shape checks,
15 after basis checks,75 graph checks and30 terminal checks. Its complete
successful control flow establishes that selected counters passed reset0 and
dormant full storage equaled prepare baselines at those checkpoints, alongside
immutable tensor checks. The counter source bindings match a fresh CPU check,
including dormant-nonzero positive cases and selected-nonzero/dormant/rw/offset
alias negative cases. No counters were zeroed by the harness.

Evidence limits remain explicit. The archive contains generated source, SASS and
resources for all three modules and their recorded .so hashes, but not the .so
binaries, so a separate local rehash of the actual libraries is unavailable.
Successful final per-counter digests/values are not individually archived; only
failure would save them. Consequently this audit verifies the executed,
source-bound successful assertions rather than independently recomputing final
tensor values. One H100 attempt, varying bake picks, and small effects with
mixed F1/paired signs do not establish reproducibility or H200 transfer.

Worker13 performed only local CPU reads/recomputation and wrote this new audit
directory. Candidate, frozen driver/manifest, queue and protocol were not edited.
Reproduce with `python3 proj_rsi/audits/smallm-v3-01/analysis.py`; structured output
is analysis.json, with per-pair dispersion, sign counts, resources, selected
policies and complete nonzero baseline metadata.
