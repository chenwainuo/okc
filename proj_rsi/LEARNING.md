# Experiment learning registry

A completed success, regression, mixed result or diagnosed setup failure should
produce a lesson before the next experiment is admitted. This registry is a
CPU-only review aid. It does not register work, alter queues, call an API, run a
GPU, promote a source or replace the root validity/protocol gates.

`learning.py` validates `learnings.json`, summarizes its latest lesson versions,
and retrieves lessons and concrete proposed tests relevant to an idea:

```bash
python3 -m proj_rsi.learning validate
python3 -m proj_rsi.learning summarize
python3 -m proj_rsi.learning query 'rq11 escape residency'
python3 -m proj_rsi.learning query 'native composition overlap' --hardware H100
python3 -m unittest proj_rsi.tests.test_learning
```

All CLI commands are read only. Validation fails on stale/missing evidence or
source bytes, unmatched JSON values/text anchors, empty outcomes, omitted
negative outcomes in mixed/regressing experiments, missing scope/limits,
unsupported transfer/promotion, unbound hypotheses, malformed version chains,
unknown fields and nonfinite benefit/cost values. Validation is structural and
checks fact anchors; a human still reviews whether a statement accurately
interprets its anchor. A digest is integrity checking, not authorship proof or a
cryptographic signature. Do not treat schema acceptance as causal validation.

Each immutable lesson has an ID, version, canonical record SHA256, previous
version digest, tested context, source-parent/comparator hashes and one or more
hashed evidence files. Observations are explicit positive/negative/mixed/invalid
facts bound to exact JSON pointers/values or verbatim text anchors. Causal
hypotheses are separate, reference those observation IDs and remain unproven or
limited to their tested scope. Every lesson states scope, transfer limits,
observation/causal/transfer confidence, falsification, a next discriminating test,
required gates, related queue/ROI IDs and conditional benefit/cost. Unknown
values are `null`; a requested reservation or planning scenario is not measured
gain. No automatic CPU-to-GPU, H100-to-H200, source-commutativity-to-additivity,
bake-to-balanced-causality or component-to-whole-score inference is supported.
Transfer confidence is deliberately low; establish a new in-scope experiment
rather than promoting an old record to a new target.

Lessons are append-only by convention and digest chain. To refine a lesson,
append the same ID with version+1, set `supersedes` to the previous record digest,
retain the old record unchanged and increment `registry_version`. New evidence
must already have a stable frozen path and SHA. Preserve those exact bytes at
that path; do not point an old record at a mutable “latest” report. Correcting an
incorrect interpretation requires a new version and an explanation in its
observations/scope. Recompute the new record digest with `learning.digest` only
after independent review. `schema_version` changes only for a schema migration.
The CLI intentionally has no write/append command and never silently repairs a
stale record. Digests cannot prevent a deliberate wholesale rewrite; external
review/version control supplies that accountability.

The six initial lessons retain parent-relative native progress despite actual
incumbent losses; commuting X4/STMatrix edits with mixed marginal timing;
recovered tiny EH observations including official CHECK sequential prepare
medians and negative alternatives; counter-contract overconstraint and the
unknown original failing tensor; full RQ11 payload/escape/cache accounting on
CPU weights without CUDA transfer; and actual prepare-selection variability
motivating the queued PW39 work. Their six **proposed, unqueued** ideas link
existing ROI/queue IDs to specific discriminators, controls, stopping rules and
cost assumptions. A reference to an existing queue ID is provenance, not a queue
state claim or execution authorization.

For a future idea, query the registry, inspect matched scope and negative
outcomes, then state which prior hypothesis the experiment discriminates and
what observation would falsify it. A parent-relative win is retained as learning
without becoming an incumbent win. Preserve tiny signed observations and
estimator disagreement; do not impose an invented gain cutoff or add cross-run
minima. Invalid setup supplies validity learning only; its timings never become
performance evidence. Root will integrate the required learning step into its
separately owned protocol/queue gate; this implementation changes neither.

Registry revision2 appends v2 metadata corrections for prepare selection and
RQ11: their future proposed controls now appear only in next_test, while measured
context names actual comparators. The original six v1 records remain unchanged.
It also appends the independently validated W-ring lesson and a seventh proposed
matched-context test: a tiny selected-policy gain coexists with a negative
equal-depth barrier contrast and mixed split-versus-coupled timing. These distinct
causal contrasts remain separate rather than being averaged into a mechanism win.

## Registry revision4: PW43 and selected-profile failures

Revision4 appends two lessons and two proposed discriminators to the unchanged
v3 lesson/idea prefixes; frozen consumer snapshots and v1-v3 archives remain
untouched. Reproduce with `PYTHONPATH=. python3 proj_rsi/audits/build-learning-v4.py`.
The resulting archive is `learning_archive/registry-v4-9e7d3bf8.json`.

PW43 is locally valid but regresses actual independently prepared PW38 under
both descriptive aggregate estimators: PW38-minus-PW43 is−13.920680us paired and
−28.966159us F1. All30 signed contrasts remain in canonical local observations,
including14 positives/12 negatives/4 ties. Later M256 confirmation cells expired;
only qkv completed. The next discriminator separates admission accounting from
comparison policy, using the immutable refinement brief and actual PW38 control.
A standalone production incumbent derived from search remains a proxy.

Selected profiler source revision2/jobv1 failed context metadata before prepare.
Revision3/jobv2 establishes a real context and completes unchanged73.931256s
prepare, then fails GV cluster launch with capture rc9. Actual GV S2/V0 follows
DSM cluster barriers, not a global-counter last-arriver route. Missing retained
postrange oracles/counter/cleanup means no profile row is admitted, including
router rows that completed collection. Hardware-only metrics versus broad
sections are a proposed recipe discriminator; no cause, gain, transfer, new ROI
or future job ID is fabricated. The archive binds actual raw collections and
the independent source-bound diagnosis by SHA256.
