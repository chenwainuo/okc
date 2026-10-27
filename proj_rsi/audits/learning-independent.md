# Independent learning-framework review: CPU_PASS

Reviewer sol17; author09;2026-10-27. **CPU_PASS for the frozen read-only learning
validator, registry revision2, retrieval and documented review workflow.** This
does not authorize an experiment, convert hypotheses into facts, admit a new
performance gain or change the queue. No GPU, API, remote or queue operation was
performed by this review.

Final reviewed SHA256:

| File | SHA256 |
|---|---|
| `proj_rsi/learning.py` | `c3115da4a23fbc264e2c2b30c2a4c01304cb0dca642b0e78835b2a796b77ca86` |
| `proj_rsi/learnings.json` | `86617478ad3d82b318d181abe240b5e84547a415497e13ee6912aa5687eaf906` |
| `proj_rsi/LEARNING.md` | `0e766bb0774d2f8f64fb8ad9818e05c36aebca76b05a8d40e6f13cc1b9ff815a` |
| `proj_rsi/tests/test_learning.py` | `1e70985a0f88f7bbfbb746c105da03c2417ef35d216194346af78138584d8f5d` |

The registry contains9 lesson versions/7 latest lessons and9 idea versions/7
latest ideas. All7 current ideas are proposed, transfer-unproven, with benefit
kind `conditional_scenario_not_measured` and unknown benefit value `null`.
Existing queue/ROI IDs are references, not registration or authorization.

## Independent evidence and immutability checks

Independently recomputed every canonical record hash and sequential per-ID
`version`/`supersedes` chain, without calling the implementation's digest or
chain validators. All18 records pass. Context/evidence pins independently match
the21 distinct referenced source/evidence files. Independently resolved all33
version-level fact selectors:20 exact JSON anchors and13 verbatim text anchors.
The independent JSON comparison used canonical encoding, including nested type
distinctions, rather than Python's loose bool/number equality. Every anchor
matches its frozen bytes.

The original6 lessons and6 ideas remain intact. Reconstructing their original
registry-version1 JSON from the first6 records of each array produces precisely
the original whole-file SHA
`03dc2fe8bf1d891e9cdd254f1b289d4afa73fec1661c850ab95a2e7b950aab62`.
This proves preservation beyond merely trusting author assertions or resealed
record hashes. Revision2 appends the two corrected lesson versions and linked
idea versions, plus the independently validated W-ring lesson/idea. No original
negative, invalid, tiny positive or estimator disagreement was overwritten.

Append-only enforcement has an explicit practical limit: a self-contained digest
chain is not a signature or a protected historical database. A deliberate
wholesale rewrite/resealing could be made structurally consistent. External
review and version control remain necessary, as the documentation states. The
CLI has no append/edit/repair operation; this review approves the exact frozen
package, not all future manually resealed content. Registry-version increments
are a documented append workflow, not proof against deletion of an entire
unreferenced lineage. Preserve this review/hash and historical bytes externally.

## Findings corrected before final approval

1. The initial prepare-selection context listed future PW39 confirmation and
   held-out fixtures as measured comparators. The author appended
   `prepare-selection-is-experiment@2`, preserving v1 and all observations. The
   current context now lists actual PW38/PW36/RS32/smallm-v3 policies; future
   PW39 controls remain only in the proposed test. Its scope explicitly records
   the metadata correction.
2. The same audit found future safe esc6 release/actual CUDA comparison in the
   RQ11 context. Appended `rq11-full-residency-ledger@2` separates actual CPU
   accounting from those proposed controls, with explicit correction scope.
   Matching idea versions now reference the corrected lesson versions.
3. The initial JSON-pointer helper accepted Python negative array indices. It
   now rejects negative/signed/leading-zero array indices and malformed `~`
   escapes. Valid object keys and RFC escapes remain supported.
4. A resealed `VALID_GAIN` containing only negative facts initially validated.
   Gain now requires a positive observation. Regression retains negative facts;
   mixed requires a negative plus positive/mixed observation; invalid setup
   requires an invalid observation. These are structural minimums, not automated
   causal adjudication.
5. Top-level type checking initially let nested `True` equal numeric1 inside a
   JSON object/list. Recursive exact comparison now checks types at every level;
   a temporary evidence fixture demonstrates rejection and a positive nested
   control passes. Current frozen facts already had the correct types.

These were actual independently reproduced negatives. No failed version is
relabelled as having passed the repaired checks.

## Fact/hypothesis and outcome review

The seven current lessons preserve distinct scopes and retain their adverse
evidence:

- Native-X records positive DP/DS parent progress while preserving the actual
  incumbent loss. It never promotes a parent-relative win into an incumbent win.
- X4/STMatrix composition records exact source commutativity separately from
  timing. Positive native-parent paired progress coexists with negative F1
  constituent contrasts and the actual incumbent loss. Additivity is unproven.
- EH keeps the tiny signed official CHECK prepare observations, the unselected
  negative hinted unpooler alternative and the separate fixed-geometry mixed
  result. Sequential prepare medians are not balanced causal estimates, whole
  board gains or leaderboard submissions.
- Counter-contract diagnosis preserves INVALID_SETUP and the unknown original
  failing tensor. A later reproduction of the old blanket-zero problem is not
  retroactive identification of that tensor. Active reset/alias protections
  remain required, and invalid timings never become performance evidence.
- RQ11 retains the exact CPU QKV packed-byte saving1069056 alongside the
  negative full-residency saving−11085360 when cached esc6 metadata is charged.
  CUDA-fixture bit identity and serving speed remain unknown; FP32 router
  truncation is excluded.
- Prepare-selection preserves the negative pooler32 full-policy observation,
  an unchanged-path positive observation and ST5/ST6 confounding. Selection
  noise is a hypothesis; PW39 improvement is an unmeasured future test.
- W-ring binds the now independently validated analysis/report. Split versus
  actual selected PW36 has positive F1+0.223999843µs and paired+0.175999478µs;
  equal-depth barrier separation is negative in both estimators. Split/coupled
  sign disagreement and identical-main context differences remain explicit.
  The overall lesson is VALID_MIXED across its distinct contrasts, preserving
  the scoped selected-policy positive without claiming a mechanism, robust,
  official or transferable win. Its full runtime validity is inherited from
  the linked independent W-ring review; this framework review does not rerun
  the295 calls or GPU gates.

All current causal hypotheses remain `unproven`. Observation confidence does not
become causal confidence; transfer confidence remains low. Every lesson supplies
scope, transfer limits, falsification, next discriminator, controls, stopping
rule and required gates. Costs with unknown reservations remain null; the known
15CPUmin/0GPU counter-test plan and60CPUmin/600s PW39 planning scenario are labeled
as future costs, not achieved gains or new approvals.

## Read-only CLI and tests

Independent execution passes all20 unit tests, including evidence mutation,
broken chains, negative preservation, counterfactual/CPU-to-GPU relabel attempts,
strict pointer and deep-value negatives, and append preservation. `validate` and
`summarize` return VALID. The four CLI operations below leave bytes and mtimes of
all4 framework files and all21 referenced files unchanged:

```bash
python3 -m proj_rsi.learning validate
python3 -m proj_rsi.learning summarize
python3 -m proj_rsi.learning query 'rq11 residency' --hardware H100
python3 -m proj_rsi.learning query 'Wring barriers'
```

RQ11/H100 retrieval returns the corrected v2 proposed idea with an unproven
transfer warning, not the CPU lesson as an H100 observation. W-ring retrieval
returns the scoped mixed lesson and its matched-context proposed discriminator.
Query uses lexical matching over latest versions, not semantic proof or an ROI
optimizer. Root should inspect linked evidence and adverse outcomes before
forming a new test; query order itself grants no priority or execution authority.

The implementation imports stdlib only and reads files. It does not construct a
QueueStore, invoke subprocess/network/GPU APIs, import candidate code or mutate
registries. Tests use temporary copies and CPU Python CLI subprocesses only.
The root-owned mandatory-learning protocol step remains a separate gate; this
module does not silently change scheduling policy.

CPU_PASS closes this review of the current framework/backfill package. Future
lesson/idea appends require new evidence pins, record hashes, scope review and
the existing independent admission process; none of the proposed tests is run
or approved by this verdict.
