# PW39 full-policy confirmation trial

Actual policies are frozen RS32(a7b6e010…), PW38(29e557ea…) and PW39(8a133891…).
Driver `proj_rsi/tools/pw39_stack_lab.py` derives from the independently reviewed
PW38 full-scope driverddb48ef0…, preserving its strict all-shape schedule and
gates. No production source is edited. PW39's five GPU literals and generated
8RQ128/11RQ32 instantiations exactly match PW38; identical code is not evidence
of a host selection improvement.

Scope remains222case checks (65normal+9exact official distributions per arm),
4350timed calls over all30shapes,180own-terminal oracles and360retained-graph
mutations. Compare RS32/PW38 controls to actual PW39 serving selections. F1 uses
11replays per block; paired timing uses30trials with all six orders five times.
No forced candidate selection or discarded samples. All raw measurements, picks,
small gains/regressions, counters and final resets are retained.

The approved PW38 gates remain: source closure, actual JIT CUDA/loaded.so/SASS/
resources before return to preparation math, mandatory actual EH/control four-
function168register/0local/canonical-PC/physical gate for PW38 and PW39, selected
RQ resource queries, current-X/immutable byte tests, physical counter-arena
alias-aware reset/P1/RW/RTK progress/dormant contracts, float workspace/output
NaN poison and actual graph drain/reset. No import-time clock reset; original
240s preparation guards remain. Compiler threads must join within300s. Fresh
mutable arenas are separate per policy. Root independently verifies hardware
cleanup after the bounded process ends.

## New confirmation evidence

`prepare_confirmation` records the complete production `rq_confirm` audit,
including every invocation's screen/incumbent, all candidate tags and scores,
completed blocks, attempting/error metadata, partial-budget reasons, decisions,
elapsed/cumulative allowance and whether any invocation actually occurred. If
budget guards skip the phase or source helper returns the original screen,
that fact remains visible. No partial/error record is filtered into a successful
confirmation.

An in-memory observer is inserted immediately before the unchanged `_time_us`
`times.sort()` statement, only for PW39's new helper. It retains the five actual
replay times in execution order for each confirmation block, without issuing an
extra event, replay, timer, tensor read or CUDA call. Removing that one observer
expression restores the timing function AST exactly; CPU negative/positive tests
bind the capture point and preserved median. Its Python overhead is charged to
the original preparation and shared2s block-admission clocks; this is not a claim
that the allowance is a hard2s wall limit. The timing/helper observers are detached
after preparation. All RQ host launch statuses in these confirmation blocks are
also retained; a nonzero/exception makes lab performance invalid even when the
production helper safely retains an incumbent. Serving RQ calls continue to reject
silent declines as in the PW38 harness.

The wrapper independently recomputes each completed confirmation's two local
lower-adjacent-incumbent ratios, stable lowest-worst-ratio winner, unchanged0.99
margin, displayed real median and fallback decision from the retained raw blocks.
Actual selected outputs/counters remain authoritative; a confirmation decision
alone is not correctness or a performance result.

```sh
python3 proj_rsi/tools/pw39_stack_lab.py --cpu-check
```

Proposed normal job `proj_rsi/scheduling/jobs/pw39-stack-01.json`: inner3300/
outer3420seconds, MAX_JOBS1, no profiling capability. Review initiallyHOLD until
independent wrapperGO. Root alone finalizes review, registers and launches. This
worker performs no GPU/prepare/API/scheduler action. Exit0 never auto-promotes;
H100 validity, H200 check/scoring and incumbent review are separate steps.
