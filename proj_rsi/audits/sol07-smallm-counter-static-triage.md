# Smallm394798 static counter triage

Root reports terminal03:07:57UTC exit1 at frozen driver's invariants counter
assertion. No JSON/counter-name payload inspected here, so this note identifies
a source-supported hypothesis, not the observed failed tensor. No GPU, remote
import, prepare, candidate/lab edit or ledger edit.

Frozen v2 driver collects every state path containing cnt/sink after all three
prepares. It requires every collected cnt tensor zero, except names containing
rw_cnt which must equal its snapshot. That combines active serving counters
with unrelated prepared plans and dormant counter state.

Original source's `_gvcall` at1981ff selects per-instance `p1cnt` for V bits27/29.
GPU code738 publishes zeroer flag through atomicAdd;794 assigns monotonic tile
tickets and799 increments the zero-flag. Consumer1083 waits against ticket/S+1.
These paths contain no per-replay zeroing of those ticket/flag words. Thus
"self-zeroing" in the Python comment refers to output zeroing, not a contract
that the bookkeeping tensor is all-zero on return. Nonzero p1cnt after prepare
can be valid even if current shape serves rq and never touches that state.
Other GV reduction modes1110–1122 and1148–1158 explicitly reset their counters;
selected rq reduction3913–3926 separately resets its two active arrays. The
router rw_cnt is already documented monotonic and specially handled by v2.

Minimal diagnostic before repair: archive exact failing path, shape, selected
policy and baseline/current values; classify its source protocol. Keep selected
rq and ordinary reduction reset obligations strict. Dormant protocol tensors
should retain baseline when untouched; active monotonic ticket protocols need
their own progress/flag contract plus output correctness. Do not blanket zero
state before checking, blanket accept nonzero counters, or discard the failed
attempt's correctness gate to admit its partial timings. Author13 and auditor14
own the actual repair/contract; root notified of this hypothesis.
