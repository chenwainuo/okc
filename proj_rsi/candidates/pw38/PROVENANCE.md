# PW38: cumulative production source with optional EH

The actual standalone source is `submission.py`,277234bytes, SHA256
`29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`.
It combines PW36's RS32 small-M RQ and PW35 RTK router with optional EH variants.
It excludes the cursor regression. This file is implemented and CPU checked;
GPU correctness, actual selection, performance and official scoring remain unrun.

Parent PW36 source SHA256:
`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`.
Six strict edits come unchanged from `../rs32_eh_backport/delta.json`, bound by
v2 manifest `a4ccf23b7ff6b9a383ad0d6299d7de2d0d0d18b60d85e287150a0e465aac9869`
and isolated EH source
`9067795eb0d01abe576744a874e3504d6371a92ebc05543708c500b8d83a4819`.
Reversing these edits restores the entire PW36 file byte-for-byte.

The only changed named AST nodes are RQCU, RQV, RQPROD and RQBAKE. RQSBAKE is
unchanged. EH adds s4s5zEH FL65538 and s4s5hzEH FL65539 at M128/ST5/GS1 only.
Original candidates remain in the compile and selection lists. The original
non-EH epilogue is preserved as the compile-time else branch. EH bit65536 is
separate from HALF16 and EARLY32; EP is deliberately unported.

All six RTK router nodes exactly match PW35/PW36. prepare, kernel_fn, fallback,
small-M bakes, M256 bakes, all30 task-shape paths,240s prepare budget and1% R6
selection margin remain unchanged. There is no LDS_CURSOR in this file. PW37's
source and manifest remain frozen; its new RESULTS annotation records why the
cursor is experimental rather than an accepted improvement.

CPU proof independently checks exact composition and reverse restoration,
parent/backport file pins, all preserved AST nodes, strict-anchor negative cases,
EH output bitmap, drain/proxy/publication order and byte-identical regenerated
EH device translation unit/flags. It reuses the pinned offline cubin/SASS only
for NT128/ST5 FL2,3,65538,65539. Non-EH controls have exact1792/1832 PC/predicate/
instruction streams. This is offline CPU evidence, not actual JIT or GPU results.
No production import, prepare, extension build, compilation or CUDA call occurs:

```sh
python3 -m proj_rsi.candidates.pw38.cpu_proof
```

## What the existing selection threshold actually does

EH is an option, not a forced active path. `_rqbake` sorts all validated RQ
candidates and selects the fastest only if it is under0.99times the lower R6 F1
baseline. There is no additional1% EH-over-old-RQ requirement: a0.13us EH gain can
be selected when the RQ family already clears R6 by1%. Conversely, a family
within1% of R6 remains on R6 despite a small EH gain. Measurement noise, ties,
failed validation and prepare-budget guards can also leave EH unselected.
Retain actual PICK records and selected graphs before saying EH is active.

Do not lower this threshold based on the historical0.13–0.16us kernel result.
The concrete next diagnostic is specified in `same_geometry_lab.json`: current
PW38 fixed FL2vs65538 and FL3vs65539 pairs with identical geometry and KS, plus
actual same-run incumbent/parent prepared-policy controls. All outcomes and
small gains remain visible; no fixed tiny-gain kill rule is imposed. A later
threshold change, if supported, must be a new version and separate causal test.
