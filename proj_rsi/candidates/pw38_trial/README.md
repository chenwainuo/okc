# PW38 full cumulative serving-policy trial

Driver: `proj_rsi/tools/pw38_stack_lab.py`. Three frozen actual policies:
RS32 a7b6e010..., PW36 2a067059..., PW38 29e557ea.... No forced EH, no rewritten
production selector, no source mutation and no import-clock reset. PW38's optional
EH status is reported from actual selected plans, including an empty selection.

`h100_bench.py` is the existing65-case/30-shape local harness. It is a screening
harness: it does not fail immediately on all numerical failures and does not
retain/check actual last timed outputs or persistent arena progress. The new
wrapper retains its exact shapes and read-only256MiB cold-L2 graph timing while
adding strict terminal and source-bound state checks.

Scope per arm: all65 normal cases (five layers x M1,5,7,8,9,16,32,33,128,130,
256,512,1000) plus the nine exact official layer/M/distribution cases from
`proj_rsi/correctness.py`, including router9/qkv512/unpooler1000 outliers. Every
case checks the task's FP64/probe oracle, X bytes, immutable prepared tensors
and the selected counter contract. Original preparation retains full-K basis,
FP64 and per-candidate numerical checks; no basis claim is made for a variant
skipped by preparation. All30 timed shapes compare all three prepared policies.

Per shape: three separate current-X/output graph workspaces;11-repetition
bracketed F1 followed by30 paired fresh-fixture trials using all six orders five
times each. Every sample is retained. Both phase boundaries check each arm's
actual last output against its recorded fixture before any replay or X/output
mutation. Then retained graphs receive zero/outlier/all-probe/fresh-normal X
mutations with output and mutable floating partial workspaces poisoned toNaN
(counters are untouched; active persistent-zero RED2 workspaces HOLD). No scratch
zeroing or timed-output repair
is added. All graphs drain/reset on success and in failure cleanup.

Physical counter arenas are deduplicated by allocation identity, including
aliases. Per checked interval, the exact replay count is retained. RQ/V12/global
last-arriver reset arenas must be zero; P1 active even tickets advance byS and
odd zero flags by1 per tile/call; RW/RTK active release words advance byK/KR.
RW/RTK slot and geometry come from the exact generated source switch passed to
load_inline. Every other word/arena remains byte-value identical. Counter wrap,
unknown modes or unexplained allocations HOLD/fail. Cross-policy mutable arena
aliases fail. Read-only weights/packed/escape tensors are byte immutable.

The load_inline interceptor saves generated CUDA, exact loaded.so hash, actual
JIT SASS and resource metadata before returning to any preparation math. Any build/gate exception is retained
and invalidates the lab even if the original production prepare catches it; a
successful actual PW38 four-function EH/control gate is required. Current
NT128/ST5 FL2/3/65538/65539 streams must exactly match pinned CPU canonical PCs,
predicates and instructions with168registers/0local. EH actual function-set,
publication/retirement/TMA drain gates rerun. Selected RQ plans additionally query
actual registers/local/shared with the source-correct info argument. Production
prepare clocks remain unchanged; all interception time is charged. Original
240s phase guards remain, background compiler threads must finish and join within
300s. This is a fresh full prepare, not a prepare-only run or promised official
cold-prepare feasibility. All BAKE/PICK stdout is retained by the outer guard.

CPU proof checks source pins,30/65/9 scope, exact official distribution list,
source-generated RW/RTK slots and geometry, counter negative controls, and JIT
missing-EH/register/local negatives without importing torch or production.

```sh
python3 proj_rsi/tools/pw38_stack_lab.py --cpu-check
```

Proposed normal scheduler job: `proj_rsi/scheduling/jobs/pw38-stack-01.json`.
Inner3300/outer3420seconds bounds include compile/prepare/validation/timing.
Review is initially HOLD until an independent reviewer binds the frozen driver.
Root alone updates approval, registers and launches. No scheduler state was
changed by this worker. Timeout/exception is invalid/incomplete; root separately
verifies actual process/container/GPU cleanup. Exit0 is not root validation or
an official H200 score. No fixed minimum-gain rejection threshold is imposed.
