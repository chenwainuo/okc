# Sol10 smallm incremental serving trial: CPU frozen, root review pending

This is a concrete executable focused lab, not a promotion or GPU result.
`tools/smallm_increment_lab.py` compares the actual freshly prepared `kernel_fn`
of pw36, frozen rs32, and Sol04's rs32-derived cursor candidate. Source hashes
are pinned in its code and `candidates/smallm_increment_trial/manifest.json`.
The official incumbent remains rs32/3200.615008us; pw36 is a development
comparator. No frozen parent, queue, production submission or ledger was edited.
No GPU context, extension load, prepare, benchmark or profiling ran here.

The default admitted scope proposed for independent review is qkv32. All five
affected/potentially affected shapes are CLI choices, but qkv16/pooler32/
unpooler32/o32 have CPU regression evidence only until separately GPU tested.
No scope-wide or official admission follows a qkv32 result. A retained r6 or
unmodified rq pick is logged and cannot be credited to the recurrence.

## Identity and first-math gate

Each policy imports its original file, builds the entire original rq32 variant
list through its original `_rqext`, records exact generated source, extension
name/path/SO hash, actual JIT SASS and resource text. The existing extension
name uses a process-local 32-bit hash of complete source. The lab explicitly
rejects a same-name/different-source collision across policies and verifies the
cached module slot's complete source. Identical source may safely share a
binary. It neither adds a variant nor rewrites dispatch.

For rs32 and increment, all three guarded NT16/ST5, NT32/ST5, NT32/ST6 FL35
runtime JIT opcode multisets must equal the respective CPU-admitted SASS.
Runtime resource text must report 168 registers and zero LOCAL. Before/after
CPU proof keeps LDS/SYNCS/HGMMA/WARPGROUP counts fixed and removes one
UIMAD.WIDE.U32 site. The selected affected kernel is then queried through its
original info-only rqrun path: 168 registers, zero local bytes, and unchanged
72840/83080/99488 shared bytes as applicable. Physical register budget is
40*128 + 232*256 = 64512 = 168*384. This is an address-cursor change; producer
publication, W/X retirement, math ordering and counter protocol are unchanged
as proved in `sol04-smallm-increment.md` and the pinned source restoration.
The exact JIT opcode gate is conservative: compiler variation is an invalid
setup requiring review, never permission to ignore a discrepancy.

The rq32 build/dump occurs BEFORE prepare, hence before the candidate's first
math bake. The CPU-only compile is removed from prepare's import-clock budget
by resetting `_T0` immediately before each full original prepare. This models
a warm build cache and records the tradeoff: compilation normally overlaps
prepare. All original bake choices, validation and margins remain intact; the
actual picks may differ from historical official/H100 policies. Full prepare
can run unrelated bake math, but is the simple trustworthy actual policy
comparator explicitly allowed by the task. No forced rq plan is labelled actual.

## Predeclared measurements and validity

One full weight fixture (reference seed 5151) is reused identically across all
three prepares. Each actual kernel_fn first receives every K basis column
(8192 qkv/pooler, 4096 unpooler, 1024 o), exact x8 scaled weight columns, then
normal/outlier FP64 checks. Timed graph nodes capture actual kernel_fn,
including its allocation/zero/main/correction work; untimed basis checking
alone adds an output copy. Graph validation includes task normal/outlier,
zero, fresh normal, all-row probes, NaN output poison and mutable scratch
poison. Weights and all packed/escape tensors are snapshotted; counters must
return to zero. The unrelated monotonic rw counter must stay at its initial
value. Fresh prepares supply independently allocated scratch and plans.

F1: 11 samples each control before, increment, controls after reversed; report
all raw samples and min(before median,after median)/increment median ratios.
All F1 arms use the same eleven fresh fixture seeds. Paired: 30 repetitions,
same fixture for all three arms, six permutations each repeated five times;
report per-shape median(control/increment) ratios and every pair. Each launch
starts after a 256MiB L2 flush. No finite sample deletion, trimming or selection.

At the terminal F1 and paired launch, BEFORE any replay/mutation/profile,
check every retained actual timed output against that policy's recorded fresh
fixture and verify current X against the last actual launch, finiteness,
FP64/task oracle, packed immutability and counters. This gate reads outputs;
it cannot repair them. Actual graph outputs are held separately by policy.
Every raw sample is immediately appended to raw.jsonl so a timeout preserves
partial timings. Failure traceback, source pins, run phases, picks, JIT and
results are retained. Invalid attempts provide no performance admission.

## CPU validation and bounded root command

`python3 -m py_compile proj_rsi/tools/smallm_increment_lab.py` passes.
`python3 proj_rsi/tools/smallm_increment_lab.py --cpu-check` passes without
importing torch: 16512 recurrence events, static opcode/barrier/math counts,
manifest hashes, and physical register equality. CPU proof lives in
`candidates/smallm_increment_trial/cpu-proof.json`.

Root alone may execute after an independent critic reviews these exact bytes.
Proposed bound: inner1140s/outer1200s, one exclusive H100 container, no NCU.
The longer bound allows three complete 240s prepares plus compile/validation;
precompiled extension cache may reduce it. Root should use its existing
SSH/transfer/guard and cleanup conventions around this exact remote payload:

```bash
PROJ_GPU_CONTAINER=okc-proj-smallm-increment01 timeout --signal=TERM --kill-after=15s 1200s /root/okc_run.sh timeout --signal=TERM --kill-after=10s 1140s python proj_rsi/tools/smallm_increment_lab.py --output proj_rsi/candidates/smallm_increment_trial/attempt01 --root-review-token proj_rsi/audits/smallm-increment-independent.md --shapes qkv32
```

The review-token argument names the actual reviewed artifact and is logged;
it is not an automatic approval parser. Root must confirm critic GO and source
freeze independently. On termination, root retains complete stdout/stderr and
checks guard/container/compute PIDs absent and GPU memory released. The lab's
finally block records cleanup responsibility, not a claim that it can inspect
its own post-exit CUDA process. No official submission is authorized here.
