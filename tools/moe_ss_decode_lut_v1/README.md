# Bounded real-serving decode LUT stage screen

`pilot.py` is runnable on the same pinned CUDA13/H100 container as the prior true
serving pilot. It reuses that pilot's verified immutable candidate tree and exact
original extension SO. It compiles only the new full signed-table T8 DIV8/DIV1
functions, applies the original accumulator lifetime gate with function count2,
and checks actual loaded resources before candidate math.

It prepares original serving weights once, shares original immutable banks, and
records actual up/down inputs by executing the same serving method code object
with a private globals dictionary in a local subclass. It changes neither the
original method/class nor its module globals. Each actual serving capture checks
the existing reference oracle and current input immutability. No stand-in input,
activation, aligner or mathematical reference replaces a serving stage.

Each arm has its own captured stage input/output buffers. Graphs call the normal
`project` wrapper, including output zeroing and the original capacity grid. Fresh
input/routing and all-nonlocal poison checks precede timings. Up and down each
receive8 balanced matched pairs on new real serving captures. Every raw event
time is written before its numeric check; exact stage outputs and current input
bytes are checked after every timed call. Each stage checks its own terminal
before the next stage. Losses remain signed evidence for the measured arm.

Optional `--profile` runs only after validated timings and own terminals are
saved. Profiler/export failure is recorded locally; numeric/runtime failures of
attempted replays still invalidate the result outside that handler. The default
queue descriptor does not enable profiling. No counter privileges are needed.

`job-root-ready.json` is HELD and has exact file pins. Root owns focused driver
review, binding the review SHA in argv plus review metadata, transfer and GPU
admission. The review placeholder is deliberate; an unrelated prior GO is not
reused. Budget is180s inner /240s outer, including one candidate JIT, without a
new candidate archive/tree snapshot. No GPU or queue action was performed here.

CPU readiness:

```
python3 tools/moe_ss_decode_lut_v1/cpu_readiness.py
```

The executed readiness checks import/help without Torch, all pinned file hashes,
the same-code/private-globals recording mechanism, and positive/negative cases
for the inherited twofunction SASS gate. They do not repeat the codec or matrix
campaign. Evidence is in `cpu-readiness.json`; CUDA remains uninitialized.

The candidate is the **8KB full signed common table** at
`serving/experiments/moe_rsi_20261027/ss_decode_lut_v1/full_table/`. Both earlier
2KB and current8KB source/proof artifacts remain preserved. A losing stage
rejects that measured configuration, not all lookup designs. If only one wins,
a separate mixed-stage full-serving validation can follow; this screen makes no
full-serving speedup claim.
