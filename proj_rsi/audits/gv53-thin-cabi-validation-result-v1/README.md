# Thin C ABI actual validation: successful bounded seam

Recommend DONE / VALID_NO_GAIN for research.gv53.thin.cabi.validation.01.
The diagnostic succeeds; no native or official speed gain is claimed. Actual
H100132SM collection1793141233697823368 completes all four interface cells.

| Measured driver/build phase | Seconds |
|---|---:|
| Torch import |0.856693|
| collected PyBind import |0.000694|
| thin native compile |1.609522|
| thin link |0.048528|
| thin compile+link |1.658050|
| thin dlopen |0.000732|
| full driver |5.985288|

The retained prior GV53 PyBind compile/load42.186038s is from another attempt,
with C++/NVCC/link/import portions uninstrumented. Actual saved build seconds
remain unknown. This demonstrates a low-cost actual thin build, not a paired
40second reservation saving.

CPU reconciliation verifies12collected artifact hashes and18dispatch pins;
raw journal exactly matches all324timed and40warm observations. All24task
checks,8exact basis sets/64columns,16own terminals and40fresh retained-graph
mutations pass. Both interfaces agree bitwise at each paired terminal and
all five mutation fixtures per cell. The two360instruction target streams
match actual collected PyBind SASS in PC, text/predicates, encodings and control.
Both arms'32combined attribute fields agree:64registers,0local/static,
17408dynamic bytes,1024threads, queried maximum CTA capacity1, reserved1024B.
The query is capacity, not measured active scheduling.

Explicit nondefault capture handles differ from replay handles in each cell;
passed C ABI handle assertions plus current-X graph mutation/output checks
validate actual captured execution in this scope. Strong-owner probe passes
after caller X/output/scratch/counter references drop; two fresh-input graph
replays pass before reset/drain and weak-owner release. Invalid native arm
returns−90 and propagates. No unsafe asynchronous fault was injected; existing
synchronization/terminal error checks remain active.

All graphs reset after drain. Exact admitted PyBind cleanup reports workspace
map1→0 with33554432B configured workspace, finalallocated/reserved0. External
collector records exit0, no process/container,0MiBGPU allocation. Saved output
tensor bytes were not archived separately; CPU review verifies source-bound
successful runtime assertions and recorded numerical values rather than
recomputing GPU tensor bytes.

All signed interface points are retained in analysis.json. PyBind-minus-CABI
paired points for parentM1/M8 are+.128000/+.064000us; keyedM1/M8−.128000/+.032000us.
Endpoint-average F1 is+.256000/~−.000001/−.047999/−.032000us. These are mixed
same-kernel replay/context observations, not native improvements or score gains.

The seam is ready to reuse in a next bounded fixed-kernel proposal, retaining
explicit current stream, Python-owned graph-buffer lifetimes, pre-capture
attributes, current-input/own-terminal/mutation checks, source/native identity
and the version-matched admitted cleanup bridge. This does not certify another
kernel's argument layout, cluster protocol, CUDA image or full policy migration.
No new broad proof campaign is needed solely because this ABI validation passed.
Root owns adoption/queue transitions. Author analysis used CPU only.

Frozen machine DONE evidence is done-evidence.json; analysis.json preserves
all signs, source/result pins and limitations. Later additions require named
supplements; delivered hashes must not be overwritten.

Loop: profile → measure → hypothesis → optimize → test → validate → stack →
submit measured gains → learn → reprioritize → repeat; idle ≤10s.
[Contract](../../ITERATION.md).
