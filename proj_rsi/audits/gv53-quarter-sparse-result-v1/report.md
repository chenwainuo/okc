# Sparse-header M8: valid three-arm regression

Same-job three-arm comparison passes recorded correctness, lifetime and source/build checks. Positive first-minus-second favors the second arm:

| First -> second | Paired median us | First-before-second / reverse us | Blocked endpoint-average us | Second wins / first wins / ties |
|---|---:|---:|---:|---|
| Parent -> quarter | -0.191999599 | -0.192000531 / -0.144000165 | -0.239999034 | 6 / 17 / 1 |
| Parent -> sparse | -0.223999843 | -0.223999843 / -0.303999521 | -0.271999277 | 5 / 19 / 0 |
| Quarter -> sparse | -0.063999556 | -0.063999556 / -0.095999800 | -0.032000244 | 10 / 14 / 0 |

All24signed effects per comparison remain in analysis.json, including five favorable sparse-versus-parent and ten favorable sparse-versus-quarter triples. All three contrasts use the same24triples; they are correlated, and median effects cannot be added. Each pair is ordered12/12 and each arm occupies each position8times.

Six blocked medians(control/quarter/sparse/sparse/quarter/control) are[21.152000874,21.407999098,21.536000073,21.536000073,21.600000560,21.376000717]us. Parent endpoint drift+0.223999843us, quarter+0.192001462us, sparse0. Parent-averaged21.264000796,quarter21.503999829,sparse21.536000073us. All blocked contrasts are negative as well. Do not overwrite predecessor trial's positive F1endpoint-average+0.256us with1.152us drift: that remains separate historical evidence. This matched three-arm result supplies no speed promotion.

Source and execution:

- Five collected artifact hashes and13dispatch pins match. New combinedCUDA/SO hashes match actual files. Exact oldquarterSO SHA4b85c059c953e91989f6d613e8148269cfbb8e0480ab120585abcac492c3e44c loads viaPyInit gv53_quarter_607e6faec9d5b5e706ac, nativearm1. Newmodulearm0 is parent,arm1 sparse; result bindings and rawarm echoes agree. No silent replacement or predecessor recompile.
- Collected newSO instruction PCs/predicates/full128-bit encodings exactly match CPU prototype for both parent and sparse. Actual VOTEU0x4b0 exists; !P1BRA0x820 skips headerSHFL0x840 and !P1BRA0x860 skips repair to firstHMMA0xab0. FourPRMT precede sparse firstHMMA versus16parent. Native mechanism is realized; no build/selection mismatch found.
- Raw attrs for all three:64registers,local0,userstatic0,dynamic17408,1024threads,queried active-CTA capacity1,reservedshared1024. These are resource legality/capacity observations, not measured achieved scheduling.
- Actual prospective Wseed5344 and both bank hashes match previous quarter trial. All4096groups/4759accepted records map exactly before timing; parent reads original bank,quarter/sparse share identical transformed bank. Immutable W/payload/banks and private current-X/output/fullcounter/scratch guards remain checked. Recorded actual output/input tensors are not archived for offline recomputation.
-129raw observations(114timed/15warm),9normal/outlier/zero task checks,24basis columns,6own terminal sites,15capturedgraphmutation replays and30recorded task/FP64 snapshots pass. MaximumFP64 L2=0.001661842624,worstrow=0.001733328999. Every terminal is checked for its own lastcurrentX before further replay.
- Exit0; no residual process/container/GPU allocation. All graphs drain/reset; postscope allocated/reserved0,BLASmapempty. Total45.678924425s,compile/load42.245794183s,oldSOload0.001914900s included inside compile/load,weight/pack0.398340467s including transport/mapping0.105248937s. Do not add nested timing fields.

Cause confidence and next decision:

PROVEN correctness/transport/current-output validity and actual sparse native implementation. PROVEN sparse loses both parent and its failed quarter predecessor on paired and blocked point estimates, with favorable individual observations preserved. PROVEN by source/accepted density that predecessor performs32768header broadcasts while sparse performs one per occupied unit(at most4759), plus4096ballots and unchanged4759record broadcasts. This is expected source-level dynamic work, not captured hardware instruction counters or cycle fractions.

The specific hypothesis that unconditional empty-unit header broadcasts explain a recoverable quarter loss is not supported by this variant's measured outcome: removing most of them did not rescue speed. UNKNOWN whether ballot/branch overhead outweighs savings, header latency already overlaps useful work, four separated repair/MMA phases lose issue-level parallelism, or larger code/dependency/memory effects dominate. Static code is still455nonNOP versus parent348 and64registers unchanged. No clock, cache or thermal cause is established from timings alone.

Root direction: no further quarter source variants before dynamic evidence. hotloop_counters owns a bounded exact-SO parent/quarter/sparse M8 counter proposal; reuse existing artifacts to inspect actual memory/issue/dependency behavior. No new sweep, unchanged rerun, fullpolicy/full30/profiler gate or automatic family retirement. Reopening condition is a concrete dynamic bottleneck that supports one changed critical path with costs priced, then root chooses a new experiment. Thin C ABI seam is validated for later new builds; current experiment stayed on reviewed Torch seam and no migration is required for exact-SO profiling.

Recommended DONE: VALID_REGRESSION, fixedM8same-W/Xonly. Do not integrate/promote sparse or quarter from these primary estimates. Root owns the transition; the family remains diagnostic-active, not rejected.

Loop: profile -> measure -> hypothesize -> implement -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](../../ITERATION.md).
