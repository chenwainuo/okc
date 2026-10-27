# One narrow three-arm down diagnostic

Compare the exact measured v1 LUT binary, the exact failed fused+rolled v2 binary,
and the new explicit two-value packed scatter. This is separate from the pair2
pipeline experiment. No other protocol or broad SS change is included.

Only the new T8DIV1 function compiles. All arms use normal project zeroing and
the original capacity grid, shared immutable banks from one original prepare,
and actual serving down inputs produced with v1 LUT up. Existing binaries and
module-factory sources are hash-checked before cache priming. Candidate execution
starts after the existing actual accumulator-lifetime/resource gate.

F1 captures seed7904 once and freezes its exact down input, sorted routes, experts
and post tensor for every block. Full byte hashes and routing counts are saved.
The five blocks are v1/v2/new/v2/v1 with3samples each. Eight same-input/routing
triples follow; each of the three pairwise comparisons has4/4 execution order.
Every raw time is saved before its own output/current-input check. Each F1 block
and every arm's paired terminal is checked before any optional profiler.

The actual first lookup/scatter instruction schedule is recorded with JIT data.
It is mechanism evidence; an unexpected CUDA13 scheduling result remains visible
alongside numerical checks and timings. Actual resources and lifetime checks are
mandatory. Mixed v1-up/new-down whole outputs must match v1 bitwise and the task
reference for current/fresh/empty inputs and their own post-timing terminal.
No mixed full-serving latency is claimed by this stage experiment.

CPU readiness passed: imports/help without Torch,20proposed job file pins, three exact
factory sources using mocked builders, onefunction actual-gate positive/negative
cases, and every pair's4/4 order. The changed scatter's4096cell/alignment/nibble
mapping was checked by the candidate builder; unchanged codec/matrix proof is not
repeated. CUDA remains uninitialized.

Bounds are130s inner/180s outer. Optional profiling is disabled and only allowed
after direct results and terminals are persisted. Root owns GPU, queue and final
admission. The proposed descriptor is HELD; independent GO must produce a bound
root-ready descriptor with top-level state BACKLOG. No GPU/API/queue action was
performed by the author. Root is prioritizing incumbent Marlin/projection work
following this one diagnostic.
