# Independent PW38 hotpath trace02 repair review

**GO for the bounded all30 diagnostic, normal inner600/outer720 seconds; GPU UNRUN.** This admits the repaired collection attempt, not timings or a performance result. Source PW38 remains `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`. No GPU/context/prepare/NCU/API/queue operation was performed by this reviewer.

Exact pins:

- `tools/pw38_hotpath_trace_v2.py`: `e71a6c1350432eb122acacffeccd2f6a6b49d0d006e4cbf8e17609c4037d57ef`.
- `candidates/pw38_hotpath_trace_v2_trial/manifest.json`: `dd02e9e58bb59e9b37d0241d11864c99b714ae94c7e32359baf5070a6027d20a`; all139 dependencies verified, including seven restored inherited CPU-proof receipts. Initial incomplete c559 manifest is preserved separately; it is not the admitted manifest.
- New support: `193666639b9fa79f81cfdb50a3758bd6be32e0940dd49fcb2b66f84c91b1e986`.

```text
python3 /okc/proj_rsi/tools/pw38_hotpath_trace_v2.py --scope all30 --output /okc/logs/scheduler/pw38-hotpath-trace-02/run --review /okc/proj_rsi/audits/pw38-hotpath-trace-v2-independent.md
```

Attempt01 failed in diagnostic parsing, after the first case's current-output terminal. The actual Chrome file contains both CPU `user_annotation` and correlated `gpu_user_annotation` entries with identical SETUP and REPLAY names. Old parser selected name/ph alone, found two REPLAY/0 scopes, and rejected. Its initial synthetic fixture and my previous review missed this legitimate dual-category format. The old failure remains INVALID; it does not establish a valid all30 trace or a kernel performance result.

The collected attempt01 inventory has23 byte-verified artifacts, exit1,30 DRAINED_RESET graph receipts, and physical cleanup observations of no container/compute/guard processes and zero GPU memory. Capture SHA `1964b8894e2a88e73c05cfa12cbae7e87ed210cca0f328a2abc33a15e3844791`; Chrome SHA `628748fdaa16a3008a6916c868a465f30b166fdedd1f122c70f41a1104019652`. All30 metadata/task-case setup records and the first case's11 event samples/four mutation receipts/profile terminal are present, but the final all30 retained-output gate was not reached. None of this upgrades the failed attempt's admission.

V2's driver is byte-identical after reversing only its new trial-directory path. The parser changes exactly two scope selectors to require CPU `cat=user_annotation` and separately retains GPU-correlated annotations as diagnostic records. GPU annotations are not kernels, CPU inclusive scopes, or extra graph calls. All strict kernel multiset, multiplicity, duration, attribution and ordered three-replay checks remain. This is a trace interpretation repair, not a GPU source, graph, selection, timing schedule, counter or output-oracle change. Original driver/trial/source remain untouched.

The exact actual Chrome reproduces the old failure and passes the new parser: three CPU replay scopes, one exact selected GV8193 kernel per replay, CUDA durations13.024/12.000/11.776us, and all four GPU annotations retained separately. These are partial instrumented observations for debugging; they are neither cold/scored latencies nor a fullscope performance admission. Six independent actual-artifact mutations reject duplicate CPU replay scopes, GPU-only replay or setup without CPU scopes, extra/missing actual kernels, and a wrong actual symbol. Fourteen author parser negatives and eight extracted own-output/immutable/counter negatives also pass.

The original [all30 admission review](pw38-hotpath-trace-independent.md) remains applicable to the unchanged execution path: fresh original-clock PW38 prepare, complete19 actual RQ canonical/resource functions plus EH physical gate, actual loaded SO/source/SASS archives, current selected graph metadata, distinct retained X/output ownership, refined physical active/dormant counter contracts,330 cold samples/90 profiled replays/120 mutations,180 current-case terminals and final30 retained-output checks, and drain/reset cleanup. Historical fullscope/basis validity is reused; normal/outlier oracles rerun. No new all-K campaign or65+9 fullpolicy suite is claimed.

Host replay spans include synchronization/wait; CUDA duration sums may overlap and differ from graph elapsed. Both remain separate from unprofiled cold event timings and historical H200 scored latencies. Closed-library source remains UNKNOWN; no bottleneck, gain, request frequency, NCU metric or H100-to-H200 transfer is inferred. Root must independently validate actual attempt02 identities, counts, outputs, physical counters, immutable bytes, JIT artifacts and cleanup after collection. Planning240s is unproven; normal600/720 and the previously stated720s reservation/followup accounting remain unchanged.

CPU reproduction: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 proj_rsi/audits/pw38-hotpath-trace-v2-independent.py`. Exact actual reparse, six additional rejection receipts, dependency/artifact hashes and author CPU proof are in the adjacent JSON. No queue edits or automatic terminal admission accompany this GO.
