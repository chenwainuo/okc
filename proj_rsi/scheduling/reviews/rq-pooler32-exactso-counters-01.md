# GO: exact incumbent RQ poolerM32 counter adapter

Independent reviewer rq_lab_review; author moe_decode_lut. Focused source/readiness GO for proposal b08d2dfd1f1cf218afda1c9e3343e56d188b74eac1d9c981f7c5ca701de70c41. No GPU run or speed claim. All19 proposed pins verified; runtime manifest29663bde96334781b9ba737c126fde63694fe8a5bd49641176d5307099a9cbce remains unchanged.

Exact RQ SO ab6c3ad7 loads its pinned PyInit module, arm0 only. Existing native wrapper, originalcodec and earlier native proof bind poolerM32/N4096/K8192/KS4 to rq_control::rq<32,6,1,0,0,0,35>. Live captured graph must contain that exact mangled function, grid32x4x1/block384x1x1/dynamic99488, supported noncluster attrs and local0. No load_inline/JIT/freshprepare/selector or offset arm. Separately pinned GV sparseSO is used only for its already-proven postscope ATenBLAS cleanup export; zero GV math/profile calls.

Current-X/output live in fixed private guarded arenas. Original weight/pack/escape immutable snapshots, full physical counter guards/active reset0 and sink checks are retained. Normal/outlier/zero/32ring-split-boundary checks precede capture; five warmups precede one NVTX-selected profile replay. Its own retained output is checked before any later replay or mutation; five new-input captured mutations follow. Numeric failure propagates separately from metric/export failure. Drained graph reset and scope release precede BLASclear/emptycache; root still validates external worker cleanup.

NCU profile=true explicitly opts into existing container capability; no NCU installation or clock setting. Existing12metrics are runtime queried against version2025.1.1. Kernel replay/graph node/cacheall/clocknone/NVTXprofile/one-launch count preserved. CSV requires one exact function/geometry row, finite nonnegative metrics, positive duration, percentage units and finite integer replay passes>=1 with pass units. Aggregate stall samples cannot attribute producer/consumer roles or removable time, and profiler duration is not a clean benchmark.

Bounds90inner/120outer; capturechild max60 and bounded overall remaining-time budget. Unique child process group is SIGTERM/reaped then unconditionally SIGKILL-drained on failure, including exited leader/live descendant; normal completion also drains descendants. ProcessLookup races are handled. No surviving capture is accepted. The original proposed files were not edited by reviewer.

Independent CPU reruns PASS: capture --cpu-check, profile --cpu-check, test_cpu seven parser negatives and real no-GPU fork/group fixture with exited leader/live descendant. No CUDA/NCU/API/SSH/queue/git execution. Reuse existing math/native/codec contracts; no new broad proof or profiling gate.

Scope: one diagnostic on an existing validated incumbent binary. Runtime compatibility, numerical terminal, metric availability and root cleanup remain actual-run observations. No PC attribution, kernel improvement, official score or automatic follow-up is authorized by this GO. Root owns admission.

Loop: profile -> measure -> hypothesize -> implement -> validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle<=10s. [Contract](../../ITERATION.md).
