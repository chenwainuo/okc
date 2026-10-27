# Native-X versus DP profile capture: validity pass, exports queued

Capture validity passes. Counter interpretation is **QUEUED_CPU_EXPORT**. Local
NCU and its Python report SDK are absent; no remote export or GPU operation was
performed by this reviewer. The parent-owned serialized normal-capability export
job must finish before numerical stall/sample conclusions can be drawn.

The immutable scheduler collection is
results/proj-rsi/scheduler/artifacts/native.profile.v1.01/collection-1793077258598870796/.
Every inventory entry is hash-verified. The report copied for exact import is
input/native-profile-01.ncu-rep,1080736bytes,SHA256
4bbde4f4d2a157dde8504dee2764c1a80706dd4ae1ff735725703681b39c6ddf.
Original capture artifacts are untouched. Host log shows precisely three selected
nodes: DP rx_pack, DP rq_px_dp, NX rq_tma_x, in that order; each receives34NCU
replay passes. Those passes are not independent benchmark repetitions.

Driver SHA16e75027 matches the frozen source. Status is
COUNTERS_CAPTURE_COMPLETE_EXPORT_REVIEW_REQUIRED, phase IMMEDIATE_ORACLES_PASSED.
Immediately after the selected range and before replay/mutation, both retained
actual outputs pass the task oracle: relativeL2 1.421e-4, worst-row2.633e-4 and
12probe rows exact. NaN output poison is overwritten; DP's complete1MiB XP is
byte-exact for currentX; W/packedW/escapes/X remain byte-immutable and both
counter arrays are zero. Native-X uses no XP; its raw TMA/LDSM storage-bit
roundtrip remains explicitly UNPROBED. The narrow counter driver does not rerun
the broad basis/FP64/mutation battery from its separate numerical trial.

Both graphs record RESET_AFTER_DRAIN. Host guard exits0 at2026-10-27T05:00:37Z.
The collected cleanup observation has no guard/container/compute PIDs, no
running containers,0MiB GPU memory and no cleanup failure. This establishes
capture/resource lifetime validity, not a speed improvement or completed
bottleneck analysis. Keep the parent capture VALIDATING until exports and
interpretation are reviewed; do not mark VALID_GAIN from exit0.

The audit reexecutes actual saved-SASS canonical PC/predicate/instruction,
physical A4 retirement/fence and source-qualified CFG role/wait gates for both
variants. The recomputed maps exactly equal the driver's saved maps. Main
resources are122registers/0local/512threads/65536rounded register pool and
occupancy1afteroptin; DP dynamicshared103600B, NX137232B. DP pack22registers/0local.

Static CFG denominators are retained separately from future dynamic samples:

| Variant | Math static PCs | Producer static PCs | Common/setup/unreachable PCs |
|---|---:|---:|---:|
| DP |324|339|185|
| NX |580|335|189|

These counts are not samples, probabilities or microseconds. Dynamic math,
producer, common, unmapped and total sample denominators remain null until the
source/raw exports reconcile. Common setup remains separate rather than guessed
as producer work. Every unclassified instruction is retained in the map.
Source regions derive Wfull102400,Bfull102448,Bempty102512,Xfull136192; fast and
outlined waits retain operand/predicate/branch/return identity. Native has
32mathLDSM,8mathbar5,8electedmathTMA and0mathLDG128 sites; DP has16math-loop
LDG128 plus16predicated common-setup LDG128 sites. These are static code facts,
not evidence that one phase dominates elapsed latency.

Required subsequent sample analysis must reconcile explicit #Samples to raw
smsp__pcsamp_sample_count, retain all-state/not-issued/reason columns and missing
reason residuals, and show dropped samples, interval, pass counts and all role
denominators. Preregistered sufficiency is≥200math and≥200producer samples per
variant, and≥30for a leading phase. Insufficient evidence stays inconclusive.
Source per-PC reasons can omit warpgroup_arrive; missing labels must not be
invented or converted to zeros. Predicated-on failed lanes are distinct from
unpredicated InstructionsExecuted; divide-by32 requires independent uniformity
proof. TRYWAIT can block even when its first predicate succeeds.

Six CTC RX/TX metrics are explicitly unavailable in the capture warning and
remain unknown. The unmodified-clock warning is retained. Per-node kernel replay
with cache-all also differs from normal DP pack-to-main cache consumption.
No stall/sample share is multiplied by a kernel duration, and no NCU duration
replaces the separate unprofiled complete-graph comparison.

Serialized export implementation is tools/native_profile_export.py. It imports
only the exact hash-pinned report, requires the recorded Nsight2025.1.1client
version at its exact target binary path, records export-binary SHA/version and
image identity, and runs fixed import-only raw/source CSV commands. Historical
capture-binary SHA was never recorded and remains UNKNOWN; export-client hash
must not be presented as capture provenance. Missing metrics stay null. The job
uses normal guard inner120/outer180 with no extra SYS_ADMIN, and no kernel target
or CUDA API. Root17 independent review is required before root launch. Output
child must be nonexistent; collect its parent. All report/source/client hashes
are rechecked and raw warnings/partial failures retained.

Recompute the present capture proof with capture_analysis.py. Numerical source
counter analysis is not complete and no new bottleneck or optimization is
claimed until the queued export artifacts are available.
