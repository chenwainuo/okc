# QKV geometry follow-up: analytic hold

State: **ANALYTIC_HOLD_NO_PERFORMANCE_VARIANT**. This is a CPU feasibility/evidence artifact, not a CUDA kernel or GPU-ready trial. The valid split4 result remains a40.288µs full-pipeline regression against actual RS32 at17.088µs and PW36 at17.408µs. No family retirement or performance forecast follows.

`analyze.py` verifies all12 collected artifacts and46 dispatch pins, binds exact executed source/route/profile identities, reconstructs fixed launch geometry, and preserves unobserved data as null. The archive contains kernel names but **no pack/main/reduce durations, SM count or actual wave telemetry**. Main post-optin occupancy is1; pack/reduce/control occupancy is not recorded. Wave quantization cannot be tested causally from these absent observations.

The current source explicitly launches pack512×256, main(36,4)×512 and reduce1152×256. Main output width is64 and NT is already128; no TT symbol exists. Selected control uses source-bound NB18/KS7 and384 threads, producing126 CTAs. Launch expressions are source-qualified, not captured launch telemetry. Actual wave counts remain unknown.

Necessary unchanged-schedule conditions admit equal SK1/2/4/8/16; SK32/64/128 fail unconditional startup or four-stage pairing. Ragged boundaries require a new tail/parity/CSR implementation, not a KS-only edit. SK8 preserves stage coverage atKCS16 but raises main CTAs144→288, doubles FP32 partial/reduction bytes4.5→9MiB and leaves logical main-X payload72MiB unchanged. Its CAP can be bounded32 from the recorded SK4 CAP32 via list-subset reasoning, but group count doubles576→1152; this is not a new GPU measurement.

No evidence currently supports positive performance ROI for a geometry-only source version. SK8 requires coherent main/CSR/workspace/reduction/oracle edits; changing only grid silently loses K coverage or violates storage bounds. If TT128 means N tile128, new WGMMA/D64/producer/shared/output/resource proofs are needed; this is not a safe launch edit. If it means NT128, the current kernel already uses it.

No changed kernel was emitted or compiled. Existing source's exhaustive current-X/output/physical proof is independently rerun; proposed partition/CSR schedule models are necessary arithmetic only. Five invalid stage-count controls reject. Actual numerical/resource/JIT validity for a changed version remains UNRUN/UNKNOWN.

Specific next decision: only a separately authorized bounded diagnostic should record actual SM count, per-node launch dimensions/shared bytes, post-optin occupancy and node GPU durations in the real current-X full pipeline with actual selected PW36/RS32 comparators. A main-dominated result plus repeatable geometry-dependent wave tail would support a tightly scoped source version. Dominant auxiliaries or no main improvement under matched geometry would refute a wave-only recovery hypothesis. Do not convert wave counts or source payloads to microseconds.

Applied learning: `wring-mechanism-policy-separation@1`, `prepare-selection-is-experiment@2`, `composition-not-additivity@1`, `rq11-full-residency-ledger@2`, and the source-bound actual QKV failure lesson `proj_rsi/audits/qkv-runtime-01/lessons.json`. Mechanism isolation and actual selected policy remain separate; all later timing must include the actual serving comparison and preserve every signed outcome.

Reproduce: `python3 proj_rsi/candidates/qkv_geometry_followup/analyze.py`. Detailed machine evidence: `analysis.json`; audit: `proj_rsi/audits/qkv-geometry-followup.md`. No GPU, prepare, API or queue action is performed.
