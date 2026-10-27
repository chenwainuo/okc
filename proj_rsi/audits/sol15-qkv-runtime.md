# Sol15 QKV128 runtime admission wrapper (CPU only)

Family group8_qkv remains ACTIVE research, measurement HOLD. No CUDA context,
extension load, prepare, benchmark, official submission, frozen-parent edit,
global ledger edit or commit was performed. This work supplies a concrete host
extension and CPU lab, not a performance result or completed runtime admission.

New version lives in `candidates/group8_qkv_trial/`; sources and all offline
artifacts are pinned by its manifest. Its extension cache key includes the exact
combined source, C++ declaration, compilation flags and Python wrapper source.
The device body is restored verbatim from canonical parent SHA
`d568e2612056b5bb37a1ebaadeb5c50d5c63345a736d660b144c25f4dac0d864`;
only its CPU-instantiation trailer is replaced with the host wrapper. No kernel
code generation adjustment or tuning is admitted by this version.

`runtime.Pipeline(w)` validates actual QKV W using pinned rs32 `_rqpack` and
`_rqunpack`, checks exact BF16 bits (including sign/outliers), derives CSR CAP
from real escape density before allocating, and owns packed W, metadata, XP
2 MiB and FP32 partial 4.5 MiB. Actual task-W validation has not run here.
`bind(x,out)` retains capture buffers for the Pipeline lifetime. The caller must
retain the Pipeline until every captured graph finishes. Each invocation launches
fresh pack -> split4 main -> ascending-slice FP32 reduction on the current stream.
Dynamic shared-memory opt-in occurs before capture, never inside the timed graph.
The new path has no counters; parent counter-zero checks remain mandatory.

Host guards require exact CUDA device, contiguous dtype, shape, size, alignment,
bounded CSR capacity, replicated table bytes and pairwise nonoverlap. The
specialization uses no tensor-map operands, so zero maps are correct. There is no
host synchronization or input-derived persistent cache in the three-node path.
Direct extension calls assume CSR contents came from the validated Pipeline;
shape guards alone are not a proof for arbitrary caller-supplied CSR bytes.

Run `python3 tools/group8_qkv_lab.py --cpu-check --compile-host` for CPU-only
admission. CUDA entrypoints are trapped during the check. It exercises all 65536
BF16 bit patterns through actual pinned pack/unpack, all seq0..127 with lane,
register, half and xor boundary fields, varying-density split4 lists and empty
metadata including sentinel capacity. It reuses the immutable parent's exhaustive
524288-word X bijection, all-output ownership, slice/slot DAG and physical
register checker, then compiles the full host wrapper to an object and inspects
that object's SASS without loading it. Kernel/resource evidence stays attached to
this version. All offline header failures are retained in `failures/`.

The environment has torch 2.13 CPU-only. Initial host compilation encountered
missing `cusparse.h` from an unnecessarily broad ATen include, repaired with the
narrow c10 CUDA stream header. A second failure was the CPU wheel's absent
`cuda_cmake_macros.h`; the offline compile uses the header's supported
`C10_CUDA_NO_CMAKE_CONFIGURE_FILE` option. This flag is not in actual extension
JIT flags: root must use its CUDA-enabled torch environment. Link/load/launch
validation is consequently still outstanding.

## Required root runtime gates

Independent source/resource critic GO is needed before GPU admission. Load the
hash-named extension and retain actual JIT `.so`, resource attributes, SASS and
physical A/D/fence proof BEFORE candidate math. Then validate actual W codec and
CSR bytes, all8192 basis columns, FP64 relative L2 <=4e-3 and row <=2e-2, task
normal/outliers using the reference checker, finite outputs, all-bit probes,
zero/fresh/current X and poisoned XP/partial/output. Preserve dense W, packed W,
metadata and input bytes. Check full XP against the CPU bijection; assert every
partial/output is overwritten and reduction sums slice0+1+2+3 in FP32 order.
The original parent proves ownership/order statically; numerical identity across
split and unsplit accumulation is not claimed.

Compare against fresh full `pw36.prepare(weights)` and its actual `kernel_fn`,
recording current selected plan. Compare the equivalent current QKV RQ split-K
plan if present in `state['rq'][('qkv_gate',128)]`; it is a causal reference,
not an assumed reverse-G8 winner. Selected bake ~16.8us is historical evidence
only and must not replace actual measurements. No timing runner is claimed
implemented here; root can invoke the concrete Pipeline from its independently
reviewed harness. This omission is an explicit integration blocker, not a family
ban or performance admission.

For every last timed fixture inspect the ACTUAL final timed output, current X,
XP and all immutable tensors BEFORE graph replay, mutation, or profiling. Keep
all F1 raw before/after and rotated paired samples and ratios for all controls.
Require exactly three new graph nodes, verify parent counters reset, and hold
all buffers through graph completion. A 240s prepare / 300s overall budget is a
proposed bound only: if fresh preparation plus correctness cannot fit, retain a
clear bounded-run failure and revise the bound before any timing claim.

Next causal question is whether 144 CTAs improve coverage enough to pay fresh
packing plus FP32 workspace/reduction costs against the selected QKV incumbent.
Gain is unknown. A loss diagnoses this version; it does not reject the family.
