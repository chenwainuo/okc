# Isolated both-projection SS prerequisite screen

This runnable screen is separate from the proposed full-expert benchmark. It
imports the frozen08 adapter and exercises actual up (K4096/N8192/gatherDIV8) and
down (K4096/N4096/DIV1),32 local experts/topk8 slots. It does not import SGLang,
Marlin, CUTLASS or cute. A temporary module exposes only the exact pinned pure
`nvfp4_marlin_process_global_scale` AST needed by actual adapter.prepare, then
restores prior modules. This shim is explicit; it does not validate serving imports.

Use the existing guarded image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,CUDA12.8,ninja,H100SM90 >=70GB.
The root's reviewed H100 has132SM and driver580. No new image/wheel installation
is expected. Runtime checks Torch2.8/CUDA12.8,actual NVCC12.8, hardware, all eight
canonical production PC streams, actual exact per-function registers/0LOCAL,
1024staticshared, maxthreads/maxdynamic and occupancy>=1 before any project call.
Shared resource scheduling and GPU execution remain exclusively root-owned.

An appended diagnostic host query and independent mover kernel have an exact
production-source inverse. Offline CPU-only compilation succeeded; all eight
production SASS streams remain exact. Actual JIT repeats that gate; no compiled
extension has been loaded during author CPU work. Diagnostic mover reuses codec
primitives and roundtrips the same shared swizzle; its independent oracle decodes
raw low/high nibbles numerically, raw E4M3 scales and independent gather semantics.
It probes all64K64 stages, first/middle/last N tiles, experts0/31 and every T/DIV.
This mover probe complements the actual WGMMA all-K/output checks.

The independent stage oracle computes numeric E2M1×rawE4M3×2^-119 rounded BF16,
FP32 matmul with TF32 disabled, then BF16 output cast followed by BF16 global
multiply (rawglobal rounded BF16 then×2^119). It never reads candidate packed words
or processed scales. Reference-only expanded weights are not candidate model banks.
All-K sparse inputs cover every4096 K coordinate for each of eight instantiations
with64 rows per replay and64 replays; their single-term BF16-boundary expectations
are exact (signed-zero equivalence). Random/cancellation probes use a predeclared
screen-only elementwise bound abs<=.001+.02*abs(reference), retaining max errors.
These are diagnostic gates, not substitutes for existing full-task numeric/model
oracles or relative-L2/norm/router/SwiGLU checks.

A failed exploratory CPU check is preserved in history: raw E4M3 subnormal bytes
1..7 do not map to numeric processed scales with the inherited codec. This lies
outside authoritative `muse_moe_w4a16/nvfp4.py` random_weights domains: normal scales
[8,448],wide scales[2^-6,448]. Screen fixtures cover all finite nonnegative NORMAL
E4M3 codes0x08..0x7e plus zero, and gate that domain explicitly.1920 independent
nibble/scale pairs prove the numeric exponent factor there. General checkpoint
scale-domain support, negatives/NaNs/subnormals remain unclaimed; do not silently
apply this fixture-domain result to arbitrary NVFP4 weights.

Each eight-instantiation edge suite also covers block48→T16, ragged M1/5/7/9/17,
balanced/skew/duplicate expert selections across unique slots, mixed nonlocal,
one valid lane, all nonlocal and a defensive four-rounded capacity/post tail.
Duplicate sorted output slots reject because they race. Route builder models
local0..31 and nonlocal-1; it does not validate the serving aligner/expert-map code.
All route IDs are current source slots; up gathers slot//8,down gathers slot.
Every projection explicitly zeros the whole output as the original adapter does.
Retained graphs check fresh X, outputNaNs, zero X, valid→empty via the SAME post
pointer, route/experts/post metadata and values, and bank/raw/scales/global full
storage bytes and view metadata. There are no global progress counters in this
kernel; route metadata/post are the counter-equivalent immutable inputs, with only
the declared post0 mutation. Every graph drains before reset, including failures.

Stage-only timings use equivalent M64/skew/alllocal fixtures for T8/16/32/64 on
both up/down. Explicit zeroing is included. Twenty position/order-balanced rounds,
11-sample F1 brackets, all raw samples and own terminal outputs are retained.
F1 uses disjoint arm seed blocks and paired rounds use same-seed fixtures. Actual
last timed output is checked BEFORE replay, input copy, poison or profiling.
One separately counted postterminal profiler replay confirms the dispatched
project<T,DIV> node plus outputzero and retains its own output oracle/trace.
Different block sizes change padding/capacity/empty-CTA overhead: T timing differences
are confounded and cannot uniquely establish producer overlap. No stock/anchor
ratio, official score, whole-MoE speedup or threefold claim is generated.

Original target remains stock861333f/source05ec9a8a, all14 original shapes and
independent holdouts, per-shape report/geometric mean and actual full serving on
8H200s with model/norm/router/TTFT/throughput/collective validation. The separate
projection3198us/3ms target cannot complete it. Source08's proposed whole-expert
anchor/current comparisons remain a separately quoted/admitted followup.

Cost quote before GPU work: additional author/review CPU budget90minutes;
provisional GPU600inner/720outer full reservation. Actual compile/prepare/oracle/
timing/drain costs are unknown and recorded, not treated as zero. Existing image
needs no expected wheel transfer; source checkout transfer bytes are measured from
manifest, but bandwidth/cache/transfer time are unknown. No whole-expert or official
followup cost is included or free. A timeout or invalid result is retained for a
bounded repair, without automatic retry/promotion or a family ban. Root decides
queue utility; no invented measured ROI or gain prior is provided.
