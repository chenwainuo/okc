# Isolated both-projection SS prerequisite screen

This runnable screen is separate from the proposed full-expert benchmark. It
imports the frozen08 v2 adapter and exercises actual up (K4096/N8192/gatherDIV8)
and down (K4096/N4096/DIV1),32 local experts/topk8 slots. It does not import
SGLang, Marlin, CUTLASS or cute. v2 directly prepares raw global scales as BF16;
no package shim or inherited processed-global helper is needed. This screen does
not validate serving imports or the original full expert method.

Use the existing guarded image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,CUDA12.8,ninja,H100SM90 >=70GB.
The root's reviewed H100 has132SM and driver580. No new image/wheel installation
is expected. Runtime checks Torch2.8/CUDA12.8,actual NVCC12.8, hardware, all eight
canonical production PC streams, actual exact per-function REG/STACK/SHARED/LOCAL,
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

The independent stage oracle uses a CPUdouble/integer BF16 lookup over all16
E2M1 nibbles and254 finite SIGNED raw E4M3 scale bytes (4064pairs). Bits upload as
int16 and reinterpret BF16, so the oracle's GPU float32 FTZ cannot silently erase
tiny expected values. Canonical v2 BF16 weights equal numeric E2M1×rawE4M3; FP32
matmul has TF32 disabled, its output rounds BF16 BEFORE multiplying rawglobal
rounded BF16. The oracle never reads candidate packed words or processed scales.
Reference-only expanded weights are not candidate model banks. Every mainbank's
first N64/K64 tile deterministically covers all4064 pairs; each group16 has q0..15,
254scale groups cover every finite signed scale and remaining2groups are zero.
Raw mover compares uint8 BF16 storage bytes including signed zero. All-K sparse
inputs cover every4096K for all eight instantiations with64rows/replay×64replays;
single-term BF16-boundary expectations use numerical equality (reduction may
canonicalize signed zero). Random/cancellation probes use a predeclared screen-only
elementwise bound abs<=.001+.02*abs(reference), retaining max errors. These are
diagnostic gates, not substitutes for full-task numeric/model/relative-L2 oracles.

A failed exploratory v1 check is preserved in history: E4M3 subnormal bytes1..7
fail the inherited processed-scale codec. They lie outside the authoritative
normal[8,448]/wide[2^-6,448] fixture generator but inside the broader numeric
format. v2 is a new isolated source: canonical BF16 E2M1×raw E4M3 operands and raw
BF16 global scale remove the inherited2^-119/2^119 representation and its operand
denormal risk. The same cast-before-global boundary remains; timing/numeric
identity with Marlin or v1 is NOT assumed. Stage fixtures include all finite
nonnegative scale bytes, plus deterministic signed finite first-tile coverage.
NaN scale bytes0x7f/0xff are rejected by fixture validation, not introduced into
model banks. General checkpoint/model validity still requires full separate gates.

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

The untimed shared-cache dependency diagnostic runs each block8/16/32/48/64 at
raggedM5, with up cache13 and down output sharing storage, separate BF16 middle,
FP32 diagnostic SwiGLU, retained current-input/NaN/zero/valid→empty graphs and
final down oracle. Up values are deliberately consumed then overwritten; only
down is a retained chain output. This tests same-stream operand lifetime and
aliasing, not the serving activation implementation, norm/router or full expert.

Timing fixtures use the same original-token64 scope: up64 input rows, down512
expanded input rows,512 output slots for each. Down inputs remain independently
generated BF16 diagnostics rather than actual serving activations. The stage
numbers cannot be summed or labelled current serving dispatch gains. Both actual
production and explicit output-zero GPU nodes must appear in postterminal traces.

Run-entry timestamps begin before verify/review/Torch/device initialization.
Startup/preflight is retained separately. Interpreter/module import, container,
lock wait and checkout transfer remain external unknowns reconciled from root's
outer-wrapper timestamps; compile/preparation counters are not total charged cost.
The thin queue launcher retains startup failures before result.json exists.

The first two executable review drafts are retained in history. Independent review
caught a translation-unit preflight bug: the frozen physical checker assumes only
production kernels, while the augmented dump includes diagnostic movers. Runtime
now retains full SASS, extracts exactly the canonical eight complete production
symbol tables, and runs the unchanged actual gate on that production-only dump.
CPU regression executes the exact frozen gate and physical bodies: filtered dump
passes; raw augmented dump rejects. Exact full resource metadata remains compared
for the production set. The container name now meets the actual frozen trial
launcher's required `okc-proj-` prefix. Existing attempt directories are refused
before Torch import, and the thin launcher never writes into a prior attempt;
startup failure artifacts use an exclusively newly created output directory.
