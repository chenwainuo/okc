# RQ32 predicated escape XOR: independent CPU source GO

GO for the exact CPU source version only: candidate manifest
`31b0079efc89c2b35cbddf7c156c558f621871f0693f9ed8bc6a736f918c9188`,
submission `cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210`.
Parent is exact frozen PW45 source835acb90 / manifest98821446. This receipt does
not admit a GPU harness, reserve GPU time, infer measured gain, or promote a
whole policy. Reviewer sol12 is independent of author sol09. No GPU context,
extension load, prepare, API, queue transition or git action occurred.

All14 candidate-file pins and47 external pins pass. Reversing the two exact
source edits restores the entire PW45 submission byte for byte. The only
changed top-level AST entry is RQCU. Native target types are exactly
NT32/ST5-or6/GS1/PRB0/WM0/INS0/FL35; the other19 of21 main streams retain every
PC, predicate, instruction and operand. All old selectors, prepare budgets,
codec/CSR, loads, rings, barriers, counters, math, launch and output source
remain inherited. Source preservation is not a claim that every instruction
in the two changed native streams is otherwise byte-identical: parameter
loads and scheduling also change, as retained in the opcode deltas.

PTX input/output semantics are sound within this source scope. All16 A words
are explicit +r read/write operands. Mask and register selector inputs are
copied into assembly-local .u32 temporaries before any A output write; lane
and target inputs are consumed by setp.eq before writes. Therefore an output
cannot overwrite a still-needed input merely because the compiler coalesces
an operand. The local .pred/.u32 declarations are braced, so inline copies do
not collide. For each i, setp.eq.and produces (reg==i)&&(lane==target), and
predicated xor reads/writes the same A[i]. The mask is exactly
(xor7 << (8+16*half)); xor0 is a no-op, xor127 and both halves are preserved.
No shared/global memory is accessed by this inline block; memory clobbers are
not needed to describe its register-only effects. A data dependency still
connects decoded A to its subsequent HGMMA operands.

The warp-uniform sequence loop and its complete cursor/refill/shuffle tail
are unchanged. The new per-lane predicate changes A only and cannot change
ei, ec, nx or nxs. The independent window model covers19 scenarios: empty,
1/2/30/31/32/33/63/64/65/95/96/97/127/128/129 entries; future sequences retained
without consumption; sequence31/32/63/64/127 boundaries; and the largest
non-sentinel encoded sequence. Original/new values, cursor, loaded32-entry
windows, trace and sentinel termination match. Sentinel is not treated as a
payload entry; actual supported sequences remain bounded by the inherited
K/KS launch contract. Repeated entries and xor0 remain correct.

An independent native-instruction interpreter evaluates the actual emitted
eight escape blocks, including mask construction, lane/word predicate
combination, predicate-register reuse and in-place LOP3 XOR. It checks393216
vectors per block (3145728 total): every low17-bit lane/register/half/xor
encoding with active and two distinct inactive-lane classes, while high
sequence fields vary. Other inactive lanes are equivalent because the lane
GPR has exactly one read, the target-equality predicate. The expected A words
are bound independently to each subsequent physical HGMMA register quartet.
Three destructive native mutations reject reversed lane predicate, wrong
half shift and XOR changed to OR. The author's separate131072 encoded-entry /
262144 predicate proof and five source/native negatives also pass.

Independent NVCC12.8 sm90a -O3 -DNDEBUG --expt-relaxed-constexpr compilation of
the independently reconstructed exact new device source produced a cubin
byte-identical to the frozen candidate:59c4864cb9a7e50bb8e5295a33827c48812b7028e81ebd8fe440665ca2dbbf26.
Fresh SASS and resource dumps are also byte-identical. All21 functions report
168 registers,zero stack/local/spills and SHARED:1024 in the offline resource
dump. The inherited dynamic launch SMEM is83080bytes/ST5 and99488bytes/ST6;
384threads imply168×384=64512 rounded register pool, matching the unchanged
dynamic232×256+40×128 policy. Actual attributes, maximum dynamic shared opt-in,
occupancy and runtime lifetime remain GPU admission gates, not CPU facts.

The change is template-scoped, not layer-scoped. The unchanged RQSBAKE table
exposes these types on qkv_gate32 (ST5 KS6/7 and ST6 KS6), o32 (both KS2),
pooler32 (both KS4), and unpooler32 (both KS2). A later pooler/unpooler focus
cannot be called an unaffected-QKV/O stack. Runtime validation must retain
actual unforced selected labels, the true current incumbent, QKV/O effects,
full-K basis/FP64/task and escape/refill fixtures, current X, own-last-fixture
terminal outputs, physical counter/sink/immutable storage, exact loaded JIT
identity/resources, graph identity and drain/reset. The author's recent local
ST5 and scored PW38 ST6 references remain distinct; this review does not
replace agent14's fresh trace or infer escape-loop dominance.

This v1 had no discarded implementation failure. Frozen parents, prior cursor
losses/gains, PW42 failures and other global history remain untouched and are
not reclassified. Static code reduction and fewer SEL/BSSY/BSYNC instructions
are discriminators, not a latency prediction. GPU numerical behavior,
occupancy, whole-policy selection, attainable gain and H200/official transfer
remain unmeasured. Root alone decides the next bounded experiment after the
fresh trace and a separately reviewed executable harness.

Reproducer: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3
proj_rsi/audits/rq32-predxor-source-independent/review.py`. It rereads frozen
sources and archived independent compilation, runs CPU semantic models and
writes only its analysis JSON; it never executes a GPU kernel. The native
interpreter is a CPU model, not a GPU bit-roundtrip or a formal NVIDIA ISA
verification. Compilation log/cubin/SASS/resources and full modeled cases are
retained beside this report.
