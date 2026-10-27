# Smallm increment: CPU source/codegen pass; GPU admission pending

Owner worker04. No GPU context, extension load, prepare, benchmark, profile, commit,
ledger or frozen-parent mutation was performed. Candidate is
`candidates/smallm_increment/submission.py`, SHA256
`0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737`.

## Actual selected policy and concrete question

Read ITERATION.md, queue.json, protocol.json and group16-feasibility.md. The
frozen official score remains rs32/fb548/3200.615008us. Official raw logs `candidates/rs32/profiles/submission.fb548b5dc6f04696.log`
and `results/proj-rsi/jobs/e22a4b9d693d4902.log` select these relevant kernels:

|shape|rs32 official|pw36 official|KCS per split|
|---|---|---|---|
|qkv32|t32s5hze /7|same|18 or19|
|qkv16|t16s5hze /7|same|18 or19|
|pooler32|t32s5hze /4|t32s6hze /4|32|
|unpooler32|t32s5hze /2|same|32|
|o32|t32s6hze /2|r6 retained|8|

All listed rq plans have GS1/PRB0/WM0/INS0/FL35. The candidate also affects o32
if its fresh bake picks rq; it must be included in regression/validity coverage.

Pooler16/unpooler16 retain r6 in this actual preparation; they are not claimed
as covered improvements. H100 pw36 takeover qkv16 retained r6, so official and
local dispatch differ. Pooler32 also chooses ST5 on that H100 versus ST6 in
this official log. Every eventual same-run comparison must inspect actual
prepared kernel_fn; neither fixed plan alone represents all preparations.

Concrete cost: consumer `ldr` at parent lines3959+ recomputes `s%ST` for its
full-barrier/LDS/Wempty addresses and `(s/ST)&1` for barrier phase, although
WM0/GS1 calls it in exact consecutive stage order. Before SASS has reciprocal
multiply/shift/remainder address arithmetic around the repeated loader, e.g.
M16/ST5 at0x27b0/0x2910/0x3360. These are codegen observations, not evidence
that this arithmetic is on a material latency critical path. Producer/CSR/
HGMMA/Bfull waits may hide it.

Profile question: does scalar ring bookkeeping constrain issue-to-next-LDS
latency in these *actual selected* kernels, or is it hidden behind W/X arrival,
escape correction and HGMMA dependencies? The new source isolates that question
without modifying decode, math order or asynchronous scheduling. Information
value is identifying whether an existing multi-load path admits low-risk
incremental address work; attainable gain remains UNKNOWN.

## One source change and exact restoration

Candidate lines3960–3984 introduce two scalar cursor values under constexpr
`(NT==16||NT==32)&&GS==1&&PRB==0&&WM==0&&INS==0&&FL==35`.
`ldr(G)` uses the current slot/parity for full wait and three LDS addresses,
reuses the slot for Wempty arrival, then increments slot and toggles parity at
ST. No change to mma/xrel/producer/rq_fix, packed layout, host dispatch, bake,
probe order, output, counter reset, tolerance or fallback.

`build.py` asserts frozen rs32 SHA
`a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`, applies
exactly four textual replacements recorded in delta.json, reverses every delta
and asserts full byte equality to that parent. The development pw36 comparator
remains SHA`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`;
this is a fresh rs32-derived source, not a pw36 integration or parent edit.

A CPU-only intermediate changed phase/slot but did not reuse the cursor for
Wempty. It increased static non-NOP instructions by6 and retained avoidable
modulo work; its source/cubin/SASS/log are retained in cpu-history/phase-only.
It was never GPU-run and is not a performance failure or family rejection.
Final v1 completes the same cursor change by reusing its slot at Wempty.

## Correctness, bytes and lifetime proof

Under this guard GS1 gives NGT=NGF=KCS and no partial round. The source calls
ldr(0) once, then only ldr(G+1) when G+1<KCS, with rounds G=0..KCS-1. Each
loader is therefore called exactly once in order. Before ldr(G), invariant
`ldslot=G%ST`, `ldphase=floor(G/ST)&1` holds by induction: initial0/0; the update
increments the slot and flips parity only on ST wrap. CPU exhaustive model
checks KCS1..128 for ST5/6, covering all above fixed KCS values and ragged KS7.
All threads execute the same loader/update path. Wempty still arrives once per
math warp at lane0 after exactly the same LDS + syncwarp. Full wait address,
phase, three LDS addresses and Wempty address are identical event-for-event.
No added memory access, prefetch, async issue, barrier, fence, early release,
register-A write or retirement occurs. Xempty, wgmma wait/commit/fence and all
producer ownership remain unchanged. Consequently every in-flight W/X/A
lifetime and byte count is unchanged. This is an address recurrence change,
not an async-overlap change requiring a new buffer or publication protocol.

Added scalar state is two integers; compiler reported register count remains168
for every compiled selected body. Runtime dynamic register policy remains two
math warpgroups232 registers/thread, producer warpgroup40, totaling64512 physical
registers per384-thread CTA. Existing dynamic shared bytes remain72840
(NT16/ST5),83080(NT32/ST5),99488(NT32/ST6); no occupancy improvement is claimed.
Actual JIT resource/lifetime inspection remains mandatory: standalone CPU
codegen does not establish runtime allocation behavior.

## CPU checks and machine evidence

`python3 -m proj_rsi lint .../submission.py` passes; the existing sm_90a reminder
is retained in lint.json. Compile selected full device bodies extracted by AST
from actual RQCU, removing only torch/ATen host includes and omitting wrappers;
no torch import or extension load. Exact RQFLAGS:
`-gencode=arch=compute_90a,code=sm_90a -O3 -DNDEBUG --expt-relaxed-constexpr`.
Only compile-only `-cubin -Xptxas=-v` added. Local nvcc12.8.93. No warning
suppression, architecture substitution or parameter sweep.

|body|non-NOP before/after|registers before/after|stack/spill bytes before/after|
|---|---:|---:|---:|
|M16/ST5/FL35|2169/2160|168/168|0/0|
|M32/ST5/FL35|2136/2122|168/168|0/0|
|M32/ST6/FL35|2161/2147|168/168|0/0|

All use16 barriers. SASS before.sass/after.sass and counts/delta JSON are saved;
manifest.json binds source, extraction, cubin and SASS hashes. All three bodies
remove one static UIMAD.WIDE.U32 site. LDS.128 remains12 static sites,
WARPGROUP.ARRIVE4, WARPGROUP.DEPBAR.LE5, HGMMA and SYNCS counts unchanged.
Static instruction counts include startup/epilogue and unrolled copies; they
are not dynamic counts, critical-path attribution, an execution proof or a
speed forecast. Added cursor compares/updates can offset the removed address
arithmetic, and compiler scheduling/register assignment changes are confounds.

## Next decision and bounded comparison contract

Request root + independent critic source/codegen admission only; no GPU job is
authorized by this audit. First decide whether this specific small address
change has enough information value for a bounded lab. If admitted, compare
actual freshly prepared frozen pw36 kernel_fn per selected same shape, frozen
rs32 predecessor and v1 complete kernel_fn pipelines. Also inspect each pick:
r6-retained shapes cannot be credited as rq wins. Keep comparison order balanced,
fixed sources and full source/JIT identity. Check all basis columns, FP64 and
task normal/outlier cases; validate outputs from the final real timed graph
BEFORE any replay/mutation. Check fresh/zero/outlier/NaN poison graph reuse,
weight/pack immutability, counter resets, resource/spill/SASS gate and cleanup.
Root is sole executor; no official job suggested yet.

Support: valid same-run complete gains reproducible on relevant actual selected
loads, with codegen matching recurrence and no validity changes. Refutation:
valid flat/slower complete timings despite correct outputs indicates removed
arithmetic was hidden or new cursor/scheduling cost offsets it. Preserve result;
next diagnose the observed issue path rather than banning address recurrence
or the rq32 family. If codegen/lifetime/output fails, timings remain invalid;
repair a new version before any performance decision.

Exact same-job ROI is the group16 audit law
`Snew=S0*exp(sum(log1p(-delta_load/Tload))/6)`. Conditional 1us genuine gain in
qkv16 and all three rq M32 layers gives decode16−1/decode32−3; with all other
loads fixed Snew=3158.126341us (42.488667us reduction). This is arithmetic only,
not measured/attainable gain. It illustrates multi-load relevance while making
no claim that this address-only change closes the200.615008us gap.
