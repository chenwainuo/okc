# RQ32 predicated escape XOR, CPU discriminator v1

GPU UNRUN. New isolated PW45 descendant, source cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210, parent835acb90 / manifest98821446. The first discriminator succeeded: offline NVCC produces different machine code, with exactly19 unrelated main streams unchanged. No frozen parent, selector, prepare budget or queue changed.

The new rq_fix_pred helper is used only by NT32/ST5or6/GS1/PRB0/WM0/INS0/FL35. It preserves the warp-uniform seq loop, entry encoding, escape index, lane-spread32-entry window loads and shuffles. For each of16 packed A words it predicates the XOR with `(lane==entry_lane)&&(word==entry_reg)`. Inputs mask/reg are copied into assembly-local temporaries before outputs are written. It uses explicit read/write A constraints, no dynamic register-array indexing and no local memory. All other types retain the old helper. RQSBAKE and all other top-level AST entries remain exact PW45. Removing the two exact edits reconstructs PW45 byte-for-byte. Its pre-host device body and parent ST5/ST6 PC streams match the PW38 ancestor.

The kernel types are shared by selected QKV/O routes too; this is a type-scoped change, not a claim that other shapes are unaffected. Initial selected-hotpath question is pooler32/N4096/K8192/KS4 and unpooler32/N8192/K4096/KS2. Both execute32k64stages per CTA and128CTAs. Official PW38 scored traces selected ST6, whereas the recent local PW38 control selected ST5. Retain both references with accurate labels; do not force ST6 and call it locally selected. PW45 is the requested stack parent, not an invented official scored promotion.

NVCC12.8 sm90a/O3 offline results:

| Type | Static instructions parent/new | SEL parent/new | BSSY and BSYNC each | Registers | Shared |
|---|---:|---:|---:|---:|---:|
| ST5 | 2160/2064 | 94/30 | 8/4 |168|83080|
| ST6 | 2184/2096 | 94/30 | 8/4 |168|99488|

Each type has64 actual predicated in-place XOR sites (four static helper copies). Branches decrease by8. Uniform parameter loading also changes: ULDC.64 12→8 and LDC.64 7→11; no bandwidth or timing interpretation is assumed. Math/LDS/TMA/store/barrier/fence/register-policy opcode counts remain unchanged. All21 functions have zero stack and spill reports; the target resource pool remains168×384=64512, and dynamic232×256+40×128=64512. Actual GPU attributes, occupancy and performance remain unmeasured. Static instruction counts are whole-function counts, not dynamic escape-hit frequency or execution time. Await agent14 fresh trace before any loop-dominance or GPU-scope claim.

Read-only CPU verification:

```
python3 proj_rsi/candidates/rq32_predxor/proof.py
```

It checks inverse edits/ancestor/body/AST equality; all19 unchanged PC/predicate/instruction streams; exact emitted inline-PTX operand order;131072 encoded lane/register/half/xor entries and262144 lane/register predicate cases; and compiled predicate-to-physical-HGMMA A-word mapping for all four helper copies in each changed kernel. Each native predicate selects the correct logical A word and four contiguous A words bind to each following HGMMA input quartet. Five destructive negatives reject wrong selector, wrong input word, wrong half shift, swapped lane/target constraints and wrong native XOR destination. This is CPU interpretation plus source/compiled structure, not a GPU roundtrip or numerical result.

`compile.py` reconstructs the exact frozen PW45 offline source factory before compiling the isolated descendant and saving SASS/resources. It invokes NVCC/cuobjdump only, no extension load or CUDA context. `build.py` reconstructs only this new directory. Do not run mutable generators after independent freeze without preserving a new version; read-only `proof.py` is the review entrypoint.

Prior-learning constraint: the old ldr cursor0d0e already showed pooler losses and small unpooler paired positives alongside all matched F1 losses. This change does not repeat that ring arithmetic experiment, add an empty-escape fastpath or duplicate MMA body. Both gains and losses remain evidence; code shrinkage alone does not establish a kernel bottleneck or latency benefit. Proposed later root-only admission must bind actual loaded JIT, current selected whole-kernel controls, all-K-basis/FP64/task escaped/refill-boundary fixtures, full physical counters/sink/immutable storage, current-X bytes, fresh outputs/workspaces, immediate own-last-fixture terminal outputs, exact graph identity and drain/reset. Shared-type QKV/O regressions must be retained if this becomes a stack candidate. H200/official score transfer is unproven.

Planning only: remaining independent review15CPUminutes and any later wrapper/analysis20CPUminutes are uncertain estimates, not measured costs. This package requests no GPU reservation. Future GPU scope/budget is for root after fresh trace and independent07/18 review. Predicted gain and actual runtime/cost are unknown. No GPU, extension prepare/load, API, queue, git or shipping action by author.
