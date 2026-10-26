# aligned_epilogue — NO-GO for the narrowed m16 GV port

Decision: CPU/bit feasibility passes, but no credible >=0.3us BOTH prepare-F1/paired-graph gain follows from measured issue evidence. Do not build/queue a rebased GV prototype. No GPU run, full submission, new shape/geometry or tuning loop. Only CPU proof and never-launched local static compiler diagnostics were produced.

## Prior implementation and narrowed novelty

Power2 weight rebasing with ONE FP32 epilogue restore was already implemented in `cleanroom/contrarian/submission.py`: `_pack_layer` (~803) shifts a segment's exponent window onto[112,127], high7B'=56; `gemm_kernel` (~370) applies `(acc+correction)*scale` before conversion/reduction. Its DESIGN explicitly records the mechanism and65task checks/probe accuracy. It is not an untried idea. That policy/layout decodes4instructions/pair and has different pipeline/exception/reduction overhead.

Parent explicitly narrowed novelty to extracting the mechanism into the actual7op-aligned r6/GV decoder. That exact port is NOT asserted previously timed. P4's aligned-window rung had garbage weights and no correct epilogue restoration; its HMUL2 formulation scales every decoded register and is a different closed path. This audit evaluates only the narrow GV extraction, without reproducing contrarian.

## Actual parent coverage

Frozen `pw36/submission.py` SHA256 `2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`; frozen rs32 `a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`. No edits.

Actual pw36 logs `takeover-r1-pw36.log` and `router-small01.host.log` agree on m16 picks below; `PICK ... m16 r6` means `_r6`'s already-selected GV path, NOT a torch or fixed cuBLAS surrogate. Exact-m rq selection at m32 does not apply to16.

|layer/m16|actual tuple FMT,MT,ST,WR,WK,KU,S,RT,XS,V|grid/threads|local baseline registers/spills|
|---|---|---|---|
|pooler|1,2,2,2,8,16,1,1,0,1|128CTA x512|128 /0|
|unpooler|1,2,2,1,4,16,1,1,2,1|512CTA x128|110 /0|

Both S1 paths use only FP32 intra-CTA reduction before final BF16 `emit`. A prospective policy must fall back if a new prepare selects any other tuple/FMT/V/m, or an exact rq pick overrides it. No gains assumed against V0's11op decode; these actual picks use V1.

## Exhaustive CPU proof and actual packs

`cpu_feasibility.py`, CPU-only, no extension/prepare/GPU call:

-121allowed B values x65,536BF16 bitpatterns =7,929,856 cases. Nearest B' multiple8, even exponent shift j=2(B'-B);7,745,266 accepted cases have injective/reversible bit mapping and exact numeric `ldexp` identity. Signed zeros preserved; nonzero subnormal/nonfinite and out-of-normal-range shifts return unsupported. `(B'*0x01010101)&0x87878787==0` proves add==OR for ALL packed C words.
-37,951,930 safe finite onehot power cases (8..128) restore exactly.299,520 cases explicitly detect finite-original/scaled-product overflow; weight bijection alone is insufficient to justify output equivalence.
-Actual CPU reference seed5151 full matrices pooler4096x8192 and unpooler8192x4096: B54->56,j+4,weight*16,FP32epilogue*1/16. All original BF16 bits restore exactly, every c12 payload byte is IDENTICAL, every escape position/list nonXOR bit is identical.
-ALL4668pooler and3240unpooler escape XOR records must be recomputed in the scaled domain; reusing old XOR fields is wrong. Frozen `_c12`/`_unc12` and `_gvesc` prove the updated mappings. AllK probe powers8..128 match; CPU FP32 matmul on normal BF16 X is bit-identical after restoration.

CPU generator sequence is not CUDA generator sequence. This is not an actual HMMA accumulation/rounding, held-out outlier or graph proof. [NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) leaves BF16 matrix accumulation details and subnormal handling unspecified. A new port would still require full GPU numerical checks.

## Homogeneity / fallback boundary

For normal finite operations without underflow/overflow, binary power scaling commutes with rounding; undoing it on FP32 accumulators BEFORE any BF16 store/partial can preserve the original result. `emit` is the only final emission on the two admitted S1 paths. Restore all4float components before `sto<bf16>`. Scaling after BF16 conversion is not equivalent; future redemit/BF16partial paths are unsupported unless every emission is covered.

Whole-layer weight-only admission must reject nonzero subnormal/nonfinite weights, unrepresentable scaled BF16, failed inverse bit proof, invalid metadata or unsupported selected path, keeping the original pack/escapes. Even a normal representable scaled weight can produce overflow on finite activations: original BF16 bits31488 (E246),j4,x128 gives finite original FP32 product but infinite scaled product. A conservative probe-domain guard can reject such weights; arbitrary accumulators still need a finite-domain argument. No per-input CPU branch, input-derived persistent state or reduced precision is allowed. The encoding+scale is invertible at BF16 precision, not authority to assume unrestricted FP32 homogeneity or blanket rule approval.

## Source/SASS issue accounting

`source_sass_cost.py` extracts unchanged device code from frozen GVCU, instantiates ONLY the two actual baseline tuples, and compiles an isolated baseline/OR decoder helper. It implements no aligned GV entry, epilogue patch, Torch wrapper or deployable candidate. Local nvcc12.8 `-arch=sm_90a -O3 -cubin`: both real baseline kernels0stack/spills,128/110registers, one barrier. Artifacts `diagnostic.cu`, cubin/SASS, `local-compile.log`, `source-sass-cost.json`.

|decode for8BF16 values/lane|PRMT|LOP3|SHF|integer additions|total hot instructions|
|---|---:|---:|---:|---:|---:|
|actualV1 base|4|2|1|2VIADD|9|
|aligned OR helper|4|2|1|0|7|

Important correction: P4's register-A context emitted IMAD.IADD; the ACTUAL GV context compiled here uses VIADD with uniform Tb. Do not silently apply a throughput/pipe claim from the old mnemonic. Standalone helper nonNOP counts21->19; equal padded32instruction sections are not equal work. Full parent body includes150/153VIADD (128of them are the64decode calls*2), plus unchanged tensor/escape/address operations. Static whole-function counts include branches; they are not measured dynamic issue times.

Each target matrix has33,554,432BF16 values. This removes262,144warp add instructions total, while917,504decode ALU warp instructions (7/8values/lane) stay. It saves NO W/X bytes, PRMTs, escape walk, loads, reductions, graph nodes or fixed startup handshake. It adds2048pooler /4096unpooler warp FMULs at emission and potentially uniform->general table/scale liveness; pooler already sits at its128register occupancy cap. No aligned full-kernel resource claim is made.

262,144instructions over132SMs at ideal4warp instructions/SM/cycle is~496SM cycles of issue work:~0.25us at1.98GHz,~0.33us at1.5GHz. This is an instruction-slot-equivalent MODEL, not an upper bound, actual VIADD latency/throughput or achieved speedup. It requires the removed slots to be fully critical; overlap and epilogue work reduce the benefit, while latency-chain effects could differ. No counter identifies these2adds as the target m16 bottleneck.

Measured r6 lineage: small-m GV sits near the stream; clean qkv m1/8 whole decode+escapes costs~0.45/~0.74us, with multiple confounded x/occupancy terms. These are sibling-shape evidence, not bounds on target m16. P4's m128 whole-decode0.03–0.32us is also not a m16 bound. Neither supports attributing >=0.3us to just the2adds while7ALUops, escapes and all input delivery remain. At m16/unpooler the x-load/LSU block remains a separate cost.

Conclusion after the narrowed assessment: bit feasibility is real, but the speedup claim is unsupported and its optimistic instruction budget is around the kill bar before costs. NO-GO; no full rebased GV prototype/graph lab is justified by this evidence. Reopening requires independent evidence of a changed critical bottleneck, not calling prior epilogue scaling new or extrapolating garbage-weight aligned rungs.
