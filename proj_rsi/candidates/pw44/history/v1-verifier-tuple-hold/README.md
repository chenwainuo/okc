# PW44: optional private output slabs for current EH

CPU-only candidate over scored PW38 source29e557…; no GPU/context/prepare/API/queue work, performance unknown. PW39/PW40 and historical EP early-poll are excluded.

Four optional M128/ST5/selfzero variants retain hint/no-hint controls:

| Variant | FL | Shared dynamic | Pre-store cross-WG bar1 |
|---|---:|---:|---|
| existing EH z/hz | 65538/65539 |144520|retained|
| private z/hz resource control |327682/327683|178176|retained|
| private z/hz independent output |851970/851971|178176|removed|

The additional2×16KiB private slabs start at logical shared offset144384 and end177152. The declared dynamic allocation includes1024Balignment headroom:178176B, +33656B versus baseline. Offline compiler metadata also reports1024Bstatic shared for all old/new kernels, yielding179200Btotal for new variants. Actual runtime occupancy must account for both; the production new-flag-only occupancy query does so from the compiled function. Register pool is64512/65536 at384threads (168static perthread, dynamic232×256+40×128).

New variants are appended to the original prepare contenders only for QKV128/KS7 and pooler128/KS4. No winner, timing or input-derived output is hardcoded. Original EH and ordinary controls, lists/order/margins/validation remain intact. Old19device streams and resource records remain exactly identical after compilation; the new private branch restores to the entire old EH body by changing only slab pointer and the optional pre-store barrier. Both post-store128thread publication rendezvous/proxy fences, first-half overlap, current zero poll/trap/global proxy fence, per-WG Dwait0 and final both-TMA read drain remain unchanged. Removing cross-WGbar1 on original reused Xring is unsafe and is a negative test, not an option.

The new-flag-only host launch guard rejects unsupported shape/KS, non-SM90, resource/API errors or insufficient actual resident CTA capacity before launching candidate math. It runs after the original shared opt-in; the exact extracted guard template is offline host-compiled. QKV's grid is126CTAs, pooler's128CTAs. CPU bounds/resource models are not actual occupancy measurements: mandatory actual JIT attributes/opt-in/occupancy/source/SASS/counters/whole-grid evidence remain required before any root GPU trial. A declined optional candidate retains the original production fallback.

Storage proof enumerates all16384BF16output coordinates into four disjoint8KiB half-slabs and checks alignment/headroom/full X/W/barrier/flag separation. Event DAGs preserve per-warp D completion, all four warp stores/publication, zero publication before first reduce, both commits and issuer read drain before CTA lifetime ends. No cross-WG convergence is assumed once private storage is used. Exact qkvKS7ragged18/19stage and poolerKS4/32stage partitions retain the old startup/steady/tail producer/A/mainloop bytes.

CPU tests: `python3 -m proj_rsi.candidates.pw44.cpu_proof` recompiles offline kernels/guard and writes only this package's proofs; `python3 -m proj_rsi.candidates.pw44.protocol_proof`, `.codegen_proof`, and `.learning_proof` rerun pure proofs. Future reviewers should use `verify_frozen` for read-only closure. No new raw hardware bit-mover claim exists; STSM/TMA numerics require actual basis/FP64/task/own-final-output/mutation oracles.

Normative boundaries: retained wait0 protects accumulator use ([PTX8.7wgmma.wait_group](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)); per-warp `.sync` is not an invented cross-WG barrier. Retained `.read0` completes source/tensormap reads, rather than asserting destination writes are already globally visible ([PTX8.7bulk wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)). Existing generic-to-async publication remains required ([PTXasync proxy](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#async-proxy)). No per-PC sample count is converted into microseconds or intrinsic barrier cost.

`cost.json` quotes all stages before implementation: CPU45+30+15=90minute cap, separate narrow300s plus cumulative3420s GPU reservation caps. Sequential CPU+local-GPU upper reservation is152minutes, not≤90minutes all-in; officialH200cost remains unknown. Runtime/benefit not measured. Equal-resource private-barrier versus private-free isolates this source scheduling change while retaining compiler epilogue rescheduling as part of the effect. Whole serving graphs versus actual current PW38 and local controls remain mandatory; no additive score claim follows from source composition.
