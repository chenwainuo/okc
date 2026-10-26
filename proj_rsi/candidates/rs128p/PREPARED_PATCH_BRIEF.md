# Prepared escape footer: separate, unimplemented hypothesis

This is **not** an extension of the failed producer-written table sweep. That mechanism is closed. No CUDA source, GPU preparer, compile or timing run is implemented here. The only code is a CPU layout model.

## Measured reason to consider it

Clean P4 takeover-pt01: producer table reduces WG0 steady loop955->937clocks (pooler) and1124->994(qkv), but raises median first-WGMMA2.05->3.58us. Graph results regress+1.760/+1.664us. Source shows an extra runtime escape-list prefill, shared table construction and barrier arrival. Static weight-derived metadata can remove those operations entirely; it is allowed preparation of immutable weights, not input-derived caching.

## Concrete layout and kernel boundary

For each original global `(nb, k64_chunk)`:

- Original12,288-byte packed payload remains byte-identical and occupies the first bytes.
- Append one128-byte footer, making a12,416-byte,128-byte-aligned stage. Overhead1.04167%.
- Footer bytes0..31: eight `uint32` lane masks, one per math warp.
- Bytes32..63: eight `uint32` base-record offsets.
- Bytes64..127: eight sparse lane records; each record has two32-bit `(xor_bits | register_index)` words.
- Records are ordered by warp then lane. For an active lane: `index=base[warp]+popc(mask & lanemask_lt)`; word1=0 means absent.
- Multiple escaped halves of one register are combined by XOR. More than8 affected lanes in a stage or more than2 distinct patched registers in one lane must disable this new path for the layer and keep the incumbent. Never truncate.

A future GPU preparer derives this footer from existing `esc6(tile, chunk, q, lane, value, exponent_xor)` and combines it with the already-packed stream. It does not inspect activations. A new source flag changes the producer's WB stride/copy size to12,416, removes the runtime escape-list prefill and extra table/barrier producer, and loads this footer alongside existing W LDS. No global pointer chase on the math-to-WGMMA critical path. Original warp payload offsets remain valid. The footer is global-chunk-addressed, so KS4/7/2 and uneven K partitions share the same pack.

The future source must retain the old path for every unsupported/overflow layer; preserve exact one-hot outputs, all normal/outlier checks, counter reset and graph replay. Compilation/baking must fit the current parent's prepare order and budget.

## CPU layout proof

`prepared_patch_layout_proof.py` is pure Python, with no Torch/CUDA dependency. Output: `takeover/prepared_patch_layout_proof.json`.

-24,576 single-escape coordinate/XOR cases match an independent direct register-XOR model.
-4 mixed cases cover both BF16 halves, two registers, all8 record slots, lane31 and empty stages.
-3 overflow/duplicate cases explicitly return fallback.
-20 complete/uneven K-split ownership cases cover qkv/pooler/unpooler/o shapes.
-128-byte stage alignment preserved.

This is an indexing/encoding proof only. It does not prove GPU packing, actual eligibility rate, TMA arithmetic, numerical outputs or performance.

## ROI and a one-run kill rule

The new data costs~0.524MB on the50.3MB pooler stream and~0.295MB on28.3MB qkv: ideal additional stream time~0.19/~0.11us at2.8TB/s H100 (or~0.14/~0.08us at3.8TB/s H200). This is a lower cost estimate, not actual achieved bandwidth. Sparse footer loads/popc/predication add instructions.

Observed no-escape upper references conflict in size: graph removes~0.672us pooler /~0.800us qkv, while CUPTI median removes only~0.384/~0.223us. Subtracting ideal footer bytes leaves little H100 headroom under the0.3us kill bar, especially qkv. Eliminating the original early escape-list global preload might save further startup, but that portion is unmeasured. Do not treat resident escape cost or a no-escape probe as an attainable speedup.

Recommendation: **conditional, small ROI**. Only one tightly scoped root-admitted prototype is justified if the head wants to test the measured startup hypothesis. It is lower priority than completing pw36/latency comparisons and should not grow into a new rewrite. No automatic GPU compile/run is authorized by this brief.

Preregistered observations for such a prototype:

1. Metadata overflow rate, pack/unpack bit exactness, prepare seconds and added bytes.
2. First-WGMMA startup <= shipped2.05us; no extra barrier arrival or runtime escape-list walk.
3. Stage clocks and registers/spills next to the shipped row.
4. Same-run graph and in-prepare gain>=0.3us against the actual shipped rq; warm/discard baseline timing and F1 pre/post checks.
5. Kill if it does not clear the bar after this one prototype, or if any correctness/fallback/resource gate fails. No scored slot for a microprobe-only win.
