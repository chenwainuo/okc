# RQ11 wordplanes v1: exact format, bounded escapes, insufficient standalone ROI

2026-10-27. CPU/offline-compiler investigation only. No GPU initialization,
extension load, prepare, benchmark, official check or submission. Own artifacts:
`candidates/rq11_codec/`. Stock sources and all global ledgers remain unchanged.

**Finding:** a legal 11-bit stream with coalesced aligned loads is straightforward,
but this concrete decoder costs substantially more instructions and its serial
escape walker handles roughly 13–25 entries per warp-unit on synthetic weights.
The raw 8.333% W-byte reduction becomes only 1.59–4.97% after fixed-stride escape
storage in this CPU sample. This is a retained feasibility prototype, not a
serving candidate or speed win. Park integration of this version pending a
specific causal test; do not retire the lossless-codec family.

## Authority and scope

Read `ITERATION.md`, `protocol.json` rules, frozen rs32/pw36 `rq_dec`, `rq_unit`,
`rq_fix`, `_c12`, `_rqpack`, `_rqesc`, `RqC` and producer/LDS code. Prior evidence:
`audits/codec-study.md` and `audits/r6-lineage.md` fixed-width analysis. Those
already identify 11-bit escape-walk cost; this version adds an actual exact
format, overflow/finite gates, aligned wordplane layout and offline SASS evidence.
It does not repeat generic hardware compression/byteplane experiments.

Authoritative official baseline remains rs32 job `fb548b5dc6f04696`,
3200.6150084373335 us. The 3000 us milestone remains unmet. Active older “3x current
benchmark” metadata has an unspecified historical baseline, as separately
documented in `sol16-score-opportunity.md`; this audit does not redefine that
goal or claim achievement. Even eliminating the raw one-bit W saving overhead
does not make an 11/12-format change a 3x design.

`native-x-01/report.md` independently retains local H100 Native-X paired median
31.856 us versus actual pw36 s4s5hz KS2 23.968 us. These are one local comparison,
not official H200 board numbers, and Native-X is not the selected speed baseline.

## Concrete representation and exactness

One warp-unit is stock's 16 rows × K64: 1024 BF16 words, represented as
`[q=4,lane=32,v=8]`. Write each BF16 word as low8, h7 (bits14:8), sign. Choose the
weight-only best contiguous four-value h7 window `[B,B+3]`, `0 <= B <= 124`.
The code is `(h7-B) | (sign<<2)` for in-window values, otherwise `0 | (sign<<2)`.
Keep every low byte verbatim; an outlier gets XOR7 correction `h7 ^ B`.

The decoder uses an unsigned high-byte table `[B,B+1,B+2,B+3]` plus the same table
OR `0x80808080`. Four packed 3-bit codes spread into PRMT selector nibbles;
selector bit2 chooses the signed table, bits1:0 choose the offset. Final stock
PRMTs interleave high and low bytes into the same 16 A registers in the same order.
XOR fix-up touches only the original h7 field. Sign, exponent bit0 in low8,
mantissa bits and payloads are never rounded, quantized or inferred from inputs.

`codec.py` proves 8,192,000 original/decoded cases: **all 65,536 BF16 storage
patterns × all 125 legal windows**. This includes both signed zeros, every
subnormal, low magnitude/outlier, Inf and every NaN payload. It separately proves
all4096 selector expansion patterns and393216 entry coordinate/XOR/boundary-seq
patterns. A CPU simulation of the actual register-half XOR walker reconstructs
all65536 storage patterns and crosses every 32-entry cache-window boundary.
All tests pass (`cpu-results.json`, `cpu-proof.log`).

Preserving a bit pattern is distinct from admitting nonfinite model weights:
`serving_admissible` requires exact bit equality and finite original/decoded
BF16 weights, including an explicit all-exponent-ones rejection. Codec proof
accepts all patterns; serving admission rejects nonfinite tensors. Four finite/
mutation gate tests pass. Full output finite/relative-L2/one-hot checks remain
mandatory before activation and have not run here.

## Escape capacity is explicit, never silently truncated

Reuse stock RQ's 32-bit entry:
`seq<<17 | lane<<12 | reg<<8 | half<<7 | xor7`,
`reg=q*4+v//2`, `half=v%2`. The layout allocates one fixed-stride list per
`(KS,NB,math warp)`, with 32-entry register-prefetch windows. Importantly, stock
RQ's capacity is **dynamic**, not a hard32 limit:
`CAP=round_up(max_list_length+1,32)`. All lists terminate in a sentinel; append
32 trailing sentinels so the last window's lane-spread prefetch is safe.

The prototype validates every field before encoding and reserves seq32767,
accepting seq0..32766 only, so no valid entry aliases `0xffffffff`. `csr` rejects
entries outside that range and returns None if the explicitly bounded allocation
limit cannot fit `CAP`. An integration must fall back to stock on every such
rejection, before narrowing counts/offsets to CUDA integers or allocating. The
modeled large sizes fit this sequence bound, but a general caller must check it.
Tests cover empty lists, counts1/31/32/33/63/64/1024, guard storage, and capacity
overflow. Duplicated entries are not emitted by `pack`; each source position has
at most one fix. No statistical escape rate is used as a correctness limit.

## Physical layout and stage accounting

Warp payload: two stock512B low planes plus three128B header wordplanes =1408B.
For each lane, concatenate the four24-bit q headers into96bits, split into three
u32s and transpose `[word=3,lane=32]`. Thus each lane performs two aligned
LDS128 low loads and three LDS32 header loads. Each header instruction accesses
32 consecutive words, one per bank. It avoids an invalid assumption that a
44B per-lane record permits aligned LDS128: consecutive44B records rotate
alignment, so that approach cannot retain the stock vector loads.

The stock128-row K64 stage is8×1536=12288B. New:8×1408=11264B. HALF64-row stage
is4×1408=**5632B**, versus6144B. Raw saving512B/HALF stage (8.333%). Both new stage
sizes and all warp/plane offsets are multiples of128B; every low vector is
16B aligned and each header word4B aligned. The actual W mover is linear
`cp.async.bulk`, not the X tensor-map swizzle. It needs source/destination
alignment and transfer-byte multiples, which these sizes preserve. The 5632B
stride alternates512B modulo1024; it is not itself1024B aligned. W is decoded
into A registers and does not use an SW128 WGMMA descriptor, so no W descriptor
requires that larger alignment. X tensor maps and descriptors remain separate.

Minimal source edits for a future integration are bounded: `_rqpack` physical
tile1408/row-block11264; W producer source stride11264 and HALF offset5632;
`RqC.WB`; `wl` warp stride1408; LDS header offsets1024/1152/1280 with lane*4;
new `rq_unit`; unchanged escape entry/walk. Full and early producers, resident
controls, barrier expect_tx and profiler byte assumptions must all be changed
consistently. This audit does not claim that a partial string replacement works.

Additional escape allocation is separate from ring bytes. Exact physical total:
`NB*KC*11264 + 4*(KS*NB*8*CAP+32)` bytes, plus the existing tensor/runtime metadata.
For HALF grid the parent128-row allocation remains shared by both half CTAs.
The four-byte decode table replaces the old four-byte table. Compare against
`NB*KC*12288 + old_escape_bytes`, rather than raw11/12 alone.

## CPU distribution evidence and compiler evidence

These are synthetic NumPy PCG64 seed5151 float32 normal samples rounded RNE to
BF16, N=1048576 per K, scaled by1/sqrt(K). They are **not actual GPU fixture
weights**, nor identical to an earlier Torch/CPU seed5151 entropy sample.
Fixed-stride accounting uses four synthetic groups of256 warp-units each;
actual KS/NB grouping can change max-list padding materially.

| K | escape % | 11+32p bits | incl synthetic fixed stride/guard bits | net bytes saved vs raw12 |
|---|---:|---:|---:|---:|
|1024|1.2372|11.39590|11.40332|4.972%|
|2048|1.7363|11.55560|11.57129|3.573%|
|3072|2.1544|11.68939|11.69629|2.531%|
|4096|2.4540|11.78528|11.80957|1.587%|
|8192|1.3499|11.43198|11.44238|4.647%|

This comparison charges all new CSR bytes against raw12, so it is conservative
relative to stock's nonzero old CSR, but actual old/new bytes must be counted.
Per-unit escape means range12.67–25.13; these require that many serial warp-walk
iterations in this unmodified `rq_fix`. Aggregate tails, not mean alone, determine
prefetch windows. No arithmetic-opcount-to-time conversion is assumed.

Offline compiler: CUDA12.8 V12.8.93, `nvcc -std=c++17 -arch=sm_90a -O3
-Xptxas=-v -cubin decode_probe.cu`. This creates a cubin only; it never loads or
executes. `decode_probe.sass` and `sass-counts.json` retain the results.

| Probe | all static instructions incl padding/prologue/stores | registers | spills | shared loads |
|---|---:|---:|---:|---|
|stock|80|32|0|3×LDS128|
|rq11|128|30|0|2×LDS128+3×LDS32|

Stock source has exactly9 decode operations per8 weights:2 masks,2 additions,
1 rotate,4 PRMT. The unit calls it4times, hence36 decode ALU operations/32weights.
Stock SASS agrees:16PRMT+8VIADD+8decodeLOP3+4SHF. RQ11 adds eight PRMT table
lookups, selector spreading and two cross-word funnel extracts. The48-instruction
whole-probe delta includes address/table load differences; it is not an isolated
fused-kernel latency prediction. The standalone register counts exclude WGMMA,
ring/pipeline and escape registers. Escape walker cost is excluded from both
probes, and actual11bit escape frequency would make it materially different.
Shared bytes in these kernels are intentionally uninitialized: they are compile
probes with no host launcher, not numerical or benchmark kernels.

## Smallest next causal test, only if root judges ROI worthwhile

First obtain actual weight-only h7 histogram and exact old/new CAP allocation for
the **selected incumbent geometry**, without changing inputs or math. If new
bytes are >= old bytes, retain this version as no-byte-gain for that fixture and
test a different representation rather than timing a known larger stream.

Then root alone may run a bounded producer control keeping source, graph and
input/output ownership frozen: stock6144/12288B versus aligned5632/11264B mover,
with a read sink and identical stage count. It estimates only the attainable
transfer reduction, not a codec gain. A second decode-only resident test compares
the supplied decoders without escapes; full correctness still requires real
packed weights and escape handling. Finally integrate all three changes only if
those bounds justify it, preserve the exact WGMMA FP32 operation/output order,
and compare actual selected incumbent, predecessor and new candidate in one run.
Keep current inputs fresh; pack only weights during prepare; preserve router
FP32 hi+lo behavior and do not port this BF16 format to FP32 weights by truncation.

All gates remain: exact W unpack proof, original W/packed/CSR immutability,
nonfinite/overflow rejection, numerical and one-hot probes, graph replay and
input mutation, complete barrier lifetime and cleanup. No promotion, score
projection, actual compression ratio or serving speed claim follows this work.
