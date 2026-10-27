# Bounded shared decode-table experiment (CPU only)

Candidate for actual-JIT representation gate and one M8 stage screen: `full_table/`.
This is a changed producer decoder, not a timing result. Exactly two offline device
functions instantiate T8 with gather divisors 1 and 8. The local adapter forces T8;
it is intended only for this bounded experiment, not general serving admission.
No original/frozen source was edited. No GPU, remote, API, queue, or git used.

Read `../ss_pipeline_v2/README.md` and its `v1-scale-counterexample.json` first:
v1 lost raw FP8 subnormal 0x01. The new table preserves canonical v2 BF16 weight
products, including signed zeros, negative scales and all FP8 subnormals.
NaNs remain NaNs and are outside finite numeric admission. Global stays after the
FP32 reduction and BF16 projection cast. Folding global into weight lookup is
invalid: a=[1,1], w=[1,2^-8], gs=1.5 yields canonical 1.5, folded 1.5078125.
Thus an exact table needs no expert dimension: 16 codes ×256 scales ×2B =8KB
shared by every expert. A global-dependent 8KB/expert table would be incorrect.

`build.py` creates the first 2KB magnitude-table iteration: 8 magnitude codes ×128
scale magnitudes, signs restored with XOR. `full_table_codegen.py` derives the
second 8KB full-table iteration and removes sign restoration from the hot path.
Both table contents are host-generated without CUDA. Each valid CTA coalescently
copies its common device-global table into private shared memory using all256
threads, then reuses the existing initialization barrier. Every produced BF16
weight uses one shared U16 lookup. No stage synchronization, matrix descriptors,
WGMMA, input gather, route behavior, BF16 boundary, global epilogue or serving
method is changed. Shared table occupies space after the ring and four barriers.

Offline CUDA12.8 codegen, T8:

| Variant | DIV8 up instructions / LOP3 / regs | DIV1 down instructions / LOP3 / regs | Dynamic shared bytes |
|---|---|---|---|
| 2KB magnitude |1064 /180 /40|1208 /190 /64|21536|
| 8KB full |944 /99 /40|1080 /109 /60|27680|

DIV8 gathers the up projection, DIV1 the down projection. Original actual SASS
from `tools/moe_ss_serving_pilot_v1/actual-analysis/analysis.json` reports up
1272instructions/215LOP3/32regs and down1416instructions/224LOP3/48regs. The
full table reduces static instruction counts but increases register usage in both
projections; actual loaded resource gates must be repeated. Each candidate
function still has20 static HGMMA across5 wait0 groups and5 reader barriers.
There are32 static LDS.U16 decoder loads; no spills/stack in either variant.
These are whole-function static counts, not dynamic cycles or throughput.

The full table saves more scalar instructions but requires8KB init reads and
shared space per CTA. Both variants add262144 logical shared U16 reads =512KB
per full K4096 valid CTA, in addition to unchanged operand stores. Direct global
lookup would add512KB lookup payload per CTA (plus cache/transaction effects),
versus8KB initial copy for this variant. Constant-memory random indexing risks
warp address serialization. A register-resident full table would require2048
32-bit registers per lane for replicated storage; distributing it needs dynamic
selection/shuffles and no longer gives free lookup. No claim is made that any
lookup option is faster.

`cpu-proof.json` gives a bank service lower-bound model: distinct 32-bit words per
bank, broadcasts collapsed. With seed7 and1000 random per-row scales/per-lane
codes, magnitude table mean2.392 rounds/max5; full table mean2.808/max5. Same
scale is1; adversarial scales separated by8 and same q need8. This is an indexing
model, not measured GPU bank conflicts or a performance prediction.

CPU verification compiles and executes each exact C++ lookup helper and compares
all4064 finite code/scale products bitwise against independent Torch FP8→BF16
multiplication. Both signed zero and scale signs/subnormals are included;32 NaN
cases check classification only. It also checks unchanged matrix and downstream
source, reuses v2 layout/protocol/routing proofs, and verifies compiled SASS has
no accumulator overwrite before wait0 and a matching reader barrier. Hardware
shared addressing and end-to-end numerics still need actual-JIT checks.

Reproduce without a CUDA context:

```
python3 build.py
python3 full_table_codegen.py
python3 compile_cpu.py full_table
python3 cpu_proof.py
```

`compile_cpu.py` additionally compiles the full Torch host wrapper offline and
writes its SASS. It never loads an extension or initializes CUDA.

Next minimal GPU experiment (root owns admission/execution):
1. Compile `full_table/adapter.py` through the existing JIT path, check the two
   actual loaded function streams/resources/wait0 lifetime gates.
2. Onehot representation checks with all finite scales/sign/zero/subnormal
   cases; exact stage oracle at T8 with DIV8 andDIV1. Retain global-after-cast.
3. Paired same-input, same-route M8 stage timing against original v2 decoder,
   retaining zeroing and wrapper behavior; same grid, no mapped-grid change.
   Time up and down separately to identify whether LDS/init cost wins or loses.
4. Only a winning, numerically passing stage merits a full-serving M8 pilot.
No broad shape sweep or claimed full-expert speedup is authorized by this artifact.
