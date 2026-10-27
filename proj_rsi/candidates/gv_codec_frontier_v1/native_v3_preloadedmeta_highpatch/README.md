# C11 descriptor preload + high-byte patch, three-arm discriminator

CPU/offline complete; GPU unrun. This changes the source meaningfully and
preserves matched C12 and v2 comparators. It is not an identical cleanup rerun.

The v2 attempt completed its numerics and timings but failed its terminal
allocated0 assertion with32MiB retained. External dispatcher cleanup was clean.
`../attempt01-analysis/` preserves both receipts and the unadmitted negative
timings (−5.408/−12.656us paired control-minus-v2). No old result is rewritten
or promoted. Actual per-case occupancy remained1/4 and local bytes0.

The strongest changed mechanism removes the per-unit **global metadata → record
copy** dependency. Each warp copies its entire KU×32B descriptor slab once,
commits, waits0 and synchronizes all32lanes before any record issue reads it.
Lane0 publishes the following unit's boundary in a separate16B pad; the final
matrix boundary comes from the host word count. That normal store and all async
copies are covered by the same warp synchronization. Slabs are immutable and
warp-private; no cross-warp publication is assumed.

The old ST1/ST2 record ring then uses landed shared offsets. It retains its
wait0 / wait1-until-final-wait0 protocol and sync before consumer reads. Four
repair steps finish before a slot is refilled. The descriptor slab and each
record slot have disjoint16B-aligned extents beyond the old reduction storage.
The high-byte patch applies identical XOR fields to two high-byte words before
the final four PRMTs. It changes no weight bits or arithmetic ordering.

| Cell | C12 / v2 / new registers | C12 / v2 / new dynamic shared bytes |
|---|---|---|
| Pooler M1 |64 /64 /64|17408 /20480 /28160|
| Unpooler M16 |110 /108 /96|5120 /6144 /8000|

All six kernels compile with0 stack/local/spills. Actual user-static/resource/
occupancy checks remain required; no occupancy gain is predicted. The larger
shared footprint is a real tradeoff. Four C12/v2 native PC/predicate/encoding
streams remain exactly equal to the frozen v2 module. The source inverse restores
new shared transport to the high-patch source and then to v2's arithmetic/input/
reduction/output source. Physical layout and capacities32/48 remain unchanged.

CPU ownership checks cover1280 unit-step generations, empty and exact-capacity
units, overflow rejection, both tail waits, matrix-end boundaries, disjoint warp
regions and two exact physical roundtrips. The separate high-field XOR proof
covers4096 cases and list composition by XOR associativity.

Unique format bytes remain48,323,952 /49,087,008. New duplicate logical boundary
words cost16380 /8188B (one4B word per warp except the final matrix boundary),
versus v2's262140B per matrix. These are source word references, not measured
cache sectors/HBM or an instruction-to-time prediction. The old repair/load
work remains substantial, so lower bytes alone do not justify promotion.

The driver reuses the reviewed two-cell protocol with three arms: current C12,
exact v2 and new source. Across each cell it retains F1 control endpoints and
24 same-seed triples, covering all six orders equally (eight visits per arm per
position). Total232 timed calls plus30 warmups,12own timed-output terminal
oracles and30retained-graph mutations. Each arm has private guarded X/output;
weights and read-only packs are shared and checked immutable. Full actual CUDA
fixture SHA, complete physical unpack/C12 reconstruction, finite/capacity/offset/
byte-saving gates precede timing. It never truncates repairs.

Cleanup now runs after graph drain/reset, function unwind and alias release.
An ATen C++ host hook records cached BLAS map entries/workspace size, clears
that cache, records entries after, then normal empty_cache runs. Allocated0
remains mandatory. The hook cannot run during timing or while graph consumers
are live. Before/after allocator bytes, Torch/CUDA versions and workspace config
are saved, so the32MiB hypothesis is tested rather than waived. The declarations
and map behavior are present in version-matched
[PyTorch2.8 CUDAContextLight](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cuda/CUDAContextLight.h)
and [cuBLAS handle source](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cuda/CublasHandlePool.cpp).
The CUDA host binding is staged for normal runner compilation; it has not been
loaded here. A successful new cleanup does not retroactively validate v2.

Proposed300s inner /360s outer, default Torch image, profiling/full prepare false.
The runtime captures actual source/SO/SASS, verifies all four live C12/v2 controls
before any launch and records all signed comparisons. Root owns review/admission,
GPU execution, DONE disposition and any subsequent integration.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../../ITERATION.md)
