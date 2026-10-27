# GO — projection-only GV53 keyed repair

Independent gv_codec_frontier focused review. Source239b530cf7eead0a2dbf0af5c29aea4ba02e449500eb80e45b255b361cf2f900;
driver1cef9a69cf7e97ca7d784619e12b4cd4afdb0bda3d480c9344b5007b57c0b87c;
runtime manifestef620956c543674d07be264386a20881231bdc8f38225dc065a4d06af89ecfc8;
frozen proposal49083648bddb3e1fc55ab0ea6985545dc808fb2177e318c577dc4e3ea268f51c.
GO for one120s-inner/180s-outer fixed pooler M1/M8 comparison, only muse_bf16_proj.

The changed predicate is exact for both matrix-half bits, without an h0-only
packing assumption: `(nx>>9)&0x3d = 4*((nx>>11)&15)+((nx>>9)&1)`.
Comparing this key to4*i is equivalent to old r==i && hm==0. Mask0x3d excludes
bit10, the within-register BF16 half selector, and target/sequence/XOR fields.
Thus hm1 remains ignored. Independently checked262144 raw words with varied high
bits and4194304 register predicates. Existing author field/cursor/native checks
also pass read-only; cursor, XOR magnitude/shift, target lane and load order are
unchanged. Key, mask and lane predicate are copied before the first output XOR,
retaining extended-asm input/output alias safety.

The exact whole-file/kernel inverse and174 unchanged original AST nodes pass.
The copied kernel is restricted to tuple1,1,1,2,16,8,1,1,0,3; production dispatch
further requires N4096/K8192/m1or8. Other actual shapes retain original dispatch.
Packing/CSR, input load/refill, HMMA and reduction/output arithmetic are reused.
The original native target matches archived PC/predicate/encoding. Candidate
removes three non-NOP instructions, retains360 aligned slots,64regs/0local/stack/
spills and P2R/SHFL/HMMA/LDG counts. This is codegen evidence, not a speed forecast.
An unchanged unused p1zero helper is emitted; only two repair targets execute.

Reviewed the existing harness delta and reran its CPU check. Two private guarded
X/output banks, normal/outlier/zero smoke, eight boundary columns, touched pooler
W/pack/CSR immutability, complete counter/scratch checks, eight own timed-output
terminal sites and20 retained-graph mutations remain.162 timed calls+20warmups
retain all AB/BA pairs and F1 endpoints. The unchanged codec's own reconstruction
check remains; its redundant extra full-matrix roundtrip was removed. Existing
codec/math/layout proof is reused rather than expanded. Runtime attributes are
recorded and require legal launch/residency only; no exact-register or relative
occupancy performance gate is added.

All graphs drain/reset before scope/closure aliases release. Target-verified
Torch2.8 CUDAContextLight BLAS cache clear then records map/allocator before/after,
requires an empty map and allocated0. The native host wrapper/JIT is staged for
the normal runner; it was not locally loaded in review. No GPU/compiler/SSH/
queue/git action or author-source mutation. No full prepare/profiler, policy or
official-score admission follows this GO. Preserve every signed result and price
any subsequent actual-policy comparison separately.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../ITERATION.md)
