# GO — projection unpooler M16 lane-relative repair key

Independent gv_codec_frontier focused source/transport review. Candidate
713c46b37e50f71410ef4c7116f7fcb94bea71bd7b64c7f33d2a9e0d93e24f6c,
driver2b1598520ee759c336b129398f6584b57695f500bed8977b0432a8196223cd27,
binding6143e7477b58d4578aa61e9041c4cbf599245c3bc7c1057b18da8a384c05d963,
manifestb212cee6f9b11e97c480be47496406434a64e0762ae89521a3998f0b4c302ca7,
proposal d120d6f5c0732b722c940be087f62e0d090805ba72337ca90dab8dd9c0a29936.
GO for one90s-inner/150s-outer fixed muse_bf16_proj unpooler M16 comparison.

The masked field equals64*target+4*r+hm. Subtracting64*lane as uint32 gives a
key matching4*i iff target==lane, hm==0 and r==i. For target>lane the integer
value is at least64; for target<lane it is between−1984 and−3 and wraps far above
60. Neither can match0..60. The low field is at most61; hm1 remains odd and
ignored. Mask0x7fd excludes the register-half selector and sequence/XOR bits.
Both XOR halves and magnitude are unchanged. Key and mask are copied to PTX
locals before any output write, preserving input/output alias safety.

Independently checked the actual complete CUDA construction, whole-file inverse
and copied kernel inverse, not just metadata declarations.174 other AST nodes,
packing/CSR, cursor/SHFL, input loads, HMMA, reduction/output are unchanged.
The new template assertion and existing unpooler boolean restrict production
dispatch to `(1,2,2,1,4,16,1,1,2,1)`, N8192/K4096/m16. The separate spilled
pooler candidate is excluded from this source/dispatch. Existing pinned semantic
binary returns524288 predicates/256 boundary chains; it was run without invoking
the mutating verify.py or recompiling/writing author artifacts.

The native old control matches the archived incumbent PC/predicate/128-bit
encoding,2080 instructions. New1992 instructions,110→108registers,0local/stack;
P2R16/SHFL17/HMMA128/LDG113 unchanged. Counts/resources do not forecast a gain.

Reviewed the one-cell harness/binding: exact grid1×512×1,128threads,5120B dynamic
shared, packed50331648B and CSR65536words. Private guarded X/output, current-X
normal/outlier/zero checks and16 boundary columns in one M16 launch per arm,
own timed terminals, immutable weight/pack/CSR and complete touched counter/
scratch checks remain. Existing codec/layout/math proof is reused; redundant
external full roundtrip is removed while the packer's own check remains.
81 timed calls+10warmups,24 AB/BA pairs,4 own terminal sites and10 retained-graph
mutations preserve all signs/F1 endpoints. Runtime attributes are recorded;
only declared launch/resource legality is required, not relative occupancy gain.

Known target Torch2.8 ATen BLAS clear runs only after all graphs drain/reset and
run-scope aliases release. Before/after cache-map/allocator evidence is retained;
empty map and allocated0 remain required. CPU harness check and source/manifest/
proposal pins pass. No GPU/compiler/SSH/queue/git operation or author-source
mutation by reviewer. Host JIT/load and runtime numerics remain pending the
bounded root-run trial. No fresh-selection, policy or official-score claim.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../ITERATION.md)
