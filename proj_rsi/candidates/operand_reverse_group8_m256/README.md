# M256 coverage port: CPU pass, runtime unrun

Minimal token tiling of frozen group8, full M256/N8192/K4096 only. Main grid
(128,2), pack grid (256,2), NT128 and descriptor outer M256/box128. Every call
writes the complete fresh2MiB XP before the main launch on the same stream.
Each y tile owns its XP1MiB and output rows128; weights are shared immutable.
All input/output offsets fit32bits. Main has no counter/atomic access.

Generated source `b9dedf18be83f32dcfa82f19674bb31dea0e32e749af74c586852f0ffca503a0`.
Parent source `e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`;
parent manifest `2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7`.
Finite reverse edits restore that exact full source, including W codec,
CSR/producers, two A32 banks, all256K16 MMA order and shared epilogue.

CPU admission passed: main122 and pack17 registers, no stack/spills/local/C75,
uniform rounded128*512=65536registers,103600B SMEM. The inherited generalized
physical checker discovers both disjoint32word A banks and D, verifies fence
coverage and async retirement, and rejects retained premature-overwrite and
synthetic missing/predicated-fence/A/D corruption controls. Scalar producer
store-form checker passes. Exhaustive models cover all524288 packed u32
source/destination pairs and consumer vectors, all2,097,152 output cells,
all65536 BF16 storage patterns, and identical complete weight sequences for
both token tiles. Inherited detailed W bit map, shared-output bijection and
per-tile schedule proof pass. This is CPU evidence, not numerical/runtime
validation or occupancy measured on a GPU.

Root-only lab prerequisites are a corrected valid runtime trial containing
frozen G8 (a GS four-arm trial is eligible) plus root/critic GO. Runtime entry
is explicitly HOLD until root reviews that canonical result schema and freezes
a new lab/manifest binding its strict status and final-output records. The
current lab intentionally refuses to load extensions while HOLD is present. The author created no CUDA context, loaded no extension and ran no GPU work.
Proposed invocation, only after the prerequisites:

```sh
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_group8_m256/lab.py \
  --json /okc/logs/group8-m256-01.json \
  --parent-validation-result /okc/logs/operand-reverse-group8-01.json \
  --review-go REVIEW_EVIDENCE_ID
```

Lab compares prepared frozen pw36 actual M256 `kernel_fn`, M256 pack+main,
and forward/reverse ordered pairs of full parent M128 calls on disjoint halves.
Both parent controls own separate2MiB workspaces and output and time both
packs+main calls (four nodes), versus new two-node M256. It verifies loaded
JIT SASS/resource/store-form before first math, all4096 onehot columns,
FP64/task normal/outliers, all65536 storage-bit pack patterns and fresh XP/output.
Five mutations times four actual timing graphs give20 graph checks, with NaN
output poison, packed workspace poison/full-byte oracle and immutable X,
**before timing**. Immediately after timing, every actual final timed output and
full2MiB XP is checked against the recorded unchanged last input fixture, before
any graph replay/profile/mutation; raw samples are saved before this gate. The
parent runtime prerequisite must require corrected final-timed-output evidence.
Final records are explicit per-policy lists with PASS_BEFORE_ANY_REPLAY_OR_MUTATION,
fixture seed/input SHA256, full XP match/Y oracle/X immutable/W-pack-escape
immutable/cnt0 fields.
All graphs reset; input/W/packedW/esc/counter gates required.
Two predeclared estimators retain all21samples/order rotations and both
comparators; no favorable-method selection. The before/after and paired
`*_m256_gain_us` fields measure new M256 gain. Both methods must exceed0.5us
against actual incumbent before any integration review. Invalid attempts retain
raw artifacts and earn no performance admission. Proposed runtime budget is
inner300s/outer360s; root may decline if remaining bounded resource budget is
insufficient. No official slot or production integration requested here.

Causal limits: M256 doubles CTA count128→256, duplicated raw W copies48→96MiB
and decoded shared-B work; per-CTA policy/resources remain fixed. This can add
another scheduling wave, W/cache capacity pressure and repeated CSR decode.
New fresh pack reads+writes2MiB each; main XP logical requests double to256MiB.
Comparing to two M128 calls confounds fewer launches with overlap/interleaving
of token tiles, changed scheduling and cache reuse. Reversing parent tile order
checks one order effect, not all cache confounds. Pack registers22→17 are a
compiler consequence of the extra uniform tile offsets, not a measured win.
No input cache, free pack, post-timing validation repair or nearby-M port.

Official conditional arithmetic uses only best rs32 jobfb548 score
3200.6150084373335 and complete verify32 total86.58484871917094us. Unpooler256
contributes24.693354263625714us. A genuine complete-shape saving0.5/1/2/3us,
all else fixed, yields score3197.527148/3194.424305/3188.173027/3181.859854.
Both U128 and U256 saving1/2/3us each yields3186.489055/3172.233727/3157.845952.
U256 alone requires27.867180us to reach3000, exceeding its entire contribution;
U128 alone also cannot. Together their impossible zero-time ceiling is2864.718291
and equal genuine savings would need13.368764us EACH. These are conditional
same-job arithmetic opportunities, not H100/H200 forecasts or measured gains.
G8 and this port have no demonstrated incumbent speedup at the time of freeze.
