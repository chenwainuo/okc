# DP + STMatrix composition: CPU/device feasibility only

One composition of two measured changes, pending independent review and root GPU
handoff. No production port, official slot, GPU timing or gain claim for this version.

Frozen DP source97dd5613 is the parent. DP runtime measured whole fresh-pack+main
F1=34.495998us against PX39.519999/STMX38.176000/incumbent24.095999;
paired gains=+5.152002/+3.904000/−10.591999us, wins21/21/0. Separate STMX
previously improved scalar-store PX by1.183998us F1/1.440004us paired. These are
different experiments; their gains need not add. DP is the best failed comparator.

Only source delta from DP: producer16scalarSTS become4
`stmatrix.sync.aligned.m8n8.x4.shared.b16`, plus kernel/cache name. Exact whole
CUDA restoration to DP97dd5613 proves all other code identical: 512threads,
producer parity, CSR skip, W6/B8, proxy fence, owner barriers3/4, publisher256/384,
Xpack, nativeA4 retired-slot schedule, cache policy, geometry and plain output.
No byte/layout/precision change. Each producer handles32stages and4warps;
`w=warp&3` gives the same mapping for either owner. Stage ownership is unchanged.

Store address-provider laneL, producer warpw, decode groupk:
`row=16w+8*((L>>3)&1)+(L&7)`, `chunk=2k+(L>>4)`,
`dst=Bslot+row*128+((chunk^(row&7))*16)`; registers `a[4k:4k+4]`.
PTX data laneD/registeri obtains row address from lane `8i+(D>>2)`,
plus word offset `4*(D&3)`. Substitution exactly matches original scalar stores.
[NVIDIA PTX12.8.1 stmatrix semantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
require full-warp converged execution and16B address alignment; weak memory means
retaining every writer's proxy fence and both128-thread owner barriers.

CPU proof covers all4096BF16/stage and all524288BF16 in the weight fixture,
all65536 storage patterns, escaped bits, full cross-lane routing/byte equality,
SW128 bijection/alignment/slot bounds, both ownership parities, inherited dense
CSR windows and missing-skip negative control. Xpack/inverse/1MiB all-bit map,
output mapping and A4 logical retirement proofs retained. No input values cached.

Local actual SM90a: main122/packer22registers, zero stack/spill/local, no
USETMAXREG/C75. Rounded main register pool128*512=65536; shared103600B.
Physical A4 SASS checker preserves strictD24 and rejects known badQ4/A2copy.
Store gate requires4unpredicatedSTSM.16.M88.4,16outputSTS,32XLDG128,16HGMMA,
noBFLY, MEMBAR+proxy fence then BOTH bar3/bar4 count128. Actual compiler lowers
owner branch to complementary predicates. The gate verifies same live predicate
with no intervening predicate writer; source proof preserves group membership.
First parser incorrectly required unpredicated BARs; evidence is retained in
checker-history/. Kernel source was unchanged by that parser repair.

Root-only bounded command after independent frozen-source GO:

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_dual_stmatrix/lab.py --json /okc/logs/operand-reverse-dual-stmatrix01.json
```

Inner240s/outer300s. Four-way actualpw36 incumbent / frozenDP / frozenSTMX /
combined. Three independent1MiB packed workspaces; every packed timed graph has
fresh pack+main, no setup input transform. Save actual loaded JIT SASS/hash BEFORE
checking each artifact and BEFORE any candidate math. All three A4 gates, new
and DP dual-barrier store forms, frozen STMX single-barrier store form must pass;
gate failure writes INVALID_ACTUAL_JIT_RETIREMENT_GATE and launches no math.
ActualDP/combined<=128regs/local0/roundedpool<=65536/maxThreads>=512/opt-in
occupancy>=1 are mandatory; no assumptions from CPU compilation alone.

All four: exact all4096Konehots/FP64random/tasknormal+outliers; all three packed:
all65536GPU storage bits, fresh workspace/output, independent current-X byte
oracle. All20actualtimed-graph mutation/output-poison checks and15packed byte
poisons retained. W/rawW/escape bytes immutable, counters0, graph reset mandatory.
Profiles verify incumbent actual selected path and every packed2node identity.
Timing uses21samples with before/after controls (minimum control median), plus
21rotated/reversed4-way paired rounds. Preserve every sample. Incumbent admission
requires>=0.5us improvement in BOTH F1 and paired medians, all validity gates.
Progress versus DP/STMX is reported independently. Nonadditive/negative result
requires residual bottleneck diagnosis; it does not close the entire family.

Prediction: fewer store/address instructions may advance each parity producer's
B publication. Same shared traffic; STSM port cost, barrier skew and additional
producer overlap can make gains nonadditive. Whole-version timings cannot prove
a unique PC cause. Root's separate DP counter diagnostic is independent; no new
profile or GPU work by this author.
