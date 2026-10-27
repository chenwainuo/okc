# PW42 v4: actual-metadata guarded empty-escape GV

CPU-feasible research source over immutable PW38 source29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e.
No GPU/context/extension load/prepare/API/queue operation occurred. PW40 and all
parents stay frozen. No gain, actual empty prevalence, selection or cumulative
policy result is claimed. The motivating opportunity is
`pw38.next.gv-empty-escape-specialization` in the retained `opportunity.json`.
Its rank2 is conditional budget arithmetic, not a live scheduler priority.

## Cost quoted before implementation

Author planning estimate:35CPUminutes implementation/proofs +20independent
review +10analysis =65CPUminutes. The common cap is separately45/30/15=90.
Expected future wall-time estimates are180s narrow +2400s cumulative =2580s,
including transfer/JIT/prepare/correctness/timing/analysis/cleanup. Separate
reservation caps are300s+3420s=3720s. These are planning estimates/reservations,
not observed runtime, kernel time, official H200 cost or dispatch authorization.
Official cost and attainable gain remain unknown. See cost-quote.json.

## Lessons retrieved before implementation

Queried the read-only registry and read the exact ranked opportunity. Applied
immutable lesson records are bound in learning-proof.json:

- counter-contract-not-name@1,6f1e6d11…: counters/workspaces follow actual selected
physical storage semantics, never a blanket named-counter-zero rule. Approved
S1GV targets have no active global counter protocol; the lab preserves exact
nonzero/dormant bytes and aliases, including NaN payloads.
- rq11-full-residency-ledger@2,d89e8697…: CPU fixture statistics do not establish
CUDA weight escape prevalence, residency or benefit. This implementation never
uses a CPU histogram to omit a correction. Actual CUDA first-sentinel prevalence
is the first future discriminator, explicitly unknown here.
- prepare-selection-is-experiment@2,78c2ea94…: making an option available is not
an actual selected-policy gain. Keep every existing option and measure actual
prepare/PICK, headroom and all30shapes before a cumulative claim.

No cross-kernel latency fractions, H100→H200 transfer, fixed tiny-gain cutoff or
added score forecast is imported from those lessons.

## Narrow source mechanism and exact fallback

Only these two old geometries receive optional guarded twins:

| Existing target | Threads | BaseV | New host traitV |
| --- | ---: | ---: | ---: |
| GV(1,1,1,2,16,8,1,1,2,1) |1024|1|1073741825|
| GV(1,1,1,1,16,8,1,1,2,8193)|512|8193|1073750017|

The old generic GV kernel is retained VERBATIM. All95old GVI IDs retain their
prefix/order; only two guarded instances are appended. Bit30 is a checked-kernel
trait, disjoint from old cache, launch-bound and counter bits. The host gvl
pointer selects `gv_empty_guard` only for the new trait and strips it before
instantiating the device kernel. All old pointers still select old gv. Exact
parent inverse, unchanged named functions/constants and metadata AST proofs
protect every other GPU definition and policy option. The two targeted old mains
also have exact actual CPU PC/predicate/instruction/operand/branch-target SASS;
other93oldGV instantiations are source-preserved, not independently recompiled
in this narrow package.

New contenders are offered only for layerqkv_gate,bucketM1/8, the exact two
geometries, a valid C12pack/list and at least one first-entry sentinel in the
ACTUAL prepared escaped tensor. This exact `.any()` metadata check may synchronize
four times during QKV1/8 prepare; its cost is unmeasured. Zeroempty/None/non-C12
and all other shape/geometry cases retain old choices. No winner is forced;
original validation/bake/margins/prepare240s guards remain. Extra compile/fullK
validation and RNG/timing context can change later selections/headroom; full
policy checks must retain those effects rather than infer unchanged picks.

At device entry all32lanes load their actual32word per-warp list, then lane0's
first word is broadcast with the existing full-mask shuffle. Only exact
UINT32_MAX enters an escape-free templated compute path. Nonempty warps execute
the old repair loops with every XOR, original unit order and original arithmetic.
Mixed empty/nonempty warps execute their own paths and rejoin BEFORE the same
CTA shared reduction/barrier/output sequence. No CTA barrier is inside either
approved compute branch. Every currentX value is still read and multiplied on
every call/replay. No outputs, input-value decisions or benchmark answers are
cached; only legitimate prepared weight metadata is inspected.

V4 scopes startup/compute locals inside templated branches and rejoins four
FP32 words before the original shared reduction. This avoids the retained v1
address spill. It preserves per-unit decode/MMA/load order and arithmetic, but
startup metadata/address scheduling can differ from generic. A future gain
would be an observed guarded-version effect, not isolated proof that one removed
instruction caused it. The generic control remains separately exact.

## Prefix, boundary and bit proofs

Parent `_gvesc` is unchanged. It initializes every32word group to sentinel,
sorts actual escape records by group×4096+unit and fills ranks0..length−1.
Overflow length33 returnsNone. Hence firstsentinel iff zeroentries; no sampled
frequency or assumption about later words is needed. Full32 lists need no tail
sentinel: ep32 synthesizes UINT32_MAX without a33rdword load. Valid sequences
are0..7 for KU8/RT1; sentinel sequence4095 cannot be a legal payload. Matrix h1,
reserved bits, zero-XOR payloads, holes and unsorted sequences are rejected by
the lab's complete actual-list validator.

CPU proofs execute the unchanged encoder on synthetic tensors, prove the exact
2304group/18432tilechunk bijection for both WR values, all1024BF16 per-tilechunk
fragment positions, total18874368weight coordinates, boundary groups, prefix
lengths0/1/2/15/31/32/overflow33, duplicate XORs and last-unit full32 end. They
reject sentinel-hole, unsorted/reserved/out-of-unit/matrix/zero-XOR and deliberate
forced-empty-on-nonempty counterexamples. Generic and guarded correction results
match arbitrary32bit decoded registers. Unchanged C12 encode/decode reconstructs
all65536synthetic BF16 storage patterns, including NaN/sign payloads. This is
codec/protocol evidence only, not a real CUDA weight prevalence measurement.
OddM1/2/3/5/7/8 input clamp and exact output-coordinate coverage are proved.

## Compiler/resource and physical evidence

Local CUDA12.8 sm90a,O3,lineinfo,C++17 compiles both guarded and generic targets.
All four use64registers,zero stack/spill stores/spill loads. Generic controls
match original PW38 at every PC/predicate/instruction/operand/control target.
V4 host eligibility did not change a single byte of the retained v3 device
translation unit, so the successful codegen is reused exactly; no inference
from a different kernel source. Offline host gvl trait-pointer instantiation
also produces those same four main streams without a CUDA context/API call.

The512thread target retains64register/two-CTA eligibility;1024thread target
retains its64register65536pool bound. Actual opt-in shared occupancy must still
be queried in the authorized trial. Dynamic shared is8704B/17408B respectively,
unchanged from each generic geometry. Both branches have32HMMA; empty has no
repair shuffle/branch, nonempty retains8static repair SHFL. Old generic empty
warps already skip the repair-body shuffles; the proposed saving is repeated
next-sequence tests/control, not eight dynamically executed repair shuffles. Both paths rejoin
one common CTA barrier, and there are no global counter atomics for S1.

Guarded static code is1528/1536instructions (24448/24576B by16BPC strides) versus
1152generic (18432B), about33% larger. This is an explicit instruction-cache/code
cost risk; static removal is not a latency or speedup measurement. Exact actual
loaded SO streams/resources must match canonical gates before candidate math.

## Retained versions and failures

- v1 full artifacts retained: original generic has zero spill, but both guarded
loop-joined twins spilled a64bit W-base address (8Bstack/stores,56Bloads) at the
64register cap. No GPU attempt or timing admission.
- v2 source/compiler failure retained: compute scoping moved tk inside a lambda,
while an unreachable generic suffix still needed its name declared. No cubin
was produced for that failed source; no stale output is represented as v2.
- v3 scoped codegen succeeds at64/zero spill. Its whole source/cubins/SASS/logs
are retained before v4's actual-metadata host eligibility was added.
- A sibling-module import error in the physical checker is retained separately;
only checker binding changed. No failed version was erased or treated as a
family-wide kill.

## Minimal frozen next lab

`lab.py` is root-only and does not import Torch/create context at module import.
It first verifies frozen files, compiles the separately retained four-arm CUDA
source, dumps actual SO and checks exact canonical streams/resources. It then
prepares UNTOUCHED actual PW38, rejects an unsupported or silently changed/fallback
selected route and verifies its actual selected target JIT. Actual CUDA weights
are C12packed/unpacked exactly; actual fixed-group lists are fully validated and
all2304first entries counted BEFORE candidate math/timing. Exact zeroempty across
both geometries exits as VALID_METADATA_DIAGNOSTIC_EMPTY_PATH_UNUSED_NO_TIMING,
without speed admission; rare-positive lists have no arbitrary cutoff.

Four fixed arms (generic/guarded for each geometry) share the same actual W,
CSR and currentX pointers. The actual PW38 selected policy is a separate fifth
arm. Reusable output buffers and currentX are allocated once perM and swapped
between each matched pair in a second role phase. This retains strict separate
terminal outputs while controlling output-address roles. Actual selected policy
has its real prepared/capture context, never a forced artificial comparator.

Fixed-arm validation covers all8192basis columns exactly, independentFP64,
normal/outlier task checks at all six odd/evenM cases. Fresh current input and
output/NaN probes, every originalweight/pack/CSR/workspace/counter byte, legitimate
nonzero dormant/alias storage,80retainedgraph mutations, actual one-node profiles
and drain/reset are required. Unknown active counter protocols or unsupported
selected geometries abort rather than weakening checks. S1fixed workspace and
counter buffers intentionally contain nonzero sentinels and must stay unchanged.

Four contexts=M1/M8×two output roles retain1132rawcalls: per context15warm,
168F1 (8×21blocks),100paired (20rounds×5arms,10cyclic/reverse orders), and10strict
own-fixture terminal output oracles. Forty total terminal outputs are checked
at F1 and paired boundaries BEFORE any further replay/mutation/profile.
All four causal contrasts per role keep every signed sample and win; no0.3/0.5us
filter. CPU fake-event extraction checks counts/order/role reuse; synthetic
terminal,NaNstorage and binary-tamper negatives reject.

Read-only verifier after freeze:

```
python3 -m proj_rsi.candidates.pw42.verify_frozen
```

After independent CPU GO and root's frozen source/host/launch review only,
proposed narrow normalcap command from iteration host `/root/okc/proj_rsi`:

```
bash tools/h100_trial.sh okc-proj-pw42-gv-empty-01 300 240 bash -lc 'cd /okc && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 -m proj_rsi.candidates.pw42.lab --json /okc/logs/pw42-gv-empty-01.json'
```

A timeout is retained as invalid setup with a new reviewed attempt, never a
silent deadline/source edit. Narrow validation does not establish actual PW42
selection or the full cumulative result. A separate reviewed fresh-prepare
PW38/PW42 complete30shape comparison, actualPICK/wholepolicybudget/JIT/counters/
terminal/cleanup and officialH200 validity are still required before integration.

Falsification: no real empty lists; unchanged compiler-equivalent saved work;
code-size, startup scheduling or occupancy cost dominates; signed gains vanish
under output-role swaps/current policy; added choices/budget regress other
shapes; new variant remains unselected. Retain every tiny signed observation,
positive or negative, and choose the next discriminator rather than declare a
family closed. No CUDA-weight prevalence, attainable gain, whole-score addition,
H100→H200 transfer or official promotion is inferred here.
