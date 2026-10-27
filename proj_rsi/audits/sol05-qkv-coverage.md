# QKV128 reverse group8 coverage: concrete CPU split4 prototype, runtime HOLD

Owner sol05. No GPU, CUDA context, extension load, prepare, benchmark, submission,
commit, frozen-parent modification or global-ledger update. This is a resource
and indexing result, not a performance admission. Group8's provisional35.04us
unpooler result loses the actual incumbent and lacks its repaired strict terminal
output gate; nothing here presumes that mechanism is a winner.

## Actual comparator and prior costs

QKV M128/N2304/K8192 has36 channel64 CTAs for a full-K reverse port, versus132
H100 SMs. Actual pw36 judge log selects **rq:s4s5hz, KS7**, not cuBLAS: bake16.80us
versus r6/Lt19.62us; its18 channel128 tiles ×7 slices provide126CTAs.
M256 instead retains r6/Lt21.63us, beating rq22.21–22.75us. Source
`rs32/submission.py` RQKS and `_rqesc` are the exact codec/protocol reference;
`pw36/profiles/submission.e22a4b9d693d4902.log:61` supplies actual selection.
Fresh preparation can select differently, so a future lab must call prepared
pw36 `kernel_fn` and report actual selected plan, not substitute this bake time.

Prior `audits/p4-h100.md:99` has plainBF16 QKV128 cuBLAS19.9–20.8us,
k2x20.83/rsk2x20.67us, k4(BLOCK_K256)27.94us; QKV256 k2x37.25 versus
cuBLAS23.0–23.7us. Those are distinct dataflows, not current reverse measurements,
but rule out calling8MMA/group, K128, register-A, or splitK new by themselves.
The current prototype tests packed-X plus dual decoded-W producers with a new
explicit FP32 reduction. It changes split/reduction order as well as coverage;
these confounds must remain named.

## Immutable prototype and fit

New files are under `candidates/group8_qkv/`. `build_cpu.py` freezes the exact
G8 extension SHA `e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`,
then generates device-only source SHA
`d568e2612056b5bb37a1ebaadeb5c50d5c63345a736d660b144c25f4dac0d864`.
No host runtime/extension entrypoint is claimed ready.

One fixed split policy: grid(36,4),512threads,32local K64 stages/slice,
64groups over all slices,16groups/slice/mathWG. Every slice has a whole number
of K128 pairs and at least6startup W stages.144CTAs means12CTAs extend beyond
one132SM wave; full occupancy is not guaranteed. Main **122regs**, pack22,
reduction16; all zero stack/spills, no C75. Main SMEM remains103600B; rounded
128×512 uses the65536register pool. This is offline nvcc12.8 codegen, not actual
JIT attributes, occupancy or runtime launchability. The inherited physical
checker passes both8MMA groups, A2×32/fence/retirement coverage in offline SASS.
Sources, cubin, SASS, resource log and result JSON are retained.

## Exact codec, packing and per-call bytes

W uses ordinary `_rqpack` lossless12bit layout[18][128][8][1536], **27MiB**,
plus table word and `_rqesc(esc6,NB=18,KC=128,KS=4,dev)` metadata. Decode is
unchanged. For slice ks, `k0=32*ks`, packed W stage is
`(nb*128+k0+s)*12288+half*6144`. CSR list is
`((ks*18+nb)*8+4*half+w)*CAP`; each producer independently starts its cursor
at0, owner1 skips local stage0, then owners consume local stages of their parity.
Entries encode **local** seq0..31, not global stage. Full-K unsplit would use
0..127, also representable. Entry31 remains disjoint from lane/reg/half fields;
sentinel0xffffffff decodes seq32767. Capacity is computed from actual escapes:
`CAP=ceil((longest_list+1)/32)*32`, 32minimum,32trailing sentinels. There is no
fixed8lane/constantCAP assumption. Pure worst-case per-warp list32stages ×32lanes
×32BF16 values yields CAP32800, not the actual allocation or expected density.
The current CPU proof checks all entry field combinations, but does not supply
actual task-W escape counts or independently run CPU bit-exact W pack/unpack.
Those remain mandatory host-wrapper gates, with a bounded allocator guard.

Pack is fresh every invocation:512blocks×256threads, four u32/thread,
**524288words=2MiB**. `idx` decomposes v/lane/chunk/warp/s;
`row=64*(warp>>2)+16*(warp&3)+(lane>>2)+8*(v&1)` and
`k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)`.
Write `XP[idx+131072*u]=X[row*4096+k/2+1024*u]`,u0..3.
This is an exhaustive bijection of all X words. Consumer uses
`XP+(k0+s)*16384+warp*2048+lane*16`, offsets0/512/1024/1536B.
No input-derived cache, reused stale pack or packing during prepare is legal.
Per-call pack logically reads2MiB and writes2MiB; all consumers request72MiB
XP and deliver36MiB decodedB to shared, and packed W total27MiB. These are
logical bytes, not measured sectors/cache/HBM traffic or time predictions.

## Slice barriers, lifetime, reduction and order

Local W6/B8 phases use local s throughout; only source W/XP indices add k0.
Wfull phase(s/6)&1, Bfull phase(s/8)&1, Bempty expected parity((s/8)&1)^1
remain unchanged. Each producer's stage s depends on same-owner s−2, W refill
s−6 and Bempty release s−8. Consumer groupg waits two Bfulls, fences, issues
8MMA in stage2g then2g+1/K16 order, commits and wait1; previous A bank is
refilled only after its covering wait. Final wait0 precedes every partial store.
`proofs.py` checks an80node per-slice DAG, exhaustive output ownership and the
inherited offline physical-register checker. This is not arbitrary-interleaving
hardware validation.

Main stores every D register directly to disjoint FP32
`partial[ks][row][channel]`, no BF16 partial rounding. Each slice independently
accumulates K in increasing order; separate reduce sums slice0+1+2+3 in that
order and rounds once to BF16. This differs from full-K FP32 accumulation and
actual RQ's BF16 split reduction, requiring full numerical validation.
Workspace is **4,718,592B=4.5MiB**, fully overwritten each call before reading;
reduction logically reads4.5MiB and writes589824B BF16. It is1152blocks×256
threads with full output coverage. No counter, memset or cross-CTA polling is
used; stream order pack→main→reduce supplies ownership. Thus counter reset is
vacuous for the new path, but actual parent counters must still be checked0
between calls. Lifetimes require retained X, XP, partial, output, W and metadata
through graph completion. All three nodes must be timed; extra-launch cost is
**unknown**, not zero or a measured constant. A fused last-CTA counter scheme
would be a separate version with release/acquire/reset proofs, not a free fix.

## Smallest useful next diagnostic, conditional only

HOLD for root criticGO and repaired G8 control evidence. If root admits further
work, first add a fixed-shape host wrapper with exact device/dtype/shape/stride/
alignment/size checks, full per-call three-node invocation and persistent-owned
workspaces. Port same-data equivalent G8 KS1 as a causal coverage control only
if needed; do not compare split4 solely to a failed unpooler parent or Lt-only
function. Required comparison is actual prepared pw36 `kernel_fn` + equivalent
QKV parent/control + this new version in one bounded serialized run.

Before first math, require actual JIT cubin/SASS physical A/D/fence and resource
checks with retained source hashes. FullK8192 onehots, FP64/task normal/outlier,
all-bit probes, nonfinite rejection, NaN-poisoned XP/partial/output, fresh X,
zero, mutation, W/metadata immutability, graph replay/workspace lifetime and
counter reset on parent apply. Inspect **actual terminal timed output AND XP
before any overwrite/replay/mutation**, then validate probes. No timing
admission from offline SASS, pre-timing numerics, static instruction counts or
PC sample fractions. If valid but losing, retain the version and diagnose
coverage/pack/reduction costs; this is not a QKV or reverse-family ban.

## Conditional score coverage

`coverage_arithmetic.json` uses only rs32 officialfb548. QKV128 contributes
13.207008322us in verify16; selected pw36 bake values are not substituted.
A genuine complete-QKV128 cut0.5/1/2/3us changes score to
3196.652053/3192.664380/3184.613504/3176.459559us, all else fixed.
Verify16 alone needs21.728430046us total-load reduction to reach3000, exceeding
all of QKV128; QKV-only cannot close the goal. Prototype coverage is exactly
M128/N2304/K8192. No m256, decode or other-layer improvement is implemented or
measured. Attainable gain and extra-node tax remain unknown; score arithmetic
is neither a forecast nor grounds for official submission.
