# Group8: two native K64 stages per math group

`state=CPU_DEVICE_PASS; GPU_UNRUN; independent_review_required; speedup=UNMEASURED`

Fixed unpooler128: M128/N8192/K4096,128CTAs,512threads,103600B shared.
Parent DP manifest`281ff5a3…`, source`97dd5613…`; control DS source`672dab44…`.
New generated source`e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e`.

Only math register grouping/scheduling changes. Producer W6/B8, old KC64
metadata, parity/CSR catchup, two producer WGs, scalar decoded-W stores,
proxy/owner barriers, output mapping, fresh1MiB X pack and all weight bits remain
exact parent code. `preflight.py` restores exact parent generated source after
reversing just A declarations/prefetch and math region/name.

```text
two native A32 banks, each holding oldK64 stages(2p,2p+1)
startup: prefetch pairs0/1 (four oldK64 stages)
for p=0..31:
  wait oldBfull(2p), oldBfull(2p+1)
  fence A32/D; four MMA on B2p then four on B2p+1; commit; wait1
  if p>0:
    previous pair p-1 is retired: release oldB(2p-2),oldB(2p-1)
    if p+1<32: prefetch pair(p+1) into that retired A32 bank
final wait0; release oldB62/63; unchanged output
```

All256 K16 MMA operations retain parent order and FP32 accumulator chain. Per
WG fences/commits/wait1 fall64→32; old W/TMA/producer/decoded-B stages and full
waits remain64. This changes the grouping, decoded-B retirement granularity and
steady lookahead192→128 K elements. It is not the parked paired-fence version,
nor physical producer K128 staging: no doubled shared rings, new W pack or
retirement scratch. Group8 prefetch follows the covering wait canonically.

Prior `pw24h/submission.py`185,434–460 documents/implements KS2, eight MMA per
commit with wait1 protecting previous A. That older path used a different
producer/shared-X/operand layout and lost to tuned Lt; its result does not
measure this fresh-packed-X/dual-producer policy. [PTX8.7 WGMMA fence/commit/wait
contract](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
permits multiple same-shape MMA in one async group and requires a covering wait
before generic access to pending A/D. A fence is still required for the generic
writes to each next A bank.

CPU/source/device gates:

```sh
python3 proj_rsi/candidates/operand_reverse_group8/preflight.py
```

PASS: main122/pack22regs, zero stack/spill/local/USETMAXREG/C75, rounded128×512
=65536registers. Actual device SASS has two fences and two8-MMA groups across
16staticMMA; two disjoint32word A banksR88..119/R56..87 and D R24..55. The
generalized checker discovers those sets, checks16initialvectorloads, every
generic A/D access against pending groups, generic-write→fence coverage, safe
refill placement, all reachable outlined paths and final wait0. It rejects the
old real premature A overwrite and synthetic missing-fence/current-A/D-write
negatives. Actual Torch JIT artifacts must repeat these gates before any math.

The symbolic proof checks all64native stage indices/256MMA quads,32pairs and
64Bempty releases. Producer P(s) waits on release(s−8), which occurs after math
G(floor(s/2)−3), strictly before its own G(floor(s/2)); the160-node dependency
graph is acyclic. B8 retains separate old-slot phases and eight warp arrivals per
stage. Output reuse follows final publications62/63 plus final math wait0.
These are bounded source/models, not GPU scheduling or occupancy evidence.
Retained CPU bit/layout models cover524288weights, all65536Xpatterns/full1MiB
pack mapping and every8192D scalar. CSR bit reconstruction is separate from its
unchanged source control-flow/encoded32-entry reload protocol.

Root-only bounded lab after critic GO:

```sh
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_group8/lab.py --json /okc/logs/operand-reverse-group8-01.json
```

Proposed inner240s/outer300s, same prior full-validation trial bounds. No GPU
launch by author. Actual .so SASS is archived before checker admission; failures
save `no_candidate_math_launch=true`. Actual regs≤128/local0/maxThreads≥512,
opt-in SMEM103600 and occupancy≥1 are required for DP/DS/G8. Compare actual
prepared pw36 `kernel_fn`, DP parent, DS control and G8. Each packed arm has a
distinct1MiB workspace and times fresh pack+main every call. Retain all-bit GPU
storage checks, complete4096onehots/FP64/normal/outliers, freshworkspace/output,
20actual timed-graph mutation/NaN-output checks with15packed poison/fullbyte
oracles, immutable X/W/packed/escape data, zero counters and reset all graphs.

First timing method is unchanged21rotated/reversed rounds plus conservative
before/after F1. Report both estimators and every comparator; no favorable
estimator selection. An incumbent improvement requires≥0.5us in both methods
before any integration proposal. Progress against DP/DS is learning evidence;
mixed/small effects require the predeclared balanced-pair followup rather than
family rejection. No official slot/full submission/integration is authorized.

Falsifiable predictions: dynamic math fences/commits/wait1 halve, MMA/input
payload/output bytes do not; no additional producer decode/store/CSR work;
launch/register/SMEM occupancy stays one CTA if runtime admits it. X main
requests remain128MiB logical and fresh pack remains1MiB read+1MiB write;
packed-W ingestion remains48MiB plus unchanged metadata. Fewer sampled ARRIVEs
need not reduce time: waits may move to Bfull/group wait or shorter X lookahead.
No sampled-stall→microsecond or unmeasured speedup claim is made.
