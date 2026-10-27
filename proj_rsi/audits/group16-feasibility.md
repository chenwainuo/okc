# Group16: conditional CPU plan; no source or GPU admitted

Decision: **A2×64 NO-FIT under current uniform512thread policy; A1×64/wait0
HOLD pending group8 runtime/mechanism evidence.** No prototype, compilation,
new GPU/profile, geometry sweep or dynamic-register rewrite. Root integration's
separate math-only group8 A2×32 version is the immediate causal experiment;
preserve its scalar-store DP parent and compare it before any composition.

Current evidence: DP counter samples833/1750math at WARPGROUP.ARRIVE and
producer Bempty626/1556; these identify dependency associations, not intrinsic
fence latency or an available microsecond budget. [Fresh DP audit](dual-producer-pc-roi.md).
Fixed-source DS now wins the separate two-policy diagnostic by1.168us paired,
but has no incumbent/official win. [Measurement audit](dual-stmatrix-measure-result.md).

## Prior implementation, not an untried family

pw24h KS2/rsk2x already uses8HGMMA/group and A0/A1[32], wait1 and explicit
retirement. Its SS family also tested physical BLOCK_K256/k4,16HGMMA/group:
L3 pooler12830.94us versus K128/k2s127.71–27.84; full bake k431.90/27.94us
pooler/qkv128 and53.79/46.40us pooler/qkv256. These are older plainBF16
geometry/dataflow, not a present DP measurement, but contradict automatic
"larger group=faster" reasoning. Current frozen pw24h selected list no longer
includes k4; historical results are retained in [P4 H100](p4-h100.md:91), not
claimed as a currently selectable compiled kernel.

Old rs128 GS2 likewise groups8MMA while retaining individual oldK64 slots.
Math-only group16 here would group FOUR oldK64 Bslots, without physically
merging producer W/B phases. Distinguish it from the separate physically
paired-stage [K128 plan](dual-producer-k128-plan.md).

## Resource lower bound / exact overlap trade

Current DP:2mathWGs+2producerWGs=512threads, uniform registers rounded128,
entire65536register pool. D=32FP32/thread, old A4×16=64u32; actual122regs.

|math policy|A payload/thread|A+D lower bound|512thread static fit|consumer overlap|
|---|---:|---:|---|---|
|current4MMA, A4×16|64|96|actual122/round128|wait1,3oldK64 prefetch ahead|
|group8, A2×32|64|96|author actual122, runtime pending|wait1,1K128pair ahead|
|group16, A2×64|128|160|**fails<=128 before temporaries**|wait1,1K256group ahead|
|group16, A1×64|64|96|plausible only; uncompiled|**wait0, next group X load after retirement**|

A2 cannot be fixed by a register cap:128A+32D are simultaneously live; spills
would move required work to local memory. Dynamic producer donations or a
different CTA/role geometry would be a separate architecture with compiler,
physical-pool, producer-register/occupancy and deadlock proofs; no such change
is authorized or currently evidenced. Do not revive an oversubscribed inc/decr
scheme or claim aggregate average registers solve physical launchability.

A1 waits for all16MMA before overwriting any of its64Awords. Initial load covers
four old stages, same initial payload as A4. After commit, no current-bank refill
is legal until wait0: registerA is asynchronous. A1 removes cross-group X-load /
MMA overlap; producer W work can still overlap math. No identity-copy/pin trick
permits early physical register reuse. Exact A1 physical SASS/fence/retirement
proof, full output sink, C75/local/spill/resource gates would be required if
later admitted. No runtime resource claim follows from the96register payload.

## W6/B8 flow / arithmetic / tail

Keep old64 K64 stages, producerowner=s%2, W6/B8, CSR skip/sequence, bar3/bar4,
6144B Wcopy/8192B decodedB, output and fresh XP. Consumer groupg0..15 waits
Bfull stages4g..4g+3, fences once, issues16HGMMA in EXACT oldstage/k16 order,
commits once. A1wait0 then releases allfour Bslots; last group60..63 drains
before output. No tail for fixed K4096 (also K1024/8192 divide256, but other
shapes remain unsupported until routing/index proofs).

Logical flow has no new W6/B8 cycle: producerP(s) depends on earlier same-owner
P(s−2), WrefillP(s−6), and R(s−8). With A1wait0, R(s−8) follows group
floor((s−8)/4)=floor(s/4)−2, strictly before groupfloor(s/4). For abstract
A2wait1, release follows at most floor(s/4)−1, also earlier. Initial W0..5
and initial B0..7 allow startup; final producer62/63 and all W transfers have
landed before output reuse. Audit CPU topological models:144nodes each, A1
wait0 and abstract A2wait1 both acyclic. This is an ownership/dependency model,
not arbitrary GPU-interleaving or hardware register proof.

Group16 halves math fence/commit/wait calls versus group8 (32→16/WG), but
retains64 Bfull waits,64 Bempty releases,64 producer proxy/bar rounds and
64raw Wcopies; same256HGMMA/WG, X/W/shared bytes and CSR work. It does not
halve producer synchronization. All64 source stages still checked individually.
The old source hot fence can migrate to new X readiness, HGMMA dependency or
Bfull waits instead of disappearing. Primary PTX covering-wait rule remains:
[commit/wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).
No claim of a documented "16MMA maximum"; historical SS k4 supports grouping
feasibility, not proof that new RS code lowers or runs efficiently.

Reopening criterion: group8 VALID actual timed graphs and a material complete
pipeline gain, with residual evidence consistent with fixed per-round overhead
rather than X-readiness/producer starvation. Then assess whether further16round
savings can credibly exceed lost X/MMA overlap by>=0.5us against actual incumbent.
Without that evidence, A1 is an unproven throughput trade and A2 is a hard
resource failure under the named policy. No blind8/16/32 grouping sweep.

## Authoritative score ROI / hard single-shape ceiling

Use ONLY best rs32 jobfb548b5dc6f04696
[raw](../candidates/rs32/profiles/submission.fb548b5dc6f04696.json),
SHAf3a8e1c1339cbaf764d0f9024b7d4c9273dc73c15788f425c8c16e6b7a037fd7.
S0=3200.6150084373335us, passed9/9. Recomputed all6 same-job SGLang/load sums:
speedup1.509334920715312,4830.8/speedup exactly S0. No other job minima or
H100 candidate times substituted.

|load|complete benchmark total us|local score-us per genuine load-us|
|---|---:|---:|
|decode1|35.291272317|15.115234|
|decode16|48.292189824|11.046006|
|decode32|52.139459709|10.230943|
|verify1|37.485332530|14.230522|
|verify16|67.511418316|7.901417|
|verify32|86.584848719|6.160845|

Exact hypothetical same-job law, fixed SGLang vector:
`Snew=S0*exp(sum_load(log1p(-delta_load/Tload))/6)`.
The slopes above are S0/(6*Tload), derivatives only, not constant large-gain
conversion factors. UnpoolerM128 contributes18.951374903us to verify16.
Goal gap200.615008us. Verify16 alone needs21.728430046us reduction to reach3000;
this exceeds the ENTIRE unpooler128 benchmark time. Even impossible ZERO-TIME
unpooler128 leaves **3029.589194531us**, max reduction171.025813906us.
Thus unpooler128-only cannot close the gap, even before physical floors.

|conditional genuine complete-shape gains, all else fixed|score us|reduction us|
|---|---:|---:|
|U128−0.5us|3196.652053|3.962955|
|U128−1us|3192.664380|7.950629|
|pooler/qkv/unpooler M128 each−1us|3176.459559|24.155450|
|same3layers at M128 AND M256 each−1us|3157.845952|42.769056|
|same6shapes each−2us|3113.855375|86.759633|
|same6shapes each−3us|3068.549774|132.065235|

Same6shapes would need **4.456254803us each** genuine improvement to reach3000
if all other shapes stayed fixed. Arithmetic scenario, not a source prediction,
additive local-H100 gains, or available kernel budget. Current reverse versions
still have no incumbent gain; their failed-version progress contributes ZERO
demonstrated official score reduction.

## Coverage needed after the seed mechanism is validated

Current nativeA/output/packer/host guards are exactly M128/N8192/K4096. Reuse
requires explicit shape policies and proofs, not changing a dispatch constant.

* **UnpoolerM256:** two M128 token tiles need XP2MiB, token-row offsets and output
  indexing. Straight duplication produces256CTAs/two waves and doubles Wdelivery;
  no automatic win over official24.693354us. A one-wave128channel geometry would
  increase D and B/W storage/resource costs, a separate design. NearbyM must
  receive complete legal tiles/zero padding or the incumbent fallback.
* **PoolerM128:** full-K N4096/K8192 gives64channel64CTAs, not128, and128old
  stages. Simply reuse loses SM coverage. Restoring it by splitK introduces a
  precision-correct reduction/output protocol and timed cost. Tail/divisibility,
  immutable pack/CSR indices and whole-output ownership must be proved.
* **QkvM128:** N2304/K8192 gives36CTAs, an even larger coverage problem. SplitK,
  narrower channel tile or token decomposition changes geometry/epilogue. Do
  not assume old single128token A/index layout supplies a competitive policy.
* **Pooler/qkvM256:** added token parallelism helps CTA count but also changes
  pack/Xbytes, output indices and ring/weight repetition; require same-job
  complete-pipeline comparison. No transfer speedup is currently measured.
* **DecodeM16/M32:** higher score slopes matter, but fixed128token WGMMA wastes
  rows and cannot read the smaller tensor with the current loader. These are
  existing GV paths, not free extensions of this nativeA128 kernel. A coverage
  roadmap must include distinct evidenced work on those loads, not endless
  M128-only grouping variants.

Queue recommendation: finish current bounded group8 evidence; use its mechanism
to decide a resource-legal next step AND an actual multi-load coverage plan.
Group16 remains conditional, not another active implementation slot. No promise
that all listed shapes improve simultaneously or that group16 reaches3ms.
