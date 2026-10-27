# CONDITIONAL: DP paired K128 producer/math phases

State=CPU/source architecture assessment ONLY; no kernel, compilation or GPU.
Frozen parentDP=manifest281ff5a3/source97dd5613, paired-store composition6785da2e
awaits runtime. Actual combined harvest has priority. Parent DP valid whole F1
34.495998us vs incumbent24.095999us; this proposal has no measured speedup.

## History / precise scope

Already tried: pw24h `bh` KS2/k2x/rsk2x groups two K64 planes into K128 slots;
one consumer phase and8m64n64k16 per group. `rsk2x` has A0/A1[32]. Its critic
verified eight HGMMA/singlecommit/wait1 and a correct finalwait0. Plain-BF16
hand pipelines remained behind tunedLt, even though stage merging improved
their previous versions. Sources: [pw24h](../candidates/pw24h/submission.py:185),
[critic](pw24h-critic.md), [anatomy/history](p4-h100.md:53).

Also already present: rs128/rs128p GS2: two per-K64 full waits/LDS/release paths,
then one fence,8HGMMA/commit/wait. The source explicitly calls this K128 rounds.
It does not physically merge its K64 W/X barrier slots and has decodedW in
registerA, unlike the reverse kernel. Thus eight-HGMMA groups and A2 are not
novel mechanisms. [Source](../candidates/rs128p/submission.py:3211).

Narrow new test, if later admitted: **physically merge reverse-DP W/B producer
publication and consumer ready/release phases**, preserving old K64 packed-W
format, exact decode/escape arithmetic, separate SW128 B planes, geometry,
fresh XP, cache policy, 512threads, two producer owners and output. Root brief
uses scalar-store DP as baseline; do not silently combine STMX unless root
chooses that measured parent after terminal evidence.

## Fixed dimensions / addresses / bytes

Shape M128/N8192/K4096, grid128, fullK KS1. Define pairP=0..31,
sub=0/1, **oldseq=2P+sub**. Keep host `p.KC=64` and all immutable pack/escape
metadata in OLD K64 units. Kernel pair-loop count32 is a different constant.
Changing host `p.KC` or CSR `_rqesc` KC to32 corrupts pack stride/escape sequence.

```
WST=6; BST=8; WBpair=12288; BBpair=16384
Wslot = base+(P%6)*12288
Bslot = bring+(P%8)*16384; bring=base+6*12288
srcW(nb,half,P,sub) = W + (nb*64+2P+sub)*12288 + half*6144
dstW(P,sub) = Wslot+sub*6144
dstB(P,sub) = Bslot+sub*8192        # two OLD SW128 planes
```

Each Wfull phase: ONE `arrive.expect_tx(12288)` then TWO bulk6144B copies
to the same barrier. These global source ranges are not contiguous: a single
12288B copy would load the opposite channel-half instead of the next same-half
K64 stage. Both completions must arrive before the single Wfull wait succeeds.
All6144B copies and destinations remain128B aligned. No packed-W reencoding.

Decoded subplanes use existing row*128/SW128 mapping at `+sub*8192`. Descriptor
`rq_desc(Bslot+sub*8192)+2k`, k0..3. Do not reinterpret as one row-major128-wide
K tile or alter descriptor stride/swizzle. Each 8192B plane is1024B aligned;
normal descriptor addresses cover its original K64 atom. Existing store and
bit proofs apply independently to each subplane; a future implementation must
prove full paired byte/slot equivalence explicitly.

XP packer stays byte-for-byte:1MiB fresh every call. Main pairA[32] loads the
old XP stages2P and2P+1, four v4 vectors each, offsets0/512/1024/1536 inside
each old16KiB stage. Initial two paired arrays load the same64u32/thread as
old four arrays. No persistent input values or setup prepack.

CPU enumeration performed during audit: all nb0..63/half0..1/P0..31/sub0..1
cover8192 old subtiles exactly once; max packed-W byte50,331,647. Ring parity,
alignment and both subplane offsets checked. Nominal main global payload remains
W50,331,648B (48MiB), X134,217,728B (128MiB); these are NOT measured HBM bytes.
Pack1MiB read+write/output2MiB and nominal shared288MiB remain unchanged.

## Ownership / CSR / phase / tail

512threads unchanged. owner=(warp−8)/4, pairP=owner; P<32; P+=2.
W6/B8 even => same owner across every physical slot reuse. Independent128-thread
bar3/bar4; publisher/refiller256/384. Owner0 last pair30(old60/61), owner1 last
pair31(old62/63). Each owner still decodes32 OLD K64 stages, now16pairs.

Before each sub: skip CSR entries whose oldseq<2P+sub, then unchanged
`rq_fix(a,2P+sub)`. Two subdecodes reuse local a[16] sequentially. Old CSR
encoding/cap/sentinels/32-entry reload windows unchanged. Assignment changes
from alternating OLD stages to alternating PAIRS: metadata traversal/skip
instruction counts need not be identical. Both ownership patterns must be
modeled against a full sequential repair, including dense>32windows/repeated
edits/empty stages/last63; missing-skip negative control remains mandatory.

Single Wfull wait; if P>=8, Bempty wait; decode/store sub0 andsub1; then every
writer executes proxy fence; owner bar3/4; publisher arrives Bfull ONCE and
refills pairP+6 if<32. Initial issuer256 initializes6W/8B slots and launches
pairs0..5 with two copies each. Wfull parity=(P/6)&1, Bfull=(P/8)&1,
Bempty previous=((P/8)&1)^1. W/B prerequisites concern earlier pairs P−6/P−8.
Consumer Bempty arrival count stays8 (one lane0 for each math warp), ONCE per
retired pair. No arrival/release between the two subplanes.

Initial pairs0..5 + refills6..31 cover exactly32pairs/64raw copies. No refill
after P25. No partial K128 tail for fixedK4096. Initial/final named math barriers
and output mapping unchanged; after wait0 and all pair waits, last producer
stores/publications and all W copies have completed. The128x64 output slab
fits the first16KiB Bslot and may be reused only after that drain.

## Native A2 lifetime / ISA limits

```
q0[32]=pair0; q1[32]=pair1
for P=0..31:
  current=q[P%2]; retired=q[(P-1)%2]
  wait Bfull(P); pin D/current; wgmma.fence
  issue sub0 k0..3 then sub1 k0..3   # exact increasing-K arithmetic order
  commit ONCE; wait_group1
  if P>0:
    pin retired (32 words); release B(P-1)
    if P+1<32: prefetch pair(P+1) into retired
wait_group0; pin D and ALL64 A words; release B31; output
```

Audit CPU model: every current buffer equals P, never overwrites pending group,
wait1 retires exactly P−1, tailwait0 retires31; maximum two committed groups
before wait1, one afterwards. Last iteration31 issues q1, releases30, no refill;
finalwait0 then releases31. q0/q1 always physically disjoint32-register banks.
Current helper `rq_fa(uint32_t(&)[16])` cannot pin A[32]; a future patch must
provide explicit32-word pinning and verify physical code, not use an unchecked
pointer cast/copy. Old Q4/A2 alias-hazard rejection remains a negative control.

Primary PTX says commit batches prior uncommitted MMA; wait1 retires older groups;
register-A cannot be reused until its covering wait. All128threads must execute
the same aligned group operations. [PTX commit/wait](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).
NVIDIA CUTLASS explicitly restricts waitN to0..7; onlyN1/N0 used here.
[Primary header](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp#L54).
No unsupported claim of a documented per-group instruction maximum: eight
HGMMA/group is already compiled/validated in local historical pw24h. New source
must still compile eight/group without C75 and satisfy a fresh actual-JIT gate.

Required checker change if implemented: classify TWO32register A banks from
eight HGMMA/group, count two waits in a two-pair unroll, preserve D32 disjointness,
conservative writes/outlined paths/two-loop wraparound/tailwait0, verify actual
group commit boundaries. The existing A4 checker cannot be applied unchanged.

## Resource and fixed-work accounting

|quantity/CTA|DP K64|paired K128|
|---|---:|---:|
|stages / stages per owner|64 /32|32 /16|
|raw W copies (6144B)|64|64|
|Wfull phases/expect arrivals|64|32|
|decodedW bytes|524288|524288|
|math HGMMA per WG|256|256|
|math fence/commit/wait1 rounds per WG|64|32|
|Bfull publications / owner-group fence+bar rounds|64|32|
|math Bempty releases per warp (including tail)|64|32|
|producer Bempty successful-stage waits|56|24|
|initial rawW stages in flight|6|12|
|decodedB capacity in oldK64 stages|8|16|
|A payload u32/thread|4×16=64|2×32=64|
|nominal X prefetch distance|3old stages=192K|1pair=128K|
|SMEM bytes|103600|206000|

SMEM=1024+6*12288+8*16384+8*6+16*8=206000,
headroom26448 below actual node opt-in232448. One CTA/SM remains the ceiling.
Same A64+D32 payload is96registers before pointers/temporaries. Fit<=128 is
plausible, NOT established: compiler allocation, two descriptors/new loop logic
may exceed it. Hard gates remain main<=128, round8*512<=65536, local/stack/spill0,
no dynamic register redistribution/C75, actualoptin206000 occupancy>=1. Shared
memory info/lab assertions must distinguish this candidate from103600B controls.

Not a pure round-count experiment: W/B byte-depth doubles, Wfull gates both
subcopies before sub0 decode, B publication waits both decoded subplanes,
X prefetch distance shrinks, MMA burst length doubles, CSR ownership changes.
Any measured gain/loss is the whole paired version. Source-op halving does not
halve hardware polling, barrier latency, byte cost or main time.

## Admission / falsification / measurement

Conditional until combined runtime and root's fresh DP counters arrive. Useful
if per-round Bfull/publication/fence or producer synchronization remains exposed;
less credible if DP is dominated by unavoidable packedW/X delivery or tensor
service, or if shorter A lookahead worsens X readiness. Prior PX samples are not
substitutes for currentDP/combined samples. No numerical speedup forecast and
no promise to close the~10us incumbent gap.

If root later admits ONE source version: full paired bit/CSR/index/descriptor/
phase/tail model, exact delta to named frozen parent, actual90a resource gate,
new A2 physical checker plus bad-copy negative control, actual loaded-JIT gate
before math. Compare actual incumbent and same-job named parent; retain best
failed control. Fresh pack+main2nodes in every timed graph, all-K exact/FP64/
normal/outliers, current-X/workspace/output poisons, all65536storage patterns,
weights/escape bytes immutable/counters0, profiles and complete graph reset.
>=0.5us BOTH F1 and paired versus incumbent for local promotion review; record
partial parent progress independently. Diagnose next measured limit on failure.
