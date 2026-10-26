# Token-pair W multicast: source feasibility and ROI decision

`date=2026-10-26; owner=/root/proj_integration; decision=NO_GO_IMPLEMENTATION; GPU=UNRUN; prototype=NONE`

Scope: unpooler `M256,N8192,K4096`; one fixed policy, channel tile128, token tile128, two token CTAs per cluster, full K in each CTA, original lossless12-bit register-W/shared-X math, plain output stores. This is a source/accounting rejection of the current >=0.5us win hypothesis, not a universal bound on multicast. No source, frozen candidate, GPU queue, or official quota was changed.

## Decision

The policy is technically plausible and avoids operand_reverse's X transpose and doubled X traffic. It preserves W source reads and X delivery bytes, removes the global K reduction, but doubles W shared fill/LDS/decode and the per-CTA stage count. Existing NT128 resident measurements already expose that shorter tensor period as the weaker point. The best available same-run resident estimate spends about3us more before any new multicast/stream/output cost, against about2.4us of entire old reduction epilogue. Moreover, old complete rq unpooler256 was already2.60us slower than the actual prepared r6 comparator. There is no measured source of the additional acceleration needed to beat r6 by0.5us.

Close without a prototype or GPU allocation. Do not reopen through a stage/cluster-size/flag sweep. A materially changed mechanism or new evidence that invalidates the resident-period accounting would be needed.

## Novelty audit

| Prior work | Actual mechanism and result | Relation to this policy |
|---|---|---|
| pw24 `bg12c` | Full-K, plain output, two token CTAs sharing W via TMA multicast; same W tile consumed in both CTAs. Paired in-prepare37.47us vs noncluster `bg12`29.02us. | Same W-over-token-pair mechanism. Proposed12-bit RS/two math WGs/channel128 differs from old BF16 SS/one math WG/channel64, so exact new policy remains unmeasured. Cannot advertise the mechanism as wholly untried. |
| CUTLASS `cw7c2` vs `cw7` | Cluster1x2 with X sharing31.97us vs noncluster32.10us. | Rejects an assumed universal1–2us or8–10us cluster launch tax. W pair lockstep is the closer negative precedent; it is not a measured unavoidable tax for this new kernel. |
| P3 cluster4 X sharing | Router128 CTAs; occupancy30 clusters=120 CTAs; two waves; self-zero protocol could deadlock across nonresident clusters. | Different operand, cluster size, geometry and global-zero dependency. This policy has no global zero/cross-cluster wait. That old deadlock is not a correctness argument against it. |
| rq NT256/two-pass NT128 | Same token256 CTA, K split, reduction epilogue; resident1027–1033clk/k64. | Did not split token halves into full-K clustered CTAs. Supplies the current tensor-period baseline. |
| rs128 stream-K / prepared patch table | Full-output trials lost. | Neither tests this token policy; neither is reopened. |
| operand_reverse | X register transpose, channel64, decoded W shared; complete output47.648us vs actual rq24.128us. | Closed. This policy retains channel128 and X shared, so its X bytes/transpose penalty does not transfer. Its m128 Nsight counters are not m256 multicast measurements. |

Primary repository evidence: [pw24 source](../candidates/pw24/submission.py), lines1670–1764,1816–1835,3040–3042; [pw24 protocol review](pw24-critic.md); [corrected cluster interpretation](m128-plan.md), item12; [original CUTLASS run ledger](../LOOP.md), lines696–701; [P3 occupancy/deadlock audit](p3-m256.md), section3. `pipeline-diff.md`'s old general launch-tax language is superseded by the explicit correction in `m128-plan.md`.

## Exact geometry and byte accounting

`NB=N/128=64; KC=K/64=64; output=M*N*2=4MiB.` Old rq: NT256,KS2 =>128 CTAs,32 k64 stages/CTA. Proposed: NT128,2 token partitions,KS1 =>128 CTAs/64 clusters,64 stages/CTA. Each CTA has two math WGs covering64 channels each; shared X is consumed by both. Tensor FLOPs/CTA are unchanged. The original pack is `[NB,KC,8,1536]`,12KiB per128channel/k64 tile.

| Whole invocation payload | Old NT256/KS2 rq | Proposed token pair/KS1 | Reason |
|---|---:|---:|---|
| W source requests, excluding metadata |48MiB|48MiB|Old disjoint K splits read pack once; one multicast read per new cluster/stage. This is requested payload, not a measured HBM-byte counter.|
| X global/L2 requests |128MiB|128MiB|`NB*M*K*2`; same channel128 tile. No doubled X penalty from channel64.|
| W TMA shared writes |48MiB|96MiB|Multicast populates both CTA shared copies.|
| W shared LDS |48MiB|96MiB|Each token CTA decodes the full K tile.|
| X TMA shared writes |128MiB|128MiB|Half tile, twice as many stages.|
| X WGMMA shared reads |256MiB|256MiB|Two math WGs; half token extent, twice the stages.|
| Mainloop shared traffic sum |480MiB|576MiB|20% increase, entirely duplicated W fill/LDS.|
| Packed-W decode/escape walks |1x|2x|Both token CTAs repeat every W stage; escape lists may share storage but execution/read requests repeat.|
| X TMA requests |4096|8192|Half-size boxes; same bytes does not imply equal request overhead.|
| Output TMA payload |8MiB reduce-add partials|4MiB plain stores|Final output has4MiB.|
| Initial output zero stores |4MiB|0|No split-K dependency in proposed policy.|

Per CTA/stage: old W12KiB+X32KiB fills, W12KiB LDS, X64KiB WGMMA reads =120KiB; new W12KiB+X16KiB fills, W12KiB LDS, X32KiB WGMMA reads =72KiB. Tensor periods are1024 vs512clk/stage; whole-CTA ideal tensor cycles both32768. An assumed aggregate128B/clk shared-port budget would be960 vs576clk/stage; that aggregate port model is UNPROVEN by port02 and is not used as a physical impossibility argument. The known resident measurements are stronger evidence.

Source anchors: [rq geometry/rings](../candidates/rs128/submission.py), `RqC`/`rq` lines3144–3240; [math LDS/decode/RS issue](../candidates/rs128/submission.py), lines3330–3405; [output store/reduce selection](../candidates/rs128/submission.py), lines3568–3608 (`FL&4` selects plain TMA store). Existing host `p.m<=NT` and X origin0 require a token-offset descriptor/launch adapter; merely setting NT128 for an M256 tensor is not a valid port.

## Measured period evidence and timing limits

Read-only raw harvest: H100 `/root/okc/logs/rs128/m256d.log`, SHA256`2bea3d1719dd5540b3cb80cd2af252a068d5ba124d1d08ac6d2f6e59ab5d6c6b`. This is the corrected build after unread-WGMMA probe DCE was fixed, as recorded in LOOP19:10. It tests pooler K8192, not the proposed unpooler K4096; the per-stage math/pack path transfers, but exact full-kernel timing does not.

| Same run resident probe | Stages/CTA | CTAs | CUPTI median/min us |
|---|---:|---:|---:|
| NT256 `r8n256-ks4` |32|128|22.21/21.92|
| Two-pass NT128 within NT256 CTA |32|128|22.19/21.98|
| NT128 `r8-ks4` |32|128|14.59/14.37|
| NT128 `r8-ks1` |128|32|46.69/46.46|
| NT256 no-output streamed `s3n256` |32|128|25.57/25.25|
| NT256 complete `n256s5m1z` |32|128|27.82/27.58|

NT256 stamped WG means1027/1028clk per k64; two-pass1031/1033 vs1024 ideal. Resident paths intentionally remove TMA/barrier traffic. Old resident throughput therefore already hides W decode under the longer tensor work. Splitting into NT128 shortens that hiding interval while retaining all decode stages.

Conditional affine NT128 estimate from32/128stage measurements: slope`(46.69-14.59)/96=0.334375us/stage`, intercept3.89us,64stage estimate25.29us,3.08us above old NT25632stage resident22.21us. **This is not a measured new resident time or a hardware lower bound:** KS1 control uses32CTAs rather than128, hence clocks/utilization and startup differ. It is a warning about the required direction/magnitude, not an exact prediction. Simply doubling14.59us would also incorrectly double startup.

Independent P4 source-identical consumer evidence supports that direction: [P4 anatomy](p4-rs128-anatomy.md), poolerNT128/32stages/128CTAs, resident749clk/stage with real escapes,646 without escapes,521 constant-A vs512 tensor. Thus full64stage workload has conditional loop budgets47936clk (real) /41344clk (noescape) versus old NT25632stage32864–33056clk. Stamped code and uninstrumented timings differ; do not turn this into a paired latency claim. It demonstrates exposed LDS/decode/list-walk work at NT128 even with no producer. Escape removal is not legal or part of this proposal.

No nvidia-smi1980MHz conversion is used: P4 constant-A521clk/stage and CUPTI stage slope implied effective tensor clock~1.5GHz. Compare clock64 inside a controlled variant; compare real complete graph timings for adoption. No m256 NCU traffic/stall counters currently establish the proposed producer or cluster bottleneck. New operand_reverse NCU is a different m128 pair with a newly baked baseline and cannot supply those causes.

## Barrier and launch feasibility, without an implementation claim

The inherited NT128 shape has384threads,168static registers,0local/spills and working producer40/math232 redistribution: `128*40+256*232=64512=384*168`. At ST5 its inherited shared allocation is144520B; NT256 ST5 is226440B. These establish room for the old local rings, not the new code's final register/cluster occupancy. Remote addresses, token offset, recipient state and tensor maps still require a real build/launch gate. No source was compiled for this audit.

A plausible protocol retains full count2 in EACH recipient CTA: its local W coordinator arrives/expect_tx12KiB and X lane arrives/expect_tx16KiB. Leader issues one packed-W multicast with mask3 to equal CTA-relative shared offsets and equal local full-barrier offsets; both local X lanes issue their own token slice. Every CTA must publish barrier initialization and participate in opening cluster sync before multicast. Local full completion covers both operands. W empty at the leader counts16 math-warp arrivals (8 per CTA), issued after W LDS; X empty remains8 local arrivals after the covering wait_group. Multicast saves a source request, not recipient shared traffic or decode.

**Additional control obligation:** a nonleader W coordinator cannot run ahead issuing expect_tx into future phases merely because it does not issue TMA. It needs an explicit bounded slot/phase handshake. One mechanically plausible version retains local empty8 in the peer plus remote-to-leader8, while leader empty counts16; each W coordinator waits its corresponding empty before its next expect_tx, and initial full phases require both local arrives. That is a design obligation, not a verified protocol. Existing pw24 instead has both producers multicast half tiles and both empty barriers count local+peer arrivals; that verified protocol cannot be silently replaced by a leader-only sketch.

After the final math wait and output shared-read drain, a closing cluster sync protects remote arrivals/multicast writes against peer exit. No initial CTA self-zero, global reduction counter, or cross-cluster wait is required. A second occupancy wave therefore need not deadlock, but could erase ROI. SM count128<=132 does not prove64 clusters co-resident: clusters must fit GPC-local allocation and the exact compiled per-CTA resources. Future admission would require shared opt-in BEFORE `cudaOccupancyMaxActiveClusters` using the exact launch config and actual cluster count, plus a real launch. The old cluster4 count30 does not determine cluster2 occupancy.

Packed layout, escape restoration and original BF16 RS accumulation remain unchanged. Both token partitions must address the same complete KS1 escape lists for their channel block, with independent walk state. Output mapping is original transposed rq mapping plus token offset0/128; two128x64 plain stores per CTA are sufficient. Exact complete one-hot/probe/outlier/fresh-input graph validation would still be mandatory if a future changed design were authorized.

## ROI and falsifiable reopening bar

The reference is actual frozen pw36 prepared dispatch, not torch. Handover04 reports same in-prepare unpooler256 r6=27.26us, rq=29.86us: old rq loses2.60us. Removing the entire≈2.4us old epilogue at zero replacement cost would only reach≈27.46us, still0.20us slower; a real plain store has nonzero cost. Those old graph numbers and resident/CUPTI numbers are separate experiments and are not algebraically combined into a claimed new timing. They jointly fail to identify a credible >=0.5us path.

Fresh official pw36 unpooler256 is24.487153us, one Nvjet24.8us. Its removable global K tail is zero; the tail belongs to our losing rq alternative. Official rs32 unpooler256=24.693354us. If a future implementation gained0.5us only there, exact same-job rs32 score would improve about3.088us: `3200.615008*(1-(1-0.5/86.584849)^(1/6))`. This is a small conditional board gain, not progress toward3000 by itself.

Reopen only with source/measured evidence for a changed mechanism that recovers the exposed NT128 work or for a valid same-unit resident64stage result that contradicts the warning above. Then one fixed complete-output trial would need >=0.5us in BOTH lower-before/after F1 and paired cold-L2 graph timing against actual prepared pw36, plus all numerical/replay/immutability gates and dispatch identity. No GPU job, probe, full submission, geometry sweep, or port is authorized by this document.
