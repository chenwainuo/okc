# Current leaderboard source and verified gain stack

The archived best remains **rs32/fb548b5dc6f04696:3200.615008us**, source
`a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`.
It does **not** include PW35's newer TMA router. That router was already stacked
with rs32 in PW36 and officially tested: **3212.415452us**, with all phases
executed. Thus there is a real source omission in the best-scoring file, but
there is no demonstrated untested additive leaderboard win waiting merely to
be merged. The combined version's point estimate was11.800444us worse; reported
speedup confidence intervals overlap. This does not establish that stacking
lost the router mechanism or that it intrinsically regresses the aggregate.

This is read-only local evidence audit, not a new live-board query, submission,
source change or GPU run. Root owns live API verification. Reproducer
[build-current-official-gains.py](build-current-official-gains.py) reads33
archived official jobs and independently reproduces every complete six-load
score; [JSON references](current-leaderboard-gains-official.refs.json) retain
job/source/log hashes, all scored rows, scored dispatch reports,75 positive
selected same-job bake observations and all exported GV bake summaries. Repeated
bakes are repeated observations, not75 independent new features. Sol11 owns
independent exact-source ancestry and Sol16 broader experiment completeness.

## Feature/current-source/evidence/stack matrix

|Feature|In best rs32 source / actual fb548 serving|Positive evidence, with comparator context|Combined test / remaining integration|
|---|---|---|---|
|Lossless12-bit register-direct GV, exact escape handling, skinny small-m paths; tuned cuBLASLt fallbacks|Yes; baseline r6 family remains underneath bake/dispatch|r6 scored lineage23cec... documented in r6-lineage; later official jobs repeatedly select its GV or tuned Lt against V12/torch controls|Inherited through every promoted descendant and9/9 rs32/PW36. No separate base delta inferred from later job scores.|
|PW23 pipelined split-K router m128|Code retained as comparator/fallback; active m128 superseded by later rw|Official17f448: same-job r6 15.01→rp13.44; intermediate rs14.53→rp13.44. Score3399.258|Stacked in PW31 and descendants; do not count rp and its later replacement gains twice.|
|PW27h/P1 one-node self-zeroing small router m8/16/32 and unpooler m1/8|Yes; actual fb548 selected rk at8/16/32 and uk at1/8|PW31 same-job router r6→selected9.41→8.80,9.44→8.99,10.21→9.41. fb548 unpooler8 8.70→8.51 (-0.19), demonstrating retained small gain below0.3 threshold|Official PW31→PW33→PW34→rs32→PW36 combined validation; source keeps alternatives and bake chooses each run.|
|P1 rr6/rr7 small router m1 alternatives|Yes; fb548 selects rr6 (source bake), profile confirms rr|Local alternating r6/rr0 7.20→rr6 6.91; official fb548 GV bake V12 4.54→rr6 2.82. Those are different controls and preparation contexts|Already stacked, run-dependent rr selection; no missing standalone change.|
|PW29h/P3 one-node rw WGMMA router m128|Yes; actual fb548 rw64k32s8z13.22; m256 r6 retained|PW31 same-job oldrp13.54→rw13.12 (-0.42); PW33 rp13.47→rw13.25 (-0.22). These smaller positives are retained, not discarded for falling below later local thresholds|Stacked with P1 in PW31, shorter qkv in PW33, RS128 in PW34, rq32 in rs32; later PW35/PW36 replace router with rtk.|
|PW33/P2 shorter per-warp qkv geometry choices m1/8/16|Yes; qkv1/8 still GV, qkv16 later rq-selected in fb548|P2 actual-r6 local bakes ~-0.6/-0.5/-1.4us; PW33 official selected new GV choices, score3332.114. Its m16 calibrated cross-job change did not reproduce the full local prediction|Already present in PW34/rs32/PW36; rq16 supersedes selected m16, not an additive separate speedup. No row-wise cross-job minima stitch.|
|PW34 RS128 register-A 12-bit split-K at m128, merged barriers/in-kernel zeroing/TMA reduce|Yes; fb548 pooler128 andqkv128 rq-selected; unpooler128 r6 retained|846e same-job pooler25.31→22.62 (-2.69),qkv19.49→16.70 (-2.79). Final49b reorder15514 same-job23.65→23.01 (-0.64),19.23→16.99 (-2.24). Different runs/choices, not a sum|Stacked in rs32 and PW36. Initial PW34 skipped phase5a393... was invalid evidence for this feature, then guard/order repaired.|
|RS32 NT16/32, HALF/EARLY, separate build/bake and true-incumbent comparison|Yes; actual fb548 allfour BF16 m32 plusqkv16 select rq|Exact same-job table below: gains3.90/1.63/1.79/0.44 at32 and0.51 qkv16|Best official fb548 passes9/9+30rows; all present in PW36. No new incremental gain assigned separately to HALF versus EARLY versus geometry without isolated measurement.|
|PW35 TMA rtk one-node router m128/256|**No**; rs32 is PW34+rq32 branch, not PW35 descendant. rs32 hypothesis paths misleadingly copied PW35 text|5d880 same-job m128 oldrw12.93→rtk11.68 (-1.25); m256 oldrw15.94→rtk13.12 (-2.82), actual r6F1 15.14→13.12 (-2.02)|**Already merged and official-tested in PW36 e22a**; same-job router improvement remains. Full aggregate3212.415 does not beat3200.615. Candidate remains validated combined reference; renewed promotion needs valid fresh combined evidence, not assumption of additive scores.|
|mx two-plane CUTLASS local qkv/pooler128 positives|No|Local actual-r6 bakes ~qkv19.49→18.50 (-0.99),pooler27.10→26.53 (-0.57); competing RS128 in same development era had larger win|Alternative kernel for same shapes, not an independent additive feature. No official or same-job win over selected rq incumbent established. Missing source does not mean missing verified stack gain.|
|Plain BF16 P4/CUTLASS pipeline apparent parity/gains|No; lessons influenced RS128|Positive controls were often untuned torch; PW30/PW32 against actual tuned r6 picked nothing. No established shipping gain|No demonstrated unmerged actual-incumbent improvement. Preserve variants/lessons, not a claim of recoverable score.|
|Four cleanroom schools / latency loader internal improvements|No|Fresh bracketed full-policy comparison found0/30 wins>=0.3us versus both actual pw36 brackets for each school; internal or fallback wins have different comparators|No ready missing policy winner. Completed verifier audit retains65/65 and paired caveats.|
|Fresh packed-X, STMatrix, dual producer, native-X, G8/async/X4 composition research|No|Several valid wins versus failed predecessors; native-X31.856 beatsDP34.096/DS35.312 but loses actual23.968. No shipping gain from that result|Keep lineage progress/source identity; new queued combined kernels require their own valid whole-pipeline actual-incumbent comparison and official validation.|
|Smallm address cursor0d0ef candidate|No|CPU/layout/codegen and GPU basis evidence are not speed evidence. First lab counter gate failed with0 raw timings; no verified benchmark win|Measurement repair/new attempt before any timing admission; no instruction-count-to-score claim.|

## RS32 incremental small-m official evidence

All rows are from the **same fb548 judge preparation**, comparing each rq choice
to the actual prior selection timed before/after variants, keeping the lower
control. In these rows the printed incumbent is r6; the bake implementation
would instead label/use sel or rq128 when that was the actual incumbent. A
printed r6 path can itself contain prior tuned GV/Lt behavior. These are bake
µs, not calibrated scored µs and not a forced plan comparison.

|shape|same-job incumbent bake|actual chosen rq bake|gain|actual rq pick|calibrated scored µs|
|---|---:|---:|---:|---|---:|
|qkv32|17.95|14.05|3.90|t32s5hze-k7|10.107945|
|pooler32|21.02|19.39|1.63|t32s5hze-k4|15.538102|
|unpooler32|20.96|19.17|1.79|t32s5hze-k2|15.149120|
|o32|10.94|10.50|0.44|t32s6hze-k2|6.365771|
|qkv16|14.56|14.05|0.51|t16s5hze-k7|9.663936|

Primary stdout lines127–163 and JSON profiles confirm these selections. Old
RS32 NOTES/H100 tables say o32 andallm16 stayr6; those describe an earlier local
bake and are superseded for this official job. Do not repeat the hypothesis's
cross-job -4.45/-0.80/-1.10/-1.69 as exact causal increments: PW35 andrs32 were
separate official jobs. No official scored timing of the source predecessor
under the same job is established by these artifacts; the same-job controls
are the preparation bake observations above. `parent_ns`-named benchmark fields
are not assumed to identify a timed source predecessor without harness proof.

The same current job also keeps earlier RS128 gains: pooler12823.87→22.94 and
qkv12819.42→16.86; unpooler12822.82→22.82 ties and staysr6. The source contains
more candidates than the actual selected subset; compiled presence is not proof
that every variant contributed positively to the measured score.

## PW36 tested the actual requested stack

Exact PW36 source2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf
combines PW35 router andrs32 RQ source. Independent source audit confirms both
embedded kernels/helpers exact parents except explicit combined prepare. The
H100 takeover comparison passed65/65 per file; official e22a4b9d passed9/9,
30/30 timed rows with60runs each and30profiles. Prepare163s: rq32 androuter
both ran, router compile38s, no skipped-budget/error. This was a real combined
run, not a hypothetical merge or absent router phase.

Within PW36 same-job bake, rq32 gains remain qkv32 18.40→14.14,pooler32
20.93→19.49,unpooler32 20.93→19.49,qkv16 14.78→14.11. o32 retainsr6:
best rq10.72 misses sequential1%margin versus10.82 control. The actual router
rtk11.84/13.31 beats oldrw13.06/16.00 at128/256; source/profile confirms one
rtk node. Thus the router gain survived the combined preparation, while actual
small-shape dispatch can change (notably o32). Both small-load increases and
large-router improvements appear in calibrated cross-job observations; they
cannot be uniquely divided into scheduling, tuning, calibration or run drift.

|load|rs32 own sum|PW36 own sum|PW36 minusrs32|
|---|---:|---:|---:|
|decode1|35.291272|36.425574|+1.134302|
|decode16|48.292190|49.178875|+0.886685|
|decode32|52.139460|53.037879|+0.898419|
|verify1|37.485333|37.840604|+0.355272|
|verify16|67.511418|65.510400|-2.001018|
|verify32|86.584849|85.080610|-1.504239|

This table is **cross-job context**, not exact change attribution. SGLang paired
baselines also changed. Correct official score is4830.8 divided by geometric
mean of the SIX same-job SGLang-load-sum / own-load-sum ratios. All33 complete
archived jobs reproduce that arithmetic independently. Old handover62×geomean
of own sums is approximate, not the exact official score law.

## Official combined history and uncertainty

|official source/job|score µs|what was actually added/tested|
|---|---:|---|
|PW23 /17f4484b97644673|3399.258293|r6 plus pipelined router|
|PW31 /bbb732a78955425b|3337.130276|P1 small paths andP3 rw router combined|
|PW33 /8eeaca43bf43467f|3332.113749|shorter qkv geometries combined|
|PW34 first /5a39372456514f20|3361.156079|rq phase skipped; no RS128 gain validation|
|PW34 /846e25d9df09433f|3293.938948|RQ phase ran; pooler/qkv128 selected|
|PW34 final49b /15514e8241db4d09|3295.545517|reordered prepare/warm discard; all phases ran|
|PW35 /5d880cd0aa504209|3278.638455|PW34 plus TMA rtk router|
|RS32 /fb548b5dc6f04696|**3200.615008**|PW34 plus rq16/32, olderrouter|
|PW36 /e22a4b9d693d4902|3212.415452|Both branches combined and executed|

Every completed scored job above passes9/9. These are observed official point
estimates, not causal contributions that add arithmetically. rs32 speedupCI
[1.486694897,1.531974945]; PW36[1.481233692,1.526347408] overlap. The archive
supports rs32 as best measured leaderboard entry andPW36 as correctness-validated
stack, without establishing a statistically distinct aggregate regression or
new best. No under3000us result is present here.

## Corrections and actionable integration status

- PW35 router is missing from rs32 **source**, but not missing from combined
  validation: PW36 tested it together withrq32. Treat current best and combined
  development source as different artifacts; do not silently replace one.
- Handover06 calls9.42→7.49/11.33→9.18 “within-run”; those are cross-job calibrated
  numbers. Same-job actual-old-path bake controls are12.93→11.68/15.94→13.12.
- LOOP's early PW34 unpooler “-0.8” compared one rq variant23.36 to another22.59.
  Actual r6 control22.75 gave0.16 improvement, below1%admission; r6 was retained.
  Final49b gave0.03, also unselected. These tiny positives are retained in raw
  evidence, not called lost official gains.
- rs32 hypothesis router paths/kill fields are copied stale PW35 prose; exact
  source and actual job profiles take precedence. Source geometry choices,
  LUT/codec/cache proposals and earlier predictions must not be counted as
  measured improvements individually.
- No archive evidence reviewed here establishes another older local mechanism
  beating the actual selected incumbent and then accidentally omitted from both
  rs32 andPW36. mx/raw-pipeline/internal-loader gains have weaker/different
  controls or are alternatives superseded by measured selected paths. Sol16's
  independent completeness inventory should be checked before saying the entire
  filesystem has been exhausted.
- User's stack objective is concrete: preserve both immutable proven source
  branches, validate a complete candidate under actual fresh dispatch and all
  required shapes, and submit/promote only on a valid official combined result.
  PW36 already supplies one combined official result; renewed work must record
  its new source/policy/measurement identity and explain what changed. No new
  score can be manufactured from best-per-shape numbers across separate jobs.

Machine refs are reproducible from primary archived JSON/stdout rather than
handover conclusions. Source hashes in older job reports may identify paths
rather than server-attested uploaded digests; local frozen sources/handles and
history provide lineage, not cryptographic server-side execution attestation.

### Independent completeness/source follow-up

[Sol11 exact source stack](current-leaderboard-stack-source.md) independently
maps18 components/all30 active profile routes; [Sol16 completeness audit](current-leaderboard-gains-completeness.md)
adds a genuine omitted small local refinement: historical rs128b EH bit65536,
reported full-kernel CUPTI -0.13..-0.16us versus R4, and recovered single qkv bake
17.25→16.96. It is absent from rs32 and distinct from EARLY bit32. Pooler and
unpooler EH bakes lost in the recovered run; raw CUPTI repetition distributions
remain missing. This is an omitted local option requiring current-source/
actual-incumbent confirmation, not a forgotten established leaderboard gain.
It must be retained despite its small magnitude. New optional backport/PW38
is a new source awaiting current combined runtime validation, not retroactive
shipping admission of the historical result.

Source identity is supported by local a7b6 equality to git fbf6371, surviving
submit_rs32.sh naming that commit/candidate and historical launch records. The
server content-hash scheme is unknown and no raw uploaded payload attestation
was recovered; local file equality alone cannot cryptographically attest the
server payload. This limitation does not license relabeling other sources as
rs32 or changing frozen artifacts.
