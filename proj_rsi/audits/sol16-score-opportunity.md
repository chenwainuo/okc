# Sol16 exact score opportunity and coverage audit

CPU-only arithmetic and source review. No GPU/context/prepare/extension load, benchmark, global queue/ledger edit or commit. Reproduce with `python3 proj_rsi/audits/sol16-score-data/recompute.py`.

Authority is **rs32 / fb548b5dc6f04696 / 3200.6150084373335us / 9 of 9 / speedup 1.509334920715312**. Raw SHA256 `f3a8e1c1339cbaf764d0f9024b7d4c9273dc73c15788f425c8c16e6b7a037fd7`; frozen source SHA256 `a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad`. JSON preserves input hashes, all 30 scored component latencies, same-job baseline components, actual printed preparation picks, actual timed profiles, per-shape derivatives and zero-component bounds. Preparation bake times and profile times are separately labelled and never substituted for scored `median_ns`.

The exact score is `4830.8 / geomean(same_job_SGLang_load_sum / proposed_load_sum)` across six equally weighted loads. Both load sums multiply by62; it cancels in ratios. Equivalent fixed-reference law: `Snew=S0*exp(sum(log1p(-delta_load/Tload))/6)`. Positive delta is genuine complete-shape improvement; negative delta is a hypothetical regression. Resulting component latencies cannot be negative. Zero component scenarios are mathematical ceilings, not attainable kernels. `62*geomean(ours)` is not the official score. Recomputed score agrees exactly with the recorded summary.

Latest detailed user target remains **3000us**. Active goal metadata's older “3x current benchmark” wording has an unspecified historical baseline. This audit does not replace either objective, mark achievement, or silently select a baseline. Anchoring “current” to the known official incumbent gives1066.871669479us (3x incumbent and4.528004762x published reference). If “3x” means published SGLang instead, target1610.266666667us. Which historical benchmark the old metadata meant remains unknown. The latest3ms milestone needs200.615008437us score reduction,6.268014363% balanced reduction across all six loads, or summed log-load gain0.388384157. The incumbent-relative3x target needs66.6667% balanced reduction and log gain6.591673732; published-relative3x needs49.6888% balanced reduction and log gain4.121659108.

|Load/M|ours sum us|same-job SGLang sum us|local score-us / genuine us|3ms cut if only this load changes|
|---|---:|---:|---:|---:|
|decode 1 / 1|35.291272317|59.343998786|15.115234|11.358433298|
|decode 16 / 16|48.292189824|75.632000808|11.046006|15.542755500|
|decode 32 / 32|52.139459709|76.752000488|10.230943|16.780992478|
|verify 1 / 8|37.485332530|75.824001804|14.230522|12.064587679|
|verify 16 / 128|67.511418316|86.688000243|7.901417|21.728430046|
|verify 32 / 256|86.584848719|101.663996466|6.160845|27.867179736|

The highest marginal value per genuine microsecond is M1, then M8, M16, M32, M128, M256. This is a derivative, not a constant conversion for large cuts. The largest absolute components are pooler25624.838us, unpooler25624.693us, unpooler12818.951us, pooler12818.399us and qkv25617.277us; larger latency does not by itself imply better attainable ROI. For3ms a single changed load must lose32.1848% of its total. For incumbent-relative3x, it must lose99.8628%; published-relative3x98.3782%. These statements alone do not rule out any physical design.

**Native-X only unpooler128 cannot close3ms:** zeroing its entire18.951374903us leaves3029.589194531us, while verify16 alone needs21.728430046us reduction. Improving additional shapes/loads is therefore necessary for this narrow family. It also cannot close either3x interpretation. No byte/floor prior is used as an impossibility proof.

Conditional, complete-shape arithmetic (all other components fixed):

|Group|each −1us regression delta|zero gain|each +1us genuine gain|each +3us genuine gain|equal cut each to3ms|
|---|---:|---:|---:|---:|---:|
|qkv16/qkv32/pooler32/unpooler32/o32|3251.376456|3200.615008|3147.322930|3031.507684|3.503737|
|qkv/pooler/unpooler at128 and256|3242.245660|3200.615008|3157.845952|3068.549774|4.456255|
|unpooler128 only|3208.468096|3200.615008|3192.664380|3176.459559|not evaluated here|
|qkv/pooler/unpooler256|3218.836231|3200.615008|3181.859854|3142.601209|not evaluated here|

No speedup estimate is available for the cursor, native-X branches, store composition or future ports against a same-hardware selected incumbent. The hypothetical1us cuts above are independent inputs, not additive cross-run measurements. The equal-cut model for either five smallm shapes or six large shapes cannot reach either3x target while keeping each component nonnegative; this is a bound on those restricted equal-cut scenarios only.

A second generous arithmetic ceiling sets one entire layer to zero at all six loads:

|Layer zeroed everywhere|remaining score us|
|---|---:|
|qkv_gate|2506.480316|
|o|2812.023621|
|pooler|2183.902853|
|unpooler|2450.255408|
|router|2829.425867|

Thus even eliminating any single entire layer at every M leaves above both3x interpretations. Broad multi-layer changes are necessary for those targets under this fixed benchmark; nothing here says that such changes are physically impossible. Pooler has the largest zero-layer ceiling, followed by unpooler, qkv, o and router. Per-shape zero ceilings in JSON offer a less extreme view.

Potential ROI ranking (conditional coverage weighted by current readiness; attainable gain UNKNOWN):

1. **Smallm cursor recurrence**: `candidates/smallm_increment/submission.py`, guarded NT16/32 GS1/PRB0/WM0/INS0/FL35. Official rs32 picks qkv16 `t16s5hze-k7`, qkv32 `t32s5hze-k7`, pooler32 `t32s5hze-k4`, unpooler32 `t32s5hze-k2`, o32 `t32s6hze-k2`. It reaches two high-sensitivity loads without a new architecture. Pooler16/unpooler16 retain r6 and are not covered. Current independently repaired focused driver initially admits qkv32 only; a qkv32 result does not validate all five shapes. Offline static instruction reduction is evidence of changed code, not latency benefit. Hypothetical1us in each five yields the table's value; no such gain is measured.
2. **Large multi-load coverage**, contingent on a useful seed mechanism: future qkv/pooler/unpooler128+256 has broader3ms arithmetic headroom. `candidates/group8_qkv` (see `sol05-qkv-coverage.md`) is a concrete CPU split4 QKV128 prototype,144CTAs with fresh pack/main/FP32-reduce, not runtime admitted. Full-K QKV128 has36CTAs; pooler12864CTAs; simple reuse underfills132SM H100. M256 needs new token/output/XP handling and more work; no free dispatch extension or doubled-tile speedup. Official256 r6/Lt controls must be retained. Coverage must spread to M1/M8/M16/M32 for a credible3x program; no current implementation promises this.
3. **Native-X U128 refinement** remains a useful seed but has the narrow hard ceiling above and a measured local deficit. Latest valid native-X31.856us versus actual selected local pw36 s4s5hzKS223.968us loses7.888us when subtracting policy medians (paired median difference−7.952us); prior DP34.096/DS35.312 progress cannot be booked as official gain. Exact paired gain and difference-of-medians are different summaries. Neither H100 number replaces the H200 official18.951us. Current async-refill/X4 branches have CPU proofs and resource evidence only. Store composition combines changed mechanisms and requires isolated controls before credit.

Information-value ranking (separate from speed ROI):

1. Admit the repaired focused qkv32 cursor lab only after independent root review: compare freshly prepared actual pw36, frozen rs32 and cursor version; inspect dispatch and actual JIT before math, retain actual terminal timed outputs. It resolves whether scalar ring bookkeeping is hidden. Then extend one admitted shape at a time to qkv16/pooler32/unpooler32/o32 if evidence warrants it. Local pw36 qkv16 can retain r6, so a retained path supplies no cursor evidence.
2. Finish bounded native-X profiling against actual selected incumbent and frozen parent. Determine X readiness/warp handoff versus producer/HGMMA/coverage limits without converting PC fractions to microseconds. `reverse_tma_x_async_refill` removes math bar5 via Xempty owner handoff; `reverse_tma_x4` isolates ring2→4 at increased shared footprint. Compare each separately before composing stores. Both remain GPU_UNRUN at the reviewed source snapshot; root is sole executor.
3. Require the seed to beat actual selected U128 or provide actionable mechanism evidence before a QKV128 split4 host wrapper/runtime trial. Measure full fresh-pack/main/reduction pipeline; this resolves whether added CTA coverage repays new nodes/workspace. Port M256 only with explicit packing/index/ownership proof and actual complete controls. Revisit group16 A1 only after group8 mechanism evidence; group16 A2 cannot fit the named512-thread uniform128-register policy before temporaries (128A+32D), and a different architecture is a separate proposal.

Next failure should trigger a specific diagnosis/new immutable version, not family retirement. No official submission or promotion follows this CPU audit. The current official score remains3200.615008437us and both3ms and3x objectives remain unmet.
