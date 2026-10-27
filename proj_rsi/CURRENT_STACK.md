# Current leaderboard stack: gv_predxor_pw45_v2

Board-verified scored job `9916d34357b745cf`: **3157.0384036526784us**,9/9 and30 timed shapes. Source `501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9`. PW45 + standalone GV predicated escape repair alternatives with original controls and actual prepare selection retained. No redundant preceding CHECK was required for this scored GV release.

The old [rs32 source audit](audits/current-leaderboard-stack-source.md) and [official history](audits/current-leaderboard-gains-official.md) are HISTORICAL snapshots retained unchanged. Current bindings below are regenerated from this frozen current release source and this scored job; they are not copied CHECK selections.

|Component|Source/selection evidence|
|---|---|
|Lossless 12-bit BF16 codec with escapes|SOURCE_PRESENT; actual applicability is in scored routes; 5 current AST/source bindings; no isolated causal delta.|
|Prepacked MMA fragment layouts|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Register-direct GV stream and fast decode|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|GV local K reduction and retained fallback reductions|SOURCE_PRESENT; actual applicability is in scored routes; 1 current AST/source bindings; no isolated causal delta.|
|PW33 short per-warp QKV chains|SOURCE_PRESENT; actual applicability is in scored routes; 1 current AST/source bindings; no isolated causal delta.|
|P1 split-K with in-kernel zeroing|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Direct FP32 router GEMV|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|Router BF16 hi+lo decomposition and FP32 accumulation|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|Old RW cp.async/WGMMA router|SOURCE_PRESENT; actual applicability is in scored routes; 6 current AST/source bindings; no isolated causal delta.|
|RS128/RQ swap-AB producer/math pipeline|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Staged register escape cursor|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|RQ fused zeroing, STMatrix and TMA reduction|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|RS32 small-token specialization|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Early first loads and L2 hints|SOURCE_PRESENT; actual applicability is in scored routes; 2 current AST/source bindings; no isolated causal delta.|
|64-row HALF/fullK plain-store alternatives|SOURCE_PRESENT; actual applicability is in scored routes; 3 current AST/source bindings; no isolated causal delta.|
|cuBLASLt tuning and valid fallback paths|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Shape-exact routing, dimensions and safe fallback|SOURCE_PRESENT; actual applicability is in scored routes; 4 current AST/source bindings; no isolated causal delta.|
|Parallel build, exact validation, bounded F1 bake|SOURCE_PRESENT; actual applicability is in scored routes; 6 current AST/source bindings; no isolated causal delta.|
|PW35 one-node TMA RTK router128/256|SELECTED_IN_SCORED_PROFILE; 5 current AST/source bindings; no isolated causal delta.|
|Optional EH early-half-slab epilogue bit65536|SOURCE_PRESENT_NOT_SELECTED_IN_SCORED_PROFILE; 4 current AST/source bindings; no isolated causal delta.|
|Optional o32 HALF52 fullK ST4/ST5|SOURCE_COMPILED_BUT_UNSELECTED_IN_SCORED_PROFILE; 6 current AST/source bindings; no isolated causal delta.|
|Standalone GV predicated escape XOR repair|SELECTED_IN_SCORED_PROFILE; 4 current AST/source bindings; no isolated causal delta.|

|Load / layer / M|Actual scored family / prepare pick|EH / HALF / EARLY flags|
|---|---|---|
|decode 1 / qkv_gate / 1|gv / See actual kernel profile|False / False / False|
|decode 1 / o / 1|gv / See actual kernel profile|False / False / False|
|decode 1 / pooler / 1|gv_predxor / See actual kernel profile|False / False / False|
|decode 1 / unpooler / 1|gv / uk-s4w4x|False / False / False|
|decode 1 / router / 1|rr / See actual kernel profile|False / False / False|
|decode 16 / qkv_gate / 16|rq / rq:t16s5hze-k7|False / False / True|
|decode 16 / o / 16|gv / r6|False / False / False|
|decode 16 / pooler / 16|gv_predxor / r6|False / False / False|
|decode 16 / unpooler / 16|gv_predxor / r6|False / False / False|
|decode 16 / router / 16|gv / rk-s16w8z|False / False / False|
|decode 32 / qkv_gate / 32|rq / rq:t32s5hze-k7|False / False / True|
|decode 32 / o / 32|rq / rq:t32s5ze-k2|False / False / True|
|decode 32 / pooler / 32|rq / rq:t32s6hze-k4|False / False / True|
|decode 32 / unpooler / 32|rq / rq:t32s6hze-k2|False / False / True|
|decode 32 / router / 32|gv / rk-s8z|False / False / False|
|verify 1 / qkv_gate / 8|gv / See actual kernel profile|False / False / False|
|verify 1 / o / 8|gv / See actual kernel profile|False / False / False|
|verify 1 / pooler / 8|gv_predxor / See actual kernel profile|False / False / False|
|verify 1 / unpooler / 8|gv / uk-s4w4x|False / False / False|
|verify 1 / router / 8|gv / rk-s16w8x|False / False / False|
|verify 16 / qkv_gate / 128|rq / rq:s4s5hz|False / False / False|
|verify 16 / o / 128|Lt/vendor / r6|False / False / False|
|verify 16 / pooler / 128|rq / rq:s4s5hz|False / False / False|
|verify 16 / unpooler / 128|Lt/vendor / r6|False / False / False|
|verify 16 / router / 128|rtk / rt64k32b3x2z|False / False / False|
|verify 32 / qkv_gate / 256|Lt/vendor / r6|False / False / False|
|verify 32 / o / 256|Lt/vendor / r6|False / False / False|
|verify 32 / pooler / 256|Lt/vendor / r6|False / False / False|
|verify 32 / unpooler / 256|Lt/vendor / r6|False / False / False|
|verify 32 / router / 256|rtk / rt64k32b3x2z|False / False / False|

Full per-route kernel reports, exact current source-node SHA256/line bindings and historical equivalence checks are in [machine view](current_leaderboard_stack.json). Source presence does not imply measured benefit on every route. Server payload hash remains opaque; no component-wise or cross-run additive gain claim is made.

PW45 historical source/routes are preserved byte-for-byte at `results/proj-rsi/releases/pw45-v1-835acb90b47b/historical-current-stack-before-gv.json`. Selected standalone GV predxor routes: **4**.

|Shape|Previous PW45 kernel|Current scored kernel|
|---|---|---|
|decode 1/pooler M1|void gv<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|void gv_predxor<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|
|decode 1/unpooler M1|void gv<1, 1, 2, 1, 8, 2, 4, 1, 0, 536870913>(G)|void gv<1, 1, 4, 1, 4, 4, 4, 1, 0, 536870913>(G)|
|decode 16/pooler M16|void gv<1, 2, 2, 2, 8, 16, 1, 1, 0, 1>(G)|void gv_predxor<1, 2, 2, 2, 8, 16, 1, 1, 0, 1>(G)|
|decode 16/unpooler M16|void gv<1, 2, 2, 1, 4, 16, 1, 1, 2, 1>(G)|void gv_predxor<1, 2, 2, 1, 4, 16, 1, 1, 2, 1>(G)|
|decode 16/router M16|void gv<2, 2, 1, 1, 8, 1, 16, 1, 0, 536870912>(G)|void gv<2, 2, 1, 1, 8, 1, 16, 1, 0, 134217728>(G)|
|decode 32/pooler M32|void rq<32, 5, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|void rq<32, 6, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|decode 32/unpooler M32|void rq<32, 5, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|void rq<32, 6, 1, 0, 0, 0, 35>(RqP, RqTmap, RqTmap)|
|verify 1/o M8|void gv<1, 1, 2, 1, 8, 2, 1, 1, 0, 0>(G)|void gv<1, 1, 2, 1, 8, 2, 1, 1, 0, 1>(G)|
|verify 1/pooler M8|void gv<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|void gv_predxor<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)|
|verify 1/unpooler M8|void gv<1, 1, 2, 1, 8, 2, 4, 1, 0, 536870913>(G)|void gv<1, 1, 4, 1, 4, 4, 4, 1, 0, 536870913>(G)|
|verify 16/o M128|nvjet_sm90_tst_128x64_64x8_1x1_v_bz_coopB_NNN|nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNN|
|verify 16/unpooler M128|nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN|nvjet_sm90_tst_128x64_64x8_1x1_v_bz_coopA_NNT|
|verify 32/o M256|nvjet_sm90_tst_256x64_64x5_1x2_v_bz_coopB_NNN|nvjet_sm90_tst_128x128_64x6_2x1_v_bz_coopA_NNN|
|verify 32/unpooler M256|nvjet_sm90_tst_128x128_64x6_1x2_v_bz_coopA_NNT|nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN|

Latest complete compatible GV+RQ+privateEH+O32 source d1ca was measured locally against this GV source and officially scored as `161bdb8de7354344`: **3187.14239967648us**,9/9; it did not replace GV. The local whole-package positive point does not establish official benefit. EH+O32 c425 was measured versusEH parent only. [Exact stacking provenance and all branch outcomes](audits/submission-stack-20261027/report.md). Latest archived board rank: **62 / 1173**.

Latest independently archived official CHECKs (receipt + own completed result; distinct from board authority): `8e9cba04c9d644c8` 3221.2172600872086us, `323514d58af84e72` 3173.2818089965185us, `abbaf85681f74e53` 3178.0129977697934us, `53e707b4e7ed470e` 3200.8783202089357us, `dc78c2473d204a92` 3191.721918275971us, `ac49231d41d441f6` 3225.893245547708us, `71a76a2c3fcd431f` 3182.3487733684065us, `fe5c45197ca041bb` 3196.414179553723us. All passed9/9; none is a scored submission. M1-only full30 and focused unpooler256 local regressions remain separate in [history](HISTORY.md). MoE LUT whole improves its SS parent but remains slower than its actual anchor; it is separate local research with no projection leaderboard claim.

Complete GV+RQ+privateEH+O32 plus M1-only optional splitCTA scored `ae061d458eaa420a`: **3186.7445157626275us**,9/9, source `55f7fb523307a577bda46ddb10ceaee8f8f13e148adf525aade9e415c1f0e456`. It remains above GV best. Its separate-job point is0.397883914us below d1ca parent; this is not an isolated causal feature gain. Local incremental versus d1ca paired1.001786585/F1average1.000614972/conservative0.995125856 conflict; no new tuple was selected locally. This scored result does not itself supply board authority; the current rank comes from the separately archived board capture.

Latest isolated projection CHECKs: keyed-half `ac49231d41d441f6` **3225.893245547708us**,9/9, new kernel actually served; O32-only `dc78c2473d204a92` **3191.721918275971us**,9/9, root REFINE. Both lose the incumbent whole-policy point. Local GV53/RQ scoped positives, their negative controls, and GV16 paired/F1 conflict remain separately preserved in [history](HISTORY.md). Composed f018 CHECK `71a76a2c3fcd431f` completed **3182.3487733684065us**,9/9; newRQflags131107 and keyedM8 selected, M1 chose old unrolledpredxor. This actual combined measurement still loses the incumbent and establishes no measured-gain stack. [Projection-only live ROI](ROI_QUEUE.md).

Optional keyed-choice CHECK `fe5c45197ca041bb` completed **3196.414179553723us**,9/9, source78f9; same-H200-prepare target comparison M1+0.0480003655us/keyed, M8−0.0160001218us/parent. WholeCHECK loses; rootREFINE. GV16 three-weight reuse-SO robustness median−0.063999556us remains a separate diagnostic regression. [Live projection ROI](ROI_QUEUE.md).

Early-load local job `research.gv53.earlyload.fixed.01` completed VALID_GAIN: fixedM8 paired+0.192000531us,15W9L; both order medians and F1 endpoints positive. Source81fd is not the scored incumbent. M1keyed+M8earlyload production composition is being built; no official or combined-policy result exists for it. [Current live ROI](ROI_QUEUE.md).
