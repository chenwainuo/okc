# Combined X-prefetch CHECK: selected target, regressed package

CHECK38821b954fb14978/source48af8c79 passed9/9. Score3225.130858483us: +49.231207759us worse than17d44d/40d856 and +68.092454830us worse than incumbent9916. Target<3000 unmet. CHECK is not a scored-quota receipt or incumbent promotion; root owns disposition.

|Prepare option|Four-pair parent−option median us|Choice|Actual profile agrees|
|---|---:|---|---|
|GV_KEYED_CHOICE|+0.000000000|parent|yes|
|GV_EARLYLOAD_CHOICE|+0.143999234|earlyload|yes|
|GV_XPREFETCH_CHOICE|+0.095999800|xprefetch|yes|

All completed receipts reconcile exactly with four AB/BA pairs and zero threshold. New X-prefetch pair effects are+.384001061,+.191999599,−.352000818,0us; median+.095999800. M1 keyed tied0 and correctly retained parent; M8 early-load selected+.143999234. Actual profile executes gv_xprefetch for unpoolerM16, gv_earlyload for poolerM8, gv_predxor for poolerM1. Thus **source inclusion, tuning selection and actual execution are separately observed**; this failure is not an unselected-new-body case. Receipt text and profile reports are hash-linked in analysis-v1.json.

## Target timings

Positive old−new favors current; raw/calibrated/baseline microseconds are distinct estimators. Cross-job deltas are descriptive, not isolated effects.

|Comparator|Target|Raw old→new us|Calibrated old→new us|Baseline raw old→new us|
|---|---|---:|---:|---:|
|40d856e085274f51|pooler M1|17.664000→17.744000|13.838391→13.706738|25.472000→25.440000|
|40d856e085274f51|unpooler M16|18.464001→18.352000|14.375761→14.226864|22.608001→22.624001|
|40d856e085274f51|pooler M8|17.664000→17.712001|13.768730→14.012077|25.440000→25.504000|
|9916d34357b745cf|pooler M1|17.632000→17.744000|13.714964→13.706738|25.504000→25.440000|
|9916d34357b745cf|unpooler M16|18.528000→18.352000|14.345175→14.226864|22.560000→22.624001|
|9916d34357b745cf|pooler M8|17.600000→17.712001|13.536974→14.012077|25.264001→25.504000|

UnpoolerM16 improves raw+.112001/calibrated+.148897us versus17d44d and raw+.175999/calibrated+.118311us versus9916 while selected tuple/shape remain eligible. Preserve this encouraging target result and the earlier matched H100+.576us/21W3L evidence; do not label the cross-job H200 deltas causal or equivalent to the H100 estimate. M8 early-load raw/calibrated both regress versus parent in this job despite positive tuning margin; that is retained transfer counterevidence.

## Whole-package confounders and cost contributors

Both comparisons have16 changed native routes and all30 changed benchmark input seeds. Every exact old/new native name, full report, source receipt, raw/event/calibrated/reference/parent latency and case identity is retained in analysis-v1.json. No UNKNOWN-only explanation substitutes for these observed differences. Against17d44d, changed-route classes include three implementation choices(M1/M8/newM16), GV geometries/flags on unpoolerM1/routerM16/routerM32/O M8, RQ KS swaps at M32 and flags at M128, vendor algorithms at M128/M256, and routerM128 RTK. Original unmarked bake code is unchanged by exact source inverse, so these other route differences were observed under fresh timing/context rather than prescribed new routes. That does not identify their timing cause.

|Comparator|Dominant calibrated cost contributor|Raw old−new us|Calibrated old−new us|Route changed?|
|---|---|---:|---:|---|
|40d856e085274f51|o M128|-0.096000|-0.977428|False|
|40d856e085274f51|unpooler M256|-0.352001|-0.854573|False|
|40d856e085274f51|pooler M128|-0.464000|-0.578888|True|
|40d856e085274f51|router M8|-0.159999|-0.554831|False|
|40d856e085274f51|router M128|-0.064000|-0.477587|True|
|40d856e085274f51|router M256|-0.064000|-0.432925|False|
|9916d34357b745cf|unpooler M256|-1.552001|-1.856804|True|
|9916d34357b745cf|qkv_gate M32|-0.080000|-0.939671|False|
|9916d34357b745cf|pooler M128|-0.016000|-0.901632|False|
|9916d34357b745cf|pooler M256|-0.384001|-0.896702|True|
|9916d34357b745cf|o M128|-0.336000|-0.705195|True|
|9916d34357b745cf|unpooler M128|-0.335999|-0.566389|True|

Ranking uses case weight×calibrated latency difference **within load**, not additive whole-score microseconds. Unchanged-route regressions such as parent O M128(−.096raw versus−.977calibrated us) show calibration/context also matters; changed-route losses do not by themselves prove a bad bake choice. Baseline raw/calibrated, parent/outside/event values remain in JSON. No cross-job clock/thermal/CPU-underflow attribution is established.

## Failure diagnosis and next falsifier

- **PROVEN:**9/9 correctness success, exact released source, reviewed optional mechanism, completed choice arithmetic and actual new target dispatch. No observed setup/marker/shape failure. Passing checks does not exclude every implementation or codegen bug. Official SASS/resources were not archived; local body identity/resource proof is separate.
- **PROVEN:**whole-package score loss, selected target descriptively improves, M8 target regresses,16 routes/all30input seeds differ. No isolated new-feature cause is proven.
- **UNKNOWN:**how much whole loss comes from calibration, changed bakes, input variation, device history or implementation throughput. Neither the positive target nor negative package quantifies isolated X-prefetch contribution.
- **Smallest reopening question:**if root prioritizes isolating transfer, compare parent/new exact unpoolerM16 on the same H200 window, original actual inputs/math, private outputs/current-X terminals and identical timing protocol. Reuse the existing narrow ABI/body contract; no full30/local proof repeat is necessary. Separately audit fresh chooser/route variance from retained receipts rather than force a former winner or add a significance gate. No GPU proposal/launch is created here.
- **Disposition:**retain source48af and native body as measured iteration; full package is VALID_REGRESSION/REFINE, not incumbent promotion or family rejection. No minimum-gain gate, automatic unchanged rerun or additive local-gain claim. Root owns quota/CHECK/scored decisions.

Loop: profile → measure → hypothesis → optimize → test → validate → stack → submit → learn → reprioritize → repeat; idle≤10s. [Contract](../../ITERATION.md).
