# Official selected-earlyload result

Job 40d856e085274f51; source17d44d11. Statusok, passed9/9; score3175.899650724us. Root owns promotion; analyzer makes no API/queue/release mutation.

## Actual selected targets

- decode 1 M1: keyed; selected names: ['void gv_keyed<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)'].
- verify 1 M8: predxor; selected names: ['void gv_predxor<1, 1, 1, 2, 16, 8, 1, 1, 0, 3>(G)'].

Receipt arithmetic verified againstall4ABBApairs andofficial-status.output.log referenced archive: M1keyed selectedat+.016000122us; M8parent retainedatparent−early−.016000122us. BothCOMPLETE_ZERO_THRESHOLD. M8earlyloadwasactuallycomparedintuningbutnotselectedforofficialprofile; no officialearlyloadkernelperformanceeffect canbeassigned.

Prepare-choice receipts: PRESENT. Actual native profile execution is separate evidence from tuning choice margins/status; absent stdout cannot establish why a choice won/skipped.

## Cross-job descriptive comparisons

Versus9916d34357b745cf: old3157.038403653→new3175.899650724us; old−new-18.861247072us.

|Target|Rawold−new us|Calibratedold−new us|Baseline raw old→new us|
|---|---:|---:|---:|
|decode 1 poolerM1|-0.032000244|-0.123427167|25.504000→25.472000|
|verify 1 poolerM8|-0.064000487|-0.231755866|25.264001→25.440000|

Changed native routes across all30cases: 10. Native names ignore rounded profiler duration; full reports and every raw/calibrated case remain in analysis.json.

Versusfe5c45197ca041bb: old3196.414179554→new3175.899650724us; old−new+20.514528829us.

|Target|Rawold−new us|Calibratedold−new us|Baseline raw old→new us|
|---|---:|---:|---:|
|decode 1 poolerM1|-0.223999843|-0.317133188|25.200000→25.472000|
|verify 1 poolerM8|-0.096000731|-0.157363857|25.152000→25.440000|

Changed native routes across all30cases: 13. Native names ignore rounded profiler duration; full reports and every raw/calibrated case remain in analysis.json.

## Actual unpooler M16 eligibility

Currentjoband9916 selectsame gv_predxor tuple[1, 2, 2, 1, 4, 16, 1, 1, 2, 1], decode16M16,N8192/K4096. Currentraw18.464000896us versusold18.527999520, calibrated14.375760897 versusold14.345174506. Rawold−new+0.063998625us whilecalibratedold−new-0.030586391us; baseline/contextdiffer. Exact-tupleoptionalXprefetcheligibilitymatches thisactualjob; futurepreparemayselectanothergeometry. Roundedprofile15.4versus14.2us remainsseparate.

RootrecordedREFINE:9916remainsincumbent; actualscore3175.899650724>3157.038403653andtarget<3000notmet. Preserve localbodygains andofficialchoicefail-to-expose distinction; no familydiscard.

These are separate jobs, with calibration/baseline/prepare/device context differences. Rawtiming, calibratedscore, roundedprofileGPUduration and localH100fixedcells are distinct estimators. No isolatedearlyloadcausality, additivegain or three-way same-window comparison is claimed. Local mixedM1−.048/M8+.032us points remain preserved. ActualsourcecontainsoptionalM1parent/keyed andM8parent/early; no keyed+earlyload body.

Loop: profile→measure→hypothesize→optimize→test→validate→stack→submit measured gains→learn→reprioritize. [Contract](../../ITERATION.md).
