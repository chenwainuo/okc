# Shared-X/SS ancestry: pw19 versus reversed operands

`scope=CPU_SOURCE_COMPARISON_ONLY; implementation=NONE; family=ACTIVE; source_counting!=hardware_ceiling`

Read actual pw19`submission.py` pwt/PwT/launcher/selector, its raw judge log09e3c6f672a24c71, current reverse kernel and historical pw19 critic/pipeline accounting. No GPU/new kernel/source edit. Historical aggregate/shared-port models and old family-wide closure language are not authority; current ITERATION.md applies.

|property / unpooler128|old pw19 tw64|current reverse RS|hypothetical reversed SS|
|---|---|---|---|
|CTA logical tile|64channels×128tokens|same|same|
|MMA roles|W shared-A/X shared-B|X register-A/W shared-B|X shared-A/W shared-B|
|math arrangement|one WG,m64n128|two WGs,m64n64,token halves|same two WGs,m64n64|
|threads|128math+128decode+32producer=288|256math+128decode=384|384if elected decode lane also feeds X|
|split/cluster for THIS shape|KS1/no cluster|KS1/no cluster|could preserve KS1/no cluster|
|rings|W8/X4/decodedW4|W6/decodedW8|adding X4 gives W6/X4/decodedW8|
|output|FP32shared stage→shared BF16transpose→16Brow stores|native token/channel D→one plain TMA|could preserve native D/plain TMA|
|X logical demand/launch|128MiB|128MiB|128MiB with unchanged tile|

**Historical source correction:** pw19 raw log lines207–215 records `pw-tw64 unpooler m=128 128x1`,43.46us vs same-run torch23.58; tw64x2 is128×2/cluster2/35.30us; tw128 is64×2/35.33. The familiar tw6443.74/pipeline43.8 ladders were pooler128/KS2. Therefore no-cluster/KS1 unpooler SS was already executed. Those old H200/prepare numbers are historical controls, not predictions or paired comparisons to current H10024us incumbent.

Meaningful distinctions remain if a new proposal swaps **operand orientation/output**, preserves modern W6/B8/escape protocol and uses two cooperative token WGs. Old Wshared-A descriptor already used canonical K-major SW128 and Xshared-B was TMA-staged in the same16KiB/token128×k64 layout. Merely staging both and calling SS repeats pw19's core mechanism; swapping descriptor roles makes the native D token×channel, changing actual instruction geometry and eliminating its transpose epilogue. Old KS1 still did FP32 shared staging/read, BF16 shared transpose/read and vector global stores; it did not have current reverse's native-output TMA. Old decode used four stmatrix operations,128individual afull arrivals, separate busy-polling producer warp; current reverse uses scalar decoded-B stores, producer-only128barrier/fence then elected publication, cached32entry escape windows and no extra producer warp. These differences need source/resource proof rather than being labeled an unchanged rerun.

Nominal k64 shared bytes for the SAME64channel×128token tile:

|source operation|pw19 SS|reverse RS|reversed SS with TMA-X|
|---|---:|---:|---:|
|packed W fill+LDS|12KiB|12KiB|12KiB|
|decoded W store|8KiB|8KiB|8KiB|
|decoded W operand reads|8KiB|16KiB|16KiB|
|X shared fill+operand read|32KiB|0|32KiB|
|total|60KiB|36KiB|68KiB|

Two token WGs each reread the decoded W tile; new SS therefore is **68KiB**, not simply old60KiB. At64stages×128CTAs this is nominal544MiB versus historical480MiB/currentreverse288MiB, excluding output/control and without asserting a measured aggregate-port ceiling. A W6/B8/X4 ring would occupy164KiBdata plus alignment/barriers, conceptually below Hopper's opt-in cap; actual compile/occupancy/lifetime remain unproven. X4 repeats old64KiBlookahead, so it does not by itself remedy the historical X-ingress-depth concern. Directly adding X TMA also needs an independently correct X-full/empty ring and cannot assume an elected decode lane feeding two rings avoids head-of-line coupling.

Current v1/v2 counter evidence gives a concrete motivation—many LDG/frontend requests and X-dependent scoreboard exposure with roughly flat DRAM—but not a guarantee that increased shared demand wins. Useful future first question: does eliminating X-LDG/demux/address work with native-output SS outweigh the added32KiBshared/stage and extra X barrier protocol? Compare actual incumbent plus a source-specific predecessor and real full-output pipeline; do not infer a win from a resident MMA/probe or port model. A native-output SS candidate is a distinct possible iteration, but its novelty and opportunity must be these explicit differences, not 'pw19 lost only because of clusters' or a blanket family dismissal.
