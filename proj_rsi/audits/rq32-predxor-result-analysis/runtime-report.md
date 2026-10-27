# RQ32 predxor collected local comparison

Actual full30 policy geometric load speedup: paired **1.0056249783832558**, F1 **1.0030822041573286**. Both local point estimates are positive. This is one fresh prepare per source; it does not rescale or replace the official PW45 score3179.285623687654.

Collected run completed; 148 correctness cases, all four affected M32 all-K/FP64 checks, fixed task/fresh escaped-weight validation, 3810 actual and590 fixed timing rows pass the existing harness. Raw actual journal equals result rows. Independently rehashed all10 collected JIT original/gated CUDA, SASS and SO files; source/review identities match. All120 actual own-terminal seeds and240 mutation receipts are present and successful. Exit0, no JIT errors, clean graph cleanup, no live containers/processes and0 GPU memory at collection.

Positive deltas below mean PW45/parent is slower; every signed result is retained. Actual pooler32/unpooler32 both select matched ST5; QKV32 both ST6. O32 changes parentST6 to newGV45, so its loss is a policy/selection effect. Fixed poolerST5 paired loss versus F1 gain remains explicit. Fixed arms have separate SO/allocation placement effects; no isolated cycle claim.

| Actual shape | Paired gain µs | F1 gain µs |
|---|---:|---:|
| decode1/qkv_gate | -0.111999922 | -0.144000631 |
| decode1/o | +0.159999356 | +0.032000244 |
| decode1/pooler | +0.095998868 | -0.032000244 |
| decode1/unpooler | +0.079999911 | +0.000000000 |
| decode1/router | -0.064000022 | -0.032000011 |
| decode16/qkv_gate | +0.000000000 | +0.080000143 |
| decode16/o | +0.000000000 | -0.063999556 |
| decode16/pooler | +0.111998990 | +0.175999478 |
| decode16/unpooler | +0.047999434 | +0.111999922 |
| decode16/router | +0.048000133 | +0.031999778 |
| decode32/qkv_gate | +0.064000022 | +0.096000265 |
| decode32/o | -0.064000022 | -0.208000187 |
| decode32/pooler | +0.144000165 | +0.128000043 |
| decode32/unpooler | +0.287999399 | +0.464001670 |
| decode32/router | +0.032000244 | -0.096000731 |
| verify1/qkv_gate | +0.176000409 | +0.351999421 |
| verify1/o | -0.128000043 | -0.128000043 |
| verify1/pooler | -0.032000244 | +0.239999034 |
| verify1/unpooler | +0.159999821 | -0.159999821 |
| verify1/router | -0.032000011 | +0.000000233 |
| verify16/qkv_gate | +0.112000853 | +0.031999312 |
| verify16/o | +0.031999778 | +0.047999900 |
| verify16/pooler | +0.031999312 | -0.064000487 |
| verify16/unpooler | -0.160000287 | +0.032000244 |
| verify16/router | +0.064000022 | +0.208000187 |
| verify32/qkv_gate | +0.575998798 | +0.176000409 |
| verify32/o | -0.015999656 | -0.128000043 |
| verify32/pooler | +0.864000060 | +0.656000338 |
| verify32/unpooler | -0.016000122 | -0.304000452 |
| verify32/router | -0.063999556 | -0.000000466 |

| Fixed layer/contrast | Paired gain µs | F1 gain µs |
|---|---:|---:|
| pooler/parent5_minus_new5 | -0.079999678 | +0.144000165 |
| pooler/parent6_minus_new6 | +0.319998711 | +0.480000861 |
| pooler/parent5_minus_parent6 | -0.160001218 | -0.128000043 |
| pooler/new5_minus_new6 | +0.223999843 | +0.208000652 |
| unpooler/parent5_minus_new5 | +0.272000208 | +0.288000330 |
| unpooler/parent6_minus_new6 | +0.128000043 | +0.064000487 |
| unpooler/parent5_minus_parent6 | -0.016000122 | +0.096000731 |
| unpooler/new5_minus_new6 | -0.239999965 | -0.127999112 |

Paired actual shapes:18positive/2ties/10negative; F1:17positive/1tie/12negative. Opposite signs remain on decode1/pooler, decode32/router, verify1/pooler, verify1/unpooler, verify1/router, verify16/pooler and verify16/unpooler.

The next decision can use the positive actual fullpolicy point with retained local regressions and diagnostic disagreement. No significance/minimum-gain hurdle is imposed. Fullpolicy gains include fresh prepare selection differences, especially verify32. Official/H200 transfer remains unmeasured.
