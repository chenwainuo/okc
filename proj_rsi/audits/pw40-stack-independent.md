# PW40 independent full-scope result review

**VALID_GAIN; GO for the next official unscored check of exact source `3bc0df52d96f62479c3d5c19ab2717c7a9f4d18c1dd087fb786e2f9f080f0278`.** Both local whole-load point observations favor PW40 over PW38: paired control/PW40 ratio `1.0031524075112948` and conservative bracketed F1 ratio `1.0011109673097403`. No minimum-size threshold was applied. These are local H100 observations; official performance remains unmeasured. Root owns execution and scheduler finalization.

All63 collected files and127 dispatched source pins match their SHA256 values. Terminal exit is0; state and phase are complete, with no JIT errors or pending graph cleanup. Prepare charged original import clocks (RS32 60.2964s, PW38 63.1868s, PW40 63.3713s), reset no budgets, used no CPU serving fallback and joined both background builders for each arm. H10080GB,132SM, SM90, device0 and UUID54a7daab-fe3e-c7fc-7f7c-6fbc40145a22 were recorded before PW40 import/prepare. Device-gate source identity matches the frozen candidate.

Fifteen actual extension-build records include all17RS32,19PW38 and20PW40 RQ functions. I reexecuted canonical PC/predicate/instruction matching and resource gates for every RQ function, EH lifetime gates, and the new Wring physical gate from collected CUDA/SASS/resource dumps. All19 prior PW40 controls match the same canonical streams as PW38. The appended query is exactly the pinned host-only suffix; removing it restores each recorded production CUDA source. Recorded SO hashes are well-formed; this collection does not contain shared objects to redump independently. Every gate was recorded before returning to prepare math.

Wring actual attributes are168registers,0local bytes,148656shared,384max threads and occupancy1 after opt-in. Static rounded registers168×384=64512 equal the dynamic232×256+40×128 pool, below65536. Its126CTA grid fits132SM×1; max dynamic shared148656 matches requested allocation. Physical gates retain X6/W4,16HGMMA,8outputSTSM, one reduce issue, finalwait0 before output, TMA-read drain and bounded-poll trap. The independent ring/lifetime and source proof remain pinned.

**The actual artifact records70 Wring preparation calls, all at128×2304/KS7 with return0.** The initial80-call summary is corrected here. `Wring_selected=[]`; PW40 EH was selected for verify16 qkv_gate/pooler/unpooler. Successful compilation, resource queries and attempted bake do not establish an isolated Wring gain. Selected picks also differ on other cells, so full-policy deltas cannot be assigned to one mechanism.

I checked all222 correctness records (65normal plus9official cases per arm), all4350 finite raw samples against the append-only journal, all30 shapes and all six balanced three-arm timing orders repeated five times. Each shape has five11-sample F1 blocks and30paired rounds. Every reported median, signed delta and ratio recomputes exactly. All180 immediate terminal records identify the correct final fixture (F1 seed9010 and paired seed9049), and all360 retained-graph mutations cover zero/outliers/all_probes/fresh_normal for each arm with output and all float scratch poisoned. Reviewed source executes the numerical oracle, current-X immutability, immutable-state and source-derived full counter-arena/alias checks before any replay or mutation can replace timed outputs. Selected RQ serving returns are strictly zero. Graphs drain then reset. This CPU review verifies the recorded evidence and exact hooks; it does not claim to recompute GPU tensors.

Cleanup observation reports no container, compute PID or guard PID, zeroGPU memory, no cleanup failure, and terminal0. Eight independent negative controls reject wrong terminal seed, fabricated gain, failed Wring status, insufficient SMs, missing function, false Wring selection, zero occupancy and production-source drift.

The ratios below use sums of per-shape paired medians within each of six loads, then a geometric mean across loads. F1 uses the faster of the control-before/control-after medians per shape. This preserves the conservative bracket. Separate-window load sums are local screening, not an official aggregate measurement.

| Load | RS32 us | PW38 us | PW40 us | PW38−PW40 us |
|---|---:|---:|---:|---:|
| decode1 | 59.471999295 | 59.232000727 | 59.023999376 | +0.208001351 |
| decode16 | 76.751998626 | 77.168000396 | 77.247999609 | -0.079999212 |
| decode32 | 78.960000537 | 79.104000703 | 79.071999528 | +0.032001175 |
| verify1 | 60.655999929 | 60.880000703 | 60.719999485 | +0.160001218 |
| verify16 | 89.312000200 | 88.239998557 | 87.695999071 | +0.543999486 |
| verify32 | 106.672001071 | 105.071998667 | 104.319999926 | +0.751998741 |

All shape outcomes are retained below; positive signed differences favor PW40. Raw values remain unrounded in the JSON. Tiny differences, losses and ties are not filtered.

| Shape | Paired RS32−PW40 us | Paired PW38−PW40 us | PW38 wins/ties/losses for PW40 | F1 PW38−PW40 us |
|---|---:|---:|---:|---:|
| decode1/qkv_gate | -0.080000143 | +0.000000000 | 13/3/14 | -0.160000287 |
| decode1/o | +0.160000287 | +0.095999800 | 21/5/4 | +0.095999800 |
| decode1/pooler | -0.000000931 | +0.160000287 | 17/1/12 | +0.512000173 |
| decode1/unpooler | +0.351999886 | -0.032000011 | 13/1/16 | +0.159999821 |
| decode1/router | +0.000000000 | +0.032000011 | 17/0/13 | +0.000000000 |
| decode16/qkv_gate | -0.079999678 | -0.159999821 | 8/0/22 | -0.032000244 |
| decode16/o | -0.032000244 | +0.047999900 | 17/0/13 | +0.000000000 |
| decode16/pooler | -0.128000043 | +0.144000165 | 18/0/12 | +0.064000487 |
| decode16/unpooler | -0.303999521 | -0.079999678 | 14/0/16 | +0.064000487 |
| decode16/router | +0.031999778 | +0.047999900 | 16/5/9 | -0.160000287 |
| decode32/qkv_gate | +0.272000208 | +0.335999765 | 25/1/4 | +0.191999599 |
| decode32/o | -0.032000244 | +0.000000000 | 14/2/14 | +0.032000244 |
| decode32/pooler | -0.176000409 | -0.159999356 | 10/0/20 | -0.543998554 |
| decode32/unpooler | +0.032000244 | +0.031998381 | 16/1/13 | +0.063998625 |
| decode32/router | -0.015999656 | +0.000000000 | 14/3/13 | -0.223999843 |
| verify1/qkv_gate | -0.064000022 | -0.095999800 | 11/1/18 | -0.191999599 |
| verify1/o | +0.128000043 | +0.000000000 | 14/3/13 | +0.063999556 |
| verify1/pooler | -0.064000487 | +0.176000409 | 23/1/6 | +0.032000244 |
| verify1/unpooler | -0.160000054 | +0.000000000 | 13/3/14 | +0.032000244 |
| verify1/router | -0.063999789 | -0.064000022 | 8/4/18 | -0.064000022 |
| verify16/qkv_gate | +0.031999312 | -0.016000122 | 14/1/15 | +0.063998625 |
| verify16/o | +0.096000731 | +0.192000531 | 28/0/2 | +0.032000244 |
| verify16/pooler | +0.048000365 | +0.079999678 | 15/1/14 | -0.159999356 |
| verify16/unpooler | +0.287999399 | +0.448000617 | 24/0/6 | -0.320000574 |
| verify16/router | +1.168000046 | +0.064000022 | 18/2/10 | +0.032000244 |
| verify32/qkv_gate | +0.175999478 | +0.287999399 | 23/0/7 | -0.095998868 |
| verify32/o | -0.016000122 | +0.175999478 | 20/2/8 | +0.128000043 |
| verify32/pooler | +0.384000130 | +0.256000087 | 18/1/11 | +0.319998711 |
| verify32/unpooler | +0.031999312 | +0.176000409 | 20/1/9 | +0.256000087 |
| verify32/router | +1.711999997 | +0.016000122 | 15/1/14 | +0.383999199 |

Against RS32, paired local ratio is1.0065552403677742 and conservative F1 ratio1.008638844266225. This trial supplies favorable measured full-policy observations against both historical RS32 and actual PW38. It does not prove repeatability across fresh preparations, isolated Wring benefit, H200 transfer or an official score. The exact positive result is retained under the user’s policy without a0.5us threshold.

Reproduce with `python3 proj_rsi/audits/pw40-stack-independent.py`. This review performed local CPU reads/analysis only; no GPU, API, queue, frozen source or candidate mutation occurred.
