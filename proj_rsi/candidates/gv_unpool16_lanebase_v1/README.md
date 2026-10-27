# GV16 lane-base hoist v1

Projection only, parent713c lane-relative key. A single opaque PTX mov identity makes lane_base an explicit value before the kernel step sequence. Replace only `(uint32_t)lane<<6` in that key subtraction with lane_base. No cursor/load/repair arithmetic/HMMA/geometry/codec/selector change. Exact inverse proof is in proof.json; predecessor domain/native proofs reused.

Focused two-function CPU compile: control501d full PC/predicate/128bit encoding equals both prior pilot and collected actual SO. New instructions1960 versus predecessor1992/control2080; S2R5 versus20/5; registers108 versus108/110, stack/spills0. SHFL17/P2R16 unchanged. The rematerialization mechanism changed as intended. No speed claim. Runtime active CTA capacity for this new binary is not yet queried; inherited driver saves/checks actual attrs before launch. Resources unchanged gives no observed new occupancy risk.

Bounded onecell screen uses original reviewed driver except distinct JIT module name, and byte-identical binding: current501d control vs new hoist;91 calls(81clean+10warm),4own timed terminals,10mutation checks,16basiscolumns,normal/outlier/zero task+FP64, exact W/pack/arena cleanup. Previous713c results remain separate: robust equal48pair median control−713c−.063999556us. This screen asks whether eliminating rematerialization improves the actual incumbent contrast; it is not a simultaneous713c comparison or official/fullpolicy claim. Existing F1/paired differentXbank limitation remains explicit. On mixed outcome, retain all signs; smallest next robust question uses same-bank fixtures rather than claiming schedule causality.

CPU compiler wall2.304s for exacttwo functions; runtime Torch extension build can cost~42s as prior. Inner90/outer150s includes that cost; no profiler/fullprepare/all95proof. ActualselectedunpoolerM16 scoreelasticity4.856283% is coverage, not forecast. Dynamic gain unknown; larger scored opportunities can take priority. No GPU/API/SSH/queue/git by author.

Loop: profile→measure→hypothesis→optimize→test→validate→stack→submit→learn→reprioritize. [Contract](../../ITERATION.md).
