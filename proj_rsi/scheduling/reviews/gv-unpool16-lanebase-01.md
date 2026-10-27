# GO — projection GV16 lane-base hoist

Independent gv_codec_frontier focused review. Source b1eb9b3cc056b3a431d897cbf5a42dacf0eb521219a19b0e2156a03ddc0e10d3; proposal b2929da59cef200dd149010085f305b776ea849322a1bb74558dd14c76d2b1e7; manifest17adb85e3acf5ea2a4d765368caa830d9003b8976956c8d7357ba4b7d43d04f6. GO for bounded90s inner/150s outer, one muse_bf16_proj unpoolerM16 cell versus actual501d control. No runtime gain claimed.

Independently verified all23 proposal pins and exact source inverse to713c after removing the single opaque lane-base mov, restoring unsigned lane subtraction and reverting JIT identity. PTX mov.b32 preserves all32 bits of uint32(lane)<<6; lane is immutable0..31. Its local register has no shared ownership/lifetime/synchronization effect. No memory clobber is necessary for a register-only move. The existing masked-key/predicate proof and math/codec/layout/source-domain proofs therefore apply unchanged. Pilot differs by only the same mov and subtraction. Binding bytes are identical; driver inverse is exact except JIT module name. Actual native control, transport, schedules, output oracles and cleanup behavior remain inherited.

Reviewed pinned offline codegen: control PC/predicate/encoding exact; candidate1960 instructions/S2R5 versus predecessor1992/S2R20,108registers/no stack or spill. These static facts do not establish dynamic savings. The prior713c three-weight paired loss−0.063999556us and estimator counterevidence remain valid; no family rejection or new gain forecast. No simultaneous713c timing is included; this bounded question compares the new implementation to actual incumbent.

Ran inherited driver.cpu_check read-only: PASS,81timings+10warmups,24balanced matched pairs,4own-terminal sites,10mutations,16boundary columns, stale-seed/missing-arm/invalid-output/touched-state negatives caught; source pins and codec extraction pass. No broader proof or compilation repeated. Runtime resources/legal launch, current-input/own-output numerics, touched-state integrity, post-drain BLAS clear and external cleanup remain trial obligations. No relative occupancy or exact offline-register admission gate is added. No full prepare/profiler/GPU/SSH/API/queue/git operation or frozen-file mutation performed by reviewer.

Root owns admission, actual validity and performance disposition. Delivered files freeze on hash delivery; supplements require new names.

Loop: profile → measure → hypothesis → optimize/test → validate → stack/submit measured gains → learn → reprioritize → repeat; idle≤10s. [Contract](../../ITERATION.md).
