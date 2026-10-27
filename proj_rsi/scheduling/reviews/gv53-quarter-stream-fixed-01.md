# GO — bounded quarter-stream M8 driver

Independent measurement_contract_audit focused review. ONLY muse_bf16_proj.
Driver `5a55b7d4f96b87d19624eb71e4137f054b8ef154071c863282b0e6adf73adf36`,
binding `39e5c37fb8f883c92aacf0b6e3431e1e9b762ed44a1c344f876dedc8e27ac596`,
manifest `ba674f0027d67ce8e74457d84d330cf1a1f54267a23cfb71f8116ce7d842fa9c`,
proposed `7f5a2296ef74c0c82af0689df101729012c19c20516bd7e6aa7d9b9be3f4c07f`.
All11 launch pins and CPU harness check pass. Reuse source/native-packet
GO4715b328; no compile or broad proof repeated.

Actual extension is exactly frozen pilot plus host binding, with only two native
targets. Host requires CUDA/same-device/contiguous dtypes, exact M8/N4096/K8192,
50331648 packed bytes and131072 escape words. It constructs original G arguments
and launches grid1x128x1,1024threads,17408 shared bytes/current stream. This blocks
unsupported-shape silent device return. Raw loaded resource fields precede legal
residency checks; no occupancy-nonregression or exact-register-count gate added.

Prospective task W seedWEIGHTS_SEED+193 is shared. Original codec AST creates
accepted parent escape bank; None overflow is not bypassed. All4096 actual
warpgroups unpack exactly to stable unit/quarter-bucketed accepted records before
timing. Explicit unsigned-to-signed conversion preserves every int32 bit;
parent/new banks are distinct, hashed and immutable. All transport/packing stays
outside timing. Both arms share unchanged W payload and have private guarded
X/output/full counter/scratch backing. S1 counter/scratch are inactive and retain
their full nonzero sentinels; no irrelevant active-counter protocol is imposed.

81 timed+10 warmup calls, F1 three blocks using the same11 fixtures,24 matched
pairs with12/12 lead order. Same cold readflush and complete native graph work.
Own current-X/task FP64/finite outputs are checked at F1-end and paired-end:
four terminal sites, not every timed sample or earlier control-before output.
These exact retained outputs are checked before another replay/mutation of their
graph. Initial normal/outlier/zero and8 exact weight-column boundaries plus10
fresh/zero/outlier/boundary poisoned graph mutations provide the focused smoke.
All signed samples/F1 contradictions remain; estimator/workload differences
between F1 and paired sets do not establish pure drift or causality.

Graphs synchronize/reset on success and exceptions. After run-scope aliases
release, reuse known ATen BLAS workspace clear and allocator cleanup with raw
before/after evidence and strict allocated0; errors invalidate cleanup. Root
checks external process/container/device cleanup in actual receipt. Driver stores
review hash; reviewed-file pin and argv review path bind that hash externally,
as supported by this reused driver protocol (no unsupported SHA CLI flag added).

GO for one90inner/150outer-second fixed-cell diagnostic, profile=false, existing
exclusive worker and unique output/container. Actual runtime integration/output/
latency remain unrun. Earlier first MMA, unchanged64regs and+114nonNOP are source
evidence only. No full prepare/fullbasis/full30/official-policy claim. Root owns
admission and any subsequent measured integration. No GPU/SSH/API/queue/git or
source/frozen-artifact mutation by reviewer; only new review/binding written.

Loop: profile -> measure -> hypothesis -> optimize/test -> validate -> stack ->
submit measured gains -> learn -> reprioritize -> repeat; idle <=10s.
[Contract](../../ITERATION.md)
