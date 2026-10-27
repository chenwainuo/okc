# GO — projection RQ descriptor-offset fixed two-cell comparison

Independent measurement_contract_audit focused CPU review. ONLY
`muse_bf16_proj`, no serving/MoE scope. Driver
`4fffe34745e4511ae2a7fc03242740a3179689437e374057ff6b8fe36ab5cb30`,
manifest `ceb7238dc9ffa271813e9c10df7d8e376c6630983a4ad3b23ef58b359968f2ca`,
proposed descriptor `afe75a9658045d21a8174428a9a2c4bbda26df0e05b85ae635bc7b7f6e898a4a`.
All14 launch file pins verified. Candidate source is
`c26284f59a5939221f082adf07677653f1f1d7e84099ec1e323cfea8de4c8cc9`,
parent exact inc501d `501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9`.

Reused focused recurrence/lifetime review: only GS1/WM0/PRB0/INS0 targeted types
replace `(s%ST)*XB>>4` descriptor displacement by an unsigned incremental offset.
Actual GS1 round dispatch visits every stage once in increasing order. Increment
XB>>4 and wrap at ST*(XB>>4) therefore give identical descriptor operands; largest
target offset bound is1536 units, safely within uint32. Descriptor base stays
uint64. Original s remains the first-accumulate condition. Loads, escape codec,
waits, releases, WGMMA operands/order, zeroing/counters and epilogue stay unchanged.
Previously independently checked source inverse/soleRQCU delta and stage-event
recurrence are reused; no repeated codec/fullbasis campaign is needed.

In-memory regeneration independently reproduced the exact combined trial.cu from
both frozen source generators plus binding.cu. Only parent/new RQ32/ST6/FL35
functions are instantiated in private namespaces. Source-bound original host
launch/map constructs384threads and99488 dynamic shared bytes; real counters are
2NB reset-zero, not the unrelated monotonic P1 contract. Both use original packed
weights/escapes and task input shapes: pooler M32/N4096/K8192/KS4, unpooler
M32/N8192/K4096/KS2. Copied actual9916 profile records establish scored-selected
geometry. This does not establish fresh prepare selection, integrated policy gain
or an official result. Offline168regs/0spill and−8 static instructions do not
predict latency. Actual runtime attributes are saved and launch legality checked
before math; no occupancy-nonregression gate is added.

Existing device compilation succeeds. Local full-host compile stops on missing
cusparse.h in CPU-only Torch headers before candidate compilation; this documented
environment limitation is not another admission gate. The bounded actual runtime
JIT remains the real integration smoke and preserves any failure.

Independent CPU check passes exact nine original codec AST nodes, scoped two
cells,32 boundary rows per arm/cell, fixed F1 fixture,16 matched pairs with8/8
order. Each cell has21 F1 calls and32 paired calls plus6 warmups:106 timed and12
warmup calls total. F1 control/new/control blocks reuse the same fixture. Private
input/output/counter/sink backings retain physical guards. Normal/outlier/zero
and changed descriptor-ring/split boundary rows precede timing; boundary outputs
equal the exact original weight columns. Current-X/task FP64 and finite checks
execute on every timed own output before next replay. F1 own block terminals and
both paired own last-seed outputs are checked before mutations. Shared raw/packed/
escape bytes, whole active reset-zero counters, counter guards and sink remain
checked. Fresh/zero/boundary poisoned-output graph mutations follow. Required
zero/reduction work remains in the timed native graph. Raw signed samples are
persisted before their output gates, and all graphs drain/reset on normal/error
paths; root still verifies external cleanup in the receipt.

GO for one120inner/180outer-second diagnostic using existing exclusive worker,
unique output/container, no full prepare/full30/profiler. Preserve every signed
comparison and F1/paired disagreement; no gain-size/significance threshold.
Any later actual-selector/fullpolicy integration is a separate measured package.
No GPU/SSH/API/queue/git action or frozen source edit by reviewer.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test -> validate
-> stack -> submit measured gains -> learn -> reprioritize -> repeat; idle <=10s.
[Contract](../../ITERATION.md)
