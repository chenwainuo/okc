# EH same-geometry attempt01 — independent result

**Runtime validity PASS, with mixed small matched-variant observations.**
Some pairs improve and others regress or disagree between F1 and paired timing.
No minimum tiny-gain cutoff is applied. This experiment does not measure actual
selected serving policies, a full cumulative stack, or an official H200 score.
It cannot justify forcing EH on every shape or adding these deltas to a board
score. Root alone finalizes queue disposition; no GPU/API/queue action performed
by this reviewer.

Authoritative collection:
`results/proj-rsi/scheduler/artifacts/research.eh.samegeometry.gpu.01/collection-1793076695219438309/`.
All4inventory files and32dispatch source pins verify. Exact admitted driver
fa8420608041ae6884518c95b44fb495b60262e5e498627ff74a2080e0b95c79 /
trial92121f07b0e4e895cdb9e5f1292971dd73776e9bb03b5e535364aefc6e47d9cc;
PW36parent2a067/PW38EH29e557 unchanged. Reproducer and machine evidence:
`audits/eh-same-geometry-01-independent.py` and sibling`.json`.

## Every signed observation retained

Positive means parent control minus EH, in microseconds. Each pair changes only
EHbit65536 at same NT128/ST5/GS1/PRB0/WM0/INS0 andKS: qkv7,pooler4,
unpooler2,o2. FL2→65538 has no hints; FL3→65539 preserves hints. Controls and
candidates are forced research variants, not actual kernel_fn bake winners.

|Shape|Flags|Paired median gain µs|Conservative F1 gain µs|EH paired wins/32|
|---|---|---:|---:|---:|
|qkv128|2→65538|+0.064000|−0.096001|17|
|qkv128|3→65539|−0.159999|+0.352001|9|
|pooler128|2→65538|−0.175999|−0.320001|12|
|pooler128|3→65539|+0.096000|+0.768000|18|
|unpooler128|2→65538|+0.128001|−0.159999|21|
|unpooler128|3→65539|+0.016000|+0.224000|16|
|o128|2→65538|+0.096000|−0.160000|19|
|o128|3→65539|+0.096000|+0.224000|22|

All raw signed pair deltas, ratios, win/tie/loss, ranges andMAD are retained in
JSON; no filtering or cross-job minima. EH3 pooler/unpooler/o have favorable
point medians in both methods, including0.016us; qkvEH3 disagrees, and allEH2
F1 values are negative. These observations do not establish a stable universal
EH benefit or a statistically resolved effect size. F1 andpaired use separate
windows; no claim that a specific instruction or memory phase intrinsically
costs these time differences. Separate policy run PW38stack already reports
actual EH selections; this isolated table must not replace that evidence.

## Source, actual JIT, resources and lifetime

Both parent/EH actual loaded-SO SASS hashes match collected artifacts and their
complete PC/predicate/instruction streams match canonical controls2/3 and
EH65538/65539. Reexecuted both source-specific binary gates. Pure AST factories
regenerate exact frozen research extension source from pinned productionRQCU;
no production import, new compilation, CUDA context or prepare performed.
Recorded SO hashes remain metadata, not archived binary payloads.

Combined actual4function SASS reruns the pinned physical EH checker: WGMMA
wait0 before all8outputSTSM, half0reduce between STSM quartets, both shared
proxyfences/WGbarriers and secondreduce, source read-depbar drain afterallissues.
CPU output bitmap covers all16384 CTA BF16 outputs exactly once, disjoint8KiB
halfslabs per WG; firstasyncread andsecondgenericwrite do not alias. Current
zero-ready acquire/trap andparent fullK/math/producer/reset protocol remain.
This is source+actualcode binding, not a general undocumented SASS interpreter.

All4 runtime attribute records equal168regs/0local/144520Bshared/occupancy1/
max384threads,static384×168 anddynamic128×40+256×232 both64512. Largest grid
128CTAs fits root's132SMH100 at1CTA residency. Attributes/JIT completed before
first candidate math. No resource lowering or occupancy assumption inferred
from timing.

## Numerics, actual outputs, state and schedule

16arm-shape validators pass allK basisEXACT plusFP64 and task normal/outlier.
Exact source-bound original RQTOL[KS] is used:0.008 atKS7,0.006 atKS4,
0.005 atKS2, unchanged from PW36/PW38; task checks independently enforce task
limits/probes. Fixed actual task pack/unpack compares BF16 storage bits before
use. Original task W/packed/escape tensor-equality snapshots and four independent
active2NB reset counters are enforced; no dormant prepared-policy counter is
incorrectly forced zero. No arbitrary-all65536hardware-bit roundtrip is claimed.

Independently reconstructed1096finite positive raw calls,274per shape:
20warm+126F1+128paired. Matched2/3 control/EH/control brackets have21samples each;
32pairedrounds use8cyclic/reverse orders4times, each arm eachposition8times.
Every raw seed/label/mode/warmup flag, F1array/median, pairedtuple/delta/ratio/win
matches result exactly. No separate journal exists in this driver; complete
raw calls are inside authoritative result JSON.

32strict terminal actual outputs pass:4arms×2boundaries×4shapes, each read
against its OWN recorded lastfixture before replay/mutation/profile. F1lastseeds
parent2=240062, eh2=240041, parent3=240125, eh3=240104; current X=240125. Paired last=250031
for allarms. Gate also enforces distinct outputs/currentX/immutable/counters.
80posttiming mutations cover original/zero/freshnormal/freshoutlier/probes for
allarms/shapes, NaN output overwrite/task/currentX/immutable/counter checks.
Every actual graph has exactly1correctrq<...FL...>node. All4shapegraphsets drain/
reset; postprofile immutable/reset checks pass via frozen successful flow.

CPU review cannot rerun numerical Tensor or counter values after GPU exit;
assertions and recorded oracle evidence are bound to the independently reviewed
exact driver/status. No step replays or repairs GPU outputs during this audit.
Synthetic analyzer negatives, with hashes recomputed in separate temporary
inventories, reject wrongpairedgain, wrongterminalownfixture and alteredJITstream.
No authoritative artifact was edited.

Inventory terminalexit0; archived cleanup has no compute/guardPIDs/containers,
named containerabsent,GPU0MiB,cleanup_failedfalse. This verifies retained root
cleanup observation, not a fresh live resource query while another job runs.
Result is valid mechanism evidence with all tiny/mixed outcomes preserved.
No full-policy/official gain or forecast, automatic promotion, source rewrite
or family rejection follows.
