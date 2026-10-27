# Two alternating producers: CPU/device pass, GPU review pending

Owner=root. No GPU admission, no timing result. Parent is frozen packed-X
0420abd3/source2c28de06, which passed all local numerical/graph/JIT gates and
measured39.648us versus actual24.224us incumbent. Fresh PC samples expose
decoded-B publication/wait work; this is an independent producer-parallelism
branch, preserving the parent's scalar stores while another branch tests STSM.

512threads: two128-thread math WGs and two128-thread producer WGs. Existing
122math registers suggest possible fit under launch_bounds512/1(max128), but
actual rounding128*512 consumes the entire65536-register SM pool. Require
zero spills/local, no dynamic register redistribution, and actual occupancy.

Producer owner0/1 handles stages0/1,2/3,...,62/63. W6/B8 slots are even, so
each producer owns its slots across every reuse. Initial issuer256 fills all6;
publisher/refill issuer256+128*owner. Separate named barriers3/4 count128,
math barriers1/2 remain256, initial CTA barrier covers512. Whole math order,
X pack/A4 schedule, W payload, shared bytes and output remain unchanged.

Both producers use the same immutable escape lists. Before repairing s, skip
records with stage<s; then apply existing rq_fix to stage s. Without skip,
the other parity's record blocks later repairs. This traverses metadata twice,
an explicit cost; it does not duplicate W decode or change original bits.

protocol_proof.py models parity/phase/CSR and a missing-skip negative control.
Independent protocol review also models the encoded32-entry cursor windows and
both math-WG retirement dependencies: Bempty(s) requires retire(s−8), emitted at
round(s−7), still strictly earlier than s. No literal return/join is claimed;
all producer stores/refills precede last B62/B63 publication, so output reuse is safe.

Local sm90a compile: main122/pack22regs,0stack/spill/local/USETMAXREG; rounded
register pool65536 at512threads. Device SASS passes the retained physical A4
retirement checker. Source restoration reproduces exact parent CUDA, including
all Xpack/math/output/decode bytes, after reversing only producer changes, block
size, names and an added metadata-only info6 maxThreads query. This is a CPU
feasibility result, not actual occupancy or GPU correctness.

Lab command after independent exact-source GO, root only:

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_dual_producer/lab.py --json /okc/logs/operand-reverse-dual-producer01.json
```

Bound inner240/outer300. All three actual packed JIT artifacts are saved and physically
checked BEFORE math; actual DP regs<=128/rounded pool<=65536/local0/maxThreads>=512
and opt-in occupancy>=1 are required. Compare actual prepared incumbent, frozen
PX, best-failed STMX and new DP, complete fresh pack+main for all three packed variants. Three separate
workspaces; full all-K/normal/outlier/FP64, storage65536, fresh workspace/output,
20actual retained-graph poison/mutation checks plus all three packed bit oracles,
immutable W/packed/escape bytes, reset counters and graph cleanup are retained.
F1 uses conservative before/after controls;21rotated/reversed paired rounds report
all gains, wins and raw samples. No official slot, port or incumbent claim before
valid complete evidence. A failed version is retained and diagnosed.

Comparator revision: the unrun earlier V1-control lab/manifest are retained in lab-history-bb4d1296/. STMX has since measured faster than PX; the frozen trial now compares incumbent/PX/STMX/DP with three distinct workspaces and15packed poison checks. Kernel97dd source is unchanged by this harness revision.
