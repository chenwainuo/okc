# Actual native-X versus DP source-counter analysis

The collected three-node report and its serialized CPU export pass independent
source/counter accounting. Analysis is now complete for this capture; there is
**no new speed or causal-latency claim**. The earlier capture-only report remains
immutable and accurately records its then-pending export state.

DPmain has3351samples=1724math+1524producer+103common; NXmain has3863samples=
2002math+1749producer+112common. Pack has314samples. Every source PC binds to
its actual saved SASS PC/predicate/instruction:56pack,848DPmain,1104NXmain rows.
No PC/sample is unmapped. All source totals equal raw smsp__pcsamp_sample_count.
Both main math/producer denominators exceed the preregistered200sample gate.
The sampling interval is2048cycles,2aggregatedPCpasses and0droppedbytes for
allthree nodes. NCU's34collection replay passes per node are not34independent
performance observations.

| Source location | DP samples / role denominator | NX samples / role denominator |
|---|---:|---:|
| math WARPGROUP.ARRIVE PCs |808/1724=46.87%|14/2002=0.70%|
| math Bfull test/predicate/slow paths |102/1724=5.92%|83/2002=4.15%|
| producer Bempty test/predicate/slow paths |475/1524=31.17%|764/1749=43.68%|
| producer Wfull test/predicate/slow paths |81/1524=5.31%|128/1749=7.32%|
| math HGMMA PCs |295/1724=17.11%|96/2002=4.80%|
| math direct-X LDG128 PCs |38/1724=2.20%|0sites|

Native-only X delivery has additional visible locations:

| NX location | Samples /2002math | Interpretation |
|---|---:|---|
| startup Xfull exact paths |100=5.00%|Startup is retained separately|
| loop Xfull exact test/predicate/slow paths |260=12.99%|Includes blocking-test dependency association|
| arithmetic interleaved between loop Xfull test and predicate branch |509=25.42%|Address/control work in a qualified wait window|
| combined loop Xfull qualified window |769=38.41%|260+509; not isolated barrier latency|
| bar5 synchronization |262=13.09%|256-reader cross-WG read-retirement/convergence|
| LDSM issue PCs |39=1.95%|Issue-site association, not full load readiness cost|
| shared proxy fences |96=4.80%|Required generic/async publication ordering|
| elected math TMA issue PCs |0|Observed zero samples is not zero cost|

The interleaved window distinction matters. At NX0x1c30 the actual instruction
is Xfull TRYWAIT;0x1c40 is IMAD.IADD and0x1ce0 is its failed-predicate branch.
0x1c40 alone has408samples, all long-scoreboard. Calling those408samples pure
integer arithmetic latency would be unjustified. The register/control dependency
and a preceding blocking TRYWAIT can move the observed PC. The script retains
exact wait sites and interleaved instructions separately before reporting their
qualified-window sum. It similarly keeps104interleaved DP Bfull-window samples
separate from102exact-path samples; the combined DP Bfull window is206/1724.
All residual math, producer and common PCs remain explicitly accounted once.

The dominant exact DP ARRIVE association is807long-scoreboard samples of808.
Native loop Xfull exact paths have250long-scoreboard of260; interleaved Xfull
arithmetic430of509; bar5 has254of262. Native producer Bempty has761long-scoreboard
of764. These are waiting/consumer-PC associations, not intrinsic instruction
latencies or a proof of where elapsed time can be removed.

All mapped fast/outlined wait lane flows reconcile: slow-test executed lanes
match failed-entry plus failed-retry lanes. Native Xfull failed-entry lane
fractions are zero at its eight fast sites, yet its test/window samples are
substantial. Producer Bempty failed-entry fractions are only0.07324%DP and
0.09068%NX, despite475/764samples. TRYWAIT may block before succeeding, so low
failed-predicate fractions do not prove data initially ready. InstructionsExecuted
ignores instruction predicates; predicated-on thread counts are retained
separately. No divide-by32 uniform-warp claim is made.

The source reason columns omit warpgroup_arrive. DP all-state reason residual207
matches the raw207category total; its not-issued residual156matches156. NX
residual52matches52; not-issued27matches27. This reconciles missing displayed
labels without inventing per-PC distribution. Source instruction totals
8,482,087DP and9,199,983NX remain distinct from raw8,176,336/9,056,364: replayed
polling and metric accounting do not justify forcing them equal.

| Metric | DPmain | NXmain |
|---|---:|---:|
| global LSU load requests |263168|1024|
| global LSU load sectors |4198400|4096|
| DRAM read bytes |51466240|51469824|
| DRAM write bytes |2359040|2365440|
| L2 read hit rate |30.915676%|30.862541%|
| eligible warps / active cycle |0.424515|0.379931|
| issue active / peak active |30.691378%|30.216213%|
| tensor-pipe active / peak elapsed |27.846970%|25.020616%|

The export uses scaled units:51.466240Mbyte is51466240bytes, not51bytes.
The parser preserves original value/unit and an explicit base-unit conversion.
NX removes math's packed-X global LSU loads; its1024requests are initial escape
metadata, so its50%L1load hit rate has a different population from DP's0.04878%
and is not an X-cache victory. Original-X TMA traffic uses a different request
path. DP/NX DRAM reads differ by only3584bytes (~0.007%): this does not support
an HBM-byte-elimination explanation. Aggregate traffic cannot isolate W/X/
metadata nor prove a bandwidth ceiling. Pack reads1051904DRAMbytes and has
8192load requests/32768sectors. Its observed0DRAMwritebytes does not mean zero
stores; writes may remain cached. Its four store PCs have245/314samples, mostly
long-scoreboard, which also does not establish intrinsic store latency.

Raw launch identities match the source contract: pack grid256/block256/22regs;
both main grids128/block512/122regs. Main dynamic shared bytes103600DP and
137232NX are distinct from total104624/138256 including1024driver bytes and
allocated104704/138368. Allthree raw rows belong to the selected profile NVTX
range in order pack→DPmain→NXmain. Six CTC metrics are unavailable, never filled
with zeros. A deliberately missing generic L2-hit metric key also stays null;
the actual exported read-hit metric is separately named and used.

The original capture's immediate actual-output oracles, DP's full1MiB current-X
permutation, weight/packed/escape/X immutability, counters0, poison overwrite,
physical A4/resource/canonical gates and both drained/reset graphs all pass.
Capture and export guard inventories independently prove no live guard/container/
compute context,0MiB and successful cleanup. The export exactly reimports the
hash-pinned original report, records Nsight2025.1.1 and export-binary SHA; original
capture-binary SHA remains unknown. No new GPU/NCU/remote work was performed by
this analysis author; only local collected files were read.

Two cautious next implementation questions are proposed for the learning
registry, with benefit and full costs unknown (null):

1. CPU-proof independent64-row X half ownership per math WG. Current row mapping
   makes WG0/1 read disjoint8KiB halves of each16KiB slot. Investigate separate
   half descriptors/completion/reuse domains and128-reader retirement, replacing
   the cross-WG256-reader bar5. Count doubled TMA requests and metadata, and prove
   every slot/phase/proxy/A-retirement lifetime before any runtime. This differs
   from merely deepening X4 or repeating the prior producer-owned async-refill
   port. No gain is predicted from the sample fractions.
2. CPU-inspect address preparation around the hot Xfull window. Precomputing
   native addresses before TRYWAIT while preserving waits/loads/refills may alter
   scheduling/register pressure. A hot-PC relocation without consistent complete
   paired/F1 improvement refutes an arithmetic-speed explanation. Reject spills,
   resource-pool overflow or lifetime changes before requesting a GPU test.

Both ideas must compare the frozen native parent, relevant current composite
and actual freshly selected incumbent complete pipeline. Producer Bempty exposure
is consistent with consumer-side backpressure; it does not prove producer decode
can never matter. Do not respond by blindly adding producers or sweeping deeper
W rings. The separate unprofiled native-X incumbent loss remains negative evidence;
this profile neither supersedes it nor measures the latest composite or H200.

Reproduce with counter_analysis.py; six CPU tests reject PC-only, predicate-only
and instruction-only source tamper and verify null/byte-unit handling. Structured
results are counter-analysis.json. The separate proposed-learning-registry-v3.json
passes the current learning.py schema and immutable digest/selector checks; it is
a proposal only, not an edit to the frozen current registry or execution queue.

Recommended scheduler disposition: capture VALID_NO_GAIN with explicit
“valid diagnostic, no speed comparison”; export CPU_PASS (serialized CPU-only
read/import). Root alone validates/updates those records. Counter information is
retained as a successful scoped diagnosis, not an achieved optimization gain.
