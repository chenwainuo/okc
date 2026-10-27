# QKV128 split4 runtime independent review — GO v2

Independent reviewer09, not author16/15. CPU-only GO for root-accepted exact
inner1140s/outer1200s(full1200s reservation), one QKV_gate M128/N2304/K8192
comparison,210unfilteredrawcalls. No GPU/context/extension/prepare/queue action by
reviewer. No automatic selected-policy or official speed claim from a forced
candidate. Root alone admits node state, launches and verifies cleanup.

Exact frozen driver `proj_rsi/tools/group8_qkv_runtime_lab.py`:
`a218e39867c56d1e66cab7d7fb08f245ed7181be59d64db2220e1d3f1b9d4746`.
Exact trial manifest `proj_rsi/candidates/group8_qkv_runtime_trial/manifest.json`:
`efff5355c0609e283dfdba6fa24aff538a6bb3384bd4ccdc4e8eea15d05bb129`.
Manifest files keys are repository-root-relative. All40pins and upstream
kernel9file/wrapper11file closures verify; original4340/fe064v1 retained under
trial/history/v1-43402311. V2 repairs critic-discovered PC-offset omission by
retaining complete orderedPC/predicate/instruction tuples and rejecting PC-only
tamper. It also refuses alternate selected RQ modules rather than associating
their binary with default generatedsource. Original frozen kernel/wrapper and
PW36/RS32 sources remain unchanged.

Candidate source29f7f11bdf3246e66b9419fce4003cb27c22a4cf263a3d1f15c0f3770371f1a2,
Pipeline identity8e2460ae67fe1c346ee16bacf5d03fb049ce199e0cb5ffe0010771ec4bd61996.
Every call packs CURRENT originalX, executes split4main, then orderedFP32reduce.
No input-derived cache is used. Live CSRcapacity derives from actual escapes
with explicit sizeguard and exact BF16 weight unpack check; partial/XP buffers
belong to retained Pipeline objects for full graph lifetime. CPU exhaustive
524288-word XP/output coverage,split4/local32stage CSR encoding/DAG and A4
physical retirement/fence proofs independently rerun. W6/B8/nativeA grouped
math retains pinned parent semantics; new reduction explicitly performs
(((partial0+partial1)+partial2)+partial3).to(BF16).

Before ANY candidate math, copy actualloaded.so and save/hash actualSASS and
resource dump. Match main/pack/reduce fullPC/predicate/instructions against
frozen offline hostSASS; apply main's actual A/D retirement/fence checker.
Runtime requires main122regs/512threads/65536roundedpool,pack22regs/reduce16regs,
zero local and adequate launchthreads. Main optinbudget>=103600B and occupancy
>=1 gate before capture. JIT differences invalidate setup, never skip gating.
Allthree candidate node names/counts must later pass CUPTI profile after both
strict terminal checks. Full pipeline cost includes freshpack and orderedreduce.

Controls are actual original fullprepare plus kernel_fn from PW36source2a067
andRS32sourcea7b6, with actual selected plans retained. No prepareclockreset,
forcedRQselection or control-outputcopy is added to timedgraphs. When RQ is
selected, its exact object must be defaultmodule_rqmod0,source generated from
frozen RQPROD must match original storedsource,selectedattributes require168regs/
0local; launchdecline fails instead of allowing fallback. For nonRQselection,
actual original policy runs and actualGPUprofile is recorded. Production
prepares naturally bake their own kernels before external comparator JITgate;
this is an explicit scope limit, not a before-preparegate claim. Unknown active
persistent/architecture-specific counter protocols HOLD conservatively.

Pinned v3 counter source/dispatch bindings independently rerun. Actual selected
counter PHYSICAL STORAGE resets0; dormantstorage equals preparebaseline even
when legitimately nonzero, including offset aliases. No harnesszeroing. Full
failurevalues and successfuldigests retained. All originalweights,prepared
packed/metadata/candidateW/escape immutable; workspace/sink excluded according
to pinned source contract. Both originalprepares must preserve taskweights.

All8192basis columns are tested exactly at x8 in64batches of128rows for each
policy. Normal/outliertask plus explicitFP64 relativeL2<=4e-3/maxrow<=2e-2 pass.
Actualfixedgraphs test original/zero/freshnormal/freshoutlier/allrowprobe with
NaNoutputpoison; candidateXP/partials also poisoned and completely revalidated.
Actualpack probes all65536BF16storage patterns with integerbit inputimmutability;
nonfinite arithmetic is explicitly storage-only, followed by fresh valid
fixtures overwriting/revalidating allworkspaces. Independent newPipeline tests
freshWpack/CSR/workspace/output ownership. XP is byte-exact versus shape-only
indexmap; everypartial finite and output equals orderedreduction. No hardware
raw-bit proof is inferred from CPU model alone.

Schedule15warm+105F1+90paired=210calls, eachseed/policy/mode/time logged immediately
to raw.jsonl and summary. F1 five21sampleblocks PW36/RS32/new/RS32/PW36;
30pairedrounds run allsixpermutations exactlyfivetimes with samefreshfixture
perround and256MiB read-onlyL2flush. No samplefiltering. F1lastfixturebindings
PW36=720104,RS32=720083,new=720062 are distinct. At index120, before nextXcopy or
replay, allactual retainedoutputs plus currentX/candidateXP/partials/counters/
immutables validate against OWNlastfixture. Pairedend repeats before anyprofile/
mutation/repair. Both controls' conservativeF1gains,pairedmediangains/ratios and
allrawrows retained. Root harvest should independently recompute all210calls.

Reviewer extracted actual nested terminal_gate/oracle/workspace_oracle and
rejected12CPUcorruptions: output,NaN,XP,partialNaN,orderedpartialsum,currentX,W,
packed,escape,counter,wronglastfixture,missingpolicy. No terminal replay/mutation
methods; sourceordering confirms F1gate before nextinputcopy and pairedgate
before profiling. Actual outputs have distinct retainedpointers. Profile outputs
are immediately revalidated; posttimingmutations repeat. Finally allcapturedgraphs
drain/reset; runtimeerrors retain traceback and require rootcleanupverification.

Read-only reviewer checks: author --cpu-check(reproducedPASS),fourCPUunittests,
`python3 proj_rsi/audits/qkv-runtime-review/negative_checks.py`, and detailed
review-cpu.json/author-cpu-check.json evidence in that auditdirectory. These
checks import no CUDA extension or productionprepare and execute no GPU work.

Root invocation uses new NONEXISTING output CHILD, for example:
`python3 /okc/proj_rsi/tools/group8_qkv_runtime_lab.py --output /okc/logs/scheduler/qkv-runtime-01/run --root-review-token proj_rsi/scheduling/reviews/qkv-runtime-01.md`.
Scheduler may create artifactPARENT, never child`run`(driver exist_ok=False).
Collect parent directory so failure beforechildcreation remains in rootlogs.
Bound1140/1200 is exact to thisGO. Preserve invalidattempts; no source/cap/flags
mutations or official submission under this review. Root must verify container,
computePID andmemoryrelease after exit; exit0 alone is not admission.
