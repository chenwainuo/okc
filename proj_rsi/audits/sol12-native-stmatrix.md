# Sol12 native-X plus STMatrix CPU audit

Candidate `candidates/reverse_tma_x_stmatrix/` is frozen at manifest SHA256
`2c132b82a854e476f433335462b2122f21bc3cb490dec6f2b691c01cd9775045`;
generated extension CUDA SHA256
`f6a765f0d0cf3174435f4981958015c242a53b5359f00064aa388b958cb9d9b6`.
State: CPU_FEASIBILITY_PASS_REVIEW_REQUIRED_GPU_UNRUN. No extension load, CUDA
context, prepare, measurement or GPU command was run by this worker.

Frozen native parent manifest
`025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc`
and source `ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377`;
DS store control manifest
`6785da2ec9881ce13148951daff2f8802d8e7bbe6c83abf588607092b3f97072`
and source `672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1`.
Only producer store block and unique kernel name differ from native parent.
`proofs.all_proofs()` reverses those changes and requires equality of the ENTIRE
generated extension CUDA source. The inherited native source/host wrapper is
therefore bound, including current original-X tensor map, two SW128 shared slots,
math barrier5/elected refill, A4, W6/B8, CSR, publisher, output and proxy/bar3/4.

The DS proof model is retained locally and executes against the new prototype's
identical pack/decode helpers. Every8192B stage is exactly equal between scalar
and STMatrix models for both64-row halves and all64 K stages, including all65536
BF16 storage patterns and CSR escapes across32-entry windows. Every one of the
32 address providers in each of four producer warps supplies a16B aligned row
address in its8192B slot. Data lane L/register i uses provider `8*i+(L>>2)` plus
`4*(L&3)`, exactly reproducing the original SW128 destination. Slot base36864 and
slot stride8192 preserve16B alignment. Both complementary owner groups execute
full warps and the copied STSM block is before the unchanged proxy/barrier code.

CUDA12.8 exact flags: `-gencode=arch=compute_90a,code=sm_90a -O3 -DNDEBUG
--expt-relaxed-constexpr`, plus `-Xptxas=-v -cubin` for CPU artifact. Resource log
reports122 registers and zero bytes stack/spill loads/spill stores, six barriers;
no C75/serialization warning. Rounded register allocation128*512=65536.
Dynamic shared allocation137232 and all native shared-region bounds are unchanged.

Local SASS gates pass: four unpredicated `STSM.16.M88.4`, remaining16 scalar STS
are unchanged output stores, 16 HGMMA, zero128B global X loads, 32 LDSM, eight
math barrier5 and eight math TMA issue sites. Four A register arrays remain
physically distinct; conservative two-iteration retirement gate passes. Last
producer STSM PC0x3ab0 precedes MEMBAR/proxy PC0x3ad0, then owner barriers4/3 at
0x3ae0/0x3af0 under complementary P0 predicates with no intervening predicate
writer. Named owner participants remain256..383 and384..511.

Initial preflight compiled the same kernel successfully, then failed because a
copied native physical checker expected dispatch PC0xaf0/target0x2cf0; this
composition shifts it to0xad0/0x2cd0. Initial checker and failure JSON are retained
in checker-history. Only checker constants changed. No failed CUDA variant was
discarded; there was no CUDA-source revision.

`actual_binary_gate.py` is a CPU-only gate component for a separate root GPU lab:
it compares the actual .so main to the frozen cubin at every PC/predicate/
instruction/branch target, then reruns A4/native-X/store/owner/proxy gates. Its
local canonical selfcheck passes. All manifest file hashes were verified after
freeze. A standalone execution driver outside this immutable candidate directory
has NOT been prepared; one is required before GPU execution. The frozen native
parent lab cannot be used unmodified because it lacks the new arm and pins old
identities.

Iteration record:

```yaml
family: reverse_tma_x_stmatrix
version: f6a765f0d0cf3174435f4981958015c242a53b5359f00064aa388b958cb9d9b6
predecessor: native reverse_tma_x ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377
observed_failure: inherited native 31.856us paired vs DP34.096/DS35.312; local incumbent23.968 (supplied task evidence, not newly measured)
causal_hypothesis: decoded-W collective stores may reduce producer instruction pressure independently of native-X loading
change_or_diagnostic: replace only16 scalar producer stores with4 proven STMatrix operations
predicted_observations: CPU instruction reduction observed; runtime benefit or nonadditive interference remains unknown
comparison: [actual_selected_incumbent, frozen_native_parent, new_native_stmatrix]
validity: CPU source, bitmap, protocol, lifetime, resources PASS; GPU numerics/graphs/mutation/immutability/cleanup pending
forecast:
  attainable_gain: unknown
  conditional_score_value: no official claim
  information_value: tests whether DS collective-store benefit composes with native-X TMA/LDSM
cost: CPU compilation and proof only; root exclusive queue required for future GPU lab
next_after_failure: retain full measurements and diagnose producer/math overlap or store cost; do not retire family from a losing version
```

Independent critic/source review and a new frozen standalone driver are required
before root executes GPU work. Runtime attributes/actual occupancy must pass;
actual-JIT checks must precede first math; all native numerical, graph, mutation,
immutability, zero-counter and cleanup gates remain required. Hardware raw-bit
TMA/LDSM roundtrip remains unprobed, inherited from parent. No global ledger,
benchmark-ready promotion, submission, official claim or commit was made.

## Separate lab follow-up

A new standalone driver and trial package are now frozen without changing any
kernel artifact: `tools/native_x_stmatrix_lab.py` SHA256
`87ec3afa2add4ce14e53f924bdc3d8310960aeee22e7947df6e9d4ec89b94f22`;
`candidates/reverse_tma_x_stmatrix_trial/manifest.json` SHA256
`7677ff2bfbac57b50a739dea72f3ea3ed1a28784d4d82fd7cf02de1c4b0a71e3`.
The earlier note that a driver had not been prepared describes the kernel freeze
stage; this separate package now satisfies that prerequisite, pending07 review.

Four labels are actual LOCAL pw36 incumbent, DS globalXP, native025f parent and
new2c132 STMatrix. Same frozen async driver48feda58 and trialc2618e31 serve as
source template; both originals remain unchanged. New native/new resource gates
require137232 bytes shared,122registers,512threads,65536pool and exactly1 active
block/SM after opt-in. New actual-JIT gate calls frozen candidate's
actual_binary_gate.py with correct per-label liveness/proofs bindings.

CPU proofs verify regenerated source equality and all canonical binary gates;
six wrong-label pairings plus three tampered HGMMA operands reject. Exact
extracted measure control flow runs with fake CPU events, confirming147F1,
128paired and20warm calls, eight terminal outputs, distinct F1 ownfixtures,
and each paired arm at each position8times. Nine terminal corruption negatives
and no-replay/no-write gate ordering pass. Full4096basis/FP64/task oracles,
20actualgraph mutations,DS-only XPbits,freshworkspaces,immutability,counters,
reset/drain and1-vs2node profiles remain present for root-only execution.

Trial README contains exact normal240/300 root command. Agent07 independent GO
is still required. Final manifest file closure, frozen kernel/old driver/old
trial identities and syntax verified CPU-only. No context,extension,prepare or
GPU call executed; no commits or global changes.

## Stacked X4 plus STMatrix composite

New candidate `candidates/reverse_tma_x4_stmatrix/` frozen manifest
`11892b99e5e7b87c6e7413202701f18f1a2fe0407596d8e7dea184881981b22c`;
source `61aad6a1d26ee56492c2b0a87b5b1e6c12c434fc166a0aad6e958a555becbf15`.
This stacks frozen X4d47ed166/e8447cbf and frozen STMatrix2c132b82/f6a765f0 on
common native025f/ffd30. Both former kernel artifacts and standalone labs stay
unchanged; worker09 owns the separate five-arm factorial job.

`composition_proof.py` verifies every frozen parent manifest/file, applies eight
finite depth edit spans (seven edit strings; both initialization and startup
loops change bounds), and the exact producer store block. Normalizing ONLY
kernel identities, full17694B extension CUDA is byte-identical for
`store(depth(native)) == depth(store(native)) == actual`. Its normalized SHA is
`e6241e4027352de8fc932c568679c6e7ed1d841e9b38a0f5c2f0054eb2d34e43`.
Exact finite inverse recovers entire X4, entire STMatrix, then native parent from
both orders; every depth edit span is disjoint from the producer store span.
Thus codec, FP32 math, A4, host/launch, output, W6/B8, CSR, publisher, proxy,
bar3/4/bar5 remain fixed except the explicit depth/store edits. Unique final
kernel name rq_tma_x4_stmx appears exactly twice, declaration and host selection;
extension cache identity uses source SHA.

Local CUDA12.8 actual compile matches prediction:122registers/512threads,
170016 shared bytes/65536 rounded pool,zero stack/spill loads/spill stores,
no C75. Source schedule and region proofs bind X4; A4 retirement gate passes.
SASS gates pass4unpredicated STSM,16output scalarSTS,32LDSM,eightbar5,
eightmathTMA,fourstartupTMA,zero mathLDG and16HGMMA. Last producer STSM0x3cb0,
proxy0x3cd0,complementary P0 owner barriers4at0x3ce0/3at0x3cf0 with no predicate
writer between. DS all8192B/all65536bit store-layout proof,16Bprovider alignment,
producer parity/CSR/barrier protocol and output bijection also pass.

Initial checker failures were checker-only: copied X4 fixed dispatch0xc50/
0x2ef0 moved to0xc30/0x2ed0; finite transform initially expected one startup-loop
occurrence, while both barrier-init and TMA-startup loops change. Initial
preflight/proof scripts are retained under checker-history. Composite CUDA
source never changed after first compile and had no codegen resource failure.

`actual_binary_gate.py` is available to worker09's separately frozen lab, with
same exact canonical stream/lifetime/native/store/owner/proxy interface. All
manifest hashes and canonical selfcheck verified. Worker09/07/root08 registry
notified. No additive speed forecast or fixed0.5us retention gate is claimed;
tiny valid effects may stack and require the factorial same-job evidence.
Hardware raw native bit movers remain unprobed. No GPU,context,extensionload,
prepare,globalqueue edit or commit occurred.
