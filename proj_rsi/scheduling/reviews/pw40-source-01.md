# GO: PW40 optional QKV128 W-ring source, CPU review only

Independent source critic sol17;2026-10-27. **GO for this exact frozen source as
an input to the separately reviewed bounded full-policy trial.** This is not
runtime validation, selection robustness, an official submission approval or a
speed claim. Root remains the sole executor/admission authority.

- PW40 source: `3bc0df52d96f62479c3d5c19ab2717c7a9f4d18c1dd087fb786e2f9f080f0278`.
- PW40 manifest: `27ea3303d909c0f9f38b885e4455172b75c0dd27aa71f2647818aa1eb790b062`;
  all46 pinned files match.
- Current root-designated scored PW38 base:
  `29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`,
  manifest `48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b`.
- Validated W-ring research ancestor:
  `207f5a1bdc4f5d7712d444d9c71e857643e6700451c88734fce898ac1fc1b7ab`,
  manifest `20ffb258d67a4ed205298103e0e34e9f06d929957c4cc670dd5985deccc1c299`.
- Independent machine evidence: `proj_rsi/audits/pw40-source-independent.json`,
  SHA `4a8ac3e8053f5f3edb8c4dce93d182c267e5cd56b70127c17f8e4c21bf388154`.

## Optional policy scope and preservation

The only extra production contender is `wring_x6w4hz`, template
`(NT128,ST6,GS1,PRB0,WM0,INS0,FL1027)`, with independent Wslots4 and the existing
QKV KS7. It is appended after the complete old EH/control sequence only for
`layer == qkv_gate` and `m == 128`. In this fixed task N2304/K8192 makes18×7=126
CTAs. N/K are task dimensions rather than new explicit production shape guards;
this review grants no general-shape port. Kernel serving remains restricted by
the unchanged actual selected shape/path dispatch.

The production RQ128 compile list keeps its original8-element prefix/order and
adds one function. The unchanged RQ32 list contributes11 functions. Named RQI
lookups absorb catalog row movement while compiled old selector indices remain
stable. No selectable or compiled `wring_x5w5hz` negative arm is added. The proven
core retains that reference type for its inherited proof/domain, which is
different from enabling it as a production contender.

Only named AST entries RQCU, RQV, RQPROD and `_rqbake` differ. Whole-source delta
replay additionally rules out changes hidden outside those AST entries.
`prepare`, `kernel_fn`, `_time_us`, `_rqstart`, `_rqsbake`, `_rsbake`, RQBAKE,
RQSBAKE, RQPROD2, RQKS, RQFLAGS, router RWCU, MARGIN and
PREPARE_BUDGET_S remain exact. The import-time240s preparation clock, compile
threads, skip guards, full-K basis/random validation, fallback and original0.99
selection margin remain unchanged. Nothing forces W-ring selection or a favorable
timing arm. There is no `_rqconfirm` or PW39 confirmation integration.

The unchanged selection mechanism does not imply unchanged selected policy:
extra compilation, validation and timing work can consume headroom or perturb
other choices. Actual PW40 compilation/selection/full30-shape behavior and
prepare cost remain unmeasured. Retain unselected/skipped paths and all non-target
regressions; availability alone is not an improvement.

## Applied learning and exact composition

Independently bound the exact three applied records and their retained retrieval:

- `wring-mechanism-policy-separation@1`, digest
  `8aed17de00967041aa8d8f3de9d75ba2adf3f9f8cadb3a2315c610cc2067b3ab`.
- `composition-not-additivity@1`, digest
  `0e3ae3fc3b9be194f5afb595c0b3bd065aeafeccfdcd3f4f344d401b59404ee0`.
- `eh-tiny-context-bound@1`, digest
  `6b58697000771856c5c3ef11cbc68049f172bcaa898e63e3e8fadfbb5f8cd8f9`.

The historical retrieval pins registry-v2 SHA
`86617478ad3d82b318d181abe240b5e84547a415497e13ee6912aa5687eaf906`.
Its archived whole snapshot verifies, and the exact three versioned records,
scope/limits/falsification/next tests remain identical after the subsequent v3
append. The current registry version is not misrepresented as the historical
retrieval. The learning proof/query files themselves are in the46-file manifest.

The source follows those lessons: prior W-ring split-versus-PW36 tiny gain is
retained alongside the negative equal-depth contrast, estimator disagreement and
same-main context gap. It is not evidence of a PW40 gain over PW38. EH alternatives
and their contexts stay available. EH and W-ring compete for the QKV128 choice;
they are not combined inside one kernel, and there is no additive score forecast.

Independently replayed the ordered byte edits and their inverse, rather than
only trusting a proof status. All seven kernel layout/publication edits exactly
equal the validated W-ring research edits. A narrow catalog anchor adds only the
split row, and two separate host edits add compilation/bake enumeration.

The W-ring kernel/catalog edits and PW38's six-edit EH delta commute over the
entire PW36-based file with zero normalization and disjoint original spans.
Applying the two host-policy edits reaches exact PW40 bytes. Removing all10PW40
edits restores exact PW38. The alternate path removes only the unused negative
equal-depth row from the frozen W-ring ancestor, applies EH plus host policy and
reaches the same PW40. Its inverse reinstates the explicit row and recovers the
entire W-ring research source. No similar-looking replacement or arbitrary
source normalization is accepted.

The split static assertion excludes HALF/EARLY/EH/EP and limits NT/GS/PRB/WM/INS
and ring-depth pairs. Independent evaluation passes the two inherited allowed
reference pairs and rejects11 out-of-domain cases. The negative equal-depth type
remains absent from the actual new catalogs. PW39 commutativity is an in-memory
compatibility check only, not included runtime code or a performance theorem.

## Machine code, resources and lifetime

Read-only `verify_frozen` passes. Independently regenerated the device-only
source strings in memory and matched both retained candidate/control compiler
inputs. No compiler, CUDA context, extension, prepare or benchmark was invoked.
Two offline `cuobjdump --dump-sass` reads of the retained cubins independently
match the saved SASS files; these never load or execute kernels.

All19 old functions exactly match PW38 controls at every PC, predicate,
instruction, operand and branch target. Every old register count is identical.
The twentieth split stream exactly matches the validated W-ring research split.
All20 candidate and19 control logs report168registers, zero stack/spill stores/
spill loads, and no C75/serialization warnings. This establishes offline codegen
identity, not identity of future loaded modules or timing under other contexts.

Split shared allocation is specifically
`1024 + 6*16384 + 4*12288 + 8*(2*6+2*4) + 16 = 148656B`.
It must not be computed with the common-ring formula. With384threads, the
uniform168×384 and dynamic232×256+40×128 register pools both equal64512≤65536.
Actual opt-in shared capacity, max threads and occupancy remain runtime gates.

Reexecuted inherited EH physical checks for flags2/3/65538/65539 and the split's
physical checks: final WGMMA wait0 precedes output STSM, correct EH half issues,
TMA read drain, no local operations, bounded-poll TRAP, and actual register-pool
allocation/deallocation are retained. The entire original zero/counter protocol
and output suffix remain byte-identical to PW38. BF16 lossless packing/escape
decode, FP32 router precision, WGMMA/D operation order, fresh input descriptors
and production fallback code are not changed by this delta.

Reexecuted the frozen lifetime proof and matched its serialized result:
53047reachable states,1089 randomized complete schedules across KC128/KS1..64,
including KS7 ragged18/19-stage partitions. Packed W covers12288bytes exactly
once; the X SW128 map covers8192 BF16 coordinates bijectively. Shared rings,
full/empty barriers, flags and alignment slack do not overlap. Xfull/Wfull have
separate count1 publication; Wempty requires all8 warp LDS+syncwarp readers;
Xempty follows wait1 retirement or final wait0. Startup, phase wrap, reuse and
drain remain consistent. Missing X publication, early X release and7-of8 W
reader negatives reject. The model aggregates four warps per math WG and assumes
finite asynchronous-engine progress; it is not arbitrary GPU CFG/scheduler proof.

The two original checker failures are retained in history: sibling import
binding and Python tuple-versus-JSON-list comparison. Neither was a timed kernel
failure, and neither is relabelled as having passed before repair. All46 frozen
artifact hashes remain unchanged after this review.

## Mandatory runtime boundary and verdict limits

**Production `rq_go` is deliberately PW38-exact and has no new runtime occupancy/
co-residency guard.** A future trial must prove the126-CTA ZI grid fits before its
first split launch, including launches inside prepare. Checking attributes only
after prepare would be too late. The separate full-scope wrapper review07
provides this boundary: device capability/name/SMcount gate before candidate
import/prepare, then an exact-symbol host-only attribute/opt-in/occupancy query
before `load_inline` returns to any prepare math. Required split values are
168registers,0local,148656B shared, at least384max threads, occupancy≥1 and
126≤SMcount×occupancy. The host-query suffix is separately reviewed and actual
augmented JIT streams must still match all20 canonical functions.

This source GO is contingent on that independent wrapper GO and those successful
before-math runtime gates. It does not approve directly running this production
candidate on an unverified smaller device. Official H200 submission is separate.
The exact full-scope wrapper admission is recorded by07 in
`proj_rsi/audits/pw40-stack-lab-independent.md`; this source critic does not replace
its distinct full counter/graph/terminal/output/lifetime review.

Root's future runtime review must retain actual original PW38 and optional PW40
prepared picks, charge original preparation clocks, inspect swallowed compile/
JIT errors, verify current/fresh input and immutable weights/pack/CSR, all task
and odd-M fallback correctness, full physical counter aliases/dormant baselines,
actual JIT/resources, real graph output oracles before replay/mutation, complete
F1/paired samples and graph/process cleanup. Matched allocation/pointer controls
remain necessary for mechanism attribution. An optional contender staying
unselected, an EH win, budget loss, tiny mixed effect or regression is useful
learning to retain, not a source-review success converted into a speed gain or
grounds for family-wide rejection.
