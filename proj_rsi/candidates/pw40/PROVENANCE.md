# PW40: optional narrow W-ring contender over PW38

PW40 branches from frozen PW38 source29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e,
the current scored base designated by root. PW38 and its official job remain
unchanged. PW40 is implemented and CPU checked; it has no GPU, actual-selection,
whole-policy, official-correctness or score result.

The only production addition is the OPTIONAL `wring_x6w4hz` QKV M128 contender,
with NT128/ST6/WS4/GS1/PRB0/WM0/INS0/FL1027 and existing QKV KS7. It is appended
after every existing EH/control option for this one bake cell. All original
options, order, validation, R6 incumbent margin0.99, compile-thread strategy and
import-clock240second prepare budget/guards remain. No fixed variant is forced.
Equal-depth X5/W5 separation is intentionally absent from selectable/production
catalogs because its actual mechanism contrast was negative. The proven core
still supports that type for reference, not for production selection.

PW39 bounded host confirmation is NOT included. Its separate typed delta
commutes in memory with this PW40 delta; that establishes source compatibility
only. A future combined candidate requires validation of both changes and a new
reviewed source. No source composition is treated as timing additivity.

## Learning retrieved before implementation

Read `proj_rsi/LEARNING.md`, then ran the read-only query retained verbatim in
`learning-query.json`:

```
python3 -m proj_rsi.learning query 'wring separate composition overlap' --hardware H100
```

Applied immutable lesson records, validated by `learning_proof.py`:

- **wring-mechanism-policy-separation@1**, digest8aed17de00967041aa8d8f3de9d75ba2adf3f9f8cadb3a2315c610cc2067b3ab.
  Retain split versus actual selected PW36 +0.224000us F1/+0.175999us paired,
 23wins/1tie/8losses, alongside equal-depth separation −0.224000F1/−0.240001paired
 and split-versus-coupled estimator disagreement. The same-main actual/coupled
 F1 gap is+0.255998us, so allocation/order/context remains a causal limitation.
 PW40's optional selection must be tested in its actual PW38 context; the PW36
 point gain is not a promised PW40 gain.
- **composition-not-additivity@1**, digest0e3ae3fc3b9be194f5afb595c0b3bd065aeafeccfdcd3f4f344d401b59404ee0.
  Exact commuting source edits can have mixed marginal timings and still lose
 the actual incumbent. Prove composition separately; do not sum historical
 medians, source-regions or independent minima into an expected score.
- **eh-tiny-context-bound@1**, digest6b58697000771856c5c3ef11cbc68049f172bcaa898e63e3e8fadfbb5f8cd8f9.
  Preserve useful tiny EH observations and negative/unselected alternatives.
 PW40 keeps every EH/control candidate and does not enable W-ring universally.
 EH and W-ring now COMPETE for QKV128 selection; they are not combined inside
 the same kernel. W-ring plus EH flag combinations remain compile-time invalid.

`learning-proof.json` binds exact lesson IDs/versions/digests, tested scopes,
transfer limits and falsification. Registry retrieval is a CPU review aid, not
queue registration, performance admission, H100→H200 transfer or promotion.

## Exact typed composition and ancestor recovery

The common ancestor is frozen PW36 source2a067059. Seven exact kernel edits
come unchanged from validated `pw36_wring_rebase` manifest20ffb258/source207f5a1b:
W depth/configuration, physical ring bases, separate full-barrier initialization,
W producer slot/phase, W read/release ring, X wait before math and split-domain
assertion. No codec, arithmetic, D pin/retirement, bounded zero/trap/counter reset
or output operation changes. HALF/EARLY/EH/EP remain excluded from split types.

A stable old variant-row anchor inserts ONLY x6w4 before EH catalog rows.
This kernel+catalog delta and PW38's exact frozen six-edit EH delta commute on
ENTIRE source bytes with ZERO normalization. Their base edit spans are proven
disjoint. Two later typed host-policy edits append one compiled contender and
restrict its bake enumeration to layerqkv_gate/M128; existing RQKS supplies7.
Thus kernel/layout, catalog, compilation and bake policy remain separate regions.

Undoing PW40's ten edits restores the entire PW38 file exactly. The alternate
path removes only the unused negative equal-depth catalog row from frozen
Wring207f, applies frozen EH and the two host-policy edits, and reaches the SAME
entire PW40 bytes. Reversing that path and restoring the explicit row recovers
the entire frozen Wring207f ancestor, not merely a similar kernel body. No
arbitrary reorder, textual normalization or unpinned best-effort merge is used.

Existing named AST changes versus PW38 are RQCU,RQV,RQPROD,_rqbake only.
prepare,kernel_fn,_time_us,_rqstart,_rqsbake,RQBAKE,RQSBAKE,RQPROD2,RQKS,RQFLAGS,
MARGIN,PREPARE_BUDGET_S and RTK router bodies remain exact. RQPROD retains its
original8-entry prefix and adds one option; RQ32 remains11entries. Existing EH
catalog indices move through named RQI lookup, while compiled selector indices
remain unchanged because original production prefixes/order survive.

## CPU compiler, controls and lifetime evidence

Local CUDA12.8 sm90a/O3/DNDEBUG/expt-relaxed-constexpr compile covers ALL20active
RQ instantiations:9RQ128 including the new split,11RQ32. A separate PW38 CPU
control compilation covers all19old production functions. All19existing EH AND
non-Wring functions match at every PC,predicate,instruction,operand and branch
target and retain identical register resources. The new split main is exactly
the frozen validated Wring split instruction stream. No performance inference
comes from equality of instructions under different allocations/contexts.

All20candidate and19control instances use168registers, zero stack/spill loads/
spill stores and no C75 warnings. The new split uses384threads/148656 shared
bytes; uniform168*384 and dynamic232*256+40*128 pools both64512≤65536. Actual
opt-in occupancy and co-resident126CTA zeroing grid remain future runtime gates.
Inherited EH output bitmap/physical checks and validated split physical checks
pass, including finalwait0 before output STSM, TMA-read drain, proxy/barrier
publication, bounded poll TRAP and actual USETMAXREG232/40. No GPU context,
extension load, prepare or timing was run by this author.

The frozen Wring protocol proof is reexecuted:53047 reachable states plus1089
random schedules over KC128/KS1..64, including KS7 ragged18/19 stages. It binds
all shared-region addresses/alignment/phases, separate startup/read/reuse/drain,
packedW12288B and XSW1288192-value bijections. Missing X publication, premature X
release and7-of8 W reads reject. The model aggregates four warps per mathWG and
assumes finite async-engine progress; it is not a general GPU CFG/scheduling
proof. Entire zero/counter and output suffixes remain PW38 byte-identical.

Initial `python -m` harness failed to import its sibling build_patch helper;
that precompile failure and original checker are retained in history. Only the
checker import binding changed. No CUDA source/codegen failure was discarded.

Read-only verification after freeze:

```
python3 -m proj_rsi.candidates.pw40.verify_frozen
```

The author build script/cpu_proof CLI regenerates artifacts and is not the
read-only verifier. Reviewers should use the verifier/pure gates to preserve
frozen resource logs/cubins/source identities.

## Required next evidence and falsification

A separately frozen reviewed complete-policy PW38/PW40 lab must run fresh
actual prepare under original240second clocks/compile strategy, retain every
BAKE/PICK record and record whether x6w4 is ACTUALLY selected. Additional compile
and8192basis validation cost can consume preparation headroom or change other
selections; none is measured here. No timing admission from merely making an
option available, compiler success or exit0.

Require exact actual JIT/source/resources, original no-fallback selected-policy
calls, full30shape correctness, current/fresh inputs, NaN/workspace overwrites,
source-bound active/dormant full-storage counter contracts, immutable weights/
pack/CSR, own-fixture terminal outputs BEFORE replay/mutation/profile, complete
balanced paired/F1 raw samples and named graph/process cleanup. Retain all tiny
signed results and estimator disagreements; impose no invented minimum gain.

For mechanism attribution also retain a matched-pointer/allocation-role fixed
control versus split contrast plus the actual selected PW38 policy. The prior
same-main context gap means independent prepared allocations cannot establish
that full-barrier separation caused the gain. Refutation includes disappearance
or reversal of split's policy gain, preserved context gaps large enough to
explain it, x6w4 remaining unselected, EH winning the same path, or other-shape
regressions/prepare-budget loss erasing a narrow gain. Keep those observations
and iterate a new version rather than retire a whole family.

This candidate supplies optional cumulative source integration for a fresh test.
It does not claim an added gain over PW38, official correctness or score,
additivity with EH/PW39, or transfer from local H100 to H200.

## Read-only verifier checkpoint

A provisional22-file freeze reached successful source/compile/resource/physical
proofs, but its new read-only verifier compared Python tuple spans directly
against JSON list spans and rejected equivalent serialized protocol facts.
The entire provisional package and manifest are retained under
`history/initial-verifier-json-shape/`. Checker revision2 compares their JSON
representations; no candidate CUDA/source, cubin, resource or timing result
changed. This was a verifier container-type mismatch, not a protocol failure.
