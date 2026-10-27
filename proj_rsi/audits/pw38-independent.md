# PW38 cumulative source independent review

**GO to run the existing full local harness as a correctness/full-policy screen
of this exact frozen source. HOLD promotion or strict gain admission from the
unmodified legacy harness alone.** A source-pinned full-scope wrapper adding
actual final timed-output, persistent-counter and JIT gates is being authored
separately and needs its own executable-driver review. Source readiness does
not certify that unfinished wrapper or historical EH performance.

No GPU/context/extension load/prepare, production-source edit, commit, queue or
global-ledger edit was performed. Independent read-only CPU check imports only
pure composition/lint/bitmap/SASS routines; no production/torch module or CUDA
API/compilation ran.

## Exact identity and manifest closure

PW38 source `candidates/pw38/submission.py`,277234B, SHA256
`29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e`;
manifest `48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b`.
Every listed file hash independently verified. PW36 parent source
`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf` and
EH backport manifest
`a4ccf23b7ff6b9a383ad0d6299d7de2d0d0d18b60d85e287150a0e465aac9869`
plus every backport file hash verified. Exact isolated EH source is
`9067795eb0d01abe576744a874e3504d6371a92ebc05543708c500b8d83a4819`.

Executed `pw38/cpu_proof.py:check()` read-only: six unique patch anchors apply
exactly and reverse to the ENTIRE PW36 source bytes; duplicate-anchor negatives
reject. PW38 RQCU equals isolated EH's RQCU. Independent occurrence-aware
comparison of ALL176 top-level AST nodes finds exactly four changed occurrences:
RQV146,RQCU149,RQPROD162,RQBAKE163. This avoids relying only on last-assignment
name dictionaries. Initial reviewer expectation ordered RQCU before RQV; actual
AST order differs, so checker corrected to compare the intended four-node set.
No candidate change was made or hidden by that reviewer-only check correction.

All original RQV tuples retain order/index; two EH tuples append. RQPROD/RQBAKE
preserve all original controls and append s4s5zEH/s4s5hzEH atM128/ST5. Small-M
RQSBAKE/RQPROD2, M256 list, full prepare, kernel_fn, fallback, flags and240s
prepare budget/MARGIN0.99 remain exact PW36. Six router named nodes RWCU,
_rwsrc,_rwext,RSPALL,RSP,_rsbake match PW35/PW36. No LDS_CURSOR, EP, native-X
or operand-reverse addition. Existing source/rules lint passes with inherited
sm90a reminder, no new forbidden import or host data behavior.

## EH mechanism, lifetime and control preservation

EH bit65536 is separate from HALF16/EARLY32. Compile-time domain restriction
admits only NT128/ST5/GS1/PRB0/WM0/INS0 and FL65538/65539. Original full epilogue
is preserved under constexpr else, preventing the first backport version's
non-EH register/address refactor from contaminating controls. Existing FL2/3
local SASS streams match their separate controls at every PC/predicate/
instruction:1792/1832 rows. Only affected fixed instances2,3,65538,65539 are
covered by reused offline artifacts; this is not all-production runtime JIT
coverage. Regenerated device translation unit and exact CUDA flags equal the
pinned offline extraction, so no new compile was required for this CPU review.

After final WGMMA wait0 and generic D-register retirement, math CTA bar1 permits
X-ring reuse as output. Each WG's128token×64channel slab is split into disjoint
8KiB halves. Four STMatrix-trans operations write tokens0..63, proxy fence and
128-thread WG barrier publish them, issuer waits current bounded zero-ready
flag then issues first tensor reduce and commits. Next four STMatrix operations
write only tokens64..127, publish via fence/barrier, issue second reduce at
output token64/shared+8192, commit, and read-wait0 drains BOTH source reads before
issuer exits. The half0 async read never overlaps half1 generic writes. Other
threads exiting earlier do not destroy shared memory while issuer remains.
No early scratch reuse, producer publication, K order, codec, accum precision,
zeroing or selected-counter reset protocol changes. Half-output descriptor is
box64channels×64tokens; original output pointer/dims/stride still encoded each
call. Full actual layer output is tiled across channelblocks and both WGs.

Pinned bitmap enumerates all16384 BF16 outputs of the128×128 CTA exactly once,
with disjoint per-WG/per-half source/token/channel ranges. Local physical gate
requires all8STSM after WGMMA drain, first reduce between STSM quartets, both
publication fences/barriers,2reduce issues forEH versus1control, and final
read-depbar after all issues. Independent scoped negatives delete anEH reduce,
proxy fences or read-drain depbar: all3 reject. Underlying fixed-CFG checker
is not a general SASS dataflow interpreter or GPU numerical oracle.

Resource log for all4 offline instances:168registers,0stack/spill loads/stores,
16barriers; static168×384=64512 and inherited dynamic40×128+232×256=64512.
No occupancy/speed inference. Actual JIT/resource/spill/selected-plan/descriptor
checks still required; no historical raw distribution or CPU bitmap establishes
current GPU correctness or latency. PTX async-proxy ordering and bulk-group read
completion are the same parent rules applied separately to disjoint halfslabs.

## Optional selection and allowed claims

Unchanged _rqbake validates allK onehot/randomFP64 and picks fastest validated
RQ only if under0.99×lowerR6 F1 baseline. There is no additional1% EH-over-oldRQ
rule. A small EH improvement can be selected if family already clearsR6margin;
noise/ties/budget/validation can leaveEH inactive. The extra2instantiations and
2M128 validations may alter build/prepare timing and selected plans. Record
PREP/BAKE/PICK and actual graph; compiled presence is not activation or gain.

Historical rs128b -0.13..-0.16us is an omitted local refinement, not current
speed evidence. Recovered oneqkv bake17.25→16.96 is positive; same historical
pooler/unpooler lost and raw repetition distributions were not recovered.
`same_geometry_lab.json` is a proposed fixedFL2vs65538/FL3vs65539 experiment,
not an executable or completed test. Its eventual source-pinned driver must
preserve exact geometry/KS and actual-incumbent controls; no selection-threshold
change justified before measurement. Small positive observations remain visible
without imposing a new minimum-gain rejection rule.

## Existing full harness and next driver gates

Read `tools/h100_bench.py`. Existing --official exercises65 layer/M normal
cases (30scored shapes plus edge sizes), prints30 graph medians, and uses read
L2 flush. It does not fail process on CHECK failure, does not include all exact
nine official outlier cases, discards retained timing outputs, and lacks strict
terminal/persistent/JIT/immutability gates. Therefore GO is for root's full
screen with explicit all65pass/30finite rows/source/picks/prepare/cleanup
inspection; its medians alone cannot establish user-requested verified gains.
Never subtract its replay probe retrospectively or treat GPU local results as
H200 score. No legacy --prepare-only warm step outside exclusive ownership.

For valid full stacked comparison, the new reviewed wrapper must compare actual
fresh RS32/PW36/PW38 kernel_fn for all30shapes, original240s prepares/no forcedEH,
source/JIT/resource identity before first affected math, exact official cases
plus allK/FP64 affected-path checks, currentX/freshzerooutlier/probes/NaN output
and mutable scratch graph reuse, active reset counters and source-classified
monotonic/dormant state, immutable W/pack/escape and ACTUAL terminal timed outputs
before replay/mutation. Fullscope includes P1/RW/RTK persistent contracts that
smallm-v3 did not exercise. Do not repeat blanket allcnt==0 error or reset valid
monotonic state to make checks pass. Root alone owns bounded exclusive GPU run
and actual PID/container/guard/memory cleanup. New wrapper freeze requires
separate GO; no official submission/promotion authorized by this source audit.

## Dashboard exposure review

No dashboard/server/site code was present in the workspace at this review:
rg --files plus dashboard/server-name search found only unrelated observer
notes and no runnable dashboard. No network listener was launched or modified.
Read-only dashboard implementation remains unreviewed until its actual code is
available; this audit does not approve exposure or infer security from absence.
