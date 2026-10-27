# PW36 separate W-ring rebase, v1

CPU source/resource/protocol/lab proofs pass. GPU UNRUN. This is a narrow research
rebase of historical rs128b's FLbits8..15 separate-W-slot mechanism onto frozen
current PW36; it makes no production/bake selection change. Historical QKV128
x6w4 observation15.23→14.85us (~0.38us, near noise) is prior evidence only.
Neither deeper-W regressions nor that small observation closes the family.

Three explicit complete-kernel arms, all M128/GS1/PRB0/WM0/INS0/HINT+ZI/KS7:

| Arm | X depth | W depth | Full barrier | Dynamic shared bytes |
| --- | ---: | ---: | --- | ---: |
| coupled | 5 | 5 | Shared count2 | 144520 |
| equal | 5 | 5 | Independent count1/count1 | 144560 |
| split | 6 | 4 | Independent count1/count1 | 148656 |

A fourth arm is actual LOCAL PW36 serving selection after complete prepare,
without a forced variant. Its actual selected rq return must be0; abort fallback.
Equal versus coupled isolates barrier separation at equal ring depth (plus40B
metadata). Split versus equal additionally changes both depths and shared bytes;
those depth/resource confounds are explicit. No broad depth sweep. No minimum
0.5us gain cutoff, additivity forecast, official score claim or promotion.

`build_patch.py` and delta.json apply eight strict unique edits to entire PW36
submission; reversing every edit restores the entire parent byte for byte.
All codec helpers, mathematical order, FP32 D pinning/retirement, current bounded
zero poll+trap, counter reset, output epilogue and host launch/map remain exact.
The split-domain static assertion permits only x5w5/FL1283 and x6w4/FL1027.
HALF16/EARLY32/EH65536/EP131072 and all probes are excluded from split arms.
EARLY's coupled initialization path is therefore never used for split rings.
RQPROD/RQBAKE/RQKS/RQFLAGS and complete prepare/selection code are unchanged.
The new variants are selectable only by this separate lab.

Local CUDA12.8 exact flags: sm90a,-O3,-DNDEBUG,--expt-relaxed-constexpr; CPU
compiler additionally uses -Xptxas=-v -cubin. Five candidate instantiations and
three parent controls use168registers, zero stack/spill stores/spill loads,
no C75. Uniform register pool168*384=64512; dynamic pool232*256+40*128=64512.
Actual SASS USETMAXREG alloc232/dealloc40 is checked. Existing current M128
ST5/FL2, ST5/FL3 and ST4/FL3 controls all match original parent at everyPC,
predicate, instruction, operand and branch target. Their source helpers remain
unchanged; no claim is made about uncompiled arbitrary configurations.

Protocol model explores53047 reachable interleavings for three arms/KCS1..8 and
1089 randomized complete schedules covering every valid QKV KC128/KS1..64
partition, including raggedKS7 lengths18/19. All barriers/regions/phase reuse and
startup/drain are proved; Wslot releases after all8warp LDS+syncwarp, Xslot after
wait1 retires previousgroup or finalwait0. Two WG witnesses each represent four
warp arrivals; finite TMA/WGMMA engine progress is an assumption. Negative
controls reject missing X publication, early X release and only7of8 W reads.
12288B packedW load mapping and8192BF16 SW128 X mapping remain bijective.
This is a bounded protocol/source model, not a general GPU CFG/scheduling proof.

`tools/wring_rebase_lab.py` is a new root-only four-arm driver. Source/manifest,
actual JIT exact-all-main SASS, physical Dwait0→output and resource/occupancy/
co-resident126CTA gates precede first candidate math. A host-only wrinfo query
wrapper adds actual occupancy/shared/thread attributes without changing kernel
body. The extension cache uses full generated-source SHA prefix.

Runtime gates: full8192basis exact, randomFP64 and official task normal/outlier
oracles for all4arms; fresh X/output; alloriginalweights/candidate+actualincumbent
packed+CSR immutability; active/dormant counter full-storage identity+guards;
20actualgraph NaN/input mutations; actual single-kernel candidate profiles;
reset/drain in finally. No silently accepted fallback. Active RQ counter prefix
is36int32 for NB18/KS7/full-widthZI. Visible candidate72int32 counters include
36dormant integers and8guard integers on either side; active entry/completion
must zero, tails/guards exactly preserve baseline. All unrelated prepared
counter allocations preserve exact prepare baseline, which may be nonzero;
physical aliases of the selected active allocation share that one contract.

Measurement retains all295calls:20warm,147F1,128paired across32rounds and8balanced
orders. Eight terminal output oracles inspect retained actual timed outputs
against eacharm's ownlastfixture at F1 and paired boundaries, before any replay,
mutation or profile. Six named causal contrasts include coupled-minus-equal;
every raw sample retained, no filtering or minimum-gain retention threshold.
CPU fake-event extraction confirms counts/orders/own-fixture boundaries. Seven
terminal corruption negatives and five counter-storage negatives pass.

CPU commands, no context, extension load or prepare:

```bash
python3 proj_rsi/candidates/pw36_wring_rebase/preflight.py
python3 proj_rsi/candidates/pw36_wring_rebase/cpu_lab_proof.py
```

After separate independent CPU review GO and root's exact frozen launch review,
root-only proposed normalcap command from iteration host `/root/okc/proj_rsi`:

```bash
bash tools/h100_trial.sh okc-proj-wring-rebase-01 300 240 bash -lc 'cd /okc/proj_rsi && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 tools/wring_rebase_lab.py --json /okc/logs/wring-rebase-01.json'
```

Cold prepare/JIT/validation may exceed normalcap; timeout retains raw INVALID
setup failure and requires a distinct reviewed attempt, with no silent cap edit.
The planning900second downstream reservation is not this execution cap and is
not an achieved runtime. No scheduler registration/mutation or GPU execution by
this author. Root owns readiness disposition/review registration/dispatch.

Retained harness failures: initial preflight reached successful3arm/parent
compile+control equality before protocol_proof.py existed; that incomplete
harness error is retained. Next physical checker looked for PTX SETMAXNREG spelling
instead of actual SASS USETMAXREG; corrected checker validates alloc232/dealloc40.
Candidate kernel source did not change for either error; no resource/codegen
failure was discarded. Future failures and losing valid measurements remain
attached to their actual version and attempt; no blanket family kill.

Manifest `files` keys are relative to THIS candidate directory. `external_files`
keys are relative to `proj_rsi/`. Parent source pin and historical source pin
are separate. Manifest does not claim GPU correctness or launch admission.
