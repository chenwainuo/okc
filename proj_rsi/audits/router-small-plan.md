# Router-small / one m8 diagonal tensor-folding prototype

State: design before code; root authorized one prototype, CPU proof + local sm90a resource/SASS gate, independent review, THEN one named GPU handoff. No sweep, full submission or source edits to pw36/pw35. Target >=0.5us at m8 in BOTH median paired cold graph and prepare-F1, fully validated baseline and candidate. Any spill, >64 registers, fault, timeout or correctness failure rejects it.

## Shipped evidence / exclusions

Read `handover_folder/{03,06,10}`, `audits/{p1-h100,router,r6-lineage,codec-study}`, pw36 source small-router paths, pw35 judge log and pw36 takeover-audit. Current judge small router: m1~3.2us rr6; m8~4.63us,16~4.87us,32~5.17us. m1 rr6 stays untouched. pw35 scored m8/16 picked rk-s16w8x, m32 rk-s8z; H100 pw36 can pick z instead.

Important byte correction: these winning rk paths use FMT2 PLAIN BF16 hi+lo, 256*8192*4 =8MiB. 6MiB is the hypothetical packed transport (or old compressed `gx-r12g8`, retaining last-arriver overhead), not all current small-router winners. New proof must preserve each hi and lo BF16 bit exactly; accumulate FP32 and check original FP32 W at1e-4 (strict shipped validator2e-5).

Closed, not to repeat: cluster/DSMEM, global partial workspace/last-arriver, memset+red, PDL and self-zero flags/zeroer planes, bulk W rings, generic TMA/mbarrier skeleton, variable-length codec E and11bit escape walks, x cache hints. P1 measured global reds themselves nearly free; ordering zero-before-red costs~0.8us even in best path. rr8 SIMT full-K/2or4rows loses7.3–14.7 kernel-side vs gv5.18: replacing tensor math with scalar FMA is not viable. RTKearly stage0-priority gate also closed; this is unrelated to m128/256.

## New mechanism: tensor folding inside a full-K CTA

128CTAs, each2experts x all8192K x up to8tokens;1024threads =32warps, each warp owns256K. Grid has enough CTAs to occupy128/132SMs without cross-CTA K splits. No global atomics, workspace, output zero, flags, cluster, TMA or mbarrier. Only one CTA barrier at final warp-partial reduction. All output stores overwrite.

For each warp, split its256K into4groups of64K,4steps of16K. A `mma.m16n8k16` uses A = X, B = W, BF16/BF16->FP32:

- B col =2*group+expert, group0..3,expert0..1; its16K row comes from that group's current16K step.
- A row =8*local_group+token; two calls use group pairs(0,1) and(2,3), each with all8tokens.
- Keep only C entries whose A group matches B group. The off-diagonal products are discarded.
- Every token/expert receives all4groups x32warps x64K exactly once. Within each lane's pair of C rows choose its matching diagonal; shuffle-reduce four groups, stage32warp partials (2KiB), one CTA barrier, sum32partials and overwrite16outputs.

The arithmetic matrix is25% useful, but it runs on BF16 tensor cores, not rr8's scalar FP32 pipe. Useful m8 math is67.1MFLOP; padding executes268.4MFLOP. Ideal H100989TFLOP cost0.27us, with128CTAs/132SMs almost full spatial coverage. Actual instruction/issue cost can be larger.32warps/CTA and64register/thread cap are strict; reject any spills instead of widening geometry.

Fragment formulas follow NVIDIA PTX `mma.m16n8k16` BF16 definition ([primary documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)). No external/private team code. CPU emulator reconstructs A/B matrices from independent PTX register-coordinate rules, computes C, selects diagonals, reduces, and compares original X@(hi+lo)^T, plus allK onehots. Layout proof must also detect deliberately wrong B group, A cohort and C diagonal choices.

## Fixed-stride lossless weight transport, no escapes

Each warp/plane owns512BF16 values. Encode a warp-adaptive high7 window: if max(h7)-min(h7)<=7, store low8bytes plus offset3/sign1 nibbles (12bits). Otherwise store that entire warp-plane as plain16bit. Header carries two7bit bases + two wide flags. Every original bit preserved; arbitrary distributions, zeros/subnormals/NaNs/Inf are losslessly representable via wide fallback. Numerical W legality still requires finite challenge weights.

Storage always1024B perwarp-plane, address affine inCTA/warp/plane; narrow transports768B, wide1024B. Header fixed-index4B perwarp. First24B/lane and header requests issue independently, before format-dependent decode; a wide warp-plane fetches its final8B/lane. No pointer table, per-lane variable stream, truncation, sparse fix, escape walk, runtime fallback to a different kernel or data-dependent output shortcut. Branch on immutable prepared weight format is warp-uniform.

CPU pilot, Gaussian FP32 weights256x8192, seeds5151/1234/17, BF16 hi and residual lo: wide hi246–261/4096warp-planes (~6%), lo945–965/4096 (~23%). Effective payload~6.30MiB plus16KiB headers; allocation8MiB. CPU and GPU generators differ, so actual trial must report its own exact rates/bytes and roundtrip all weights before launching candidate.

## ROI / risks / one trial

Potential removal:~0.8us global zero-ordering protocol, plus~1.7MiB fewer transported W bytes (~0.45us at3.8TB/s). Costs: redundant tensor products, more X traffic (16MiB L2->SM at m8, vs about2MiB in 16row-tile baseline), per-warp metadata/decoding, one CTA1024-thread barrier/reduction.128CTA one-wave alone does not guarantee bandwidth. A .5–1us gain is credible but low confidence; no claim until graph data. If local resource gate fails, close before GPU. If first trial fails, close all geometry/window/cohort tuning of this prototype.

Comparator must import hash-frozen actual pw36, run its prepare/bake and call its actual selected router path; record the pick. Full baseline correctness must pass the same allK onehots, original FP32/FP64 tolerance, zero input, normal, outliers, fresh input, post-timing captured output checks as prototype. Prepare style F1 baseline min(pre,post), median paired AB/BA cold graph. No r6/torch-only surrogate, no finite-only baseline, no subtraction of instrumentation from production time.
