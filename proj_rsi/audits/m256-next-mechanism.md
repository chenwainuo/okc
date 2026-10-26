# Next mechanism: reverse the RS operands; falsify first on unpooler m128

`decision=GO_CPU_DEVICE_FEASIBILITY_ONLY; primary=unpooler(M128,N8192,K4096),KS1; GPU=UNRUN; m256_followup=CONDITIONAL`

The materially untried mechanism is **X in register-A, losslessly decoded W in shared-B**, with one decoded W tile reused by two cooperative math warpgroups. It removes X TMA/shared delivery from the closed transform pipeline. The narrow first falsifier is unpooler m128:64 output channels×all128 tokens×full K;128 CTAs, two math WGs plus one producer/decode WG, one plain `{64,128}` TMA output store. No global split/reduce/counter/memset. This is a design proposal, not a measured improvement or a universal m256 solution.

## Current score and measured bottleneck

Latest pw36 job`e22a4b9d693d4902`:9/9,3212.415452µs, source2a067059…813bccf. Best scored rs32`fb548b5dc6f04696`:3200.615008µs. The scores overlap CI; do not promote pw36 on noise. Exact benchmark and profiler observations are separate contexts:

| Shape | rs32 benchmark/kernel µs | fresh pw36 benchmark/kernel µs |
|---|---:|---:|
| qkv m256 |17.277/17.6|17.768/17.8|
| pooler m256 |24.838/25.2|24.370/25.0|
| unpooler m256 |24.693/25.6|24.487/24.8|
| qkv m128 |—|12.922/12.8 rq|
| pooler m128 |—|18.152/18.6 rq|
| unpooler m128 |—|19.103/18.6 Nvjet|

All three current m256 profiles contain one Nvjet kernel. They do not expose a removable extra-node tax. Handover04's measured rq closure is more diagnostic: resident NT256 n256/two-pass1027–1033clk versus1024 tensor floor; streamed/no-epilogue25.6µs versus22.2 resident; output reduce tail≈2.4µs/8MB partials. That closes “tune the tensor issue loop” as the missing lever in that design. Actual tensor clocks differ from nvidia-smi; do not convert cycles with a reported clock.

The source/measurement reason for operand reversal is pw19's failure: packed-W→decoded-shared-A plus shared-X/SS WGMMA. At the64channel×128token shape, its stage traffic is60KiB/k64 and W+x delivery cost≈30.4µs versus19.9 W-only; full43.8. The new path uses the same lossless W arithmetic but removes the X shared fill and X shared read. It also removes the consumer's dependency on an X full barrier. This does not reduce X's global/L2 byte demand, and register layout formation is expensive; those are the explicit counter-hypotheses.

## Architecture and accounting

Fixed primary geometry: channel tile64, allM128, full K4096,128 CTAs. WG0 processes token0..63, WG1 token64..127. W producer WG's four warps decode64 W rows. Packed W uses inherited6KiB/64channel/k64 chunks and original exact escape lists. Keep packed-W TMA staging; **do not reuse the newly failed direct-GPR W loader**. Producer publishes one8KiB SW128 BF16 shared-B tile per k64. Both math WGs read it; X loads directly into GPRs using coalesced vector loads and a four-lane register transpose.

Proposed fixed rings: packed-W6 slots×6KiB=36KiB; decoded-B8 slots×8KiB=64KiB. Including alignment/control≈103.6KiB/CTA, below232448B opt-in. One producer WG can prefetch/refill W with an elected lane while all four producer warps decode; explicit producer-only barriers order reuse. No fourth WG is planned. Every producer writer must obey generic→async proxy visibility; this is not assumed to be a free single-lane fence. Math WGs release B only after their covering WGMMA groups retire.

| Shared traffic per k64, same64channel tile | pw19 SS m128 | reverse m128 | pw19-style SS m256 | reverse m256 |
|---|---:|---:|---:|---:|
| packed-W fill+LDS |12KiB|12KiB|12KiB|12KiB|
| decoded-W store |8KiB|8KiB|8KiB|8KiB|
| X shared fill+read |32KiB|0|64KiB|0|
| decoded-W WGMMA reads |8KiB|16KiB|8KiB|32KiB|
| total |60KiB|36KiB|92KiB|52KiB|

W in shared-B is read once per64-token set, hence more W shared reads than old shared-A. The net reductions are40% at m128 and43.5% at m256. The old128B/clk aggregate shared-port model is **conditional**, not established by port02 (its intended traffic rate failed). With that model, m128 reverse288clk vs256 tensor; m256416clk vs512 tensor. These figures are traffic budgets, not hardware ceilings or latency predictions.

X register layout is not free. A64-token×k64 set requires four coalesced128-bit loads/thread, four4×4 lane transposes:32SHFL+32SELP/thread plus addressing. Two token sets/WG at m256 double that. Preserve BF16 bits exactly; no casts to lower precision, TF32, FP8, pruning, or approximate input transform.

X register pressure at m128: D32 + double A32 + raw-X prefetch4×16=64 ⇒128 core registers/thread, plus addresses/transposition/control. At m256: D64 + double A64 + raw-X prefetch2×32=64 ⇒192 core registers/thread before overhead; that is much tighter and may underfeed X. The m256 case is not admitted now.

**Launch-resource gate:** dynamic register sums do not prove launchability. A384-thread kernel with static numRegs200 would require76800 registers and fail before setmaxnreg. Existing rq's `__launch_bounds__(384,1)`, static168/0spills, producer40/math232 is evidence that Hopper/compiler support the inherited redistribution pattern, not evidence that new code fits. The initial64/224 proposal incorrectly requested65536 from a64512-register initial CTA pool; independent review caught it before any GPU launch. Corrected producer64/math216 requests63488≤168×384=64512, leaving1024 registers in that CTA pool. Unassigned SM registers are not assumed available. The source must still compile under384-thread launch bounds with static≤168, zero stack/local/spills, and no C75 serialization; exact USETMAXREG immediates and consumer liveness need review. Then runtime must set dynamic shared opt-in **before** occupancy queries and verify occupancy≥1 and a launch. If the compile/resource gate fails, stop without flag/geometry sweeps. No claim of runtime fit precedes those gates.

Output native WGMMA coordinates are now `[token][channel]`; the old transposed rq epilogue must not be copied unchanged. Store every D pair into a SW128 `{M,64}` slab. The full slab is one TMA store, not per-warp stores. CPU proof must cover X register transpose, exact decoded W SW128 placement, D output bijection, and all bit patterns/escape reloads. A full GPU one-hot/probe/replay matrix is still required before timing admission.

## Transfer and ROI, without mixing scores or timing units

The same64channel×allM policy gives pooler64 row tiles×KS2=128CTAs, unpooler128 row tiles×KS1=128CTAs. At m128 unpooler uses plain store; pooler requires a split tail. At m256 unpooler still uses KS1/plain store; pooler's partials would be4MB/128TMA reductions versus old rq8MB/256 reductions if assembled into one `{64,256}` slab/CTA. That is a conditional followup, not permission to code pooler/m256 now. qkv's36 row tiles need a different scheduling decision (KS3=108CTAs or KS4=144); no qkv performance promise follows automatically.

For the same64channel tiling, global/L2 X is134.2MB at m128 and268.4MB at m256 on pooler/unpooler, plus50.3MB packed W. Operand reversal reduces shared delivery but does not reduce these bytes. Current rq's128channel tiling reads half as much X. If global/L2 X delivery or register transpose becomes binding, the new path loses despite the smaller shared traffic. This is why a resident-only result cannot establish a win.

Exact reported score is4830.8/geomean(SGLang_load_sum/ours_load_sum), using the same job's six SGLang load sums. The old62×geomean shortcut is stale. Each what-if below uses `score_after=score_before×geomean(ours_after/ours_before)`, holding that job's actual SGLang vector fixed. With rs32's actual3200.615/verify32 sum86.58485, all three m256 layers at their SOL/0.85 would yield3104.756 (−95.859). At full SOL they still yield3045.674. With fresh pw36, the corresponding values are3115.707 and3055.254. Therefore m256 alone cannot reach3000; the earlier≈85score-µs estimate is not a complete roadmap. A hypothetical3µs total reduction in fresh pw36 verify16 yields3187.416 (−24.999). Unpooler gains1µs at m128 plus2µs at m256 yield3191.512 (−20.904) on that same score vector. These are exact conditional projections, not addable gains from different judged runs.

## Narrow first falsifier

1. CPU/source/SM90a device feasibility only: one unpooler m128 KS1 prototype; mandatory resource gate above. No full submission or inherited edit.
2. Independent review before root schedules one bounded complete-output trial. Comparator is actual frozen pw36 prepared `kernel_fn`, including its tuned Lt/rq choices; never untuned torch.
3. Require all4096 one-hot columns exact, lossless weight pack, normal/outlier FP64/probe checks, fixed-pointer fresh/zero replay, input immutability, and all D fragments observed. Reject any error or silent fallback.
4. First performance kill:≥0.5µs gain in BOTH lower-before/after F1 baseline and median paired cold-L2 graph timing. Otherwise close this mechanism, no automatic parameter/flag retries. No instruction/period or CUPTI-only victory substitutes for complete graph timing.
5. Only after that survives may root consider pooler/m256 transfer. An85%-peak claim would require actual complete-kernel measurements and register/ingress accounting, not this document. Targeting≥20score-µs requires additional measured shapes; the first0.5µs unpooler win alone is≈4score-µs.

Distinctness/closures respected: pw19 staged both operands and used SS; pw15 staged W and had one WG or alternating-K ping-pong; current rq decodes W in math and shares X; port02 was inconclusive with an underfed producer; o_direct_w's W-GPR loader lost and is not reopened; stream-K and runtime patch-table tracks lost. None tested Xregister-A/decodedWshared-B/allM/two cooperative math WGs/plain unpooler store. This is permission for a narrow CPU feasibility exercise only; numerical/rule constraints are unchanged.

## CPU feasibility update

The authorized single unpooler128 prototype is now in `candidates/operand_reverse/`. Corrected64/216 allocation compiled under the first fixed geometry/flags:168static registers,0stack/spills,no C75. SASS confirms DEFACTO64producer/216math instructions,16HGMMA/4WARPGROUP.ARRIVE/one plain output TMA/no X TMA. Independent CFG review bounds producer register useR53/mathR158 and finds no remaining source lifetime blocker. CPU mapping proves524288 arbitrary X bits,524288 W bits and every native output D scalar. The CSR bit reconstruction and sequential slot-tag model do not substitute for actual escape-window/barrier-interleaving runtime tests. Generated source SHA`ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8`; runtime/full-Torch-build/GPU correctness/performance remain pending final review and root scheduling. No pooler or m256 kernel was implemented.
