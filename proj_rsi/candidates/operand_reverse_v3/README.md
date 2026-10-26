# Operand reverse V3: one coalesced8B / pairlane2x2 X-loader hypothesis

state: CPU_PROOF_AND_LOCAL_SM90A_COMPILE_ONLY; GPU_UNRUN; root_named_handoff + independent_critic_required.

scope: unpooler M128/N8192/K4096; KS1;128CTA;384threads; channel_tile64; W6/B8; plain output. Parent `rs32` helper SHA a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad. Actual incumbent is frozen pw36 SHA2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf `prepare` + `kernel_fn`; never force its RQ pick.

## Causal change / controls

- V1: four16B `ld.global.nc.L1::no_allocate.v4.u32` loads,32SHFL+32SEL conversion/thread/stage; rawQ4, separate asyncA2; four-stage lookahead; dynamic64producer/216math,168static entry registers.
- V2: sixteen4B `ld.global.ca.u32` native-A loads, no transpose; asyncA4 lookahead three; uniform162registers. Valid but57.024us vsV147.232/inc24.224 in its own unprofiled trial. This comparison changed cache policy, width, staging, and resource allocation; counters cannot isolate one alone.
- V3: eight8B `ld.global.nc.L1::no_allocate.v2.u32` loads +16SHFL/16SEL; restore V1 rawQ4/separateA2 four-ahead schedule and cache policy. Uniform152registers, no `setmaxnreg`, because source requests no role reallocation and static pool58368<65536. Same W producer/output bytewise except removed dynamic declaration; source proof checks this. Relative V1 changed width/demux/static resources, relative V2 changed width/cache/staging/demux. No claim that this is a one-variable cache experiment.
- V1/V2 memory diagnostic from root: L1requests262656->1049088; L1sectors4196352->8390656; V2hits4194304 while misses4196352 unchanged; DRAM52.152->51.473MB (~flat). Extra frontend traffic is plausible; no DRAM explosion. Diagnostic durations/stall ratios are not benchmark speed or a causal time decomposition.
- Predicted V3 logical demand per warp/stage:64touched=64unique32B sectors, same V1; V2=128touch/64unique. V3 has twice V1 load instructions but half conversion instructions. This trades load issue against transpose issue; speed remains unmeasured.

## Exact map / lifetime

For lane=(4g+t), warp w within math WG, half h, kstep j, stage s:

```
row = 64*wg +16*w +g +8*h
k_vector =64*s +16*j +2*(t&2) +8*(t&1)  #8B-aligned uint2
q={X[row,k_vector:k_vector+2], X[row,k_vector+2:k_vector+4]} as two rawu32
u=SHFL_XOR(q.x,1); v=SHFL_XOR(q.y,1)
A[4*j+h]   = (t&1) ? v : q.x
A[4*j+h+2] = (t&1) ? q.y : u
```

The resulting two pairs are exactly k=(64s+16j+2t) and (64s+16j+8+2t), all524288 symbolic BF16 indices, with all65536 possible pattern labels. No arithmetic/conversion of X bits. Inline PTX snapshots both inputs and evaluates predicate before output writes, so tied input/output allocation cannot early-clobber an input.

Initial rawQ stages0..3. Round s converts rawQ[s%4] to separateA[s%2], issues/commits WGMMA, refills rawQ slot withs+4, waits_group1, pins oldA after retiring group(s-1), then releases B(s-1). Thus no current async-A overwrite by prefetch. A slot reused ats+2 only after priorwait1/pin; finalwait0 then pin both A and releaseB63. CPU negative control catches62 absent-retirement hazards. Proof is an ordered source contract, not a GPU interleaving simulator. All barriers/zero-workspace semantics inherited.

Weight proof: independent packed12 decode/CSR repair -> exact524288 arbitrary BF16 storage bits, including signzero/subnormal/Inf/NaN payload labels, dense escape lists spanning many32entry windows. No lossless-weight or FP32 accumulation policy changes. Every8192Dscalar maps once to final128x64 SW128 slab, inherited FP32 WGMMA before BF16 emission.

## Build gate / compiler evidence

`python3 proj_rsi/candidates/operand_reverse_v3/preflight.py` never calls CUDA or launches a kernel. It checks CPU mapping/bit/lifetime/source identity, then one local nvcc sm90a cubin build, ptxas resources and SASS.

Actual local:152registers,0stack/spill,0LDL/STL,0USETMAXREG;103600B dynamic shared;16HGMMA/4WARPGROUP.ARRIVE/1outputTMA;0X-TMA;64static64B-LDG mnemonics (`LDG.E.NA.64.CONSTANT` means8B per lane,64-bit width).

`resource-first.log`/`sass-first.txt` preserve the initial C++-demux compilation. Initial96static BFLY was mistakenly suspected to be dynamic rematerialization. Inspection corrected this:64fixed-mask main-path BFLY plus32outlined collective alternatives entered by BRA.DIV for the first two unrolled rounds. Current explicit PTX also has96static BFLY, same152registers. Each alternative replaces16main-path BFLY and returns after that demux; remaining outlined TRYWAIT blocks are separate. Preflight checks fixed-mask64/collective32 and return CFG, without adding alternatives as runtime work or assuming they never execute. No measured lowering speed benefit is claimed. First source was not separately archived; first SASS/resource evidence is preserved.

Runtime gate repeats actual extension attributes: <=168registers,0local, opt-in occupancy>=1, shared103600; all extensions compile before parent prepare budget begins. V1 dynamic requests63488fit its64512entry pool; V2/V3 have no dynamic requests. Actual host extension build is still a required root GPU gate.

## Frozen trial contract

Command (root container only):

```
MAX_JOBS=1 python3 proj_rsi/candidates/operand_reverse_v3/lab.py --json /okc/logs/operand-reverse-v301.json
```

Budget: inner240s/outer300s, serialized exclusive root runner. No retries, geometry/cache sweeps, official slots, or full-submission port.

Lab hashes own manifest and frozen V1/V2 six executable/proof files; V1 README has later audit text and is intentionally not treated as executable provenance. Builds three source-named extensions, then actual pw36 prepare. Records scalar selected pick only. Four controls: actual incumbent, frozenV1, frozenV2, V3, each with separate real outputs and candidate counters. GPU checks for all: all4096K onehot exact, randomFP64, officialnormal/outlier, actualtimedgraphs withoriginal/zero/freshnormal/freshoutliers/probes, NaNpoison overwritten, input/weight/packed/escape immutability, workspace countersreset, profiler source identity.20actualgraphmutation checks total. No finite-only baseline loophole, no extra baseline timing copy.

Timing:256MiBflush outside events,5discarded warmups,21F1samples with each control before+after V3 (control=min of its two medians),21four-way rotated/reversed paired rounds. Report all samples, all three V3 deltas, wins, source/resource identities; reset all graphs before exit. Incumbent admission requires>=.5us both F1+paired; progress versusV1/V2 is retained even if incumbent faster. Invalid version requires diagnosis and a reviewed revision; no automatic family closure. No speedup promise or additive official-score projection before a valid result.
