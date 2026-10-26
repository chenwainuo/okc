# r6-c12 lineage audit: why `gv` is at the stream at m=1, what breaks it at m≥16, and the minimal extension

Source: `proj_rsi/candidates/r6-c12/submission.py` (line refs below). Judge job `23cecdef1bb54c9e`
(`profiles/submission.23cecdef1bb54c9e.txt`). SASS from nvcc 12.8 `-arch=sm_90a -O3` on the GVCU block (lines 501–733).
I instantiated the 14 deployed `gv<>` instances and 22 proposed ones. Scratch:
`/tmp/claude-1000/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/scratchpad/r6/` (gv.cu, prop*.cu, hot.py).
No okc runs.

Correction to the brief: in r6's **own** scored run, decode32 qkv and pooler are `gv`, not nvjet. The nvjet names
are from pw7's report, where pw7's bake fell to cuBLAS. In r6 only **decode32 o** is cuBLAS at m≤32.

## 1. Who serves what

`gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>` (l.548). FMT 1 = 12-bit BF16, 2 = router hi+lo plain BF16.
MT = number of n8 token tiles (m/8). ST = register-ring depth in 64-k chunks. WR = warps along rows (16-row tiles).
WK = warps along K. KU = chunks per warp. S = CTAs per cluster along K. RT = row tiles per warp.
XS: 0 = x in registers via LDG, 1 = x in a smem slab, 2 = x LDG'd at use through L1.
CTA rows = 16·WR·RT. K = 64·KU·WK·S (checked in `gvl`, l.736). Grid = (m/8MT, N/(16·WR·RT), S) (l.751).

| shape (m bucket) | kernel (judge profile) | CTAs × thr | CTA/SM | regs | x path | split-K | kernel µs / judge |
|---|---|---|---|---|---|---|---|
| qkv m1/8 | gv<1,1,2,1,8,16,1,1,0,1> | 144 × 256 | 1–2 | 106 | regs | 8 warps in CTA | 10.1 / 9.97 |
| o m1/8 | gv<1,1,2,1,8,2,1,1,0,0> | 512 × 256 | 4 | 64 | regs | 8 warps | 4.4 / 5.30 |
| pooler m1/8 | gv<1,1,1,2,16,8,1,1,0,3> (rolled) | 128 × 1024 | 1 | 64 | regs | 16 warps | 14.3 / 13.95 |
| unpooler m1/8 (n=1024) | gv<1,1,2,1,16,4,1,1,0,1> | 64 × 512 | 1 | 96 | regs | 16 warps | 3.5 / 4.88 |
| router m1 | `rr<1,1>` FP32 direct (l.760) | 256 × 512 | – | 28 | regs | 16 warps | 2.9 / 4.14 |
| router m8/16 | gv<2,1|2,2,1,8,2,8,1,0,0> | 128 × 256, cluster 8 | 1 | 86/110 | regs | DSMEM S=8 | 4.0 / 5.08, 4.4 / 5.14 |
| router m32 | gv<2,4,2,1,4,2,16,1,0,0> | 256 × 128, cluster 16 | 2 | 152 | regs | DSMEM S=16 | 5.2 / 5.88 |
| qkv m16 | gv<1,2,1,1,4,8,4,1,2,1> | 576 × 128, cluster 4 | 4 | 96 | L1 | DSMEM S=4 | 10.4 / 11.20 |
| o m16 | gv<1,2,2,1,8,2,1,1,0,1> | 512 × 256 | 2–3 | 79 | regs | 8 warps | 4.9 / 5.76 |
| pooler m16 | gv<1,2,2,2,8,16,1,1,0,1> | 128 × 512 | 1 | 128 | regs (L1 shared by 2 row-warps) | 8 warps | 14.5 / 14.30 |
| unpooler m16 | gv<1,2,2,1,4,16,1,1,2,1> | 512 × 128 | 4 | 110 | L1 | 4 warps | 14.9 / 14.84 |
| qkv m32 | gv<1,4,1,1,4,16,2,1,2,1> | 288 × 128, cluster 2 | 2–3 | 128 | L1 | DSMEM S=2 | 14.1 / 14.21 |
| o m32 | **nvjet** 128x16 (gv lost the bake) | – | – | – | – | – | 4.5 / 6.62 |
| pooler m32 | gv<1,4,2,1,8,16,1,1,2,1> | 256 × 256 | 2 | 128 | L1 | 8 warps | 16.8 / 16.94 |
| unpooler m32 | gv<1,4,2,1,4,16,1,1,2,0> | 512 × 128 | 4 | 128 | L1 | 4 warps | 16.5 / 16.73 |
| all m=128/256 | cuBLASLt nvjet on **uncompressed** bf16 (`_ltsearch` l.1294). Router: one GEMM on [hi;lo] + an elementwise add (l.1436–1443) | | | | | | 15–25 |

Dispatch: `kernel_fn` (l.1415) looks up `state["gv"][(layer, bucket)]`, which `_gvprep` (l.1016) bakes per shape
against `_v12` (cuBLAS or the older skinny/rtr kernels) with `GVMARGIN` 0.99. The table above lists the bake's
winners. `GVRB = []` (l.855) keeps gv off m=128/256. The router never uses 12-bit: `css = … if GVC12 and not hl`
(l.1028). The deployed router moves 8.4 MB of hi+lo BF16, while 12-bit hi+lo would be 6.3 MB.

## 2. Load path, decode, multiply, reduction, and why it sits at the stream at m=1

- **W load**: `gnc` = `ld.global.nc.L1::no_allocate.v4` (l.509). Per lane, one 64-k chunk of one 16-row tile is
  3 × 16 B: 2 low-byte words plus 1 code word. A warp-chunk is 1536 B, contiguous. Each warp streams one contiguous
  range of KU chunks (24.6 KB for qkv, 12.3 KB for pooler; offset `wo`, l.561).
- **Depth**: the prologue `ld(s,s)` for s<ST (l.578) is the first memory instruction after about 30 address ops.
  For qkv m1, the SASS issues **9 W LDGs (3 chunks; ptxas hoists one beyond ST=2) + 6 x LDGs** before any compute.
  The steady state has 3–4 chunks (144–192 B per thread) in flight. Pooler m1 is the rolled loop with ST=1 and
  1024 threads/SM: 48 B/thread, about 49 KB/SM, 6.3 MB GPU-wide.
  Bandwidth × latency ≈ 3.8 TB/s × ~1 µs ≈ 3.8 MB, so the whole bandwidth-delay product is requested within about
  100 ns of launch.
- **Decode** (`dec1`, l.532, V=1): `(C & 0x87878787) + base`, a funnel-shift, and 4 PRMTs make 8 values.
  That is 9 ops per 8 weights, **36 per chunk per warp, decoded once and reused for all MT token tiles**.
  Escapes come from a per-warp list of 32 sorted entries, one `nx>>20` compare per chunk (l.639); the slow path is
  branched around.
- **Multiply**: `mma.sync m16n8k16` with **W as the A operand** directly from the decoded registers, and x as B
  fragments. K is permuted inside each 64-chunk by `_pack` (l.1379), so lane t's 16 k values are contiguous, and x is
  two `ld.global.nc.v4` per n8 tile per chunk (l.563, l.572/604).
  Hot path per warp-chunk from the SASS: qkv m1 **59 instr** (3+2 LDG, 4 HMMA, 36 decode);
  pool16 66; pool32 79; qkv32 94 (16 HMMA + 8 x-LDG + 3 W-LDG).
- **Reduction**: K is split **across warps inside one CTA** (WK = 8/16) and the CTA owns only 16–32 rows.
  So N/16 ≈ 128–144 CTAs fill the GPU **without any cross-CTA split-K** at m≤8 for qkv/o/pooler/unpooler (S=1).
  The epilogue is `STS.128` → one `BAR.SYNC` → 16 `LDS.128` adds → `STG.U16` (l.684–694). There is no workspace,
  no atomics, no fence, no counter and no second kernel.
  S>1 uses a cluster barrier plus `ld.shared::cluster` (l.699–715); the last-arriver workspace (l.717) exists only
  for sm<90.
- **Epilogue cost**: about 32 outputs per CTA, ≤0.2 µs, paid per CTA while the other CTAs are still streaming.

**Why gv is at the stream and pw3–pw7 are 4.6 µs off.** Every warp is an independent stream with **no barrier
between launch and its last load**: no mbarrier handshake, no producer warp, no smem staging of W. All CTAs are
resident in one wave and the loads are in flight at launch. The kernel ends at the slowest warp's last load plus one
smem reduce, because **no CTA waits on another CTA**. pw tiles N coarsely (BN=256), so it needs cross-CTA split-K to
fill 132 SMs. That gives it a serialized tail: red/fence/counter/finalize, or a cluster barrier for every CTA.
pw4/pw5 showed this tail is the gap. r6 sidesteps it with **16-row N tiles and an in-CTA K split**.

## 3. What stops it scaling

**The price of 16-row tiles is that x is re-fetched once per 16 rows.**
x traffic is (N/R)·m·K·2 B, with R = rows sharing one x fetch (16 for most current configs).

| m | x bytes / W(12-bit) bytes at R=16 | LDG x per warp-chunk | HMMA per warp-chunk |
|---|---|---|---|
| 1 | 0.08 (all g-lanes hit one row) | 2 | 4 |
| 8 | 0.67 | 2 | 4 |
| 16 | 1.33 | 4 | 8 |
| 32 | 2.67 | 8 | 16 |

Fitting `t ≈ max(W/3.8, (W+X_L2)/B_L2) + ~0.7` to r6's own kernels gives B_L2 (L2→SM) ≈ 7–7.5 TB/s.
- qkv32: 28 + 75 = 103 MB in 14.1 µs, 7.3 TB/s.
- pool32: 50 + 67 MB in 16.8 µs, with the 2 co-resident CTAs sharing x via L1.
- pool16: 84 MB, still HBM-bound at 14.5.

The exact L1 sharing is not measured; X lies between the per-SM-shared and unshared values.

A second, smaller limit: an x `LDG.128` touches 8 rows × 64 B (half of 4 sectors). By my estimate that is about 8 L1
wavefronts per 512 B, which puts the LSU at about 45 % busy at m=16 and about 80 % at m=32 at stream rate.

What the data rules out:
- Decode does not scale with m: it is shared by all MT tiles.
- HMMA issue is about 25 % of the mma.sync peak at m=32.
- Issue slots are about 21 % at m=32.
- No spills: every deployed instance is clean, at 128 regs where capped.

**qkv m1 (+1.5 µs over ceiling while pooler is at ceiling): geometry.** N=2304 gives 144 tiles of 16 rows, so 144
CTAs on 132 SMs. 12 SMs carry 2 CTAs (393 KB each) while 120 carry 196 KB, a 1.83× max/mean load. Pooler has
exactly 128 × 1 CTA. The bake's alternatives (S=2: 288 CTAs, 1.38×; S=4: 576 CTAs, 1.15×) pay a cluster reduction
and more x traffic, so they lost. Modelled cost: the doubled SMs finish about 1.5 µs late, which matches 10.1 vs
8.6 µs. This is inferred, not measured.

**o (all m)**: 6.3 MB is dominated by fixed latency. 512 thin CTAs (each warp loads only 2 chunks, then a WK=8 smem
reduce) run at 4.4 µs, against the 3.4 kernel-µs bare 6 MB read. At m=32, x is 5× W at R=16 (33.5 MB), so gv lost
to nvjet.

## 4. Minimal extension: keep the load path, decode, mma.sync and the in-CTA/DSMEM reduction; raise R and balance CTAs

Almost everything below is a **new GVC table entry** (l.837). The template already supports any WR, RT, XS and S.
I compiled each one with nvcc 12.8, sm_90a.

| shape | proposed geo (WR,WK,KU,S,RT,XS) [ST] | CTAs × thr | R | X_L2 (MB) | regs / spill | r6 judge → expected |
|---|---|---|---|---|---|---|
| qkv m1/8 | (9,1,16,8,1,0) [2] | 128 × 288, cl 8 | 144 | ~0 | 106 / 0 | 9.97/9.90 → **~8.7** |
| qkv m16 | (9,1,16,8,1,2) [2] | 128 × 288 | 144 | 4.2 | 108 / 0 | 11.20 → **~8.9** |
| qkv m32 | (9,1,16,8,1,2) [2] or XS1 [3] | 128 × 288 | 144 | 8.4 | 152 / 0 (XS1: 128 / 0) | 14.21 → **~9.0** |
| pooler m32 | (8,2,16,4,1,2) [2] or XS1 [3] | 128 × 512, cl 4 | 128 | 16.8 | 128 / 8 B (XS1: 128 / 0) | 16.94 → **~14.3** |
| pooler m16 | same, MT=2 (XS1: 106 / 0) | 128 × 512 | 128 | 4.2 | | 14.30 → ~14.2 |
| unpooler m16/32 | (8,2,16,2,1,2) [2] or XS1 [3] | 128 × 512, cl 2 | 128 | 8.4 / 16.8 | 128 / 8 B (XS1: 128 / 0) | 14.84/16.73 → **~14.3** |
| o m16/32 | (4,4,4,1,1,1) [3] | 128 × 512 | 64 | 4.2 / 8.4 | 111 / 0 | 5.76/6.62 → ~4.6 |
| o m1/8 | (4,4,4,1,1,0), all chunks in flight | 128 × 512 | 64 | – | – | 5.30 → ~4.7 (low confidence) |
| router m8–32 | FMT 3 (12-bit hi+lo), (1,4,2,16,1,0) or WR=2 | 128–256, cl 16 | 16–32 | ≤4 | 156 / 0 | 5.08/5.14/5.88 → ~4.0 |
| router m1 | FMT 3 (1,8,1,16,1,0) | 128 × 256 | 16 | – | 80 / 0 | 4.14 → ~3.7 |

Reasoning:
- With R = 128–144, X_L2 is ≤ 0.33 W at m=32, so the shape returns to HBM-bound.
- 128 CTAs at 1/SM balances qkv like pooler.
- The S-way split is r6's existing DSMEM path. Its cost is visible in qkv16 (S=4, 10.4 vs 10.1 at S=1) and the
  router (S=8/16): ≤0.3–0.5 µs. Partials are tiny (NI = WR·RT·MT·32 float4, ≤16 KB).
- XS=2 with WR≥8 relies on the WR row-warps of one CTA reading identical x lines, so L1 dedups them. The CTA's x slice
  is ≤128 KB, which fits in L1 next to ≤48 KB of smem.
- XS=1 (smem slab, conflict-free LDS.128, half the wavefronts) fixes the LSU term too. It front-loads the slab
  (16.8 MB GPU-wide at pooler m32, about 2 µs of L2→SM) behind ST=3 chunks of W already in flight. Bake both.
- **If the LSU term binds at m=32** (XS=2 lands around 15), the next step is RT=2: x fragments reused from
  registers, halving both terms. It needs one code change: the launch-bounds rule at l.549 forces ≥512 threads/SM
  for XS=2, which caps regs at 128, and RT=2 then spills 184–776 B. With minBlocks=1, RT=2 compiles at 198–226 regs
  with 0 spill, but in flight drops to about 25 KB/SM (8 warps), which is below the bandwidth-delay product. Second
  choice.
- Load sums:

  | load | r6 → expected (µs) | ln |
  |---|---|---|
  | decode1 | 38.2 → ~36.0 | 0.06 |
  | decode16 | 51.2 → ~46.0 | 0.11 |
  | decode32 | 60.4 → ~46.2 | 0.27 |
  | verify1 | 38.9 → ~35.4 | 0.10 |

  That is about **0.53 ln of the 0.67 available** for these four loads at ceiling. With pw9-class m=128 gains
  (about 0.3) this clears the 0.80 needed.

Risks:
1. **Cluster residency.** 16 clusters of 8 (or 32 of 4) at 1 CTA/SM need each GPC to host ≥16 co-resident CTAs.
   On a 132-SM H100 with uneven GPCs this can force a partial second wave. Check
   `cudaOccupancyMaxActiveClusters` in prepare and fall back to S=4/R=72 if it is under 16.
   pw7's cluster epilogue was 5–10 µs worse; r6's own S=4/8/16 instances are not, but that difference is unexplained.
2. The L2-bandwidth model is fitted, not measured. One bake line answers it (pooler m32 R=128 XS2 vs XS1 vs current).
3. Escape lists are 32 per warp. A warp now spans the same chunks (KU=16) but distinct rows, so `_gvesc` (l.940)
   overflow rates are unchanged per warp. The router lo matrix under 12-bit is the real unknown: overflow means the
   plan is dropped at prepare, not wrong results.
4. 8 B spill at 128 regs in the XS=2 R=128 variants: harmless, but prefer XS=1 or trim ST.
5. qkv uses 128 of 132 SMs (3 % idle); `rr` already wins router m1, so FMT 3 there is marginal.

## m=128/256

r6 is cuBLAS on **uncompressed** bf16 (67 MB pooler) with a cuBLASLt pinned-config search. The router is one
[hi;lo] GEMM plus a 1.6 µs add kernel. `GVRB=[]`: gv at m=128 was slower because mma.sync is issue-bound there.

**r6's load path can feed wgmma directly.** The decoded `A[16]` per lane is the m16n8k16 A fragment for 4 k16 steps,
and wgmma's register-A layout gives each warp of a warpgroup one 16-row slab in that same per-thread layout. So
WR=4 warps (one warpgroup), each decoding its own 16-row tile of the same chunk, form an m64 A operand.

Two adjustments are needed:
1. **Repack W in natural K order.** `_pack` permutes K inside each 64-chunk to make x LDGs contiguous. That is free
   at prepare and keeps the codec, but TMA-staged x in canonical smem layout cannot reproduce the permutation.
   This is the "K-order correction" in `wgmma-layout.md`.
2. **Double-buffer the A registers.** A must stay live until `wgmma.wait_group`.

The rest stays as r6 has it: W register-direct with no smem ring, x via 2D TMA per CTA, and the split across
K by ≤2 using a DSMEM cluster with small partials. Compute at pooler m128 is 8.6 GFLOP ≈ 11 µs at ~800 TF, against
a 13.2 µs stream, so the whole decode must overlap async wgmma.

## Post-mortem of pw11 (check 824f32ce203541c4: no gx path won; in-prepare µs from its log)

**1. Which assumption failed**

- **The cluster designs were mostly never measured.** Every 1-CTA/SM cluster variant with KS=4 or 8 was `declined-occ 120<128`.
  This H200 co-schedules only 120 CTAs in clusters of 4 or 8 (my audit's risk 1). The cluster variants that did run are
  q128a (2 CTAs/SM) and u128a (KS=2).
- **The last-arriver tail is the 2 µs.** Same geometry, b vs a:
  - q128 at m=1: 17.34 vs 15.26 (+2.1), with only 0.6 MB of partials.
  - u128 at m=32: +1.2 (x via L1) and +4.1 (x in smem); at m=16: +0.2.

  The pw11 SASS shows the dependent chain: partial `STG.STRONG` → `MEMBAR.ALL.GPU` (waits for L2 to acknowledge) → BAR →
  `ATOMG` (result needed) → BAR → `MEMBAR` → `LDG.STRONG` of the peers → STG. That is 3 dependent L2 round trips plus 2
  CTA barriers.
  - In a one-wave grid where every CTA finishes together, nothing hides that chain.
  - Counter contention is not the cause (≤8 arrivals per counter), and neither is traffic.
  - The +4.1 at m=32 with x in smem is larger than this chain explains; I cannot attribute it without a probe.
- **The fitted 7 TB/s L2→SM limit is wrong.** unpooler m=32 u128a-s cuts x L2 traffic about 4×, to 16.8 MB, and removes the
  L1-wavefront term, yet only ties r6 (21.47 vs 21.15) while paying a KS=2 cluster reduction worth ≤1 µs.
  So x traffic costs r6 at most about 1 µs at m=32, not the 2.5 µs the model said. What makes r6's m=32 slower than its m=1
  is still unattributed.
- **Per-warp split:** secondary. qkv144 at 1 CTA/SM has 9 warps × ST=2 ≈ 27 KB in flight per SM, against r6's 43–87 KB.
  The p128/u128 variants match r6's in-flight bytes. The XS=2 / MT=4 variants also spill 8 B (about 120 B of LDL).

**2. The 12-bit router**

- **Not the escape rate.** The codec's 3-bit offset indexes the upper 7 exponent bits, so its window spans 16 binades (the
  0.05% case). The judge log shows 1063 lo escapes out of 2.1 M weights, about 1 per warp, and no list overflowed.
- **Not the N=256 geometry.** r12c8 has exactly r6's router geometry (cluster 8, 128 CTAs) and is still +1.6 at m=8
  (11.14 vs 9.57).
- **The cause is decode on the critical path.** With KU=2, each warp's whole W slice is requested at launch, so the kernel is
  one round trip plus drain. It is latency-bound (r6's 8.4 MB in about 4.3 kernel-µs against a 2.2 µs stream), so cutting
  2.1 MB saves nothing.
  - What it adds is the decode of both halves plus the escape walk between data arrival and the HMMAs: 936 vs 512
    instructions, about 420 more on each warp's critical path.

## pw14 ladder: what the m=32 excess is made of (check 066cf9bd503040cf, in-prepare µs)

This corrects the pw11 post-mortem. x costs **2.5–2.8 µs**, not ≤1. The u128a-s comparison was also paying for a cluster
reduction.

| step | pooler m=1 (`1.1.1.2.16.8.1.1.0.3`) | pooler m=32 (`1.4.2.1.8.16.1.1.2.0`) | qkv m=1 (`1.1.2.1.8.16.1.1.0.1`) | qkv m=32 (`1.4.1.1.4.16.2.1.2.1`, ST 1) |
|---|---|---|---|---|
| L0 W only | 18.24 | 16.93 (126 regs, 2/SM) | 13.31 | 12.51 (4/SM) |
| L1 + x | – | 19.74 (**+2.8**) | – | 15.01 (**+2.5**) |
| L2 + decode | – | 20.51 (+0.8) | – | 18.53 (74 regs, **6/SM**) |
| L3 + mma | 18.40 | 20.74 (+0.2) | 13.95 | 17.22 (4/SM) |
| L4 real | 18.05 | 21.02 (+0.3 reduce + epilogue) | 14.21 | 18.56 = cuBLAS (r6's bake picked v12 this run) |

**pooler.** The m=32 excess over m=1 (+3.0) is the x loads. Decode, mma and the epilogue add about 1.3 together.

Caveat: m=32 L0 has all 48 W loads hoisted (126 regs). Part of L0→L1 is therefore W depth lost to x's 32 registers at
the 128-register cap.

**qkv: why decode costs more at m=32 than at m=1.** Decode instructions per W byte are the same at both m. The difference
is the geometry:
- **m=1** runs 8 warps × 16 chunks (full K) per CTA at 2 CTAs/SM with ST=2. ptxas hoists 3–4 chunks, so about 70–90 KB
  are in flight per SM.
- **m=32** runs 4 warps × 16 chunks (half of K, S=2) per CTA with **ST=1** (the forced 7th field), at 4 CTAs/SM. That
  leaves 16 warps × 1 chunk ≈ 24 KB in flight per SM, and each warp's next W load waits for the current chunk's x loads
  and decode.

So at m=32 decode and x sit on the per-chunk critical path instead of under the stream. L2 (6/SM) being slower than L3
(4/SM) shows contention for the LSU and L1 between x loads and more warps, not ALU cost.

## Router (pw17 analysis; in-prepare offset-subtracted bake lines from r6, pw14 and pw16)

- **Above the stream:** rr (FP32, m=1) runs 3.2 µs, about 0.7 over 8.4 MB / 3.8 TB/s. That is one DRAM latency plus
  launch, since all bytes are requested at t=0.
- **gv's structure costs about 1.0 µs over rr at the same bytes.** c8 runs 4.2 at m=1 and m=8 against rr's 3.2. The
  cost is the cluster launch, 2 cluster barriers, the DSMEM gather, the in-CTA BAR plus 8-way reduce, and a 16-deep HMMA
  chain (about 0.27 µs, because hi and lo share one accumulator).
- **What grows with m:**
  - x is re-fetched by every 16-row tile: 2, 4 and 8.4 MB at m=8, 16 and 32, against 8.4 MB of W.
  - c8 at m=32 needs 152 regs, so 1 CTA/SM. With 120 co-resident clusters of 8 against 128 CTAs it runs a second wave
    (8.1 µs), which is why r6 switches to c16 (5.1).
- **What does not help:** smaller K slices (r6's KU=1 geometry is 1.5–2 µs slower), interleaved hi/lo (loads are not on
  the critical path), and the FP32 write (already direct).
(The router variants were built as `candidates/pw17r-router-dropped/` and then dropped on the priority change; never run.)

## Fixed-width 11-bit codec: what it can deliver (analysis before building)

Data: Gaussian weights with the reference generator's distribution, 4096×8192, matching the judge STATS (esc8_best
0.0136).

**Window.** The best 8-exponent window is 114..121. It starts on an even exponent, so r6's own decode
(`(C & mask) + base`, PRMT with the verbatim low byte) works unchanged with a 2-bit offset of `e>>1` plus the sign: a 3-bit
field. Escapes are 1.35%: 0.88% below the window, 0.48% above.

**Escape counts and the walk.**
- Per warp-chunk (16×64 = 1024 weights): 13.9 on average, at most 30. Per warp list at KU=16: about 222.
- r6's walk is warp-serial: one shfl per entry, then a 16-way predicated select of about 40 instructions. That is about
  550 instructions per chunk against the hot loop's 60, so it is not usable.
- Per lane-chunk (32 values): 0 escapes 65%, 1 29%, 2 6.2%, 3 0.85%, 4 or more 0.09%.
- A SIMT per-lane fix-up (XOR a byte into the hl/hh word before the PRMT) runs once per iteration of the warp's max lane
  count. The expected iterations are about 2.3 per chunk at about 25 instructions each, so roughly 60 per chunk: 2–2.5×
  r6's decode.

**Bits, including a decodable escape form.** An escape needs its position (5 bits within the lane-chunk) plus 8 exponent
bits, about 14–16 bits, so 1.35% × 16 ≈ 0.22 bits per weight before list or slot overhead.
- Per-lane inline slots: one slot covers only 94% of lane-chunks, so a second-level list fires in 86% of warp-chunks;
  two slots cost a full bit, which is back to 12.
- Realistic total: **11.3–11.5 bits**, a 4–6% byte cut.

**Payoff.** pooler m=1 saves 0.5–0.8 µs of stream at best, with decode doubled. The kill bar (≥0.7) sits at the
theoretical maximum. The what-if row for 11.4 bits is 3353, not the 3297 of 11.0.

**Where the entropy is.** Exponent mass: 120 32%, 119 24%, 121 15%, 118 14%, 117 7%. The best 4-window covers 85.5% and
the best 16-window 99.99%. Reaching about 10.6 needs a variable-length exponent, where a 2-bit code covers 85%.

## Variable-length exponent code: per-m economics and gv integration (analysis; build waits for the decode routine)

**1. Budget.** At 3.8 TB/s and 1.98 GHz each SM gets 14.5 B/clk, so one 12-bit warp-chunk (1536 B, 32 weights per
lane) arrives every 106 clk and one ~11-bit chunk (1408 B) every 97 clk. That is 4 × 97 ≈ 390 warp-instructions of issue
per chunk. r6's hot instructions per warp-chunk are measured from SASS, escape walk excluded, prologue and epilogue
amortized.

| shape | r6 geometry | warps/SM | r6 instr/chunk | issue use | extra instr/weight at 70% | per-warp clk/chunk | regs (cap) |
|---|---|---|---|---|---|---|---|
| pooler m1/8 | 2.16.8 rolled | 32 | 77 | 20% | 6.1 | 3100 | 64 (64) |
| qkv m1/8 | 1.8.16 | 16 (2 CTAs) | 59 | 15% | 6.7 | 1550 | 106 (255) |
| pooler m16 | 2.8.16 | 16 | 66 | 17% | 6.5 | 1550 | 128 (128) |
| qkv m16 | 1.4.8 S4 | ~18 | 113 | 29% | 5.0 | 1750 | 96 (128) |
| pooler m32 | 1.8.16 XS2 | 16 | 79 | 20% | 6.1 | 1550 | 128 (128) |
| qkv m32 | 1.4.16 S2 | 16 | 94 | 24% | 5.6 | 1550 | 128 (128) |

Issue is not the limit at any m. These three are:
- **Registers.** The bit buffer, position and refill ring add about 10–16 registers. pooler m1/8 sits at its 64 cap (1024
  threads), so it must move to r6's own 2.8.16 / 512-thread geometry (0.2 µs slower in r6's bake). Every m=16/32
  geometry sits at 128.
- **The serial chain.** Each lane decodes 32 symbols per chunk in sequence. At about 25–35 clk per symbol (or per pair
  with a lookup) that is 500–1000 clk, inside the 1550–3100 budget but not by much.
- **LSU / MIO at m=32.** The pw14 ladder put x loads at +2.5–2.8 µs there. A shared-memory lookup table adds 16 LDS
  (2 symbols each) per lane-chunk on the same pipe.

Verdict: bake VL per (layer, m). Expect picks at m ≤ 8, maybe m=16, and r6's 12-bit at m=32. Keep both packings
resident.

**2. Integration.**
- **Stream layout.** Use per-lane streams **interleaved at 16 B** inside each warp's K range: lane l's k-th 16-B piece at
  `(k·32 + l)·16`. The lane owns the weights it holds in A-fragment order. Addresses are computable, so no offset table
  and no dependent first load.
- **Padding.** Each warp-stream is padded to its longest lane: E[max of 32] ≈ mean + 2σ ≈ +10 B on about 700 B per lane
  (about 1.4%).
- **Loads.** Lanes drift by at most about one piece, so the warp LDG.128s stay nearly coalesced and sector-complete.
  Prefetch addresses do not depend on the data, so a ring of D pieces per lane, refilled by a predicated
  `LDG.128 if consumed`, keeps r6's depth.
- **Decode table.** A 256 × 4 B table has about 3.5-way random bank conflicts. Either replicate it per bank (32 KB, lane
  index = bank, conflict-free) or decode the 5-class code arithmetically.
- **Unchanged parts.** Escapes drop to the 0.01% outside-window list (r6's walk is fine at that rate), and the A-fragment
  registers come out as today, so mma and the epilogue do not change.

**3. What breaks.**
- **First chunk.** The prologue issues ring pieces at launch exactly as r6 does. The first decode waits for piece 0 only,
  so there is no extra latency.
- **Tail.** Lanes finish at different pieces; ring loads past a lane's end are clamped to the padded warp-stream.
- **Bake.** Two packings (about 150 MB each), per-(layer, m) selection, and a pack-time `STATS<layer> bits=<effective incl.
  padding> esc=<rate>` line.
- **Validation.** One-hot ×8 exact per served m, plus odd m.
- **Compile.** One more FMT instantiated per geometry.
