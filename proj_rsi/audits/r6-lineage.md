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
