# latency: one node, every byte in flight at launch (muse_bf16_proj, m <= 32)

Clean-room design (no reading of the incumbent). School: latency and launch structure.

## Thesis

At m <= 32 every layer is a weight stream; what separates the incumbent from the HBM floor on the
small shapes (router, unpooler slice, o) and on qkv is fixed cost, not bandwidth: extra graph nodes,
launch ramps, cross-CTA reductions and wave tails. So: one kernel node per call, 128 CTAs in one wave,
no clusters, no cross-CTA reduction, all resident stages issued by TMA at the first instruction, the
output written exactly once. The same kernel template serves all five layers and every m <= 32.

## Data layout (prepare, lossless)

bf16 weight w = [s][E:8][mant:7]. Per layer a 16-exponent window [2hb, 2hb+15] is chosen from the
maximum exponent (hb = ceil((Emax-15)/2)); it holds all but ~1e-4 of the weights.

* L plane (byte): the low byte of the bf16, E0|mant (unchanged bits).
* H plane (nibble): s<<3 | ((E>>1) - hb).
* Weights outside the window are stored as the escape code (H=0, L=0 -> +2^(2hb-127)) and listed per CTA
  as (n_local, k, delta = true - escape) in a side table; the kernel adds x[k]*delta to the accumulator.
  Counts: qkv 2601, o 1628, pooler 4650, unpooler 3307, slice 417 (seed 1234).
* 12 bits/weight: qkv 28.3 MB, o 12.6 MB, pooler 50.3 MB, unpooler 50.3 MB, slice 6.3 MB.
* Router (fp32): bf16 hi + lo pair as SGLang does, each 12-bit packed with its own window: 6.3 MB
  (vs 8.4 MB fp32 / hi+lo bf16). Exceptions of both planes share one table (4166).

Packed order is CTA-major, then stage (8 k64 blocks, one per warp), then block, then row, then lane;
inside a block the L plane is [n][64 B] and the H plane [n][32 B], so each lane's 16 B + 8 B smem reads are
conflict-free, and k is permuted so that a lane's 8 consecutive weights are the 8 consecutive x values it
needs (k = 32h + 8c + i for lane%4 = c), which makes the x fragment loads full 16 B sectors from L2.

## Kernel

* grid = N/NB CTAs of 256 threads (NB per layer: qkv 18 -> 128 CTAs with a masked third n-tile; o 64;
  pooler 32; unpooler 64; slice 8; router 2), one wave, no cluster.
* Thread 0 inits S mbarriers and issues S `cp.async.bulk` 1-D copies (one contiguous stage each) at
  launch; each warp waits on the stage's mbarrier, consumes its own k64 block of the stage, and the
  CTA re-arms the buffer with the stage S further on. Stage bytes = 8*NB*96 (router: *192), S chosen
  so S*stage <= 96 KB (router/slice/o: the whole CTA stream is in flight from the first instruction).
* Decode (per 4 weights, 7 ALU ops): prmt LUT for the exponent byte, prmt sign-replicate trick for the
  sign, two prmt to interleave with the L bytes -> two bf16x2 B-fragment registers.
* mma.sync m16n8k16 bf16, fp32 accumulate; x A-fragments are read directly from global (L2) one stage
  ahead; rows >= m are zero, lanes beyond NB carry zero B.
* The 8 warps split K; the epilogue reduces their partials once through smem and writes out[m][NB] once.

## Measured (H100 box, seed 1234)

Correctness: all 20 (layer, m in 1/8/16/32) checks pass, probe rows exact to 0 (router 2.9e-6).
--official (m = 5, 7, 9, 33, 130, 512, 1000) not yet run: m > 32 is the torch.mm / hi+lo fallback.

Kernel-only durations (CUPTI, graph replay after an L2 flush; the box's graph replay overhead was 12.6 us
during these runs, so harness totals are inflated by ~8 us vs a quiet box), TMA 1-D bulk loader:

| layer | packed MB | m=1 | m=8 | m=16 | m=32 | torch (cuBLAS, bf16) |
|---|---|---|---|---|---|---|
| qkv_gate | 28.3 | 18.8 | 21.8 | 23.1 | 26.0 | 14.5-15.0 |
| o | 12.6 | 9.1 | 9.6 | 9.6 | 11.4 | 7.7-7.9 |
| pooler | 50.3 | 25.5 | 27.6 | 29.2 | 33.8 | 23.8-24.0 |
| unpooler (slice m<=8) | 6.3 / 50.3 | 9.3 | 10.5 | 24.5 | 27.2 | 5.5 / 23.7-24.2 |
| router (hi+lo 12-bit) | 6.3 | 15.9 | 18.7 | 20.5 | 23.4 | - |

Finding: the kernel is nowhere near the byte count. Time scales with the number of bulk copies per CTA
(router: 16 copies of 3 KB -> 16 us; slice: 8 x 6 KB -> 9.3 us; o: 2 x 48 KB -> 9.1 us), i.e. the 1-D
`cp.async.bulk` copies of one CTA are effectively serialized at ~1 us latency + ~16 GB/s per SM, far below
the ~23 GB/s per SM that 128 CTAs need to reach the HBM rate, and crippling for the small layers.
The compute side is fine (m=32 costs only +2-4 us over m=1 with 8 warps on mma.sync).

## What I would do next (queued on the box when this report was due)

1. `mb_load.py`: microbenchmark of TMA-1D (copy size x depth x split) vs plain `ld.global.v4` streams
   (4/8/16 loads in flight per thread) vs 16 B `cp.async` rings, 128 CTAs x 48/96/384 KB after an L2 flush,
   to pick the loader that reaches ~2.9 TB/s on the H100 with everything issued at launch.
2. `LATENCY_LOADER=1` build (already compiled): the same kernel with all 256 threads issuing 16 B cp.async
   into the same smem ring, completion via `cp.async.mbarrier.arrive.noinc`, so the stage depth and the
   consumer side are unchanged. Profile run queued (`logs/latency/prof_l1.log`).
3. If plain loads win: move the L/H planes straight into registers per warp (6 regs per n-tile per k64)
   with 2-3 k64 blocks in flight per warp, no smem at all for the weights.
4. Then the m=32 L2 question (x traffic = 2m/(1.5 NB) of the weight traffic): NB=64 variants or x via smem.
5. Run --official, then the full harness table, then tune S / NB per layer on a quiet box.
