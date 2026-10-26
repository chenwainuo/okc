# DeepGEMM sm90 bf16 GEMM: pipeline structure

Source: DeepGEMM at commit 5081f85 (2026-10-20), clone at
`/tmp/claude-1000/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/scratchpad/ref/DeepGEMM`.
Paths below are relative to that clone. Abbreviations:

| tag | file |
|---|---|
| K | `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` (399 lines, the whole kernel) |
| H | `csrc/jit_kernels/impls/sm90_bf16_gemm.hpp` (JIT instantiation + launch) |
| S | `csrc/jit_kernels/heuristics/sm90.hpp` (tile/stage/cluster selection) |
| SCH | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` |
| MMA | `deep_gemm/include/deep_gemm/mma/sm90.cuh` (wgmma selector, smem descriptors) |
| TMA | `deep_gemm/include/deep_gemm/common/tma_copy.cuh` |
| WG | `deep_gemm/include/deep_gemm/ptx/wgmma.cuh` |
| RU | `csrc/jit_kernels/impls/runtime_utils.hpp` (cuTensorMapEncodeTiled wrappers) |
| API | `csrc/apis/gemm.hpp` |

CUTLASS primitives referenced (`ClusterTransactionBarrier`, `SM90_TMA_LOAD*`, `warpgroup_reg_*`) were read in CUTLASS 3.9.2 headers; the kernel pins a different CUTLASS submodule (f3fde58) but these primitives are unchanged PTX wrappers.

## 0. Summary (what matters for the m=128/256 problem)

1. **It is a plain TMA-ring + SS-wgmma warp-specialised kernel, one wgmma group in flight.** Producer = 1 elected thread in a 128-thread TMA warpgroup (setmaxnreg 48); consumers = 1 or 2 math warpgroups (setmaxnreg 248 / 224). Per stage the math warpgroup does `wgmma.fence -> BLOCK_K/16 wgmmas -> commit -> wait_group 0 -> arrive(empty)`. Nothing is pipelined across stages on the consumer side; all latency hiding is the TMA ring depth (K:254-281).
2. **Stage merging.** When the ring has >= 10 stages (K-major NT, one math warpgroup), stages are merged so BLOCK_K becomes 128 or 192 with >= 5 stages; this halves/thirds the number of fence/commit/wait rounds per K (K:45-56). For pooler m=128 the heuristic's 64x64x64, 13 stages becomes **6 stages of 64x64x128 (32 KB/stage, 192 KB in flight), 8 wgmma m64n64k16 per group**.
3. **No swap-AB, no register-A, no split-K, no stream-K on SM90.** `swap_ab` is asserted 0 (S:125); bf16 uses only `*_SS` wgmma (MMA:104-141); the scheduler is persistent round-robin over MxN tiles only (SCH:185-272). Skinny M is handled by BLOCK_M in {16, 32} with the wgmma M=64 reading past the tile (K:76-78, S:25-26). Swap-AB exists only in the SM100 kernels.
4. **Epilogue:** registers -> `stmatrix` into a single swizzled smem_d buffer -> `cp.async.bulk.tensor` store per 64-column slab, `commit_group`, no wait; the store drains during the next tile's mainloop and is waited (`wait_group.read 0`) at the next epilogue (K:297-388). Math warps stall during their own epilogue; the producer keeps prefetching the next tile's stages meanwhile.
5. **Multicast:** cluster <= 2, exactly one operand multicast (A if cluster_n > 1, else B), issued only by cluster rank 0 with mask 0b11; empty barriers count arrivals from both CTAs' math warps (K:116, 161-166, 238-245; TMA:78-85). The host cost model **disables clusters whenever the grid is a single wave** (S:235-236), which is every one of our shapes.
6. **`bf16_gemm_nt` routes to cuBLASLt by default.** The sm90 kernel only runs when `num_sms` was lowered below the device count or deterministic mode is on (API:514-521, `runtime.hpp:89-91`). DeepSeek ships this kernel but does not use it for bf16 on a full GPU; `tests/test_bf16.py:56-58` benches it against `nvjet`.
7. For our shapes the heuristic (emulated, section 11) chooses 1 CTA/SM, 128 tiles for pooler/unpooler/o and only 96 (BN=48) or 116 tiles for qkv, no cluster. The resulting L2->SM traffic is the cuBLAS-like 2x W + (N/BN) x x.

## 1. Host side: dispatch, config, descriptors, launch

| item | value | ref |
|---|---|---|
| API | `bf16_gemm_nt(a[M,K], b[N,K], d[M,N])`; `_nn/_tn/_tt` aliases pass MN-major operands | API:489-531 |
| cuBLASLt bypass | `if (!deterministic && is_cublaslt_available() && alpha.has_value()) cublaslt_gemm(...)`; identity epilogue has alpha=1.0; `is_cublaslt_available() == (num_sms == device SMs)` | API:514-518, `epilogue_class.hpp:49`, `runtime.hpp:89-91` |
| Template params | majors, compiled M/N/K (0 = runtime), BLOCK_M/N/K, swizzle A/B/D, stages, TMA/math thread counts, multicast count + on-A flag, num_sms, GemmType, accumulate flag, D dtype | K:27-37, H:33-64 |
| compiled dims | `compiled_dims` chars among "mnk" bake those dims as constants (default from Python: "nk") | RU:31-40 |
| Grid / block / cluster | grid = `num_sms` (132, must be even), block = 128 TMA + 128 or 256 math threads, cluster = cluster_m*cluster_n in {1, 2}; `__launch_bounds__(threads, 1)` | H:119-124, S:190-200, K:38, `config.hpp:60` |
| Tensor maps | 2D, inner = K (K-major), box = {64, BLOCK_M or BLOCK_N} elements; `smem_inner_dim` forced to `swizzle/elem = 64`; SWIZZLE_128B; L2_PROMOTION_256B; OOB_FILL_NONE (zeros); INTERLEAVE_NONE | H:99-113, RU:119-158 |
| D tensor map | box = {swizzle_cd/2 = 64 cols, BLOCK_M rows}, SWIZZLE_128B for bf16 D (swizzle 0 for fp32 D) | H:109-113, S:137-139 |
| PDL | `cudaGridDependencySynchronize()` before the scheduler starts | K:131 |

### 1.1 Config heuristic (S:16-118, 148-239)

| knob | rule | ref |
|---|---|---|
| BLOCK_M candidates | {64, 128}; +16 if m <= 16; +32 if m <= 32; +256 if D is bf16 | S:23-30 |
| BLOCK_N candidates | 16..256 step 16 (`block_n_multiple_of` default 1) | S:40-57 |
| BLOCK_K | fixed 128 B of K = **64 bf16** | S:60 |
| reject | BLOCK_M > 128 and BLOCK_N > 128 (registers); swizzle A/B must be 64 or 128 B | S:94-95, 102-103 |
| cluster | (1,1), (2,1), (1,2); multicast on A iff cluster_n > 1 | S:71-79, H:61 |
| stages | `min(16, (232448 - smem_cd - 256) / (BLOCK_M*BLOCK_K*2 + BLOCK_N*BLOCK_K*2))`; need >= 3, or >= 4 when BLOCK_M*BLOCK_N < 128*192 | S:149-188, 105-108 |
| math threads | 128 if BLOCK_M <= 64 else 256; TMA threads always 128 | S:190-200 |
| cost model | per-tile L2 bytes `K*(BM/cluster_n + BN/cluster_m)*2 + D`, L1 bytes `K*(BM+BN)*2 + K*(max(64,BM)+BN)*2 + D`; L2 bw `min(64*SMs, 6154)` B/clk, L1 bw `128*SMs` B/clk; cycles = max(L1,L2)/wave_efficiency; **cluster candidates get infinite cost if num_waves <= 1**; pick min cycles | S:202-243 |

## 2. Warp roles and registers

| role | warps (math=128 / 256 threads) | setmaxnreg | work | ref |
|---|---|---|---|---|
| TMA warpgroup | warps 4-7 / 8-11 | `setmaxnreg.dec 48` | one elected lane of warp `math/32 + 2` (warp 6 / 10) runs the whole producer loop; the rest of the warpgroup idles after dealloc | K:127, 147-153 |
| barrier init | one lane of warp `math/32 + 1` | | `full.init(1)`, `empty.init(cluster * math_warps)`, `fence.mbarrier_init.release.cluster` | K:112-121 |
| descriptor prefetch | one lane of warp `math/32` | | `prefetch.tensormap` for A, B, D | K:85-90 |
| math warpgroup(s) | warps 0-3 (+4-7) | `setmaxnreg.inc 248` (1 WG) / `224` (2 WGs) | each WG owns 64 rows of the tile (`a_desc` offset `math_wg_idx * 64`); with BLOCK_M=256 each WG owns rows {wg*64, 128+wg*64} (two "waves") | K:128, 212-230 |
| sync after init | `cluster_sync` (relaxed arrive) if cluster > 1 else `__syncthreads` | | | K:124 |

Register budget: 128x48 + 128x248 = 37,888 (1 WG) or 128x48 + 256x224 = 63,488 (2 WGs) of 65,536; 1 CTA/SM by construction (smem ~210-231 KB).

## 3. Shared memory layout

```
[ smem_d : align1024(BLOCK_M*BLOCK_N*sizeof(D)) ]
[ A stage 0 .. kNumStages-1 : BLOCK_M*BLOCK_K*2 each ]
[ B stage 0 .. kNumStages-1 : BLOCK_N*BLOCK_K*2 each ]
[ full[kNumStages] | empty[kNumStages] : 8 B each ]        (host reserves 16*8*2 = 256 B)
```
K:72-109; `extern __shared__ __align__(1024)`, every region a multiple of 1024 B (K:93-95).

**Stage merge (K:45-56):** if `kNumStages_ >= 10 && Normal && A,B K-major && math threads == 128`, then `per_merge = kNumStages_ / 5`, `BLOCK_K = 64 * per_merge`, `kNumStages = kNumStages_ / per_merge`. The merged stage is `per_merge` consecutive 64-wide K atoms laid out back to back (A atoms then B atoms per stage); descriptors step `atom_k_idx * BLOCK_M * 64` elements between atoms (K:218, 266-270; MMA:239-241). Allocated smem is still the unmerged 13 (or 15) x stage bytes, so up to one unmerged stage is wasted.

## 4. TMA loads (K:185-200, TMA:39-104)

| aspect | value |
|---|---|
| instruction | `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`, hint EVICT_NORMAL for both operands (TMA:43) |
| box | inner 64 elements (128 B, one SW128 atom), outer BLOCK_M (A) / BLOCK_N (B); for BLOCK_K > 64 the producer issues `BLOCK_K/64` copies per operand per stage into consecutive `BLOCK_OUTER*64`-element slabs (TMA:62-66) |
| coordinates | A at (k, m_block*BLOCK_M), B at (k, n_block*BLOCK_N); grouped variants add offsets through `scheduler.get_global_idx` |
| expect_tx | one `mbarrier.arrive.expect_tx` per stage with `SMEM_A + SMEM_B` bytes, issued after the copies (K:200) |
| swizzle | 128 B on A and B (K-major, 128 B rows); matches gmma LayoutType B128, SBO = 1024 B, LBO = 0 (MMA:249-259) |
| multicast | only when `scheduler.is_tma_multicast_valid()` for this tile (K:163-165; SCH:275-292). Rank 0 issues `...multicast::cluster` with mask `(1<<2)-1`; rank 1 issues nothing for that operand (TMA:78-85). Both CTAs still `expect_tx(A+B)` on their own full barrier; the multicast completes tx on both. Which operand: A if `kIsTMAMulticastOnA` (= cluster_n > 1), else B (K:164-165). |
| multicast drain | after the last tile the producer waits every empty barrier once more so the peer cannot arrive on a freed barrier (K:204-208) |

## 5. mbarrier protocol (K:106-124, 170-171, 237-245, 254, 280-281)

| barrier | init count | arrive | wait | parity |
|---|---|---|---|---|
| `full[s]` | 1 | producer thread: `arrive.expect_tx(bytes)`; TMA engine completes the bytes | every math thread: `mbarrier.try_wait.parity` spin (`ClusterBarrier::wait`) on `phase` | |
| `empty[s]` | `kNumTMAMulticast * kNumMathThreads/32` (4, 8, or 8/16 with cluster) | lane 0 of each math warp `mbarrier.arrive.shared::cta`; with cluster, lanes 0..1 each arrive on CTA `lane_idx`'s barrier via `mapa` + `mbarrier.arrive.shared::cluster` (if the peer has no tile, both arrivals go local) | producer thread waits with `phase ^ 1` (passes trivially on the first lap) | |
| phase | one bit shared by both loops; `stage_idx` wraps mod kNumStages and `phase ^= (stage_idx == 0)`; the counters are **not reset between tiles**, so the ring runs continuously across the persistent loop | | | K:138-145 |

The producer therefore runs up to `kNumStages` stages ahead, including across tile boundaries (it fetches the next tile's first stages while the consumers are in the epilogue).

## 6. wgmma issue (K:211-282, MMA:77-148, WG:1-25)

| aspect | value |
|---|---|
| instruction | `wgmma.mma_async.sync.aligned.m64n{BLOCK_N}k16.f32.bf16.bf16` **SS** (both operands via smem descriptors), `scale_d = 1` always (K:271), accumulators zeroed in registers per tile (K:230) |
| N | BLOCK_N, any multiple of 8 up to 256 (MMA:109-140) |
| per stage | `(BLOCK_K/16) * (BLOCK_M / WAVE_BLOCK_M)` wgmmas, where WAVE_BLOCK_M = BLOCK_M if <= 64 else 128; e.g. 64x64x128: 8; 64x128x64: 4; 128x128x64: 4 per warpgroup; 256xN: 2 waves x 4 |
| descriptors | built once per tile for stage 0 (`make_gmma_desc`, K:219-222) and broadcast with `__shfl_sync`; per stage add `stage_bytes/16` to the low word; per k16 add 32 B (+2) within the atom; per 64-wide atom add `BLOCK_MN*64*2/16` (K:250-251, 266-270; MMA:239-241) |
| sequence per stage | `fence_operand(accum)` x all; `wgmma.fence` (K:260); issue all wgmmas; `wgmma.commit_group` (K:274); `fence_operand(accum)`; **`wgmma.wait_group 0`** (K:278); `empty[s].arrive` (K:281) |
| groups in flight | **1**; the wait sits immediately after the commit of the same stage. No wgmma overlaps the next stage's `full` wait or descriptor math. The fp8 1d2d kernel (`sm90_fp8_gemm_1d2d.cuh:311-323`) does the same `wait<0>` per stage, followed by the scale promotion FMAs. |
| 2 warpgroups | each WG independently waits/arrives; no inter-WG sync in the mainloop (BLOCK_M >= 128 only; `kNumMathThreads == 256`) |

Smem traffic implied for 64x64x128 (pooler m=128 config): per stage 32 KB TMA writes + 32 KB wgmma reads (A 16 KB + B 16 KB), i.e. the same 1:1 ratio as the 64x64x64 config, with half as many fence/commit/wait rounds.

## 7. Slot release

Empty arrive happens **after `wait_group 0`** of the stage's own group (K:278-281), i.e. a slot is held for TMA latency + full-wait + issue + full MMA drain. There is no early release and no "release stage s-1 after wait_group 1". With 6-16 stages the producer still has >= 5 stages of lookahead per Little's-law budget. This is the part our pw10 ladder identified as the exposed-wgmma cost at 4 stages; DeepGEMM compensates only with depth (192-216 KB in flight) and the merged BLOCK_K.

## 8. Epilogue (K:284-388)

| step | detail | ref |
|---|---|---|
| who | `do_wgmma_store` threads: all 128/256 math threads when BLOCK_M >= 64; only `WAVE_BLOCK_M*2` threads (1-2 warps) for BLOCK_M = 16/32 | K:233-235, 294-295 |
| wait previous store | threads that issued the previous tile's TMA stores run `cp.async.bulk.wait_group.read 0`, then named barrier over the store threads | K:297-300 |
| bf16 D | `cvt` pairs to bf16x2, `stmatrix.sync.aligned.x2.m8n8` into `smem_d` laid out as `[BLOCK_N/64 slabs][BLOCK_M rows][64 cols]` with the SW128 XOR applied by hand (`col ^= row % 8`) | K:302-349 |
| fp32 D | plain `st.shared` of float2, unswizzled | K:350-364 |
| fence | `fence.proxy.async.shared::cta` then named barrier | K:365-366 |
| store | thread `t < BLOCK_N/64` issues `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` for slab t (or `cp.reduce...add` when accumulating into C == D), then `cp.async.bulk.commit_group`; **no wait** before the next tile | K:368-388 |
| overlap | the store drains during the next tile's mainloop; the math warps are idle for the stmatrix + fence + issue (a few hundred cycles). The producer is already filling the ring for the next tile. Single `smem_d` buffer. | |
| accumulate (C given) | host copies C into D first when they differ (API:49-51); the kernel then uses `cp.reduce.async.bulk.tensor...add` into D instead of a plain store | K:374-385, API:49-51 |

## 9. Tile scheduler (SCH:14-26, 108-144, 185-272)

| aspect | value |
|---|---|
| persistence | grid = num_sms CTAs; CTA b processes linear tile ids `b, b+132, b+264, ...` (`next_block_idx = (++iter) * kNumSMs + blockIdx.x`, SCH:186). Both producer and consumers run identical scheduler copies. |
| rasterisation | tiles are grouped along the "primary" dim in groups of `kNum1DBlocksPerGroup` in {8, 16}, chosen to minimise `g*BLOCK_N + ceil(132/g)*BLOCK_M` (multicast on A: primary = N) or `g*BLOCK_M + ceil(132/g)*BLOCK_N` (otherwise: primary = M) (SCH:14-26). Within a group, consecutive ids walk the primary dim fastest, then the secondary. So one wave of 132 CTAs covers `g` primary blocks x `132/g` secondary blocks, bounding the live A+B footprint in L2 (SCH:108-144). |
| cluster pairing | tile ids `2j, 2j+1` land on the two CTAs of a cluster and share the secondary-dim block, so the multicast operand is the shared one; odd group tails are fixed up (`num_blocks_in_group ^ 1`) so a pair never straddles groups (SCH:120-133); `is_peer_cta_alive` handles the last odd tile (SCH:266-268). |
| K | never split; one CTA walks the full K for its tile (K:168-169, 248-249). |
| grouped variants | masked / contiguous / k-grouped / batched offsets through `get_global_idx` (SCH:146-174); irrelevant here. |

## 10. Skinny shapes, operand swap

- **No swap-AB on SM90**: `DG_HOST_ASSERT(layout.swap_ab == 0)` (S:125); the kernel has no `kSwapAB` parameter. SM100 kernels implement it (`sm100_bf16_gemm.cuh:64, UMMA_N = BLOCK_M`, `epilogue/sm100_store_cd_swap_ab.cuh`).
- Small M (<= 32): BLOCK_M in {16, 32} (S:25-26) with one math warpgroup; the wgmma still reads 64 A rows from smem (the static assert at K:76-78 only guarantees the over-read stays inside the allocation); only the first `BLOCK_M/8` warps' results are stored (K:233-235). So MMA rows are wasted rather than operands swapped.
- M = 128/256 tokens is not "skinny" in DeepGEMM's orientation: A = x (M = tokens), B = W (N = W rows, BLOCK_N <= 256). The heuristic then picks BLOCK_M = 64 tokens and BLOCK_N = 48..128 W rows (section 11), i.e. `m64n{48,64,128}k16` with W as the N operand, the opposite of our pw9 swap (W rows as M, tokens as N).
- The Python API applies no swap either (`deep_gemm/__init__.py:59`, `tests/test_bf16.py:27-58` only transposes to exercise majors).

## 11. What the heuristic selects for our shapes (emulated from S:16-243, 132 SMs, bf16 D, no accumulation)

Emulator: `/tmp/claude-1000/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/scratchpad/dg/heur.py` (reproduces S:148-188 smem/stage arithmetic exactly, e.g. 64x64x64 -> 13 stages, 221,440 B). Ties are broken by enumeration order (cluster_m, cluster_n, BLOCK_M asc, BLOCK_N asc), as `compare` is strict (S:242-244, `config.hpp:273-276`).

| layer (N, K) | m | BM x BN x BK (host) | stages (host) | merged BLOCK_K x stages (kernel) | bytes/stage, in flight | wgmma/group | tiles on 132 SMs | math WGs |
|---|---|---|---|---|---|---|---|---|
| qkv_gate (2304, 8192) | 128 | 64x48x64 | 15 | 192 x 5 | 42 KB, 210 KB | 12 x m64n48k16 | 96 (36 SMs idle) | 1 |
| qkv_gate | 256 | 64x80x64 | 12 | 128 x 6 | 36 KB, 216 KB | 8 x m64n80k16 | 116 | 1 |
| o (8192, 1024) | 128 | 64x128x64 | 8 | 64 x 8 | 24 KB, 192 KB | 4 x m64n128k16 | 128 | 1 |
| o | 256 | 128x128x64 | 6 | 64 x 6 | 32 KB, 192 KB | 4 x m64n128k16 per WG | 128 | 2 |
| pooler (4096, 8192) | 128 | 64x64x64 | 13 | 128 x 6 | 32 KB, 192 KB | 8 x m64n64k16 | 128 | 1 |
| pooler | 256 | 64x128x64 | 8 | 64 x 8 | 24 KB, 192 KB | 4 x m64n128k16 | 128 | 1 |
| unpooler (8192, 4096) | 128 | 64x128x64 | 8 | 64 x 8 | 24 KB, 192 KB | 4 x m64n128k16 | 128 | 1 |
| unpooler | 256 | 128x128x64 | 6 | 64 x 6 | 32 KB, 192 KB | 4 x m64n128k16 per WG | 128 | 2 |

Cluster is (1,1) for all eight: every candidate with a cluster is a 1-wave grid and gets infinite cost (S:235-236). Traffic for pooler m=128 under this config: W read twice from L2 (2 token blocks) = 134 MB, x read 64 times = 128 MB, total L2->SM ~262 MB, HBM 67 MB; identical geometry to the nvjet config recorded in `audits/m128-study.md` (64x64, 13 stages) except DeepGEMM merges to 6 x 128-K stages and has no cluster.

## 12. Structural comparison with our pipelines (facts from the code; conclusions are inferences)

| point | DeepGEMM | our pw9/10/13/19/24 | note |
|---|---|---|---|
| ring depth | 192-216 KB in flight, 5-8 (merged) stages | 4-8 stages of 22 KB = 88-177 KB (pw10), 12 x 14 KB (pw24) | DeepGEMM's depth is at the smem ceiling; the host computes stages from smem, never from a fixed number |
| slot release | after `wait_group 0` of the same stage | pw10: after `wait_group 1`; pw7: after decode | DeepGEMM is *later* than ours and compensates with depth; so release timing alone is not what separates us from cuBLAS |
| wgmma groups in flight | 1 | 1-2 | same |
| wgmma per sync round | 8 (merged BLOCK_K=128) or 12 (192) | 4 (BLOCK_K = 64) | fewer fence/commit/wait rounds per K; a cheap thing to copy |
| operands | SS, W as the N operand, x as M | pw9/10: register-A W, x as N (SS B); pw19/24: SS both | DeepGEMM's orientation makes the x tile the 64-row A operand and W the wide B operand |
| split-K / cluster reduce | none | cluster DSMEM / last-arriver | DeepGEMM never reduces across CTAs; it accepts 96-128 tiles in one wave |
| multicast | off for 1-wave grids by the cost model | pw13/pw24 multicast lost 5-10 us | consistent with our measurements |
| epilogue | stmatrix + TMA store, unwaited, drains under the next tile | pw5/pw10 cluster or last-arriver epilogues 4-8 us | single-tile-per-CTA here, so nothing overlaps the store at the end of the kernel |
| producer | one thread, polls `empty` with try_wait spin | same | same |
| register split | 48 / 248 (1 WG) or 48 / 224 (2 WG) | pw lineage not using setmaxnreg in all variants | |
| bf16 on a full GPU | **routes to cuBLASLt** (API:514-518) | n/a | DeepSeek's own default agrees with our finding that cuBLAS wins this regime |

Experiments this suggests for the H100 (none run here): (a) build DeepGEMM's sm90 bf16 kernel standalone for the eight shapes (force it with `set_num_sms(130)` or deterministic mode, or instantiate `sm90_bf16_gemm_impl` directly from the JIT string at H:27-47) and ladder it against cuBLAS; (b) the same kernel with `kDoMergeStages` forced off to measure the value of 8 vs 4 wgmmas per round; (c) BLOCK_N = 128/256 variants (fewer tiles, wider wgmma) to test the smem-port hypothesis from `audits/m128-study.md`.
