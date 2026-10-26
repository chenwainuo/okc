# DeepGEMM sm90 bf16 GEMM: host/JIT side, what it would pick for our shapes, and embedding it in one submission.py

Source read: DeepGEMM at commit 5081f85 (2026-10-19), `csrc/jit_kernels/impls/sm90_bf16_gemm.hpp`, `csrc/jit_kernels/heuristics/{sm90,common,config,utils,runtime}.hpp`, `csrc/jit_kernels/impls/runtime_utils.hpp`, `csrc/runtime/jit.hpp`, `csrc/apis/gemm.hpp`, `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` and its 15 device headers; DeepJIT (submodule 2efdab4, cloned separately) `include/deep_jit/backend/cuda/{options,backend,kernel}.hpp`. The config heuristic was replicated in Python (`scratchpad/dg_host/heur.py`) and the picked instantiations were compiled locally with nvcc 12.8 against CUTLASS 3.9.2 (`scratchpad/dg_host/`). No GPU here; nothing below is a measured kernel time.

## 0. Two facts that frame everything

- DeepGEMM's own `bf16_gemm_nt` goes to **cuBLASLt** unless `use_deterministic_algorithms(True)` is set (`csrc/apis/gemm.hpp:519-522`: `if (not deterministic and runtime->is_cublaslt_available() and alpha) { cublaslt_gemm(...); return; }`). The sm90 bf16 kernel is DeepSeek's deterministic fallback, not their fast path for bf16. `tests/test_bf16.py` prints its time as a ratio to nvjet + the split-K reduce.
- For every one of our 8 shapes the heuristic lands on **one wave with no cluster** (cluster candidates are discarded when `num_waves <= 1`, `sm90.hpp` `get_layout_info`: `if (cluster > 1 and num_waves <= 1) num_cycles = INT64_MAX`). So DeepGEMM never multicasts on our problem; that matches pw24's finding that cluster-2 W multicast lost 9 us.

## 1. Configuration DeepGEMM would pick

### Orientation
Use the natural orientation, not swap-AB: `A = x [M = tokens][K]` K-major, `B = W [N = rows][K]` K-major, `D = out [M = tokens][N = rows]` row-major. That is exactly `deep_gemm.bf16_gemm_nt(x, W, out)` and the judge's output layout; `BLOCK_M = 64` splits 128 tokens into two M-blocks, so M = tokens is not a problem (the heuristic even allows `BLOCK_M` 16/32 for m <= 32). Swap-AB (A = W) would produce `D[rows][tokens]` (the kernel only stores N-major D, `check_major_type_cd`), i.e. a transposed output, and it picks the same tiles anyway (table below). `compiled_dims` default is `"nk"` (`gemm.hpp:842`): N and K are template constants, M is a runtime argument.

### Candidate space (`SM90ArchSpec::get_layout_candidates`)
- `block_m in {64, 128, 256}` (+16 if m <= 16, +32 if m <= 32; 256 only for bf16 D); `block_n = 16..256 step 16`; `block_k = 128 * 8 / 16 = 64` always; `cluster (m, n) in {(1,1), (2,1), (1,2)}`.
- Dropped: `block_m > 128 and block_n > 128`; swizzle A/B not a multiple of 64 B (never, since block_k * 2 = 128 B); `num_stages < 3`, or `< 4` when `block_m * block_n < 128 * 192`.
- Stages: `smem_cd = align(bm * bn * 2, 1024)`, `smem_barriers = 16 * 8 * 2 = 256`, `per_stage = (bm + bn) * 64 * 2`, `num_stages = min((232448 - smem_cd - 256) / per_stage, 16)`. Always fills smem to the 227 KB cap.
- Cost model (`get_layout_info`, the only thing compared): `num_blocks = ceil(M/bm) * ceil(N/bn)`; `l2_bw = min(64 * 132, 8e6/1.3e3) = 6153 B/clk`, `l1_bw = 128 * 132 B/clk`; `l2_bytes = K * (bm/cluster_n + bn/cluster_m) * 2 + bm*bn*2`, `l1_bytes = K*(bm+bn)*2 + K*(max(64,bm)+bn)*2 + 2*bm*bn*2`; `cycles = max(l1_bytes*blocks/l1_bw, l2_bytes*blocks/l2_bw) / wave_efficiency`, `wave_efficiency = blocks / (waves * 132)`. HBM and tensor time are not modelled ("constant across configs"). The first candidate with the minimal cycles wins (strict `<`, enumeration order cluster_m, cluster_n, block_m list order, block_n ascending).
- Launch: `grid = num_sms = 132` persistent CTAs (`Scheduler::get_next_block`: `block = iter * 132 + blockIdx.x`), `block = 128 TMA threads + (128 if bm <= 64 else 256) math threads`, `cluster = 1`.
- Kernel-side post-processing (`sm90_bf16_gemm.cuh:48-55`): if `num_stages >= 10`, Normal GEMM, both K-major, one math warpgroup, stages are **merged**: `per_merge = num_stages / 5`, `BLOCK_K = 64 * per_merge`, `num_stages = num_stages / per_merge`. One `warpgroup_wait<0>` then covers 8 (BLOCK_K 128) or 12 (BLOCK_K 192) wgmmas instead of 4.

### Picks for our shapes (natural orientation; replicated heuristic, 132 SMs, bf16 D)

| layer | tokens | M x N x K | pick bm x bn, cluster | stages (host) | kernel after merge | smem bytes | threads | blocks | regs (ptxas) |
|---|---|---|---|---|---|---|---|---|---|
| pooler | 128 | 128 x 4096 x 8192 | 64 x 64, 1x1 | 13 | BLOCK_K 128, 6 stages x 32 KB | 221,440 | 256 | 128 | 248 |
| unpooler | 128 | 128 x 8192 x 4096 | 64 x 128, 1x1 | 8 | BLOCK_K 64, 8 x 24 KB | 213,248 | 256 | 128 | 248 |
| o | 128 | 128 x 8192 x 1024 | 64 x 128, 1x1 | 8 | BLOCK_K 64, 8 x 24 KB | 213,248 | 256 | 128 | 248 |
| qkv_gate | 128 | 128 x 2304 x 8192 | 64 x 48, 1x1 | 15 | BLOCK_K 192, 5 x 42 KB | 221,440 | 256 | 96 | 248 |
| pooler | 256 | 256 x 4096 x 8192 | 64 x 128, 1x1 | 8 | BLOCK_K 64, 8 x 24 KB | 213,248 | 256 | 128 | 248 |
| unpooler | 256 | 256 x 8192 x 4096 | 128 x 128, 1x1 | 6 | BLOCK_K 64, 6 x 32 KB | 229,632 | 384 | 128 | 168 (launch-bounds) |
| o | 256 | 256 x 8192 x 1024 | 128 x 128, 1x1 | 6 | BLOCK_K 64, 6 x 32 KB | 229,632 | 384 | 128 | 168 |
| qkv_gate | 256 | 256 x 2304 x 8192 | 64 x 80, 1x1 | 12 | BLOCK_K 128, 6 x 36 KB | 231,680 | 256 | 116 | 248 |

Swap-AB gives identical picks except qkv at 256 tokens (64 x 96, 10 stages). Runner-ups for pooler m=128: (64x32, cluster 1x2, 16 stages, 256 blocks = 2 waves) 49,279 cycles; (64x80, 12 stages) 50,832; (128x32, 10 stages) 56,412; the winner is 45,165.

Why: with 128 tokens the model's L2 term is `K * (bm + bn) * 2 * blocks`, which for fixed `bm * bn` tile area is minimised by square tiles, and the wave-efficiency divisor punishes anything that leaves blocks < 132 (qkv's 2304 rows: 48-wide tiles give 96 blocks, 64-wide only 72). Nothing in the model sees HBM, so it is indifferent to W being streamed once (it is: each W row-tile is read by the 2 (m=128) or 4 (m=256) token tiles, i.e. W leaves L2 twice/four times, 134 MB L2->SM for pooler m=128 plus 128 MB of x re-reads = the 262 MB cuBLAS number from audits/m128-study.md).

Pooler m=128 is cuBLAS nvjet's geometry (64x64, 13 stages, 128 CTAs, 2 token tiles per W tile) minus nvjet's 1x2 cluster. It is also pw24's `bg12` geometry (64 rows x 64 tokens, 12 stages, no cluster: 28.9 in-prepare vs cuBLAS 24.2) with these differences: BLOCK_K 128 merged stages (one fence/commit/wait/arrive per 32 KB instead of 16 KB), `warpgroup_wait<0>` after every batch (no wgmma overlap across stages; the slot is released only after the batch retires, `sm90_bf16_gemm.cuh:273-282`), a 132-CTA persistent grid (4 idle), producer = one elected lane of warp 2 of the TMA warpgroup (`warp_idx == kNumMathThreads/32 + 2`), `setmaxnreg` 48/248, TMA loads with `L2_PROMOTION_L2_256B` and EVICT_NORMAL hint, D written by `stmatrix` into a 128B-swizzled smem tile and one `cp.async.bulk.tensor` store per 64 columns. The SASS of the pooler-128 instantiation has 16 `HGMMA.64x64x16.F32.BF16`, 16 `UTMALDG.2D`, 1 `UTMASTG.2D`, 2 `USETMAXREG`, 0 `CALL`, and ptxas prints no "wgmma serialized" warning.

### Register/smem detail per k-stage, pooler m=128
8 x m64n64k16 SS per 128-k stage = 256 tensor clk; smem traffic per stage = 32 KB TMA fill + 32 KB wgmma reads (A 16 KB + B 16 KB, read once per k16 by one warpgroup) = 64 KB = 512 clk at 128 B/clk. That is the same smem-port arithmetic audits/m128-study.md used to explain pw24's +11 us wgmma rung; DeepGEMM does nothing special about it. Whether that rung shows up here is the first thing to measure on the H100.

## 2. What the kernel needs at runtime

Kernel signature (`sm90_bf16_gemm.cuh:40-45`):
`sm90_bf16_gemm_impl<...>(int* grouped_layout, uint32_t m, uint32_t n, uint32_t k, const __grid_constant__ CUtensorMap a, const __grid_constant__ CUtensorMap b, const __grid_constant__ CUtensorMap cd)`; `grouped_layout = nullptr` for `GemmType::Normal`; shape args that have a compiled dim are overridden by the template constant.

Template arguments (`sm90_bf16_gemm.hpp:28-52`), pooler m=128 natural: `<cute::UMMA::Major::K, cute::UMMA::Major::K, /*SHAPE_M*/0, 4096, 8192, /*groups*/1, 64, 64, 64, /*swizzle A,B,D*/128, 128, 128, /*stages*/13, /*TMA,math threads*/128, 128, /*multicast*/1, false, /*SMs*/132, GemmType::Normal, /*accumulate*/false, cutlass::bfloat16_t>`. Swizzle D = `get_swizzle_mode(block_n, 2)`: 128 for bn in {64,128}, 64 for 96, 32 for 48/80.

Tensor maps (`runtime_utils.hpp make_tma_2d_desc`, all via `cuTensorMapEncodeTiled`, driver API loaded lazily by DeepJIT; in a load_inline build either `-lcuda` or `cudaGetDriverEntryPoint("cuTensorMapEncodeTiled")`, which r7's `make_xmaps` already proved on the judge):
- A (x): dtype BFLOAT16, rank 2, `gmem_dims = {K, M}`, `gmem_strides = {x.stride(0) * 2}` (= 2K bytes), `box = {64, load_block_m = 64}` (inner box forced to `swizzle / elem = 64`), elem strides {1,1}, INTERLEAVE_NONE, SWIZZLE_128B, L2_PROMOTION_L2_256B, OOB_FILL_NONE. With BLOCK_K 128/192 the kernel issues 2/3 TMA copies per operand per stage (`tma_copy.cuh copy_nd`, `BLOCK_INNER / 64` loop), each into a 64-row x 128 B swizzle atom.
- B (W): same with `gmem_dims = {K, N}`, stride 2K, `box = {64, load_block_n = bn}`.
- D (out): `gmem_dims = {N, M}`, stride `out.stride(0) * 2` = 2N bytes, `box = {swizzle_cd / 2 = 64, store_block_m = bm}`; the epilogue issues `BLOCK_N / 64` TMA stores per tile (`TMA_D_BLOCK_N`), from a smem D tile written by `stmatrix` (bf16) in the matching 128 B swizzle.
- Alignment asserted: base pointers 16 B, outer strides multiples of 16 B. OOB rows/cols (qkv 2304 / 80 = 28.8 tiles at m=256) are zero-filled on load and clipped on store; `expect_tx` still counts the full box bytes.

Workspace: none (no split-K, no reduction, no counters). Per-stage mbarriers: `full[i].init(1)`, `empty[i].init(multicast * math_warps)`, in dynamic smem after the stages; `fence_barrier_init` + `__syncthreads` (or `cluster_sync` when multicast > 1).

Launch (`DeepJIT kernel.hpp Kernel::launch`): `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size)` on every launch, then `cuLaunchKernelEx` with `gridDim = (132,1,1)`, `blockDim = (256|384,1,1)`, `sharedMemBytes = smem_size`, attributes: `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` only when `cluster_dim.x > 1` (never for our picks), `PROGRAMMATIC_STREAM_SERIALIZATION` only when PDL is enabled (`set_pdl`, default off; the kernel's unconditional `cudaGridDependencySynchronize()` compiles to one `ACQBULK` and is a no-op without PDL), `COOPERATIVE` off. A `cudaLaunchKernelEx` from a load_inline host file is equivalent; no cluster launch means the pw-lineage `pw_wlaunch` with `numAttrs = 0` suffices. `__launch_bounds__(256|384, 1)`: 1 CTA/SM.

Dynamic smem bytes (= `pipeline_config.smem_size`): 221,440 (pooler-128, qkv-128), 213,248 (o-128, unpooler-128, pooler-256), 231,680 (qkv-256), 229,632 (o-256, unpooler-256); plus 1,024 B static reported by cuobjdump (`extern __shared__ __align__(1024)`). All within the 232,448 B (227 KB) opt-in limit on both H100 and H200. The kernel's own layout after merging uses less than the host figure (pooler-128: 8,192 D + 6 x 32,768 + 96 barriers = 204,896), the rest is slack.

Other requirements: `sm_90a` (wgmma, setmaxnreg; fails on plain `sm_90` with "Instruction 'wgmma.fence' not supported on .target 'sm_90'", exactly pw24's first-check failure), driver/toolkit >= 12.3 for `cuTensorMapEncodeTiled` + `cuLaunchKernelEx` (judge is CUDA 12.x), `cute::elect_one_sync`, `cutlass::arch::ClusterTransactionBarrier`, `warpgroup_reg_alloc/dealloc`.

## 3. Embedding in a single submission.py

Licence: MIT (DeepGEMM, (c) 2025 DeepSeek); the copyright and permission notice must travel with the copied headers (1.1 KB), as pw25 already does for CUTLASS's BSD-3 LICENSE.txt. No other terms.

Header closure of `sm90_bf16_gemm.cuh` (nvcc -M, CUTLASS 3.9.2): 72 project files = 16 deep_gemm (118 KB raw), 28 cutlass non-cute (573 KB), 28 cute (4.02 MB, of which `cute/arch/mma_sm90_gmma_ext.hpp` is 2.79 MB and `mma_sm90_gmma.hpp` 0.97 MB), plus toolkit headers that exist on the judge (`cuda.h`, `cuda_bf16.h`, `cuda/std/cstdint` from the CUDA 12 CCCL tree, cuda_runtime) and libstdc++. The 16 deep_gemm files: `impls/sm90_bf16_gemm.cuh`, `comm/barrier.cuh`, `common/{compile,exception,math,packing,tma_copy,types,utils}.cuh`, `layout/{mega_moe,sym_buffer}.cuh`, `mma/sm90.cuh`, `ptx/{ld_st,utils,wgmma}.cuh`, `scheduler/gemm.cuh`. `comm/barrier.cuh` (and through it `layout/mega_moe.cuh`, `layout/sym_buffer.cuh`) is needed only for `cluster_sync_with_relaxed_arrive`, unused at cluster 1; dropping those three leaves 13 files and removes the one C++17 warning (`mega_moe.cuh:269`, a `DG_DEVICE_ASSERT` inside a constexpr constructor). Host-only dependencies not needed in the submission: torch, cublasLt, DeepJIT (nvcc driver + cubin loader), `<format>` (C++20). DeepGEMM has no SM90 mixed-input or dequant kernel (the fp8 1d2d kernel scales fp32 accumulators per block; A is still a TMA-loaded 8-bit tensor), so step B (12-bit A) is our own work in any case.

Sizes measured (tar, xz -9e, base64), `scratchpad/dg_host`:
- full 72-file closure: 4,771,840 raw -> 126,200 xz -> **168,268 b64**;
- delta over pw25's bundle (457 headers; it already holds 55 of the 56 cutlass/cute files): 16 deep_gemm files + `mma_sm90_gmma_ext.hpp` = 2,918,400 raw -> **52,512 b64**;
- delta without `mma_sm90_gmma_ext.hpp` (needed only for N not in {8,16,32,64,96,128,192,256}, i.e. qkv's 48/80; set `block_n` to multiples of 64 instead: qkv-128 then takes 64x64 with 72 blocks, qkv-256 64x128 with 72 blocks, or hand-copy the two `MMA_64x48x16` / `MMA_64x80x16` structs, ~150 lines each): 16 files = **29,032 b64**; 13 files (no comm/layout) = **23,468 b64**.
pw25 is 816,800 bytes with 642 KB of bundle; adding 23-53 KB stays under 1,000,000.

Compile time (nvcc 12.8, 48-core host; the judge took 52 s for pw25's 3 CUTLASS instantiations that took 23.8 s here, ratio ~2.2 including the ~10 s fixed load_inline cost):
- one instantiation, device only (`--cubin`, DeepJIT's way): 1.6-1.8 s each, 0 spills, 248 registers (168 at 384 threads);
- all 8 instantiations in one TU, `--cubin`: 2.35 s; host+device `-c` as load_inline does: 4.2 s for the bare include, 4.3 s for one, 5.15 s for all eight;
- judge estimate: 10-15 s for the extension including the torch host file, i.e. r6 118 s + router 6 s + DeepGEMM ~15 s + bake ~30 s = ~170 s, inside the 240 s guard with room for a few extra tile variants (each extra instantiation ~0.2 s locally).
DeepGEMM's kernels are an order of magnitude cheaper to compile than the CUTLASS collective (no CollectiveBuilder, the MMA selector is one `if constexpr` chain).

Mechanics: no network (headers embedded), no subprocess beyond load_inline (DeepJIT is replaced by static template instantiations in the .cu plus a 40-line host wrapper: 3 `cuTensorMapEncodeTiled`, `cudaFuncSetAttribute`, `cudaLaunchKernelEx`), nothing persistent across calls except the compiled module; graph-capturable (plain kernel launch, no host sync). `-std=c++17` (load_inline's default) compiles the kernel with warnings only; `--expt-relaxed-constexpr` is mandatory (3 hard errors without it, `tma_copy.cuh:57`, `mma/sm90.cuh:219,247`); `sm_90a` mandatory. Build it as its own extension with its own flags, as pw24's post-mortem requires, with the prepare-time fallback on build failure.

## 4. Exact nvcc flags DeepGEMM uses (DeepJIT `options.hpp get_flags` + `backend.hpp` + DeepGEMM `jit.hpp`)

```
nvcc kernel.cu --cubin --output-file kernel.cubin
  --gpu-architecture=sm_90a            # device.get_arch(): "90a" on Hopper (family suffix only for sm_100+)
  -O3 --compiler-options=-O3
  --ptxas-options=--register-usage-level=10
  -std=c++20 --compiler-options=-fPIC --compiler-options=-fconcepts
  --expt-relaxed-constexpr --expt-extended-lambda
  --diag-suppress=39,161,174,177,186,940              # added by DeepGEMM init_jit
  --compiler-options=-Wno-deprecated-declarations,-Wno-abi
  --include-path <deep_gemm>/include                  # cute/ and cutlass/ are copied into it by setup.py
```
Optional, env-driven: `--ptxas-options=--verbose` (DG_JIT_PTXAS_VERBOSE/DEBUG), `--ptxas-options=--warn-on-spills` / `--warn-on-local-memory-usage` (CHECK_NO_SPILLS / CHECK_NO_LOCAL_MEMORY; DeepJIT then greps the output and aborts), `--generate-line-info` (WITH_LINEINFO/DEBUG), `--use_fast_math` only when `fast_math` is set (default false). Not used: `-DNDEBUG` (DeepGEMM's asserts are its own `DG_DEVICE_ASSERT` printf+trap macros; in this kernel they sit only on the non-sm90 branch and in MN-major descriptor code, and the SASS has 0 `CALL`), `-lineinfo` by default, `-rdc`, `--maxrregcount` (registers come from `__launch_bounds__(threads, 1)` + `setmaxnreg`). Compilation is device-only to a cubin loaded through `cuLibraryLoadFromFile` / `cuKernelGetFunction`; there is no host stub.

What our builds lacked: r6's gv extension passes `["-O3", "-lineinfo"]` with torch's default arch (plain sm_90 -> wgmma rejected, pw24 check 1); pw25/pw26's `CUTFLAGS = ["-O3", "-lineinfo", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"]` lack `--ptxas-options=--register-usage-level=10` (DeepJIT's one non-default ptxas knob; its effect on this kernel is unmeasured, the local build above used it), `-std=c++20`, `--expt-extended-lambda` (not needed by this kernel), and add `--use_fast_math` and `-lineinfo` that DeepGEMM does not use. `-DNDEBUG` matters for CUTLASS's collective (pw26 critic) but not for DeepGEMM's kernel.

## 5. Open questions for the H100

1. Pooler m=128 time of the 64x64x(128)x6 DeepGEMM instantiation vs cuBLAS on the same device; it is structurally pw24-bg12 plus merged stages, so the prior is parity at best (bg12 was +4.7 us). DeepSeek's own default of cuBLASLt for bf16 is consistent with that prior.
2. Whether `register-usage-level=10` or the merged BLOCK_K changes the +wgmma rung; ladder: W+x only (comment out the wgmma batch) vs full.
3. qkv: 64x48 (96 blocks, ext header) vs 64x64 (72 blocks) vs 128x16-style tall tiles; the model prefers 48.
4. `set_num_sms(128)` (grid 128, no idle CTAs) is a free variant: same pick, identical blocks.
