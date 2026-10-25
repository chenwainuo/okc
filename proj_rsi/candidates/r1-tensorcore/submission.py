# okc: problem=muse_bf16_proj
"""BF16 projections: split-K CUDA cores for m<=32, tensor cores for aligned m>=64.

Compiled in prepare only when CUDA is present. CPU and every other shape use the
starter matmul. Router fast paths use a prepare-time BF16 hi+lo pair and fp32 acc.
"""

from __future__ import annotations

import torch

_MOD = None

# ~2 waves on 132 SMs. Each CTA is one 64-wide N tile of one K split.
_TARGET_CTAS = 256
_MAX_SPLIT_M = 32

_CPP = r"""
void launch_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partials, torch::Tensor out, int64_t splits, int64_t accumulate);
void launch_tc(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t accumulate);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

// m16n8k16.row.col.f32.bf16.bf16.f32 fragment map (PTX ISA):
//   lane group = lane>>2, tid = lane&3
//   A row-major 16x16: reg i holds (row, col)
//     row = group            if i in {0,1,4,5} else group+8
//     col = (tid*2)+(i&1)    if i<4 else (tid*2)+(i&1)+8
//   B col-major 16x8 (logical K x N): 
//     k   = (tid*2)+(i&1)    if i<2 else (tid*2)+(i&1)+8
//     n   = group
//   C 16x8 fp32:
//     row = group if i<2 else group+8
//     col = (tid*2)+(i&1)

__device__ __forceinline__ float fma_bf16x2(float acc, unsigned xbits, unsigned wbits) {
    float x0 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xbits)));
    float x1 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(xbits >> 16)));
    float w0 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(wbits)));
    float w1 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(wbits >> 16)));
    acc = fmaf(x0, w0, acc);
    acc = fmaf(x1, w1, acc);
    return acc;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned addr;
    asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(addr) : "l"(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Split-K CUDA-core. One CTA = N-tile 64 x one K split. Weights are staged
// through smem with 16-byte ld.global.cs so consecutive threads cover a row.
// Partials are fp32 [splits, m, n]. Leftover rows (m not in {1,8,16,32}) are
// predicated. Ws stride 72 keeps the 16-byte smem reload at a 4-way conflict,
// which is the minimum when 32 threads each take 4 banks.
// ---------------------------------------------------------------------------

template <int M>
__global__ void splitk_partial(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    float* __restrict__ partials,
    int m, int n, int k, int k_split) {
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WSTRIDE = 72;
    constexpr int MAXR = (M == 0) ? 32 : M;
    const int rows = (M == 0) ? m : M;
    const int tile = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int n0 = tile * BN;
    const int k0 = split * k_split;

    __shared__ __align__(16) __nv_bfloat16 Xs[32][BK];
    __shared__ __align__(16) __nv_bfloat16 Ws[BN][WSTRIDE];

    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r) acc[r] = 0.f;

    for (int kb = 0; kb < k_split; kb += BK) {
        const int kbase = k0 + kb;
#pragma unroll
        for (int rep = 0; rep < 4; ++rep) {
            const int e = tid + rep * 128;
            const int row = e >> 3;
            const int part = e & 7;
            const uint4* src = reinterpret_cast<const uint4*>(
                W + (static_cast<long long>(n0 + row) * k + kbase + part * 8));
            uint4 v = __ldcs(src);
            *reinterpret_cast<uint4*>(&Ws[row][part * 8]) = v;
        }
#pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int e = tid + rep * 128;
            const int row = e >> 3;
            const int part = e & 7;
            if (row < rows) {
                const uint4* src = reinterpret_cast<const uint4*>(
                    X + (static_cast<long long>(row) * k + kbase + part * 8));
                uint4 v = *src;
                *reinterpret_cast<uint4*>(&Xs[row][part * 8]) = v;
            }
        }
        __syncthreads();

        if (tid < BN) {
#pragma unroll 1
            for (int t = 0; t < BK; t += 8) {
                uint4 w = *reinterpret_cast<const uint4*>(&Ws[tid][t]);
                unsigned wb[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int r = 0; r < MAXR; ++r) {
                    if constexpr (M == 0) {
                        if (r >= rows) continue;
                    }
                    unsigned xb0 = *reinterpret_cast<const unsigned*>(&Xs[r][t]);
                    unsigned xb1 = *reinterpret_cast<const unsigned*>(&Xs[r][t + 2]);
                    unsigned xb2 = *reinterpret_cast<const unsigned*>(&Xs[r][t + 4]);
                    unsigned xb3 = *reinterpret_cast<const unsigned*>(&Xs[r][t + 6]);
                    float a = acc[r];
                    a = fma_bf16x2(a, xb0, wb[0]);
                    a = fma_bf16x2(a, xb1, wb[1]);
                    a = fma_bf16x2(a, xb2, wb[2]);
                    a = fma_bf16x2(a, xb3, wb[3]);
                    acc[r] = a;
                }
            }
        }
        __syncthreads();
    }

    if (tid < BN) {
#pragma unroll
        for (int r = 0; r < MAXR; ++r) {
            if constexpr (M == 0) {
                if (r >= rows) continue;
            }
            partials[(static_cast<long long>(split) * rows + r) * n + (n0 + tid)] = acc[r];
        }
    }
}

template <int IsFp32, int Accumulate>
__global__ void reduce_partials(const float* __restrict__ partials, void* __restrict__ out, int total, int splits) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        acc += partials[static_cast<long long>(s) * total + idx];
    }
    if constexpr (IsFp32) {
        float* o = static_cast<float*>(out);
        if constexpr (Accumulate) o[idx] = o[idx] + acc;
        else o[idx] = acc;
    } else {
        reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(acc);
    }
}

void launch_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partials, torch::Tensor out, int64_t splits, int64_t accumulate) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && partials.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && partials.is_contiguous() && out.is_contiguous());
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = static_cast<int>(w.size(0));
    TORCH_CHECK(m > 0 && m <= 32);
    TORCH_CHECK(n % 64 == 0 && k % 64 == 0);
    TORCH_CHECK(splits > 0 && k % splits == 0);
    TORCH_CHECK((k / splits) % 64 == 0);
    TORCH_CHECK(partials.numel() >= splits * static_cast<int64_t>(m) * n);
    const int k_split = static_cast<int>(k / splits);
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    float* pp = partials.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 block(128);
    dim3 grid(n / 64, static_cast<int>(splits));
    if (m == 1) {
        splitk_partial<1><<<grid, block, 0, stream>>>(xp, wp, pp, m, n, k, k_split);
    } else if (m == 8) {
        splitk_partial<8><<<grid, block, 0, stream>>>(xp, wp, pp, m, n, k, k_split);
    } else if (m == 16) {
        splitk_partial<16><<<grid, block, 0, stream>>>(xp, wp, pp, m, n, k, k_split);
    } else if (m == 32) {
        splitk_partial<32><<<grid, block, 0, stream>>>(xp, wp, pp, m, n, k, k_split);
    } else {
        splitk_partial<0><<<grid, block, 0, stream>>>(xp, wp, pp, m, n, k, k_split);
    }
    const int total = m * n;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const bool fp32 = out.scalar_type() == at::kFloat;
    if (fp32 && accumulate) {
        reduce_partials<1, 1><<<blocks, threads, 0, stream>>>(pp, out.data_ptr(), total, static_cast<int>(splits));
    } else if (fp32) {
        reduce_partials<1, 0><<<blocks, threads, 0, stream>>>(pp, out.data_ptr(), total, static_cast<int>(splits));
    } else {
        reduce_partials<0, 0><<<blocks, threads, 0, stream>>>(pp, out.data_ptr(), total, static_cast<int>(splits));
    }
}

// ---------------------------------------------------------------------------
// Tensor-core tile BM=BN=64, BK=32. 128 threads: warp w owns rows [16w, 16w+16)
// and all 64 columns as eight m16n8 accumulators. 2-stage cp.async pipeline.
// Smem K stride 40 is 16-byte aligned and conflict-free for the fragment loads.
// ---------------------------------------------------------------------------

constexpr int TC_BM = 64;
constexpr int TC_BN = 64;
constexpr int TC_BK = 32;
constexpr int TC_STRIDE = 40;

__device__ __forceinline__ unsigned ld_u32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(a));
    const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(b));
    return lo | (hi << 16);
}

__device__ __forceinline__ void load_ab(
    __nv_bfloat16 As[2][TC_BM][TC_STRIDE],
    __nv_bfloat16 Bs[2][TC_BN][TC_STRIDE],
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    int stage, int m0, int n0, int k0, int k) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int e = tid + rep * 128;  // 256 chunks of 8 bf16 cover 64*32
        const int row = e >> 2;
        const int col = (e & 3) * 8;
        cp_async16(&As[stage][row][col], X + (static_cast<long long>(m0 + row) * k + k0 + col));
        cp_async16(&Bs[stage][row][col], W + (static_cast<long long>(n0 + row) * k + k0 + col));
    }
}

template <int IsFp32, int Accumulate>
__global__ void tc_gemm(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    void* __restrict__ Out,
    int m, int n, int k) {
    __shared__ __align__(16) __nv_bfloat16 As[2][TC_BM][TC_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][TC_BN][TC_STRIDE];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int m0 = static_cast<int>(blockIdx.y) * TC_BM;
    const int n0 = static_cast<int>(blockIdx.x) * TC_BN;
    const int row_base = warp * 16;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }

    // Prologue loads tile 0. Each iteration prefetches the next K tile into the
    // other stage, waits until only that prefetch is outstanding, then computes.
    load_ab(As, Bs, X, W, 0, m0, n0, 0, k);
    cp_commit();
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += TC_BK) {
        const int next = k0 + TC_BK;
        if (next < k) {
            load_ab(As, Bs, X, W, buf ^ 1, m0, n0, next, k);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            const unsigned a0 = ld_u32(&As[buf][row_base + group][kk + tidg * 2]);
            const unsigned a1 = ld_u32(&As[buf][row_base + group + 8][kk + tidg * 2]);
            const unsigned a2 = ld_u32(&As[buf][row_base + group][kk + 8 + tidg * 2]);
            const unsigned a3 = ld_u32(&As[buf][row_base + group + 8][kk + 8 + tidg * 2]);
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const int ncol = ni * 8 + group;
                const unsigned b0 = ld_u32(&Bs[buf][ncol][kk + tidg * 2]);
                const unsigned b1 = ld_u32(&Bs[buf][ncol][kk + 8 + tidg * 2]);
                mma_m16n8k16(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

    const int row0 = m0 + row_base + group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int col = n0 + ni * 8 + tidg * 2;
        const float v0 = acc[ni][0];
        const float v1 = acc[ni][1];
        const float v2 = acc[ni][2];
        const float v3 = acc[ni][3];
        if constexpr (IsFp32) {
            float* o = static_cast<float*>(Out);
            const long long i0 = static_cast<long long>(row0) * n + col;
            const long long i1 = static_cast<long long>(row1) * n + col;
            if constexpr (Accumulate) {
                o[i0] += v0;
                o[i0 + 1] += v1;
                o[i1] += v2;
                o[i1 + 1] += v3;
            } else {
                o[i0] = v0;
                o[i0 + 1] = v1;
                o[i1] = v2;
                o[i1 + 1] = v3;
            }
        } else {
            auto* o = reinterpret_cast<__nv_bfloat16*>(Out);
            *reinterpret_cast<unsigned*>(o + (static_cast<long long>(row0) * n + col)) = pack_bf16(v0, v1);
            *reinterpret_cast<unsigned*>(o + (static_cast<long long>(row1) * n + col)) = pack_bf16(v2, v3);
        }
    }
}

void launch_tc(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t accumulate) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = static_cast<int>(w.size(0));
    TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % 32 == 0);
    TORCH_CHECK(m > 0 && n > 0 && k > 0);
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 block(128);
    dim3 grid(n / 64, m / 64);
    const bool fp32 = out.scalar_type() == at::kFloat;
    if (fp32 && accumulate) {
        tc_gemm<1, 1><<<grid, block, 0, stream>>>(xp, wp, out.data_ptr(), m, n, k);
    } else if (fp32) {
        tc_gemm<1, 0><<<grid, block, 0, stream>>>(xp, wp, out.data_ptr(), m, n, k);
    } else {
        tc_gemm<0, 0><<<grid, block, 0, stream>>>(xp, wp, out.data_ptr(), m, n, k);
    }
}
"""


def choose_splits(n: int, k: int) -> int:
    """Smallest split count with about _TARGET_CTAS blocks and k_per a multiple of 64."""
    n_tiles = n // 64
    if n_tiles <= 0 or k < 64 or (k % 64) != 0:
        return 1
    k_chunks = k // 64
    best = 1
    for splits in range(1, k_chunks + 1):
        if k_chunks % splits != 0:
            continue
        best = splits
        if n_tiles * splits >= _TARGET_CTAS:
            break
    return best


def _max_partial_elems(state: dict) -> int:
    need = 1
    shapes = []
    for name in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        weight = state[name]
        n_full, k_full = int(weight.shape[0]), int(weight.shape[1])
        shapes.append((n_full, k_full))
        if name == "unpooler":
            shapes.append((min(1024, n_full), k_full))
    for n_full, k_full in shapes:
        if k_full % 64 != 0:
            continue
        for n in range(64, n_full + 1, 64):
            splits = choose_splits(n, k_full)
            need = max(need, splits * _MAX_SPLIT_M * n)
    return need


def _compile():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline

        _MOD = load_inline(
            name="muse_bf16_r1_tensorcore",
            cpp_sources=[_CPP],
            cuda_sources=[_CUDA],
            functions=["launch_splitk", "launch_tc"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            with_cuda=True,
            verbose=False,
        )
    return _MOD


def prepare(weights):
    state = {
        name: weights[name].contiguous()
        for name in ("qkv_gate", "o", "pooler", "unpooler", "router")
    }
    state["cuda"] = False
    if not torch.cuda.is_available():
        return state
    router = state["router"]
    if router.device.type != "cuda":
        return state
    # FP32 router as BF16 hi + residual lo. Original BF16 weights stay BF16.
    hi = router.to(dtype=torch.bfloat16)
    lo = (router - hi.float()).to(dtype=torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["mod"] = _compile()
    state["partials"] = torch.empty(_max_partial_elems(state), dtype=torch.float32, device=router.device)
    state["cuda"] = True
    return state


def _starter(x, state, layer, n):
    weight = state[layer][:n]
    if weight.dtype == torch.float32:
        return x.float() @ weight.T
    return x @ weight.T


def _slice_weight(weight, n):
    if weight.shape[0] == n:
        return weight
    return weight[:n]


def _splitk(state, x, layer, n, splits):
    mod = state["mod"]
    partials = state["partials"]
    if layer == "router":
        out = torch.empty(x.shape[0], n, dtype=torch.float32, device=x.device)
        mod.launch_splitk(x, _slice_weight(state["router_hi"], n), partials, out, splits, 0)
        mod.launch_splitk(x, _slice_weight(state["router_lo"], n), partials, out, splits, 1)
        return out
    out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=x.device)
    mod.launch_splitk(x, _slice_weight(state[layer], n), partials, out, splits, 0)
    return out


def _tc(state, x, layer, n):
    mod = state["mod"]
    if layer == "router":
        out = torch.empty(x.shape[0], n, dtype=torch.float32, device=x.device)
        mod.launch_tc(x, _slice_weight(state["router_hi"], n), out, 0)
        mod.launch_tc(x, _slice_weight(state["router_lo"], n), out, 1)
        return out
    out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=x.device)
    mod.launch_tc(x, _slice_weight(state[layer], n), out, 0)
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["x"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = x.shape[0]
    k = x.shape[1]
    weight = state[layer]
    cuda_ok = (
        state["cuda"]
        and x.is_cuda
        and x.is_contiguous()
        and weight.shape[0] >= n
        and weight.shape[1] == k
    )
    if cuda_ok and 0 < m <= _MAX_SPLIT_M and n % 64 == 0 and k % 64 == 0:
        splits = choose_splits(n, k)
        k_per = k // splits
        if k_per % 64 == 0 and splits * m * n <= state["partials"].numel():
            sliced = _slice_weight(weight, n)
            if sliced.is_contiguous():
                return _splitk(state, x, layer, n, splits)
    if cuda_ok and m >= 64 and m % 64 == 0 and n % 64 == 0 and k % 32 == 0:
        sliced = _slice_weight(weight, n)
        if sliced.is_contiguous():
            return _tc(state, x, layer, n)
    return _starter(x, state, layer, n)
