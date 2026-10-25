# okc: problem=muse_bf16_proj
"""Per-shape CUDA-graph autotune: split-K and a fused router, else torch.

Projections stay on cuBLAS unless a custom kernel is faster under graph replay
and inside the reference tolerance. The router is the shape cuBLAS leaves on
the table: one FP32 pass for short M, fused BF16 tensor cores for M=128/256.
CPU and any shape that loses the bake-off use the starter matmul.
"""

from typing import TypedDict

import torch

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float dot8(float4 a, float4 b) {
    const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&b);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(ap[i]);
        float2 fb = __bfloat1622float2(bp[i]);
        s = fmaf(fa.x, fb.x, s);
        s = fmaf(fa.y, fb.y, s);
    }
    return s;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

extern __shared__ __align__(16) char muse_smem[];

// One warp, one output column, so the grid can fill the SMs. X for the K split
// sits in smem and is reused across the columns of the CTA. Two weight vectors
// are in flight before the FMAs.
template <int M, int COLS>
__global__ void sk_bf16(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                        float* __restrict__ partial, int N, int K, int k_split, int ld_m, int ld_n) {
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(muse_smem);
    const int split = static_cast<int>(blockIdx.y);
    const int k0 = split * k_split;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int threads = COLS * 32;

    const int span = k_split >> 3;
    const int vecs = M * span;
    for (int v = tid; v < vecs; v += threads) {
        const int row = v / span;
        const int kv = v - row * span;
        const float4* src = reinterpret_cast<const float4*>(
            X + (static_cast<size_t>(row) * K + k0 + kv * 8));
        float4* dst = reinterpret_cast<float4*>(xs + (static_cast<size_t>(row) * k_split + kv * 8));
        *dst = __ldcg(src);
    }
    __syncthreads();

    const int col = static_cast<int>(blockIdx.x) * COLS + warp;
    if (col >= N) return;
    const __nv_bfloat16* wrow = W + (static_cast<size_t>(col) * K + k0);
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.f;
    for (int base = 0; base < k_split; base += 512) {
        const int kk = base + lane * 8;
        float4 w0 = __ldcs(reinterpret_cast<const float4*>(wrow + kk));
        if (base + 256 < k_split) {
            float4 w1 = __ldcs(reinterpret_cast<const float4*>(wrow + kk + 256));
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float4 x0 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk));
                float4 x1 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk + 256));
                acc[m] = acc[m] + dot8(x0, w0) + dot8(x1, w1);
            }
        } else {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float4 x0 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk));
                acc[m] += dot8(x0, w0);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_sum(acc[m]);
        if (lane == 0)
            partial[(static_cast<size_t>(split) * ld_m + m) * ld_n + col] = v;
    }
}

template <int M, int COLS>
__global__ void sk_f32(const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                       float* __restrict__ partial, int N, int K, int k_split, int ld_m, int ld_n) {
    float* xs = reinterpret_cast<float*>(muse_smem);
    const int split = static_cast<int>(blockIdx.y);
    const int k0 = split * k_split;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int threads = COLS * 32;

    const int span = k_split >> 3;
    const int vecs = M * span;
    for (int v = tid; v < vecs; v += threads) {
        const int row = v / span;
        const int kv = v - row * span;
        float4 packed = __ldcg(reinterpret_cast<const float4*>(
            X + (static_cast<size_t>(row) * K + k0 + kv * 8)));
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&packed);
        float2 a = __bfloat1622float2(p[0]);
        float2 b = __bfloat1622float2(p[1]);
        float2 c = __bfloat1622float2(p[2]);
        float2 d = __bfloat1622float2(p[3]);
        float* dst = xs + (static_cast<size_t>(row) * k_split + static_cast<size_t>(kv) * 8);
        *reinterpret_cast<float4*>(dst) = make_float4(a.x, a.y, b.x, b.y);
        *reinterpret_cast<float4*>(dst + 4) = make_float4(c.x, c.y, d.x, d.y);
    }
    __syncthreads();

    const int col = static_cast<int>(blockIdx.x) * COLS + warp;
    if (col >= N) return;
    const float* wrow = W + (static_cast<size_t>(col) * K + k0);
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.f;
    for (int base = 0; base < k_split; base += 256) {
        const int kk = base + lane * 4;
        float4 w0 = __ldcs(reinterpret_cast<const float4*>(wrow + kk));
        if (base + 128 < k_split) {
            float4 w1 = __ldcs(reinterpret_cast<const float4*>(wrow + kk + 128));
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float4 x0 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk));
                float4 x1 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk + 128));
                acc[m] = acc[m] + dot4(x0, w0) + dot4(x1, w1);
            }
        } else {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float4 x0 = *reinterpret_cast<const float4*>(xs + (static_cast<size_t>(m) * k_split + kk));
                acc[m] += dot4(x0, w0);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_sum(acc[m]);
        if (lane == 0)
            partial[(static_cast<size_t>(split) * ld_m + m) * ld_n + col] = v;
    }
}

__global__ void reduce_bf16(const float* __restrict__ partial, __nv_bfloat16* __restrict__ out,
                            int M, int N, int splits, int ld_m, int ld_n) {
    const int total = M * N;
    for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
         idx += static_cast<int>(blockDim.x * gridDim.x)) {
        const int m = idx / N;
        const int n = idx - m * N;
        float s = 0.f;
        for (int sp = 0; sp < splits; ++sp)
            s += partial[(static_cast<size_t>(sp) * ld_m + m) * ld_n + n];
        out[static_cast<size_t>(m) * N + n] = __float2bfloat16(s);
    }
}

__global__ void reduce_f32(const float* __restrict__ partial, float* __restrict__ out,
                           int M, int N, int splits, int ld_m, int ld_n) {
    const int total = M * N;
    for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < total;
         idx += static_cast<int>(blockDim.x * gridDim.x)) {
        const int m = idx / N;
        const int n = idx - m * N;
        float s = 0.f;
        for (int sp = 0; sp < splits; ++sp)
            s += partial[(static_cast<size_t>(sp) * ld_m + m) * ld_n + n];
        out[static_cast<size_t>(m) * N + n] = s;
    }
}

template <int M, int COLS>
void launch_sk_bf16_m(const __nv_bfloat16* x, const __nv_bfloat16* w, float* partial,
                      int N, int K, int k_split, int ld_m, int ld_n, cudaStream_t stream) {
    dim3 block(COLS * 32);
    dim3 grid(N / COLS, K / k_split);
    const size_t smem = static_cast<size_t>(M) * k_split * sizeof(__nv_bfloat16);
    sk_bf16<M, COLS><<<grid, block, smem, stream>>>(x, w, partial, N, K, k_split, ld_m, ld_n);
}

template <int M, int COLS>
void launch_sk_f32_m(const __nv_bfloat16* x, const float* w, float* partial,
                     int N, int K, int k_split, int ld_m, int ld_n, cudaStream_t stream) {
    dim3 block(COLS * 32);
    dim3 grid(N / COLS, K / k_split);
    const size_t smem = static_cast<size_t>(M) * k_split * sizeof(float);
    sk_f32<M, COLS><<<grid, block, smem, stream>>>(x, w, partial, N, K, k_split, ld_m, ld_n);
}

void launch_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out,
                 int64_t k_split, int64_t cols) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && partial.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && partial.is_contiguous());
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(w.size(0));
    TORCH_CHECK(M == 1 || M == 8 || M == 16 || M == 32);
    TORCH_CHECK(cols == 8 || cols == 16 || cols == 32);
    TORCH_CHECK(N % cols == 0 && k_split > 0 && K % k_split == 0 && k_split % 256 == 0);
    TORCH_CHECK(partial.size(0) >= K / k_split && partial.size(1) >= M && partial.size(2) >= N);
    const int ld_m = static_cast<int>(partial.size(1));
    const int ld_n = static_cast<int>(partial.size(2));
    const int splits = K / static_cast<int>(k_split);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    float* pp = partial.data_ptr<float>();
    if (cols == 8 && M == 1) launch_sk_bf16_m<1, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8 && M == 8) launch_sk_bf16_m<8, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8 && M == 16) launch_sk_bf16_m<16, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8) launch_sk_bf16_m<32, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 1) launch_sk_bf16_m<1, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 8) launch_sk_bf16_m<8, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 16) launch_sk_bf16_m<16, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16) launch_sk_bf16_m<32, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 1) launch_sk_bf16_m<1, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 8) launch_sk_bf16_m<8, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 16) launch_sk_bf16_m<16, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else launch_sk_bf16_m<32, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    const int total = M * N;
    const int threads = 256;
    const int blocks = std::min((total + threads - 1) / threads, 2048);
    reduce_bf16<<<blocks, threads, 0, stream>>>(pp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                M, N, splits, ld_m, ld_n);
}

void launch_f32(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out,
                int64_t k_split, int64_t cols) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && partial.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && partial.is_contiguous());
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(w.size(0));
    TORCH_CHECK(M == 1 || M == 8 || M == 16 || M == 32);
    TORCH_CHECK(cols == 8 || cols == 16 || cols == 32);
    TORCH_CHECK(N % cols == 0 && k_split > 0 && K % k_split == 0 && k_split % 256 == 0);
    TORCH_CHECK(partial.size(0) >= K / k_split && partial.size(1) >= M && partial.size(2) >= N);
    const int ld_m = static_cast<int>(partial.size(1));
    const int ld_n = static_cast<int>(partial.size(2));
    const int splits = K / static_cast<int>(k_split);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* wp = reinterpret_cast<const float*>(w.data_ptr());
    float* pp = partial.data_ptr<float>();
    if (cols == 8 && M == 1) launch_sk_f32_m<1, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8 && M == 8) launch_sk_f32_m<8, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8 && M == 16) launch_sk_f32_m<16, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 8) launch_sk_f32_m<32, 8>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 1) launch_sk_f32_m<1, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 8) launch_sk_f32_m<8, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16 && M == 16) launch_sk_f32_m<16, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 16) launch_sk_f32_m<32, 16>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 1) launch_sk_f32_m<1, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 8) launch_sk_f32_m<8, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else if (cols == 32 && M == 16) launch_sk_f32_m<16, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    else launch_sk_f32_m<32, 32>(xp, wp, pp, N, K, (int)k_split, ld_m, ld_n, stream);
    const int total = M * N;
    const int threads = 256;
    const int blocks = std::min((total + threads - 1) / threads, 2048);
    reduce_f32<<<blocks, threads, 0, stream>>>(pp, out.data_ptr<float>(), M, N, splits, ld_m, ld_n);
}

// m16n8k16.row.col.f32.bf16.bf16.f32. A is row-major MxK, B is W[n, k] (column-major KxN).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int TC_BM = 64;
constexpr int TC_BN = 64;
constexpr int TC_BK = 32;
constexpr int TC_STRIDE = 40;

__device__ __forceinline__ unsigned ld_u32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

__device__ __forceinline__ void load_ab(
    __nv_bfloat16 As[TC_BM][TC_STRIDE], __nv_bfloat16 Bs[TC_BN][TC_STRIDE],
    const __nv_bfloat16* X, const __nv_bfloat16* W,
    int m0, int n0, int k0, int K) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int e = tid + rep * 128;
        const int row = e >> 2;
        const int col = (e & 3) * 8;
        cp_async16(&As[row][col], X + (static_cast<long long>(m0 + row) * K + k0 + col));
        cp_async16(&Bs[row][col], W + (static_cast<long long>(n0 + row) * K + k0 + col));
    }
}

__device__ __forceinline__ void load_b(
    __nv_bfloat16 Bs[TC_BN][TC_STRIDE], const __nv_bfloat16* W, int n0, int k0, int K) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int e = tid + rep * 128;
        const int row = e >> 2;
        const int col = (e & 3) * 8;
        cp_async16(&Bs[row][col], W + (static_cast<long long>(n0 + row) * K + k0 + col));
    }
}

__device__ __forceinline__ void mma_k32(
    float acc[8][4], __nv_bfloat16 As[TC_BM][TC_STRIDE], __nv_bfloat16 Bs[TC_BN][TC_STRIDE]) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int row_base = warp * 16;
#pragma unroll
    for (int kk = 0; kk < TC_BK; kk += 16) {
        const unsigned a0 = ld_u32(&As[row_base + group][kk + tidg * 2]);
        const unsigned a1 = ld_u32(&As[row_base + group + 8][kk + tidg * 2]);
        const unsigned a2 = ld_u32(&As[row_base + group][kk + 8 + tidg * 2]);
        const unsigned a3 = ld_u32(&As[row_base + group + 8][kk + 8 + tidg * 2]);
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int ncol = ni * 8 + group;
            const unsigned b0 = ld_u32(&Bs[ncol][kk + tidg * 2]);
            const unsigned b1 = ld_u32(&Bs[ncol][kk + 8 + tidg * 2]);
            mma_m16n8k16(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3], a0, a1, a2, a3, b0, b1);
        }
    }
}

// Fused router: acc = X @ Hi^T + X @ Lo^T over one K split. 64x64 tile.
// Double-buffered so the next K tile's copies overlap the current MMA.
__global__ void __launch_bounds__(128)
tc_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ Hi,
          const __nv_bfloat16* __restrict__ Lo, float* __restrict__ partial,
          int M, int N, int K, int k_split) {
    __shared__ __align__(16) __nv_bfloat16 As[2][TC_BM][TC_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][TC_BN][TC_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 Ls[2][TC_BN][TC_STRIDE];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int split = static_cast<int>(blockIdx.z);
    const int m0 = static_cast<int>(blockIdx.y) * TC_BM;
    const int n0 = static_cast<int>(blockIdx.x) * TC_BN;
    const int k_begin = split * k_split;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    load_ab(As[0], Bs[0], X, Hi, m0, n0, k_begin, K);
    load_b(Ls[0], Lo, n0, k_begin, K);
    cp_commit();
    int buf = 0;
    for (int kk = 0; kk < k_split; kk += TC_BK) {
        const int next = kk + TC_BK;
        if (next < k_split) {
            load_ab(As[buf ^ 1], Bs[buf ^ 1], X, Hi, m0, n0, k_begin + next, K);
            load_b(Ls[buf ^ 1], Lo, n0, k_begin + next, K);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        mma_k32(acc, As[buf], Bs[buf]);
        mma_k32(acc, As[buf], Ls[buf]);
        __syncthreads();
        buf ^= 1;
    }

    const int row_base = warp * 16;
    const int row0 = m0 + row_base + group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int col = n0 + ni * 8 + tidg * 2;
        partial[(static_cast<long long>(split) * M + row0) * N + col] = acc[ni][0];
        partial[(static_cast<long long>(split) * M + row0) * N + col + 1] = acc[ni][1];
        partial[(static_cast<long long>(split) * M + row1) * N + col] = acc[ni][2];
        partial[(static_cast<long long>(split) * M + row1) * N + col + 1] = acc[ni][3];
    }
}

void launch_tc(torch::Tensor x, torch::Tensor hi, torch::Tensor lo,
               torch::Tensor partial, torch::Tensor out, int64_t k_split) {
    TORCH_CHECK(x.is_cuda() && hi.is_cuda() && lo.is_cuda() && partial.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && hi.is_contiguous() && lo.is_contiguous());
    TORCH_CHECK(partial.is_contiguous() && out.is_contiguous());
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(hi.size(0));
    TORCH_CHECK(M % 64 == 0 && N % 64 == 0 && k_split > 0 && K % k_split == 0 && k_split % 32 == 0);
    TORCH_CHECK(partial.numel() >= static_cast<long long>(K / k_split) * M * N);
    const int splits = K / static_cast<int>(k_split);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(128);
    dim3 grid(N / 64, M / 64, splits);
    tc_router<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(hi.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(lo.data_ptr()),
        partial.data_ptr<float>(), M, N, K, static_cast<int>(k_split));
    const int total = M * N;
    const int threads = 256;
    const int blocks = std::max(1, std::min((total + threads - 1) / threads, 2048));
    reduce_f32<<<blocks, threads, 0, stream>>>(
        partial.data_ptr<float>(), out.data_ptr<float>(), M, N, splits, M, N);
}

void init_kernels() {
    const void* fns[] = {
        (const void*)sk_bf16<1, 8>, (const void*)sk_bf16<8, 8>, (const void*)sk_bf16<16, 8>, (const void*)sk_bf16<32, 8>,
        (const void*)sk_bf16<1, 16>, (const void*)sk_bf16<8, 16>, (const void*)sk_bf16<16, 16>, (const void*)sk_bf16<32, 16>,
        (const void*)sk_bf16<1, 32>, (const void*)sk_bf16<8, 32>, (const void*)sk_bf16<16, 32>, (const void*)sk_bf16<32, 32>,
        (const void*)sk_f32<1, 8>, (const void*)sk_f32<8, 8>, (const void*)sk_f32<16, 8>, (const void*)sk_f32<32, 8>,
        (const void*)sk_f32<1, 16>, (const void*)sk_f32<8, 16>, (const void*)sk_f32<16, 16>, (const void*)sk_f32<32, 16>,
        (const void*)sk_f32<1, 32>, (const void*)sk_f32<8, 32>, (const void*)sk_f32<16, 32>, (const void*)sk_f32<32, 32>,
    };
    for (const void* fn : fns) {
        cudaError_t err = cudaFuncSetAttribute(
            fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 192 * 1024);
        TORCH_CHECK(err == cudaSuccess, "smem opt-in failed: ", static_cast<int>(err));
    }
}
"""


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


_SHAPES = ((1, 1), (16, 16), (32, 32), (8, 8), (128, 128), (256, 256))
_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_KSPLITS = (256, 512, 1024)
_COLS = (8, 16, 32)


def _torch_out(x, state, layer, n):
    weight = state["weights"][layer]
    if weight.shape[0] != n:
        weight = weight[:n]
    if layer != "router":
        return x @ weight.T
    if x.shape[0] <= 8 or not x.is_cuda:
        return x.float() @ weight.T
    hi_t = state["hi_t"]
    lo_t = state["lo_t"]
    if hi_t.shape[1] != n:
        hi_t = hi_t[:, :n].contiguous()
        lo_t = lo_t[:, :n].contiguous()
    return torch.mm(x, hi_t, out_dtype=torch.float32) + torch.mm(x, lo_t, out_dtype=torch.float32)


def _rel(out, ref):
    denom = ref.norm().clamp(min=1e-12)
    return float((out.double() - ref).norm() / denom)


def _matches(out, ref, probe, layer):
    tol = 1e-4 if layer == "router" else 5e-3
    if _rel(out, ref) > tol:
        return False
    return _rel(out[0], probe) <= 1e-3


def _compile():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="proj_r4_fast",
        cpp_sources=[
            "#include <torch/extension.h>\n"
            "void init_kernels();\n"
            "void launch_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out, int64_t k_split, int64_t cols);\n"
            "void launch_f32(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out, int64_t k_split, int64_t cols);\n"
            "void launch_tc(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor partial, torch::Tensor out, int64_t k_split);\n"
        ],
        cuda_sources=[_CUDA],
        functions=["init_kernels", "launch_bf16", "launch_f32", "launch_tc"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=False,
    )


def _run_sk(state, x, w, out, k_split, cols):
    mod = state["mod"]
    partial = state["partial"]
    fp32 = w.dtype == torch.float32
    launch = mod.launch_f32 if fp32 else mod.launch_bf16
    m = x.shape[0]
    if m <= 32:
        launch(x, w, partial, out, int(k_split), int(cols))
        return out
    for row in range(0, m, 32):
        launch(x[row:row + 32], w, partial, out[row:row + 32], int(k_split), int(cols))
    return out


def _run_tc(state, x, hi, lo, out, k_split):
    state["mod"].launch_tc(x, hi, lo, state["tc_partial"], out, int(k_split))
    return out


def _graph_us(fn, ref, probe, layer, flush):
    holder = [None]

    def wrapped():
        holder[0] = fn()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            wrapped()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if not _matches(holder[0], ref, probe, layer):
        return None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapped()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    if not _matches(holder[0], ref, probe, layer):
        return None
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(3):
        flush.view(torch.float32).sum()
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0))
    samples.sort()
    return samples[1]


def _consider(cands, name, fn, ref, probe, layer, flush):
    try:
        us = _graph_us(fn, ref, probe, layer, flush)
    except Exception as error:
        print(f"skip {layer} {name}: {type(error).__name__}: {error}", flush=True)
        return
    if us is None:
        print(f"reject {layer} {name}: accuracy", flush=True)
        return
    print(f"cand {layer} {name} {us:.2f}", flush=True)
    cands.append((us, name))


def _bind(state, layer, m, n, k, device):
    # Projections are already on the cuBLAS roofline. Only the router has
    # enough slack to move the geomean, so the bake-off stays there.
    if layer != "router":
        state["plan"][(layer, m)] = "torch"
        print(f"plan {layer:10} m={m:<4} torch      (kept)", flush=True)
        return
    x = torch.randn(m, k, device=device, dtype=torch.float32)
    x[0].zero_()
    x[0, 7] = 32.0
    x = x.to(torch.bfloat16)
    if layer == "router":
        w_ref = state["router"][:n]
        w_fast = w_ref
        hi = state["hi"][:n]
        lo = state["lo"][:n]
        out = torch.empty(m, n, dtype=torch.float32, device=device)
    else:
        w_ref = state["weights"][layer][:n]
        w_fast = w_ref
        hi = lo = None
        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    acc = torch.empty(m, n, dtype=torch.float32, device=device)
    acc2 = torch.empty(m, n, dtype=torch.float32, device=device)
    ref = x.double() @ w_ref.double().T
    probe = w_ref[:, 7].double() * 32.0
    cands = []

    def torch_fn(x=x):
        return _torch_out(x, state, layer, n)

    flush = state["flush"]
    _consider(cands, "torch", torch_fn, ref, probe, layer, flush)

    mod = state["mod"]
    sk_ok = mod is not None and m in (1, 8, 16, 32) and k % 256 == 0
    if sk_ok:
        for cols in _COLS:
            if n % cols != 0:
                continue
            for k_split in _KSPLITS:
                if k % k_split != 0:
                    continue
                tile_m = 32 if m > 32 else m
                smem = tile_m * k_split * (4 if layer == "router" else 2)
                if smem > 164 * 1024:
                    continue
                if (k // k_split) > state["partial"].shape[0]:
                    continue

                def sk(out=out, k_split=k_split, cols=cols, x=x, w_fast=w_fast):
                    return _run_sk(state, x, w_fast, out, k_split, cols)

                _consider(cands, f"sk-{cols}-{k_split}", sk, ref, probe, layer, flush)

    tc_ok = (
        mod is not None
        and layer == "router"
        and m in (128, 256)
        and n % 64 == 0
        and k % 256 == 0
    )
    if tc_ok:
        for k_split in _KSPLITS:
            if k % k_split != 0:
                continue
            if (k // k_split) * m * n > state["tc_partial"].numel():
                continue

            def tc(out=out, k_split=k_split, x=x, hi=hi, lo=lo):
                return _run_tc(state, x, hi, lo, out, k_split)

            _consider(cands, f"tc-{k_split}", tc, ref, probe, layer, flush)

    dg = state["dg"]
    if dg is not None and layer != "router":
        def dg_fn(out=out, acc=acc, x=x, w_fast=w_fast):
            dg.bf16_gemm_nt(x, w_fast, acc)
            out.copy_(acc)
            return out
        _consider(cands, "dg", dg_fn, ref, probe, layer, flush)
        def lt_fn(out=out, x=x, w_fast=w_fast):
            dg.cublaslt_gemm_nt(x, w_fast, out)
            return out
        _consider(cands, "lt", lt_fn, ref, probe, layer, flush)
    if dg is not None and layer == "router":
        def dg_r(acc=acc, acc2=acc2, x=x, hi=hi, lo=lo):
            acc.zero_()
            acc2.zero_()
            dg.bf16_gemm_nt(x, hi, acc)
            dg.bf16_gemm_nt(x, lo, acc2)
            acc.add_(acc2)
            return acc
        _consider(cands, "dg", dg_r, ref, probe, layer, flush)

    if not cands:
        state["plan"][(layer, m)] = "torch"
        print(f"plan {layer:10} m={m:<4} torch (no candidate)", flush=True)
        return
    cands.sort(key=lambda item: item[0])
    best_us, best_name = cands[0]
    torch_us = next((us for us, name in cands if name == "torch"), None)
    if best_name != "torch" and torch_us is not None and best_us > torch_us * 0.97:
        best_name = "torch"
        best_us = torch_us
    state["plan"][(layer, m)] = best_name
    if best_name != "torch":
        state["slot"][(layer, m)] = {
            "out": out, "acc": acc, "acc2": acc2, "w": w_fast, "hi": hi, "lo": lo,
        }
    torch_txt = "na" if torch_us is None else f"{torch_us:8.2f}"
    print(f"plan {layer:10} m={m:<4} {best_name:10} {best_us:8.2f} us  torch {torch_txt}", flush=True)


def prepare(weights: Weights):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    packed = {name: tensor.contiguous() for name, tensor in weights.items()}
    router = packed["router"]
    hi = router.to(torch.bfloat16).contiguous()
    lo = (router - hi.float()).to(torch.bfloat16).contiguous()
    state = {
        "weights": packed,
        "router": router,
        "hi": hi,
        "lo": lo,
        "hi_t": hi.t().contiguous(),
        "lo_t": lo.t().contiguous(),
        "plan": {},
        "slot": {},
        "mod": None,
        "partial": None,
        "tc_partial": None,
        "flush": None,
        "dg": None,
    }
    if not torch.cuda.is_available() or router.device.type != "cuda":
        return state
    device = router.device
    try:
        state["mod"] = _compile()
        state["mod"].init_kernels()
        state["partial"] = torch.empty((32, 32, 8192), dtype=torch.float32, device=device)
        state["tc_partial"] = torch.empty((32, 256, 256), dtype=torch.float32, device=device)
        state["flush"] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    except Exception as error:
        print(f"r3 compile failed: {type(error).__name__}: {error}", flush=True)
        state["mod"] = None
    try:
        import deep_gemm
        probe_a = torch.zeros(64, 128, dtype=torch.bfloat16, device=device)
        probe_b = torch.zeros(64, 128, dtype=torch.bfloat16, device=device)
        probe_d = torch.empty(64, 64, dtype=torch.float32, device=device)
        deep_gemm.bf16_gemm_nt(probe_a, probe_b, probe_d)
        torch.cuda.synchronize()
        state["dg"] = deep_gemm
    except Exception as error:
        print(f"deep_gemm disabled: {type(error).__name__}: {error}", flush=True)

    if state["flush"] is None:
        state["flush"] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    for _load_m, m in _SHAPES:
        for layer in _LAYERS:
            n = 1024 if layer == "unpooler" and m <= 8 else packed[layer].shape[0]
            k = packed[layer].shape[1]
            _bind(state, layer, m, n, k, device)
    print(
        "plan-table "
        + " ".join(
            f"{layer[0]}{m}:{state['plan'].get((layer, m), '-')}"
            for _load_m, m in _SHAPES
            for layer in _LAYERS
        ),
        flush=True,
    )
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    layer = cfg["layer"]
    m = cfg["m"]
    n = cfg["n"]
    x = data["x"]
    plan = state["plan"].get((layer, m), "torch")
    if plan == "torch" or (layer, m) not in state["slot"]:
        return _torch_out(x, state, layer, n)
    if x.shape[0] != m or x.shape[1] != cfg["k"]:
        return _torch_out(x, state, layer, n)
    slot = state["slot"][(layer, m)]
    if plan.startswith("sk-"):
        _cols, _k_split = plan.split("-")[1:]
        return _run_sk(state, x, slot["w"], slot["out"], int(_k_split), int(_cols))
    if plan.startswith("tc-"):
        return _run_tc(state, x, slot["hi"], slot["lo"], slot["out"], int(plan.split("-")[1]))
    if plan == "dg" and layer == "router":
        slot["acc"].zero_()
        slot["acc2"].zero_()
        state["dg"].bf16_gemm_nt(x, slot["hi"], slot["acc"])
        state["dg"].bf16_gemm_nt(x, slot["lo"], slot["acc2"])
        slot["acc"].add_(slot["acc2"])
        return slot["acc"]
    if plan == "dg":
        state["dg"].bf16_gemm_nt(x, slot["w"], slot["acc"])
        slot["out"].copy_(slot["acc"])
        return slot["out"]
    if plan == "lt":
        state["dg"].cublaslt_gemm_nt(x, slot["w"], slot["out"])
        return slot["out"]
    return _torch_out(x, state, layer, n)
