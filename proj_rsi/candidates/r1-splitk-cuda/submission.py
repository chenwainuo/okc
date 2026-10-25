# okc: problem=muse_bf16_proj
"""Split-K GEMM for small-M projections. CPU and large M use the starter matmul."""

import torch

# Keep in sync with the CUDA launchers. N-tile is 8 warps * 8 rows.
MAX_M = 32
MAX_N = 8192
MAX_SPLITS = 64
N_TILE = 64
TARGET_CTAS = 2 * 132
SMEM_CAP = 192 * 1024

_MOD = None

_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" {
int muse_init_kernels();
void muse_launch_bf16(
    const void* x, const void* w, float* partials, void* out,
    int M, int N, int K, int splits, int k_split, int max_m, int max_n, void* stream);
void muse_launch_f32(
    const void* x, const void* w, float* partials, void* out,
    int M, int N, int K, int splits, int k_split, int max_m, int max_n, void* stream);
}

void init_kernels() {
  int rc = muse_init_kernels();
  TORCH_CHECK(rc == 0, "split-k kernel attribute setup failed: ", rc);
}

void launch_bf16(
    at::Tensor x, at::Tensor w, at::Tensor partials, at::Tensor out,
    int64_t M, int64_t N, int64_t K, int64_t splits, int64_t k_split,
    int64_t max_m, int64_t max_n) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && partials.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(partials.scalar_type() == at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(M >= 1 && M <= max_m && M <= 32);
  TORCH_CHECK(N >= 64 && (N % 64) == 0 && N <= max_n);
  TORCH_CHECK(K >= 8 && splits >= 1 && k_split >= 8 && (k_split % 8) == 0);
  TORCH_CHECK(splits * k_split == K);
  TORCH_CHECK(x.size(0) == M && x.size(1) == K);
  TORCH_CHECK(w.size(0) == N && w.size(1) == K);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(partials.size(0) >= splits);
  TORCH_CHECK(partials.size(1) == max_m && partials.size(2) == max_n);
  const int64_t smem = M * k_split * static_cast<int64_t>(2);
  TORCH_CHECK(smem > 0 && smem <= 192 * 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  muse_launch_bf16(
      x.data_ptr(), w.data_ptr(), partials.data_ptr<float>(), out.data_ptr(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(splits), static_cast<int>(k_split),
      static_cast<int>(max_m), static_cast<int>(max_n),
      static_cast<void*>(stream));
}

void launch_f32(
    at::Tensor x, at::Tensor w, at::Tensor partials, at::Tensor out,
    int64_t M, int64_t N, int64_t K, int64_t splits, int64_t k_split,
    int64_t max_m, int64_t max_n) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && partials.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w.scalar_type() == at::kFloat);
  TORCH_CHECK(partials.scalar_type() == at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kFloat);
  TORCH_CHECK(M >= 1 && M <= max_m && M <= 32);
  TORCH_CHECK(N >= 64 && (N % 64) == 0 && N <= max_n);
  TORCH_CHECK(K >= 8 && splits >= 1 && k_split >= 8 && (k_split % 8) == 0);
  TORCH_CHECK(splits * k_split == K);
  TORCH_CHECK(x.size(0) == M && x.size(1) == K);
  TORCH_CHECK(w.size(0) == N && w.size(1) == K);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(partials.size(0) >= splits);
  TORCH_CHECK(partials.size(1) == max_m && partials.size(2) == max_n);
  const int64_t smem = M * k_split * static_cast<int64_t>(4);
  TORCH_CHECK(smem > 0 && smem <= 192 * 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  muse_launch_f32(
      x.data_ptr(), w.data_ptr(), partials.data_ptr<float>(), out.data_ptr(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(splits), static_cast<int>(k_split),
      static_cast<int>(max_m), static_cast<int>(max_n),
      static_cast<void*>(stream));
}
"""

_CU = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Split-K: blockIdx.x is the N-tile (64 rows), blockIdx.y is the K-split.
// Each used partial is stored exactly once; the reduce kernel consumes those stores.

extern __shared__ __align__(16) char muse_dyn_smem[];

static constexpr int kMaxAcc = 32;

__device__ __forceinline__ void muse_fma_bf16(float& acc, float4 wv, float4 xv) {
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 wf = __bfloat1622float2(w2[i]);
    float2 xf = __bfloat1622float2(x2[i]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }
}

__device__ __forceinline__ void muse_fma_f32(float& acc, float4 wv, float4 xv) {
  acc = fmaf(wv.x, xv.x, acc);
  acc = fmaf(wv.y, xv.y, acc);
  acc = fmaf(wv.z, xv.z, acc);
  acc = fmaf(wv.w, xv.w, acc);
}

__device__ __forceinline__ float muse_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
  }
  return v;
}

template <int M_FIXED>
__device__ __forceinline__ void muse_apply_bf16(
    float* acc, const __nv_bfloat16* xs, const __nv_bfloat16* wrow, int M, int k_split, int kk) {
  float4 wv = __ldcs(reinterpret_cast<const float4*>(wrow + kk));
  if constexpr (M_FIXED > 0) {
#pragma unroll
    for (int m = 0; m < M_FIXED; ++m) {
      float4 xv = *reinterpret_cast<const float4*>(
          xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kk)));
      muse_fma_bf16(acc[m], wv, xv);
    }
  } else {
    for (int m = 0; m < M; ++m) {
      float4 xv = *reinterpret_cast<const float4*>(
          xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kk)));
      muse_fma_bf16(acc[m], wv, xv);
    }
  }
}

template <int M_FIXED>
__device__ __forceinline__ void muse_apply_f32(
    float* acc, const float* xs, const float* wrow, int M, int k_split, int kk) {
  float4 wv = __ldcs(reinterpret_cast<const float4*>(wrow + kk));
  if constexpr (M_FIXED > 0) {
#pragma unroll
    for (int m = 0; m < M_FIXED; ++m) {
      float4 xv = *reinterpret_cast<const float4*>(
          xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kk)));
      muse_fma_f32(acc[m], wv, xv);
    }
  } else {
    for (int m = 0; m < M; ++m) {
      float4 xv = *reinterpret_cast<const float4*>(
          xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kk)));
      muse_fma_f32(acc[m], wv, xv);
    }
  }
}

template <int M_FIXED>
__device__ __forceinline__ void muse_store_acc(
    float* acc, float* partials, int M, int split, int nrow, int lane, int max_m, int max_n) {
  const int nstore = M_FIXED > 0 ? M_FIXED : M;
  if constexpr (M_FIXED > 0) {
#pragma unroll
    for (int m = 0; m < M_FIXED; ++m) {
      float v = muse_warp_sum(acc[m]);
      if (lane == 0) {
        partials[(static_cast<size_t>(split) * static_cast<size_t>(max_m) + static_cast<size_t>(m)) *
                     static_cast<size_t>(max_n) +
                 static_cast<size_t>(nrow)] = v;
      }
    }
  } else {
    for (int m = 0; m < nstore; ++m) {
      float v = muse_warp_sum(acc[m]);
      if (lane == 0) {
        partials[(static_cast<size_t>(split) * static_cast<size_t>(max_m) + static_cast<size_t>(m)) *
                     static_cast<size_t>(max_n) +
                 static_cast<size_t>(nrow)] = v;
      }
    }
  }
}

template <int M_FIXED>
__global__ void __launch_bounds__(256)
muse_splitk_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ partials,
    int M, int N, int K, int k_split, int max_m, int max_n) {
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(muse_dyn_smem);
  const int split = static_cast<int>(blockIdx.y);
  const int n_tile = static_cast<int>(blockIdx.x);
  const int k_begin = split * k_split;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int span8 = k_split >> 3;
  const int vecs = M * span8;
  for (int v = tid; v < vecs; v += 256) {
    const int m = v / span8;
    const int kv = v - m * span8;
    const __nv_bfloat16* src =
        x + (static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k_begin) +
             static_cast<size_t>(kv) * 8);
    __nv_bfloat16* dst =
        xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kv) * 8);
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
  }
  __syncthreads();

  if (warp < 8) {
    const int row0 = n_tile * 64 + warp * 8;
#pragma unroll 1
    for (int r = 0; r < 8; ++r) {
      const int nrow = row0 + r;
      if (nrow < N) {
        const __nv_bfloat16* wrow =
            w + (static_cast<size_t>(nrow) * static_cast<size_t>(K) + static_cast<size_t>(k_begin));
        constexpr int ACCN = M_FIXED > 0 ? M_FIXED : kMaxAcc;
        float acc[ACCN];
#pragma unroll
        for (int i = 0; i < ACCN; ++i) {
          acc[i] = 0.f;
        }
        constexpr int VEC = 8;
        constexpr int SPAN = 256;
        if ((k_split & (SPAN - 1)) == 0) {
#pragma unroll 1
          for (int kk = lane * VEC; kk < k_split; kk += SPAN) {
            muse_apply_bf16<M_FIXED>(acc, xs, wrow, M, k_split, kk);
          }
        } else {
          const int n_iter = (k_split + SPAN - 1) / SPAN;
#pragma unroll 1
          for (int it = 0; it < n_iter; ++it) {
            const int kk = it * SPAN + lane * VEC;
            if (kk < k_split) {
              muse_apply_bf16<M_FIXED>(acc, xs, wrow, M, k_split, kk);
            }
          }
        }
        muse_store_acc<M_FIXED>(acc, partials, M, split, nrow, lane, max_m, max_n);
      }
    }
  }
}

template <int M_FIXED>
__global__ void __launch_bounds__(256)
muse_splitk_f32(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ w, float* __restrict__ partials,
    int M, int N, int K, int k_split, int max_m, int max_n) {
  float* xs = reinterpret_cast<float*>(muse_dyn_smem);
  const int split = static_cast<int>(blockIdx.y);
  const int n_tile = static_cast<int>(blockIdx.x);
  const int k_begin = split * k_split;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Stage BF16 activations as FP32 so the router keeps its FP32 weights.
  const int span8 = k_split >> 3;
  const int vecs = M * span8;
  for (int v = tid; v < vecs; v += 256) {
    const int m = v / span8;
    const int kv = v - m * span8;
    const __nv_bfloat16* src =
        x + (static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k_begin) +
             static_cast<size_t>(kv) * 8);
    float4 packed = *reinterpret_cast<const float4*>(src);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&packed);
    float2 a = __bfloat1622float2(p[0]);
    float2 b = __bfloat1622float2(p[1]);
    float2 c = __bfloat1622float2(p[2]);
    float2 d = __bfloat1622float2(p[3]);
    float* dst = xs + (static_cast<size_t>(m) * static_cast<size_t>(k_split) + static_cast<size_t>(kv) * 8);
    *reinterpret_cast<float4*>(dst) = make_float4(a.x, a.y, b.x, b.y);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(c.x, c.y, d.x, d.y);
  }
  __syncthreads();

  if (warp < 8) {
    const int row0 = n_tile * 64 + warp * 8;
#pragma unroll 1
    for (int r = 0; r < 8; ++r) {
      const int nrow = row0 + r;
      if (nrow < N) {
        const float* wrow =
            w + (static_cast<size_t>(nrow) * static_cast<size_t>(K) + static_cast<size_t>(k_begin));
        constexpr int ACCN = M_FIXED > 0 ? M_FIXED : kMaxAcc;
        float acc[ACCN];
#pragma unroll
        for (int i = 0; i < ACCN; ++i) {
          acc[i] = 0.f;
        }
        constexpr int VEC = 4;
        constexpr int SPAN = 128;
        if ((k_split & (SPAN - 1)) == 0) {
#pragma unroll 1
          for (int kk = lane * VEC; kk < k_split; kk += SPAN) {
            muse_apply_f32<M_FIXED>(acc, xs, wrow, M, k_split, kk);
          }
        } else {
          const int n_iter = (k_split + SPAN - 1) / SPAN;
#pragma unroll 1
          for (int it = 0; it < n_iter; ++it) {
            const int kk = it * SPAN + lane * VEC;
            if (kk < k_split) {
              muse_apply_f32<M_FIXED>(acc, xs, wrow, M, k_split, kk);
            }
          }
        }
        muse_store_acc<M_FIXED>(acc, partials, M, split, nrow, lane, max_m, max_n);
      }
    }
  }
}

__global__ void muse_reduce_bf16(
    const float* __restrict__ partials, __nv_bfloat16* __restrict__ out,
    int M, int N, int splits, int max_m, int max_n) {
  const int total = M * N;
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx >= total) {
    return;
  }
  const int m = idx / N;
  const int n = idx - m * N;
  const size_t stride = static_cast<size_t>(max_m) * static_cast<size_t>(max_n);
  size_t base = static_cast<size_t>(m) * static_cast<size_t>(max_n) + static_cast<size_t>(n);
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) {
    sum += partials[base + static_cast<size_t>(s) * stride];
  }
  out[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = __float2bfloat16(sum);
}

__global__ void muse_reduce_f32(
    const float* __restrict__ partials, float* __restrict__ out,
    int M, int N, int splits, int max_m, int max_n) {
  const int total = M * N;
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx >= total) {
    return;
  }
  const int m = idx / N;
  const int n = idx - m * N;
  const size_t stride = static_cast<size_t>(max_m) * static_cast<size_t>(max_n);
  size_t base = static_cast<size_t>(m) * static_cast<size_t>(max_n) + static_cast<size_t>(n);
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) {
    sum += partials[base + static_cast<size_t>(s) * stride];
  }
  out[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
}

static int muse_set_attr(const void* fn) {
  cudaError_t err = cudaFuncSetAttribute(
      fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 192 * 1024);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int muse_init_kernels() {
  const void* fns[] = {
      (const void*)muse_splitk_bf16<0>, (const void*)muse_splitk_bf16<1>,
      (const void*)muse_splitk_bf16<8>, (const void*)muse_splitk_bf16<16>,
      (const void*)muse_splitk_bf16<32>, (const void*)muse_splitk_f32<0>,
      (const void*)muse_splitk_f32<1>, (const void*)muse_splitk_f32<8>,
      (const void*)muse_splitk_f32<16>, (const void*)muse_splitk_f32<32>,
  };
  for (const void* fn : fns) {
    int rc = muse_set_attr(fn);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

extern "C" void muse_launch_bf16(
    const void* x, const void* w, float* partials, void* out,
    int M, int N, int K, int splits, int k_split, int max_m, int max_n, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int smem = M * k_split * static_cast<int>(sizeof(__nv_bfloat16));
  dim3 grid(N / 64, splits);
  dim3 block(256);
  const __nv_bfloat16* x_p = static_cast<const __nv_bfloat16*>(x);
  const __nv_bfloat16* w_p = static_cast<const __nv_bfloat16*>(w);
  __nv_bfloat16* o_p = static_cast<__nv_bfloat16*>(out);
  if (M == 1) {
    muse_splitk_bf16<1><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 8) {
    muse_splitk_bf16<8><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 16) {
    muse_splitk_bf16<16><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 32) {
    muse_splitk_bf16<32><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else {
    muse_splitk_bf16<0><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  }
  const int total = M * N;
  const int threads = 256;
  const int rblocks = (total + threads - 1) / threads;
  muse_reduce_bf16<<<rblocks, threads, 0, s>>>(partials, o_p, M, N, splits, max_m, max_n);
}

extern "C" void muse_launch_f32(
    const void* x, const void* w, float* partials, void* out,
    int M, int N, int K, int splits, int k_split, int max_m, int max_n, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int smem = M * k_split * static_cast<int>(sizeof(float));
  dim3 grid(N / 64, splits);
  dim3 block(256);
  const __nv_bfloat16* x_p = static_cast<const __nv_bfloat16*>(x);
  const float* w_p = static_cast<const float*>(w);
  float* o_p = static_cast<float*>(out);
  if (M == 1) {
    muse_splitk_f32<1><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 8) {
    muse_splitk_f32<8><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 16) {
    muse_splitk_f32<16><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else if (M == 32) {
    muse_splitk_f32<32><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  } else {
    muse_splitk_f32<0><<<grid, block, smem, s>>>(x_p, w_p, partials, M, N, K, k_split, max_m, max_n);
  }
  const int total = M * N;
  const int threads = 256;
  const int rblocks = (total + threads - 1) / threads;
  muse_reduce_f32<<<rblocks, threads, 0, s>>>(partials, o_p, M, N, splits, max_m, max_n);
}
"""


def _get_module():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline

        _MOD = load_inline(
            name="muse_r1_splitk_cuda_v1",
            cpp_sources=_CPP,
            cuda_sources=_CU,
            functions=["init_kernels", "launch_bf16", "launch_f32"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            with_cuda=True,
            verbose=False,
        )
    return _MOD


def _choose_k_split(n, k, m, fp32):
    """Pick a K-split so n/64 * splits is near 2*132 and the activation tile fits in smem."""
    n_tiles = n // N_TILE
    if n_tiles < 1 or m < 1 or k < 8:
        return None
    elem_bytes = 4 if fp32 else 2
    warp_span = 128 if fp32 else 256
    best = None
    best_key = None
    ks = 8
    while ks <= k:
        if k % ks == 0:
            splits = k // ks
            smem = m * ks * elem_bytes
            if 1 <= splits <= MAX_SPLITS and 0 < smem <= SMEM_CAP:
                blocks = n_tiles * splits
                aligned = 0 if ks % warp_span == 0 else 1
                key = (abs(blocks - TARGET_CTAS), aligned)
                if best_key is None or key < best_key:
                    best_key = key
                    best = ks
        ks += 8
    return best


def _starter(x, w):
    if w.dtype == torch.float32:
        return x.float() @ w.T
    return x @ w.T


def _can_fast(x, w, cfg, state):
    mod = state.get("mod")
    partials = state.get("partials")
    if mod is None or partials is None:
        return False
    m = cfg["m"]
    n = cfg["n"]
    k = cfg["k"]
    if not isinstance(m, int) or not isinstance(n, int) or not isinstance(k, int):
        return False
    if m <= 0 or m > MAX_M or n <= 0 or k <= 0:
        return False
    if n % N_TILE != 0 or k % N_TILE != 0 or n > MAX_N:
        return False
    if x.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
        return False
    if (not x.is_cuda) or (not w.is_cuda) or x.device != w.device or partials.device != x.device:
        return False
    if (not x.is_contiguous()) or (not w.is_contiguous()) or (not partials.is_contiguous()):
        return False
    if x.dim() != 2 or w.dim() != 2:
        return False
    if x.shape[0] != m or x.shape[1] != k or w.shape[0] != n or w.shape[1] != k:
        return False
    if partials.shape[0] < 1 or partials.shape[1] != MAX_M or partials.shape[2] != MAX_N:
        return False
    return True


def prepare(weights):
    state = {}
    for name, tensor in weights.items():
        state[name] = tensor if tensor.is_contiguous() else tensor.contiguous()
    state["mod"] = None
    state["partials"] = None
    if not torch.cuda.is_available():
        return state
    mod = _get_module()
    state["mod"] = mod
    device = None
    for tensor in weights.values():
        if torch.is_tensor(tensor) and tensor.is_cuda:
            device = tensor.device
            break
    if device is None:
        device = torch.device("cuda")
    # [split, row, col]. Epilogue overwrites every used entry; no memset on the replay path.
    state["partials"] = torch.empty((MAX_SPLITS, MAX_M, MAX_N), dtype=torch.float32, device=device)
    mod.init_kernels()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["x"]
    w = state[cfg["layer"]][: cfg["n"]]
    if not _can_fast(x, w, cfg, state):
        return _starter(x, w)
    m = cfg["m"]
    n = cfg["n"]
    k = cfg["k"]
    fp32 = w.dtype == torch.float32
    k_split = _choose_k_split(n, k, m, fp32)
    if k_split is None or k_split % 8 != 0:
        return _starter(x, w)
    splits = k // k_split
    smem = m * k_split * (4 if fp32 else 2)
    if splits < 1 or splits > MAX_SPLITS or smem <= 0 or smem > SMEM_CAP or splits * k_split != k:
        return _starter(x, w)
    partials = state["partials"]
    if fp32:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        state["mod"].launch_f32(x, w, partials, out, m, n, k, splits, k_split, MAX_M, MAX_N)
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        state["mod"].launch_bf16(x, w, partials, out, m, n, k, splits, k_split, MAX_M, MAX_N)
    return out
