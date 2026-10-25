# okc: problem=muse_bf16_proj
"""Per-shape pick of DeepGEMM, cuBLASLt, or a split-K kernel, else cuBLAS.

prepare() times each backend on this GPU and keeps a winner only when it is
faster than torch and inside the reference tolerance. kernel_fn only replays
that choice. CPU runs stay on torch.
"""

from typing import TypedDict

import torch

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

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

__device__ __forceinline__ float4 ldcs4(const void* p) {
    return __ldcs(reinterpret_cast<const float4*>(p));
}

__device__ __forceinline__ float4 ldcg4(const void* p) {
    return __ldcg(reinterpret_cast<const float4*>(p));
}

// One warp, one output column, a K-split. X is reused from L2 across columns.
__global__ void __launch_bounds__(512)
splitk_kernel(const __nv_bfloat16* __restrict__ X,
              const __nv_bfloat16* __restrict__ W,
              const __nv_bfloat16* __restrict__ Lo,
              float* __restrict__ partial,
              int M, int N, int K, int splits, int ld_m, int ld_n) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int col = blockIdx.x * blockDim.y + warp;
    const int split = blockIdx.y;
    if (col >= N) return;

    const int chunk = K / splits;
    const int k0 = split * chunk;
    float acc[32];
#pragma unroll
    for (int m = 0; m < 32; ++m) acc[m] = 0.f;

    const __nv_bfloat16* wrow = W + static_cast<size_t>(col) * K;
    const __nv_bfloat16* lrow = Lo ? Lo + static_cast<size_t>(col) * K : nullptr;

    for (int k = k0 + lane * 8; k < k0 + chunk; k += 256) {
        float4 wv = ldcs4(wrow + k);
        float4 lv;
        if (lrow) lv = ldcs4(lrow + k);
#pragma unroll 1
        for (int m = 0; m < M; ++m) {
            float4 xv = ldcg4(X + static_cast<size_t>(m) * K + k);
            float d = dot8(xv, wv);
            if (lrow) d += dot8(xv, lv);
            acc[m] += d;
        }
    }

#pragma unroll 1
    for (int m = 0; m < M; ++m) {
        float v = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0)
            partial[(static_cast<size_t>(split) * ld_m + m) * ld_n + col] = v;
    }
}

__global__ void reduce_kernel(const float* __restrict__ partial,
                              void* __restrict__ out,
                              int M, int N, int splits, int ld_m, int ld_n,
                              int store_fp32) {
    const int total = M * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int m = idx / N;
        const int n = idx - m * N;
        float s = 0.f;
        for (int sp = 0; sp < splits; ++sp)
            s += partial[(static_cast<size_t>(sp) * ld_m + m) * ld_n + n];
        if (store_fp32)
            static_cast<float*>(out)[idx] = s;
        else
            static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(s);
    }
}

void splitk(torch::Tensor x, torch::Tensor w, torch::Tensor lo,
            torch::Tensor partial, torch::Tensor out,
            int64_t splits, int64_t warps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(partial.is_contiguous() && partial.scalar_type() == torch::kFloat);
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);
    TORCH_CHECK(M > 0 && M <= 32);
    TORCH_CHECK(w.size(1) == K);
    TORCH_CHECK(out.size(0) == M && out.size(1) == N);
    TORCH_CHECK(splits > 0 && K % (splits * 256) == 0);
    TORCH_CHECK(partial.size(0) >= splits && partial.size(1) >= M && partial.size(2) >= N);
    const bool has_lo = lo.defined() && lo.numel() > 0;
    if (has_lo) {
        TORCH_CHECK(lo.is_contiguous() && lo.size(0) == N && lo.size(1) == K);
    }
    const int store_fp32 = out.scalar_type() == torch::kFloat;
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 block(32, warps);
    dim3 grid((N + warps - 1) / warps, splits);
    splitk_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        has_lo ? reinterpret_cast<const __nv_bfloat16*>(lo.data_ptr()) : nullptr,
        partial.data_ptr<float>(),
        M, N, K, static_cast<int>(splits),
        static_cast<int>(partial.size(1)), static_cast<int>(partial.size(2)));
    const int total = M * N;
    const int threads = 256;
    const int blocks = std::min((total + threads - 1) / threads, 2048);
    reduce_kernel<<<blocks, threads, 0, stream>>>(
        partial.data_ptr<float>(), out.data_ptr(),
        M, N, static_cast<int>(splits),
        static_cast<int>(partial.size(1)), static_cast<int>(partial.size(2)),
        store_fp32);
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


# Scored shapes. load is None for cases we only need to answer correctly.
_SHAPES = (
    ("decode 1", 1),
    ("decode 16", 16),
    ("decode 32", 32),
    ("verify 1", 8),
    ("verify 16", 128),
    ("verify 32", 256),
)
_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_EMPTY = None  # filled with a 0-size tensor on first CUDA prepare


def _valid_splits(k: int) -> list[int]:
    return [s for s in (2, 4, 8, 16, 32) if k % (s * 256) == 0]


def _torch_out(x, state, layer, n):
    if layer == "router":
        if x.shape[0] <= 8:
            return x.float() @ state["router"][:n].T
        return torch.mm(x, state["hi_t"], out_dtype=torch.float32) + torch.mm(
            x, state["lo_t"], out_dtype=torch.float32
        )
    return x @ state["weights"][layer][:n].T


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref.double()).norm() / ref.double().norm().clamp(min=1e-12))


def _accurate(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, layer: str) -> bool:
    ref = x.double() @ w.double().T
    tol = 1e-4 if layer == "router" else 5e-3
    if _rel(out, ref) > tol:
        return False
    # One-hot power-of-two row: the output must be that weight column.
    probe = (w[:, 7].double() * 32.0)
    return _rel(out[0], probe) <= 1e-3


def _bench(fn, iters: int = 8) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def _compile():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="proj_splitk_r2",
        cpp_sources=["#include <torch/extension.h>\nvoid splitk(torch::Tensor x, torch::Tensor w, torch::Tensor lo, torch::Tensor partial, torch::Tensor out, int64_t splits, int64_t warps);"],
        cuda_sources=[_CUDA],
        functions=["splitk"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def _run_splitk(state, x, w, lo, out, splits, warps):
    partial = state["partial"]
    mod = state["mod"]
    empty = state["empty"]
    m = x.shape[0]
    if m <= 32:
        mod.splitk(x, w, lo if lo is not None else empty, partial, out, int(splits), int(warps))
        return out
    for row in range(0, m, 32):
        rows = min(32, m - row)
        mod.splitk(
            x[row:row + rows], w, lo if lo is not None else empty, partial,
            out[row:row + rows], int(splits), int(warps),
        )
    return out


def _consider(cands, name, fn, x, w, layer):
    try:
        out = fn()
        torch.cuda.synchronize()
        if not _accurate(out, x, w, layer):
            return
        us = _bench(fn)
        cands.append((us, name, fn))
    except Exception as error:
        print(f"skip {layer} {tuple(x.shape)} {name}: {type(error).__name__}: {error}", flush=True)


def prepare(weights: Weights):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    packed = {name: tensor.contiguous() for name, tensor in weights.items()}
    router = packed["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state = {
        "weights": packed,
        "router": router,
        "hi": hi.contiguous(),
        "lo": lo.contiguous(),
        "hi_t": hi.t().contiguous(),
        "lo_t": lo.t().contiguous(),
        "plan": {},
        "slot": {},
        "mod": None,
        "partial": None,
        "empty": None,
        "dg": None,
    }
    if not torch.cuda.is_available():
        return state

    device = router.device
    state["empty"] = torch.empty(0, dtype=torch.bfloat16, device=device)
    try:
        state["mod"] = _compile()
        state["partial"] = torch.empty((32, 32, 8192), dtype=torch.float32, device=device)
    except Exception as error:
        print(f"splitk compile failed: {type(error).__name__}: {error}", flush=True)
        state["mod"] = None
    try:
        import deep_gemm
        state["dg"] = deep_gemm
    except Exception as error:
        print(f"deep_gemm unavailable: {type(error).__name__}: {error}", flush=True)

    # Touch every scored shape once so the dispatch is frozen before capture.
    for _load, m in _SHAPES:
        for layer in _LAYERS:
            n = 1024 if layer == "unpooler" and m <= 8 else packed[layer].shape[0]
            k = packed[layer].shape[1]
            _bind_shape(state, layer, m, n, k, device)
    _log_plan(state)
    return state


def _bind_shape(state, layer, m, n, k, device):
    x = torch.randn(m, k, device=device, dtype=torch.float32)
    x[0].zero_()
    x[0, 7] = 32.0
    x = x.to(torch.bfloat16)
    if layer == "router":
        w_ref = state["router"][:n]
        w_b = state["hi"][:n]
        lo = state["lo"][:n]
        out_dtype = torch.float32
    else:
        w_ref = state["weights"][layer][:n]
        w_b = w_ref
        lo = None
        out_dtype = torch.bfloat16
    out = torch.empty(m, n, dtype=out_dtype, device=device)
    acc = torch.empty(m, n, dtype=torch.float32, device=device)
    acc2 = torch.empty(m, n, dtype=torch.float32, device=device)
    cands = []

    def torch_fn():
        return _torch_out(x, state, layer, n)

    _consider(cands, "torch", torch_fn, x, w_ref, layer)

    dg = state["dg"]
    if dg is not None and layer != "router":
        def dg_fn(out=out, acc=acc, w_b=w_b, x=x):
            dg.bf16_gemm_nt(x, w_b, acc)
            out.copy_(acc)
            return out
        _consider(cands, "dg", dg_fn, x, w_ref, layer)
        def lt_fn(out=out, w_b=w_b, x=x):
            dg.cublaslt_gemm_nt(x, w_b, out)
            return out
        _consider(cands, "lt", lt_fn, x, w_ref, layer)
    if dg is not None and layer == "router":
        def dg_r(acc=acc, acc2=acc2, x=x, w_b=w_b, lo=lo):
            dg.bf16_gemm_nt(x, w_b, acc)
            dg.bf16_gemm_nt(x, lo, acc2)
            acc.add_(acc2)
            return acc
        _consider(cands, "dg_router", dg_r, x, w_ref, layer)

    if state["mod"] is not None and _valid_splits(k):
        for splits in _valid_splits(k):
            for warps in (8, 16):
                def sk(out=out, splits=splits, warps=warps, x=x, w_b=w_b, lo=lo):
                    return _run_splitk(state, x, w_b, lo, out, splits, warps)
                _consider(cands, f"sk-{splits}-{warps}", sk, x, w_ref, layer)

    cands.sort(key=lambda item: item[0])
    best_us, best_name, _best_fn = cands[0]
    torch_us = next(us for us, name, _fn in cands if name == "torch")
    # Keep a custom kernel only when it is clearly ahead of cuBLAS / the hi-lo path.
    if best_name != "torch" and best_us > torch_us * 0.97:
        best_name = "torch"
    state["plan"][(layer, m)] = best_name
    state["slot"][(layer, m)] = {"out": out, "acc": acc, "acc2": acc2, "w": w_b, "lo": lo}
    print(f"plan {layer:10} m={m:<4} {best_name:14} {best_us:8.2f} us  torch {torch_us:8.2f}", flush=True)


def _log_plan(state):
    print("plan-table " + " ".join(f"{layer[0]}{m}:{state['plan'].get((layer, m), '-')}" for _load, m in _SHAPES for layer in _LAYERS), flush=True)


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    layer = cfg["layer"]
    m = cfg["m"]
    n = cfg["n"]
    x = data["x"]
    plan = state["plan"].get((layer, m), "torch")
    if plan == "torch" or (layer, m) not in state["slot"]:
        return _torch_out(x, state, layer, n)
    slot = state["slot"][(layer, m)]
    out = slot["out"]
    w = slot["w"]
    if plan == "dg":
        state["dg"].bf16_gemm_nt(x, w, slot["acc"])
        out.copy_(slot["acc"])
        return out
    if plan == "lt":
        state["dg"].cublaslt_gemm_nt(x, w, out)
        return out
    if plan == "dg_router":
        state["dg"].bf16_gemm_nt(x, w, slot["acc"])
        state["dg"].bf16_gemm_nt(x, slot["lo"], slot["acc2"])
        slot["acc"].add_(slot["acc2"])
        return slot["acc"]
    if plan.startswith("sk-"):
        _splits, _warps = plan.split("-")[1:]
        return _run_splitk(state, x, w, slot["lo"], out, int(_splits), int(_warps))
    return _torch_out(x, state, layer, n)
