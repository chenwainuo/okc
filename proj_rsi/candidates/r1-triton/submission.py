# okc: problem=muse_bf16_proj
"""Triton BF16 GEMM for Muse projections. Torch fallback if CUDA/Triton is unavailable.

out = x[m, k] @ W[n, k].T. Router weights are a BF16 hi/lo pair accumulated in FP32.
Tile configs are a fixed table: no autotune on the replay path.
"""

from typing import TypedDict

import torch


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")

# Host cap. Larger m, or a split-K tile that does not fit the workspace, uses torch.
# Pointer offsets for the scored/test shapes stay well inside int32.
MAX_M = 4096
MAX_N = 8192

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, SPLIT_K)
# Keyed by m-bucket (0: m<=32, 1: m<=64, 2: else) and whether K>=4096.
# SPLIT_K>1 for m<=32 so cdiv(N, BLOCK_N)*SPLIT_K sits near 132 SMs on the wide
# layers (unpooler slice N=1024 -> 128 CTAs; qkv N=2304 -> 288).
# m>=128 is bucket 2: SPLIT_K=1 and tensor-core tl.dot tiles. BLOCK_M=64 (not 128)
# keeps M=128 at grid_m=2 so qkv/pooler still occupy the SMs.
# Bucket 0 also uses BLOCK_M=64 so Hopper WGMMA can pipeline; M<=32 is masked.
_TILES = {
    (0, True): (64, 64, 64, 4, 3, 8),
    (0, False): (64, 128, 64, 8, 3, 4),
    (1, True): (64, 64, 64, 4, 3, 4),
    (1, False): (64, 128, 64, 8, 3, 2),
    (2, True): (64, 64, 64, 4, 3, 1),
    (2, False): (64, 128, 64, 8, 3, 1),
}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tile_for(m: int, k: int):
    bucket = 0 if m <= 32 else (1 if m <= 64 else 2)
    return _TILES[(bucket, k >= 4096)]


def _max_partial_elems() -> int:
    best = 0
    for (bucket, _k_large), (bm, bn, _bk, _w, _s, split) in _TILES.items():
        if split <= 1:
            continue
        max_m = 32 if bucket == 0 else 64
        gm = _cdiv(max_m, bm)
        gn = _cdiv(MAX_N, bn)
        best = max(best, split * gm * gn * bm * bn)
    return best


_MAX_PARTIAL = _max_partial_elems()


def _shape_ok(m, n, k, n_full: int, partial_elems: int) -> bool:
    """True when the hardcoded tile covers this shape.

    N must be a multiple of BLOCK_N so the K-loop can load B with no mask.
    Odd M is masked. Uneven K splits and oversized partials fall back.
    """
    if type(m) is not int or type(n) is not int or type(k) is not int:
        return False
    if m < 1 or n < 1 or k < 1 or m > MAX_M or n > n_full:
        return False
    if m * k > 2**30 or n * k > 2**30 or m * n > 2**30:
        return False
    bm, bn, bk, _w, _s, split = _tile_for(m, k)
    if n % bn != 0 or k % (split * bk) != 0:
        return False
    if split > 1:
        need = split * _cdiv(m, bm) * _cdiv(n, bn) * bm * bn
        if need > partial_elems:
            return False
    return True


def _fallback(x: torch.Tensor, state, layer: str, n: int) -> torch.Tensor:
    """Starter math: BF16 matmul, router in FP32."""
    w = state[layer][:n]
    if w.dtype == torch.float32:
        return x.float() @ w.T
    return x @ w.T


def _warmup_cases():
    """Scored shapes plus the official odd-m cases, so prepare compiles every replay config."""
    shape = {
        "qkv_gate": (2304, 8192),
        "o": (8192, 1024),
        "pooler": (4096, 8192),
        "unpooler": (8192, 4096),
        "router": (256, 8192),
    }
    cases = []
    for m in (1, 8, 16, 32, 128, 256):
        for layer in _LAYERS:
            n, k = shape[layer]
            if layer == "unpooler" and m <= 8:
                n = 1024
            cases.append((m, layer, n, k))
    extra = (
        (5, "unpooler"),
        (7, "pooler"),
        (9, "router"),
        (33, "o"),
        (33, "pooler"),
        (33, "router"),
        (64, "o"),
        (64, "qkv_gate"),
        (130, "unpooler"),
        (512, "qkv_gate"),
        (1000, "unpooler"),
    )
    for m, layer in extra:
        n, k = shape[layer]
        if layer == "unpooler" and m <= 8:
            n = 1024
        cases.append((m, layer, n, k))
    return cases


def _build_kernels(triton_mod):
    """JIT kernels. `tl` is a module global set by prepare before this runs.

    Nested defs keep the source in this file so Triton can inspect.getsource it.
    `tl` is not a closure cell; the compiler resolves it from module globals.
    """

    @triton_mod.jit
    def gemm_kernel(
        a_ptr, b_ptr, blo_ptr, c_ptr, p_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        k_per, k_tiles, grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        IS_ROUTER: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        # Host only launches when N % BLOCK_N == 0 and K splits evenly, so B/K are in range.
        # Clamp masked-off M rows onto row 0 so the address itself is in range.
        offs_m_l = tl.where(mask_m, offs_m, 0)
        k_start = pid_k * k_per
        a_ptrs = a_ptr + offs_m_l[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        if IS_ROUTER:
            blo_ptrs = blo_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ki in range(k_tiles):
            # Cast back to bf16 so a masked load cannot promote the dot into TF32.
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
            b = tl.load(b_ptrs).to(tl.bfloat16)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            if IS_ROUTER:
                blo = tl.load(blo_ptrs).to(tl.bfloat16)
                acc = tl.dot(a, blo, acc, out_dtype=tl.float32)
                blo_ptrs += BLOCK_K * stride_bk
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        mask_n = offs_n < N
        if SPLIT_K == 1:
            offs_m_s = tl.where(mask_m, offs_m, 0)
            offs_n_s = tl.where(mask_n, offs_n, 0)
            c_ptrs = c_ptr + offs_m_s[:, None] * stride_cm + offs_n_s[None, :] * stride_cn
            mask = mask_m[:, None] & mask_n[None, :]
            if IS_ROUTER:
                tl.store(c_ptrs, acc, mask=mask)
            else:
                tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)
        else:
            # Partials are fp32 [split, grid_m, grid_n, BM, BN]. Invalid M lanes are zeros.
            acc = tl.where(mask_m[:, None], acc, 0.0)
            tile = (pid_k * grid_m + pid_m) * grid_n + pid_n
            block = BLOCK_M * BLOCK_N
            p_ptrs = p_ptr + tile * block + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            tl.store(p_ptrs, acc)

    @triton_mod.jit
    def reduce_kernel(
        p_ptr, c_ptr,
        M, N,
        stride_cm, stride_cn,
        grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
        STORE_BF16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        block = BLOCK_M * BLOCK_N
        # tile(s) = (s * grid_m + pid_m) * grid_n + pid_n
        base = (pid_m * grid_n + pid_n) * block
        split_stride = grid_m * grid_n * block
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in range(SPLIT_K):
            p_ptrs = p_ptr + base + s * split_stride
            p_ptrs = p_ptrs + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            acc += tl.load(p_ptrs)
        offs_m_s = tl.where(mask_m, offs_m, 0)
        offs_n_s = tl.where(mask_n, offs_n, 0)
        c_ptrs = c_ptr + offs_m_s[:, None] * stride_cm + offs_n_s[None, :] * stride_cn
        mask = mask_m[:, None] & mask_n[None, :]
        if STORE_BF16:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)
        else:
            tl.store(c_ptrs, acc, mask=mask)

    return gemm_kernel, reduce_kernel


def _pack_transposed(state) -> None:
    """Contiguous [K, N] copies. BF16 weights stay BF16; router becomes a hi/lo pair."""
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[name + "_t"] = state[name].transpose(0, 1).contiguous()
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.transpose(0, 1).contiguous()
    state["router_lo_t"] = lo.transpose(0, 1).contiguous()
    if state["router_hi_t"].shape != state["router_lo_t"].shape:
        raise RuntimeError("router hi/lo shape")
    if state["router_hi_t"].stride() != state["router_lo_t"].stride():
        raise RuntimeError("router hi/lo stride")


def _enable_triton(state, triton_mod) -> None:
    gemm_kernel, reduce_kernel = _build_kernels(triton_mod)
    _pack_transposed(state)
    device = state["qkv_gate"].device
    if device.type != "cuda":
        raise RuntimeError("weights are not CUDA")
    for name in _LAYERS:
        if state[name].device != device:
            raise RuntimeError("weight device mismatch")
    state["n_full"] = {name: int(state[name].shape[0]) for name in _LAYERS}
    state["partial"] = torch.empty(_MAX_PARTIAL, dtype=torch.float32, device=device)

    def launch(data, state, require_fast=False):
        cfg = data["config"]
        x = data["x"]
        layer = cfg["layer"]
        m = cfg["m"]
        n = cfg["n"]
        k = cfg["k"]
        partial = state["partial"]
        n_full = state["n_full"].get(layer, 0)
        ok = (
            layer in _LAYERS
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.shape[0] == m
            and x.shape[1] == k
            and _shape_ok(m, n, k, n_full, partial.numel())
        )
        is_router = layer == "router"
        if ok:
            b = state["router_hi_t"] if is_router else state[layer + "_t"]
            blo = state["router_lo_t"] if is_router else b
            ok = (
                b.is_cuda
                and b.shape[0] == k
                and b.shape[1] >= n
                and b.stride(1) == 1
                and (not is_router or (blo.shape == b.shape and blo.stride() == b.stride()))
            )
        else:
            b = blo = None
        if not ok:
            if require_fast:
                raise RuntimeError(f"fast path refused m={m} n={n} k={k} layer={layer}")
            return _fallback(x, state, layer, n)
        if not x.is_contiguous():
            x = x.contiguous()
        bm, bn, bk, warps, stages, split = _tile_for(m, k)
        grid_m = _cdiv(m, bm)
        grid_n = _cdiv(n, bn)
        k_per = k // split
        k_tiles = k_per // bk
        out_dtype = torch.float32 if is_router else torch.bfloat16
        out = torch.empty((m, n), device=x.device, dtype=out_dtype)
        gemm_kernel[(grid_m, grid_n, split)](
            x, b, blo, out, partial,
            m, n, k,
            x.stride(0), x.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            k_per, k_tiles, grid_m, grid_n,
            bm, bn, bk, split, is_router,
            num_warps=warps,
            num_stages=stages,
        )
        if split > 1:
            reduce_kernel[(grid_m, grid_n)](
                partial, out,
                m, n,
                out.stride(0), out.stride(1),
                grid_m, grid_n,
                bm, bn, split, (not is_router),
                num_warps=4,
                num_stages=1,
            )
        return out

    for m, layer, n, k in _warmup_cases():
        x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
        y = launch(
            {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}},
            state,
            require_fast=True,
        )
        if y.shape != (m, n):
            raise RuntimeError("warmup shape")
        expect = torch.float32 if layer == "router" else torch.bfloat16
        if y.dtype != expect:
            raise RuntimeError("warmup dtype")
    state["launch"] = launch
    state["use_triton"] = True


def prepare(weights: Weights):
    """Repack weights once. Import, compile, and warmup Triton only when CUDA is up."""
    state = {name: weights[name].contiguous() for name in _LAYERS}
    state["use_triton"] = False
    state["launch"] = None
    state["max_m"] = MAX_M
    if not torch.cuda.is_available():
        return state
    try:
        import triton
        import triton.language as tl
        globals()["triton"] = triton
        globals()["tl"] = tl
        _enable_triton(state, triton)
    except Exception as exc:
        state["use_triton"] = False
        state["launch"] = None
        print(f"r1-triton: torch fallback ({type(exc).__name__}: {exc})")
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """x @ W[:n].T. CUDA-graph safe: no sync, no autotune, no first-time compile."""
    cfg = data["config"]
    x = data["x"]
    if not state.get("use_triton", False):
        return _fallback(x, state, cfg["layer"], cfg["n"])
    return state["launch"](data, state)
