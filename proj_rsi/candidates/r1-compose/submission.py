# okc: problem=muse_bf16_proj
"""BF16 projections via Hopper GEMV (M=1) and tiny_gemm where the shape fits.

Backends are imported in prepare only if CUDA is up. A predicate miss or a
kernel exception uses torch, so a shape miss cannot fail correctness.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F

# Set in prepare. None disables that backend. Not derived from activations.
_BE = {
    "use_hopper": None,
    "hopper": None,
    "can_tiny": None,
    "tiny": None,
    "skip": set(),
}

# Scored token counts plus official cases that can take a fast path.
_WARM_M = (1, 5, 7, 8, 16, 32, 128, 256, 512)


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
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


def _reset_backends() -> None:
    _BE["use_hopper"] = None
    _BE["hopper"] = None
    _BE["can_tiny"] = None
    _BE["tiny"] = None
    _BE["skip"] = set()


def _import_backends() -> None:
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _BE["hopper"] = hopper_bf16_gemv
        _BE["use_hopper"] = use_hopper_bf16_gemv
    except Exception:
        _BE["hopper"] = None
        _BE["use_hopper"] = None
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

        _BE["tiny"] = tiny_gemm_bf16
        _BE["can_tiny"] = can_use_tiny_gemm
    except Exception:
        _BE["tiny"] = None
        _BE["can_tiny"] = None


def _warmup(state: dict) -> None:
    """Compile fast-path shapes in prepare. A failed launch is not tried again."""
    device = state["qkv_gate"].device
    if device.type != "cuda":
        return
    use_hopper = _BE["use_hopper"]
    hopper = _BE["hopper"]
    can_tiny = _BE["can_tiny"]
    tiny = _BE["tiny"]
    if hopper is None and tiny is None:
        return
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        full = state[layer]
        for m in _WARM_M:
            n = 1024 if layer == "unpooler" and m <= 8 else full.shape[0]
            w = full[:n]
            k = w.shape[1]
            if hopper is not None and m == 1:
                sig = ("h", n, k)
                try:
                    if use_hopper(1, n, k):
                        hopper(torch.zeros((1, k), dtype=torch.bfloat16, device=device), w)
                except Exception:
                    _BE["skip"].add(sig)
            if tiny is None:
                continue
            max_m = max(m, 16)
            sig = ("t", n, k, max_m)
            if sig in _BE["skip"]:
                continue
            try:
                if m <= max_m and can_tiny(n, k, max_m=max_m):
                    tiny(
                        torch.zeros((1, k), dtype=torch.bfloat16, device=device),
                        w,
                        max_m=max_m,
                    )
            except Exception:
                _BE["skip"].add(sig)


def prepare(weights: Weights) -> dict:
    _reset_backends()
    state = {
        name: weights[name].contiguous()
        for name in ("qkv_gate", "o", "pooler", "unpooler", "router")
    }
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    if torch.cuda.is_available():
        _import_backends()
        _warmup(state)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer = cfg["layer"]
    m = cfg["m"]
    n = cfg["n"]
    if layer == "router":
        w = state["router"][:n]
        # Starter FP32 path at m<=8. CPU torch has no mm out_dtype, so it stays here at any m.
        if m <= 8 or not x.is_cuda:
            return x.float() @ w.T
        hi_t = state["router_hi_t"]
        lo_t = state["router_lo_t"]
        if hi_t.shape[1] != n:
            hi_t = hi_t[:, :n].contiguous()
            lo_t = lo_t[:, :n].contiguous()
        try:
            acc = torch.mm(x, hi_t, out_dtype=torch.float32)
            acc += torch.mm(x, lo_t, out_dtype=torch.float32)
            return acc
        except Exception:
            return x.float() @ w.T

    w = state[layer][:n]
    wn, wk = w.shape
    use_hopper = _BE["use_hopper"]
    if (
        use_hopper is not None
        and m == 1
        and ("h", wn, wk) not in _BE["skip"]
        and use_hopper(m, wn, wk)
    ):
        try:
            return _BE["hopper"](x, w)
        except Exception:
            _BE["skip"].add(("h", wn, wk))

    can_tiny = _BE["can_tiny"]
    if can_tiny is not None:
        max_m = max(m, 16)
        sig = ("t", wn, wk, max_m)
        if m <= max_m and sig not in _BE["skip"] and can_tiny(wn, wk, max_m=max_m):
            try:
                return _BE["tiny"](x, w, max_m=max_m)
            except Exception:
                _BE["skip"].add(sig)

    return F.linear(x, w)
