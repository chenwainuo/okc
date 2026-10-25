"""Isolated full expert adapter replacing DOWN only. Development, unpromoted.

Copied _marlin_expert_outputs prefix is fingerprint-bound to incumbent method;
all routing/gateup/SwiGLU lines remain verbatim. _forward_marlin (norm andfinalizer)
is inherited unchanged. No monkeypatch of module/global functions.
"""
import hashlib
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path
import torch
from sglang.srt.layers.muse_spark_fp4_moe import (
    MuseSpark1xFp4Experts, _MARLIN_BLOCK_SIZE_M_CANDIDATES)
from serving_moe import prepare_layer

_ADAPTER_PATH=Path(__file__).with_name('routed_adapter.py')
_SPEC=importlib.util.spec_from_file_location('_rsi_scoped_wgmma_down',_ADAPTER_PATH)
_down=importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name]=_down
_SPEC.loader.exec_module(_down)
SOURCE_METHOD_SHA256="781aecba1270deb918366284e4f8245ac48cce5e99f77f1a1ea95025aeaef266"

class RoutedDownExperts(MuseSpark1xFp4Experts):
    def _marlin_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from sgl_kernel.scalar_type import scalar_types

        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        hidden_states = hidden_states.contiguous()
        num_tokens, size_k = hidden_states.shape
        topk = topk_ids.shape[1]
        size_n = self.intermediate_size_per_partition
        gemm1_n = 2 * size_n
        expert_map = self._expert_map

        local_topk_ids = topk_ids.to(torch.int32)
        if expert_map is not None:
            local_topk_ids = expert_map[local_topk_ids]
        local_topk_ids = local_topk_ids.contiguous()
        topk_weights = topk_scores.to(torch.float32).contiguous()

        for block_size_m in _MARLIN_BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * topk / self.num_local_experts / block_size_m < 0.9:
                break

        # Hopper EP8: measured small-batch tiles avoid overpadding sparse local
        # routes. Preserve stock tiles for larger prefills and other layouts.
        if (
            expert_map is not None
            and num_tokens <= 512
            and topk == 8
            and self.num_local_experts == 32
            and self.global_num_experts == 256
            and size_k == 4096
            and size_n == 4096
            and torch.cuda.get_device_capability(hidden_states.device) == (9, 0)
        ):
            if num_tokens <= 256:
                block_size_m = min(block_size_m, 16)
            elif num_tokens <= 512:
                block_size_m = min(block_size_m, 32)

        if num_tokens == 1 and topk <= 32 and expert_map is None:
            from sglang.kernels.ops.moe.moe_align_single_token import (
                moe_align_single_token,
            )

            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_single_token(local_topk_ids, block_size_m)
            )
        else:
            align_num_experts = (
                self.global_num_experts
                if expert_map is not None
                else self.num_local_experts
            )
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                local_topk_ids, block_size_m, align_num_experts
            )

        workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)
        cache2 = torch.empty(
            (num_tokens * topk, size_n),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cache13 = torch.empty(
            (num_tokens * topk * max(gemm1_n, size_k),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cache1 = cache13[: num_tokens * topk * gemm1_n].view(-1, gemm1_n)
        cache3 = cache13[: num_tokens * topk * size_k].view(-1, size_k)

        cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            cache1,
            self.w13_weight,
            None,
            self.w13_weight_scale,
            self.w13_weight_scale_2,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=num_tokens,
            size_n=gemm1_n,
            size_k=size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        silu_and_mul(
            cache1.view(-1, gemm1_n),
            cache2,
            expert_ids=local_topk_ids.view(-1),
            expert_step=1,
        )
        projected = _down.project(cache2, self._rsi_down_bank, sorted_token_ids,
                                  expert_ids, num_tokens_post_padded, block_size_m,
                                  out=cache3)
        return projected.view(num_tokens, topk, size_k), local_topk_ids

def prepare(weights):
    actual=textwrap.dedent(inspect.getsource(MuseSpark1xFp4Experts._marlin_expert_outputs)).strip()
    if hashlib.sha256(actual.encode()).hexdigest()!=SOURCE_METHOD_SHA256:
        raise RuntimeError('Incumbent expert method changed; recopy/review candidate prefix')
    # Retain original local packed tensors BEFORE baseline.prepare repacks its
    # copied layer parameters. No inverse Marlin permutation and no weight change.
    bank=_down.prepare(weights['w2'][:32],weights['w2_sf'][:32],weights['w2_gs'][:32])
    layer=prepare_layer(weights,RoutedDownExperts)
    layer._rsi_down_bank=bank
    return layer

def kernel_fn(data,layer):
    return layer._forward_marlin(data['hidden_states'],data['topk_weights'],
                                 data['topk_ids'],layer._post_expert_norm)
