"""Full-expert experimental adapter: BOTH projections use the new SS engine.
Routing, SwiGLU, BF16 boundaries and inherited post-expert norm/router are preserved.
No serving tree modification or global monkeypatch.
"""
import hashlib,importlib.util,inspect,sys,textwrap
from pathlib import Path
import torch
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts,_MARLIN_BLOCK_SIZE_M_CANDIDATES
from serving_moe import prepare_layer
p=Path(__file__).with_name('adapter.py');sp=importlib.util.spec_from_file_location('_ss_t64_m3012_adapter_v1',p);_engine=importlib.util.module_from_spec(sp);sys.modules[sp.name]=_engine;sp.loader.exec_module(_engine)
SOURCE_METHOD_SHA256="781aecba1270deb918366284e4f8245ac48cce5e99f77f1a1ea95025aeaef266"
class BothSSExperts(MuseSpark1xFp4Experts):
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

        cache1 = _engine.project(hidden_states, self._ss_up_bank, sorted_token_ids,
                                 expert_ids, num_tokens_post_padded, block_size_m,
                                 out=cache1, gather_divisor=topk)
        silu_and_mul(
            cache1.view(-1, gemm1_n),
            cache2,
            expert_ids=local_topk_ids.view(-1),
            expert_step=1,
        )
        projected = _engine.project(cache2, self._ss_down_bank, sorted_token_ids,
                                    expert_ids, num_tokens_post_padded, block_size_m,
                                    out=cache3, gather_divisor=1)
        return projected.view(num_tokens, topk, size_k), local_topk_ids

def prepare(weights):
    actual=textwrap.dedent(inspect.getsource(MuseSpark1xFp4Experts._marlin_expert_outputs)).strip()
    assert hashlib.sha256(actual.encode()).hexdigest()==SOURCE_METHOD_SHA256,'Serving parent drift; rebase review needed'
    up=_engine.prepare(weights['w13'][:32],weights['w13_sf'][:32],weights['w13_gs'][:32])
    down=_engine.prepare(weights['w2'][:32],weights['w2_sf'][:32],weights['w2_gs'][:32])
    layer=prepare_layer(weights,BothSSExperts);layer._ss_up_bank=up;layer._ss_down_bank=down
    return layer

def kernel_fn(data,layer):
    return layer._forward_marlin(data['hidden_states'],data['topk_weights'],data['topk_ids'],layer._post_expert_norm)
