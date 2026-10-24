"""Adapter for the patched serving class, using synthetic prepared expert weights.

This exercises the real _forward_marlin method, including its workspace allocation
and finalizer. Model loading, pre-expert norm and collectives remain outside scope.
Not a kernel-challenge submission: this adapter imports the local baseline.
"""
import baseline
import inspect
from pathlib import Path
import torch
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts


def prepare(weights):
    expected = Path(__file__).resolve().parents[1] / "serving/sglang/python"
    if not Path(inspect.getfile(MuseSpark1xFp4Experts)).resolve().is_relative_to(expected):
        raise RuntimeError("Serving adapter must import the patched checkout; set PYTHONPATH")
    return prepare_layer(weights, MuseSpark1xFp4Experts)


def prepare_layer(weights, expert_class):
    prepared = baseline.prepare(weights)
    bank = prepared["banks"][(0, 32)]
    # Construct only the state required by this method; the full constructor
    # requires distributed process groups and a model configuration.
    layer = expert_class.__new__(expert_class)
    torch.nn.Module.__init__(layer)
    layer.intermediate_size_per_partition = bank["size_n"]
    layer.num_local_experts = bank["count"]
    layer.global_num_experts = baseline.N_EXPERTS
    layer._expert_map = bank["expert_map"]
    for attr, key in (("w13_weight", "w13"), ("w2_weight", "w2"),
                      ("w13_weight_scale", "w13_s"), ("w2_weight_scale", "w2_s"),
                      ("w13_weight_scale_2", "w13_gs"), ("w2_weight_scale_2", "w2_gs")):
        setattr(layer, attr, bank[key])
    # The model binds these cached norm fields before graph capture.
    layer._post_expert_norm = object()
    layer._post_expert_norm_weight = prepared["norm_w"]
    layer._post_expert_norm_eps = baseline.POST_NORM_EPS
    layer._post_expert_norm_weight_first = True
    return layer


def kernel_fn(data, layer):
    return layer._forward_marlin(data["hidden_states"], data["topk_weights"],
                                 data["topk_ids"], layer._post_expert_norm)
