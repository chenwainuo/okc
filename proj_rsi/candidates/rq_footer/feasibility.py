"""Bounded CPU/source audit of the128B immutable footer; no CUDA generation/compile/launch."""
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "rs32" / "submission.py"
PARENT_SHA256 = "a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad"
MODEL = HERE.parent / "rs128p" / "prepared_patch_layout_proof.py"
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assess(w, parent, model, layer):
    N, K = w.shape
    NB, KC = N // 128, K // 64
    started = time.monotonic()
    pk = parent._rqpack(w)
    assert pk is not None
    lay, tab, esc = pk
    unpack = parent._rqunpack(lay, tab, esc, N, K)
    assert torch.equal(unpack, w.view(torch.int16).int() & 65535)
    del unpack
    KS = parent.RQKS[layer]
    stock_list, _ = parent._rqesc(esc, NB, KC, KS, w.device)
    groups = KS * NB * 8
    cap = (stock_list.numel() - 32) // groups
    list_entries = (stock_list[:-32].view(groups, cap) != -1).sum(dim=1)
    stages = defaultdict(list)
    for tile, ch, q, lane, value, xor in esc.tolist():
        stages[(tile // 8, ch)].append((tile % 8, lane, q, value, xor))
    record_counts, register_counts, hits = Counter(), Counter(), Counter()
    overflows = []
    first_examples = []
    for nb in range(NB):
        for ch in range(KC):
            entries = stages.get((nb, ch), [])
            patch = model.original_patches(entries)
            assert patch is not None
            lane_counts = Counter((warp, lane) for warp, lane, reg in patch)
            record_counts[len(lane_counts)] += 1
            register_counts[max(lane_counts.values(), default=0)] += 1
            hits[len(entries)] += 1
            footer = model.make_footer(entries)
            if footer is None:
                reason = "more_than8_lane_records" if len(lane_counts) > 8 else "more_than2_registers_per_lane"
                overflows.append({"nb":nb,"chunk":ch,"reason":reason,"lanes":len(lane_counts),
                                  "max_registers_in_lane":max(lane_counts.values(),default=0)})
            else:
                assert len(footer) == 128
                assert model.read_footer(footer) == model.stock_list_oracle(entries)
                if entries and len(first_examples) < 2:
                    first_examples.append({"nb":nb,"chunk":ch,"escapes":len(entries),
                                           "lane_records":len(lane_counts),"footer_hex":footer.hex()})
    return {"N":N,"K":K,"NB":NB,"KC":KC,"stages":NB*KC,
            "KS":KS,"grid_CTAs":KS*NB,"stock_list_capacity":cap,
            "max_stock_list_entries":int(list_entries.max()),
            "stock_initial_prefix_load_bytes":groups*32*4,
            "stock_list_refill_needed":bool((list_entries>=32).any()),
            "packed_payload_bytes":lay.numel()*lay.element_size(),"footer_bytes":NB*KC*128,
            "global_escape_entries":len(esc),"codec_roundtrip_bit_exact":True,
            "all_eligible_footer_patches_match_stock_lists":True,
            "lane_record_histogram":dict(sorted(record_counts.items())),
            "max_registers_per_lane_histogram":dict(sorted(register_counts.items())),
            "escape_hits_per_stage_histogram":dict(sorted(hits.items())),
            "overflow_count":len(overflows),"overflow_examples":overflows[:8],
            "whole_layer_eligible":not overflows,"first_examples":first_examples,
            "CPU_pack_proof_s":time.monotonic()-started,
            "scope":"Exact reference CPU-generated BF16 weights; CUDA generator sequence/judge weights NOT asserted identical"}


def main():
    torch.set_num_threads(4)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA256
    parent=load(PARENT,"rq_footer_frozen_rs32")
    model=load(MODEL,"rq_footer_layout_model")
    sys.path.insert(0,str(ROOT/"muse_bf16_proj"))
    import reference
    result={"state":"CPU_SOURCE_FEASIBILITY_GPU_UNRUN","parent_sha256":PARENT_SHA256,
            "layout_model_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            "policy":{"variant":"s4s5z","NT":128,"ST":5,"GS":1,"FL":2,
                      "footer_bytes":128,"record_capacity":8,"registers_per_lane":2},
            "model_self_proof":model.proof(),"weights":{}}
    for seed in (5151,1234):
        weights=reference.generate_weights(seed,device="cpu")
        result["weights"][seed]={}
        for layer in ("pooler","qkv_gate","unpooler"):
            stats=assess(weights[layer],parent,model,layer)
            result["weights"][seed][layer]=stats
            print(json.dumps({"seed":seed,"layer":layer,"stages":stats["stages"],
                              "eligible":stats["whole_layer_eligible"],"overflows":stats["overflow_count"],
                              "escapes":stats["global_escape_entries"],
                              "max_lanes":max(stats["lane_record_histogram"]),
                              "max_registers_per_lane":max(stats["max_registers_per_lane_histogram"])}),flush=True)
            (HERE/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
        del weights
    result["static_cost"]={"baseline_W_stage_bytes":12288,"footer_W_stage_bytes":12416,
                           "baseline_smem_bytes":144520,"footer_smem_bytes":145160,
                           "extra_ring_shared_bytes":640,"original_arrival_counts_unchanged":True,
                           "eliminated_W_X_producers_or_barriers":0,
                           "footer_must_be_read_before_W_empty_release":True,
                           "register_allocation":"Not compiled; cannot claim exact regs/spills",
                           "new_live_uint32_t_estimate":5,
                           "retired_list_state_uint32_t_estimate":4}
    (HERE/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
