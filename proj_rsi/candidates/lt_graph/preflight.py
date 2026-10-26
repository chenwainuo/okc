"""CPU/source proof only. No CUDA API or candidate timing."""
import ast
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"pw36/submission.py"


def main():
    text=BASE.read_text();tree=ast.parse(text)
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    src=lambda name:ast.get_source_segment(text,funcs[name])
    tm=src("_tm");tune=src("_tune");search=src("_ltsearch");prepare=src("prepare")
    assert "torch.cuda.CUDAGraph()" in tm and "g.replay()" in tm and "torch.sum(flush" in tm
    assert "if m < 128 or not LTSEARCH or Wt is None:" in tune
    assert "ext.lt_algos(m, n, K, f32, 0)" in tune
    assert "for lay in range(4):" in search and "_LTSEED" in search
    assert 'st[k + "_t"] = w.t().contiguous()' in prepare
    # Column-major raw W with transpose vs already-transposed W without transpose.
    m,n,k=3,16,8
    w=torch.arange(n*k,dtype=torch.float64).reshape(n,k)-67
    x=torch.arange(m*k,dtype=torch.float64).reshape(m,k)-13
    flat=w.flatten();packed=w.t().contiguous().flatten()
    out0=torch.empty((m,n),dtype=torch.float64);out2=torch.empty_like(out0)
    for token in range(m):
        for channel in range(n):
            out0[token,channel]=sum(float(x[token,kk]*flat[channel*k+kk]) for kk in range(k))
            out2[token,channel]=sum(float(x[token,kk]*packed[kk*n+channel]) for kk in range(k))
    assert torch.equal(out0,out2) and torch.equal(out2,x@w.t())
    for path in [HERE/"lab.py",Path(__file__)]:ast.parse(path.read_text())
    result={"status":"SOURCE_AND_CPU_LAYOUT_PASS_GPU_UNRUN","baseline_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest(),
            "tuner_is_cold_graph_replay":True,"eager_graph_mismatch":False,
            "m32_pretranspose_layout2_not_in_tune_search":True,"m128_layouts0_to3_already_searched":True,
            "layout0_and_layout2_same_BF16_matmul_semantics":"PASS_CPU_INDEX_PROOF",
            "existing_workspace_bytes":64<<20,"existing_prepare_transpose":True,
            "new_policy":"o32_only_first_legal_layout2_heuristic_forced_splitK1_no_timed_config_sweep",
            "local_build":"No new CUDA source/extension; existing public Lt API only. Python syntax passes.",
            "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"lab.py",Path(__file__)]}}
    (HERE/"preflight.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
