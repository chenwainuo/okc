"""One fixed policy: o32, first legal layout2 heuristic, explicit splitK1. Root GPU scheduling only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = HERE.parent/"pw36/submission.py"
FROZEN = "2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
sys.path.insert(0,str(ROOT/"muse_bf16_proj"))
import reference


def load_baseline():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest()==FROZEN
    spec = importlib.util.spec_from_file_location("ltgraph_frozen_pw36",BASELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_pick(B,state):
    rp = state.get("rq",{}).get(("o",32))
    if rp:
        names = B.RQPROD2 if "mod" in rp else B.RQPROD
        return {"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
    p = state.get("sel",{}).get(("o",B._bk(32)))
    if p:return {"kind":"sel","pick":list(p)}
    code = state.get("lt",{}).get(("o",32))
    return {"kind":"r6","Lt_code":code,"Lt_cfg":list(B._ext().lt_cfg(32,8192,1024,0,code)) if code is not None else None,
            "gv_pick":str(state.get("gv",{}).get(("o",B._bk(32))))}


def choose_one(ext):
    # Metadata/AlgoCheck only. Stop at the FIRST legal candidate; never time a grid of configurations.
    records = []
    for i in range(min(int(ext.lt_algos(32,8192,1024,0,2)),32)):
        cfg = list(ext.lt_cfg(32,8192,1024,0,200000+i))
        rec = {"heuristic_rank":i,"original_cfg":cfg}
        if cfg[2]>1:
            rec["skip"]="splitK>1 heuristic excluded by fixed policy"; records.append(rec);continue
        requested = cfg.copy(); requested[2]=1; requested[3]=0
        code = int(ext.lt_add(32,8192,1024,0,2,requested))
        rec["requested_cfg"]=requested;rec["code"]=code
        records.append(rec)
        if code<0:continue
        actual = list(ext.lt_cfg(32,8192,1024,0,code))
        assert actual[2]==1 and actual[3]==0,actual
        return code,actual,records
    return None,None,records


def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn(x,out)
    return g


def validate(B,fn,w):
    gen = torch.Generator(device="cuda").manual_seed(6011)
    ok,msg = B._rqval(fn,w,32,8192,gen,4e-3)
    assert ok,msg
    results = [{"all1024_onehot_columns":"EXACT","random_FP64":"PASS","detail":msg}]
    data = reference.generate_input(32,"o",3317)
    x = data["x"].clone();out = torch.empty((32,8192),device="cuda",dtype=torch.bfloat16)
    g = capture(fn,x,out)
    for kind in ("original","zero","fresh_outliers"):
        fresh = data["x"] if kind=="original" else torch.zeros_like(x) if kind=="zero" else reference.generate_input(32,"o",7711,dist="outliers")["x"]
        x.copy_(fresh);g.replay();torch.cuda.synchronize()
        assert torch.equal(x,fresh),"input mutation"
        if kind=="zero":assert torch.count_nonzero(out)==0
        else:
            ok,msg = reference.check_implementation({"x":x,"config":data["config"]},out,{"o":w})
            assert ok,msg
        results.append({"graph_replay_input":kind,"status":"PASS"})
    return results


def profile_replay(g):
    # Diagnostic only; never use profiler durations to select the candidate.
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay();torch.cuda.synchronize()
    return [{"name":e.name(),"start_ns":e.start_ns(),"duration_ns":e.duration_ns()}
            for e in p.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]


def main():
    ap = argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    B = load_baseline()
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    state = B.prepare(weights)  # Same state serves the baseline for the entire experiment.
    ext = B._ext()
    pick = baseline_pick(B,state)
    code,cfg,metadata = choose_one(ext)
    result = {"baseline_sha256":FROZEN,"baseline_pick":pick,"policy":"o32_layout2_first_legal_heuristic_force_splitK1",
              "selection_metadata":metadata,"candidate_code":code,"candidate_cfg":cfg,
              "workspace":"same existing process-owned64MiB Lt workspace, sequential graph replays"}
    if code is None:
        result["status"]="CLOSE_NO_LEGAL_CANDIDATE"
        args.json.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result));return
    w,wt = state["o_raw"],state["o_t"]
    def fn(x,out):
        r = int(ext.lt_mm(x,w,wt,out,8192,code))
        if r:raise RuntimeError("Lt launch failed %d" % r)
        return 0
    result["validation"] = validate(B,fn,w)
    data = reference.generate_input(32,"o",5179);x=data["x"]
    out=torch.empty((32,8192),dtype=torch.bfloat16,device="cuda")
    hold=[None]
    def incumbent(xx,oo):
        hold[0]=B.kernel_fn({"x":xx,"config":data["config"]},state)
        return hold[0]
    flush=torch.empty(64<<20,device="cuda",dtype=torch.float32)
    fsink=torch.empty((),device="cuda",dtype=torch.float32)
    B._time_us(incumbent,x,out,flush,fsink,it=5);B._time_us(fn,x,out,flush,fsink,it=5)
    before=B._time_us(incumbent,x,out,flush,fsink,it=21)
    candidate=B._time_us(fn,x,out,flush,fsink,it=21)
    after=B._time_us(incumbent,x,out,flush,fsink,it=21)
    ga=capture(incumbent,x,out);keepalive=hold[0]
    gb=capture(fn,x,out)
    pairs=[]
    for r in range(21):
        row={}
        for name,g in (("baseline",ga),("candidate",gb)) if r%2==0 else (("candidate",gb),("baseline",ga)):
            torch.sum(flush,0,out=fsink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();g.replay();b.record();torch.cuda.synchronize()
            row[name]=a.elapsed_time(b)*1000
        pairs.append(row)
    # Profiles diagnose actual node/kernel families, separately from admitted event timings.
    baseline_events=profile_replay(ga);candidate_events=profile_replay(gb)
    ga.replay();gb.replay();torch.cuda.synchronize()
    assert torch.isfinite(keepalive).all()
    ok,msg=reference.check_implementation(data,out,{"o":w});assert ok,msg
    f1gain=min(before,after)-candidate
    paired=statistics.median(p["baseline"]-p["candidate"] for p in pairs)
    result.update({"F1_baseline_before_us":before,"F1_baseline_after_us":after,"F1_candidate_us":candidate,"F1_gain_us":f1gain,
                   "paired_graph":pairs,"paired_graph_median_gain_us":paired,
                   "baseline_profile_events":baseline_events,"candidate_profile_events":candidate_events,
                   "status":"SURVIVES_BOUNDED_LAB_ONLY" if f1gain>=0.5 and paired>=0.5 else "CLOSE_POLICY_NO_SWEEP"})
    args.json.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT "+json.dumps(result),flush=True)
    print("LAB_DONE no H200/score/promotion claim",flush=True)


if __name__=="__main__":main()
