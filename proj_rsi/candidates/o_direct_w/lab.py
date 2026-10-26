"""Root-scheduled one-shot comparison only. No parameter or geometry sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import prototype as P

sys.path.insert(0,str(P.ROOT/"muse_bf16_proj"))
import reference


def checked_launch(ext,idx,lay,tab,esc,cnt,empty,sink,x,out):
    r = ext.rqrun(idx,lay,x,out,esc,cnt,1,tab,0,empty,sink,0)
    if r:raise RuntimeError("prototype launch error %d" % r)
    return 0


def validate(fn,w,m):
    gen = torch.Generator(device="cuda").manual_seed(4167+m)
    ok,msg = P.S._rqval(fn,w,m,8192,gen,4e-3)
    assert ok,msg
    print("EXACT_ONEHOT_ALL_K m=%d %s" % (m,msg),flush=True)
    checks = []
    for dist in ("normal","outliers"):
        data = reference.generate_input(m,"o",9417+m,dist=dist)
        out = torch.empty((m,8192),dtype=torch.bfloat16,device="cuda")
        original = data["x"].clone()
        fn(data["x"],out)
        ok,msg = reference.check_implementation(data,out,{"o":w})
        assert ok and torch.equal(data["x"],original),msg
        checks.append({"dist":dist,"result":msg})
    # Captured pointers see fresh X values; no stale-input or persistent-counter assumption.
    data = reference.generate_input(m,"o",1709+m)
    x = data["x"].clone()
    out = torch.empty((m,8192),dtype=torch.bfloat16,device="cuda")
    fn(x,out); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn(x,out)
    for kind in ("original","zero","fresh_outliers"):
        fresh = (data["x"] if kind=="original" else torch.zeros_like(x) if kind=="zero" else
                 reference.generate_input(m,"o",7719+m,dist="outliers")["x"])
        x.copy_(fresh); g.replay(); torch.cuda.synchronize()
        assert torch.equal(x,fresh),"input mutated"
        if kind=="zero":assert torch.count_nonzero(out)==0
        else:
            d = {"x":x,"config":data["config"]}
            ok,msg = reference.check_implementation(d,out,{"o":w})
            assert ok,msg
        checks.append({"replay":kind,"status":"PASS"})
    return {"all1024_onehot_columns":"EXACT", "random_FP64_4e3_and_row5x":"PASS","official_and_replay":checks}


def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn(x,out)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",type=Path,required=True)
    ap.add_argument("--reps",type=int,default=21)
    args = ap.parse_args()
    assert 9<=args.reps<=41
    expected = json.loads((P.HERE/"preflight.json").read_text())
    sha = hashlib.sha256(P.source().encode()).hexdigest()
    assert sha==expected["source_sha256"],"source changed since preflight"
    B = P.load(P.HERE.parent/"pw36/submission.py","od_frozen_incumbent")
    baseline_sha = hashlib.sha256((P.HERE.parent/"pw36/submission.py").read_bytes()).hexdigest()
    assert baseline_sha=="2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
    print("IDENTITY prototype=%s baseline_pw36=%s" % (sha,baseline_sha),flush=True)
    ext = P.extension()
    empty = torch.empty(0,dtype=torch.int64,device="cuda")
    sink = torch.zeros(4,dtype=torch.int32,device="cuda")
    for i,m in enumerate((32,128)):
        attrs = [int(ext.rqrun(i,empty,empty,empty,empty,empty,0,0,0,empty,sink,k)) for k in (1,2,3)]
        assert attrs[1]==0,attrs
        print("ATTR m=%d regs=%d local=%d smem=%d" % (m,*attrs),flush=True)
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    t = time.time()
    state = B.prepare(weights)
    print("BASELINE_PREP_SECONDS %.3f" % (time.time()-t),flush=True)
    w = weights["o"]
    lay,tab,esc6 = P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_ = P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda")
    cnt = torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda")
    flush = torch.empty(64<<20,dtype=torch.float32,device="cuda")
    fsink = torch.empty((),dtype=torch.float32,device="cuda")
    results = []
    for idx,m in enumerate((32,128)):
        fn = lambda xx,oo,idx=idx:checked_launch(ext,idx,lay,tab,esc,cnt,empty,sink,xx,oo)
        checks = validate(fn,w,m)
        print("VALID m=%d PASS" % m,flush=True)
        data = reference.generate_input(m,"o",1337+m)
        x = data["x"]
        out = torch.empty((m,8192),dtype=torch.bfloat16,device="cuda")
        hold = [None]
        def inc(xx,oo):
            hold[0] = B.kernel_fn({"x":xx,"config":data["config"]},state)
            return hold[0]
        rp = state.get("rq",{}).get(("o",m))
        if rp:
            names = B.RQPROD2 if "mod" in rp else B.RQPROD
            path = {"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
        elif ("o",B._bk(m)) in state.get("sel",{}):
            path = {"kind":"sel","pick":list(state["sel"][("o",B._bk(m))])}
        else:
            path = {"kind":"r6","Lt_code":state.get("lt",{}).get(("o",m)),
                    "gv_pick":str(state.get("gv",{}).get(("o",B._bk(m))))}
        # Discard warmups; bracket the candidate with the same frozen prepared dispatch.
        B._time_us(inc,x,out,flush,fsink,it=5); B._time_us(fn,x,out,flush,fsink,it=5)
        before = B._time_us(inc,x,out,flush,fsink,it=21)
        candidate = B._time_us(fn,x,out,flush,fsink,it=21)
        after = B._time_us(inc,x,out,flush,fsink,it=21)
        f1gain = min(before,after)-candidate
        ga = capture(inc,x,out)
        baseline_output_keepalive = hold[0]
        gb = capture(fn,x,out)
        pairs = []
        for r in range(args.reps):
            row = {}
            for name,g in (("baseline",ga),("prototype",gb)) if r%2==0 else (("prototype",gb),("baseline",ga)):
                torch.sum(flush,0,out=fsink)
                a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();g.replay();b.record();torch.cuda.synchronize()
                row[name] = a.elapsed_time(b)*1000
            pairs.append(row)
        ga.replay();gb.replay();torch.cuda.synchronize()
        assert torch.isfinite(baseline_output_keepalive).all()
        ok,msg = reference.check_implementation(data,out,{"o":w})
        assert ok,msg
        gain = statistics.median(p["baseline"]-p["prototype"] for p in pairs)
        record = {"m":m,"baseline_path":path,"checks":checks,"F1_baseline_before_us":before,
                  "F1_baseline_after_us":after,"F1_prototype_us":candidate,"F1_gain_us":f1gain,
                  "paired_graph":pairs,"paired_graph_median_gain_us":gain,
                  "decision":"SURVIVES_BOUNDED_LAB_ONLY" if f1gain>=0.5 and gain>=0.5 else "CLOSE_FIXED_POLICY_NO_SWEEP"}
        results.append(record)
        print("RESULT "+json.dumps(record),flush=True)
    args.json.write_text(json.dumps({"status":"COMPLETE","source_sha256":sha,"baseline_sha256":baseline_sha,"results":results},indent=2)+"\n")
    print("LAB_DONE no promotion or H200 speedup claim",flush=True)


if __name__=="__main__":main()
