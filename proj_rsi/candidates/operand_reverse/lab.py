"""Root-scheduled unpooler128-only complete-output trial. No parameter/flag sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
import prototype as P

sys.path.insert(0,str(P.ROOT/"muse_bf16_proj"))
import reference


def baseline_module():
    import importlib.util
    path=P.HERE.parent/"pw36/submission.py"
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha=="2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf"
    spec=importlib.util.spec_from_file_location("reverse_frozen_pw36",path)
    B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
    return B,sha


def capture(fn,x,out):
    fn(x,out);fn(x,out);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn(x,out)
    return g


def validate(fn,w):
    gen=torch.Generator(device="cuda").manual_seed(6209)
    ok,msg=P.S._rqval(fn,w,128,8192,gen,4e-3)
    assert ok,msg
    print("ONEHOT4096_EXACT "+msg,flush=True)
    checks=[{"all4096_onehot_columns":"EXACT","random_FP64":"PASS","detail":msg}]
    for dist in ("normal","outliers"):
        data=reference.generate_input(128,"unpooler",7289,dist=dist)
        before=data["x"].clone();out=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda")
        fn(data["x"],out)
        ok,msg=reference.check_implementation(data,out,{"unpooler":w})
        assert ok and torch.equal(before,data["x"]),msg
        checks.append({"official_input_dist":dist,"result":msg})
    data=reference.generate_input(128,"unpooler",1321)
    x=data["x"].clone();out=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda")
    g=capture(fn,x,out)
    for kind in ("original","zero","fresh_outliers"):
        fresh=data["x"] if kind=="original" else torch.zeros_like(x) if kind=="zero" else reference.generate_input(128,"unpooler",7771,dist="outliers")["x"]
        x.copy_(fresh);g.replay();torch.cuda.synchronize()
        assert torch.equal(x,fresh),"input modified"
        if kind=="zero":assert torch.count_nonzero(out)==0
        else:
            ok,msg=reference.check_implementation({"x":x,"config":data["config"]},out,{"unpooler":w})
            assert ok,msg
        checks.append({"fixed_pointer_replay":kind,"status":"PASS"})
    return checks


def validate_baseline(B,state,w):
    config={"layer":"unpooler","m":128,"n":8192,"k":4096}
    # Copy only for this untimed validator; actual timed baseline graph has no added copy.
    def fn(x,out):
        out.copy_(B.kernel_fn({"x":x,"config":config},state))
        return 0
    gen=torch.Generator(device="cuda").manual_seed(6211)
    ok,msg=P.S._rqval(fn,w,128,8192,gen,4e-3)
    assert ok,msg
    checks=[{"all4096_onehot_columns":"EXACT","random_FP64":"PASS","detail":msg}]
    for dist in ("normal","outliers"):
        data=reference.generate_input(128,"unpooler",7291,dist=dist)
        before=data["x"].clone()
        out=B.kernel_fn(data,state)
        ok,msg=reference.check_implementation(data,out,{"unpooler":w})
        assert ok and torch.equal(before,data["x"]),msg
        checks.append({"official_input_dist":dist,"result":msg})
    print("BASELINE_ONEHOT4096_AND_ORACLES PASS",flush=True)
    return checks


def profile(g):
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay();torch.cuda.synchronize()
    return [{"name":e.name(),"duration_ns":e.duration_ns()} for e in p.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]


def validate_actual_graphs(ga,gb,baseline_out,candidate_out,x,config,w):
    """Validate the exact timed graph objects and their real captured output buffers."""
    assert baseline_out.data_ptr()!=candidate_out.data_ptr()
    original=x.clone()
    probes=torch.zeros_like(x)
    rows=torch.arange(128,device="cuda")
    cols=(rows*37+19)%4096
    values=(2.0**((rows%5)+3))*(2*(rows%2)-1)
    probes[rows,cols]=values.to(torch.bfloat16)
    cases=[("original",original),("zero",torch.zeros_like(x)),
           ("fresh_normal",reference.generate_input(128,"unpooler",9911)["x"]),
           ("fresh_outliers",reference.generate_input(128,"unpooler",9913,dist="outliers")["x"]),
           ("all_probes",probes)]
    checks=[]
    for name,fresh in cases:
        x.copy_(fresh)
        for label,g,dst in (("baseline",ga,baseline_out),("candidate",gb,candidate_out)):
            dst.fill_(float("nan"))
            before=x.clone()
            g.replay();torch.cuda.synchronize()
            assert torch.equal(x,before),"actual timed graph modified X"
            assert torch.isfinite(dst).all(),"actual timed graph left poisoned/incomplete output"
            if name=="zero":
                assert torch.count_nonzero(dst)==0
                msg="all output elements exactly zero"
            else:
                ok,msg=reference.check_implementation({"x":x,"config":config},dst,{"unpooler":w})
                assert ok,msg
            checks.append({"graph":label,"input":name,"NaN_poison_overwritten":True,"input_immutable":True,"oracle":msg})
    x.copy_(original)
    ga.replay();gb.replay();torch.cuda.synchronize()
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    preflight=json.loads((P.HERE/"preflight.json").read_text())
    source_sha=hashlib.sha256(P.source().encode()).hexdigest()
    assert source_sha==preflight["source_sha256"],"source drift after proof"
    B,base_sha=baseline_module()
    print("IDENTITY candidate=%s baseline=%s"%(source_sha,base_sha),flush=True)
    ext=P.extension()
    empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
    info=lambda code:int(ext.rqrun(0,empty,empty,empty,empty,empty,0,0,0,empty,sink,code))
    attrs={"regs":info(1),"local":info(2),"smem":info(3),"max_dynamic_before":info(5)}
    attrs["occupancy_after_optin"]=info(4);attrs["max_dynamic_after"]=info(5)
    attrs["initial_CTA_pool"]=attrs["regs"]*384
    attrs["requested_role_pool"]=64*128+216*256
    print("ATTR "+json.dumps(attrs),flush=True)
    assert attrs["regs"]<=168 and attrs["local"]==0 and attrs["occupancy_after_optin"]>=1
    assert attrs["max_dynamic_after"]>=attrs["smem"] and attrs["requested_role_pool"]<=attrs["initial_CTA_pool"]<=65536
    weights=reference.generate_weights(reference.WEIGHTS_SEED)
    state=B.prepare(weights)
    w=weights["unpooler"]
    lay,tab,esc6=P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda")
    cnt=torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda")
    def fn(x,out):
        assert x.is_contiguous() and out.is_contiguous()
        r=int(ext.rqrun(0,lay,x,out,esc,cnt,1,tab,0,empty,sink,0))
        if r:raise RuntimeError("operand_reverse launch failed %d"%r)
        return 0
    checks=validate(fn,w)
    baseline_checks=validate_baseline(B,state,w)
    print("VALID complete-output unpooler128 PASS",flush=True)
    data=reference.generate_input(128,"unpooler",5237);x=data["x"]
    out=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda");hold=[None]
    def incumbent(xx,oo):
        hold[0]=B.kernel_fn({"x":xx,"config":data["config"]},state)
        return hold[0]
    original=x.clone();incumbent(x,out);torch.cuda.synchronize()
    ok,baseline_oracle=reference.check_implementation(data,hold[0],{"unpooler":w})
    assert ok and torch.equal(x,original),baseline_oracle
    print("BASELINE_CALLABLE_ORACLE "+baseline_oracle,flush=True)
    rp=state.get("rq",{}).get(("unpooler",128))
    if rp:
        names=B.RQPROD2 if "mod" in rp else B.RQPROD
        pick={"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
    else:pick={"kind":"actual_kernel_fn","Lt_code":state.get("lt",{}).get(("unpooler",128)),"sel":str(state.get("sel",{}).get(("unpooler",B._bk(128))))}
    flush=torch.empty(64<<20,dtype=torch.float32,device="cuda");fsink=torch.empty((),dtype=torch.float32,device="cuda")
    B._time_us(incumbent,x,out,flush,fsink,it=5);B._time_us(fn,x,out,flush,fsink,it=5)
    before=B._time_us(incumbent,x,out,flush,fsink,it=21)
    candidate=B._time_us(fn,x,out,flush,fsink,it=21)
    after=B._time_us(incumbent,x,out,flush,fsink,it=21)
    ga=capture(incumbent,x,out);keepalive=hold[0]
    gb=capture(fn,x,out);pairs=[]
    for r in range(21):
        row={}
        for name,g in (("baseline",ga),("candidate",gb)) if r%2==0 else (("candidate",gb),("baseline",ga)):
            torch.sum(flush,0,out=fsink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();g.replay();b.record();torch.cuda.synchronize()
            row[name]=a.elapsed_time(b)*1000
        pairs.append(row)
    baseline_profile=profile(ga);candidate_profile=profile(gb)
    actual_graph_checks=validate_actual_graphs(ga,gb,keepalive,out,x,data["config"],w)
    f1=min(before,after)-candidate
    paired=statistics.median(p["baseline"]-p["candidate"] for p in pairs)
    result={"source_sha256":source_sha,"baseline_sha256":base_sha,"attrs":attrs,"checks":checks,"baseline_checks":baseline_checks,
            "baseline_callable_oracle":baseline_oracle,
            "actual_timed_graph_checks":actual_graph_checks,"baseline_pick":pick,
            "F1_baseline_before_us":before,"F1_baseline_after_us":after,"F1_candidate_us":candidate,"F1_gain_us":f1,
            "paired_graph":pairs,"paired_graph_median_gain_us":paired,"baseline_profile":baseline_profile,"candidate_profile":candidate_profile,
            "status":"SURVIVES_BOUNDED_LAB_ONLY" if f1>=0.5 and paired>=0.5 else "CLOSE_POLICY_NO_SWEEP"}
    args.json.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT "+json.dumps(result),flush=True)
    print("LAB_DONE no H200/score/promotion claim",flush=True)


if __name__=="__main__":main()
