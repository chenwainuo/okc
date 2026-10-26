"""V2 direct-A loader: root-scheduled three-way lab, no geometry/policy sweep."""
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


def actual_graph_checks(graphs, outputs, x, config, w):
    assert len({out.data_ptr() for out in outputs.values()}) == 3
    original = x.clone()
    probes = torch.zeros_like(x)
    rows = torch.arange(128, device=x.device)
    probes[rows, (rows * 37 + 19) % 4096] = ((2.0 ** ((rows % 5) + 3)) * (2 * (rows % 2) - 1)).to(torch.bfloat16)
    mutations = [("original", original), ("zero", torch.zeros_like(x)),
                 ("fresh_normal", reference.generate_input(128,"unpooler",9911)["x"]),
                 ("fresh_outliers", reference.generate_input(128,"unpooler",9913,dist="outliers")["x"]),
                 ("all_probes", probes)]
    checks = []
    for name, fresh in mutations:
        x.copy_(fresh)
        for label, graph in graphs.items():
            outputs[label].fill_(float("nan")); before = x.clone()
            graph.replay(); torch.cuda.synchronize()
            assert torch.equal(x, before) and bool(torch.isfinite(outputs[label]).all()), (label,name)
            if name == "zero":
                assert not int(torch.count_nonzero(outputs[label])); message = "exact zero complete output"
            else:
                ok, message = reference.check_implementation({"x":x,"config":config},outputs[label],{"unpooler":w})
                assert ok, (label,name,message)
            checks.append({"graph":label,"input":name,"NaN_poison_overwritten":True,
                           "input_immutable":True,"oracle":message})
    x.copy_(original)
    for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    return checks


def measure(graphs):
    labels = ("incumbent", "v1", "v2")
    flush = torch.empty(64<<20,dtype=torch.float32,device="cuda")
    sink = torch.empty((),dtype=torch.float32,device="cuda")
    def one(label):
        torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        return a.elapsed_time(b)*1000
    for label in labels:
        for _ in range(5):one(label)
    series = lambda label: [one(label) for _ in range(21)]
    raw = {"incumbent_before":series("incumbent"),"v1_before":series("v1"),"v2":series("v2"),
           "v1_after":series("v1"),"incumbent_after":series("incumbent")}
    med = {k:statistics.median(v) for k,v in raw.items()}
    f1 = {label:min(med[label+"_before"],med[label+"_after"])-med["v2"] for label in labels[:2]}
    orders = [(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
    pairs = []
    for repetition in range(21):
        row={}
        for i in orders[repetition%len(orders)]:row[labels[i]]=one(labels[i])
        pairs.append(row)
    paired={label:statistics.median(row[label]-row["v2"] for row in pairs) for label in labels[:2]}
    return {"F1_medians_us":med,"F1_samples_us":raw,"F1_v2_gain_us":f1,
            "paired_graph_us":pairs,"paired_v2_median_gain_us":paired,
            "paired_v2_wins":{label:sum(row[label]>row["v2"] for row in pairs) for label in labels[:2]}}


def main():
    import importlib.util
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    manifest=json.loads((P.HERE/"manifest.json").read_text())
    for name,sha in manifest["files"].items():
        assert hashlib.sha256((P.HERE/name).read_bytes()).hexdigest()==sha,("v2 drift",name)
    old=P.HERE.parent/"operand_reverse"
    frozen=json.loads((old/"manifest.json").read_text())
    for name in ("prototype.py","kernel_body.cuh","extension_source.cu","device_source.cu","preflight.py","lab.py"):
        assert hashlib.sha256((old/name).read_bytes()).hexdigest()==frozen["files"][name],("v1 drift",name)
    spec=importlib.util.spec_from_file_location("reverse_v1_frozen",old/"prototype.py")
    V1=importlib.util.module_from_spec(spec);spec.loader.exec_module(V1)
    v2sha=hashlib.sha256(P.source().encode()).hexdigest()
    v1sha=hashlib.sha256(V1.source().encode()).hexdigest()
    assert v2sha==manifest["source_sha256"] and v1sha=="ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8"
    # Build both extensions before importing pw36 and starting its prepare budget clock.
    ext2=P.extension();ext1=V1.extension()
    B,base_sha=baseline_module()
    result={"status":"STARTED_NO_TIMING_ADMISSION","parent_sha256":base_sha,
            "v1_source_sha256":v1sha,"v2_source_sha256":v2sha,
            "loader_policy":"direct native-A16 scalaru32/ca per thread/stage; four retired-slot-prefetch buffers",
            "attrs":{}}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():args.json.write_text(json.dumps(result,indent=2)+"\n")
    empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
    for label,ext in (("v1",ext1),("v2",ext2)):
        info=lambda c:int(ext.rqrun(0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c))
        attrs={"regs":info(1),"local":info(2),"smem":info(3),"max_dynamic_before":info(5)}
        attrs["occupancy_after_optin"]=info(4);attrs["max_dynamic_after"]=info(5)
        assert attrs["local"]==0 and attrs["regs"]<=168 and attrs["occupancy_after_optin"]>=1
        assert attrs["max_dynamic_after"]>=attrs["smem"]==103600
        if label=="v1":assert 64*128+216*256<=attrs["regs"]*384<=65536
        else:assert attrs["regs"]*384<=65536 # No dynamic pool request at all.
        attrs["register_mode"]="uniform static" if label=="v2" else "64/216 dynamic frozen v1"
        result["attrs"][label]=attrs
    weights=reference.generate_weights(reference.WEIGHTS_SEED);state=B.prepare(weights);w=weights["unpooler"]
    weight_before=w.clone()
    lay,tab,esc6=P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda")
    lay_before=lay.clone();esc_before=esc.clone()
    counters={name:torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda") for name in ("v1","v2")}
    def make_fn(label,ext):
        def fn(x,out):
            assert x.is_contiguous() and out.is_contiguous()
            r=int(ext.rqrun(0,lay,x,out,esc,counters[label],1,tab,0,empty,sink,0))
            if r:raise RuntimeError(label+" launch failed "+str(r))
            return 0
        return fn
    funcs={"v1":make_fn("v1",ext1),"v2":make_fn("v2",ext2)}
    graphs={};hold=[None]
    try:
        result["validation"]={name:validate(fn,w) for name,fn in funcs.items()}
        result["validation"]["incumbent"]=validate_baseline(B,state,w);save()
        data=reference.generate_input(128,"unpooler",5237);x=data["x"]
        out1=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda");out2=torch.empty_like(out1)
        def incumbent(xx,unused):
            hold[0]=B.kernel_fn({"x":xx,"config":data["config"]},state)
            return hold[0]
        graphs["incumbent"]=capture(incumbent,x,out1);baseline_out=hold[0]
        graphs["v1"]=capture(funcs["v1"],x,out1);graphs["v2"]=capture(funcs["v2"],x,out2)
        outputs={"incumbent":baseline_out,"v1":out1,"v2":out2}
        rp=state.get("rq",{}).get(("unpooler",128))
        if rp:
            names=B.RQPROD2 if "mod" in rp else B.RQPROD
            result["incumbent_pick"]={"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
        else:
            result["incumbent_pick"]={"kind":"actual_kernel_fn","Lt_code":state.get("lt",{}).get(("unpooler",128)),
                                      "sel_present":state.get("sel",{}).get(("unpooler",B._bk(128))) is not None}
        result["timing"]=measure(graphs);save()
        result["actual_timed_graph_checks"]=actual_graph_checks(graphs,outputs,x,data["config"],w)
        result["profile"]={name:profile(graph) for name,graph in graphs.items()}
        assert len(result["profile"]["v1"])==len(result["profile"]["v2"])==1
        assert "rq<" in result["profile"]["v1"][0]["name"] and "rq_v2<" in result["profile"]["v2"][0]["name"]
        assert torch.equal(w,weight_before) and torch.equal(lay,lay_before) and torch.equal(esc,esc_before)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp["cnt"].abs().sum())
        result["weights_inputs_counters"]="UNCHANGED_OR_RESET"
        timing=result["timing"]
        wins=all(timing[key]["incumbent"]>=.5 for key in ("F1_v2_gain_us","paired_v2_median_gain_us"))
        improves_v1=all(timing[key]["v1"]>0 for key in ("F1_v2_gain_us","paired_v2_median_gain_us"))
        result["status"]="INCUMBENT_GAIN_REVIEW_ONLY" if wins else "V2_IMPROVES_V1_DIAGNOSE_NEXT_LIMIT" if improves_v1 else "V2_LOADER_DIAGNOSIS_REQUIRED"
        result["next_measurement_question"]="If not incumbent-fast: distinguish scalar-X load issue/sector pressure from decoded-W production; no automatic variant/profile."
        save();print("V2_RESULT "+json.dumps({"status":result["status"],"timing":{k:v for k,v in timing.items() if "samples" not in k and "graph_us" not in k}}),flush=True)
    except Exception as error:
        result["status"]="INVALID_VERSION_RUNTIME_OR_CORRECTNESS";result["error"]=repr(error);save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset()
            graphs.clear()
        result["graph_cleanup"]="RESET_ALL_THREE";save()


if __name__=="__main__":main()


