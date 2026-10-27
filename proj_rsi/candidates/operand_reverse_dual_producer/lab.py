"""One producer alternating dual-producer version; all three packed controls time fresh pack + main."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import subprocess
import shutil
import liveness

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


def actual_graph_checks(graphs, outputs, x, config, w, packed_workspaces, pack_index):
    assert len({out.data_ptr() for out in outputs.values()}) == 4
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
            if label in packed_workspaces:packed_workspaces[label].fill_(0x5a39c7e1)
            graph.replay(); torch.cuda.synchronize()
            assert torch.equal(x, before) and bool(torch.isfinite(outputs[label]).all()), (label,name)
            if name == "zero":
                assert not int(torch.count_nonzero(outputs[label])); message = "exact zero complete output"
            else:
                ok, message = reference.check_implementation({"x":x,"config":config},outputs[label],{"unpooler":w})
                assert ok, (label,name,message)
            if label in packed_workspaces:assert torch.equal(packed_workspaces[label],x.view(torch.int32).reshape(-1)[pack_index]),("stale/partial packed workspace",label,name)
            checks.append({"graph":label,"input":name,"packed_poison_overwritten_byte_exact":label in packed_workspaces, "NaN_poison_overwritten":True,
                           "input_immutable":True,"oracle":message})
    x.copy_(original)
    for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    return checks


def measure(graphs):
    labels=("incumbent","px","smx","dp")
    flush=torch.empty(64<<20,dtype=torch.float32,device="cuda")
    sink=torch.empty((),dtype=torch.float32,device="cuda")
    def one(label):
        torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        return a.elapsed_time(b)*1000
    for label in labels:
        for _ in range(5):one(label)
    series=lambda label:[one(label) for _ in range(21)]
    raw={}
    for label in labels[:3]:raw[label+"_before"]=series(label)
    raw["dp"]=series("dp")
    for label in reversed(labels[:3]):raw[label+"_after"]=series(label)
    med={k:statistics.median(v) for k,v in raw.items()}
    f1={label:min(med[label+"_before"],med[label+"_after"])-med["dp"] for label in labels[:3]}
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
    pairs=[]
    for repetition in range(21):
        row={}
        for index in orders[repetition%len(orders)]:row[labels[index]]=one(labels[index])
        pairs.append(row)
    paired={label:statistics.median(row[label]-row["dp"] for row in pairs) for label in labels[:3]}
    return {"F1_medians_us":med,"F1_samples_us":raw,"F1_dp_gain_us":f1,
            "paired_graph_us":pairs,"paired_dp_median_gain_us":paired,
            "paired_dp_wins":{label:sum(row[label]>row["dp"] for row in pairs) for label in labels[:3]}}


def main():
    import importlib.util
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    manifest=json.loads((P.HERE/"manifest.json").read_text())
    for name,sha in manifest["files"].items():
        assert hashlib.sha256((P.HERE/name).read_bytes()).hexdigest()==sha,("dp drift",name)
    modules={};source_shas={}
    expected={"px":"2c28de068a57235fcfec8af61a5678572ef76dc9833bc280e134623e8b6ff04a",
              "smx":"cc4f5189bf0018a1e48941377b639b9c0983ffc2eb5cb6e44e6fa105f7d20b38"}
    for label,folder in [("px","operand_reverse_packed_x"),("smx","operand_reverse_stmatrix")]:
        old=P.HERE.parent/folder;frozen=json.loads((old/"manifest.json").read_text())
        files=["prototype.py","kernel_body.cuh","extension_source.cu","device_source.cu","preflight.py","lab.py"]
        if label in ("px","smx"):files.append("liveness.py")
        for name in files:
            assert hashlib.sha256((old/name).read_bytes()).hexdigest()==frozen["files"][name],(label+" drift",name)
        spec=importlib.util.spec_from_file_location("dual_control_"+label,old/"prototype.py")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        source_shas[label]=hashlib.sha256(mod.source().encode()).hexdigest();assert source_shas[label]==expected[label]
        modules[label]=mod
    source_shas["dp"]=hashlib.sha256(P.source().encode()).hexdigest();assert source_shas["dp"]==manifest["source_sha256"]
    modules["dp"]=P
    extensions={label:mod.extension() for label,mod in modules.items()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    # Research CPU verification of exact loaded binaries before any candidate math.
    jit_records={};cuobjdump=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    for label in ("px","smx","dp"):
        binary=Path(extensions[label].__file__).resolve();assert binary.is_file() and binary.suffix==".so"
        sass=subprocess.check_output([cuobjdump,"--dump-sass",str(binary)],text=True,timeout=20)
        artifact=args.json.with_suffix("."+label+".jit.sass");artifact.write_text(sass)
        record={"artifact":str(artifact),"sha256":hashlib.sha256(sass.encode()).hexdigest(),"module":str(binary)}
        try:
            checker=liveness
            if label=="smx":
                spec=importlib.util.spec_from_file_location("dual_frozen_smx_liveness",P.HERE.parent/"operand_reverse_stmatrix/liveness.py")
                checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
            record["retirement_gate"]=checker.check(sass)
            if label=="smx":record["store_form_gate"]=checker.store_form(sass,"rq_stmx",True)
        except Exception as error:
            args.json.write_text(json.dumps({"status":"INVALID_ACTUAL_JIT_RETIREMENT_GATE","failed_variant":label,
                "candidate_source_sha256":source_shas,"actual_JIT_SASS":{**jit_records,label:record},
                "error":repr(error),"no_candidate_math_launch":True,"graph_cleanup":"NO_GRAPHS_CREATED"},indent=2)+"\n")
            raise
        jit_records[label]=record
    print("ACTUAL_JIT_PHYSICAL_A4_ALL_THREE_PACKED_GATES PASS",flush=True)
    B,base_sha=baseline_module()
    result={"status":"STARTED_NO_TIMING_ADMISSION","parent_sha256":base_sha,"candidate_source_sha256":source_shas,
            "policy":"512threads/two alternating producerWGs with separatebar3/4+CSRskip; sameWpack/scalarstores/mathA4/cache/pack/output",
            "actual_JIT_SASS":jit_records,"attrs":{},"packer_attrs":{}}
    def save():args.json.write_text(json.dumps(result,indent=2)+"\n")
    empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
    for label,ext in extensions.items():
        def info(c):
            argv=[0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c]
            if label in ("px","smx","dp"):argv.append(empty)
            return int(ext.rqrun(*argv))
        attrs={"regs":info(1),"local":info(2),"smem":info(3),"max_dynamic_before":info(5)}
        attrs["occupancy_after_optin"]=info(4);attrs["max_dynamic_after"]=info(5)
        assert attrs["local"]==0 and attrs["regs"]<=168 and attrs["occupancy_after_optin"]>=1
        assert attrs["max_dynamic_after"]>=attrs["smem"]==103600
        threads=512 if label=="dp" else 384
        attrs["threads"]=threads
        attrs["rounded_register_pool"]=((attrs["regs"]+7)//8)*8*threads
        assert attrs["rounded_register_pool"]<=65536
        if label=="dp":
            assert attrs["regs"]<=128 and info(6)>=512
            attrs["max_threads"]=info(6)
        attrs["register_mode"]="uniform static"
        result["attrs"][label]=attrs
        if label in ("px","smx","dp"):
            packattrs={"regs":int(ext.packinfo(1)),"local":int(ext.packinfo(2)),"max_threads":int(ext.packinfo(3))}
            assert 0<packattrs["regs"]<=64 and packattrs["local"]==0 and packattrs["max_threads"]>=256
            result["packer_attrs"][label]=packattrs
    weights=reference.generate_weights(reference.WEIGHTS_SEED);state=B.prepare(weights);w=weights["unpooler"]
    weight_before=w.clone();lay,tab,esc6=P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda");lay_before=lay.clone();esc_before=esc.clone()
    counters={label:torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda") for label in extensions}
    packed_workspaces={label:torch.empty(262144,dtype=torch.int32,device="cuda") for label in ("px","smx","dp")}
    assert len({a.data_ptr() for a in packed_workspaces.values()})==3
    # Shape-only oracle index metadata, never passed to the kernels.
    idx=torch.arange(262144,dtype=torch.int64);vv=idx&3;ll=(idx>>2)&31;jj=(idx>>7)&3;ww=(idx>>9)&7;ss=idx>>12
    rr=64*(ww//4)+16*(ww%4)+(ll>>2)+8*(vv&1);kk=64*ss+16*jj+8*((vv>>1)&1)+2*(ll&3)
    pack_index=(rr*2048+kk//2).cuda()
    def make_fn(label,ext,workspace=None):
        def fn(x,out):
            assert x.is_contiguous() and out.is_contiguous()
            argv=[0,lay,x,out,esc,counters[label],1,tab,0,empty,sink,0]
            if label in packed_workspaces:argv.append(packed_workspaces[label] if workspace is None else workspace)
            code=int(ext.rqrun(*argv))
            if code:raise RuntimeError(label+" launch failed "+str(code))
            return 0
        return fn
    funcs={label:make_fn(label,ext) for label,ext in extensions.items()};graphs={};hold=[None]
    try:
        bitx=(torch.arange(524288,dtype=torch.int32,device="cuda")&65535).to(torch.int16).view(torch.bfloat16).reshape(128,4096)
        bit_before=bitx.view(torch.int16).clone();bitout=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda")
        result["GPU_pack_all65536_BF16_storage_bits"]={}
        for label in packed_workspaces:
            packed_workspaces[label].fill_(0x37a8f219);funcs[label](bitx,bitout);torch.cuda.synchronize()
            assert torch.equal(bitx.view(torch.int16),bit_before)
            assert torch.equal(packed_workspaces[label],bitx.view(torch.int32).reshape(-1)[pack_index])
            result["GPU_pack_all65536_BF16_storage_bits"][label]="PASS_1MiB_EXACT_INPUT_IMMUTABLE_STORAGE_ONLY"
        del bitx,bit_before,bitout
        result["validation"]={label:validate(fn,w) for label,fn in funcs.items()}
        result["validation"]["incumbent"]=validate_baseline(B,state,w);save()
        data=reference.generate_input(128,"unpooler",5237);x=data["x"];result["fresh_workspace_and_output"]={}
        for label in packed_workspaces:
            freshpack=torch.empty_like(packed_workspaces[label]);freshout=torch.empty((128,8192),dtype=torch.bfloat16,device="cuda")
            freshpack.fill_(0x37a8f219);freshout.fill_(float("nan"));make_fn(label,extensions[label],freshpack)(x,freshout)
            torch.cuda.synchronize();assert torch.equal(freshpack,x.view(torch.int32).reshape(-1)[pack_index])
            ok,message=reference.check_implementation(data,freshout,{"unpooler":w});assert ok,message
            result["fresh_workspace_and_output"][label]={"status":"PASS","oracle":message,"workspace_bytes":1048576}
            del freshpack,freshout
        candidate_outputs={label:torch.empty((128,8192),dtype=torch.bfloat16,device="cuda") for label in funcs}
        def incumbent(xx,unused):
            hold[0]=B.kernel_fn({"x":xx,"config":data["config"]},state);return hold[0]
        graphs["incumbent"]=capture(incumbent,x,candidate_outputs["smx"]);baseline_out=hold[0]
        for label,fn in funcs.items():graphs[label]=capture(fn,x,candidate_outputs[label])
        outputs={"incumbent":baseline_out,**candidate_outputs}
        rp=state.get("rq",{}).get(("unpooler",128))
        if rp:
            names=B.RQPROD2 if "mod" in rp else B.RQPROD
            result["incumbent_pick"]={"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
        else:result["incumbent_pick"]={"kind":"actual_kernel_fn","Lt_code":state.get("lt",{}).get(("unpooler",128))}
        result["timing"]=measure(graphs);save()
        result["actual_timed_graph_checks"]=actual_graph_checks(graphs,outputs,x,data["config"],w,packed_workspaces,pack_index)
        result["profile"]={label:profile(graph) for label,graph in graphs.items()}
        for label,token in [("px","rq_px<"),("smx","rq_stmx<"),("dp","rq_px_dp<")]:
            events=result["profile"][label];assert len(events)==2
            assert sum("rx_pack" in event["name"] for event in events)==1
            assert sum(token in event["name"] for event in events)==1
        result["full_pipeline_in_timed_graph"]="FRESH_PACK_PLUS_MAIN_2_NODES_EVERY_CALL_FOR_ALL_THREE_PACKED_VARIANTS"
        assert torch.equal(w,weight_before) and torch.equal(lay,lay_before) and torch.equal(esc,esc_before)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp["cnt"].abs().sum())
        result["weights_inputs_counters"]="UNCHANGED_OR_RESET";timing=result["timing"]
        wins=all(timing[key]["incumbent"]>=.5 for key in ("F1_dp_gain_us","paired_dp_median_gain_us"))
        progress=[label for label in ("px","smx") if all(timing[key][label]>0 for key in ("F1_dp_gain_us","paired_dp_median_gain_us"))]
        result["status"]="INCUMBENT_GAIN_REVIEW_ONLY" if wins else "DP_PROGRESS_DIAGNOSE_NEXT_LIMIT" if progress else "DP_STORE_DIAGNOSIS_REQUIRED"
        result["versions_improved_both_methods"]=progress
        result["next_measurement_question"]="Two producers may advanceBpublication; sameWbytes/scalarstores, extraCSRtraversal/registers/threads/bar4. Comparepredecessorandincumbent; diagnoseresidualphase, no familyclosure."
        save();print("DP_RESULT "+json.dumps({"status":result["status"],"timing":{k:v for k,v in timing.items() if "samples" not in k and "graph_us" not in k}}),flush=True)
    except Exception as error:
        result["status"]="INVALID_VERSION_RUNTIME_OR_CORRECTNESS";result["error"]=repr(error);save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset()
            graphs.clear()
        result["graph_cleanup"]="RESET_ALL_FOUR";save()


if __name__=="__main__":main()
