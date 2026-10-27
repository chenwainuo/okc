"""Root-only five-arm native-X X4 plus STMatrix research driver; import creates no CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import subprocess
import shutil
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=ROOT/"candidates"
sys.path.insert(0,str(CANDIDATES/"reverse_tma_x4_stmatrix"))
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
    torch.cuda.synchronize();g.reset()
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


def actual_graph_checks(graphs, outputs, x, config, w, packed_workspaces, pack_index, immutable, counters, incumbent_counter):
    assert len({out.data_ptr() for out in outputs.values()}) == 5
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
            assert all(torch.equal(a,b) for a,b in immutable),"graph modified immutable tensors"
            assert all(not int(c.abs().sum()) for c in counters.values())
            if incumbent_counter is not None:assert not int(incumbent_counter.abs().sum())
            checks.append({"graph":label,"input":name,"packed_poison_overwritten_byte_exact":label in packed_workspaces, "NaN_poison_overwritten":True,
                           "input_immutable":True,"oracle":message})
    x.copy_(original)
    for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    return checks


def immediate_timed_output_checks(outputs, x, config, w, packed_workspaces, pack_index,
                                  fixtures, last_timed_input, final_record, immutable, counters, incumbent_counter):
    # This function performs reads/checks only: no X/out/XP write, graph replay,
    # output repair, profile, or recapture is permitted before this gate passes.
    assert set(last_timed_input)==set(outputs)=={"incumbent","native","x4","stmx","new"}
    assert torch.equal(x,fixtures[final_record["input_seed"]]),"timed X changed"
    checks=[]
    for label,out in outputs.items():
        seed=last_timed_input[label];fresh=fixtures[seed]
        assert bool(torch.isfinite(out).all()),(label,"nonfinite actual timed output")
        ok,message=reference.check_implementation({"x":fresh,"config":config},out,{"unpooler":w})
        assert ok,(label,"immediate actual timed output",seed,message)
        if label in packed_workspaces:
            assert torch.equal(packed_workspaces[label],fresh.view(torch.int32).reshape(-1)[pack_index]),(label,"actual timed XP",seed)
        checks.append({"graph":label,"actual_timed_output_before_any_replay_or_mutation":"PASS",
                       "input_seed":seed,"XP_current_recorded_fixture_exact":label in packed_workspaces,"oracle":message})
    assert all(torch.equal(a,b) for a,b in immutable),"W/packed/escape modified during timing"
    assert all(not int(c.abs().sum()) for c in counters.values()),"candidate counter not reset"
    if incumbent_counter is not None:assert not int(incumbent_counter.abs().sum()),"incumbent counter not reset"
    return {"checks":checks,"current_X_matches_last_actual_call_seed":final_record["input_seed"],
            "W_packed_escape_immutable_counters_zero":True,"no_output_repair_replay":True}


def measure(graphs,x,fixtures,strict_gate):
    labels=("incumbent","native","x4","stmx","new")
    flush=torch.empty(64<<20,dtype=torch.float32,device="cuda")
    sink=torch.empty((),dtype=torch.float32,device="cuda")
    calls=[];last_timed_input={};strict_checks={}
    def one(label,seed,mode,warmup=False):
        x.copy_(fixtures[seed])
        torch.sum(flush,0,out=sink)
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
        us=a.elapsed_time(b)*1000
        calls.append({"label":label,"input_seed":seed,"mode":mode,"warmup":warmup,"us":us})
        if not warmup:last_timed_input[label]=seed
        return us
    for label in labels:
        for _ in range(5):one(label,239999,"warmup",True)
    def series(label,block):return [one(label,240000+21*block+i,"F1") for i in range(21)]
    raw={};block=0
    for label in labels[:4]:
        raw[label+"_before"]=series(label,block);block+=1
    raw["new"]=series("new",block);block+=1
    for label in reversed(labels[:4]):
        raw[label+"_after"]=series(label,block);block+=1
    # All5 retained outputs can hold different fixtures. Read their exact recorded
    # fixture outputs now, before paired timing changes X or replays any graph.
    strict_checks["F1"]=strict_gate(fixtures,dict(last_timed_input),dict(calls[-1]))
    med={k:statistics.median(v) for k,v in raw.items()}
    f1={label:min(med[label+"_before"],med[label+"_after"])-med["new"] for label in labels[:4]}
    orders=[]
    for seq in [(0,1,2,3,4),(4,3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(5)])
    pairs=[]
    for repetition in range(40):
        row={};seed=250000+repetition
        for index in orders[repetition%len(orders)]:row[labels[index]]=one(labels[index],seed,"paired")
        pairs.append(row)
    strict_checks["paired"]=strict_gate(fixtures,dict(last_timed_input),dict(calls[-1]))
    paired={label:statistics.median(row[label]-row["new"] for row in pairs) for label in labels[:4]}
    return {"predeclared_paired_policy":"40 trials, 10 rotation/reverse orders each4 times; same fresh fixture for all5 arms of each round; report every sample",
            "F1_medians_us":med,"F1_samples_us":raw,"F1_new_gain_us":f1,
            "paired_graph_us":pairs,"paired_new_median_gain_us":paired,
            "paired_new_median_ratio":{label:statistics.median(row[label]/row["new"] for row in pairs) for label in labels[:4]},
            "F1_new_median_ratio":{label:min(med[label+"_before"],med[label+"_after"])/med["new"] for label in labels[:4]},
            "paired_new_wins":{label:sum(row[label]>row["new"] for row in pairs) for label in labels[:4]},
            "all_timed_and_warmup_calls":calls,"terminal_last_timed_input_seed_per_policy":dict(last_timed_input),
            "strict_immediate_actual_timed_output_checks":strict_checks}


def main():
    import importlib.util
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(CANDIDATES/"reverse_tma_x4_stmatrix_trial"))
    import native_x4_stmatrix_support as support
    support.verify_frozen()
    support.verify_lab()
    modules={label:support.load(folder,"prototype",label) for label,folder in
             (("native","reverse_tma_x"),("x4","reverse_tma_x4"),("stmx","reverse_tma_x_stmatrix"))}
    modules["new"]=P
    source_shas={label:hashlib.sha256(mod.source().encode()).hexdigest() for label,mod in modules.items()}
    for label,mod in modules.items():
        assert mod.source()==(CANDIDATES/support.FOLDERS[label]/"extension_source.cu").read_text(),(label,"generated source drift")
    extensions={label:mod.extension() for label,mod in modules.items()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    # Research CPU verification of exact loaded binaries before any candidate math.
    jit_records={};cuobjdump=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    for label in ("native","x4","stmx","new"):
        binary=Path(extensions[label].__file__).resolve();assert binary.is_file() and binary.suffix==".so"
        sass=subprocess.check_output([cuobjdump,"--dump-sass",str(binary)],text=True,timeout=20)
        artifact=args.json.with_suffix("."+label+".jit.sass");artifact.write_text(sass)
        record={"artifact":str(artifact),"sha256":hashlib.sha256(sass.encode()).hexdigest(),"module":str(binary),"module_sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
        try:
            # Each actual loaded .so is inspected before any candidate math.
            record.update(support.actual_binary_gate(label,sass))
        except Exception as error:
            args.json.write_text(json.dumps({"status":"INVALID_ACTUAL_JIT_RETIREMENT_GATE","failed_variant":label,
                "candidate_source_sha256":source_shas,"actual_JIT_SASS":{**jit_records,label:record},
                "error":repr(error),"no_candidate_math_launch":True,"graph_cleanup":"NO_GRAPHS_CREATED"},indent=2)+"\n")
            raise
        jit_records[label]=record
    print("ACTUAL_JIT_ALL_THREE_CANONICAL_MAIN_AND_LAYOUT_GATES PASS",flush=True)
    B,base_sha=baseline_module()
    result={"status":"STARTED_NO_TIMING_ADMISSION","parent_sha256":base_sha,"candidate_source_sha256":source_shas,
            "policy":"actual LOCAL pw36 incumbent, native-X parent, X4 depth, native STMatrix, X4 plus STMatrix combo; all current original input/noXP",
            "actual_JIT_SASS":jit_records,"attrs":{},"packer_attrs":{}}
    def save():args.json.write_text(json.dumps(result,indent=2)+"\n")
    empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
    try:
        for label,ext in extensions.items():
            def info(c):
                argv=[0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c]
                if label in ("native","x4","stmx","new"):argv.append(empty)
                return int(ext.rqrun(*argv))
            attrs={"regs":info(1),"local":info(2),"smem":info(3),"max_dynamic_before":info(5)}
            attrs["occupancy_after_optin"]=info(4);attrs["max_dynamic_after"]=info(5)
            result["attrs"][label]=attrs
            assert attrs["local"]==0 and attrs["regs"]<=128 and attrs["occupancy_after_optin"]==1
            assert attrs["regs"]==122
            assert attrs["max_dynamic_after"]>=attrs["smem"]==({"native":137232,"x4":170016,"stmx":137232,"new":170016}[label])
            threads=512
            attrs["threads"]=threads
            attrs["rounded_register_pool"]=((attrs["regs"]+7)//8)*8*threads
            assert attrs["rounded_register_pool"]<=65536
            if label in ("native","x4","stmx","new"):
                assert attrs["regs"]<=128 and info(6)==512
                attrs["max_threads"]=info(6)
            attrs["register_mode"]="uniform static"
            result["attrs"][label]=attrs
    except Exception as error:
        result.update(status="INVALID_ACTUAL_JIT_RESOURCE_GATE",error=repr(error),
                      no_candidate_math_launch=True,graph_cleanup="NO_GRAPHS_CREATED")
        save();raise
    weights=reference.generate_weights(reference.WEIGHTS_SEED);state=B.prepare(weights);w=weights["unpooler"]
    weight_snapshots={name:a.clone() for name,a in weights.items()}
    selected_rp=state.get("rq",{}).get(("unpooler",128))
    assert "cpu" not in state and selected_rp is not None and selected_rp["n"]==8192,"UNSUPPORTED_INCUMBENT_FALLBACK_NO_ADMISSION"
    actual_rqcall=B._rqcall
    def checked_actual_rqcall(st,p,xx,out):
        assert p is selected_rp,"incumbent serving policy changed"
        code=actual_rqcall(st,p,xx,out)
        assert code==0,("INCUMBENT_FALLBACK_NO_ADMISSION",code)
        return code
    B._rqcall=checked_actual_rqcall
    incumbent_immutable=[(selected_rp[key],selected_rp[key].clone()) for key in ("lay","esc")]
    weight_before=w.clone();lay,tab,esc6=P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda");lay_before=lay.clone();esc_before=esc.clone()
    counters={label:torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda") for label in extensions}
    packed_workspaces={}  # All four candidates consume current original X. No XP allocations.
    pack_index=None
    def make_fn(label,ext,workspace=None):
        def fn(x,out):
            assert x.is_contiguous() and out.is_contiguous()
            argv=[0,lay,x,out,esc,counters[label],1,tab,0,empty,sink,0]
            argv.append((packed_workspaces[label] if workspace is None else workspace) if label in packed_workspaces else empty)
            code=int(ext.rqrun(*argv))
            if code:raise RuntimeError(label+" launch failed "+str(code))
            return 0
        return fn
    funcs={label:make_fn(label,ext) for label,ext in extensions.items()};graphs={};hold=[None]
    try:
        result["GPU_pack_all65536_BF16_storage_bits"]={}  # No XP exists for any arm.
        result["native_variants_storage_bits_gate"]={"status":"NO_XP_BY_DESIGN_RAW_TMA_LDSM_BITS_UNPROBED", "scope":"Full basis/FP64 numeric tests and NVIDIA BF16 mover semantics; no hardware raw-bit roundtrip claimed"}
        result["validation"]={label:validate(fn,w) for label,fn in funcs.items()}
        result["validation"]["incumbent"]=validate_baseline(B,state,w);save()
        data=reference.generate_input(128,"unpooler",5237);x=data["x"];result["fresh_workspace_and_output"]={}
        for label in ("native","x4","stmx","new"):
            freshout=torch.full((128,8192),float("nan"),dtype=torch.bfloat16,device="cuda")
            funcs[label](x,freshout);torch.cuda.synchronize()
            ok,message=reference.check_implementation(data,freshout,{"unpooler":w});assert ok,message
            result["fresh_workspace_and_output"][label]={"status":"PASS","oracle":message,"workspace_bytes":0,"XP":"NONE"}
            del freshout
        candidate_outputs={label:torch.empty((128,8192),dtype=torch.bfloat16,device="cuda") for label in funcs}
        def incumbent(xx,unused):
            hold[0]=B.kernel_fn({"x":xx,"config":data["config"]},state);return hold[0]
        graphs["incumbent"]=capture(incumbent,x,candidate_outputs["native"]);baseline_out=hold[0]
        for label,fn in funcs.items():graphs[label]=capture(fn,x,candidate_outputs[label])
        outputs={"incumbent":baseline_out,**candidate_outputs}
        rp=state.get("rq",{}).get(("unpooler",128))
        if rp:
            names=B.RQPROD2 if "mod" in rp else B.RQPROD
            result["incumbent_pick"]={"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
        else:result["incumbent_pick"]={"kind":"actual_kernel_fn","Lt_code":state.get("lt",{}).get(("unpooler",128))}
        # Prepare fixture tensors before timing. Fixtures are test inputs only;
        # graphs still run freshpack+main on every call. No output state is cached.
        fixture_seeds=[239999]+list(range(240000,240000+9*21))+list(range(250000,250040))
        fixtures={seed:reference.generate_input(128,"unpooler",seed)["x"] for seed in fixture_seeds}
        immutable=[(a,weight_snapshots[name]) for name,a in weights.items()]+[(lay,lay_before),(esc,esc_before)]+incumbent_immutable
        def strict_gate(fixtures,last_timed_input,final_record):
            return immediate_timed_output_checks(outputs,x,data["config"],w,packed_workspaces,pack_index,
                fixtures,last_timed_input,final_record,immutable,counters,rp["cnt"] if rp else None)
        result["timing"]=measure(graphs,x,fixtures,strict_gate);save()
        result["actual_timed_graph_checks"]=actual_graph_checks(graphs,outputs,x,data["config"],w,packed_workspaces,pack_index,immutable,counters,rp["cnt"] if rp else None)
        assert all(torch.equal(a,b) for a,b in immutable),"mutation gates modified W/packed/escape"
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp["cnt"].abs().sum())
        result["profile"]={label:profile(graph) for label,graph in graphs.items()}
        for label,token in [("native","rq_tma_x<"),("x4","rq_tma_x4<"),("stmx","rq_tma_x_stmx<"),("new","rq_tma_x4_stmx<")]:
            events=result["profile"][label];assert len(events)==1,(label,events)
            assert not any("rx_pack" in event["name"] for event in events)
            assert sum(token in event["name"] for event in events)==1
        result["full_pipeline_in_timed_graph"]={label:"CURRENT_ORIGINAL_X_SINGLE_MAIN_1_NODE_NO_XP" for label in funcs}
        assert all(torch.equal(a,b) for a,b in immutable)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp["cnt"].abs().sum())
        result["weights_inputs_counters"]="UNCHANGED_OR_RESET";timing=result["timing"]
        wins=all(timing[key]["incumbent"]>=.5 for key in ("F1_new_gain_us","paired_new_median_gain_us"))
        progress=[label for label in ("native","x4","stmx") if all(timing[key][label]>0 for key in ("F1_new_gain_us","paired_new_median_gain_us"))]
        result["status"]="INCUMBENT_GAIN_REVIEW_ONLY" if wins else "NATIVE_X4_STMX_PROGRESS_DIAGNOSE_NEXT_LIMIT" if progress else "NATIVE_X4_STMX_DIAGNOSIS_REQUIRED"
        result["versions_improved_both_methods"]=progress
        result["next_measurement_question"]="Does stacking Xring4 and native STMatrix improve over both single edits, native parent and actual LOCAL incumbent? Retain all outcomes."
        save();print("NATIVE_X4_STMX_RESULT "+json.dumps({"status":result["status"],"timing":{k:v for k,v in timing.items() if "samples" not in k and "graph_us" not in k}}),flush=True)
    except Exception as error:
        result["status"]="INVALID_VERSION_RUNTIME_OR_CORRECTNESS";result["error"]=repr(error);save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset()
            graphs.clear()
        result["graph_cleanup"]="RESET_ALL_FIVE";save()


if __name__=="__main__":main()
