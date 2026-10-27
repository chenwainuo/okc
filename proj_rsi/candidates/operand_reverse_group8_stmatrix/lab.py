"""Group8 plus STMatrix composition: actual incumbent, DS, G8, new; every packed graph times fresh pack+main."""
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


def immediate_timed_output_checks(outputs, x, config, w, packed_workspaces, pack_index,
                                  fixtures, last_timed_input, final_record, immutable, counters, incumbent_counter):
    # This function performs reads/checks only: no X/out/XP write, graph replay,
    # output repair, profile, or recapture is permitted before this gate passes.
    assert set(last_timed_input)==set(outputs)=={"incumbent","ds","g8","new"}
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
    labels=("incumbent","ds","g8","new")
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
    for label in labels[:3]:
        raw[label+"_before"]=series(label,block);block+=1
    raw["new"]=series("new",block);block+=1
    for label in reversed(labels[:3]):
        raw[label+"_after"]=series(label,block);block+=1
    # All4 retained outputs can hold different fixtures. Read their exact recorded
    # fixture outputs now, before paired timing changes X or replays any graph.
    strict_checks["F1"]=strict_gate(fixtures,dict(last_timed_input),dict(calls[-1]))
    med={k:statistics.median(v) for k,v in raw.items()}
    f1={label:min(med[label+"_before"],med[label+"_after"])-med["new"] for label in labels[:3]}
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
    pairs=[]
    for repetition in range(32):
        row={};seed=250000+repetition
        for index in orders[repetition%len(orders)]:row[labels[index]]=one(labels[index],seed,"paired")
        pairs.append(row)
    strict_checks["paired"]=strict_gate(fixtures,dict(last_timed_input),dict(calls[-1]))
    paired={label:statistics.median(row[label]-row["new"] for row in pairs) for label in labels[:3]}
    return {"predeclared_paired_policy":"32 trials, 8 rotation/reverse orders each4 times; same fresh fixture for all4 arms of each pair; report every sample",
            "F1_medians_us":med,"F1_samples_us":raw,"F1_new_gain_us":f1,
            "paired_graph_us":pairs,"paired_new_median_gain_us":paired,
            "paired_new_wins":{label:sum(row[label]>row["new"] for row in pairs) for label in labels[:3]},
            "all_timed_and_warmup_calls":calls,"terminal_last_timed_input_seed_per_policy":dict(last_timed_input),
            "strict_immediate_actual_timed_output_checks":strict_checks}


def main():
    import importlib.util
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path,required=True);args=ap.parse_args()
    manifest=json.loads((P.HERE/"manifest.json").read_text())
    for name,sha in manifest["files"].items():
        assert hashlib.sha256((P.HERE/name).read_bytes()).hexdigest()==sha,("group8 drift",name)
    modules={};source_shas={}
    expected={"ds":"672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1",
              "g8":"e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e"}
    manifest_pins={"ds":"6785da2ec9881ce13148951daff2f8802d8e7bbe6c83abf588607092b3f97072",
                   "g8":"2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7"}
    for label,folder in [("ds","operand_reverse_dual_stmatrix"),("g8","operand_reverse_group8")]:
        old=P.HERE.parent/folder;frozen=json.loads((old/"manifest.json").read_text())
        assert hashlib.sha256((old/"manifest.json").read_bytes()).hexdigest()==manifest_pins[label]
        files=["prototype.py","kernel_body.cuh","extension_source.cu","device_source.cu","preflight.py","lab.py"]
        if label in ("ds","g8"):files.append("liveness.py")
        for name in files:
            assert hashlib.sha256((old/name).read_bytes()).hexdigest()==frozen["files"][name],(label+" drift",name)
        spec=importlib.util.spec_from_file_location("dual_control_"+label,old/"prototype.py")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        source_shas[label]=hashlib.sha256(mod.source().encode()).hexdigest();assert source_shas[label]==expected[label]
        modules[label]=mod
    source_shas["new"]=hashlib.sha256(P.source().encode()).hexdigest();assert source_shas["new"]==manifest["source_sha256"]
    modules["new"]=P
    extensions={label:mod.extension() for label,mod in modules.items()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    # Research CPU verification of exact loaded binaries before any candidate math.
    jit_records={};cuobjdump=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    for label in ("ds","g8","new"):
        binary=Path(extensions[label].__file__).resolve();assert binary.is_file() and binary.suffix==".so"
        sass=subprocess.check_output([cuobjdump,"--dump-sass",str(binary)],text=True,timeout=20)
        artifact=args.json.with_suffix("."+label+".jit.sass");artifact.write_text(sass)
        record={"artifact":str(artifact),"sha256":hashlib.sha256(sass.encode()).hexdigest(),"module":str(binary)}
        try:
            # Each actual loaded .so is inspected before any candidate math.
            checker=liveness
            if label in ("ds","g8"):
                folder="operand_reverse_dual_stmatrix" if label=="ds" else "operand_reverse_group8"
                spec=importlib.util.spec_from_file_location("group8_frozen_liveness_"+label,P.HERE.parent/folder/"liveness.py")
                checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
            record["retirement_and_fence_gate"]=checker.check(sass)
            # Independent retained store-form gate binds unchanged scalar DP/G8
            # producers and DS's distinct matrix-store control.
            spec=importlib.util.spec_from_file_location("group8_retained_store_form_"+label,P.HERE.parent/"operand_reverse_dual_stmatrix/liveness.py")
            storechecker=importlib.util.module_from_spec(spec);spec.loader.exec_module(storechecker)
            token={"ds":"rq_px_dp_stmx","g8":"rq_px_g8","new":"rq_px_g8_stmx"}[label]
            record["store_form_gate"]=storechecker.store_form(sass,token,label in ("ds","new"),dual=True)
        except Exception as error:
            args.json.write_text(json.dumps({"status":"INVALID_ACTUAL_JIT_RETIREMENT_GATE","failed_variant":label,
                "candidate_source_sha256":source_shas,"actual_JIT_SASS":{**jit_records,label:record},
                "error":repr(error),"no_candidate_math_launch":True,"graph_cleanup":"NO_GRAPHS_CREATED"},indent=2)+"\n")
            raise
        jit_records[label]=record
    print("ACTUAL_JIT_PHYSICAL_FENCE_AND_STORE_ALL_THREE_PACKED_GATES PASS",flush=True)
    B,base_sha=baseline_module()
    result={"status":"STARTED_NO_TIMING_ADMISSION","parent_sha256":base_sha,"candidate_source_sha256":source_shas,
            "policy":"G8 math exact plus DS matrix producer stores; W6B8/CSR/103600SMEM512threads/A2x32 unchanged from G8; actual incumbent/DS/G8/new",
            "actual_JIT_SASS":jit_records,"attrs":{},"packer_attrs":{}}
    def save():args.json.write_text(json.dumps(result,indent=2)+"\n")
    empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
    try:
        for label,ext in extensions.items():
            def info(c):
                argv=[0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c]
                if label in ("ds","g8","new"):argv.append(empty)
                return int(ext.rqrun(*argv))
            attrs={"regs":info(1),"local":info(2),"smem":info(3),"max_dynamic_before":info(5)}
            attrs["occupancy_after_optin"]=info(4);attrs["max_dynamic_after"]=info(5)
            result["attrs"][label]=attrs
            assert attrs["local"]==0 and attrs["regs"]<=128 and attrs["occupancy_after_optin"]>=1
            assert attrs["max_dynamic_after"]>=attrs["smem"]==103600
            threads=512
            attrs["threads"]=threads
            attrs["rounded_register_pool"]=((attrs["regs"]+7)//8)*8*threads
            assert attrs["rounded_register_pool"]<=65536
            if label in ("ds","g8","new"):
                assert attrs["regs"]<=128 and info(6)>=512
                attrs["max_threads"]=info(6)
            attrs["register_mode"]="uniform static"
            result["attrs"][label]=attrs
            if label in ("ds","g8","new"):
                packattrs={"regs":int(ext.packinfo(1)),"local":int(ext.packinfo(2)),"max_threads":int(ext.packinfo(3))}
                assert 0<packattrs["regs"]<=64 and packattrs["local"]==0 and packattrs["max_threads"]>=256
                result["packer_attrs"][label]=packattrs
    except Exception as error:
        result.update(status="INVALID_ACTUAL_JIT_RESOURCE_GATE",error=repr(error),
                      no_candidate_math_launch=True,graph_cleanup="NO_GRAPHS_CREATED")
        save();raise
    weights=reference.generate_weights(reference.WEIGHTS_SEED);state=B.prepare(weights);w=weights["unpooler"]
    weight_before=w.clone();lay,tab,esc6=P.S._rqpack(w)
    assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
    esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,"cuda");lay_before=lay.clone();esc_before=esc.clone()
    counters={label:torch.zeros(4*lay.shape[0],dtype=torch.int32,device="cuda") for label in extensions}
    packed_workspaces={label:torch.empty(262144,dtype=torch.int32,device="cuda") for label in ("ds","g8","new")}
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
        graphs["incumbent"]=capture(incumbent,x,candidate_outputs["g8"]);baseline_out=hold[0]
        for label,fn in funcs.items():graphs[label]=capture(fn,x,candidate_outputs[label])
        outputs={"incumbent":baseline_out,**candidate_outputs}
        rp=state.get("rq",{}).get(("unpooler",128))
        if rp:
            names=B.RQPROD2 if "mod" in rp else B.RQPROD
            result["incumbent_pick"]={"kind":"rq","variant":names[rp["i"]],"KS":rp["KS"]}
        else:result["incumbent_pick"]={"kind":"actual_kernel_fn","Lt_code":state.get("lt",{}).get(("unpooler",128))}
        # Prepare fixture tensors before timing. Fixtures are test inputs only;
        # graphs still run freshpack+main on every call. No output state is cached.
        fixture_seeds=[239999]+list(range(240000,240000+7*21))+list(range(250000,250032))
        fixtures={seed:reference.generate_input(128,"unpooler",seed)["x"] for seed in fixture_seeds}
        immutable=[(w,weight_before),(lay,lay_before),(esc,esc_before)]
        def strict_gate(fixtures,last_timed_input,final_record):
            return immediate_timed_output_checks(outputs,x,data["config"],w,packed_workspaces,pack_index,
                fixtures,last_timed_input,final_record,immutable,counters,rp["cnt"] if rp else None)
        result["timing"]=measure(graphs,x,fixtures,strict_gate);save()
        result["actual_timed_graph_checks"]=actual_graph_checks(graphs,outputs,x,data["config"],w,packed_workspaces,pack_index)
        result["profile"]={label:profile(graph) for label,graph in graphs.items()}
        for label,token in [("ds","rq_px_dp_stmx<"),("g8","rq_px_g8<"),("new","rq_px_g8_stmx<")]:
            events=result["profile"][label];assert len(events)==2
            assert sum("rx_pack" in event["name"] for event in events)==1
            assert sum(token in event["name"] for event in events)==1
        result["full_pipeline_in_timed_graph"]="FRESH_PACK_PLUS_MAIN_2_NODES_EVERY_CALL_FOR_ALL_THREE_PACKED_VARIANTS"
        assert torch.equal(w,weight_before) and torch.equal(lay,lay_before) and torch.equal(esc,esc_before)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp["cnt"].abs().sum())
        result["weights_inputs_counters"]="UNCHANGED_OR_RESET";timing=result["timing"]
        wins=all(timing[key]["incumbent"]>=.5 for key in ("F1_new_gain_us","paired_new_median_gain_us"))
        progress=[label for label in ("ds","g8") if all(timing[key][label]>0 for key in ("F1_new_gain_us","paired_new_median_gain_us"))]
        result["status"]="INCUMBENT_GAIN_REVIEW_ONLY" if wins else "G8_STMX_PROGRESS_DIAGNOSE_NEXT_LIMIT" if progress else "G8_STMX_DIAGNOSIS_REQUIRED"
        result["versions_improved_both_methods"]=progress
        result["next_measurement_question"]="Single composition adds DS STMatrix producer stores to exact G8 math. Compare both parents and incumbent; retain failure and diagnose phase interaction without family closure."
        save();print("G8_STMX_RESULT "+json.dumps({"status":result["status"],"timing":{k:v for k,v in timing.items() if "samples" not in k and "graph_us" not in k}}),flush=True)
    except Exception as error:
        result["status"]="INVALID_VERSION_RUNTIME_OR_CORRECTNESS";result["error"]=repr(error);save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset()
            graphs.clear()
        result["graph_cleanup"]="RESET_ALL_FOUR";save()


if __name__=="__main__":main()
