"""Fixed DP/DS measurement diagnostic; no kernel edits, parent prepare or score claim.

Root only after independent review. Suggested inner90s/outer120s, normal GPU.
Every timed graph contains fresh1MiB pack+main. Primary64 paired comparisons
follow TASK's cold-L2/taking-turns/swapped-order contract; steady blocks diagnose
switch sensitivity. Exact server estimator/overhead calibration is unavailable.
All raw times are full replay times, with no overhead subtraction or selection.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"proj_rsi/candidates"
FROZEN={
    "dp":("operand_reverse_dual_producer","281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241",
          "97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b","rq_px_dp"),
    "ds":("operand_reverse_dual_stmatrix","6785da2ec9881ce13148951daff2f8802d8e7bbe6c83abf588607092b3f97072",
          "672dab448f74dbeecccd60b4070e8a6686c0533ae69f5827afd5a4283bb0cbd1","rq_px_dp_stmx"),
}
ORDER_SEED=270151
BLOCK_SAMPLES=32
PAIR_COUNT=64
WARMUPS_PER_LABEL=5


def plan():
    directions=[0]*16+[1]*16
    random.Random(ORDER_SEED).shuffle(directions)
    paired=[]
    for block,direction in enumerate(directions):
        first=("dp","ds") if direction==0 else ("ds","dp")
        for within,order in enumerate((first,first[::-1])):
            paired.append({"pair":2*block+within,"swapped_block":block,
                           "within_swapped_block":within,"order":list(order),
                           "input_seed":110000+2*block+within})
    assert len(paired)==PAIR_COUNT
    assert sum(p["order"]==["dp","ds"] for p in paired)==32
    assert all(paired[2*b]["order"]==paired[2*b+1]["order"][::-1] for b in range(32))
    return {"order_seed":ORDER_SEED,"pair_count":PAIR_COUNT,"paired_orders":paired,
            "steady_before":["dp","ds","dp"],"steady_after":["ds","dp","ds"],
            "steady_samples_per_block":BLOCK_SAMPLES,"warmups_per_label_per_mode":WARMUPS_PER_LABEL,
            "timed_replays":6*BLOCK_SAMPLES+2*PAIR_COUNT,
            "flush_bytes":256<<20,"timed_pipeline":"freshpack+main2nodes",
            "primary":"64 cold paired DP/DS comparisons;32 adjacent swapped-order blocks",
            "secondary":"all6 steady medians and two bracketed effects; never best-block selection",
            "ratios":"report median64rawDP/DS and median32swapped-geometric-ratios explicitly; no calibrated replay-overhead correction",
            "estimator_scope":"contract-aligned local estimators; exact server two-ratio combination and overhead calibration unavailable",
            "interpretation":"paired representative of localTASK; steady-only gain is sensitivity, not promotion; no incumbent/official claim"}


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(sass,token):
    tables=[t for t in re.split(r"\n\s*Function : ",sass)[1:] if token in t.splitlines()[0]]
    assert len(tables)==1,(token,len(tables))
    return [(int(m[1],16),m[2].strip()) for m in re.finditer(r"/\*([0-9a-f]+)\*/\s+([^\n]*?);",tables[0])]


def summarize(records):
    timed=[r for r in records if not r["warmup"]]
    pairs=[]
    for i in range(PAIR_COUNT):
        row=[r for r in timed if r["mode"]=="paired" and r["pair"]==i]
        assert len(row)==2 and {r["label"] for r in row}=={"dp","ds"}
        values={r["label"]:r["us"] for r in row}
        pairs.append({"pair":i,"order":[r["label"] for r in row],"dp_us":values["dp"],"ds_us":values["ds"],
                      "gain_us":values["dp"]-values["ds"],"ratio_DP_over_DS":values["dp"]/values["ds"]})
    swapped=[(pairs[2*b]["ratio_DP_over_DS"]*pairs[2*b+1]["ratio_DP_over_DS"])**.5 for b in range(32)]
    medians={mode:[statistics.median(r["us"] for r in timed if r["mode"]==mode and r["block"]==b) for b in range(3)]
             for mode in ("steady_before","steady_after")}
    before,after=medians["steady_before"],medians["steady_after"]
    by_position={}
    for order in (("dp","ds"),("ds","dp")):
        chosen=[p for p in pairs if tuple(p["order"])==order]
        by_position["_".join(order)]={"n":len(chosen),"median_gain_us":statistics.median(p["gain_us"] for p in chosen)}
    steady_positions={}
    for mode in ("steady_before","steady_after"):
        steady_positions[mode]=[]
        for b in range(3):
            samples=[r for r in timed if r["mode"]==mode and r["block"]==b]
            steady_positions[mode].append({"label":samples[0]["label"],"first_sample_us":samples[0]["us"],
                "first_sample_previous_case":samples[0]["previous_case"],
                "first_sample_preceding_same_variant_run_length":samples[0]["preceding_same_variant_run_length"],
                "median_remaining31_us":statistics.median(r["us"] for r in samples[1:]),
                "median_first4_us":statistics.median(r["us"] for r in samples[:4]),
                "median_last28_us":statistics.median(r["us"] for r in samples[4:])})
    return {"paired_comparisons":pairs,"paired_median_gain_us":statistics.median(p["gain_us"] for p in pairs),
            "paired_wins":sum(p["gain_us"]>0 for p in pairs),"paired_losses":sum(p["gain_us"]<0 for p in pairs),
            "median64_raw_ratios_DP_over_DS":statistics.median(p["ratio_DP_over_DS"] for p in pairs),
            "swapped_block_geometric_ratios":swapped,"median32_swapped_geometric_ratios":statistics.median(swapped),
            "by_order":by_position,"steady_block_medians_us":medians,
            "steady_positions_descriptive_only":steady_positions,
            "bracketed_effects_us":{"before_mean_of_DP_brackets_minus_DS":(before[0]+before[2])/2-before[1],
                                    "after_DP_minus_mean_of_DS_brackets":after[1]-(after[0]+after[2])/2},
            "no_incumbent_or_official_claim":True,"no_best_block_or_sample_selection":True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--json",type=Path);ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args();frozen_plan=plan()
    if args.plan_only:
        print(json.dumps(frozen_plan,indent=2));return
    assert args.json is not None,"--json is required for GPU mode"
    import torch
    sys.path.insert(0,str(ROOT/"muse_bf16_proj"));import reference
    args.json.parent.mkdir(parents=True,exist_ok=True)
    result={"status":"STARTED","driver_sha256":sha(__file__),"plan":frozen_plan,"raw":[],"validation":{},"phase":"SOURCE"}
    graphs={};tracker={"previous_label":None,"same_variant_run_length":0,"previous_timed_label":None}
    last_timed_input={}
    def save():args.json.write_text(json.dumps(result,indent=2)+"\n")
    try:
        modules={};extensions={};result["frozen"]={}
        for label,(folder,manifest_sha,source_sha,token) in FROZEN.items():
            here=BASE/folder;assert sha(here/"manifest.json")==manifest_sha
            manifest=json.loads((here/"manifest.json").read_text())
            assert all(sha(here/name)==expected for name,expected in manifest["files"].items()),label
            mod=load(here/"prototype.py","measurement_frozen_"+label)
            assert hashlib.sha256(mod.source().encode()).hexdigest()==source_sha
            modules[label]=mod;extensions[label]=mod.extension()
            result["frozen"][label]={"manifest_sha256":manifest_sha,"source_sha256":source_sha}
        checker=load(BASE/"operand_reverse_dual_stmatrix/liveness.py","measurement_strict_A4_stores")
        result["phase"]="ACTUAL_JIT_BEFORE_ANY_MATH";result["actual_JIT"]={}
        for label,ext in extensions.items():
            here=BASE/FROZEN[label][0];binary=Path(ext.__file__).resolve()
            assert binary.is_file() and binary.suffix==".so"
            sass=subprocess.check_output([shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump","--dump-sass",str(binary)],text=True,timeout=20)
            artifact=args.json.with_suffix("."+label+".jit.sass");artifact.write_text(sass)
            record={"artifact":str(artifact),"sha256":hashlib.sha256(sass.encode()).hexdigest(),"module":str(binary),"module_sha256":sha(binary)}
            result["actual_JIT"][label]=record;save() # retain drift before gate
            record["A4"]=checker.check(sass)
            record["store_proxy_both_owner_barriers"]=checker.store_form(sass,FROZEN[label][3],label=="ds",dual=True)
            for token in (FROZEN[label][3],"rx_pack"):
                assert canonical(sass,token)==canonical((here/"sass.txt").read_text(),token),(label,token,"actualJITdrift")
            record["exact_frozen_PC_instruction_identity_main_and_pack"]=True
        result["phase"]="RESOURCE_GATE_BEFORE_MATH"
        empty=torch.empty(0,dtype=torch.int64,device="cuda");sink=torch.zeros(4,dtype=torch.int32,device="cuda")
        result["attrs"]={}
        for label,ext in extensions.items():
            def info(c):return int(ext.rqrun(0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c,empty))
            attrs={"regs":info(1),"local":info(2),"smem":info(3),"occupancy":info(4),"max_dynamic_after":info(5),"max_threads":info(6),
                   "pack_regs":int(ext.packinfo(1)),"pack_local":int(ext.packinfo(2)),"pack_max_threads":int(ext.packinfo(3))}
            assert attrs["regs"]==122 and attrs["local"]==0 and attrs["smem"]==103600 and attrs["occupancy"]>=1
            assert attrs["max_dynamic_after"]>=103600 and attrs["max_threads"]>=512
            assert ((attrs["regs"]+7)//8)*8*512<=65536
            assert attrs["pack_regs"]==22 and attrs["pack_local"]==0 and attrs["pack_max_threads"]>=256
            result["attrs"][label]=attrs
        S=modules["dp"].S
        weights=reference.generate_weights(reference.WEIGHTS_SEED);w=weights["unpooler"]
        packed,table,esc6=S._rqpack(w)
        assert torch.equal(S._rqunpack(packed,table,esc6,*w.shape),w.view(torch.int16).int()&65535)
        esc,_=S._rqesc(esc6,packed.shape[0],packed.shape[1],1,"cuda")
        immutable=[(a,a.clone()) for a in (w,packed,esc)]
        data=reference.generate_input(128,"unpooler",5237);x=data["x"].clone();original=x.clone();config=data["config"]
        xp={l:torch.empty(262144,dtype=torch.int32,device="cuda") for l in FROZEN}
        out={l:torch.empty((128,8192),dtype=torch.bfloat16,device="cuda") for l in FROZEN}
        cnt={l:torch.zeros(4*packed.shape[0],dtype=torch.int32,device="cuda") for l in FROZEN}
        assert len({a.data_ptr() for a in [x,*xp.values(),*out.values()]})==5
        def launch(label,workspace=None,output=None):
            code=int(extensions[label].rqrun(0,packed,x,out[label] if output is None else output,esc,cnt[label],1,table,0,empty,sink,0,xp[label] if workspace is None else workspace))
            assert code==0,(label,code)
        result["phase"]="CAPTURE_AND_INITIAL_VALIDATION"
        for label in FROZEN:
            launch(label);launch(label);torch.cuda.synchronize()
            graphs[label]=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[label]):launch(label)
        idx=torch.arange(262144,dtype=torch.int64,device="cuda");v=idx&3;lane=(idx>>2)&31;j=(idx>>7)&3;warp=(idx>>9)&7;s=idx>>12
        row=64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1);k=64*s+16*j+8*((v>>1)&1)+2*(lane&3)
        pack_index=row*2048+k//2
        def mark(label,timed=False):
            tracker["same_variant_run_length"]=tracker["same_variant_run_length"]+1 if tracker["previous_label"]==label else 1
            tracker["previous_label"]=label
            if timed:tracker["previous_timed_label"]=label
        def check_state():
            assert all(torch.equal(a,b) for a,b in immutable)
            assert all(not int(c.abs().sum()) for c in cnt.values())
        def validate_mode(mode):
            checks=[]
            if mode!="initial":
                # Check the ACTUAL final timed outputs before poisoning, changing
                # X or replaying anything. Steady controls can hold different
                # last fixtures, so use each policy's recorded seed.
                final=result["raw"][-1]
                assert not final["warmup"] and torch.equal(x,fixtures[final["input_seed"]])
                for label in FROZEN:
                    seed=last_timed_input[label];fresh=fixtures[seed]
                    assert torch.equal(xp[label],fresh.view(torch.int32).reshape(-1)[pack_index])
                    ok,msg=reference.check_implementation({"x":fresh,"config":config},out[label],{"unpooler":w})
                    assert ok,(mode,label,"immediate actual timed output",msg)
                    checks.append({"label":label,"actual_timed_output_before_any_replay_or_mutation":"PASS",
                                   "input_seed":seed,"XP_current_recorded_input_exact":True,"oracle":msg})
                check_state()
            bitx=(torch.arange(524288,dtype=torch.int32,device="cuda")&65535).to(torch.int16).view(torch.bfloat16).reshape(128,4096)
            x.copy_(bitx);before=x.view(torch.int16).clone()
            for label,g in graphs.items():
                xp[label].fill_(0x37a8f219);g.replay();torch.cuda.synchronize();mark(label)
                assert torch.equal(x.view(torch.int16),before) and torch.equal(xp[label],x.view(torch.int32).reshape(-1)[pack_index])
                checks.append({"label":label,"storage_all65536bits":"PASS","output_not_used_for_nonfinite_storage_test":True})
            del bitx,before
            probes=torch.zeros_like(x);rows=torch.arange(128,device="cuda");cols=(rows*37+19)%4096
            cols[:4]=torch.tensor([0,63,64,4095],device="cuda");powers=((2.**((rows%5)+3))*(2*(rows%2)-1)).to(torch.bfloat16)
            probes[rows,cols]=powers
            mutations=[("original",original),("zero",torch.zeros_like(x)),
                       ("fresh_normal",reference.generate_input(128,"unpooler",9951+len(result["validation"])*2)["x"]),
                       ("fresh_outliers",reference.generate_input(128,"unpooler",9952+len(result["validation"])*2,dist="outliers")["x"]),
                       ("all_probes",probes)]
            for kind,fresh in mutations:
                x.copy_(fresh);before=x.clone()
                for label,g in graphs.items():
                    xp[label].fill_(0x5a39c7e1);out[label].fill_(float("nan"));g.replay();torch.cuda.synchronize();mark(label)
                    assert torch.equal(x,before) and torch.equal(xp[label],x.view(torch.int32).reshape(-1)[pack_index])
                    assert bool(torch.isfinite(out[label]).all())
                    if kind=="zero":assert not int(torch.count_nonzero(out[label]));message="EXACT_ZERO"
                    elif kind=="all_probes":
                        expected=(w[:,cols].T.float()*powers[:,None].float()).to(torch.bfloat16)
                        assert torch.equal(out[label],expected);message="EXACT128_ONEHOT_ROWS"
                    else:
                        ok,message=reference.check_implementation({"x":x,"config":config},out[label],{"unpooler":w});assert ok,(label,kind,message)
                    checks.append({"label":label,"input":kind,"oracle":message,"XP_all1MiB_currentX":True,"input_immutable":True,"output_NaN_and_XP_poison_overwritten":True})
            x.copy_(original)
            for label in FROZEN:
                freshxp=torch.empty_like(xp[label]);freshout=torch.empty_like(out[label])
                freshxp.fill_(0x37a8f219);freshout.fill_(float("nan"));launch(label,freshxp,freshout);torch.cuda.synchronize();mark(label)
                ok,msg=reference.check_implementation({"x":x,"config":config},freshout,{"unpooler":w});assert ok,msg
                assert torch.equal(freshxp,x.view(torch.int32).reshape(-1)[pack_index]) and torch.equal(x,original)
                checks.append({"label":label,"fresh_workspace_and_output":"PASS","oracle":msg});del freshxp,freshout
            check_state()
            identity={}
            for label,ext in extensions.items():
                assert hashlib.sha256(modules[label].source().encode()).hexdigest()==FROZEN[label][2]
                assert sha(ext.__file__)==result["actual_JIT"][label]["module_sha256"]
                def again(c):return int(ext.rqrun(0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c,empty))
                identity[label]={"regs":again(1),"local":again(2),"smem":again(3),"occupancy":again(4),
                                 "same_loaded_binary_sha256":True,"same_generated_source_sha256":True}
                assert identity[label]["regs"]==122 and identity[label]["local"]==0 and identity[label]["smem"]==103600 and identity[label]["occupancy"]>=1
            result.setdefault("identity_after_validation",{})[mode]=identity
            result["validation"][mode]=checks;save()
        validate_mode("initial")
        # Pre-generate fresh lab fixtures; never kernel state. Timed paths still
        # pack current X on every call. A/B of a comparison use the SAME fixture.
        fixture_seeds=[120000+i for i in range(96)]+[p["input_seed"] for p in frozen_plan["paired_orders"]]+[130000+i for i in range(96)]
        fixtures={seed:reference.generate_input(128,"unpooler",seed)["x"] for seed in fixture_seeds}
        flush=torch.ones(64<<20,dtype=torch.float32,device="cuda");flush_sink=torch.empty((),dtype=torch.float32,device="cuda")
        def one(label,mode,warmup=False,**metadata):
            previous=dict(tracker);host_start=time.monotonic_ns()
            torch.sum(flush,0,out=flush_sink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            before_event=time.monotonic_ns();a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
            host_end=time.monotonic_ns();us=a.elapsed_time(b)*1000
            result["raw"].append({"mode":mode,"label":label,"warmup":warmup,"us":us,"previous_case":previous["previous_label"],
                "previous_timed_case":previous["previous_timed_label"],"preceding_same_variant_run_length":previous["same_variant_run_length"] if previous["previous_label"]==label else 0,
                "host_start_monotonic_ns":host_start,"host_before_start_event_monotonic_ns":before_event,"host_end_monotonic_ns":host_end,**metadata})
            if not warmup:last_timed_input[label]=metadata["input_seed"]
            mark(label,not warmup)
        modes=("steady_before","paired","steady_after")
        for mode_index,mode in enumerate(modes):
            result["phase"]="TIMING_"+mode;save()
            warm_order=("dp","ds") if mode_index%2==0 else ("ds","dp")
            x.copy_(original)
            for label in warm_order:
                for i in range(WARMUPS_PER_LABEL):one(label,mode,warmup=True,warmup_index=i)
            if mode=="paired":
                for p in frozen_plan["paired_orders"]:
                    x.copy_(fixtures[p["input_seed"]])
                    for position,label in enumerate(p["order"]):
                        one(label,mode,pair=p["pair"],swapped_block=p["swapped_block"],within_swapped_block=p["within_swapped_block"],position=position,input_seed=p["input_seed"])
            else:
                seed_start=120000 if mode=="steady_before" else 130000
                for block,label in enumerate(frozen_plan[mode]):
                    for position in range(BLOCK_SAMPLES):
                        seed=seed_start+block*BLOCK_SAMPLES+position;x.copy_(fixtures[seed])
                        one(label,mode,block=block,position=position,input_seed=seed)
            save();validate_mode(mode)
        result["summary"]=summarize(result["raw"])
        assert len([r for r in result["raw"] if not r["warmup"]])==320
        assert len(result["raw"])==350
        # Instrumented node identity only AFTER all unprofiled measurements.
        from torch.profiler import profile,ProfilerActivity
        result["profile"]={}
        for label,g in graphs.items():
            x.copy_(original)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
            mark(label)
            events=[{"name":e.name(),"duration_ns":e.duration_ns()} for e in prof.profiler.kineto_results.events() if e.device_type()==torch.autograd.DeviceType.CUDA]
            assert len(events)==2 and sum("rx_pack" in e["name"] for e in events)==1 and sum(FROZEN[label][3]+"<" in e["name"] for e in events)==1
            result["profile"][label]=events
            assert torch.equal(xp[label],x.view(torch.int32).reshape(-1)[pack_index])
            ok,msg=reference.check_implementation({"x":x,"config":config},out[label],{"unpooler":w});assert ok,msg
        check_state();result.update(status="VALID_FIXED_SOURCE_MEASUREMENT_REVIEW_REQUIRED",phase="COMPLETE",
            weights_pack_escape_immutable_counters_zero=True,no_parent_prepare=True,no_incumbent_or_official_claim=True)
        save();print("DP_DS_MEASUREMENT "+json.dumps(result["summary"]),flush=True)
    except Exception as error:
        result.update(status="INVALID_FIXED_SOURCE_MEASUREMENT",error=repr(error));save();raise
    finally:
        if graphs:
            try:
                torch.cuda.synchronize()
                for g in graphs.values():g.reset()
                graphs.clear();result["graph_cleanup"]="RESET_BOTH_AFTER_DRAIN"
            except Exception as error:
                result["graph_cleanup"]="FAILED";result["cleanup_error"]=repr(error);save();raise
        else:result["graph_cleanup"]="NO_GRAPHS_CREATED"
        save()


if __name__=="__main__":main()
