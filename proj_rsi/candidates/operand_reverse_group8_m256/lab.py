"""Root-only M256 full-pipeline lab; never run before parent validation + root/critic GO."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import subprocess
import shutil
import torch
import prototype as P
import liveness
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
    ok,msg=P.S._rqval(fn,w,256,8192,gen,4e-3)
    assert ok,msg
    print("ONEHOT4096_EXACT "+msg,flush=True)
    checks=[{"all4096_onehot_columns":"EXACT","random_FP64":"PASS","detail":msg}]
    for dist in ("normal","outliers"):
        data=reference.generate_input(256,"unpooler",7289,dist=dist)
        before=data["x"].clone();out=torch.empty((256,8192),dtype=torch.bfloat16,device="cuda")
        fn(data["x"],out)
        ok,msg=reference.check_implementation(data,out,{"unpooler":w})
        assert ok and torch.equal(before,data["x"]),msg
        checks.append({"official_input_dist":dist,"result":msg})
    data=reference.generate_input(256,"unpooler",1321)
    x=data["x"].clone();out=torch.empty((256,8192),dtype=torch.bfloat16,device="cuda")
    g=capture(fn,x,out)
    for kind in ("original","zero","fresh_outliers"):
        fresh=data["x"] if kind=="original" else torch.zeros_like(x) if kind=="zero" else reference.generate_input(256,"unpooler",7771,dist="outliers")["x"]
        x.copy_(fresh);g.replay();torch.cuda.synchronize()
        assert torch.equal(x,fresh),"input modified"
        if kind=="zero":assert torch.count_nonzero(out)==0
        else:
            ok,msg=reference.check_implementation({"x":x,"config":data["config"]},out,{"unpooler":w})
            assert ok,msg
        checks.append({"fixed_pointer_replay":kind,"status":"PASS"})
    return checks


def validate_baseline(B,state,w):
    config={"layer":"unpooler","m":256,"n":8192,"k":4096}
    # Copy only for this untimed validator; actual timed baseline graph has no added copy.
    def fn(x,out):
        out.copy_(B.kernel_fn({"x":x,"config":config},state))
        return 0
    gen=torch.Generator(device="cuda").manual_seed(6211)
    ok,msg=P.S._rqval(fn,w,256,8192,gen,4e-3)
    assert ok,msg
    checks=[{"all4096_onehot_columns":"EXACT","random_FP64":"PASS","detail":msg}]
    for dist in ("normal","outliers"):
        data=reference.generate_input(256,"unpooler",7291,dist=dist)
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
    rows = torch.arange(256, device=x.device)
    probes[rows, (rows * 37 + 19) % 4096] = ((2.0 ** ((rows % 5) + 3)) * (2 * (rows % 2) - 1)).to(torch.bfloat16)
    mutations = [("original", original), ("zero", torch.zeros_like(x)),
                 ("fresh_normal", reference.generate_input(256,"unpooler",9911)["x"]),
                 ("fresh_outliers", reference.generate_input(256,"unpooler",9913,dist="outliers")["x"]),
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
    labels=("incumbent","parent_pair","parent_reverse","m256")
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
    raw["m256"]=series("m256")
    for label in reversed(labels[:3]):raw[label+"_after"]=series(label)
    med={k:statistics.median(v) for k,v in raw.items()}
    f1={label:min(med[label+"_before"],med[label+"_after"])-med["m256"] for label in labels[:3]}
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
    pairs=[]
    for repetition in range(21):
        row={}
        for index in orders[repetition%len(orders)]:row[labels[index]]=one(labels[index])
        pairs.append(row)
    paired={label:statistics.median(row[label]-row["m256"] for row in pairs) for label in labels[:3]}
    return {"F1_medians_us":med,"F1_samples_us":raw,"F1_m256_gain_us":f1,
            "paired_graph_us":pairs,"paired_m256_median_gain_us":paired,
            "paired_m256_wins":{label:sum(row[label]>row["m256"] for row in pairs) for label in labels[:3]}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--parent-validation-result',type=Path,required=True);ap.add_argument('--review-go',required=True)
    args=ap.parse_args()
    assert args.review_go.strip(), 'root and critic review evidence id required'
    # Freeze HOLD until root reviews the corrected canonical G8-containing trial
    # schema. A valid GS four-arm trial containing frozen G8 is also eligible.
    # Lift only in a new source/manifest after canonical status/records are known.
    parent_runtime_admission='HOLD_CANONICAL_PARENT_VALIDATION_SCHEMA_PENDING'
    assert parent_runtime_admission=='READY', parent_runtime_admission
    parent_result=json.loads(args.parent_validation_result.read_text())
    manifest=json.loads((P.HERE/'manifest.json').read_text())
    for name,sha in manifest['files'].items():assert hashlib.sha256((P.HERE/name).read_bytes()).hexdigest()==sha,('frozen source drift',name)
    for name,sha in json.loads((P.PARENT/'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((P.PARENT/name).read_bytes()).hexdigest()==sha,('parent drift',name)
    assert hashlib.sha256(P.source().encode()).hexdigest()==manifest['source_sha256']
    extensions={'parent_pair':P.G.extension(),'m256':P.extension()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'STARTED_NO_ADMISSION','review_go':args.review_go,'parent_validation_result_sha256':hashlib.sha256(args.parent_validation_result.read_bytes()).hexdigest(),'source_sha256':manifest['source_sha256'],'policy':'M256 full fresh2MiB pack+main versus prepared pw36 and two ordered/reversed full parent M128 calls; no cached X','actual_JIT_SASS':{},'attrs':{}}
    def save():args.json.write_text(json.dumps(result,indent=2)+'\n')
    graphs={}
    candidate_math_launched=[False]
    try:
        # Inspect loaded code before first candidate math.
        for label,ext in extensions.items():
            binary=Path(ext.__file__).resolve()
            sass=subprocess.check_output([shutil.which('cuobjdump') or '/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True,timeout=20)
            artifact=args.json.with_suffix('.'+label+'.jit.sass');artifact.write_text(sass)
            result['actual_JIT_SASS'][label]={'module':str(binary),'artifact':str(artifact),'sha256':hashlib.sha256(sass.encode()).hexdigest(),'physical':liveness.check(sass)}
            checker=__import__('preflight').load('m256_store_gate',P.HERE.parent/'operand_reverse_dual_stmatrix/liveness.py')
            result['actual_JIT_SASS'][label]['store_form']=checker.store_form(sass,'rq_px_g8',False,dual=True)
        empty=torch.empty(0,dtype=torch.int64,device='cuda');sink=torch.zeros(4,dtype=torch.int32,device='cuda')
        for label,ext in extensions.items():
            def info(c):return int(ext.rqrun(0,empty,empty,empty,empty,empty,0,0,0,empty,sink,c,empty))
            attrs={'regs':info(1),'local':info(2),'SMEM':info(3),'occupancy_after_optin':info(4),'max_dynamic_after':info(5),'max_threads':info(6),'pack_regs':int(ext.packinfo(1)),'pack_local':int(ext.packinfo(2)),'pack_threads':int(ext.packinfo(3))}
            assert 0<attrs['regs']<=128 and attrs['local']==0 and attrs['SMEM']==103600 and attrs['occupancy_after_optin']>=1
            assert attrs['max_dynamic_after']>=103600 and attrs['max_threads']>=512
            assert 0<attrs['pack_regs']<=64 and attrs['pack_local']==0 and attrs['pack_threads']>=256
            result['attrs'][label]=attrs
        B,base_sha=baseline_module();result['baseline_sha256']=base_sha
        weights=reference.generate_weights(reference.WEIGHTS_SEED);state=B.prepare(weights);w=weights['unpooler'];weight_before=w.clone()
        lay,tab,esc6=P.S._rqpack(w)
        assert torch.equal(P.S._rqunpack(lay,tab,esc6,*w.shape),w.view(torch.int16).int()&65535)
        esc,_=P.S._rqesc(esc6,lay.shape[0],lay.shape[1],1,'cuda');lay_before=lay.clone();esc_before=esc.clone()
        labels=('parent_pair','parent_reverse','m256')
        packed_workspaces={label:torch.empty(524288,dtype=torch.int32,device='cuda') for label in labels}
        counters={label:torch.zeros(4*lay.shape[0],dtype=torch.int32,device='cuda') for label in labels}
        assert len({t.data_ptr() for t in packed_workspaces.values()})==3
        idx=torch.arange(524288,dtype=torch.int64);tile=idx//262144;local=idx%262144
        v=local&3;lane=(local>>2)&31;chunk=(local>>7)&3;warp=(local>>9)&7;s=local>>12
        row=tile*128+64*(warp//4)+16*(warp%4)+(lane>>2)+8*(v&1);k=64*s+16*chunk+8*((v>>1)&1)+2*(lane&3)
        pack_index=(row*2048+k//2).cuda()
        def make_fn(label,workspace=None):
            xp=packed_workspaces[label] if workspace is None else workspace
            def launch(ext,x,out,workspace):
                candidate_math_launched[0]=True
                code=int(ext.rqrun(0,lay,x,out,esc,counters[label],1,tab,0,empty,sink,0,workspace))
                if code:raise RuntimeError((label,'launch',code))
            def fn(x,out):
                if label=='m256':launch(extensions['m256'],x,out,xp)
                else:
                    for tile in ((0,1) if label=='parent_pair' else (1,0)):
                        launch(extensions['parent_pair'],x[tile*128:(tile+1)*128],out[tile*128:(tile+1)*128],xp[tile*262144:(tile+1)*262144])
                return 0
            return fn
        funcs={label:make_fn(label) for label in labels}
        bitx=(torch.arange(1048576,dtype=torch.int32,device='cuda')&65535).to(torch.int16).view(torch.bfloat16).reshape(256,4096)
        bit_before=bitx.view(torch.int16).clone();bitout=torch.empty((256,8192),dtype=torch.bfloat16,device='cuda')
        result['storage_bits']={}
        for label in labels:
            packed_workspaces[label].fill_(0x37a8f219);funcs[label](bitx,bitout);torch.cuda.synchronize()
            assert torch.equal(bitx.view(torch.int16),bit_before)
            assert torch.equal(packed_workspaces[label],bitx.view(torch.int32).reshape(-1)[pack_index])
            result['storage_bits'][label]='PASS_ALL65536_PATTERNS_2MiB_INPUT_IMMUTABLE_STORAGE_ONLY'
        del bitx,bit_before,bitout
        result['validation']={label:validate(fn,w) for label,fn in funcs.items()}
        result['validation']['incumbent']=validate_baseline(B,state,w);save()
        data=reference.generate_input(256,'unpooler',5237);x=data['x']
        result['fresh_workspace_and_output']={}
        for label in labels:
            xp=torch.empty_like(packed_workspaces[label]);out=torch.empty((256,8192),dtype=torch.bfloat16,device='cuda')
            xp.fill_(0x37a8f219);out.fill_(float('nan'));make_fn(label,xp)(x,out);torch.cuda.synchronize()
            assert torch.equal(xp,x.view(torch.int32).reshape(-1)[pack_index])
            ok,message=reference.check_implementation(data,out,{'unpooler':w});assert ok,message
            result['fresh_workspace_and_output'][label]={'oracle':message,'bytes':2097152};del xp,out
        outputs={label:torch.empty((256,8192),dtype=torch.bfloat16,device='cuda') for label in labels};hold=[None]
        def incumbent(xx,unused):hold[0]=B.kernel_fn({'x':xx,'config':data['config']},state);return hold[0]
        graphs['incumbent']=capture(incumbent,x,outputs['m256']);outputs['incumbent']=hold[0]
        for label,fn in funcs.items():graphs[label]=capture(fn,x,outputs[label])
        rp=state.get('rq',{}).get(('unpooler',256))
        result['incumbent_pick']={'kind':'rq','variant':(B.RQPROD2 if 'mod' in rp else B.RQPROD)[rp['i']],'KS':rp['KS']} if rp else {'kind':'actual_kernel_fn','Lt_code':state.get('lt',{}).get(('unpooler',256))}
        # Validity gates execute on the exact graphs before timing, without repair.
        result['actual_timed_graph_checks']=actual_graph_checks(graphs,outputs,x,data['config'],w,packed_workspaces,pack_index)
        assert len(result['actual_timed_graph_checks'])==20
        result['profile']={label:profile(g) for label,g in graphs.items()}
        for label in labels:
            events=result['profile'][label];count=1 if label=='m256' else 2
            assert len(events)==2*count
            assert sum('rx_pack' in e['name'] for e in events)==count
            assert sum('rq_px_g8' in e['name'] for e in events)==count
        timed_x_before=x.clone()
        fixture_sha=hashlib.sha256(timed_x_before.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        result['timing']=measure(graphs);save()
        # No graph replay, mutation or profile between measurement and these gates.
        # Every timing replay used the exact same recorded fixture.
        assert torch.equal(x,timed_x_before)
        final_pack_want=timed_x_before.view(torch.int32).reshape(-1)[pack_index]
        assert torch.equal(w,weight_before) and torch.equal(lay,lay_before) and torch.equal(esc,esc_before)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp['cnt'].abs().sum())
        result['final_timed_output_checks']=[]
        for label,out in outputs.items():
            assert bool(torch.isfinite(out).all()),(label,'final timed output nonfinite')
            ok,message=reference.check_implementation({'x':timed_x_before,'config':data['config']},out,{'unpooler':w})
            assert ok,(label,'actual final timed output',message)
            if label in packed_workspaces:
                assert torch.equal(packed_workspaces[label],final_pack_want),(label,'actual final timed packed-X')
            result['final_timed_output_checks'].append({'mode':'fixed_input','policy':label,'status':'PASS_BEFORE_ANY_REPLAY_OR_MUTATION','fixture_seed':5237,'input_sha256':fixture_sha,'Y_oracle':message,'full_XP_matched':True if label in packed_workspaces else 'NOT_APPLICABLE_BASELINE','XP_bytes':2097152 if label in packed_workspaces else 0,'X_immutable':True,'W_pack_escape_immutable':True,'cnt0':True,'no_replay_mutation_profile_since_timing':True})
        assert len(result['final_timed_output_checks'])==4
        assert torch.equal(x,timed_x_before)
        assert torch.equal(w,weight_before) and torch.equal(lay,lay_before) and torch.equal(esc,esc_before)
        assert all(not int(c.abs().sum()) for c in counters.values())
        if rp:assert not int(rp['cnt'].abs().sum())
        result['weights_inputs_counters']='UNCHANGED_OR_ZERO';result['full_pipeline']='M256_TWO_NODES_PARENT_CONTROLS_FOUR_NODES_ALL_PACKS_FRESH'
        result['status']='VALID_M256_MEASUREMENT_REVIEW_REQUIRED';save()
    except Exception as error:
        result.update(status='INVALID_M256_ATTEMPT_RETAIN_DIAGNOSE',error=repr(error),no_candidate_math_launch=not candidate_math_launched[0]);save();raise
    finally:
        if graphs:
            torch.cuda.synchronize()
            for graph in graphs.values():graph.reset()
            graphs.clear()
        result['graph_cleanup']='RESET_ALL_FOUR' if result.get('actual_timed_graph_checks') else 'RESET_ALL_CREATED_OR_NONE';save()
if __name__=='__main__':main()
