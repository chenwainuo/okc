"""CPU-only adversarial test of extracted lab gate; never imports GPU lab/module."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent

def proof():
    text=(HERE.parents[1]/"tools/native_x_lab.py").read_text();tree=ast.parse(text)
    gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="immediate_timed_output_checks")
    # The terminal read gate may not call any replay/profile or write methods.
    calls=[n.func.attr for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{"replay","copy_","fill_","reset","capture_begin","capture_end"}
    ns={"torch":torch,"reference":SimpleNamespace(check_implementation=lambda data,out,weights:(torch.equal(out,data["x"].float()),"synthetic CPU recorded fixture oracle"))}
    exec(compile(ast.Module(body=[gate],type_ignores=[]),str(HERE.parents[1]/"tools/native_x_lab.py"),"exec"),ns)
    check=ns[gate.name];labels=("incumbent","ds","dp","new")
    def case(corruption=None):
        fixtures={i:torch.arange(8,dtype=torch.int32).reshape(2,4)+100*i for i in range(4)}
        seeds=dict(zip(labels,range(4)));x=fixtures[3].clone();outputs={l:fixtures[seeds[l]].float().clone() for l in labels}
        xp={l:fixtures[seeds[l]].reshape(-1).clone() for l in labels[1:3]};idx=torch.arange(8)
        w=torch.tensor([1.]);packed=torch.tensor([2]);esc=torch.tensor([3]);immutable=[(a,a.clone()) for a in (w,packed,esc)]
        cnt={l:torch.zeros(4,dtype=torch.int32) for l in labels[1:]};ic=torch.zeros(4,dtype=torch.int32)
        if corruption=="output":outputs["ds"][0,0]+=1
        if corruption=="XP":xp["dp"][0]+=1
        if corruption=="currentX":x[0,0]+=1
        if corruption=="weight":w[0]+=1
        if corruption=="packed":packed[0]+=1
        if corruption=="escape":esc[0]+=1
        if corruption=="counter":cnt["new"][0]=1
        if corruption=="incumbent_counter":ic[0]=1
        if corruption=="wrong_per_policy_seed":seeds["ds"]=seeds["new"]
        return check(outputs,x,{},w,xp,idx,fixtures,seeds,{"input_seed":3},immutable,cnt,ic)
    valid=case();rejections={}
    for name in ("output","XP","currentX","weight","packed","escape","counter","incumbent_counter","wrong_per_policy_seed"):
        try:case(name)
        except AssertionError as error:rejections[name]=str(error)
        else:raise AssertionError("Gate admitted "+name)
    # Bind control-flow location to this exact lab: F1 gate precedes paired
    # replays; terminal paired gate precedes return into mutation/profile caller.
    measure=text[text.index("def measure("):text.index("\ndef main():")]
    assert measure.index('strict_checks["F1"]=strict_gate')<measure.index('for repetition in range(32):')
    assert measure.index('strict_checks["paired"]=strict_gate')<measure.index('    return {')
    main=text[text.index('def main():'):]
    assert main.index('result["timing"]=measure(')<main.index('result["actual_timed_graph_checks"]=actual_graph_checks')<main.index('result["profile"]=')
    return {"status":"CPU_SYNTHETIC_ADVERSARIAL_GATE_PASS","lab_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "distinct_last_fixtures_all4_policies":"PASS","no_gate_replay_or_mutation":True,"phase_order":"F1 gate then paired timing then paired gate then mutation/profile",
            "negative_controls":rejections,"gpu_calls":0,"limits":"CPU synthetic gate semantics and source ordering; real CUDA numerics/runtime remain unrun"}

if __name__=="__main__":
    result=proof();(HERE/"timed-output-gate-proof.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
