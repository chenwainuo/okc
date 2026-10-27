"""CPU-only five-arm adversarial last-actual-output gate proof; GPU never imported."""
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]/'tools/native_x4_stmatrix_lab.py'
def proof():
    text=LAB.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='immediate_timed_output_checks')
    calls=[n.func.attr for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'replay','copy_','fill_','reset','capture_begin','capture_end'}
    ns={'torch':torch,'reference':SimpleNamespace(check_implementation=lambda data,out,weights:(torch.equal(out,data['x'].float()),'CPU_SYNTHETIC_NOT_HARDWARE'))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(LAB),'exec'),ns);check=ns[node.name]
    labels=('incumbent','native','x4','stmx','new')
    def case(corruption=None):
        fixtures={i:torch.arange(8,dtype=torch.int32).reshape(2,4)+100*i for i in range(5)}
        seeds=dict(zip(labels,range(5)));x=fixtures[4].clone();outputs={l:fixtures[seeds[l]].float().clone() for l in labels}
        w=torch.tensor([1.]);packed=torch.tensor([2]);esc=torch.tensor([3]);immutable=[(a,a.clone()) for a in (w,packed,esc)]
        counters={l:torch.zeros(4,dtype=torch.int32) for l in labels[1:]};ic=torch.zeros(4,dtype=torch.int32);xp={}
        if corruption=='output':outputs['x4'][0,0]+=1
        if corruption=='unauthorized_XP':xp['native']=torch.ones(8)
        if corruption=='currentX':x[0,0]+=1
        if corruption=='weight':w[0]+=1
        if corruption=='packed':packed[0]+=1
        if corruption=='escape':esc[0]+=1
        if corruption=='counter':counters['new'][0]=1
        if corruption=='incumbent_counter':ic[0]=1
        if corruption=='wrong_per_policy_seed':seeds['native']=seeds['new']
        return check(outputs,x,{},w,xp,None,fixtures,seeds,{'input_seed':4},immutable,counters,ic)
    case();rejections={}
    for name in ('output','unauthorized_XP','currentX','weight','packed','escape','counter','incumbent_counter','wrong_per_policy_seed'):
        try:case(name)
        except AssertionError as e:rejections[name]=str(e)
        else:raise AssertionError('Gate admitted '+name)
    measure=text[text.index('def measure('):text.index('\ndef main():')]
    assert measure.index('strict_checks["F1"]=strict_gate')<measure.index('for repetition in range(40):')
    assert measure.index('strict_checks["paired"]=strict_gate')<measure.index('    return {')
    main=text[text.index('def main():'):]
    assert main.index('result["timing"]=measure(')<main.index('result["actual_timed_graph_checks"]=actual_graph_checks')<main.index('result["profile"]=')
    return {'status':'CPU_SYNTHETIC_ADVERSARIAL_GATE_PASS','lab_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'distinct_last_fixtures_all5_policies':'PASS','negative_controls':rejections,'no_gate_replay_or_mutation':True,'GPU_calls':0,
            'limits':'CPU synthetic gate plus source ordering, no GPU numeric/runtime claim'}
if __name__=='__main__':
    r=proof();(HERE/'timed-output-gate-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
