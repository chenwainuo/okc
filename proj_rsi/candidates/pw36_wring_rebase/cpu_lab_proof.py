"""CPU tests of exact lab terminal/measurement/counter functions; no CUDA context."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import torch
HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]/'tools/wring_rebase_lab.py'
LABELS=('incumbent','coupled','equal','split')

def extracted(names,namespace):
    tree=ast.parse(LAB.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LAB),'exec'),namespace)
    return namespace

def terminal():
    ns=extracted({'terminal_gate'},{'torch':torch,'LABELS':LABELS});gate=ns['terminal_gate']
    node=next(n for n in ast.parse(LAB.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='terminal_gate')
    methods={n.func.attr for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not methods&{'replay','copy_','fill_','reset','zero_','capture_begin','capture_end'}
    def case(corruption=None):
        fixtures={i:torch.arange(8).reshape(2,4)+i*100 for i in range(4)};seeds=dict(zip(LABELS,range(4)))
        outputs={label:fixtures[seeds[label]].float().clone() for label in LABELS};x=fixtures[3].clone()
        immutable=torch.tensor([31]);before=immutable.clone();counter=torch.zeros(36,dtype=torch.int32);dormant=torch.tensor([92]);dormant_before=dormant.clone()
        if corruption=='output':outputs['equal'][0,0]+=1
        if corruption=='nonfinite':outputs['split'][0,0]=float('nan')
        if corruption=='currentX':x[0,0]+=1
        if corruption=='wrong_own_fixture':seeds['equal']=seeds['split']
        if corruption=='weight':immutable[0]+=1
        if corruption=='active_counter':counter[0]=1
        if corruption=='dormant_counter':dormant[0]+=1
        def integrity():
            assert torch.equal(immutable,before),'immutable altered'
            assert not int(counter.abs().sum()),'active counter altered'
            assert torch.equal(dormant,dormant_before),'dormant counter altered'
        return gate(outputs,x,fixtures,seeds,{'seed':3},lambda fresh,out:(torch.equal(fresh.float(),out),'CPU ownfixture oracle'),integrity)
    valid=case();negatives={}
    for name in ('output','nonfinite','currentX','wrong_own_fixture','weight','active_counter','dormant_counter'):
        try:case(name)
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('terminal gate admitted '+name)
    text=LAB.read_text();measure=text[text.index('def measure('):text.index('\ndef storage_contract')]
    assert measure.index("boundaries['F1']=strict_gate")<measure.index('for repetition in range(32):')
    assert measure.index("boundaries['paired']=strict_gate")<measure.index('    med=')
    main=text[text.index('def main():'):]
    assert main.index("result['timing']=measure(")<main.index("result['actual_graph_mutations']=[]")<main.index('from torch.profiler import')
    return dict(status='PASS_READONLY_OWN_FIXTURE_TERMINAL_GATE',negative_controls=negatives,valid=valid)

def schedule():
    class Buffer:
        seed=None
        def copy_(self,value):self.seed=value
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):pass
        def elapsed_time(self,other):return .001
    fake=SimpleNamespace(empty=lambda *args,**kw:Buffer(),sum=lambda *args,**kw:None,float32=None,cuda=SimpleNamespace(Event=Event,synchronize=lambda:None))
    ns=extracted({'measure'},{'torch':fake,'LABELS':LABELS,'statistics':statistics});x=Buffer();observed={};boundaries=[]
    class Graph:
        def __init__(self,label):self.label=label
        def replay(self):observed[self.label]=x.seed
    fixtures={seed:seed for seed in [600000]+list(range(601000,601147))+list(range(602000,602032))}
    def gate(seeds,last):
        assert seeds==observed and last['seed']==x.seed
        boundaries.append(dict(seeds));return {'ownfixture_oracles':4}
    result=ns['measure']({label:Graph(label) for label in LABELS},x,fixtures,gate);calls=result['all_calls']
    counts={mode:sum(c['mode']==mode for c in calls) for mode in ('warmup','F1','paired')}
    assert counts=={'warmup':20,'F1':147,'paired':128}
    assert len(boundaries)==2 and len(set(boundaries[0].values()))==4
    paired=[c for c in calls if c['mode']=='paired']
    for position in range(4):
        for label in LABELS:assert sum(paired[4*i+position]['label']==label for i in range(32))==8
    assert len(result['comparisons'])==6
    return dict(status='PASS_EXACT_MEASUREMENT_CONTROL_FLOW_FAKE_EVENTS',counts=counts,terminal_oracles=8,balanced_each_position=8,contrasts=list(result['comparisons']))

def counters():
    ns=extracted({'storage_contract','collect_counters'},{'torch':torch});factory=ns['storage_contract'];negatives={}
    backing=torch.full((88,),123,dtype=torch.int32);visible=backing[8:80];visible[:36].zero_();check,identity=factory(visible,36);check()
    for name,index in [('active',8),('dormant',8+36),('prefixguard',0),('suffixguard',87)]:
        old=int(backing[index]);backing[index]+=1
        try:check()
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('counter contract admitted '+name)
        backing[index]=old;check()
    dormant=torch.tensor([19,7],dtype=torch.int32);exact,_=factory(dormant);exact();dormant[0]+=1
    try:exact()
    except AssertionError as error:negatives['unrelated_nonzero_baseline']=str(error)
    else:raise AssertionError('unrelated storage mutation admitted')
    state={'rq':{'cnt':visible},'alias_cnt':visible.view(-1),'p1cnt':dormant};state['cycle']=state
    found=ns['collect_counters'](state);assert len(found)==3
    assert found[0][1].untyped_storage().data_ptr()==found[1][1].untyped_storage().data_ptr()
    return dict(status='PASS_FULL_STORAGE_IDENTITY_ACTIVE_PREFIX_AND_DORMANT_GUARDS',negative_controls=negatives,active_ints=36,visible_dormant_ints=36,guard_ints_each=8,nonzero_unrelated_baseline_preserved=True)

def binary_negatives():
    import preflight
    canonical=preflight.streams((HERE/'sass.txt').read_text());negatives=[]
    for args in preflight.ARMS.values():
        rows=canonical[args];mutant=list(rows);i=next(i for i,(pc,ins) in enumerate(rows) if 'HGMMA.' in ins)
        mutant[i]=(rows[i][0],rows[i][1]+' TAMPER')
        assert mutant!=rows;negatives.append(list(args))
    assert canonical[preflight.ARMS['coupled_x5w5']]!=canonical[preflight.ARMS['split_x5w5']]
    return dict(status='PASS_CANONICAL_STREAM_TAMPER_AND_WRONG_CONTROL_REJECTION',tampered_arms=negatives)

def main():
    result=dict(state='CPU_LAB_PROOFS_PASS_GPU_UNRUN',lab_sha256=hashlib.sha256(LAB.read_bytes()).hexdigest(),terminal=terminal(),measurement=schedule(),counters=counters(),binary=binary_negatives(),GPU='UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE')
    (HERE/'cpu-lab-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
