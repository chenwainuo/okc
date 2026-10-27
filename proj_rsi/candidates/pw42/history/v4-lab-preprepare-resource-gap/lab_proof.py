"""CPU-only exact narrow-lab schedule, terminal, storage and binary gates."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
LABELS=('incumbent','generic_a','guarded_a','generic_b','guarded_b')

def extract(names,env):
    nodes=[n for n in ast.parse((HERE/'lab.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<exact-lab-functions>','exec'),env);return env

def schedule():
    class Buffer:
        value=None
        def copy_(self,value):self.value=value
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
        def elapsed_time(self,other):return .001
    fake=SimpleNamespace(empty=lambda *args,**kwargs:Buffer(),sum=lambda *args,**kwargs:None,float32=None,cuda=SimpleNamespace(Event=Event,synchronize=lambda:None))
    ns=extract({'measure'},{'torch':fake,'statistics':statistics,'LABELS':LABELS});x=Buffer();observed={};boundaries=[]
    class Graph:
        def __init__(self,label):self.label=label
        def replay(self):observed[self.label]=x.value
    fixtures={seed:seed for seed in [800000]+list(range(801000,801168))+list(range(802000,802020))}
    def gate(seeds,last):
        assert observed==seeds and x.value==last['seed'];boundaries.append(dict(seeds));return {'outputs':5}
    result=ns['measure']({label:Graph(label) for label in LABELS},x,fixtures,gate);calls=result['all_calls']
    counts={mode:sum(c['mode']==mode for c in calls) for mode in ('warmup','F1','paired')}
    assert counts=={'warmup':15,'F1':168,'paired':100}
    assert len(boundaries)==2 and len(set(boundaries[0].values()))==5
    pairs=[call for call in calls if call['mode']=='paired']
    for position in range(5):
        for label in LABELS:assert sum(pairs[5*r+position]['label']==label for r in range(20))==4
    text=(HERE/'lab.py').read_text();tree=ast.parse(text)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    mloop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='m')
    role=next(n for n in mloop.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='role')
    allocations=[n for n in mloop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='buffers' for t in n.targets)]
    assert len(allocations)==1 and mloop.body.index(allocations[0])<mloop.body.index(role)
    assert 'indices=(0,1,2,3) if role==0 else (1,0,3,2)' in text
    assert text.index("context['timing']=measure(")<text.index("context['graph_mutations']=[]")<text.index('from torch.profiler import')
    return dict(state='PASS_EXACT_LAB_SCHEDULE_AND_MATCHED_OUTPUT_SWAP',per_context_counts=counts,total_contexts=4,total_calls=1132,terminal_outputs=40,paired_position_occurrences=4,same_buffer_pool_reused_between_roles=True)

def terminal_storage():
    ns=extract({'terminal','storage'},{'torch':torch,'LABELS':LABELS});checks=ns['terminal'];negative={}
    def case(kind=None):
        fixtures={i:torch.arange(8,device="cpu").reshape(2,4)+100*i for i in range(5)};seeds=dict(zip(LABELS,range(5)));x=fixtures[4].clone();outputs={label:fixtures[seeds[label]].float() for label in LABELS}
        immutable=torch.tensor([17],device="cpu");before=immutable.clone();dormant=torch.tensor([23],device="cpu");dormant_before=dormant.clone()
        if kind=='output':outputs['guarded_a'][0,0]+=1
        if kind=='nonfinite':outputs['guarded_b'][0,0]=float('nan')
        if kind=='currentX':x[0,0]+=1
        if kind=='wrongfixture':seeds['generic_a']=seeds['generic_b']
        if kind=='weight':immutable[0]+=1
        if kind=='dormant':dormant[0]+=1
        def integrity():assert torch.equal(immutable,before) and torch.equal(dormant,dormant_before)
        return checks(outputs,x,fixtures,seeds,{'seed':4},lambda fresh,out:(torch.equal(fresh.float(),out),'CPU ownfixture'),integrity)
    valid=case()
    for kind in ('output','nonfinite','currentX','wrongfixture','weight','dormant'):
        try:case(kind)
        except AssertionError as error:negative[kind]=str(error)
        else:raise AssertionError('terminal admitted '+kind)
    # Dormant NaN payload/nonzero bytes preserve exactbaseline, no blanketzero.
    raw=torch.tensor([0x7fc00001,123,9],dtype=torch.int32,device="cpu");floats=raw.view(torch.float32);check,identity=ns['storage'](floats);check();raw[0]^=1
    try:check()
    except AssertionError:negative['NaN_payload_dormant_storage']='REJECT'
    else:raise AssertionError('NaNpayload mutation admitted')
    return dict(state='PASS_TERMINAL_OWNFIXTURE_AND_DORMANT_BIT_STORAGE',negative_controls=negative,valid=valid)

def binaries():
    spec=importlib.util.spec_from_file_location('pw42_lab_CPU_gates',HERE/'lab.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sass=(HERE/'sass.txt').read_text();mod.actual_binary_gate(sass);negative={}
    mutants={'HMMA_operand':sass.replace('HMMA.','HMMA_TAMPER.',1),'PC':sass.replace('/*0000*/','/*0010*/',1),'missing_twins':(HERE/'control-sass.txt').read_text()}
    for name,mutant in mutants.items():
        try:mod.actual_binary_gate(mutant)
        except AssertionError as error:negative[name]=str(error)
        else:raise AssertionError('actualbinary admitted '+name)
    assert mod.source()==(HERE/'lab_source.cu').read_text()
    return dict(state='PASS_ACTUAL_BINARY_COMPONENT_AND_TAMPER_NEGATIVES',negative_controls=negative,lab_source_sha256=hashlib.sha256(mod.source().encode()).hexdigest())
if __name__=='__main__':
    result=dict(state='CPU_PW42_LAB_PROOFS_PASS_GPU_UNRUN',schedule=schedule(),terminal_storage=terminal_storage(),binaries=binaries(),GPU_calls=0,prepare_calls=0)
    (HERE/'lab-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
