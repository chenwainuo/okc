"""Independent CPU extracted-timing and actual frozen-codec edge-fixture review."""
import ast,hashlib,importlib.util,json,statistics,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
TRIAL=ROOT/'proj_rsi/candidates/rq32_predxor_trial'
DRIVER=ROOT/'proj_rsi/tools/rq32_predxor_lab.py'
sys.path.insert(0,str(TRIAL))
import support as S
import mechanism as M

def full_schedule():
    tree=ast.parse(DRIVER.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    outer=next(n for n in run.body[10].body if False) if False else next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='load' for t in n.target.elts))
    layer=next(n for n in outer.body if isinstance(n,ast.For));body=layer.body
    one=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name=='one')
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='f1' for t in n.targets))
    end=next(i for i,n in enumerate(body) if i>start and isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='terminal' and n.value.args[0].value=='paired')
    class X:
        def __init__(self):self.seed=None
        def copy_(self,x):self.seed=x
    class Out:seed=None
    work={};counts={'pw45':0,'rq32_predxor':0}
    class Graph:
        def __init__(self,label):self.label=label
        def replay(self):x,out,g=work[self.label];out.seed=x.seed
    for label in counts:work[label]=(X(),Out(),Graph(label))
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
        def elapsed_time(self,b):return .001
    fake=SimpleNamespace(sum=lambda *a,**kw:None,cuda=SimpleNamespace(Event=Event,synchronize=lambda:None));boundaries=[];last={};result={'raw':[]}
    def terminal(kind):
        for label in counts:
            x,out,g=work[label];assert out.seed==last[label]==x.seed
        boundaries.append((kind,len(result['raw']),dict(counts)))
    with tempfile.TemporaryDirectory() as directory:
        env=dict(S=S,torch=fake,work=work,counts=counts,fixtures={seed:seed for seed in range(9000,9064)},flush=None,sink=None,name='synthetic',result=result,args=SimpleNamespace(output=Path(directory),f1_reps=21,pairs=32),json=json,last=last,expected_plan=S.policy_plan(),sample_index=[0],terminal=terminal,entry={},statistics=statistics,LABELS=('pw45','rq32_predxor'))
        exec(compile(ast.Module(body=[one,*body[start:end+1]],type_ignores=[]),'<EXACT-fullpolicy-timing-AST>','exec'),env)
    assert len(result['raw'])==127 and env['sample_index']==[127]
    assert boundaries==[('F1',63,{'pw45':42,'rq32_predxor':21}),('paired',127,{'pw45':74,'rq32_predxor':53})]
    assert [{k:r[k] for k in ('label','mode','seed')} for r in result['raw']]==S.policy_plan()
    pairs=result['raw'][63:]
    assert [sum(pairs[2*i+j]['label']=='pw45' for i in range(32)) for j in range(2)]==[16,16]
    return dict(state='PASS_EXACT_PRODUCTION_DRIVER_TIMING_AST_CPU_FAKE_EVENTS',per_shape=127,all30_raw=3810,terminal_boundaries=boundaries,AB_BA_each_position=16)

def codec_edge():
    import torch
    torch.set_num_threads(1)
    text=(S.PARENT/'submission.py').read_text();tree=ast.parse(text);env={'torch':torch}
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_OS','_SS') for t in n.targets):exec(compile(ast.Module(body=[n],type_ignores=[]),'<codec-literals>','exec'),env)
    for name in ('_pack_nat','_unc12','_c12','_rqpack','_rqunpack','_rqesc'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<EXACT-frozen-codec>','exec'),env)
    class CPU:
        def __getattr__(self,name):return getattr(torch,name)
        def full(self,*args,**kwargs):kwargs['device']='cpu';return torch.full(*args,**kwargs)
    mod=SimpleNamespace(**env);records=[]
    for k,ks in ((8192,4),(4096,2)):
        w=M.edge_weight(CPU(),mod,128,k);lay,tab,esc=env['_rqpack'](w)
        assert torch.equal(env['_rqunpack'](lay,tab,esc,128,k),w.view(torch.int16).int()&65535)
        info=S.verify_edge_rows(esc.tolist(),tab);packed,cap=env['_rqesc'](esc,1,k//64,ks,'cpu');words=packed.reshape(-1).tolist();groups=ks*8
        assert cap==1056
        rows=words[:-32];assert len(rows)==groups*cap
        groupsrows=[rows[i*cap:(i+1)*cap] for i in range(groups)]
        active=[i for i,row in enumerate(groupsrows) if (row[0]&0xffffffff)!=0xffffffff];assert active==[0]
        payload=[x&0xffffffff for x in groupsrows[0] if (x&0xffffffff)!=0xffffffff];assert len(payload)==1026
        assert [(x>>17) for x in payload].count(0)==1024 and [(x>>17) for x in payload].count(1)==1 and [(x>>17) for x in payload].count(31)==1
        assert {(x>>12&31,x>>8&15,x>>7&1) for x in payload[:1024]}=={(lane,word,half) for lane in range(32) for word in range(16) for half in range(2)}
        records.append(dict(K=k,KS=ks,reduced_CPU_N=128,real_launch_N='4096pooler/8192unpooler_not_CPU_prevalence',metadata=info,CAP=cap,all_warp_groups=groups,nonempty_groups=active,payload=1026,window_loads_including_initial=33,refills=32,full_exact_codec=True))
    return dict(state='PASS_ACTUAL_EDGE_FUNCTION_AND_FROZEN_CODEC_CPU',records=records)

def main():
    result=dict(fullpolicy=full_schedule(),edge=codec_edge(),fixed=dict(calls_per_shape=len(M.plan()),two_shapes=2,total_raw=590),GPU_calls=0,prepare_calls=0,limits='Fake timing is schedule proof only; CPU edge fixture does not assert actual CUDA bytes/bit correctness or prevalence.')
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
