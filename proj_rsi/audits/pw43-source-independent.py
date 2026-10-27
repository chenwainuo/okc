"""CPU-only PW43 source critic: never imports production or writes frozen artifacts."""
import ast,contextlib,hashlib,importlib.util,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'proj_rsi/candidates/pw43'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(source,names,env=None):
    env={} if env is None else env
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-host-AST>','exec'),env)
    return env

def main():
    assert sha(D/'manifest.json')=='f05e27c4136b4d9771d16bd015df18e2f92c0f1c55f61b7e98bcc64cf70fb67f'
    manifest=json.loads((D/'manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(D/n)==h
    parent=ROOT/manifest['parent']['path'];assert sha(parent)==manifest['parent']['sha256']
    old,new=parent.read_text(),(D/'submission.py').read_text();assert sha(D/'submission.py')=='fd8e9f73c74a03cb7b979094b6c24997a27c217cc23989d0af82ae03847f2d81'
    a,b=ast.parse(old),ast.parse(new);defs={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b.body=[defs[n.name] if isinstance(n,ast.FunctionDef) and n.name in ('_tune','_ltsearch') else n for n in b.body if not(isinstance(n,ast.FunctionDef) and n.name=='_lt43confirm')]
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    # Reexecute frozen author proof while capturing its sole write in memory.
    sys.path.insert(0,str(D));spec=importlib.util.spec_from_file_location('independent_pw43_author_cpu',D/'cpu_proof.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    original=Path.write_text;captured=[]
    def capture(p,text,*args,**kwargs):
        assert p.resolve()==(D/'cpu-proof.json').resolve();captured.append(json.loads(text));return len(text)
    Path.write_text=capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):mod.main()
    finally:Path.write_text=original
    assert captured==[json.loads((D/'cpu-proof.json').read_text())]
    helper=extract(new,{'_lt43confirm'})['_lt43confirm'];extra=[]
    def check(name,codes,valid,timing,expected,deadline=10,clock=lambda:0.):
        events=[]
        def factory(f):
            def v(c):events.append(('valid',f,c));return valid(f,c)
            def t(c):events.append(('time',f,c));return timing(f,c)
            return v,t
        selected,rec=helper(codes,factory,deadline,clock);assert selected==expected,(name,selected,rec)
        json.dumps(rec,allow_nan=False);extra.append({'name':name,'selected':selected,'status':rec['status'],'events':events,'record':rec});return events,rec
    ev,rec=check('dedup_and_two_finalist_cap',[3,3,4,5],lambda f,c:True,lambda f,c:1 if c<0 else .999999,3)
    assert [e[2] for e in ev if e[0]=='time']==[-1,3,-1,-1,4,-1,-1,4,-1,-1,3,-1]*2
    assert all(e[2]!=5 for e in ev)
    check('invalid_second_finalist_rejects_partial_first',[3,4],lambda f,c:c==3,lambda f,c:1 if c<0 else .9,None)
    check('one_bad_fixture_rejects_fast_overall_mean',[3],lambda f,c:True,lambda f,c:1 if c<0 else .1 if f==0 else 1.000001,None)
    check('arbitrarily_tiny_positive_no_abs_or_ratio_gate',[3],lambda f,c:True,lambda f,c:1 if c<0 else 1.-1e-12,3)
    # Compare actual old/new non-target search decisions and extension/timing calls.
    non=[]
    for layer,m,f32 in [('qkv_gate',128,0),('o',128,0),('pooler',128,0),('unpooler',128,0),('router',128,1),('router',256,1)]:
        def execute(src):
            calls=[]
            class Time:
                def time(self):return 0.
                def monotonic(self):return 0.
            class Ext:
                def lt_algos(self,*a):calls.append(('algos',a));return 2
                def lt_cfg(self,*a):calls.append(('cfg',a));return [66,1,1,0,0,0,35,0,0]
                def lt_caps(self,*a):calls.append(('caps',a));return []
            def ok(c):calls.append(('valid',c));return True
            def run(c):calls.append(('run',c));return .9 if c==0 else 1.1
            def tm(fn,flush,it):calls.append(('timer',it));return fn()
            env=dict(time=Time(),_LTDONE=[],_LT0=[0.],_T0=0.,PREPARE_BUDGET_S=240.,LTBUDGET=22.,LTW={'qkv_gate':1.6,'o':.3,'pooler':1.,'unpooler':.7,'router':1.6},_LTSEED=[],_tm=tm)
            env=extract(src,{'_ltsearch','_lt43confirm'},env)
            with contextlib.redirect_stdout(io.StringIO()):choice=env['_ltsearch'](Ext(),layer,m,8,8,f32,ok,run,lambda:1.,None,0.)
            return choice,calls
        before,after=execute(old),execute(new);assert before==after,(layer,m)
        non.append({'layer':layer,'m':m,'f32':f32,'selected':after[0],'all_extension_and_timing_calls_equal':True})
    result={'verdict':'GO_SOURCE_CPU_ONLY_RUNTIME_UNPROVEN','source_sha256':sha(D/'submission.py'),'manifest_sha256':sha(D/'manifest.json'),'all_frozen_file_hashes_match':True,'whole_module_AST_inverse':True,'frozen_author_cases_reexecuted':23,'independent_helper_cases':extra,'non_target_actual_AST_cases':non,'no_production_import':True,'no_GPU_or_prepare':True,'evidence_limits':['_tm retains aggregate medians, not raw per-replay samples; future disjoint evaluation must retain raw samples.','No hard preemption of blocking CUDA operations; host configuration queries can occur after timing deadline.','Non-target decision logic is identical; extra host bookkeeping can alter shared prepare timing and later selected policies.','PW39 mixed local signs and official regression prohibit transfer/gain inference.']}
    out=ROOT/'proj_rsi/audits/pw43-source-independent.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('independent_helper_cases','non_target_actual_AST_cases')},indent=2))
if __name__=='__main__':main()
