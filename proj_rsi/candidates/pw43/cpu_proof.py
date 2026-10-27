"""CPU-only actual source extraction, clock and estimator adversaries."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import math
import time
from build import HERE, PARENT, SHA, build

def sha(b):return hashlib.sha256(b).hexdigest()
def named(s):
    result={}
    for n in ast.parse(s).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):result[n.name]=ast.dump(n,include_attributes=False)
        elif isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):result[t.id]=ast.dump(n,include_attributes=False)
    return result

def main():
    started=time.monotonic();old,new=build()
    assert sha(old.encode())==SHA and (HERE/'submission.py').read_text()==new
    a,b=named(old),named(new)
    changed=sorted(k for k in a if a[k]!=b.get(k))
    assert changed==['_ltsearch','_tune'],changed
    assert set(b)-set(a)=={'_lt43confirm'}
    gpu={}
    for n in ast.parse(old).body:
        if not isinstance(n,ast.Assign):continue
        try:v=ast.literal_eval(n.value)
        except (ValueError,TypeError):continue
        if isinstance(v,str) and any(t in v for t in ('__global__','__device__','#include')):
            for target in n.targets:
                if isinstance(target,ast.Name):
                    assert a[target.id]==b[target.id]
                    gpu[target.id]={'bytes':len(v.encode()),'sha256':sha(v.encode())}
    env={}
    for n in ast.parse(new).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_lt43confirm','_ltsearch'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<PW43-source>','exec'),env)
    helper=env['_lt43confirm'];passed=[]
    class Clock:
        def __init__(self,step=.001,start=0):self.t=start;self.step=step
        def __call__(self):self.t+=self.step;return self.t
    def run(label, timings, expected, clock=None, deadline=10, valid=True,codes=(3,4), factory_cost=0):
        c=clock or Clock();calls=[]
        def factory(f):
            c.t+=factory_cost
            def timed(code):
                calls.append((f,code));return timings(f,code,len(calls))
            return lambda code:valid,timed
        choice,record=helper(codes,factory,deadline,c)
        assert choice==expected,(label,choice,record)
        json.dumps(record,allow_nan=False)
        if record['status']=='complete':
            assert len(record['rows'])==4*len(set(codes))
            assert len(calls)==12*len(set(codes))
        passed.append({'case':label,'selected':choice,'status':record['status'],'reason':record['reason']})
        return record,calls
    run('tiny_positive_below_one_percent',lambda f,c,i:1 if c<0 else .999999 if c==3 else 1.1,3)
    run('exact_tie',lambda f,c,i:1,None)
    run('all_loss',lambda f,c,i:1 if c<0 else 1.01,None)
    run('opposite_fixture_sign',lambda f,c,i:1 if c<0 else .9 if f==0 else 1.1,None)
    run('second_leader_wins',lambda f,c,i:1 if c<0 else .95 if c==3 else .9,4)
    run('tie_finalists_stable_order',lambda f,c,i:1 if c<0 else .9,3)
    for value in (float('nan'),float('inf'),-1,0,True):
        run('invalid_'+str(value),lambda f,c,i,v=value:1 if c<0 else v,None)
    run('expired_before_start',lambda f,c,i:1,None,deadline=0)
    run('fixture_overrun',lambda f,c,i:1,None,factory_cost=20)
    run('one_inflight_timing_overrun',lambda f,c,i:.9,None,clock=Clock(step=.1),deadline=.8)
    run('invalid_legality',lambda f,c,i:1,None,valid=False)
    run('no_finalists',lambda f,c,i:1,None,codes=())
    run('nonfinite_deadline',lambda f,c,i:1,None,deadline=float('nan'))
    class Rollback(Clock):
        def __call__(self):
            self.t+=1
            return self.t if self.t<5 else 0
    run('clock_rollback',lambda f,c,i:.9,None,clock=Rollback(),deadline=100)
    run('nonfinite_clock',lambda f,c,i:.9,None,clock=Clock(start=float('inf')))
    # Exercise actual dispatch binding and target deadline with fake extension only.
    class Ext:
        def lt_algos(self,*args):return 2
        def lt_cfg(self,*args):return [66,1,1,0,0,0,35,0,0]
        def lt_add(self,*args):return -1
        def lt_caps(self,*args):return []
    class Times:
        def __init__(self):self.c=Clock()
        def time(self):return self.c()
        def monotonic(self):return self.c()
    for layer in ('qkv_gate','o','pooler','unpooler'):
        times=Times();env.update(time=times,_LTDONE=[],_LT0=[],_T0=0,PREPARE_BUDGET_S=240.,LTBUDGET=22.,LTW={'qkv_gate':1.6,'o':.3,'pooler':1.,'unpooler':.7,'router':1.6},_LTSEED=[],_tm=lambda fn,flush,it:1.)
        factories=[]
        def factory(f):
            factories.append(f)
            return lambda c:True,lambda c:1 if c<0 else .999
        with contextlib.redirect_stdout(io.StringIO()) as output:
            picked=env['_ltsearch'](Ext(),layer,256,8,8,0,lambda c:True,lambda c:0,lambda:None,None,0,factory)
        rec=json.loads(output.getvalue().split('PW43_LT ',1)[1]);assert picked is not None and factories==[0,1]
        assert rec['selected']==picked and len(rec['exploration_ms'])==8
        passed.append({'case':'actual_target_binding_'+layer,'selected':picked})
    # Existing behavior stays byte-identical for original final decision block and timer.
    tail=old[old.index('    best = (t_torch, -1)',old.index('def _ltsearch')):old.index('\n\n_LTDONE',old.index('def _ltsearch'))]
    assert tail in new
    assert a['_tm']==b['_tm'] and a['prepare']==b['prepare'] and a['kernel_fn']==b['kernel_fn']
    result={'status':'CPU_PASS_GPU_UNRUN_REVIEW_REQUIRED','parent_sha256':SHA,'candidate_sha256':sha(new.encode()),'changed_existing_definitions':changed,'gpu_literals':gpu,'cases':passed,'case_count':len(passed),'no_torch_import':True,'no_cuda_context':True}
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in result.items() if k not in ('gpu_literals','cases')}, 'cpu_seconds':time.monotonic()-started},indent=2))
if __name__=='__main__':main()
