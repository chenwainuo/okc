"""CPU source identity and adversarial host selection tests; no production import."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from build_patch import apply,PARENT_MANIFEST,PARENT_SOURCE

def digest(data):return hashlib.sha256(data).hexdigest()
def value(text,name):
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise KeyError(name)
def named(text):
    out={}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):out[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):out[target.id]=ast.dump(node,include_attributes=False)
    return out

def source_proof():
    parent=HERE.parent/'pw38';assert digest((parent/'manifest.json').read_bytes())==PARENT_MANIFEST
    m=json.loads((parent/'manifest.json').read_text())
    for path,sha in m['files'].items():assert digest((parent/path).read_bytes())==sha,path
    old=(parent/'submission.py').read_text();new=(HERE/'submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
    assert digest(old.encode())==PARENT_SOURCE
    assert apply(old,delta)==new and apply(new,delta,True)==old
    a,b=named(old),named(new);changed=[key for key in a if a[key]!=b.get(key)]
    assert sorted(changed)==['_rqbake','_rqsbake'] and set(b)-set(a)=={'_rqconfirm'}
    for key in ('prepare','kernel_fn','_time_us','_rqstart','_rqext','_rqsrc','_rqwg','_rqss','_rqcall','_rqval','RQCU','RWCU','RQFLAGS','RQPROD','RQBAKE','RQSBAKE','RQPROD2','RQV','RQKS','RQTOL','PREPARE_BUDGET_S','MARGIN','_rsbake'):
        assert a[key]==b[key],key
    assert value(new,'PREPARE_BUDGET_S')==240.0 and value(new,'MARGIN')==.99
    # Every literal GPU source string stays byte identical; no compilation needed.
    gpu_strings={}
    for node in ast.parse(old).body:
        if not isinstance(node,ast.Assign):continue
        try:literal=ast.literal_eval(node.value)
        except (ValueError,TypeError):continue
        if isinstance(literal,str) and any(token in literal for token in ('__global__','__device__','#include')):
            for target in node.targets:
                if isinstance(target,ast.Name):
                    other=value(new,target.id);assert literal==other
                    gpu_strings[target.id]=dict(bytes=len(literal.encode()),sha256=digest(literal.encode()))
    def generate(source,names):
        env={'RQCU':value(source,'RQCU')}
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss','_rqsrc'):
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure-factory>','exec'),env)
        variants={v[0]:v for v in value(source,'RQV')}
        return env['_rqsrc']([variants[name] for name in names])
    productions=value(old,'RQPROD');small=list(dict.fromkeys(name for values in value(old,'RQSBAKE').values() for name,ks in values))
    generated={}
    for label,names in [('RQ128',productions),('RQ32',small)]:
        first,second=generate(old,names),generate(new,names);assert first==second
        generated[label]=dict(instances=len(names),bytes=len(first.encode()),sha256=digest(first.encode()))
    negatives=[]
    for edit in delta['edits']:
        try:apply(old.replace(edit['before'],edit['before']+'\n'+edit['before'],1),delta)
        except ValueError:negatives.append(edit['id'])
        else:raise AssertionError('duplicate anchor accepted '+edit['id'])
    from proj_rsi.static_check import check_source
    lint=check_source(new,str(HERE/'submission.py'));assert lint['ok'],lint
    return dict(state='PASS_ENTIRE_PARENT_REVERSE_AND_ALL_CUDA_IDENTITY',parent_source_sha256=PARENT_SOURCE,
                changed_named_AST=sorted(changed),added_named_AST=['_rqconfirm'],GPU_source_strings=gpu_strings,
                generated_production_CUDA=generated,strict_anchor_negatives=negatives,prepare_clock_and_threads='ENTIRE_PREPARE_AND_RQSTART_AST_EXACT',lint=lint)

class Clock:
    def __init__(self,wall=0.0,mono=0.0):self.wall=wall;self.mono=mono
    def time(self):return self.wall
    def monotonic(self):return self.mono
    def advance(self,seconds):self.wall+=seconds;self.mono+=seconds

def harness(values,*,duration=.001,wall=0.0,used=0.0,cands=None,screen=None,throw_at=None,return_code=0,existing_state=None,layer='arbitrary',m=32,wall_shift_per_call=0.0):
    code=ast.parse((HERE/'submission.py').read_text());fn=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='_rqconfirm')
    clock=Clock(wall=wall);calls=[];requests=[];log=io.StringIO()
    cands=cands if cands is not None else [(90.0,'A',{'id':'A'}),(91.0,'B',{'id':'B'})]
    screen=screen if screen is not None else ('rq:A',cands[0][2],cands[0][0])
    st=existing_state if existing_state is not None else {'rq_confirm':{'used_s':used,'records':[]}}
    def rqcall(state,plan,x,out):requests.append(plan['id']);return return_code
    def incumbent(x,out):requests.append('incumbent');return 0
    def timing(call,x,out,flush,sink,it):
        assert it==5
        i=len(calls);call(x,out);calls.append(requests[-1]);clock.advance(duration);clock.wall+=wall_shift_per_call
        if throw_at==i:raise RuntimeError()
        assert i<len(values),('extra timing request',i,len(values))
        return values[i]
    env={'time':clock,'_T0':0.0,'PREPARE_BUDGET_S':240.,'MARGIN':.99,'_time_us':timing,'_rqcall':rqcall}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<extracted-production-confirm>','exec'),env)
    with contextlib.redirect_stdout(log):selected=env['_rqconfirm'](st,layer,m,cands,incumbent,None,None,None,None,screen,('incumbent',None,100.0))
    return dict(selected=selected,calls=calls,record=st['rq_confirm']['records'][-1],state=st,clock=clock,log=log.getvalue())

def selection_proof():
    evidence={}
    cases={
      'screening_downward_outlier_corrected':([100,101,97,100,97,101,100],'rq:B'),
      'stable_fast_candidate':([100,97,98,100,98,97,100],'rq:A'),
      'noisy_low_average_loses_to_stable':([100,90,96,100,96,98,100],'rq:B'),
      'both_confirmation_blocks_required':([100,80,103,100,103,102,100],'incumbent'),
      'strict_equal_margin_rejected':([100,99,100,100,100,99,100],'incumbent'),
      'tiny_below_margin_retained':([100,98.999,100,100,100,98.999,100],'rq:A'),
      'lower_adjacent_control_is_baseline':([100,90,91,90,91,90,100],'incumbent'),
      'local_brackets_handle_drift':([100,97,98,110,108,107,120],'rq:A'),
    }
    for name,(values,expected) in cases.items():
        result=harness(values);assert result['selected'][0]==expected,(name,result['record'])
        assert result['calls']==['incumbent','A','B','incumbent','B','A','incumbent']
        assert len(result['record']['blocks'])==7
        evidence[name]=result['record']
    # A third validated candidate is never timed during confirmation.
    result=harness([100,97,98,100,98,97,100],cands=[(90,'A',{'id':'A'}),(91,'B',{'id':'B'}),(92,'C',{'id':'C'})])
    assert 'C' not in result['calls'];evidence['top_two_only']=result['record']
    one=harness([100,97,100,97,100],cands=[(90,'A',{'id':'A'})])
    assert one['selected'][0]=='rq:A' and one['calls']==['incumbent','A','incumbent','A','incumbent'];evidence['one_candidate']=one['record']
    none=harness([],screen=('incumbent',None,100));assert not none['calls'] and none['selected'][1] is None;evidence['screen_no_winner']=none['record']
    # Same decisions across all30task-shaped host identities; no layer/M branch.
    identities=[]
    for layer in ('qkv_gate','o','pooler','unpooler','router'):
        for m in (1,8,16,32,128,256):
            r=harness(cases['stable_fast_candidate'][0],layer=layer,m=m);assert r['selected'][0]=='rq:A';identities.append([layer,m])
    return dict(state='PASS_SELECTION_ADVERSARIAL_CASES',cases=evidence,shape_agnostic_identities=identities)

def budgets_errors_proof():
    evidence={}
    for name,kwargs in [('global_budget_empty',dict(used=2.0)),('prepare_cutoff',dict(wall=210.0)),('global_partial',dict(duration=.5)),('prepare_partial',dict(wall=209.5,duration=.2))]:
        r=harness([100,97,98,100,98,97,100],**kwargs)
        assert r['selected'][0]=='rq:A' and r['record']['decision']=='rq:A' and 'retain-screen' in r['record']['reason']
        assert r['record']['status']=='skipped';assert len(r['record']['blocks'])<7
        evidence[name]=r['record']
    state={'rq_confirm':{'used_s':0.0,'records':[]}}
    for i in range(3):r=harness([100,97,98,100,98,97,100],duration=.125,existing_state=state)
    assert len(state['rq_confirm']['records'])==3 and state['rq_confirm']['used_s']==2.0
    assert len(r['record']['blocks'])==2 and 'global-confirmation-budget' in r['record']['reason']
    evidence['shared_budget_across_bakes']=state['rq_confirm']
    # A local monotonic allowance still charges time when wallclock changes.
    backwards=harness([100,97,98,100,98,97,100],duration=.5,wall_shift_per_call=-10.)
    assert len(backwards['record']['blocks'])==4 and backwards['state']['rq_confirm']['used_s']==2.0
    assert 'global-confirmation-budget' in backwards['record']['reason']
    evidence['monotonic_budget_survives_wallclock_rollback']=backwards['record']
    for name,kwargs in [('empty_exception_message',dict(throw_at=2)),('launch_error',dict(return_code=-6))]:
        r=harness([100,97,98,100,98,97,100],**kwargs);assert r['selected']==('incumbent',None,100.0) and r['record']['status']=='error';evidence[name]=r['record']
    for name,bad in [('NaN',float('nan')),('infinite',float('inf')),('zero',0.),('negative',-1.)]:
        r=harness([100,bad]);assert r['selected'][1] is None and r['record']['status']=='error';evidence[name]=r['record']
    return dict(state='PASS_SHARED_BUDGET_INCOMPLETE_AND_ERROR_FALLBACKS',cases=evidence,
                budget_limit='Admission checked before each timing block; one in-flight 5replay block may cross allowance, no hard interrupt of original timing primitive')

def main():
    result=dict(state='CPU_PW39_HOST_SELECTION_PASS_GPU_UNRUN_REVIEW_REQUIRED',source=source_proof(),selection=selection_proof(),budget_errors=budgets_errors_proof(),
      candidate_source_sha256=digest((HERE/'submission.py').read_bytes()),production_imports=0,CUDA_calls=0,prepare_calls=0,extension_builds=0,
      performance='UNMEASURED; confirmation can reject genuine gains or retain old noisy screen if budget incomplete; full-policy validation needed')
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('state','candidate_source_sha256','production_imports','CUDA_calls','prepare_calls')}))
if __name__=='__main__':main()
