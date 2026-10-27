"""Execute actual new host bodies with mocked native/timing and small CPU tensor math."""
import ast,json,types
from pathlib import Path
import torch
D=Path(__file__).resolve().parent
nodes={n.name:n for n in ast.parse((D/'submission.py').read_text()).body if isinstance(n,ast.FunctionDef)}
torch.set_num_threads(1)
TARGET=(1,1,1,2,16,8,1,1,0,3)
records=[]
# Advertise production shape but use K32 solely to bound CPU transport arithmetic.
base=torch.full((4096,32),.125,dtype=torch.bfloat16)
class Weight:
    shape=(4096,8192);device=base.device
    def __getitem__(self,k):return self
    def double(self):return base.double()
proxy=types.SimpleNamespace(**{n:getattr(torch,n) for n in ['Generator','bfloat16','uint8','empty','equal','isfinite']})
proxy.randn=lambda shape,**kw:torch.randn((shape[0],32),**kw)
for name,old,new,expected,opts in [
 ('win',21,20,'keyed',{}),('m8_win',21,20,'keyed',{'m':8}),('loss',20,21,'parent',{}),('tie',21,21,'parent',{}),('tiny_positive',21.000001,21,'keyed',{}),
 ('invalid_timing',0,21,'parent',{}),('sel_owner',21,20,'keyed',{'sel':True}),('rq_precedence',21,20,'parent',{'rq':True}),
 ('budget_skip',21,20,'parent',{'clock':lambda:241}),('late_budget',21,20,'parent',{'late':True}),
 ('wrong_template',21,20,'parent',{'wrong':True}),('numerical_failure',21,20,'RAISE',{'bad':True})]:
    m=opts.get('m',1);p=(0,1,2,3,4096,4,5);owner='sel'if opts.get('sel')else'gv'
    st={'sel':{},'gv':{},'rq':{},'pooler_raw':Weight()};st[owner][('pooler',m)]=p
    if opts.get('rq'):st['rq'][('pooler',1)]={'n':4096}
    calls=[];clock=opts.get('clock',lambda:241 if opts.get('late')and len(calls)>=3 else 1)
    def launch(state,plan,x,out):
        calls.append('keyed'if len(plan)==8 else'parent')
        out.copy_((x.double()@base.double().T).to(torch.bfloat16))
        if opts.get('bad') and len(plan)==8:out.fill_(float('nan'))
        return 0
    def timing(fn,flush,it):
        assert it==9;fn();return(old if calls[-1]=='parent'else new)/1000
    env={'torch':proxy,'GVI':[TARGET if not opts.get('wrong')else(0,)*10],'_bk':lambda m:m,'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=clock),'_gvcall':launch,'_tm':timing}
    exec(compile(ast.Module(body=[nodes['_gvkey_confirm']],type_ignores=[]),'<actual-confirm-CPU>','exec'),env)
    try:env['_gvkey_confirm'](st,None)
    except AssertionError:
        assert expected=='RAISE';records.append({'case':name,'numeric_failure_propagates':True});continue
    assert expected!='RAISE'
    record=st['gvkey_choices'][('pooler',m)];assert record['choice']==expected,(name,record)
    assert(len(st[owner][('pooler',m)])==8)==(expected=='keyed')
    if record['status']=='COMPLETE_ZERO_THRESHOLD':
        assert len(calls)==8
        assert [r['order']for r in record['pairs_us']]==['AB','BA','AB','BA']
    records.append({'case':name,'choice':record['choice'],'status':record['status'],'calls':len(calls)})
# Actual dispatch body: bucket8 markers never redirect actualM2..7.
calls=[]
class Ext:
    def gvkeyrun(self,*a):calls.append('keyed');return 0
    def gvrun(self,*a):calls.append('parent');return 0
ext=Ext();env={'torch':torch,'GVI':[TARGET],'_gvext':lambda:ext}
exec(compile(ast.Module(body=[nodes['_gvcall']],type_ignores=[]),'<actual-gvcall-CPU>','exec'),env)
p=(0,1,2,3,4096,4,5);st={'gvws':1,'gvcnt':2}
for m in range(1,17):
    for marked in [False,True]:
        env['_gvcall'](st,p+('gv_keyed_choice',)if marked else p,torch.empty((m,8192)),torch.empty((m,4096)))
        assert calls[-1]==('keyed'if marked and m in(1,8)else'parent')
assert not torch.cuda.is_initialized()
report={'state':'PASS_FOCUSED_CPU_HOST_TRANSPORT','cases':records,'actual_m1_through_m16_marked_unmarked_routes':32,'native_calls_mocked':True,'math_fixture_K32_with_shape_proxy':True,'CUDA_initialized':False}
(D/'cpu-proof.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
