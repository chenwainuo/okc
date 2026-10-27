"""Independent new-gate edge sweep against the actual production helper; CPU only."""
import ast,copy,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=ROOT/'proj_rsi/candidates/pw43_trial/observer.py';sp=importlib.util.spec_from_file_location('pw43_v2_independent_observer',p);o=importlib.util.module_from_spec(sp);sp.loader.exec_module(o)
node=next(n for n in ast.parse((ROOT/'proj_rsi/candidates/pw43/submission.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_lt43confirm');env={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<realhelper>','exec'),env)
def case(codes,deadline,decline=None):
 c={'codes':codes,'validations':[],'timings':[]};e={'timers':[],'confirmations':[c],'deadlines':[{}]};t=[0.]
 def clock():t[0]+=.001;return t[0]
 def factory(f):
  def valid(code):
   yes=(f,code)!=decline;c['validations'].append({'fixture':f,'code':code,'valid':yes});return yes
  def timed(code):
   v=1. if code<0 else .9999 if code==7 else 1.0001;c['timings'].append({'fixture':f,'code':code,'returned_ms':v});e['timers'].append({'it':3,'raw_ms':[v]*3,'returned_ms':v,'context':'confirmation_timing','algorithm_code':code,'confirmation_fixture':f});return v
  return valid,timed
 selected,record=env['_lt43confirm'](codes,factory,deadline,clock);c.update(returned=selected,record=record);e['returned']=selected;a={'searches':[e],'observer_detached':True,'lt_mm_status':[],'printed_records':[]};o.gate(a);return a
count=0
for codes in [[],[7],[7,8]]:
 for n in range(130):case(codes,.001*n);count+=1
for decline in [(0,7),(0,8),(1,7),(1,8)]:case([7,8],100,decline);count+=1
positive=case([7,8],100)
negative=copy.deepcopy(positive);negative['searches'][0]['timers'][-1]['confirmation_fixture']=0
try:o.gate(negative)
except AssertionError:pass
else:raise AssertionError('wrong raw timer fixture accepted')
negative=copy.deepcopy(positive)
for row in negative['searches'][0]['confirmations'][0]['record']['rows']:row['fixture']=0;row['round']=0
try:o.gate(negative)
except AssertionError:pass
else:raise AssertionError('original concrete wrong-fixture gap remains')
print(json.dumps({'state':'PW43_V2_INDEPENDENT_PREFIX_SWEEP_PASS','positive_deadline_validation_cases':count,'extra_negative_controls':['wrong_raw_timer_fixture','original_all_fixture0_round0_gap'],'GPU_calls':0}))
