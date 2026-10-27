"""Retained concrete v1 observer gap; CPU only, no production import."""
import ast,copy,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=HERE/'observer-v1.py';sp=importlib.util.spec_from_file_location('pw43_v1_retained_observer',p);o=importlib.util.module_from_spec(sp);sp.loader.exec_module(o)
node=next(n for n in ast.parse((ROOT/'proj_rsi/candidates/pw43/submission.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_lt43confirm');env={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<realhelper>','exec'),env)
c={'codes':[7,8],'validations':[],'timings':[]};e={'timers':[],'confirmations':[c]};t=[0.]
def clock():t[0]+=.001;return t[0]
def factory(f):
 def valid(code):c['validations'].append({'fixture':f,'code':code,'valid':True});return True
 def timed(code):
  v=1. if code<0 else .9999 if code==7 else 1.0001;c['timings'].append({'fixture':f,'code':code,'returned_ms':v});e['timers'].append({'it':3,'raw_ms':[v]*3,'returned_ms':v});return v
 return valid,timed
selected,record=env['_lt43confirm'](c['codes'],factory,100,clock);c.update(returned=selected,record=record);e['returned']=selected;a={'searches':[e],'observer_detached':True,'lt_mm_status':[],'printed_records':[]};o.gate(a)
b=copy.deepcopy(a)
for row in b['searches'][0]['confirmations'][0]['record']['rows']:row['fixture']=0;row['round']=0
o.gate(b)
print(json.dumps({'old_gate_wrong_fixture_order_accepted':True,'source_helper':'actual extracted _lt43confirm','source_unchanged':True,'GPU_calls':0,'repair_required':'exact identity/order/alignment of rows, validation IDs and timer triples'}))
