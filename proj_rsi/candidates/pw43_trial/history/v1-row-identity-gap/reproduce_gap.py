"""Historical v1 metadata gate gap reproduction, pure CPU, no production import."""
import ast,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v1_gate',HERE/'observer.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
source=HERE.parents[2]/'pw43/submission.py';env={}
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_lt43confirm')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-helper>','exec'),env)
c={'codes':[7,8],'validations':[],'timings':[]}
def factory(fixture):
 def valid(code):c['validations'].append(dict(fixture=fixture,code=code,valid=True));return True
 def timed(code):
  value=1. if code<0 else .999
  c['timings'].append(dict(fixture=fixture,code=code,returned_ms=value));return value
 return valid,timed
choice,record=env['_lt43confirm'](c['codes'],factory,100,lambda:0);c.update(returned=choice,record=record)
for row in record['rows']:row['fixture']=row['round']=0
old.gate(dict(searches=[dict(timers=[],confirmations=[c],returned=choice)],printed_records=[],lt_mm_status=[],observer_detached=True))
print(json.dumps(dict(status='V1_WRONG_ALL_ZERO_FIXTURE_ROUND_ACCEPTED_CPU_REPRODUCTION',GPU_calls=0)))
