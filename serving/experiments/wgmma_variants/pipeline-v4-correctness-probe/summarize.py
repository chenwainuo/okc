"""Validate both correctness reports and bind output hashes; no timing claims."""
import hashlib,json
from pathlib import Path
out=Path('/output');source=Path('/probe')
rows=[]
for line in (out/'routed.log').read_text().splitlines():
    try:value=json.loads(line)
    except ValueError:continue
    if isinstance(value,dict):rows.append(value)
tests=[r for r in rows if r.get('pass') is True and 'cases' in r]
assert len(tests)==1 and tests[0]['cases']==['mixed','skew','empty','graph_valid','graph_empty_after_valid']
provenance=[r for r in rows if r.get('status')=='complete' and 'jit_libraries' in r]
assert len(provenance)==1 and provenance[0]['jit_libraries']
full=json.loads((out/'full-correctness.json').read_text())
assert full['status']=='complete' and len(full['results'])==9
assert all(r['pass'] is True for r in full['results'])
expected=['eager_4_normal','eager_33_normal','eager_257_normal','eager_128_skewed','eager_1_normal','eager_empty','graph_valid','graph_empty_after_valid','graph_fresh_skewed']
assert [r['kind'] for r in full['results']]==expected
pins=json.loads(Path('/control/source-pins.json').read_text())
for f,h in pins['source_files'].items():assert hashlib.sha256((source/f).read_bytes()).hexdigest()==h
report={'status':'complete','scope':'development correctness only; no performance or promotion claim','routed_cases':5,'full_expert_cases':9,'source_pins':pins,'report_sha256':{f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('routed.log','full.log','full-correctness.json','input-pins.json')}}
(out/'correctness-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
