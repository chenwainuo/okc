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
extended=json.loads((out/'routed-extended.json').read_text())
assert extended['status']=='complete' and extended['case_count']==42 and len(extended['cases'])==42
expected={(b,k) for b in (8,16,24,32,48,64) for k in ('partial','skew','empty','allpad','graph_valid','graph_empty_after_valid','graph_valid_restored')}
assert {(c['block_m'],c['kind']) for c in extended['cases']}==expected
assert all(c['pass'] is True and c['token_tile']==(16 if c['block_m']%16==0 else 8) for c in extended['cases'])
prefill=json.loads((out/'prefill-correctness.json').read_text())
assert prefill['status']=='complete' and [c['batch'] for c in prefill['cases']]==[3012,16384]
assert all(c['official_reference_pass'] is True for c in prefill['cases'])
pins=json.loads(Path('/control/source-pins.json').read_text())
for f,h in pins['source_files'].items():assert hashlib.sha256((source/f).read_bytes()).hexdigest()==h
report={'status':'complete','scope':'development correctness only; no performance or promotion claim','routed_original_cases':5,'routed_extended_cases':42,'full_expert_original_cases':9,'full_expert_prefill_cases':2,'source_pins':pins,'report_sha256':{f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ('routed.log','routed-extended.json','full.log','full-correctness.json','prefill-correctness.json','input-pins.json')}}
(out/'correctness-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
