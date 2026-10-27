import importlib.util,json,copy
from pathlib import Path
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('analyzer',D/'analyze.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=m.receipts((D.parents[1]/'candidates/gv53_keyed_choice_v1/cpu-test.log').read_text())
assert any(x['same_prepare_effect_us'] is not None and x['same_prepare_effect_us']>0 for x in rows)
assert any(x['record']['status']=='BUDGET_SKIP_PARENT_RETAINED'and len(x['record']['samples_us'])==3 for x in rows)
for effect,choice in [(0,'parent'),(-1,'parent'),(1e-6,'keyed')]:
 r={'m':1,'template':m.TPL,'choice':choice,'status':'COMPLETE_ZERO_THRESHOLD','pairs_us':[],'samples_us':[],'paired_median_parent_minus_keyed_us':effect}
 for b in range(4):
  r['pairs_us'].append({'parent':20+effect,'keyed':20,'order':'AB'if b%2==0 else'BA'})
  r['samples_us'] += [{'block':b,'arm':arm,'us':20+effect if arm=='parent'else 20}for arm in (['parent','keyed']if b%2==0 else['keyed','parent'])]
 assert m.receipts('GV_KEYED_CHOICE '+repr(r))[0]['record']['choice']==choice
 bad=copy.deepcopy(r);bad['choice']='parent'if choice=='keyed'else'keyed'
 try:m.receipts('GV_KEYED_CHOICE '+repr(bad))
 except AssertionError:pass
 else:raise AssertionError('wrong choice accepted')
r={'m':8,'choice':'parent','status':'INVALID_TIMING_PARENT_RETAINED','pairs_us':[],'samples_us':[{'block':0,'arm':'parent','us':float('nan')}]}
assert m.receipts('GV_KEYED_CHOICE '+repr(r))[0]['evidence_class']=='SKIP_NO_MATCHED_EFFECT'
a=json.loads((D.parents[2]/'results/proj-rsi/jobs/ac49231d41d441f6.json').read_text());actual=m.official(a);assert len(actual)==30 and sum(x['actual_profile_implementation']=='keyed'for x in actual)==2
assert m.receipts('PREP done in 140s')==[]
print(json.dumps({'state':'PASS','actual_helper_log_records':len(rows),'actual_official_rows':len(actual),'wrong_choice_negative_cases':3,'nan_partial_receipt_supported':True,'missing_receipt_not_imputed':True},indent=2))
