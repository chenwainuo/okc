"""Derive scoped RQ source with a strict-positive threshold only for its new option."""
from pathlib import Path
import ast, difflib, hashlib, json
D=Path(__file__).resolve().parent
V3=D.parent/'rq_unpooler32_descriptor_offset_v3'
ROOT=D.parents[2]
BASE=ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
old=BASE.read_text();v3=(V3/'submission.py').read_text()
assert sha(old.encode())=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
assert sha(v3.encode())=='108791d4518f21bacef618d0e62157d92d6a6bc52559fcf3e4735bea6d26c860'
a='''            for us, tag, p in sorted(cands, key=lambda c: c[0]):
                if us < best[2] * MARGIN:
                    best = ("rq:" + tag, p, us)
                break'''
b='''            for us, tag, p in sorted(cands, key=lambda c: c[0]):
                margin = 1.0 if layer == "unpooler" and m == 32 and tag == "t32s6hze_offset_unpool-k2" else MARGIN
                if us < best[2] * margin:
                    best = ("rq:" + tag, p, us)
                break'''
assert v3.count(a)==1
new=v3.replace(a,b)
delta=json.loads((V3/'delta.json').read_text())+[[a,b]]
repro=old
for before,after in delta:
    assert repro.count(before)==1
    repro=repro.replace(before,after)
assert repro==new
back=new
for before,after in reversed(delta):
    assert back.count(after)==1
    back=back.replace(after,before)
assert back==old
def nodes(t):
    return {getattr(n,'name',getattr(n.targets[0],'id','')) if isinstance(n,ast.Assign) else getattr(n,'name',''):ast.dump(n,include_attributes=False) for n in ast.parse(t).body}
x,y=nodes(v3),nodes(new)
assert x.keys()==y.keys()
assert [k for k in x if x[k]!=y[k]]==['_rqsbake']
# Execute the exact changed selection block without importing Torch/submission.
namespace={}
exec('def choose(layer,m,cands):\n    MARGIN=.99\n    best=("inc",None,100.0)\n'+ '\n'.join('    '+line[12:] for line in b.splitlines())+'\n    return best\n',namespace)
choose=namespace['choose'];name='t32s6hze_offset_unpool-k2'
cases=[]
for layer,m,tag in [('unpooler',32,name),('pooler',32,name),('unpooler',16,name),('unpooler',32,'t32s6hze-k2')]:
    for us in [98.9,99.0,99.8,100.0,100.1]:
        got=choose(layer,m,[(us,tag,{'i':1})])[1] is not None
        special=layer=='unpooler' and m==32 and tag==name
        expected=us<100*(1.0 if special else .99)
        assert got==expected
        cases.append(dict(layer=layer,m=m,tag=tag,us=us,selected=got))
assert choose('unpooler',32,[(99.8,name,1),(98.8,'old-k2',2)])[1]==2
assert choose('unpooler',32,[(99.8,name,1),(99.9,'old-k2',2)])[1]==1
assert choose('unpooler',32,[])[1] is None
(D/'submission.py').write_text(new)
(D/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
(D/'delta.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='inc501d',tofile='rq_unpooler32_descriptor_offset_v4')))
proof={'state':'CPU_SOURCE_POLICY_PROVEN_ACTUAL_SELECTION_UNRUN','source_sha256':sha(new.encode()),'parent_sha256':sha(old.encode()),'scoped_v3_source_sha256':sha(v3.encode()),'exact_eight_delta_inverse_to_parent':True,'only_v3_AST_delta':'_rqsbake','only_v3_change':{'layer':'unpooler','m':32,'tag':name,'factor':1.0,'comparison':'strict <; ties rejected'},'old_candidates_factor':.99,'unchanged_sorting_fastest_only_budget_timing_validation':True,'policy_cases':cases,'additional_order_empty_cases':3,'RQCU_exact_v3':True,'native_evidence_reused':'../rq_unpooler32_descriptor_offset_v3/compile-proof.json','limits':['Actual preparation exposure, selected execution, new SO placement and official score remain unmeasured.','No new kernel compilation or GPU run; v3 native proof applies because RQCU is byte-identical.']}
(D/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
