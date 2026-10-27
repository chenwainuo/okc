"""Small read-only source-composition proof; stdlib only, no production imports/builds."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
PARENTS={
 'pw45':('proj_rsi/candidates/pw45/submission.py','835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3'),
 'rq32_predxor':('proj_rsi/candidates/rq32_predxor/submission.py','cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210'),
 'gv_predxor_pw45_v2':('proj_rsi/candidates/gv_predxor_pw45_v2/submission.py','501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9')}
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def changes(a,b):
 assert len(a)==len(b)==176
 return [i for i,(x,y) in enumerate(zip(a,b))if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False)]
def apply(text,old,new,indices):
 a=ast.parse(old).body;b=ast.parse(new).body
 for i in indices:
  before=ast.get_source_segment(old,a[i]);after=ast.get_source_segment(new,b[i]);assert text.count(before)==1
  text=text.replace(before,after,1)
 return text
s={k:(ROOT/p).read_text()for k,(p,h)in PARENTS.items()}
for k,(p,h)in PARENTS.items():assert sha(s[k])==h,(k,'parent drift')
b,r,g=(s[k]for k in PARENTS);c=(D/'submission.py').read_text();bt,rt,gt,ct=(ast.parse(x).body for x in (b,r,g,c))
assert changes(bt,rt)==[149]
assert changes(bt,gt)==[10,57]
assert changes(bt,ct)==[10,57,149]
assert changes(rt,ct)==[10,57]
assert changes(gt,ct)==[149]
assert apply(r,b,g,[10,57])==c==apply(g,b,r,[149]),'byte-exact commutativity'
assert apply(c,g,b,[10,57])==r,'removing GV returns exact RQ parent'
assert apply(c,r,b,[149])==g,'removing RQ returns exact GV parent'
assert apply(apply(c,g,b,[10,57]),r,b,[149])==b
m=D/'manifest.json'
if m.exists():
 manifest=json.loads(m.read_text())
 for p,h in manifest['files'].items():assert hashlib.sha256((D/p).read_bytes()).hexdigest()==h,p
 for p,h in manifest['parents'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
receipt={'state':'CPU_SOURCE_COMPOSITION_PASS_GPU_UNRUN','source_sha256':sha(c),'parent_source_sha256':{k:h for k,(p,h)in PARENTS.items()},'top_level_AST_nodes':176,'base_to_RQ_nodes':[{'index':149,'name':'RQCU'}],'base_to_GV_nodes':[{'index':10,'name':'GVCU'},{'index':57,'name':'_gvext'}],'combined_changed_nodes':[10,57,149],'overlapping_nodes':[],'byte_exact_patch_orders_commute':True,'remove_GV_recovers_exact_RQ':True,'remove_RQ_recovers_exact_GV':True,'remove_both_recovers_exact_PW45':True,'all_other_AST_nodes_exact':True,'compile_or_GPU_validation':False,'combined_gain_claim':False}
print(json.dumps(receipt,indent=2))
