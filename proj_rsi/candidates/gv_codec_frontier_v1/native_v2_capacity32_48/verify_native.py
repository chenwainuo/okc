"""Read-only focused source inverse and native control audit; CPU only."""
import ast
import json
import re
from pathlib import Path

D=Path(__file__).resolve().parent
ROOT=D.parents[3]
source=(ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()
cu=next(ast.literal_eval(n.value)for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
prefix='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__'
start=cu.index(prefix,cu.index('void gv_fix_pred'))
control=cu[start:cu.index('\n__global__ void p1zero',start)].strip()
native=(D/'native.cu').read_text()
assert native.count(control)==1
start=native.index(prefix,native.index(' gv_predxor(G p)'))
candidate=native[start:native.index('\ntemplate __global__ void gv_predxor',start)].strip()
delta=json.loads((D/'delta.json').read_text())
restored=candidate
for change in reversed(delta['changes']):
    assert restored.count(change['new'])==change['occurrences'],change['new'][:80]
    restored=restored.replace(change['new'],change['old'])
restored=restored.replace(' gv_frontier(G p, Frontier fp) {',' gv_predxor(G p) {',1)
assert restored==control

def streams(text):
    return {sym:[(int(m[1],16),re.sub(r'\s+',' ',m[2].strip()),m[3],m[4])for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',body)]for sym,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S)}
actual=streams((D/'native.sass').read_text())
archive=streams((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/candidate-full.sass').read_text())
assert len(actual)==4
controls={k:v for k,v in actual.items()if'gv_predxor'in k}
assert len(controls)==2
for symbol,rows in controls.items():
    assert rows==archive[symbol],symbol
print(json.dumps(dict(status='SOURCE_INVERSE_AND_TWO_ARCHIVED_CONTROLS_PASS_GPU_UNRUN',
    changes=len(delta['changes']),unmodified_math_reduction_output_source=True,
    control_native_PC_predicate_encoding_exact=True,kernel_count=4),indent=2))
