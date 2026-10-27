"""Read-only two-condition source proof; no Torch/production import or compiler."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT='proj_rsi/candidates/rq32_gv_predxor_pw45_v1/submission.py'
PARENT_SHA='137e0dbce0f0126ef9c72d4b6e9b48e0448b02d0c397d9cc0ba927bac2d2a4a6'
EDITS=[
 {'function':'_gvprep',
  'before':'if best[1] is not None and best[0] > t12 * GVMARGIN:',
  'after':'if best[1] is not None and best[0] > t12 * (1.0 if (layer, bm) in (("pooler", 1), ("pooler", 8), ("pooler", 16), ("unpooler", 16)) else GVMARGIN):'},
 {'function':'_rqsbake',
  'before':'if us < best[2] * MARGIN:',
  'after':'if us < best[2] * (1.0 if (layer, m) in (("qkv_gate", 32), ("o", 32), ("pooler", 32), ("unpooler", 32)) else MARGIN):'}]

def sha(raw):return hashlib.sha256(raw).hexdigest()
def transform(source,reverse=False):
    for edit in reversed(EDITS) if reverse else EDITS:
        nodes={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
        node=nodes[edit['function']]
        segment=ast.get_source_segment(source,node)
        before,after=(edit['after'],edit['before']) if reverse else (edit['before'],edit['after'])
        assert segment.count(before)==1,edit['function']
        changed=segment.replace(before,after,1)
        assert source.count(segment)==1
        source=source.replace(segment,changed,1)
    return source

def check():
    parent=(ROOT/PARENT).read_bytes();child=(HERE/'submission.py').read_bytes()
    assert sha(parent)==PARENT_SHA
    assert transform(parent.decode()).encode()==child
    assert transform(child.decode(),True).encode()==parent
    old=ast.parse(parent).body;new=ast.parse(child).body
    assert len(old)==len(new)==176
    changed=[getattr(a,'name',None) for a,b in zip(old,new)if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False)]
    assert changed==['_gvprep','_rqsbake'],changed
    # Every changed expression's untargeted branch remains exact original margin.
    shapes={'GV':(('pooler',1),('pooler',8),('pooler',16),('unpooler',16)),
            'RQ32':(('qkv_gate',32),('o',32),('pooler',32),('unpooler',32))}
    for edit in EDITS:
        before=ast.parse(edit['before']+'\n pass').body[0].test
        after=ast.parse(edit['after']+'\n pass').body[0].test
        old_factor=next(n for n in ast.walk(before)if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Mult)).right
        new_factor=next(n for n in ast.walk(after)if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Mult)).right
        assert isinstance(new_factor,ast.IfExp) and isinstance(new_factor.body,ast.Constant) and new_factor.body.value==1.0
        assert ast.dump(old_factor)==ast.dump(new_factor.orelse)
    # Direct numerical examples show admission-window change without a new floor.
    examples=[]
    for incumbent,candidate in ((10.0,9.95),(10.0,9.999999),(10.0,10.0),(10.0,10.01)):
        examples.append({'incumbent_us':incumbent,'candidate_us':candidate,
            'RQ_old_accept':candidate<incumbent*.99,'RQ_target_accept':candidate<incumbent*1.0,
            'GV_old_final_accept':not(candidate>incumbent*.99),'GV_target_final_accept':not(candidate>incumbent*1.0)})
    return {'state':'CPU_TWO_ALLOWLISTED_MARGIN_EDITS_PASS_GPU_UNRUN','parent_sha256':sha(parent),
        'source_sha256':sha(child),'top_level_nodes':176,'changed_functions':changed,
        'all_other_174_nodes_exact':True,'strict_forward_inverse_byte_exact':True,
        'allowlists':shapes,'examples':examples,'unchanged':['all CUDA/source/compile factories','GVMARGIN=.99 and MARGIN=.99 globals','all other selection gates','candidate catalogs/order/retests','numerical/pack/launch/counter guards','prepare clocks/time budgets','fallbacks and serving contract'],
        'limits':['No measured gain or statistical confidence claim.','Original GV > final rejection comparator is retained, so exact final ties can survive the targeted gate; RQ < retains incumbent on ties.','No production/execution state changed.']}

if __name__=='__main__':print(json.dumps(check(),indent=2))
