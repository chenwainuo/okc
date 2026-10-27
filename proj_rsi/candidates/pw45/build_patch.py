"""Strict optional o32 HALF+EARLY full-K contenders over frozen PW38."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38'
PARENT_SHA='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
PARENT_MANIFEST='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)
def apply(text,delta,reverse=False):
    edits=delta['edits'] if isinstance(delta,dict) else delta
    for e in reversed(edits) if reverse else edits:
        before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
        if text.count(before)!=1:raise ValueError('nonunique/missing '+e['id'])
        text=text.replace(before,after,1)
    return text

def build():
    parent=(PARENT/'submission.py').read_text()
    assert digest(parent)==PARENT_SHA and digest((PARENT/'manifest.json').read_text())==PARENT_MANIFEST
    anchor='    ("h128s8e", 128, 8, 1, 0, 0, 0, 52),\n'
    row='    ("o", 32): [("t32s5hze", 2), ("t32s6hze", 2), ("t32s5ze", 2)],'
    launch='  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;'
    loop='            for nm, KS in ps[:4]:'
    edits=[dict(id='two-optional-half-early-types',before=anchor,after=anchor+'    ("h32s4e", 32, 4, 1, 0, 0, 0, 52),\n    ("h32s5e", 32, 5, 1, 0, 0, 0, 52),\n'),
      dict(id='o32-only-append-options-preserve-controls',before=row,after=row[:-2]+', ("h32s4e", 1), ("h32s5e", 1)],'),
      dict(id='new-types-exact-shape-fullK-only',before=launch,after=launch+'\n  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {\n    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;\n  }'),
      dict(id='bounded-five-only-o32-keep-original-budget',before=loop,after='            for nm, KS in ps[:(5 if layer == "o" and m == 32 else 4)]:')]
    candidate=apply(parent,edits);assert apply(candidate,edits,True)==parent
    result={'version':'pw45-o32-half-early-fullK-v1','parent_source_sha256':PARENT_SHA,'parent_manifest_sha256':PARENT_MANIFEST,'edits':edits,
      'scope':'Only o32 optional ST4/ST5 unhinted FL52 HALF+EARLY+plain-store at KS1. Existing controls/order, EH, math, codec, input validation and prepare clock/guards preserved; no forced selection.'}
    (HERE/'submission.py').write_text(candidate);(HERE/'delta.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':build()
