"""Compose exact retained PW44 edits onto a pinned current parent; CPU only."""
import ast, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PARENTS={'pw45':('proj_rsi/candidates/pw45/submission.py','835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3'),'rq32_predxor':('proj_rsi/candidates/rq32_predxor/submission.py','cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210')}
def sha(b):return hashlib.sha256(b).hexdigest()
def apply(text,edits,reverse=False):
 for e in reversed(edits) if reverse else edits:
  before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
  assert text.count(before)==1,('strict unique anchor',e['id'])
  text=text.replace(before,after,1)
 return text

def check():
 d=json.loads((HERE/'composition.json').read_text())
 for rel,pin in d['pins'].items():assert sha((ROOT/rel).read_bytes())==pin,rel
 for label,e in d['compositions'].items():
  old=(ROOT/e['parent']).read_text();new=(HERE/e['file']).read_text()
  assert apply(old,d['edits'])==new and apply(new,d['edits'],True)==old
  ast.parse(new)
 return {'state':'CPU_DELTA_INVERSE_PASS_GPU_UNRUN','parents':d['compositions'],'reused':'PW44 kernel/protocol proofs; PW45 or RQ32 parent proofs; no new fullscope harness'}

def build():
 edits=json.loads((ROOT/'proj_rsi/candidates/pw44/delta.json').read_text())['edits'];rows={};pins={}
 for label,(rel,pin) in PARENTS.items():
  old=(ROOT/rel).read_text();assert sha(old.encode())==pin
  new=apply(old,edits);assert apply(new,edits,True)==old;ast.parse(new)
  name=label+'.submission.py';(HERE/name).write_text(new)
  rows[label]={'parent':rel,'parent_sha256':pin,'file':name,'source_sha256':sha(new.encode())};pins[rel]=pin;pins[str((HERE/name).relative_to(ROOT))]=sha(new.encode())
 pins['proj_rsi/candidates/pw44/delta.json']=sha((ROOT/'proj_rsi/candidates/pw44/delta.json').read_bytes())
 (HERE/'composition.json').write_text(json.dumps({'edits':edits,'pins':pins,'compositions':rows},indent=2)+'\n')
 (HERE/'cpu-proof.json').write_text(json.dumps(check(),indent=2)+'\n')
if __name__=='__main__':build();print(json.dumps(check(),indent=2))
