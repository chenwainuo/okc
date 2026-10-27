"""Typed narrow W-ring option over PW38; exact commuting kernel/catalog edits."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38'
WRING=HERE.parent/'pw36_wring_rebase'
PW38_MANIFEST='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
WRING_MANIFEST='20ffb258d67a4ed205298103e0e34e9f06d929957c4cc670dd5985deccc1c299'
BASE_SOURCE='2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)
def apply(text,edits,reverse=False):
    if isinstance(edits,dict):edits=edits['edits']
    for e in reversed(edits) if reverse else edits:
        before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
        if text.count(before)!=1:raise ValueError('nonunique/missing strict anchor '+e['id'])
        text=text.replace(before,after,1)
    return text

def build():
    assert sha((PARENT/'manifest.json').read_text())==PW38_MANIFEST
    assert sha((WRING/'manifest.json').read_text())==WRING_MANIFEST
    parent=(PARENT/'submission.py').read_text();base=(HERE.parent/'pw36/submission.py').read_text()
    assert sha(parent)=='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e' and sha(base)==BASE_SOURCE
    research_delta=json.loads((WRING/'delta.json').read_text());eh_delta=json.loads((PARENT/'delta.json').read_text())
    # Seven proven kernel edits are reused exactly. Catalog edit uses a stable
    # old-row anchor before EH, so both orders have identical full source bytes.
    core=[dict(e,region='kernel.layout-and-publication') for e in research_delta['edits'][:-1]]
    anchor='    ("h128s8z", 128, 8, 1, 0, 0, 0, 18),\n'
    split='    ("wring_x6w4hz", 128, 6, 1, 0, 0, 0, 1027),\n'
    core.append(dict(id='optional-split-catalog-only',region='variant.catalog',before=anchor,after=anchor+split))
    original_compile='RQPROD = ["s4s5z", "s4s5hz", "s4hz", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]'
    policy=[dict(id='compile-one-optional-split',region='host.compile-catalog',before=original_compile,after=original_compile[:-1]+', "wring_x6w4hz"]'),
      dict(id='qkv128-only-optional-bake',region='host.bake-candidate-enumeration',before='            for nm in RQBAKE[m]:',after='            for nm in RQBAKE[m] + (["wring_x6w4hz"] if layer == "qkv_gate" and m == 128 else []):')]
    # Exact full-source composition before optional host policy, no normalization.
    ew=apply(apply(base,eh_delta),core);we=apply(apply(base,core),eh_delta)
    assert ew==we
    candidate=apply(ew,policy)
    assert apply(candidate,core+policy,True)==parent
    # Frozen research ancestor differs from portable core only by its unused
    # negative equal-depth catalog row. Its entire source is restored in inverse.
    equal='    ("wring_x5w5hz", 128, 5, 1, 0, 0, 0, 1283),\n'
    research=(WRING/'submission.py').read_text();assert research.count(equal)==1
    portable=research.replace(equal,'',1);assert portable==apply(base,core)
    assert apply(apply(portable,eh_delta),policy)==candidate
    restored=apply(apply(candidate,policy,True),eh_delta,True)
    restored=restored.replace(split,equal+split,1);assert restored==research
    delta=dict(version='pw40_v1_optional_qkv_wring_over_pw38',parent_source_sha256=sha(parent),parent_manifest_sha256=PW38_MANIFEST,
      wring_parent_source_sha256=sha(research),wring_parent_manifest_sha256=WRING_MANIFEST,
      edits=core+policy,kernel_catalog_edits=core,host_policy_edits=policy,EH_ancestor_delta_sha256=sha((PARENT/'delta.json').read_text()),
      alternate_ancestor_bridge=dict(remove_unused_negative_equal_catalog_row=equal,split_row=split),
      semantics='Optional qkv128 x6w4 HINT+ZI KS7 contender. Existing EH/control candidates remain; no PW39 confirmation, no force, no Wring+EH same-kernel flag combination.')
    (HERE/'submission.py').write_text(candidate);(HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
    return delta
if __name__=='__main__':build()
