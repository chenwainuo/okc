"""Construct a bounded EH-only patch; CPU file operations, never imports submission."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAND=HERE.parent

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)

def apply(text,delta,reverse=False):
    entries=list(reversed(delta['edits'])) if reverse else delta['edits']
    for e in entries:
        before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
        if text.count(before)!=1:raise ValueError('nonunique/missing strict anchor: '+e['id'])
        text=text.replace(before,after,1)
    return text

def build():
    rs=(CAND/'rs32/submission.py').read_text();pw=(CAND/'pw36/submission.py').read_text()
    old=(CAND/'rs128b/submission.py').read_text()
    assert value(rs,'RQCU')==value(pw,'RQCU')
    # Exact historical EH block, with current cb/NW and trapped zero-poll safety.
    start='    const unsigned long long ce0 = INS ? clock64() : 0;'
    stop='  } else {\n    if (sink =='
    oldcu=value(old,'RQCU');cur=value(rs,'RQCU')
    historical=oldcu[oldcu.index(start):oldcu.index(stop)]
    before=cur[cur.index(start):cur.index(stop)]
    after=historical.replace('bar.sync 1, 256;','bar.sync 1, %0;').replace('asm volatile("bar.sync 1, %0;\\n" ::: "memory");','asm volatile("bar.sync 1, %0;\\n" :: "r"(NW * 32) : "memory");')
    after=after.replace('nb * 128 + 64 * g','cb + 64 * g')
    after=after.replace('constexpr bool EH = (FL >> 16) & 1, EP = (FL >> 17) & 1;', 'constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported')
    after=after.replace('bounded spin)', 'bounded spin, trap on timeout)')
    anchor='        asm volatile("fence.proxy.async.global;\\n" ::: "memory");'
    assert after.count(anchor)==1
    after=after.replace(anchor,'        if (!f) asm volatile("trap;");\n'+anchor)
    after='    if constexpr (FL & 65536) {\n'+after+'    } else {\n'+before+'    }\n'
    edits=[dict(id='current-safe-EH-epilogue',before=before,after=after),
           dict(id='EH-only-approved-domain',before='  constexpr bool HALF = (FL & 16) != 0;',after='  constexpr bool HALF = (FL & 16) != 0;\n  static_assert(!(FL & 65536) || (NT == 128 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && (FL == 65538 || FL == 65539)), "EH backport approved M128/ST5/selfzero full-output variants only");'),
           dict(id='EH-output-map-half-token-box',before='  r = rq_map(&to, out.data_ptr(), p.N, p.m, NT);',after='  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);'),
           dict(id='EH-selectable-variants',before=']\nRQI = {v[0]: i for i, v in enumerate(RQV)}',after='    ("s4s5zEH", 128, 5, 1, 0, 0, 0, 65538),\n    ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),\n]\nRQI = {v[0]: i for i, v in enumerate(RQV)}'),
           dict(id='EH-compile-selection',before='RQPROD = ["s4s5z", "s4s5hz", "s4hz", "n256s4hz", "n256s5hz", "n256s5z"]',after='RQPROD = ["s4s5z", "s4s5hz", "s4hz", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]'),
           dict(id='EH-existing-validation-bake-selection',before='RQBAKE = {128: ["s4s5z", "s4s5hz", "s4hz"], 256: ["n256s4hz", "n256s5hz", "n256s5z"]}',after='RQBAKE = {128: ["s4s5z", "s4s5hz", "s4hz", "s4s5zEH", "s4s5hzEH"], 256: ["n256s4hz", "n256s5hz", "n256s5z"]}')]
    delta=dict(version='rs32_eh_backport_v2',source_parent_sha256=sha(rs),compatible_RQCU_parent_sha256=sha(cur),historical_source_sha256=sha(old),historical_epilogue_sha256=sha(historical),edits=edits,
               semantics='EH bit65536 only; preserve HALF16/EARLY32, current zero-poll trap and generic D retirement; existing validated preparation bake selects optional variants')
    candidate=apply(rs,delta);assert apply(candidate,delta,True)==rs
    cpw=apply(pw,delta);assert apply(cpw,delta,True)==pw
    assert value(candidate,'RQCU')==value(cpw,'RQCU')
    (HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
    (HERE/'submission.py').write_text(candidate)
    (HERE/'historical-EH-epilogue.cuh').write_text(historical)
    return delta
if __name__=='__main__':build()
