"""PW41: one optional compile-time P1 ticket-placement bit over frozen PW38."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38/submission.py'
PARENT_SHA='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
LATE=1<<30
TARGETS=[(2,1,1,1,8,1,16,1,0,1<<29),
         (2,2,1,1,8,1,16,1,0,1<<27),
         (2,4,2,1,8,2,8,1,0,1<<27),
         (1,1,2,1,8,2,4,1,0,(1<<29)|1)]

def value(source,name):
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)

def catalog(source):
    """Execute only the reviewed pure catalog region, never import submission."""
    start=source.index('_0, _1, _01 = ');end=source.index('_gvmod = None',start)
    nodes=ast.parse(source[start:end]).body
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(ast.Module(body=nodes,type_ignores=[])))
    env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-GV-catalog>','exec'),env)
    return env

def edits():
    anchor='  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow\n'
    domain='\n'.join([
      '  constexpr bool P1_LATE_TICKET = ((V >> 30) & 1) != 0;',
      '  static_assert(!P1_LATE_TICKET || (WR == 1 && WK == 8 && RT == 1 && XS == 0 && (',
      '    (FMT == 2 && MT == 1 && ST == 1 && KU == 1 && S == 16 && (V & ~(1 << 30)) == (1 << 29)) ||',
      '    (FMT == 2 && MT == 2 && ST == 1 && KU == 1 && S == 16 && (V & ~(1 << 30)) == (1 << 27)) ||',
      '    (FMT == 2 && MT == 4 && ST == 2 && KU == 2 && S == 8 && (V & ~(1 << 30)) == (1 << 27)) ||',
      '    (FMT == 1 && MT == 1 && ST == 2 && KU == 2 && S == 4 && (V & ~(1 << 30)) == ((1 << 29) | 1))',
      '  )), "PW41 late ticket only four selected P1 geometry/types; no PDL/cluster/codec/other flags");',''])
    old='      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);'
    wait='    if (threadIdx.x == 0) {\n      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;'
    late='    if (threadIdx.x == 0) {\n      if constexpr (P1_LATE_TICKET) tk = atomicAdd(p.cnt + 2 * (bz * GX + bx), 1);\n      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;'
    additions='''# PW41: append four optional ticket-placement instances AFTER the old GVI
# inventory, so every old selector/counter identity remains at its original index.
_P1LT = 1 << 30
P1LATE = [
    ("rk-s16w8xLT", "router", (8,), (2, 1, 8, 1, 16, 1, 0, 1), _XZ | _P1LT),
    ("rk-s16w8zLT", "router", (16,), (2, 1, 8, 1, 16, 1, 0, 1), _ZF | _P1LT),
    ("rk-s8zLT", "router", (32,), (2, 1, 8, 2, 8, 1, 0, 2), _ZF | _P1LT),
    ("uk-s4xLT", "unpooler", (1, 8), (1, 1, 8, 2, 4, 1, 0, 2), 1 | _XZ | _P1LT),
]
GVI += list(dict.fromkeys(_gxinst(g, v, bm) for _, _, ms, g, v in P1LATE for bm in ms))
GXP += P1LATE

'''
    return [dict(id='compile-time-selected-domain',region='kernel.compile-time-domain',before=anchor,after=anchor+domain),
      dict(id='old-prologue-only-for-old-variant',region='kernel.ticket-placement',before=old,after='      if constexpr (!P1_LATE_TICKET) { if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1); }'),
      dict(id='late-ticket-immediately-before-first-acquire',region='kernel.ticket-placement',before=wait,after=late),
      dict(id='append-only-optional-catalog',region='host.catalog',before='_gvmod = None\n',after=additions+'_gvmod = None\n')]

def apply(source,changes,reverse=False):
    for e in reversed(changes) if reverse else changes:
        before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
        assert source.count(before)==1,('nonunique/missing patch anchor',e['id'])
        source=source.replace(before,after,1)
    return source

def build():
    source=PARENT.read_text();assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    changes=edits();candidate=apply(source,changes);assert apply(candidate,changes,True)==source
    old,new=catalog(source),catalog(candidate)
    assert new['GVI'][:len(old['GVI'])]==old['GVI']
    assert new['GXP'][:len(old['GXP'])]==old['GXP']
    assert new['GVI'][len(old['GVI']):]==[a[:-1]+(a[-1]|LATE,) for a in TARGETS]
    (HERE/'submission.py').write_text(candidate)
    (HERE/'delta.json').write_text(json.dumps(dict(parent_source_sha256=PARENT_SHA,source_sha256=hashlib.sha256(candidate.encode()).hexdigest(),edits=changes,old_GVI_count=len(old['GVI']),new_GVI_count=len(new['GVI']),targets=TARGETS,scope=[['router',8],['router',16],['router',32],['unpooler',1],['unpooler',8]],semantics='Only optional ticket issue placement; exactly same per-tile S even tickets and one odd flag. No forced selection/no counter reset/no removed acquire.'),indent=2)+'\n')
    print('PW41_BUILT',len(old['GVI']),len(new['GVI']))
if __name__=='__main__':build()
