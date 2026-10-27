"""PW42 source patch: exact generic GV retained; two guarded empty-escape twins."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38'
PIN='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
BASES=[(1,1,1,2,16,8,1,1,2,1),(1,1,1,1,16,8,1,1,2,8193)]
TRAIT=1<<30

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def value(text,name):
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)
def apply(text,edits,reverse=False):
    if isinstance(edits,dict):edits=edits['edits']
    for edit in reversed(edits) if reverse else edits:
        before,after=(edit['after'],edit['before']) if reverse else (edit['before'],edit['after'])
        if text.count(before)!=1:raise ValueError('nonunique/missing strict anchor '+edit['id'])
        text=text.replace(before,after,1)
    return text

def clone_kernel(cu):
    start='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__ void __launch_bounds__'
    a=cu.index(start);b=cu.index('\n__global__ void p1zero',a);original=cu[a:b]
    edits=[]
    def change(name,before,after):
        edits.append(dict(id=name,before=before,after=after))
    change('unique-twin-name','gv(G p) {','gv_empty_guard(G p) {\n  static_assert(FMT==1 && MT==1 && ST==1 && WK==16 && KU==8 && S==1 && RT==1 && XS==2 && ((WR==2 && V==1) || (WR==1 && V==8193)), "PW42 approved QKV empty-escape geometries only");')
    change('compile-time-repair-step','  auto step = [&](const int u, const int s) {','  auto step = [&](auto REPAIR, const int u, const int s) {')
    # Both repair sites in step are gated; initial sentinel load is unchanged.
    step=original[original.index('  auto step ='):original.index('  if (V & 2)')]
    before='if ((C12 || CE) && LDEC) {';after='if constexpr (decltype(REPAIR)::value && (C12 || CE) && LDEC) {'
    assert step.count(before)==2
    modified_step=step.replace(before,after)
    change('compile-time-repair-sites',step,modified_step.replace('  auto step = [&](const int u, const int s) {','  auto step = [&](auto REPAIR, const int u, const int s) {'))
    # Previous step edit already changed the anchor; merge signature into this
    # one strict replacement to keep unique reversible edit boundaries.
    edits.pop(1)
    loop_start='  if (V & 2) {';loop_end='  if constexpr (LAD != 0)'
    i=original.index(loop_start);j=original.index(loop_end,i);loop=original[i:j]
    empty=loop.replace('step(u0 + s, s)','step(std::false_type{}, u0 + s, s)').replace('step(u, u % ST)','step(std::false_type{}, u, u % ST)')
    repair=loop.replace('step(u0 + s, s)','step(std::true_type{}, u0 + s, s)').replace('step(u, u % ST)','step(std::true_type{}, u, u % ST)')
    change('warp-uniform-entry-dispatch',loop,'  if (nx == ~0u) {\n'+empty+'  } else {\n'+repair+'  }\n')
    clone=apply(original,edits);assert apply(clone,edits,True)==original
    return original,clone,edits

def build():
    parent=(PARENT/'submission.py').read_text();assert sha((PARENT/'manifest.json').read_text())==PIN
    assert sha(parent)=='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
    original,clone,clone_edits=clone_kernel(value(parent,'GVCU'))
    marker='\n__global__ void p1zero(uint4* o, int n16) {'
    edits=[dict(id='guarded-twin-kernel-only',region='GPU.append-only-twin',before=marker,after='\n#include <type_traits>\n\n'+clone+marker)]
    before='  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;'
    after='''  auto k = []() {
    if constexpr (V & 1073741824) return gv_empty_guard<FMT, MT, ST, WR, WK, KU, S, RT, XS, (V & ~1073741824)>;
    else return gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }();'''
    edits.append(dict(id='trait-only-host-kernel-pointer',region='host.GV-dispatch',before=before,after=after))
    constants='\nGV_EMPTY_TRAIT = 1 << 30\nGV_EMPTY_BASES = '+repr(tuple(BASES))+'\nGVI += [c[:-1] + (c[-1] | GV_EMPTY_TRAIT,) for c in GV_EMPTY_BASES]\n'
    edits.append(dict(id='append-two-optional-instances',region='host.instance-catalog',before='\n_gvmod = None\n',after=constants+'\n_gvmod = None\n'))
    helper='''def _gvee_choices(layer, bm, geo, vs, c12):
    if not c12:
        return (0,)
    extra = tuple(v | GV_EMPTY_TRAIT for v in vs
                  if layer == "qkv_gate" and bm in (1, 8)
                  and (1, 1, _gvst(geo)) + geo[:6] + (v,) in GV_EMPTY_BASES)
    return tuple(vs) + extra


'''
    edits.append(dict(id='optional-validated-bake-choices',region='host.prepare-options',before='def _gvprep(st, flush):',after=helper+'def _gvprep(st, flush):'))
    edits.append(dict(id='use-optional-choice-enumerator',region='host.prepare-options',before='                for v in vs if c12 else (0,):',after='                for v in _gvee_choices(layer, bm, geo, vs, c12):'))
    candidate=apply(parent,edits);assert apply(candidate,edits,True)==parent
    delta=dict(version='pw42_v1_guarded_empty_escape',parent_manifest_sha256=PIN,parent_source_sha256=sha(parent),edits=edits,clone_edits=clone_edits,
       original_kernel_sha256=sha(original),clone_kernel_sha256=sha(clone),approved_bases=BASES,trait=TRAIT,
       semantics='Append-only guarded GV twins; actual warp-uniform first sentinel selects empty step. Nonempty executes original repair semantics; no host histogram or assumed CUDA weight distribution.')
    (HERE/'submission.py').write_text(candidate);(HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
if __name__=='__main__':build()
