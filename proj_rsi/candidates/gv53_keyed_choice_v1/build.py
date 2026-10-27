"""New immutable optional contender; does not modify its frozen parent."""
import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'gv53_keyed_half_v1/submission.py'
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def make():
    original=P.read_text(); assert digest(original)=='239b530cf7eead0a2dbf0af5c29aea4ba02e449500eb80e45b255b361cf2f900'
    s=original; edits=[]
    def edit(old,new):
        nonlocal s
        assert s.count(old)==1,(old[:100],s.count(old));s=s.replace(old,new,1);edits.append({'old':old,'new':new})
    edit('int XS, int V>\nstatic int64_t gvl(', 'int XS, int V, bool KEYED = false>\nstatic int64_t gvl(')
    edit('  if constexpr(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3) {\n    if(p.N==4096 && p.K==8192 && (p.m==1 || p.m==8)) k=gv_keyed<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;\n  }', '  if constexpr(KEYED) {\n    static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"keyed host entry exact tuple");\n    if(p.N!=4096 || p.K!=8192 || (p.m!=1 && p.m!=8)) return -3;\n    k=gv_keyed<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;\n  }')
    a=s.index('static int64_t gvgo(');b=s.index('\nint64_t gvrun(',a)
    wrapper=s[a:b].replace('static int64_t gvgo(int64_t inst,','int64_t gvkeyrun(',1).replace(', int O) {',') {',1)
    wrapper=wrapper[:wrapper.index('  const int occ = O;')]+'  return gvl<1,1,1,2,16,8,1,1,0,3,true>(p,st,0);\n}\n'
    edit('\nint64_t gvrun(int64_t inst,','\n'+wrapper+'\nint64_t gvrun(int64_t inst,')
    edit('load_inline("gv53_keyed_half_v1_%x"','load_inline("gv53_keyed_choice_v1_%x"')
    declaration='int64_t gvkeyrun(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); '
    edit('cpp_sources="int64_t gvrun(int64_t, torch::Tensor','cpp_sources="'+declaration+'int64_t gvrun(int64_t, torch::Tensor')
    edit('functions=["gvrun", "gvocc", "gvregs"','functions=["gvkeyrun", "gvrun", "gvocc", "gvregs"')
    branch='''    if len(p) == 8 and p[-1] == "gv_keyed_choice" and x.shape[0] in (1,8):
        assert x.shape[1] == 8192 and out.shape[1] == 4096 and tuple(GVI[p[0]]) == (1,1,1,2,16,8,1,1,0,3)
        return _gvext().gvkeyrun(p[1],p[5],x,out,p[2],st["gvws"],st["gvcnt"],p[3],p[6])
'''
    edit('def _gvcall(st, p, x, out):\n','def _gvcall(st, p, x, out):\n'+branch)
    helper=(D/'confirm.py').read_text()
    edit('\ndef prepare(weights):','\n'+helper+'\n\ndef prepare(weights):')
    edit('    st.pop("gvc12", None)\n','    if GV:\n        _gvkey_confirm(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))\n    st.pop("gvc12", None)\n')
    restored=s
    for e in reversed(edits): assert restored.count(e['new'])==1; restored=restored.replace(e['new'],e['old'],1)
    assert restored==original
    before=ast.parse(original);after=ast.parse(s)
    bn={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};an={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    changed=[k for k in bn if ast.dump(bn[k],include_attributes=False)!=ast.dump(an[k],include_attributes=False)]
    assert changed==['_gvext','_gvcall','prepare'],changed
    (D/'submission.py').write_text(s);(D/'delta.json').write_text(json.dumps(edits,indent=2)+'\n')
    (D/'source-proof.json').write_text(json.dumps({'state':'PASS_CPU_SOURCE_INVERSE','parent':str(P),'parent_sha256':digest(original),'source_sha256':digest(s),'strict_inverse':True,'changed_existing_functions':changed,'unchanged_existing_functions':len(bn)-len(changed),'native_kernel_bodies_unchanged':True,'only_host_GVCU_edits':True},indent=2)+'\n')
    print(digest(s))
if __name__=='__main__':make()
