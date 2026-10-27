"""Immutable integration, original501d bakes + disjoint late implementation choices."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2];P=D.parent/'gv53_keyed_choice_v1/submission.py';E=D.parent/'gv53_earlyload_v1'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def build():
 s=P.read_text();assert sha(s)=='78f93581a011fc7c0088032542faccb7617c6a5402ef79aded44a1aa33988597';old=s;edits=[]
 def edit(a,b):
  nonlocal s
  assert s.count(a)==1,(a[:120],s.count(a));s=s.replace(a,b,1);edits.append({'old':a,'new':b})
 early=json.loads((E/'delta.json').read_text())['new_kernel'];assert 'gv_earlyload(G p)'in early
 edit('\n__global__ void p1zero','\n'+early+'\n__global__ void p1zero')
 edit('int XS, int V, bool KEYED = false>\nstatic int64_t gvl(', 'int XS, int V, bool KEYED = false, bool EARLY = false>\nstatic int64_t gvl(')
 marker='  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;'
 branch='''  if constexpr(EARLY) {
    static_assert(!KEYED && FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"earlyload host entry exact tuple");
    if(p.N!=4096 || p.K!=8192 || p.m!=8) return -3;
    k=gv_earlyload<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
  }
'''
 edit(marker,branch+marker)
 a=s.index('int64_t gvkeyrun(');b=s.index('\nint64_t gvrun(',a);keywrapper=s[a:b];earlywrapper=keywrapper.replace('gvkeyrun(','gvearlyrun(',1).replace('1,1,0,3,true>','1,1,0,3,false,true>');assert earlywrapper!=keywrapper
 edit('\nint64_t gvrun(', '\n'+earlywrapper+'\nint64_t gvrun(')
 edit('load_inline("gv53_keyed_choice_v1_%x"','load_inline("gv53_selected_earlyload_v1_%x"')
 declaration='int64_t gvearlyrun(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); '
 edit('cpp_sources="int64_t gvkeyrun(', 'cpp_sources="'+declaration+'int64_t gvkeyrun(')
 edit('functions=["gvkeyrun",','functions=["gvearlyrun", "gvkeyrun",')
 edit('if len(p) == 8 and p[-1] == "gv_keyed_choice" and x.shape[0] in (1,8):','if len(p) == 8 and p[-1] == "gv_keyed_choice" and x.shape[0] == 1:')
 dispatch='''    if len(p) == 8 and p[-1] == "gv_earlyload_choice" and x.shape[0] == 8:
        assert x.shape[1] == 8192 and out.shape[1] == 4096 and tuple(GVI[p[0]]) == (1,1,1,2,16,8,1,1,0,3)
        return _gvext().gvearlyrun(p[1],p[5],x,out,p[2],st["gvws"],st["gvcnt"],p[3],p[6])
'''
 edit('def _gvcall(st, p, x, out):\n','def _gvcall(st, p, x, out):\n'+dispatch)
 # Preserve exactM1 helper behavior; M8 compares only its measured earlyload alternative.
 edit('    for m in (1,8):\n        layer, n, key = "pooler", 4096, ("pooler",_bk(m))','    for m in (1,):\n        layer, n, key = "pooler", 4096, ("pooler",_bk(m))')
 helper=(D.parent/'gv53_keyed_choice_v1/confirm.py').read_text().replace('_gvkey_confirm','_gvearly_confirm').replace('for m in (1,8):','for m in (8,):').replace('gvkey_choices','gvearly_choices').replace('GV_KEYED_CHOICE','GV_EARLYLOAD_CHOICE').replace('gv_keyed_choice','gv_earlyload_choice').replace('"keyed"','"earlyload"').replace('parent_minus_keyed','parent_minus_earlyload').replace('r["keyed"]','r["earlyload"]')
 # unquoted pairedfield replacement already covered literalquotedkey; assert no stale keylabel.
 assert '"keyed"'not in helper and 'keyed_us'not in helper
 edit('\ndef prepare(weights):','\n'+helper+'\n\ndef prepare(weights):')
 edit('        _gvkey_confirm(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))','        implementation_flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)\n        _gvkey_confirm(st, implementation_flush)\n        _gvearly_confirm(st, implementation_flush)')
 restored=s
 for e in reversed(edits):assert restored.count(e['new'])==1;restored=restored.replace(e['new'],e['old'],1)
 assert restored==old
 (D/'submission.py').write_text(s);(D/'delta.json').write_text(json.dumps(edits,indent=2)+'\n');(D/'early-confirm.py').write_text(helper)
 pilot=(D.parent/'gv53_keyed_half_v1/pilot.cu').read_text()+'\n'+early+'\ntemplate __global__ void gv_earlyload<1,1,1,2,16,8,1,1,0,3>(G);\n';(D/'pilot.cu').write_text(pilot)
 proof={'state':'CPU_SOURCE_INVERSE_PASS','source_sha256':sha(s),'parent_sha256':sha(old),'root_incumbent_sha256':'501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9','whole_source_inverse_to_78f_exact':True,'early_kernel_exact_frozen_81fd':True,'keyed_kernel_unchanged':True,'default_parent_gvrun_bakes_unchanged':True,'M1_optional_choices':['parent','keyed'],'M8_optional_choices':['parent','earlyload'],'keyed_plus_earlyload_composition':False,'M8_keyed_evidence_preserved_not_selected_here':True,'M2_through_M7_markers_fall_back_to_parent':True}
 (D/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(sha(s))
if __name__=='__main__':build()
