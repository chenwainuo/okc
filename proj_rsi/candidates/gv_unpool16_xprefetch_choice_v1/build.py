"""Exact inverse production delta to17d44d; no CUDA execution."""
import hashlib,json,pathlib,difflib
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];P=D.parent/'gv53_selected_earlyload_v1/submission.py';X=D.parent/'gv_unpool16_xprefetch_v1'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def build():
 s=P.read_text();parent=s;assert sha(s)=='17d44d1140ab09cdbab1f2835841e2891fc1c7c01c1eb4d9a2833c242d99a3e6';edits=[]
 def edit(a,b):
  nonlocal s
  assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b,1);edits.append({'old':a,'new':b})
 pilot=(X/'pilot.cu').read_text();pos=pilot.index(' gv_xprefetch(G p)');a=pilot.rfind('template <int FMT',0,pos);b=pilot.index('\ntemplate __global__ void',pos);body=pilot[a:b].rstrip()
 edit('\n__global__ void p1zero','\n'+body+'\n__global__ void p1zero')
 edit('bool KEYED = false, bool EARLY = false>\nstatic int64_t gvl(', 'bool KEYED = false, bool EARLY = false, bool XPREFETCH = false>\nstatic int64_t gvl(')
 marker='  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;'
 branch='''  if constexpr(XPREFETCH) {
    static_assert(!KEYED && !EARLY && FMT==1 && MT==2 && ST==2 && WR==1 && WK==4 && KU==16 && S==1 && RT==1 && XS==2 && V==1,"X-prefetch host entry exact tuple");
    if(p.N!=8192 || p.K!=4096 || p.m!=16) return -3;
    k=gv_xprefetch<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
  }
'''
 edit(marker,branch+marker)
 a=s.index('int64_t gvearlyrun(');b=s.index('\nint64_t gvrun(',a);wrapper=s[a:b].replace('gvearlyrun(','gvxprefetchrun(',1).replace('1,1,1,2,16,8,1,1,0,3,false,true>','1,2,2,1,4,16,1,1,2,1,false,false,true>')
 assert 'return gvl<1,2,2,1,4,16,1,1,2,1,false,false,true>'in wrapper
 edit('\nint64_t gvrun(','\n'+wrapper+'\nint64_t gvrun(')
 edit('load_inline("gv53_selected_earlyload_v1_%x"','load_inline("gv_unpool16_xprefetch_choice_v1_%x"')
 decl='int64_t gvxprefetchrun(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); '
 edit('cpp_sources="int64_t gvearlyrun(','cpp_sources="'+decl+'int64_t gvearlyrun(')
 edit('functions=["gvearlyrun",','functions=["gvxprefetchrun", "gvearlyrun",')
 dispatch='''    if len(p) == 8 and p[-1] == "gv_xprefetch_choice" and x.shape[0] == 16:
        assert x.shape[1] == 4096 and out.shape[1] == 8192 and tuple(GVI[p[0]]) == (1,2,2,1,4,16,1,1,2,1)
        return _gvext().gvxprefetchrun(p[1],p[5],x,out,p[2],st["gvws"],st["gvcnt"],p[3],p[6])
'''
 edit('def _gvcall(st, p, x, out):\n','def _gvcall(st, p, x, out):\n'+dispatch)
 helper=(D.parent/'gv53_selected_earlyload_v1/early-confirm.py').read_text().replace('_gvearly_confirm','_gvxprefetch_confirm').replace('for m in (8,):','for m in (16,):').replace('target = (1,1,1,2,16,8,1,1,0,3)','target = (1,2,2,1,4,16,1,1,2,1)').replace('layer, n, key = "pooler", 4096, ("pooler",_bk(m))','layer, n, key = "unpooler", 8192, ("unpooler",_bk(m))').replace('(4096,8192)','(8192,4096)').replace('gvearly_choices','gvxprefetch_choices').replace('GV_EARLYLOAD_CHOICE','GV_XPREFETCH_CHOICE').replace('gv_earlyload_choice','gv_xprefetch_choice').replace('earlyload','xprefetch')
 edit('\ndef prepare(weights):','\n'+helper+'\n\ndef prepare(weights):')
 edit('        _gvearly_confirm(st, implementation_flush)','        _gvearly_confirm(st, implementation_flush)\n        _gvxprefetch_confirm(st, implementation_flush)')
 restored=s
 for e in reversed(edits):assert restored.count(e['new'])==1;restored=restored.replace(e['new'],e['old'],1)
 assert restored==parent
 compile(s,'submission.py','exec')
 (D/'submission.py').write_text(s);(D/'delta.json').write_text(json.dumps(edits,indent=2)+'\n');(D/'xprefetch-confirm.py').write_text(helper)
 (D/'source.diff').write_text(''.join(difflib.unified_diff(parent.splitlines(True),s.splitlines(True),fromfile='17d44d',tofile='xprefetch_choice')))
 proof={'state':'CPU_EXACT_SOURCE_INVERSE_PASS','source_sha256':sha(s),'parent_sha256':sha(parent),'whole_source_inverse_to17d44d_exact':True,'new_native_body_sha256':sha(body),'new_native_body_exact_measured_pilot':True,'M1_keyed_and_M8_earlyload_bodies_helpers_dispatch_unchanged':True,'original_unmarked_bakes_unchanged':True,'unpooler_M16_options':['parent','xprefetch'],'global_prepare_budget_seconds':240,'new_numeric_and_perf_validation':'optional same actual weight four AB/BApairs, immediate own-output/currentX check, zero threshold; incomplete/deadline/loss retains parent','scope':'CPU-only production construction; not fresh GPU/official validation'}
 (D/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(sha(s))
if __name__=='__main__':build()
