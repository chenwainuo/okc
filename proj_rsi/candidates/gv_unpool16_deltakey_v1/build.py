import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2];P=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py';PS='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
def cu(s):return next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
helper='''__device__ __forceinline__ void gv_fix_deltakey(uint32_t (&a)[16],uint32_t key,uint32_t xr){
  asm volatile("{ .reg .b32 k, mask; .reg .pred pr; mov.b32 k, %16; mov.b32 mask, %17; '''+' '.join('setp.eq.u32 pr, k, %d; @pr xor.b32 %%%d, %%%d, mask;'%(4*i,i,i)for i in range(16))+''' }" : '''+', '.join('"+r"(a[%d])'%i for i in range(16))+''' : "r"(key),"r"(xr));
}
'''
def build():
 s=P.read_text();assert hashlib.sha256(s.encode()).hexdigest()==PS;old=cu(s);prefix='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__'
 a=old.index(prefix,old.index('void gv_fix_pred'));b=old.index('\n__global__ void p1zero',a);kernel=old[a:b];new=kernel.replace(' gv_predxor(G p) {',' gv_unpool16_deltakey(G p) {',1)
 sa=new.index('  static_assert(');sb=new.index('  // pw14 probes only',sa)
 scope='  static_assert(FMT==1 && MT==2 && ST==2 && KU==16 && S==1 && RT==1 && V==1 && WR==1 && WK==4 && XS==2, "only officialunpoolerM16targettemplate");\n'
 new=new[:sa]+scope+new[sb:]
 before='''          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);'''
 after='''          const uint32_t key = ((nx >> 9) & 0x7fdu) - ((uint32_t)lane << 6), xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_deltakey(A[j][0],key,xr);'''
 assert new.count(before)==1;new=new.replace(before,after,1)
 newcu=old[:b]+'\n'+helper+'\n'+new+old[b:]
 route='''    if constexpr (FMT==1 && MT==2 && ST==2 && KU==16 && S==1 && RT==1 && V==1 && WR==1 && WK==4 && XS==2) {
      if (unpooler) k=gv_unpool16_deltakey<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
    }
'''
 anchor='    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;\n';assert newcu.count(anchor)==1;newcu=newcu.replace(anchor,anchor+route,1)
 # Keep parent's entire source text, adding only nativebody/helper/route and distinct JIT prefix.
 oldliteral='GVCU = r"""'+old+'"""';newliteral='GVCU = r"""'+newcu+'"""';assert s.count(oldliteral)==1;source=s.replace(oldliteral,newliteral,1)
 marker='load_inline("gvp45_predxor_v2_%x"';assert marker in source;source=source.replace(marker,'load_inline("gv_unpool16_deltakey_v1_%x"',1)
 (D/'submission.py').write_text(source);(D/'helper.cuh').write_text(helper)
 pilot=old[:b]+'\n'+helper+'\n'+new
 pilot=pilot.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
 for geo in ('1,2,2,1,4,16,1,1,2,1',):
  for fn in ('gv_predxor','gv_unpool16_deltakey'):pilot+='\ntemplate __global__ void '+fn+'<'+geo+'>(G);\n'
 (D/'pilot.cu').write_text(pilot)
 (D/'delta.json').write_text(json.dumps({'parent_path':str(P.relative_to(R)),'parent_sha256':PS,'old_GVCU_literal':oldliteral,'new_GVCU_literal':newliteral,'old_prefix':marker,'new_prefix':'load_inline("gv_unpool16_deltakey_v1_%x"','old_kernel':kernel,'new_kernel':new,'old_repair':before,'new_repair':after,'new_route':route},indent=2)+'\n')
 print(hashlib.sha256(source.encode()).hexdigest())
if __name__=='__main__':build()
