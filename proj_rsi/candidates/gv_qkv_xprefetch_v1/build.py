import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2];P=D.parent/'gv_predxor_pw45_v2/submission.py'
def build():
 s=P.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 cu=next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
 prefix='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__';a=cu.index(prefix);b=cu.index('__device__ __forceinline__ void gv_fix_pred',a);old=cu[a:b];new=old.replace(' gv(G p) {',' gv_qkv_xprefetch(G p) {',1)
 assert new!=old
 anchor='  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];'
 prologue='''  static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==2 && V==1,"onlyofficialQKVsmalltarget");
  uint4 xnext[MT][2];
#pragma unroll
  for (int n=0;n<MT;n++) { xnext[n][0]=gxp<(V >> 15) & 3>(xp[n]); xnext[n][1]=gxp<(V >> 15) & 3>(xp[n]+8); }
'''
 assert new.count(anchor)==1;new=new.replace(anchor,prologue+anchor,1)
 before='''      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }'''
 after='''      for (int n = 0; n < MT; n++) { xd[n][0]=xnext[n][0]; xd[n][1]=xnext[n][1]; }
      if(u+1<KU) {
#pragma unroll
        for(int n=0;n<MT;n++) { xnext[n][0]=gxp<(V >> 15) & 3>(xp[n]+(u+1)*64); xnext[n][1]=gxp<(V >> 15) & 3>(xp[n]+(u+1)*64+8); }
      }'''
 assert new.count(before)==1;xi=new.rfind(before);new=new[:xi]+new[xi:].replace(before,after,1)
 newcu=cu[:b]+new+cu[b:]
 route='''  if constexpr(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==2 && V==1) {
    if(p.N==2304 && p.K==8192 && (p.m==1 || p.m==8)) k=gv_qkv_xprefetch<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
  }
'''
 anchorroute='  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;';assert newcu.count(anchorroute)==1;newcu=newcu.replace(anchorroute,route+anchorroute,1)
 oldlit='GVCU = r"""'+cu+'"""';newlit='GVCU = r"""'+newcu+'"""';assert s.count(oldlit)==1
 out=s.replace(oldlit,newlit,1).replace('load_inline("gvp45_predxor_v2_%x"','load_inline("gv_qkv_xprefetch_v1_%x"',1);(D/'submission.py').write_text(out)
 # Onlyoriginalgenericgvprefix plusnewbody; no unrelatedtargetinstantiations.
 pilot=(cu[:a]+old+new).replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
 for name in['gv','gv_qkv_xprefetch']:pilot+='\ntemplate __global__ void '+name+'<1,1,1,2,16,8,1,1,2,1>(G);\n'
 (D/'pilot.cu').write_text(pilot);(D/'delta.json').write_text(json.dumps({'old_literal':oldlit,'new_literal':newlit,'old_kernel':old,'new_kernel':new,'prologue':prologue,'old_X_issue':before,'new_X_issue':after,'new_route':route},indent=2)+'\n')
if __name__=='__main__':build()
