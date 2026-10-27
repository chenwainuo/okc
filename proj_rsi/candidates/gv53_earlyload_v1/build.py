import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2];P=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
def build():
 s=P.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 cu=next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
 a=cu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__',cu.index('void gv_fix_pred'));b=cu.index('\n__global__ void p1zero',a);old=cu[a:b]
 new=old.replace(' gv_predxor(G p) {',' gv_earlyload(G p) {',1)
 sa=new.index('  static_assert(');sb=new.index('  // pw14 probes only',sa)
 new=new[:sa]+'  static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"onlyactualGV53M8");\n'+new[sb:]
 # Stage exactly the original source raw operands before overwriting ST1 ring.
 anchor='    uint32_t A[RT][NA][16], B[MT][8];'
 stage='''    uint4 early_w[RT][NA][NL], early_x[MT][2];
#pragma unroll
    for (int j=0;j<RT;j++)
#pragma unroll
      for (int h=0;h<NA;h++)
#pragma unroll
        for (int i=0;i<NL;i++) early_w[j][h][i]=wb[s][j][h][i];
#pragma unroll
    for (int n=0;n<MT;n++) { early_x[n][0]=xb[s%XB][n][0]; early_x[n][1]=xb[s%XB][n][1]; }
    if (u+ST<KU) ld(s,u+ST);
'''
 assert new.count(anchor)==1;new=new.replace(anchor,stage+anchor,1)
 oldq='for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];';assert new.count(oldq)==1;new=new.replace(oldq,'for (int i = 0; i < NL; i++) qv[i] = early_w[j][h][i];',1)
 oldx='else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }';assert new.count(oldx)==2;xi=new.rfind(oldx);new=new[:xi]+new[xi:].replace(oldx,'else { b0 = early_x[n][0]; b1 = early_x[n][1]; }',1)
 # Only ordinary production path's refill, retain dead constexpr alternative text unchanged.
 target='    if (u + ST < KU) ld(s, u + ST);\n    if ((C12 || CE) && LDEC) {'
 assert new.count(target)==1;new=new.replace(target,'    if ((C12 || CE) && LDEC) {',1)
 route='''    if constexpr(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3) {
      if (pooler && p.m==8) k=gv_earlyload<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
    }
'''
 anchorroute='    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;\n';assert cu.count(anchorroute)==1
 newcu=(cu[:b]+'\n'+new+cu[b:]).replace(anchorroute,anchorroute+route,1)
 oldlit='GVCU = r"""'+cu+'"""';newlit='GVCU = r"""'+newcu+'"""';assert s.count(oldlit)==1
 out=s.replace(oldlit,newlit,1).replace('load_inline("gvp45_predxor_v2_%x"','load_inline("gv53_earlyload_v1_%x"',1);(D/'submission.py').write_text(out)
 pilot=(cu[:b]+'\n'+new).replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
 for fn in ['gv_predxor','gv_earlyload']:pilot+='\ntemplate __global__ void '+fn+'<1,1,1,2,16,8,1,1,0,3>(G);\n'
 (D/'pilot.cu').write_text(pilot);bridge=(R/'proj_rsi/candidates/gv53_thin_cabi_validation_v1/bridge.cuh').read_text().replace('gv_keyed<','gv_earlyload<').replace('(m!=1&&m!=8)','m!=8');(D/'bridge.cuh').write_text(bridge);(D/'thin.cu').write_text(pilot+'\n'+bridge)
 (D/'delta.json').write_text(json.dumps({'parent_sha256':hashlib.sha256(s.encode()).hexdigest(),'old_kernel':old,'new_kernel':new,'old_literal':oldlit,'new_literal':newlit,'route':route,'old_q':oldq,'old_x':oldx,'stage':stage,'removed_refill':target,'scope':'actualGV53poolerM8only'},indent=2)+'\n')
if __name__=='__main__':build()
