"""One exact repair-field key change atop frozen official GV501d; CPU only."""
import ast,hashlib,json,re
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
P=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
PARENT_SHA='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
TPL=(1,1,1,2,16,8,1,1,0,3)
sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
def value(s):return next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
def parts(cu):
 a=cu.index('// Predicated format1 repair;');b=cu.index('\ntemplate <int FMT',a)
 helper=cu[a:b]
 start=cu.index('template <int FMT',b);end=cu.index('\n__global__ void p1zero',start)
 kernel=cu[start:end]
 newhelp=helper.replace('gv_fix_pred(', 'gv_fix_keyed(').replace(',uint32_t hm)',')')
 newhelp=newhelp.replace('.pred pl, pa, pr;', '.pred pl, pr;').replace(' setp.eq.and.u32 pa, %20, 0, pl;','')
 newhelp=re.sub(r'pr, ri, (\d+), pa;',lambda m:'pr, ri, '+str(4*int(m[1]))+', pl;',newhelp)
 newhelp=newhelp.replace(',"r"(hm)','')
 assert '%20'not in newhelp and ' pa'not in newhelp
 old='''          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);'''
 new='''          const uint32_t key = (nx >> 9) & 0x3du, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_keyed(A[j][0],key,xr,lane,(nx>>15)&31);'''
 assert kernel.count(old)==1
 newkernel=kernel.replace('gv_predxor(G p)', 'gv_keyed(G p)',1).replace(old,new,1)
 anchor='  // pw14 probes only'
 assertion='  static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"keyed repair only current rolled pooler");\n'
 newkernel=newkernel.replace(anchor,assertion+anchor,1)
 dispatch='''
  if constexpr(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3) {
    if(p.N==4096 && p.K==8192 && (p.m==1 || p.m==8)) k=gv_keyed<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
  }
'''
 at='  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;'
 assert cu.count(at)==1
 return helper,kernel,newhelp,newkernel,dispatch,at

def make():
 old=P.read_text();assert sha(old)==PARENT_SHA;cu=value(old)
 helper,kernel,newhelp,newkernel,dispatch,at=parts(cu)
 added='\n'+newhelp+'\n'+newkernel+'\n'
 out=cu.replace('\n__global__ void p1zero',added+'\n__global__ void p1zero',1).replace(at,dispatch+at,1)
 # Preserve all original literals outside exact GVCU node and JIT cache identity.
 n=next(n for n in ast.parse(old).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
 lines=old.splitlines(keepends=True);prefix=''.join(lines[:n.lineno-1]);suffix=''.join(lines[n.end_lineno:]);literal='GVCU = r"""'+out+'"""\n'
 result=prefix+literal+suffix
 assert value(result)==out
 result=result.replace('load_inline("gvp45_predxor_v2_%x"','load_inline("gv53_keyed_half_v1_%x"',1)
 restored=result.replace('load_inline("gv53_keyed_half_v1_%x"','load_inline("gvp45_predxor_v2_%x"',1)
 rc=value(restored).replace(added,'',1).replace(dispatch,'',1);assert rc==cu
 # Literal source spelling remains exact parent since its literal wrapper is retained.
 assert restored.replace(out,cu,1)==old
 (D/'submission.py').write_text(result);(D/'helper.cuh').write_text(newhelp+'\n')
 # Compile only the old/new exact rolled functions, not gvl or other inventory.
 stop=cu.index('// FP32 router, small m:')
 pilot=(cu[:stop].replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>',''))
 pilot=pilot.replace('\n__global__ void p1zero',added+'\n__global__ void p1zero',1)
 for fn in ['gv_predxor','gv_keyed']:pilot+='\ntemplate __global__ void '+fn+'<'+','.join(map(str,TPL))+'>(G);\n'
 (D/'pilot.cu').write_text(pilot)
 (D/'delta.json').write_text(json.dumps({'parent_sha256':PARENT_SHA,'source_sha256':sha(result),'scope':'Only current rolled pooler M1/M8, same gvl geometries/rejections; unchanged routes for all other actual shapes','helper_before':helper,'helper_after':newhelp,'added_kernel_sha256':sha(newkernel),'dispatch':dispatch,'semantic_identity':'key=(nx>>9)&0x3d=4*r+hm; comparekey4*i iff r==i and hm==0; bit10 within-register half selector excluded'},indent=2)+'\n')
 return {'state':'CPU_SOURCE_READY_GPU_UNRUN','parent_sha256':PARENT_SHA,'source_sha256':sha(result),'helper_sha256':sha(newhelp+'\n'),'strict_inverse':True,'changed_original_AST_nodes':['GVCU','_gvext'],'other_original_AST_nodes_unchanged':174,'new_kernel_math_load_reduction_output_exact':True,'ignored_hm1_preserved':True,'template':TPL,'GPU_calls':0}
if __name__=='__main__':
 p=make();(D/'source-proof.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
