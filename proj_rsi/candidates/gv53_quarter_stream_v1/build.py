"""Fixed M8 native prototype, exact parent source; no GPU or production integration."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent/'gv_predxor_pw45_v2/submission.py'
def make():
 s=P.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 cu=next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
 a=cu.index('template <int FMT',cu.index('// Predicated format1 repair;'));b=cu.index('\n__global__ void p1zero',a)
 parent=cu[a:b];new=parent.replace(' gv_predxor(G p)',' gv_quarter(G p)',1)
 # Generic prologue/reduction retained; exact compile-time shape narrows the new function.
 pos=new.index('  // pw14 probes only')
 new=new[:pos]+'  static_assert(FMT==1 && MT==1 && ST==1 && WR==2 && WK==16 && KU==8 && S==1 && RT==1 && XS==0 && V==3,"quarter prototype exact tuple");\n  if(p.m!=8 || p.N!=4096 || p.K!=8192) return;\n'+new[pos:]
 oldcursor='  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }'
 assert new.count(oldcursor)==1
 new=new.replace(oldcursor,'  e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane];',1)
 start=new.index('  auto step =');end=new.index('  if (V & 2)',start)
 oldstep=new[start:end];newstep=(D/'step.cuh').read_text();new=new[:start]+newstep+new[end:]
 helper='''__device__ __forceinline__ void gv_fix_quarter(uint32_t (&a)[4],uint32_t r,uint32_t xr,int lane,uint32_t target,uint32_t hm){
  asm volatile("{ .reg .b32 ri, mask; .reg .pred pl, pa, pr; mov.b32 ri, %4; mov.b32 mask, %5; setp.eq.u32 pl, %6, %7; setp.eq.and.u32 pa, %8, 0, pl; setp.eq.and.u32 pr, ri, 0, pa; @pr xor.b32 %0, %0, mask; setp.eq.and.u32 pr, ri, 1, pa; @pr xor.b32 %1, %1, mask; setp.eq.and.u32 pr, ri, 2, pa; @pr xor.b32 %2, %2, mask; setp.eq.and.u32 pr, ri, 3, pa; @pr xor.b32 %3, %3, mask; }" : "+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]) : "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm));
}
'''
 pilot=cu[:b]+'\n'+helper+'\n'+new
 pilot=pilot.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
 for name in ['gv_predxor','gv_quarter']:pilot+='\ntemplate __global__ void '+name+'<1,1,1,2,16,8,1,1,0,3>(G);\n'
 (D/'pilot.cu').write_text(pilot);(D/'helper.cuh').write_text(helper);(D/'kernel.cuh').write_text(new)
 (D/'derivation.json').write_text(json.dumps({'parent_path':str(P),'parent_sha256':hashlib.sha256(s.encode()).hexdigest(),'control_exact_parent_kernel':True,'new_kernel':{'clone_name':'gv_quarter','only_changes':['exact_tuple_and_M8_shape_guard','quarter_packet_load_no_old_cursor_shuffle','step_replaced_with_four_ordered_decode4_repair4_HMMA_phases']},'old_step':oldstep,'new_step':newstep,'packet_bytes_per_warpgroup':128,'new_kernel_sha256':hashlib.sha256(new.encode()).hexdigest()},indent=2)+'\n')
 print(hashlib.sha256(pilot.encode()).hexdigest())
if __name__=='__main__':make()
