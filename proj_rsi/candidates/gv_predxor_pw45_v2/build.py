"""Isolated PW45 GV predicated escape discriminator, CPU source construction only."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2];PARENT=ROOT/'proj_rsi/candidates/pw45/submission.py'
SHA='835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3'
TARGETS=[(1,1,1,2,16,8,1,1,0,3),(1,1,2,2,8,16,1,1,0,0),(1,1,2,2,8,16,1,1,0,1),(1,2,2,2,8,16,1,1,0,0),(1,2,2,2,8,16,1,1,0,1),(1,2,2,1,4,16,1,1,2,1),(1,1,2,1,8,16,1,1,0,1)]
PARAMS=('FMT','MT','ST','WR','WK','KU','S','RT','XS','V')
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def value(s,key):return next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==key for t in n.targets))
def build():
 old=PARENT.read_text();assert sha(old)==SHA;cu=value(old,'GVCU')
 cond=' ||\n      '.join('('+' && '.join(f'{k} == {v}'for k,v in zip(PARAMS,t))+')'for t in TARGETS)
 kernel_start=cu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__')
 kernel_end=cu.index('\n__global__ void p1zero',kernel_start);kernel=cu[kernel_start:kernel_end]
 helper='''// Predicated format1 repair; source inputs are copied before any output write.
__device__ __forceinline__ void gv_fix_pred(uint32_t (&a)[16],uint32_t r,uint32_t xr,int lane,uint32_t target,uint32_t hm){
  asm volatile("{ .reg .b32 ri, mask; .reg .pred pl, pa, pr; mov.b32 ri, %16; mov.b32 mask, %17; setp.eq.u32 pl, %18, %19; setp.eq.and.u32 pa, %20, 0, pl; '''
 helper+=' '.join(f'setp.eq.and.u32 pr, ri, {i}, pa; @pr xor.b32 %{i}, %{i}, mask;'for i in range(16))
 helper+=''' }" : '''+', '.join(f'"+r"(a[{i}])'for i in range(16))+''' : "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm));
}
'''
 before='''          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }'''
 after='''          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);'''
 assert kernel.count(before)==1
 newkernel=kernel.replace(' gv(G p) {',' gv_predxor(G p) {',1).replace(before,after)
 newkernel=newkernel.replace('  // pw14 probes only','  static_assert('+cond+', "GVpredxor only seven reviewed format1 templates");\n  // pw14 probes only',1)
 insert=helper+'\n'+newkernel+'\n'
 # Append the isolated kernel without modifying the original gv definition.
 newcu=cu[:kernel_end]+'\n'+insert+cu[kernel_end:]
 anchor='  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;'
 dispatch='''
  if constexpr ('''+cond+''') {
    const bool pooler = p.N == 4096 && p.K == 8192 && ((MT == 1 && (p.m == 1 || p.m == 8)) || (MT == 2 && p.m == 16));
    const bool unpooler = MT == 2 && p.N == 8192 && p.K == 4096 && p.m == 16;
    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }'''
 assert newcu.count(anchor)==1;newcu=newcu.replace(anchor,anchor+dispatch,1)
 # Replace the exact raw string payload while leaving all other Python bytes untouched.
 node=next(n for n in ast.parse(old).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets));literal=ast.get_source_segment(old,node.value);assert literal.startswith('r"""')and literal.endswith('"""')
 newliteral='r"""'+newcu+'"""';assert old.count(literal)==1
 new=old.replace(literal,newliteral,1)
 prefix='load_inline("gv20_%x"';assert new.count(prefix)==1;new=new.replace(prefix,'load_inline("gvp45_predxor_v2_%x"',1)
 assert value(new,'GVCU')==newcu
 (D/'submission.py').write_text(new);(D/'helper.cuh').write_text(helper)
 (D/'delta.json').write_text(json.dumps({'parent_source_sha256':SHA,'target_instantiations':TARGETS,'typed_edits':[{'id':'append-standalone-GV-copy','region':'GVCU.new-kernel','original_gv_source_sha256':sha(kernel),'new_gv_source_sha256':sha(newkernel),'old_escape':before,'new_escape':after},{'id':'new-only-shape-dispatch','region':'GVCU.gvl-pointer','anchor':anchor,'addition':dispatch},{'id':'unique-JIT-module-prefix','region':'host._gvext','before':prefix,'after':'load_inline("gvp45_predxor_v2_%x"'}],'strict_whole_file_inverse':{'before_sha256':sha(old),'after_sha256':sha(new),'original_GVCU_literal':literal,'candidate_GVCU_literal':newliteral}},indent=2)+'\n')
 return old,new
if __name__=='__main__':
 a,b=build();print(sha(b))
