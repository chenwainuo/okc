"""CPU generator: preserve C12/v2 kernels; add two metadata-preloaded kernels."""
import hashlib
import json
from pathlib import Path

D=Path(__file__).resolve().parent
V2=D.with_name('native_v2_capacity32_48')
HIGH=D.with_name('native_v3_highpatch')
base=(V2/'native.cu').read_text()
old=(HIGH/'native.cu').read_text()
prefix='template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__'
a=old.index(prefix,old.index(' gv_predxor(G p)'))
b=old.index('\ntemplate __global__ void gv_predxor',a)
kernel=old[a:b].strip()
kernel=kernel.replace(' gv_frontier(G p, Frontier fp)', ' gv_frontier_preload(G p, Frontier fp)',1)
kernel=kernel.replace('frontier_dec(', 'frontier_dec_preloaded(')
changes=[]
def replace(old,new):
    global kernel
    assert kernel.count(old)==1,(old[:90],kernel.count(old))
    kernel=kernel.replace(old,new,1)
    changes.append(dict(old=old,new=new))

start=kernel.index('  constexpr int FCAP = ')
end=kernel.index('  auto ld = ',start)
oldprologue=kernel[start:end]
newprologue=r'''  constexpr int FCAP=MT==1?32:48, RBYTES=FCAP*2;
  constexpr int DBYTES=KU*32+16, WBYTES=DBYTES+ST*RBYTES;
  const uint32_t fs=(uint32_t)__cvta_generic_to_shared(smem)+L::SMEM;
  const uint32_t wd=fs+w*WBYTES;
  uint32_t* const warp_desc=(uint32_t*)(smem+L::SMEM+w*WBYTES);
  static_assert(RT==1&&NA==1&&!L::BK&&S==1&&(ST==1||ST==2)&&KU<=16,"exact preloaded descriptor scope");
  const unsigned first_unit=(bx*WR+wr)*p.KC+c0;
  // Every warp owns its KU*32B slab. No cross-warp publication or reuse.
  if(lane<KU*2) {
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(wd+lane*16),"l"(p.esc+(size_t)first_unit*8+lane*4));
  }
  // Only the next unit's first offset is needed; last matrix unit uses words.
  if(lane==0) warp_desc[KU*8]=first_unit+KU<fp.units?__ldg(p.esc+(size_t)(first_unit+KU)*8):fp.words;
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  __syncwarp(); // both async descriptors and lane0's boundary are now visible
'''
replace(oldprologue,newprologue)
start=kernel.index('    const uint32_t begin = ')
end=kernel.index('    asm volatile("cp.async.commit_group;',start)
oldissue=kernel[start:end]
newissue=r'''    const uint32_t begin=warp_desc[u*8],finish=warp_desc[(u+1)*8];
    const uint32_t dest=wd+DBYTES+s*RBYTES;
    if(lane<FCAP/8&&lane*16<(finish-begin)*2) {
      asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(dest+lane*16),"l"(fp.records+(size_t)begin*2+lane*16));
    }
'''
replace(oldissue,newissue)
replace('    const uint32_t* desc=(const uint32_t*)(smem+L::SMEM+(w*ST+s)*FSLOT);\n    const unsigned short* fixes=(const unsigned short*)(smem+L::SMEM+(w*ST+s)*FSLOT+32);',
        '    const uint32_t* desc=warp_desc+u*8;\n    const unsigned short* fixes=(const unsigned short*)(smem+L::SMEM+w*WBYTES+DBYTES+s*RBYTES);')
a=old.index('__device__ __forceinline__ void frontier_dec(')
b=old.index('\n}',a)+2
decode=old[a:b].replace('frontier_dec(', 'frontier_dec_preloaded(')
header=(HIGH/'high_patch.cuh').read_text()
text=base+'\n'+header+'\n'+decode+'\n'+kernel+'\n'
for tpl in ('1,1,1,2,16,8,1,1,0,3','1,2,2,1,4,16,1,1,2,1'):
    text+='template __global__ void gv_frontier_preload<'+tpl+'>(G,Frontier);\n'
(D/'native.cu').write_text(text)
(D/'delta.json').write_text(json.dumps(dict(parent_V2_sha256=hashlib.sha256((V2/'native.cu').read_bytes()).hexdigest(),
    highpatch_sha256=hashlib.sha256((HIGH/'native.cu').read_bytes()).hexdigest(),changes=changes,
    descriptor_slab_bytes_per_warp={'pooler':272,'unpooler':528},
    record_slots_bytes_per_warp={'pooler':64,'unpooler':192},
    extra_dynamic_bytes={'pooler':10752,'unpooler':2880},
    total_dynamic_bytes={'pooler':28160,'unpooler':8000},
    additional_boundary_source_bytes={'pooler':16380,'unpooler':8188}),indent=2)+'\n')
print('Generated six functions: two C12, two exact V2, two preloaded/highpatch.')
