from pathlib import Path
import hashlib,json,re,difflib
P=Path(__file__).resolve().parent;B=P.parent/'wgmma-pipeline-v4';parent=(B/'routed_down.cuh').read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='50120fdc47b2026be404447aa3bba0aab28bff331e6613a919e080f4399b8e24'
ks=parent.index('__global__ void routed_nvfp4_down');ke=parent.index('\n\n\nstruct RoutedDownKernel',ks)
original_kernel=parent[ks:ke];t16=original_kernel.replace('int position=int(blockIdx.y)*8;','int position=int(blockIdx.y)*16;').replace('Bf16 sb[4][128]','Bf16 sb[4][256]').replace('float c[4]={0,0,0,0};','float c[8]={0,0,0,0,0,0,0,0};')
old='''      int route=sorted[position+tid/16];
      auto b=cute::make_tensor(cute::make_smem_ptr(sb[step]), BMap{});
      b(tid/16,tid%16)=(route>=0 && route<rows)
          ?activation[int64_t(route)*K+base+step*16+tid%16]:Bf16(0.0f);'''
new='''      auto b=cute::make_tensor(cute::make_smem_ptr(sb[step]), BMap{});
      #pragma unroll
      for(int token_half=0;token_half<2;++token_half) {
        int token=tid/16+token_half*8;
        int route=sorted[position+token];
        b(token,tid%16)=(route>=0 && route<rows)
            ?activation[int64_t(route)*K+base+step*16+tid%16]:Bf16(0.0f);
      }'''
assert old in t16;t16=t16.replace(old,new)
a=t16.index('    asm volatile(\n      "{');b=t16.index('      : "memory");',a)
asm=t16[a:b]
# Shift only input operand positions: eight accumulators precede same22inputs.
asm=re.sub(r'%(\d+)\b',lambda m:'%'+str(int(m[1])+4) if int(m[1])>=4 else m[0],asm)
asm=asm.replace('m64n8k16','m64n16k16').replace('{%0, %1, %2, %3},','{%0, %1, %2, %3, %4, %5, %6, %7},')
asm=asm.replace(': "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])',': "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),\n        "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])')
t16=t16[:a]+asm+t16[b:]
t16=t16.replace('for(int v=0;v<4;++v) {\n    int linear=CMap{}','for(int v=0;v<8;++v) {\n    int linear=CMap{}')
traits='''namespace token16 {
using Op = cute::SM90::GMMA::MMA_64x16x16_F32BF16BF16_RS<
    cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>;
using Traits = cute::MMA_Traits<Op>;
using AMap = Traits::ALayout;
using CMap = Traits::CLayout;
using BMap = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_SW32_Atom<Bf16>{}, cute::Shape<cute::_16,cute::_16>{}));
static_assert(cute::size(AMap{})==64*16);
static_assert(cute::size(CMap{})==64*16);
static_assert(cute::cosize(BMap{})==16*16);
static_assert(sizeof(Op::CRegisters)==8*sizeof(float));
// Verify actual pinned CUTLASS mappings; no hand-derived CMap or descriptor.
constexpr bool map_contracts_valid() {
  bool outputs[64*16]={};
  for(int tid=0;tid<128;++tid) {
    for(int v=0;v<8;++v) {
      if(AMap{}(cute::make_coord(tid,v))!=::AMap{}(cute::make_coord(tid,v)))return false;
      int linear=CMap{}(cute::make_coord(tid,v));
      if(linear<0 || linear>=64*16 || outputs[linear])return false;
      outputs[linear]=true;
    }
  }
  bool shared[16*16]={};
  for(int token=0;token<16;++token)for(int k=0;k<16;++k) {
    int offset=BMap{}(cute::make_coord(token,k));
    if(offset<0 || offset>=16*16 || shared[offset])return false;
    shared[offset]=true;
  }
  return true;
}
static_assert(map_contracts_valid(),"Pinned CUTLASS token16 maps/bounds invalid");
static_assert((16*16*sizeof(Bf16))%128==0,"Each shared slice descriptor must remain aligned");
'''
candidate=parent[:ke]+'\n\n'+traits+t16+'\n} // namespace token16\n'+parent[ke:]
candidate=candidate.replace('    LaunchKernel(dim3(n/64,S.unwrap()/8),128,device.unwrap())(\n      routed_nvfp4_down,', '''    // Each16-token tile stays wholly inside one existing expert block.
    // T8 kernel body remains byte-identical for block_m8 or other nonmultiples16.
    const bool use_token16=(block_m%16)==0;
    const int token_tile=use_token16?16:8;
    auto kernel=use_token16?token16::routed_nvfp4_down:routed_nvfp4_down;
    LaunchKernel(dim3(n/64,S.unwrap()/token_tile),128,device.unwrap())(
      kernel,''')
assert candidate[ks:ke]==original_kernel
(P/'routed_down.cuh').write_text(candidate);(P/'parent-routed_down.cuh').write_text(parent)
(P/'candidate.diff').write_text(''.join(difflib.unified_diff(parent.splitlines(True),candidate.splitlines(True),fromfile='v4/routed_down.cuh',tofile='v5/routed_down.cuh')))
for name in ['routed_adapter.py','serving_wgmma_down.py','check_routed.py','check_full_expert.py','run_routed_probe.py','check_pack_cpu.py']:
 data=(B/name).read_bytes()
 if name=='routed_adapter.py':data=data.replace(b'rsi_wgmma_k64group_down',b'rsi_wgmma_token16_down')
 (P/name).write_bytes(data)
# CPU metadata ownership and activation-store coverage independent of GPU.
cases=0
for block_m in [8,16,24,32,40,48,56,64]:
 tile=16 if block_m%16==0 else 8
 for blocks in [1,2,17,256]:
  post=block_m*blocks
  for tail in [0,4,8,12,60]:
   capacity=post+tail
   seen=[]
   for y in range(capacity//tile):
    pos=y*tile
    if pos>=post:continue
    slots=list(range(pos,pos+tile));assert all(x//block_m==pos//block_m for x in slots)
    assert slots[-1]<post<=capacity
    seen.extend(slots)
   assert seen==list(range(post));cases+=1
stores=[(tid//16+half*8,tid%16) for tid in range(128) for half in range(2)]
assert len(set(stores))==256 and set(stores)=={(t,k) for t in range(16) for k in range(16)}
report={'status':'local_source_and_index_checks_pass_CUDA_compile_pending','parent_sha256':hashlib.sha256(parent.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'T8_kernel_byte_identical':True,'metadata_ownership_cases':cases,'activation_coordinates_bijection':True,'CUTLASS_map_checks':'constexpr AMap equality+CMap bijection+BMap bounds written; evaluated only upon pinned CUDA compilation','source_counts':'256GMMAops/64commitwaitgroups perCTA; T16 handles twice tokens; no pruning/layout/grouping change','GPU_correctness':False,'performance':'unknown'}
(P/'status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
