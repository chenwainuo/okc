from pathlib import Path
import hashlib,json,difflib
P=Path(__file__).resolve().parent
B=P.parent/'wgmma-coalesced-v3'
s=(B/'routed_down.cuh').read_text()
start=s.index('  __shared__ __align__(128) Bf16 sb[128];')
end=s.index('  #pragma unroll\n  for(int v=0;v<4;++v)',start)
old=s[start:end]
new='''  // K64 grouping only: four independent K16 shared/register operands survive
  // until their common wait. Grid, packed layout and arithmetic remain v3.
  __shared__ __align__(128) Bf16 sb[4][128];
  uint64_t desc[4];
  #pragma unroll
  for(int step=0;step<4;++step) {
    auto b=cute::make_tensor(cute::make_smem_ptr(sb[step]), BMap{});
    desc[step]=cute::SM90::GMMA::make_gmma_desc<cute::SM90::GMMA::Major::K>(b);
  }
  int tid=threadIdx.x;
  float c[4]={0,0,0,0};
  for(int base=0;base<K;base+=64) {
    #pragma unroll
    for(int step=0;step<4;++step) {
      int route=sorted[position+tid/16];
      auto b=cute::make_tensor(cute::make_smem_ptr(sb[step]), BMap{});
      b(tid/16,tid%16)=(route>=0 && route<rows)
          ?activation[int64_t(route)*K+base+step*16+tid%16]:Bf16(0.0f);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    alignas(16) nv_bfloat162 a[4][4];
    #pragma unroll
    for(int step=0;step<4;++step) {
      uint32_t payload=packed[(base/16+step)*128+tid];
      int row0=((tid/4)%8)+16*(tid/32);
      uint32_t scale0=scales[(base/16+step)*64+row0];
      uint32_t scale1=scales[(base/16+step)*64+row0+8];
      uint32_t sc=scale0 | scale0<<16 | scale1<<8 | scale1<<24;
      nv_bfloat162 fs[2];
      marlin::dequant_fp8_scales<nv_bfloat162,sglang::host::kFE4M3fn.id()>(sc,fs);
      #pragma unroll
      for(int h=0;h<2;++h) {
        uint32_t halfword=payload>>(h*16);
        uint32_t w=((halfword&0xff)<<8)|((halfword&0xff00)<<16);
        marlin::dequant<nv_bfloat162,sglang::host::kFE2M1f.id(),true>(w,a[step]+h*2);
        a[step][h*2]=__hmul2(a[step][h*2],fs[0]);
        a[step][h*2+1]=__hmul2(a[step][h*2+1],fs[1]);
      }
    }
    auto* ar=reinterpret_cast<uint32_t*>(a);
    // Keep complete asynchronous accumulator lifetime in one asm region.
    // Same-shape WGMMA accumulators have instruction ordering by PTX contract;
    // all register-A fragments and shared-B slices remain untouched until wait.
    asm volatile(
      "{\\n.reg .pred p, t;\\n"
      "setp.ne.b32 p, %[accumulate], 0;\\n"
      "setp.eq.b32 t, 0, 0;\\n"
      "wgmma.fence.sync.aligned;\\n"
'''
for step in range(4):
    names=', '.join('%[a'+str(step*4+x)+']' for x in range(4))
    new+=f'      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {{%0, %1, %2, %3}}, {{{names}}}, %[desc{step}], '+('p' if step==0 else 't')+', 1, 1, %[major];\\n"\n'
new+='''      "wgmma.commit_group.sync.aligned;\\n"
      "wgmma.wait_group.sync.aligned 0;\\n}\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : '''
new+=',\n        '.join(f'[a{i}] "r"(ar[{i}])' for i in range(16))
new+=',\n        '+', '.join(f'[desc{i}] "l"(desc[{i}])' for i in range(4))
new+=''',
        [accumulate] "r"(int(base!=0)), [major] "n"(int(cute::GMMA::Major::K))
      : "memory");
    __syncthreads(); // All four shared-B slices may now be reused.
  }
'''
# NVCC inline PTX accepts positional operands, not symbolic GCC names.
for index,name in enumerate([f'a{i}' for i in range(16)]+[f'desc{i}' for i in range(4)]+['accumulate','major'],start=4):
    new=new.replace('%['+name+']','%'+str(index)).replace('['+name+'] ','')
candidate=s[:start]+new+s[end:]
(P/'routed_down.cuh').write_text(candidate)
(P/'parent-routed_down.cuh').write_text(s)
(P/'candidate.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile='v3/routed_down.cuh',tofile='v4/routed_down.cuh')))
for name in ['routed_adapter.py','serving_wgmma_down.py','check_routed.py','check_full_expert.py','run_routed_probe.py','check_pack_cpu.py']:
 data=(B/name).read_bytes()
 if name=='routed_adapter.py':data=data.replace(b'rsi_wgmma_coalesced_down',b'rsi_wgmma_k64group_down')
 (P/name).write_bytes(data)
# Structural identities: all preloop dispatch and entire epilogue/host wrapper unchanged.
assert candidate[:start]==s[:start]
assert candidate[candidate.index('  #pragma unroll\n  for(int v=0;v<4;++v)'):]==s[end:]
indices=[base+step*16+k for base in range(0,4096,64) for step in range(4) for k in range(16)]
assert indices==list(range(4096))
report={'status':'source_and_CPU_structural_checks_pass','mechanism':'K64 grouping; not interiteration double buffering','parent_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'candidate_source_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'unchanged':['dispatch','packed layout','perK16 dequant arithmetic','ascendingK16 accumulation','output epilogue','host validation','grid'],'source_counts':{'GMMA_ops':256,'commit_wait_pairs':64,'CTA_barriers':192},'CUDA_compile':'pending','GPU_correctness':False,'performance':'unknown'}
(P/'status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
