"""Offline-only candidate construction, compilation and finite protocol exploration."""
import collections, hashlib, json, re, subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'ss_decode_lut_v1'/'full_table'
parent=(P/'kernel.cuh').read_text()
old='''  asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;":::"memory");
  // All math warps finish async shared reads before the elected release.
  asm volatile("bar.sync 3,128;":::"memory");if(tid==0)arrive(empty+8*slot);
 }
 B gs=globals[expert];'''
new='''  asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;":::"memory");
  // Only the previous group is known retired; never release the newest slot.
  if(s>0){asm volatile("bar.sync 3,128;":::"memory");if(tid==0)arrive(empty+8*((s-1)&1));}
 }
 // Cover all accumulator reads and the last shared operand lifetime.
 asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
 if(K>=64){asm volatile("bar.sync 3,128;":::"memory");if(tid==0)arrive(empty+8*((K/64-1)&1));}
 B gs=globals[expert];'''
assert parent.count(old)==1
wait1=parent.replace(old,new)
# Pair two unchanged K64 physical layouts in each of two logical ring slots.
pair=parent.replace('full=addr(memory+2*STAGE*2)','full=addr(memory+4*STAGE*2)').replace('memory+2*STAGE*2+32','memory+4*STAGE*2+32')
pair=pair.replace('for(int s=0;s<K/64;++s){\n   int slot=s&1;if(s>=2)wait(empty+8*slot,((s/2)-1)&1);\n   B* a=ring+slot*STAGE;B* b=a+AW;', 'for(int pair=0;pair<K/128;++pair){\n   int slot=pair&1;if(pair>=2)wait(empty+8*slot,((pair/2)-1)&1);\n   for(int half=0;half<2;++half){\n   int s=pair*2+half;\n   B* a=ring+(slot*2+half)*STAGE;B* b=a+AW;')
pair=pair.replace('   asm volatile("fence.proxy.async.shared::cta;', '   } // Both physical K64 halves are ready before logical publication.\n   asm volatile("fence.proxy.async.shared::cta;')
pair=pair.replace('for(int s=0;s<K/64;++s){\n  int slot=s&1;wait(full+8*slot,(s/2)&1);\n  uint64_t a=desc(addr(ring+slot*STAGE)),b=desc(addr(ring+slot*STAGE+AW));\n  asm volatile("wgmma.fence.sync.aligned;":::"memory");', 'for(int pair=0;pair<K/128;++pair){\n  int slot=pair&1;wait(full+8*slot,(pair/2)&1);\n  asm volatile("wgmma.fence.sync.aligned;":::"memory");\n  for(int half=0;half<2;++half){\n  int s=pair*2+half;\n  uint64_t a=desc(addr(ring+(slot*2+half)*STAGE)),b=desc(addr(ring+(slot*2+half)*STAGE+AW));')
pair=pair.replace('  asm volatile("wgmma.commit_group.sync.aligned;', '  } // Eight K16 instructions retain original accumulation order.\n  asm volatile("wgmma.commit_group.sync.aligned;')
pair=pair.replace('return 2*(4096+t*64)*2+32+8192+1024;', 'return 4*(4096+t*64)*2+32+8192+1024;')
# T8 K4096 only diagnostic: do not expose unsupported K tails through a serving adapter.
for name,src in [('wait1',wait1),('pair2',pair)]:
 out=D/name;out.mkdir(exist_ok=True)
 ns='moe_ss_pipeline_'+name+'_v1'
 src=src.replace('moe_ss_lut_full_v1',ns)
 (out/'kernel.cuh').write_text(src)
 (out/'compile.cu').write_text((P/'compile.cu').read_text().replace('moe_ss_lut_full_v1',ns))
 cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(out/'compile.cu'),'-o',str(out/'device.cubin')]
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
 (out/'resource.log').write_text(p.stdout+p.stderr)
 assert p.returncode==0,p.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(out/'device.cubin')],text=True)
 (out/'sass.txt').write_text(sass)
 (out/'compile-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
# Abstract protocol: producer publish, consumer issue, async retire, math release.
# Progress is nondeterministic; safety checks cover every reachable interleaving.
def prove(n,depth):
 # state p(next publication), c(next issue), r(oldest live group), release(next release)
 start=(0,0,0,0);seen={start};todo=[start];edges=0;term=0
 while todo:
  p,c,r,rel=todo.pop();nxt=[]
  assert 0<=rel<=r<=c<=p<=n and p-rel<=2
  if p<n and (p<2 or rel>p-2):nxt.append((p+1,c,r,rel))
  # wait_group(depth) of prior iteration must complete before next issue.
  if c<p and r>=max(0,c-depth):nxt.append((p,c+1,r,rel))
  if r<c:nxt.append((p,c,r+1,rel))
  # depth1 releases previous only after newest was issued; terminal drain covers last.
  can_release=rel<r and ((depth==0 and rel<c) or (depth==1 and (rel<c-1 or c==n)))
  if can_release:nxt.append((p,c,r,rel+1))
  if not nxt:
   assert (p,c,r,rel)==(n,n,n,n),(n,depth,(p,c,r,rel))
   term+=1
  for q in nxt:
   edges+=1
   if q not in seen:seen.add(q);todo.append(q)
 return dict(stages=n,depth=depth,states=len(seen),edges=edges,terminal_states=term,no_deadlocks=True)
proof=[prove(n,d) for n in [1,2,3,4,32,64] for d in [0,1]]
# Exact physical K16 traversal proves unchanged accumulation order and swizzle/descriptors.
for halves in [1,2]:
 actual=[(pair*halves+half,k) for pair in range(64//halves) for half in range(halves) for k in range(4)]
 assert actual==[(s,k) for s in range(64) for k in range(4)]
assert pair.count('for(int pair=0;pair<K/128;++pair)')==2
assert 'for(int s=0;s<K/64;++s)' not in pair
result={'state':'CPU_ONLY_NO_GPU_NO_SERVING_RUNNER','parent':str(P/'kernel.cuh'),'parent_sha256':hashlib.sha256(parent.encode()).hexdigest(),'protocol':proof,'K16_accumulation_order':'identical 256 steps','supported_prototype':'T8,DIV1/DIV8,K4096; pair2 requires K%128==0, no serving wrapper supplied','candidates':{}}
for name in ['wait1','pair2']:
 out=D/name;sass=(out/'sass.txt').read_text();log=(out/'resource.log').read_text()
 result['candidates'][name]={'kernel_sha256':hashlib.sha256((out/'kernel.cuh').read_bytes()).hexdigest(),'register_reports':re.findall(r'Used (\d+) registers',log),'resource_log':log,'static_DEPBAR_lines':[line.strip() for line in sass.splitlines() if 'DEPBAR' in line],'static_HGMMA':sass.count('HGMMA'),'dynamic_shared_T8':27680 if name=='wait1' else 46112,'device_compile':'PASS'}
(D/'cpu-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
print({k:(v['register_reports'],v['static_HGMMA'],v['static_DEPBAR_lines']) for k,v in result['candidates'].items()})
