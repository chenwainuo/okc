"""Dedicated compact HALF compile option; no new bake policy or old variant edits."""
from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent;R=D.parents[2]
P=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
old=P.read_text();assert hashlib.sha256(old.encode()).hexdigest()=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
FL=262195;NAME='t32s6hze_half_compact256';delta=[]
def edit(a,b):delta.append([a,b])
edit('__global__ void __launch_bounds__(384, 1) rq(', '__global__ void __launch_bounds__((FL & 262144) ? 256 : 384, (FL & 262144) ? 2 : 1) rq(')
edit('  constexpr bool HALF = (FL & 16) != 0;', '  constexpr bool COMPACT = (FL & 262144) != 0;\n  static_assert(!COMPACT || (NT == 32 && ST == 6 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && FL == 262195), "compact HALF approved unpooler32 type only");\n  constexpr bool HALF = (FL & 16) != 0;')
edit('  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;', '  // Map physical producer WG1 to the original logical WG2; logical WG1 is absent.\n  const int tid = threadIdx.x + ((COMPACT && threadIdx.x >= 128) ? 128 : 0), warp = tid >> 5, lane = tid & 31;')
edit('  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\\n");', '  if constexpr (COMPACT) asm volatile("setmaxnreg.inc.sync.aligned.u32 216;\\n");\n  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\\n");')
edit('  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;', '  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;\n  if constexpr (FL & 262144) {\n    if (p.m != 32 || p.N != 8192 || p.K != 4096 || p.KC != 64 || KS != 2 || p.walias != 0) return -47;\n  }')
edit('KS), dim3(384), C::SMEM, at::cuda::getCurrentCUDAStream()', 'KS), dim3((FL & 262144) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()')
edit('    ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),\n]', '    ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),\n    ("t32s6hze_half_compact256", 32, 6, 1, 0, 0, 0, 262195),\n]')
new=old
for a,b in delta:
 assert new.count(a)==1,(a,new.count(a))
 new=new.replace(a,b)
back=new
for a,b in reversed(delta):assert back.count(b)==1;back=back.replace(b,a)
assert back==old
assert (FL&~262144)==51
(D/'submission.py').write_text(new);(D/'delta.json').write_text(json.dumps(delta,indent=2)+'\n');(D/'delta.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='inc501d',tofile='compact256HALFv1')))
# A bijection between every active old HALF math/producer lane and compact lanes.
mapping={t:t+(128 if t>=128 else 0) for t in range(256)}
assert set(mapping.values())==set(range(128))|set(range(256,384))
assert all((p&31)==(l&31) and p//128==(0 if l<128 else 1) for p,l in mapping.items())
assert mapping[128]==256 and mapping[192]==320
for ks in range(2):
 for hh in range(2):
  # Zeroing is exactly 64-lane partition, and each half-CTA writes unique rows/cols.
  oldcells=set();newcells=set()
  for p,l in mapping.items():
   if l//32>=10:
    for e in range(l-320,16*8,64):
     cell=(ks*16+e//8,hh*64+(e%8)*8)
     oldcells.add(cell);newcells.add(cell)
  assert oldcells==newcells and len(newcells)==128
assert sum(l//32<4 for l in mapping.values())==128
assert sum(l//32>=8 for l in mapping.values())==128
def astkeys(t):
 return [ast.dump(n,include_attributes=False) for n in ast.parse(t).body]
x,y=astkeys(old),astkeys(new);changed=[i for i,(a,b)in enumerate(zip(x,y)) if a!=b];assert len(x)==len(y) and len(changed)==2
proof={'state':'CPU_ROLE_MAPPING_PROVEN_RUNTIME_LIFETIME_UNRUN','source_sha256':hashlib.sha256(new.encode()).hexdigest(),'parent_source_sha256':hashlib.sha256(old.encode()).hexdigest(),'exact_inverse':True,'changed_AST_nodes':['RQV','RQCU'],'old_bake_catalog_and_selectors_unchanged':True,'new_variant':[NAME,32,6,1,0,0,0,FL],'new_variant_not_offered_to_bake':True,'physical_to_logical_mapping':'0..127 ->0..127;128..255 ->256..383','mapping_bijective_256_active_old_HALF_threads':True,'math_WG0_and_producer_WG1_each128_physical_lanes':True,'preserved_logical_roles':{'EARLY_init':256,'W_producer':256,'X_producer':288,'prefetch':320,'zeroing':list(range(320,384))},'preserved_source_lifetime':'Only entry thread mapping/register budget/launch bounds and metadata guard change. Entire math, descriptor, codec, ring full/wempty/xempty arrivals/phases, TMA byte counts, WGMMA fences/commits/waits, zero/reduce counter protocol, epilogue barriers and async store waits byte-identical. Logical omitted mathWG1 already returned in old HALF; its exit had no barrier participation. __syncthreads now covers exactly256 present lanes.','setmaxnreg':'Each physical WG collectively executes dec40 or inc216; no partial WG.128*(216+40)=32768 allocation target.','runtime_guard':{'m':32,'N':8192,'K':4096,'KC':64,'KS':2,'walias':0},'limitations':['Compile cap/spills and actual runtime co-residency unproven until measured; finite mapping does not prove GPU scheduling/zero handshake progress.','No old HALF source or historical loss modified; no new selector is enabled.']}
(D/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
