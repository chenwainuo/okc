"""Read-only PW44 critic. CPU cubin inspection and fake-runtime host-guard test only."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
from proj_rsi.candidates.pw44 import verify_frozen as V,cpu_proof as C,protocol_proof as P,codegen_proof as G
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'proj_rsi/candidates/pw44'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard_test():
 delta=json.loads((D/'delta.json').read_text());body=delta['edits'][3]['after'].split('  k<<<')[0]
 cpp='''#include <cassert>
#include <iostream>
struct RqP{int m,N,NB;};
int fail=0,smmajor=9,smminor=0,sms=132,resident=1,calls=0;
constexpr int cudaSuccess=0,cudaDevAttrComputeCapabilityMajor=1,cudaDevAttrComputeCapabilityMinor=2,cudaDevAttrMultiProcessorCount=3;
int cudaGetDevice(int* d){calls++;*d=0;return fail==1?1:0;}
int cudaDeviceGetAttribute(int* v,int a,int){calls++;*v=a==1?smmajor:a==2?smminor:sms;return fail==a+1?1:0;}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* v,int,int threads,int shared){calls++;assert(threads==384&&shared==178176);*v=resident;return fail==5?1:0;}
struct C{static constexpr int SMEM=178176;};
template<int FL>int guard(RqP p,int KS){int k=0;
'''+body+'''return 0;}
int main(){
 RqP q{128,2304,18},p{128,4096,32};int tests=0;
 int(*gs[])(RqP,int)={guard<327682>,guard<327683>,guard<851970>,guard<851971>};for(auto g:gs){assert(g(q,7)==0);assert(g(p,4)==0);tests+=2;}
 q.m=127;assert(guard<851971>(q,7)==-40);q.m=128;assert(guard<851971>(q,4)==-40);q.N=8192;assert(guard<851971>(q,7)==-40);q.N=2304;tests+=3;
 for(fail=1;fail<=4;fail++){assert(guard<851971>(q,7)==-41);tests++;}fail=5;assert(guard<851971>(q,7)==-43);tests++;fail=0;
 smmajor=8;assert(guard<851971>(q,7)==-42);smmajor=9;smminor=1;assert(guard<851971>(q,7)==-42);smminor=0;tests+=2;
 resident=0;assert(guard<851971>(q,7)==-44);resident=1;sms=125;assert(guard<851971>(q,7)==-44);sms=126;assert(guard<851971>(q,7)==0);assert(guard<851971>(p,4)==-44);sms=128;assert(guard<851971>(p,4)==0);tests+=5;
 resident=2;sms=64;assert(guard<851971>(p,4)==0);tests++;
 calls=0;fail=1;smmajor=0;assert(guard<65539>(q,1)==0&&calls==0);tests++;
 std::cout<<tests;}
'''
 with tempfile.TemporaryDirectory(prefix='pw44-source-review-') as tmp:
  src=Path(tmp)/'guard.cpp';exe=Path(tmp)/'guard';src.write_text(cpp)
  subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(exe)],check=True,capture_output=True)
  count=int(subprocess.check_output([str(exe)],text=True))
 return {'exact_extracted_guard_sha256':hashlib.sha256(body.encode()).hexdigest(),'fake_runtime_assertions':count,'scope':'C++17 exact branch with fake CUDA APIs; not an occupancy measurement or CUDA runtime test.'}

def main():
 assert sha(D/'manifest.json')=='9add0a2ad776d77e68c8a6869d81d71865422bb3694bf179364f696db80990e8'
 closure=V.check();new=(D/'submission.py').read_text();old=(D.parent/'pw38/submission.py').read_text()
 delta=json.loads((D/'delta.json').read_text());forward=old
 for edit in delta['edits']:
  assert forward.count(edit['before'])==1;forward=forward.replace(edit['before'],edit['after'],1)
 assert forward==new;inverse=new
 for edit in reversed(delta['edits']):
  assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
 assert inverse==old
 assert C.device(new)==(D/'device_source.cu').read_text() and C.device(old)==(D/'control_source.cu').read_text()
 for stem,saved in [('device','sass.txt'),('control','control-sass.txt')]:
  actual=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(D/(stem+'.sm90a.cubin'))],text=True)
  assert actual==(D/saved).read_text()
 codegen=G.proof();protocol=P.proof()
 # Separate whole logical-coordinate enumeration from parent register/ISA mapping.
 sets=[];coords=set()
 for wg in range(2):
  for half in range(2):
   addresses={P.OUT+wg*16384+token*128+(((col//8)^(token%8))*16)+2*(col%8) for token in range(half*64,(half+1)*64) for col in range(64)}
   expected=set(range(P.OUT+wg*16384+half*8192,P.OUT+wg*16384+(half+1)*8192,2));assert addresses==expected
   assert not any(addresses&s for s in sets);sets.append(addresses)
   for token in range(half*64,(half+1)*64):
    for col in range(64):assert (token,64*wg+col) not in coords;coords.add((token,64*wg+col))
 assert coords=={(t,c) for t in range(128) for c in range(128)}
 # Equal-resource arms differ in presence of only the pre-store cross-WG named barrier at source level.
 # Compiler epilogue scheduling changes are explicitly permitted and separately frozen.
 rows=C.streams((D/'sass.txt').read_text());control=C.streams((D/'control-sass.txt').read_text())
 assert len(rows)==23 and len(control)==19 and all(rows[k]==v for k,v in control.items())
 source=C.source_proof();guard=guard_test()
 proof={'verdict':'GO_SOURCE_CPU_ONLY_RUNTIME_UNPROVEN','source_sha256':sha(D/'submission.py'),'manifest_sha256':sha(D/'manifest.json'),'closure':closure,'independent_seven_edit_byte_inverse':True,'pure_device_sources_exact':True,'retained_cubins_redumped_exact':True,'old19_full_streams_and_resources_exact':True,'independent_output_coordinates':len(coords),'private_half_slabs_disjoint':len(sets),'dynamic_shared':178176,'static_shared':1024,'total_shared':179200,'registers':168,'threads':384,'register_pool':64512,'source':source,'codegen':codegen,'protocol':protocol,'guard':guard,'GPU':'UNRUN','occupancy':'UNMEASURED','performance_forecast':'NONE'}
 out=ROOT/'proj_rsi/audits/pw44-source-independent.json';out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k not in ('source','codegen','protocol')},indent=2))
if __name__=='__main__':main()
