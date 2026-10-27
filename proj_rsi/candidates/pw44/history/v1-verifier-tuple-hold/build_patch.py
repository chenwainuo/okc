"""PW44 private EH output slabs over exact scored PW38; no production import."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'pw38'
PARENT_SOURCE='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
PARENT_MANIFEST='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
PRIVATE=262144;NOBAR=524288
VARIANTS=[('ehpriv_z_bar',327682),('ehpriv_hz_bar',327683),('ehpriv_z_free',851970),('ehpriv_hz_free',851971)]
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
 for n in ast.parse(text).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
 raise KeyError(name)
def apply(text,edits,reverse=False):
 if isinstance(edits,dict):edits=edits['edits']
 for e in reversed(edits) if reverse else edits:
  before,after=(e['after'],e['before']) if reverse else (e['before'],e['after'])
  if text.count(before)!=1:raise ValueError('strict anchor '+e['id'])
  text=text.replace(before,after,1)
 return text

def build():
 old=(PARENT/'submission.py').read_text();assert sha(old)==PARENT_SOURCE and sha((PARENT/'manifest.json').read_text())==PARENT_MANIFEST
 edits=[]
 def add(id,region,before,after):edits.append(dict(id=id,region=region,before=before,after=after))
 before='  static constexpr int SMEM = 1024 + ST * (XB + WB) + 24 * ST + 16;'
 after='''  // PW44 bit18: two 16KiB output slabs beyond rings/barriers. Bit19 removes only the alias-protecting pre-store barrier.
  static constexpr int OUT = ((ST * (XB + WB) + 24 * ST + 16 + 1023) / 1024) * 1024;
  static constexpr int SMEM = (FL & 262144) ? 1024 + OUT + 2 * XB : 1024 + ST * (XB + WB) + 24 * ST + 16;'''
 add('private-layout','kernel.layout',before,after)
 before='(FL == 65538 || FL == 65539)), "EH backport approved M128/ST5/selfzero full-output variants only");'
 after='(FL == 65538 || FL == 65539 || FL == 327682 || FL == 327683 || FL == 851970 || FL == 851971)), "EH backport approved M128/ST5/selfzero full-output variants only");\n  static_assert(!(FL & 262144) || (FL & 65536), "PW44 private requires EH");\n  static_assert(!(FL & 524288) || (FL & 262144), "PW44 barrier removal requires private slabs");'
 add('private-flags','kernel.static-admission',before,after)
 # Preserve the original complete EH branch byte for byte; add a disjoint constexpr branch.
 start='    if constexpr (FL & 65536) {\n';lo=old.index(start);end='    } else {\n    const unsigned long long ce0 = INS ? clock64() : 0;';hi=old.index(end,lo)
 original=old[lo:hi]
 private=original.replace(start,'    if constexpr (FL & 262144) {\n',1)
 bar='    asm volatile("bar.sync 1, %0;\\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free'
 private=private.replace(bar,'    if constexpr (!(FL & 524288)) { '+bar.strip()+'\n    }',1)
 private=private.replace('    const uint32_t slab = xring + g * XB;','    const uint32_t slab = sbase + C::OUT + g * XB;',1)
 assert private!=original and private.count('EP = false')==1
 add('private-EH-branch','kernel.epilogue',start,private+'    } else if constexpr (FL & 65536) {\n')
 # No math, producer, codec, zeroing or TMA retirement changes. New-only host check prevents a non-co-resident ZI launch.
 anchor='  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);'
 guard='''  if constexpr (FL & 262144) {
    if (p.m != 128 || !((p.N == 2304 && KS == 7) || (p.N == 4096 && KS == 4))) return -40;
    int dev = 0, major = 0, minor = 0, sms = 0, resident = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return -41;
    if (major != 9 || minor != 0) return -42;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k, 384, C::SMEM) != cudaSuccess) return -43;
    if (resident < 1 || (long long)sms * resident < (long long)p.NB * KS) return -44;
  }
'''
 add('private-launch-co-residency','host.new-variant-prelaunch',anchor,guard+anchor)
 anchor='    ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),\n'
 rows=''.join(f'    ("{name}", 128, 5, 1, 0, 0, 0, {fl}),\n' for name,fl in VARIANTS)
 add('optional-private-catalog','host.catalog',anchor,anchor+rows)
 anchor='RQPROD = ["s4s5z", "s4s5hz", "s4hz", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]'
 add('compile-private-arms','host.compile-catalog',anchor,anchor[:-1]+', '+', '.join(repr(n) for n,_ in VARIANTS)+']')
 anchor='            for nm in RQBAKE[m]:'
 extra=[n for n,_ in VARIANTS]
 add('two-targets-only-optional-bake','host.bake-enumeration',anchor,'            for nm in RQBAKE[m] + ('+repr(extra)+' if layer in ("qkv_gate", "pooler") and m == 128 else []):')
 delta=dict(version='pw44_v1_private_EH_slabs',parent_source_sha256=PARENT_SOURCE,parent_manifest_sha256=PARENT_MANIFEST,edits=edits)
 new=apply(old,delta);assert apply(new,delta,True)==old
 (HERE/'submission.py').write_text(new);(HERE/'delta.json').write_text(json.dumps(delta,indent=2)+'\n')
 return new,delta
if __name__=='__main__':
 new,d=build();print(sha(new))
