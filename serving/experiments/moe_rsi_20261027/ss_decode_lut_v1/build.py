"""Generate an isolated T8/DIV1+8 candidate. Never touches frozen inputs."""
from pathlib import Path
import hashlib,json,struct
D=Path(__file__).resolve().parent
P=D.parent/'ss_pipeline_v2'

def bf16_bits(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0]
    return ((u+0x7fff+((u>>16)&1))>>16)&65535

def scale(s):
    e,m=s>>3,s&7
    return m*2**-9 if e==0 else (1+m/8)*2**(e-7)
grid=[0.,.5,1.,1.5,2.,3.,4.,6.]
# Scale-major, eight adjacent positive FP4 magnitudes. Signed values use XOR.
table=[bf16_bits(grid[q]*scale(s)) if s!=127 else 0x7fc0 for s in range(128) for q in range(8)]
helper='''// Raw FP4 and FP8 signs restore signed zero exactly; global stays in epilogue.
__host__ __device__ __forceinline__ uint16_t lookup_bits(uint32_t q,uint32_t s,const uint16_t* lut){
 return uint16_t(lut[(s&127u)*8u+(q&7u)]^(((q&8u)<<12)^((s&128u)<<8)));
}
'''
lut='__device__ const uint16_t magnitude_lut[1024]={\n'+',\n'.join(','.join(hex(v) for v in table[i:i+16]) for i in range(0,1024,16))+'\n};\n'
kernel=(P/'kernel.cuh').read_text()
kernel=kernel.replace('namespace moe_ss_v2','namespace moe_ss_lut_v1')
# Remove now-unused decoder functions; retain matrix/protocol bodies byte for byte.
start=kernel.index('// Canonical NVFP4 numeric representation')
end=kernel.index('template<int T> __device__ __forceinline__ void mma',start)
kernel=kernel[:start]+helper+lut+kernel[end:]
kernel=kernel.replace('uint32_t full=addr(memory+2*STAGE*2),empty=full+16;', 'uint32_t full=addr(memory+2*STAGE*2),empty=full+16;\n uint16_t* lut=reinterpret_cast<uint16_t*>(memory+2*STAGE*2+32);')
kernel=kernel.replace('if(tid==0){init(full);','for(int i=tid;i<1024;i+=256)lut[i]=magnitude_lut[i];\n if(tid==0){init(full);')
old='''    uint32_t sc=s0|(s0<<16)|(s1<<8)|(s1<<24);__nv_bfloat162 fs[2],f[4];scale(sc,fs);
#pragma unroll
    for(int h=0;h<2;++h){uint32_t hw=payload>>(16*h);uint32_t q=((hw&255)<<8)|((hw&0xff00)<<16);dequant(q,f+2*h);f[2*h]=__hmul2(f[2*h],fs[0]);f[2*h+1]=__hmul2(f[2*h+1],fs[1]);}
    B* values=reinterpret_cast<B*>(f);'''
new='''    B values[8];
#pragma unroll
    for(int v=0;v<8;++v){
     constexpr int offsets[8]={0,8,4,12,16,24,20,28};
     uint32_t q=(payload>>offsets[v])&15u;
     values[v]=__ushort_as_bfloat16(lookup_bits(q,(v/2)%2?s1:s0,lut));
    }'''
assert old in kernel
kernel=kernel.replace(old,new)
kernel=kernel.replace('return 2*(4096+t*64)*2+32+1024;', 'return 2*(4096+t*64)*2+32+2048+1024;')
(D/'kernel.cuh').write_text(kernel)
adapter=(P/'adapter.py').read_text().replace('moe_ss_v2','moe_ss_lut_v1').replace('moe_ss_both_v2_','moe_ss_lut_v1_')
adapter=adapter.replace('#define PICK(D) if(bm%64==0){LAUNCH(64,D);}else if(bm%32==0){LAUNCH(32,D);}else if(bm%16==0){LAUNCH(16,D);}else{LAUNCH(8,D);}', '#define PICK(D) LAUNCH(8,D); // Bounded T8 changed-stage experiment only.')
(D/'adapter.py').write_text(adapter)
compile_cu=(P/'compile.cu').read_text().replace('moe_ss_v2','moe_ss_lut_v1')
(D/'compile.cu').write_text('\n'.join(l for l in compile_cu.splitlines() if 'project<' not in l or 'project<8,' in l)+'\n')
# Preserve entire serving method; local adapter module has identical bank/public API.
(D/'serving_both.py').write_text((P/'serving_both.py').read_text())
(D/'magnitude-lut.json').write_text(json.dumps(table)+'\n')
(D/'source-manifest.json').write_text(json.dumps({'parent':'../ss_pipeline_v2','parent_hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['kernel.cuh','adapter.py','serving_both.py','cpu-proof.json']},'candidate_hashes':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['kernel.cuh','adapter.py','compile.cu','serving_both.py']},'variants':['T8_DIV1','T8_DIV8'],'status':'CPU_ONLY_GPU_UNRUN'},indent=2)+'\n')
