"""CPU-only device prototype generation; never loads CUDA or a torch extension."""
from pathlib import Path
import hashlib,json,subprocess,re
H=Path(__file__).resolve().parent
P=H.parent/'operand_reverse_group8'
src=(P/'extension_source.cu').read_text().split('typedef CUresult (*RqEnc)')[0]
assert hashlib.sha256((P/'extension_source.cu').read_bytes()).hexdigest()=='e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e'
src=src.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
src=src.replace('const uint8_t* W;', 'float* partial;\n  const uint8_t* W;',1)
src=src.replace('rq_px_g8','rq_qkv_g8_sk4')
src=src.replace('constexpr int KC=64,WST=6', 'constexpr int KC=32,WST=6')
src=src.replace('row*2048u+(k>>1)','row*4096u+(k>>1)')
src=src.replace('idx+65536u*u]=x[source+512u*u]', 'idx+131072u*u]=x[source+1024u*u]')
src=src.replace('int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;', 'int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;\n  const int ks=blockIdx.y,k0=32*ks;')
src=src.replace('(nb*8+4*half+w)', '((ks*p.NB+nb)*8+4*half+w)')
src=src.replace('xbase+(uint32_t(s)<<14)', 'xbase+(uint32_t(k0+s)<<14)')
src=src.replace('((size_t)nb*KC+s)', '((size_t)nb*128+k0+s)')
src=src.replace('((size_t)nb*KC+s+WST)', '((size_t)nb*128+k0+s+WST)')
start=src.index('  asm volatile("bar.sync 1,256;')
src=src[:start]+'''  // Each D value owns one FP32 partial; no BF16 rounding before reduction.
#pragma unroll
  for(int j=0;j<8;++j){
    const int row=wg*64+w*16+g,col=cb+8*j+2*t;
    float* q=p.partial+(size_t)ks*128*2304+(size_t)row*2304+col;
    q[0]=d[4*j];q[1]=d[4*j+1];
    q[8*2304]=d[4*j+2];q[8*2304+1]=d[4*j+3];
  }
#endif
}
__global__ void qkv_reduce(const float* partial,__nv_bfloat16* out){
  const unsigned i=blockIdx.x*256u+threadIdx.x;
  if(i<128u*2304u){
    float d=partial[i];
#pragma unroll
    for(int s=1;s<4;++s)d+=partial[s*128u*2304u+i];
    out[i]=__float2bfloat16_rn(d);
  }
}
void instantiate(RqP p,RqTmap x,RqTmap o){
  rx_pack<<<512,256>>>((const uint32_t*)p.X,(uint32_t*)p.XP);
  rq_qkv_g8_sk4<128,8,1,0,0,0,20><<<dim3(36,4),512,RqC<128,8,1,0,0,0,20>::SMEM>>>(p,x,o);
  qkv_reduce<<<1152,256>>>(p.partial,p.out);
}
'''
(H/'device_source.cu').write_text(src)
cmd=['/usr/local/cuda-12.8/bin/nvcc','-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr','-Xptxas=-v','-cubin',str(H/'device_source.cu'),'-o',str(H/'device.sm90a.cubin')]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(H/'resource.log').write_text(r.stdout+r.stderr)
result={'status':'COMPILED_CPU_ONLY' if r.returncode==0 else 'FAILED_CPU_RETAINED','source_sha256':hashlib.sha256(src.encode()).hexdigest(),'parent_source_sha256':hashlib.sha256((P/'extension_source.cu').read_bytes()).hexdigest(),'gpu_calls':0,'compiler_log':r.stdout+r.stderr}
if r.returncode==0:
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(H/'device.sm90a.cubin')],text=True)
 (H/'sass.txt').write_text(sass)
 result['registers']=re.findall(r'Used (\d+) registers',r.stdout+r.stderr)
(H/'compile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
