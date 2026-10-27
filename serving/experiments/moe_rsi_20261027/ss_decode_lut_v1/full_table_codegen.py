"""Second bounded offline variant: 8KB full signed table, same two kernels."""
from pathlib import Path
import json,re,subprocess,hashlib
D=Path(__file__).resolve().parent;F=D/'full_table';F.mkdir(exist_ok=True)
table=json.loads((D/'magnitude-lut.json').read_text())
full=[table[(s&127)*8+(q&7)]^(((q&8)<<12)^((s&128)<<8)) for s in range(256)for q in range(16)]
src=(D/'kernel.cuh').read_text()
a=src.index('__host__ __device__ __forceinline__ uint16_t lookup_bits');b=src.index('template<int T> __device__ __forceinline__ void mma',a)
replacement='''__host__ __device__ __forceinline__ uint16_t lookup_bits(uint32_t q,uint32_t s,const uint16_t* lut){
 return lut[s*16u+q];
}
__device__ const uint16_t magnitude_lut[4096]={\n'''+',\n'.join(','.join(hex(v)for v in full[i:i+16])for i in range(0,4096,16))+'\n};\n'
src=src[:a]+replacement+src[b:]
src=src.replace('i<1024','i<4096').replace('+32+2048+1024','+32+8192+1024').replace('moe_ss_lut_v1','moe_ss_lut_full_v1')
(F/'kernel.cuh').write_text(src)
for name in ['compile.cu','adapter.py','serving_both.py']:
 (F/name).write_text((D/name).read_text().replace('moe_ss_lut_v1','moe_ss_lut_full_v1'))
(F/'magnitude-lut.json').write_text(json.dumps(full)+'\n')
cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(F/'compile.cu'),'-o',str(F/'device.cubin')]
p=subprocess.run(cmd,capture_output=True,text=True,check=True);(F/'resource.log').write_text(p.stdout+p.stderr)
sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(F/'device.cubin')],text=True);(F/'sass.txt').write_text(sass)
results=[]
for f in sass.split('Function : ')[1:]:
 rows=re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]*?);',f)
 results.append({'function':f.splitlines()[0],'instructions':len(rows),'LOP3':sum('LOP3'in i for i in rows),'LDS_U16':sum('LDS.U16'in i for i in rows),'HGMMA':sum('HGMMA'in i for i in rows)})
assert not re.search(r'[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',p.stdout+p.stderr)
(F/'codegen.json').write_text(json.dumps({'state':'CPU_ONLY_GPU_UNRUN','SASS':results,'registers':list(map(int,re.findall(r'Used (\d+) registers',p.stdout+p.stderr))),'extra_shared_bytes':8192,'shared_T8_bytes':27680,'kernel_sha256':hashlib.sha256((F/'kernel.cuh').read_bytes()).hexdigest()},indent=2)+'\n')
print((F/'codegen.json').read_text())
