"""Two bounded offline register-lifetime trials; reuse exact full-table codec."""
from pathlib import Path
import hashlib,json,re,subprocess
D=Path(__file__).resolve().parent;P=D.parent/'ss_decode_lut_v1/full_table'
old='''    B values[8];
#pragma unroll
    for(int v=0;v<8;++v){
     constexpr int offsets[8]={0,8,4,12,16,24,20,28};
     uint32_t q=(payload>>offsets[v])&15u;
     values[v]=__ushort_as_bfloat16(lookup_bits(q,(v/2)%2?s1:s0,lut));
    }
#pragma unroll
    for(int v=0;v<8;++v){int row=row0+8*((v/2)%2),k=2*(lane%4)+(v%2)+8*(v/4);a[sw(row,step*16+k)]=values[v];}'''
new='''#pragma unroll
    for(int v=0;v<8;++v){
     constexpr int offsets[8]={0,8,4,12,16,24,20,28};
     uint32_t q=(payload>>offsets[v])&15u;
     int row=row0+8*((v/2)%2),k=2*(lane%4)+(v%2)+8*(v/4);
     a[sw(row,step*16+k)]=__ushort_as_bfloat16(lookup_bits(q,(v/2)%2?s1:s0,lut));
    }'''
parent=(P/'kernel.cuh').read_text();assert parent.count(old)==1
results=[]
for variant,rolled in [('fused',False),('fused_rolled_step',True)]:
 F=D/variant;F.mkdir(exist_ok=True);ns='moe_ss_lut_v2_'+variant
 kernel=parent.replace(old,new).replace('moe_ss_lut_full_v1',ns)
 if rolled:
  needle='#pragma unroll\n   for(int step=0;step<4;++step)';assert kernel.count(needle)==1
  kernel=kernel.replace(needle,'#pragma unroll 1\n   for(int step=0;step<4;++step)')
 (F/'kernel.cuh').write_text(kernel)
 for name in ['adapter.py','compile.cu']:(F/name).write_text((P/name).read_text().replace('moe_ss_lut_full_v1',ns))
 cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(F/'compile.cu'),'-o',str(F/'device.cubin')]
 p=subprocess.run(cmd,text=True,capture_output=True,check=True);log=p.stdout+p.stderr;(F/'resource.log').write_text(log)
 sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(F/'device.cubin')],text=True);(F/'sass.txt').write_text(sass)
 functions=[]
 for f in sass.split('Function : ')[1:]:
  if '7projectILi'not in f.splitlines()[0]:continue
  div=int(re.search(r'projectILi8ELi(\d+)E',f)[1]);rows=re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)
  functions.append({'DIV':div,'instructions':len(rows),'LOP3':sum('LOP3'in ins for _,ins in rows),'LDS_U16':sum('LDS.U16'in ins for _,ins in rows),'HGMMA':sum('HGMMA'in ins for _,ins in rows)})
 regs=list(map(int,re.findall(r'Used (\d+) registers',log)));assert len(regs)==2
 # NVCC emits DIV8 then DIV1 for this compile.cu; preserve labeled function logs.
 labeled={}
 for block in log.split('Compiling entry function ')[1:]:
  div=int(re.search(r'projectILi8ELi(\d+)E',block)[1]);labeled[div]=int(re.search(r'Used (\d+) registers',block)[1])
 for f in functions:f['registers']=labeled[f['DIV']]
 row={'variant':variant,'state':'CPU_OFFLINE_ONLY_GPU_UNRUN','parent_kernel_sha256':hashlib.sha256(parent.encode()).hexdigest(),'kernel_sha256':hashlib.sha256(kernel.encode()).hexdigest(),'functions':functions,'spills_or_stack':bool(re.search(r'[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log)),'same_codec':'byte-identical helper and8192Btable, no new scalar proof','same_matrix_protocol':'unchanged, no new proof'}
 (F/'codegen.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
(D/'codegen.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
