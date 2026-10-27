"""Target-only production-shape CPU codegen check; no full inventory/GPU."""
import ast,hashlib,json,re,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
SOURCE=R/'results/proj-rsi/releases/gv53-keyed-half-v1-239b530cf7ee/submission.py'
LAB=R/'proj_rsi/candidates/gv53_keyed_half_v1/pilot.cu'
TPL='1,1,1,2,16,8,1,1,0,3'
def cu(s):return next(ast.literal_eval(n.value)for n in ast.parse(s).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GVCU'for t in n.targets))
def function(text,name):
 pos=text.index(name+'(');a=text.rfind('template <int FMT',0,pos)if name=='gv_keyed'else text.rfind('__device__',0,pos)
 # All target braces are conventional C++ braces; PTX helper also balanced.
 b=text.index('{',pos);depth=1;i=b+1
 while depth:
  if text[i]=='{':depth+=1
  elif text[i]=='}':depth-=1
  i+=1
 return text[a:i]
def streams(s):
 return {sym:[(int(m[1],16),re.sub(r'\s+',' ',m[2].strip()),m[3],m[4])for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',body)]for sym,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',s,re.S)}
def main():
 source=SOURCE.read_text();full=cu(source);lab=LAB.read_text();report={'state':'CPU_DIAGNOSIS_NOT_ACTUAL_OFFICIAL_BINARY','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'helper_source_exact':function(full,'gv_fix_keyed')==function(lab,'gv_fix_keyed'),'target_body_source_exact':function(full,'gv_keyed')==function(lab,'gv_keyed'),'actual_official_SO_SASS_available':False,'production_config':'_gvext extra_cuda_cflags=[-O3,-lineinfo]; architecture externally inferred/unarchived; full moduleinstantiatesGVI.','lab_config':'Explicitcompute90a/sm90a; sameO3/lineinfo; direct2functionbinding.','target_comparisons':{}}
 assert report['source_sha256']=='239b530cf7eead0a2dbf0af5c29aea4ba02e449500eb80e45b255b361cf2f900';assert report['helper_source_exact']and report['target_body_source_exact']
 prefix=full[:full.index('// FP32 router, small m:')].replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
 target=prefix+'\ntemplate int64_t gvl<'+TPL+'>(const G&,cudaStream_t,int);\n'
 (D/'production-target.cu').write_text(target)
 lab_stream=streams((LAB.with_suffix('.sass')).read_text());key=next(k for k in lab_stream if'gv_keyed'in k)
 for arch in ['90','90a']:
  stem=D/('production-sm'+arch);argv=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch,'-Xptxas=-v','-c',str(D/'production-target.cu'),'-o',str(stem.with_suffix('.o'))]
  t=time.monotonic();p=subprocess.run(argv,text=True,capture_output=True,timeout=90);stem.with_suffix('.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
  sass=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(stem.with_suffix('.o'))],text=True);stem.with_suffix('.sass').write_text(sass);res=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(stem.with_suffix('.o'))],text=True);stem.with_suffix('.resources').write_text(res)
  ss=streams(sass);rows=ss[key];report['target_comparisons'][arch]={'argv':argv,'wall_s':time.monotonic()-t,'native_target_exact_PC_predicate_encoding_vs_lab_sm90a':rows==lab_stream[key],'nonNOP':sum(x[1]!='NOP'for x in rows),'slots':len(rows),'register_resource_excerpt':res[res.index(' Function '+key):res.index(' Function '+key)+250]}
 (D/'codegen.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
