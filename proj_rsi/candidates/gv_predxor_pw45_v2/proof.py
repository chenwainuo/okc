"""Source, exact actual compiled SASS and finite semantics, CPU only."""
import ast,collections,hashlib,importlib.util,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
s=importlib.util.spec_from_file_location('B',D/'build.py');B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(p):return dict((m.group(1),m.group(2))for m in re.finditer(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',p.read_text(),re.S))
def instructions(x):return re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;',x)
def counts(x):return collections.Counter(re.sub(r'^@!?[UP]\d+\s+','',i).split()[0]for i in instructions(x))
def resources(p):return {k:{a:int(b)for a,b in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',v)}for k,v in re.findall(r'Function (\S+):\n\s*(.*)',p.read_text())}
def checked(test):
 try:test()
 except (AssertionError,ValueError,KeyError):return True
 raise AssertionError('negative accepted')
def helper_check(h):
 assert h.count('mov.b32 ri, %16; mov.b32 mask, %17;')==1
 assert h.index('mov.b32 mask')<h.index('@pr xor.b32')
 assert 'setp.eq.u32 pl, %18, %19; setp.eq.and.u32 pa, %20, 0, pl;'in h
 ops=re.findall(r'setp.eq.and.u32 pr, ri, (\d+), pa; @pr xor.b32 %(\d+), %(\d+), mask;',h)
 assert ops==[(str(i),str(i),str(i))for i in range(16)]
 assert h.count('"+r"(a[')==16
 assert ': "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm)'in h
 return True

def main():
 parent=B.PARENT.read_text();new=(D/'submission.py').read_text();delta=json.loads((D/'delta.json').read_text());assert B.sha(parent)==B.SHA
 inv=delta['strict_whole_file_inverse'];assert B.sha(new)==inv['after_sha256']
 restored=new.replace(inv['candidate_GVCU_literal'],inv['original_GVCU_literal'],1).replace('load_inline("gvp45_predxor_v2_%x"','load_inline("gv20_%x"',1);assert restored==parent
 oldcu=B.value(parent,'GVCU');newcu=B.value(new,'GVCU');assert oldcu.count('gv_fix_pred')==0
 # Exactly one isolated kernel copied; reversing only typed body edits recovers original kernel.
 kstart=oldcu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__');kend=oldcu.index('\n__global__ void p1zero',kstart);oldk=oldcu[kstart:kend]
 k2start=newcu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n__global__',newcu.index('gv_fix_pred'))
 k2end=newcu.index('\n__global__ void p1zero',k2start);newk=newcu[k2start:k2end].removesuffix('\n')
 e=delta['typed_edits'][0];fixed=newk.replace(' gv_predxor(G p) {',' gv(G p) {',1).replace(e['new_escape'],e['old_escape'],1)
 fixed=re.sub(r'  static_assert\(.*?"GVpredxor only seven reviewed format1 templates"\);\n','',fixed,count=1,flags=re.S);assert fixed==oldk
 # No other globals/Python functions changed; GVI/tuning/prepare/math/timer untouched.
 def nodes(t):return {getattr(n,'name',None)or','.join(x.id for x in n.targets if isinstance(x,ast.Name)):ast.dump(n,include_attributes=False)for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Assign))}
 pn,nn=nodes(parent),nodes(new);changed=[k for k in pn if pn[k]!=nn[k]];assert changed==['GVCU','_gvext']
 # Literal source delta removes all additions and exactly recovers entire original CUDA source.
 addition=delta['typed_edits'][1]['addition'];remove=newcu.replace(addition,'',1);helper=(D/'helper.cuh').read_text();remove=remove.replace('\n'+helper+'\n'+newk+'\n','',1);assert remove==oldcu
 helper_check(helper)
 negatives={'wrong_destination':checked(lambda:helper_check(helper.replace('@pr xor.b32 %7, %7','@pr xor.b32 %8, %7'))),'missing_ri_copy':checked(lambda:helper_check(helper.replace('mov.b32 ri, %16; ',''))),'wrong_hm_gate':checked(lambda:helper_check(helper.replace('%20, 0, pl','%20, 1, pl')))}
 # Exhaust all GVI and representative geometry grid; exact target condition from source.
 archive=ROOT/'results/proj-rsi/scheduler/artifacts/research.pw45.gpu.01/collection-1793111000996134576/logs/scheduler/pw45-stack-01/run'
 tuples=list(dict.fromkeys(tuple(map(int,x.split(',')))for x in re.findall(r'return gvl<([0-9,]+)>',(archive/'pw38.gv20_7bb50cec.original.cu').read_text())))
 assert len(tuples)==95
 scope=[];non=0
 for t in tuples:
  for m in (1,8,16,32,128,256):
   for N,K in ((4096,8192),(8192,4096),(2304,8192),(512,8192)):
    select=t in B.TARGETS and ((N==4096 and K==8192 and ((t[1]==1 and m in(1,8))or(t[1]==2 and m==16)))or(t[1]==2 and N==8192 and K==4096 and m==16))
    if select:scope.append({'template':t,'m':m,'N':N,'K':K})
    else:non+=1
 assert len(scope)==14
 eligible=[r for r in scope if r['K']==r['template'][6]*r['template'][4]*r['template'][5]*64 and r['N']%(16*r['template'][3]*r['template'][7])==0];assert len(eligible)==11
 assert all(t[0]==1 and t[7]==1 and t[9]<4 for t in B.TARGETS) # NA1/no alternate escape path
 a=functions(D/'control-full.sass');b=functions(D/'candidate-full.sass');arch=functions(archive/'pw38.gv20_7bb50cec.sass')
 assert len(a)==96 and len(b)==103
 assert all(a[k]==b[k]==arch[k]for k in a)
 ra=resources(D/'control-full.resources');rb=resources(D/'candidate-full.resources');assert all(ra[k]==rb[k]for k in a)
 rows=[]
 for k,v in b.items():
  if 'gv_predxor'not in k:continue
  old=k.replace('10gv_predxor','2gv');assert rb[k]==ra[old];assert rb[k]['STACK']==rb[k]['LOCAL']==0
  t=tuple(map(int,re.findall(r'Li(\d+)E',k)));assert t in B.TARGETS
  c0,c1=counts(a[old]),counts(v);assert c0['HMMA.16816.F32.BF16']==c1['HMMA.16816.F32.BF16']
  # Every appended helper expands to 16 actual predicate-protected in-place XORs.
  xor=re.findall(r'@P\d+ LOP3.LUT (R\d+), (R\d+)(?:\.reuse)?, (U?R\d+)(?:\.reuse)?, RZ, 0x3c, !PT',v)
  blocks=1 if t[2]==1 else 16;assert len(xor)==16*blocks
  assert all(dst==src for dst,src,mask in xor)
  # Compiler may interleave compares, but each static16-output group is physically distinct.
  assert all(len({x[0]for x in xor[i:i+16]})==16 for i in range(0,len(xor),16))
  threads=t[3]*t[4]*32;allocated_regs=((rb[k]['REG']+7)//8)*8*threads;assert allocated_regs<=65536
  NI=t[3]*t[7]*t[1]*32;smem=(t[4]+1)*NI*16
  rows.append({'template':t,'resource':rb[k],'threads':threads,'register_allocation_upper_bound':allocated_regs,'dynamic_smem':smem,'old_instructions':len(instructions(a[old])),'new_instructions':len(instructions(v)),'opcode_counts_old':dict(c0),'opcode_counts_new':dict(c1),'predicated_xors':len(xor),'repair_blocks':blocks,'hmma_count':c1['HMMA.16816.F32.BF16']})
 assert len(rows)==7
 # Actual parent diagnostic flags, all0spill. No contexts, dlopen, torch/prepare imports.
 logs=(D/'candidate-full.log').read_text();properties=re.findall(r'Function properties for (\S+)\n(.*?)(?=ptxas info|\Z)',logs,re.S);assert all(not re.search(r'[1-9][0-9]* bytes spill',v) for k,v in properties if 'gv_predxor' in k);assert sum('gv_predxor' in k for k,v in properties)==7
 subprocess.run(['g++','-std=c++17','-O3',str(D/'semantic.cpp'),'-o',str(D/'semantic_cpu')],check=True)
 semantic=json.loads(subprocess.check_output([str(D/'semantic_cpu')],text=True));assert semantic['scalar_cases']==8388608 and semantic['cursor_chain_cases']==135168
 out={'state':'CPU_SOURCE_SEMANTIC_PHYSICAL_PASS_NOT_GPU_ADMITTED','parent_sha256':B.SHA,'candidate_sha256':sha(D/'submission.py'),'whole_file_inverse':True,'changed_AST_nodes':changed,'old_kernel_source_inverse':True,'original_functions_exact_actual_archived_SASS':len(a),'original_resources_exact':len(a),'appended_functions':7,'scope_cases':scope,'eligible_launch_cases':eligible,'invalid_redirected_cases_return_minus1':3,'non_target_dispatch_cases':non,'semantic':semantic,'negative_controls':negatives,'physical':rows,'limits':['No GPU/context/extension/prepare/benchmarks','Instruction counts are static, not sampled hotspots or cost','No numeric admission until actual current-X fullstorage strict GPU oracles and balanced measurements','H100 actual archive is source-equivalent; official H200 binary/latency not inferred','Other CUDA source literals byte-identical; device compile here covers all95GV plusp1zero, not every other family']}
 (D/'proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('scope_cases','physical')},indent=2))
if __name__=='__main__':main()
