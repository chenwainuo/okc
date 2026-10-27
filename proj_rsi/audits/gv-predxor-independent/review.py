"""CPU-only immutable GV source/compiled-dispatch review; writes own workdir only."""
import argparse, ast, hashlib, importlib.util, json, re, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def check_xors(body,blocks):
 x=re.findall(r'@P\d+ LOP3.LUT (R\d+), (R\d+)(?:\.reuse)?, (U?R\d+)(?:\.reuse)?, RZ, 0x3c, !PT',body)
 assert len(x)==16*blocks
 for i in range(0,len(x),16):
  block=x[i:i+16];dest={v[0]for v in block};mask={v[2]for v in block}
  assert all(a==b for a,b,c in block)and len(dest)==16 and len(mask)==1 and not dest&mask
 return len(x)
def review(version):
 D=ROOT/f'proj_rsi/candidates/gv_predxor_pw45_{version}'
 W=Path(__file__).resolve().parent/version;W.mkdir(exist_ok=True)
 manifest=json.loads((D/'manifest.json').read_text())
 for group in ('files','external_inputs'):
  for row in manifest[group]:assert sha((D if group=='files'else ROOT)/row['path'])==row['sha256']
 P=load(D/'proof.py','independent_gv_proof_'+version);B=P.B
 # Replay author proof in an isolated writable directory, retaining every frozen byte.
 for p in D.iterdir():
  if p.is_file():shutil.copy2(p,W/p.name)
 P.D=W
 P.main()
 proof=json.loads((W/'proof.json').read_text());assert proof==json.loads((D/'proof.json').read_text())
 for stem in ('control-full','candidate-full'):
  for flag,suffix in (('--dump-sass','.sass'),('--dump-resource-usage','.resources')):
   assert subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump',flag,str(D/(stem+'.o'))],text=True,timeout=60)==(D/(stem+suffix)).read_text()
 # Strengthen physical carry evidence: one stable XOR mask, no mask/output alias.
 physical=[]
 for symbol,body in P.functions(D/'candidate-full.sass').items():
  if 'gv_predxor'not in symbol:continue
  t=tuple(map(int,re.findall(r'Li(\d+)E',symbol)));physical.append({'symbol':symbol,'xors':check_xors(body,1 if t[2]==1 else 16)})
 # Compile the ACTUAL new gvl pointer-selection prefix and unchanged shape guard.
 cu=B.value((D/'submission.py').read_text(),'GVCU')
 start=cu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\nstatic int64_t gvl')
 guard='  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;'
 end=cu.index(guard,start)+len(guard)
 prefix=cu[start:end]
 header='#include <cstdint>\n#include <cassert>\n#include <cstdio>\nusing cudaStream_t=void*;struct G{int m,N,K;};\n'
 template='template<int FMT,int MT,int ST,int WR,int WK,int KU,int S,int RT,int XS,int V>\n'
 stubs=template+'void gv(G){}\n'+template+'void gv_predxor(G){}\n'
 tail='\n return k==gv_predxor<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>?1:0;}\n'
 # Expected behavior uses explicit four authorized shapes, then original K/N guard.
 main='int main(){long checks=0,redirects=0;int ms[]={0,1,2,7,8,9,15,16,17,32,128,256};int nk[][2]={{4096,8192},{8192,4096},{2304,8192},{512,8192},{4095,8192},{4096,8191}};\n'
 tuples=list(dict.fromkeys(tuple(map(int,x.split(',')))for x in re.findall(r'return gvl<([0-9,]+)>',(D/'control-full.cu').read_text())))
 # Explicit instantiations in full TU also bind the complete original template set.
 if not tuples:tuples=[tuple(map(int,x.split(',')))for x in re.findall(r'template int64_t gvl<([0-9,]+)>',(D/'control-full.cu').read_text())]
 assert len(tuples)==95
 for t in tuples:
  reviewed=t in B.TARGETS;fmt,mt,st,wr,wk,ku,s,rt,xs,v=t
  call='gvl<'+','.join(map(str,t))+'>(p,nullptr,0)'
  expected_shape='((p.N==4096&&p.K==8192&&((p.m==1||p.m==8)&&'+str(mt)+'==1||p.m==16&&'+str(mt)+'==2))||(p.N==8192&&p.K==4096&&p.m==16&&'+str(mt)+'==2))'
  main+='for(int m:ms)for(auto &q:nk){G p{m,q[0],q[1]};int expected=(p.N%'+str(16*wr*rt)+'||p.K!='+str(s*wk*ku*64)+'||m<1)?-1:('+str(int(reviewed))+'&&'+expected_shape+'?1:0);int actual='+call+';if(actual!=expected)return 2;checks++;redirects+=actual==1;}\n'
 main+='printf("{\\\"checks\\\":%ld,\\\"redirects\\\":%ld}\\n",checks,redirects);}\n'
 # Use a regular printf format string (only C++ escaping, never shell interpolation).
 main=main.replace('\\\"','\\"').replace('\\n','\\n')
 src=header+stubs+prefix+tail+main
 cpp=W/'actual-prefix.cpp';cpp.write_text(src)
 exe=W/'actual-prefix'
 subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
 actual=json.loads(subprocess.check_output([str(exe)],text=True));assert actual['checks']==95*12*6
 # A wrong M8 boundary in the actual prefix must fail these same expected fixtures.
 mutant=src.replace('p.m == 8','p.m == 7',1);assert mutant!=src
 mcpp=W/'wrong-scope.cpp';mcpp.write_text(mutant);mexe=W/'wrong-scope'
 subprocess.run(['g++','-std=c++17','-O2',str(mcpp),'-o',str(mexe)],check=True,capture_output=True,text=True)
 assert subprocess.run([str(mexe)],capture_output=True).returncode==2
 revision=None
 if version=='v2':
  V1=ROOT/'proj_rsi/candidates/gv_predxor_pw45_v1';B1=load(V1/'build.py','independent_gv_v1_build')
  added=(1,1,2,1,8,16,1,1,0,1)
  assert B.TARGETS==B1.TARGETS+[added] and (D/'helper.cuh').read_bytes()==(V1/'helper.cuh').read_bytes()
  clause=' ||\n      ('+' && '.join(f'{k} == {v}'for k,v in zip(B.PARAMS,added))+')'
  assert cu.count(clause)==2
  restored=cu.replace(clause,'').replace('only seven reviewed','only six reviewed')
  assert restored==B.value((V1/'submission.py').read_text(),'GVCU')
  f1=P.functions(V1/'candidate-full.sass');f2=P.functions(D/'candidate-full.sass')
  assert len(f1)==102 and len(f2)==103 and all(f1[k]==f2[k]for k in f1)
  r1=P.resources(V1/'candidate-full.resources');r2=P.resources(D/'candidate-full.resources');assert all(r1[k]==r2[k]for k in f1)
  capture=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.02/collection-1793115060902266190/logs/scheduler/pw38-hotpath-trace-02/run/capture.json'
  assert sha(capture)=='6621431f187721855daab0668b53d6db2b5e28378df66ac94ed70d17587e79aa'
  actualcapture=json.loads(capture.read_text());targets=[]
  for name in ('decode1-pooler','verify1-pooler','decode16-pooler','decode16-unpooler'):
   symbol=actualcapture['cases'][name]['actual_graph_nodes'][0]['function_mangled']
   target=tuple(map(int,re.findall(r'Li(\d+)E',symbol)));assert target in B.TARGETS;targets.append({'case':name,'actual_template':target})
  revision={'only_added_template':added,'all102_v1_SASS_and_resources_exact':True,'four_actual_trace_targets_covered':targets}
 for group in ('files','external_inputs'):
  for row in manifest[group]:assert sha((D if group=='files'else ROOT)/row['path'])==row['sha256']
 return {'decision':'GO_SOURCE_CPU_ONLY_GPU_UNRUN','version':version,'manifest_SHA256':sha(D/'manifest.json'),'source_SHA256':sha(D/'submission.py'),
 'frozen_pins_verified':sum(len(manifest[k])for k in ('files','external_inputs')),'disposable_author_proof_exact':True,
 'compiled_actual_host_prefix':actual,'wrong_M8_scope_negative':'REJECTED','physical_stable_mask_disjoint_destination_checks':physical,
 'semantic':proof['semantic'],'resources':proof['physical'],'v2_delta':revision,'GPU_contexts_extensions_prepare_API_queue_calls':0}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--version',default='v1');a=ap.parse_args();out=review(a.version);(Path(__file__).parent/(a.version+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='resources'},indent=2))
