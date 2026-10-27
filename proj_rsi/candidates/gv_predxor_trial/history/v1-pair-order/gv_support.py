"""Pure CPU lab source/physical controls; no production/torch import."""
import ast,collections,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];CAND=ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2';TRIAL=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gvpred_frozen_build',CAND/'build.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
TARGETS=B.TARGETS;NEW='gv_predxor_pw45_v2'
SIG='int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t'
DECL='int64_t gvpred_attrs(int64_t,int64_t,int64_t);\nint64_t gvpred_oldrun('+SIG+');'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def streams(text):
 result={}
 for m in re.finditer(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S):
  rows=[(int(x[1],16),re.sub(r'\s+',' ',x[2].strip()),x[3],x[4])for x in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',m[2])]
  assert len(rows)==len(re.findall(r'/\*[0-9a-f]+\*/',m[2]))and rows
  result[m[1]]=rows
 return result

def resources(text):return {k:{a:int(b)for a,b in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',v)}for k,v in re.findall(r'Function (\S+):\n\s*(.*)',text)}
def value(text,key):return B.value(text,key)
def append_helpers(cu,mod,label):
 cases='\n'.join('    case %d: return gvl<%s>(p, st, occ);'%(i,','.join(map(str,c)))for i,c in enumerate(mod.GVI))
 assert cu==mod.GVCU.replace('@@CASES@@',cases),'actual generated GV source drift'
 source=(ROOT/f'proj_rsi/candidates/{label}/submission.py').read_text();assert mod.GVCU==value(source,'GVCU')
 start=cu.index('template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\nstatic int64_t gvl(');end=cu.index('\n// FP32 router, small m:',start)
 oldlaunch=cu[start:end].replace('static int64_t gvl(','static int64_t gvpred_oldlaunch(',1)
 if label==NEW:
  addition=json.loads((CAND/'delta.json').read_text())['typed_edits'][1]['addition'];assert oldlaunch.count(addition)==1;oldlaunch=oldlaunch.replace(addition,'',1)
 assert 'auto k = gv<'in oldlaunch and 'gv_predxor<'not in oldlaunch
 a=cu.index('static int64_t gvgo(');b=cu.index('\nint64_t gvrun(',a)
 oldgo=cu[a:b].replace('static int64_t gvgo(','static int64_t gvpred_oldgo(',1).replace('return gvl<','return gvpred_oldlaunch<')
 exported=cu[b:cu.index('\nint64_t gvocc(',b)].replace('gvrun(','gvpred_oldrun(',1).replace('return gvgo(','return gvpred_oldgo(',1)
 helper='\n'+oldlaunch+'\n'+oldgo+exported+'\n'
 # Separate query bodies avoid referencing an absent newkernel in the parent module.
 for newer in ((0,1)if label==NEW else(0,)):
  kernel='gv_predxor'if newer else'gv'
  helper+='''template<int FMT,int MT,int ST,int WR,int WK,int KU,int S,int RT,int XS,int V>
static int64_t gvpred_query%d(int field){auto k=%s<FMT,MT,ST,WR,WK,KU,S,RT,XS,V>;
 constexpr size_t sm=(WK+1)*(WR*RT*MT*32)*16;
 cudaFuncAttributes a;if(cudaFuncGetAttributes(&a,k)!=cudaSuccess)return -1;
 if(field==1)return a.numRegs;if(field==2)return a.localSizeBytes;if(field==3)return a.sharedSizeBytes;
 if(field==4)return a.maxDynamicSharedSizeBytes;if(field==5)return a.maxThreadsPerBlock;
 if(field==6){int n=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,WR*WK*32,sm)!=cudaSuccess)return -2;return n;}
 if(field==7){int dev=0,n=0;if(cudaGetDevice(&dev)!=cudaSuccess||cudaDeviceGetAttribute(&n,cudaDevAttrReservedSharedMemoryPerBlock,dev)!=cudaSuccess)return -3;return n;}
 return -4;}
'''%(newer,kernel)
 helper+='int64_t gvpred_attrs(int64_t inst,int64_t newer,int64_t field){\n'
 for newer in ((0,1)if label==NEW else(0,)):
  helper+='if(newer==%d){switch(inst){\n'%newer
  for t in TARGETS:helper+='case %d:return gvpred_query%d<%s>((int)field);\n'%(mod.GVI.index(t),newer,','.join(map(str,t)))
  helper+='default:return -5;}}\n'
 helper+='return -6;}\n'
 return cu+helper,{'appended_source_sha256':sha(helper),'oldlaunch_sha256':sha(oldlaunch),'oldgo_sha256':sha(oldgo),'targets':TARGETS}
def canonical_gate(sass,res,label):
 actual=streams(sass);ref=streams((CAND/('candidate-full.sass'if label==NEW else'control-full.sass')).read_text());ra=resources(res);rr=resources((CAND/('candidate-full.resources'if label==NEW else'control-full.resources')).read_text())
 assert len(ref)==(103 if label==NEW else 96)
 for key in ref:assert actual[key]==ref[key],('actual GV PC/predicate/encoding drift',label,key);assert ra[key]==rr[key],('actual GV resource drift',label,key)
 actual_new={k for k in actual if 'gv_predxor' in k};assert actual_new=={k for k in ref if 'gv_predxor'in k}
 return {'exact_original_functions':96,'exact_new_functions':7 if label==NEW else 0,'full_actual_SASS_PC_predicate_encoding':True}
def check_attrs(t,attrs,reg):
 sm=(t[4]+1)*(t[3]*t[7]*t[1]*32)*16;threads=t[3]*t[4]*32
 assert attrs['regs']==reg and attrs['local']==0 and attrs['user_static']==0 and attrs['reserved']==1024,('live GV registers/local/static/reserved',t,attrs)
 assert attrs['max_threads']>=threads and attrs['max_dynamic']>=sm and attrs['occupancy']>=1,(t,attrs,sm,threads)
 assert ((reg+7)//8)*8*threads<=65536
 return dict(template=list(t),threads=threads,dynamic_smem=sm,**attrs)
def actual_gate(original,cu,sass,res,ext,mod,label):
 rebuilt,meta=append_helpers(original,mod,label);assert rebuilt==cu
 gate=canonical_gate(sass,res,label);rr=resources((CAND/'candidate-full.resources').read_text());live=[]
 for t in TARGETS:
  suffix='ILi'+'ELi'.join(map(str,t))+'EEv1G';symbol='_Z10gv_predxor'+suffix;reg=rr[symbol]['REG']
  for newer in((0,1)if label==NEW else(0,)):
   attrs={key:int(ext.gvpred_attrs(mod.GVI.index(t),newer,i))for key,i in [('regs',1),('local',2),('user_static',3),('max_dynamic',4),('max_threads',5),('occupancy',6),('reserved',7)]}
   live.append(dict(newer=newer,**check_attrs(t,attrs,reg)))
 return dict(gate,helper=meta,live_attrs=live,before_first_prepare_math=True)
def selected_is_new(pick,m,layer):
 if pick.get('family')!='GV':return False
 t=tuple(pick['variant'])
 return t in TARGETS and ((layer=='pooler'and((t[1]==1 and m in(1,8))or(t[1]==2 and m==16)))or(layer=='unpooler'and t[1]==2 and m==16))
def selected_graph_gate(mod,pick,events,m,layer):
 if pick.get('family')!='GV':return
 newer=mod.__file__.endswith('/'+NEW+'/submission.py')and selected_is_new(pick,m,layer)
 token=('gv_predxor<'if newer else'gv<')+', '.join(map(str,pick['variant']))+'>'
 assert any(token in e for e in events),('actual selected GV kernel identity',events,token)
 if newer:assert len(events)==1,('new GV graph must have one main',events)
 else:assert not any('gv_predxor<'in e for e in events)
def cpu_check(A):
 import types
 delta=json.loads((CAND/'delta.json').read_text());old=(ROOT/'proj_rsi/candidates/pw45/submission.py').read_text();new=(CAND/'submission.py').read_text()
 # Execute the actual Python _gvext source-generation statements with a fake non-GPU JIT.
 def factory(text):
  archived=ROOT/'results/proj-rsi/scheduler/artifacts/research.pw45.gpu.01/collection-1793111000996134576/logs/scheduler/pw45-stack-01/run/pw38.gv20_7bb50cec.original.cu'
  tuples=list(dict.fromkeys(tuple(map(int,s.split(',')))for s in re.findall(r'return gvl<([0-9,]+)>',archived.read_text())))
  env={'GVCU':value(text,'GVCU'),'GVI':tuples,'_gvmod':None};captured={}
  def fake(*a,**kw):captured.update(kw);return object()
  env['load_inline']=fake;node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='_gvext');exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-pure-GVfactory>','exec'),env);env['_gvext']();return types.SimpleNamespace(**env),captured['cuda_sources']
 rows={};mutations=[]
 for label,text in [('pw45',old),(NEW,new)]:
  mod,cu=factory(text);gated,meta=append_helpers(cu,mod,label);assert gated.startswith(cu)and sha(cu)==sha(mod.GVCU.replace('@@CASES@@','\n'.join('    case %d: return gvl<%s>(p, st, occ);'%(i,','.join(map(str,c)))for i,c in enumerate(mod.GVI))))
  canonical=(CAND/('candidate-full.sass'if label==NEW else'control-full.sass')).read_text();res=(CAND/('candidate-full.resources'if label==NEW else'control-full.resources')).read_text();rows[label]=canonical_gate(canonical,res,label)
  for name,ss,rs in [('wrong_predicate',canonical.replace('@P6 LOP3.LUT','@P5 LOP3.LUT',1),res),('wrong_resources',canonical,res.replace('REG:110','REG:111',1)),('missing_main',canonical.replace('Function : _Z2gv','Function : _Z2bad',1),res)]:
   if name=='wrong_predicate'and label!='gv_predxor_pw45_v2':continue
   try:canonical_gate(ss,rs,label)
   except (AssertionError,KeyError):mutations.append(label+'/'+name)
   else:raise AssertionError('negative accepted '+name)
  # Host attr/control-launch helper gets a real CUDA12.8 compile with all original GV device bodies.
  source=gated[:gated.index('// FP32 router, small m:')].replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <type_traits>').replace('#include <ATen/cuda/CUDAContext.h>','')
  helper=gated[len(cu):];hstart=helper.index('template<int FMT,int MT');query=helper[hstart:]
  # queries + function switch instantiate all7kernels, genuine CUDA runtime APIs unresolved/unloaded.
  path=TRIAL/(label+'-attrs.cu');path.write_text(source+query)
  p=subprocess.run(['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-c',str(path),'-o',str(path.with_suffix('.o'))],text=True,capture_output=True,timeout=240);path.with_suffix('.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
  dumped=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(path.with_suffix('.o'))],text=True);dumpres=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(path.with_suffix('.o'))],text=True)
  path.with_suffix('.sass').write_text(dumped);path.with_suffix('.resources').write_text(dumpres)
  actual_functions=streams(dumped);expected_functions=streams(canonical);ar=resources(dumpres);er=resources(res)
  assert len(actual_functions)==(15 if label==NEW else 8)
  for symbol,body in actual_functions.items():assert body==expected_functions[symbol] and ar[symbol]==er[symbol]
  rows[label]['real_cuda_host_attributes_compile']=True;rows[label]['query_compile_device_functions_exact']=len(actual_functions);rows[label]['appended_source_sha256']=meta['appended_source_sha256']
 import mechanism
 plan=mechanism.plan();assert len(plan)==156
 # Exact terminal validator negative controls use stdlib fake tensors and callbacks.
 terminal=mechanism.cpu_terminal_negatives();counter_neg=[]
 model=types.SimpleNamespace(__file__=str(CAND/'submission.py'));t=TARGETS[0];pick={'family':'GV','variant':list(t)};newname='gv_predxor<'+', '.join(map(str,t))+'>';oldname='gv<'+', '.join(map(str,t))+'>'
 selected_graph_gate(model,pick,[newname],1,'pooler');selected_graph_gate(model,pick,[oldname],32,'pooler')
 for name,events,m in [('wrong_label',[oldname],1),('extra_node',[newname,newname],1),('wrong_scope',[newname],32)]:
  try:selected_graph_gate(model,pick,events,m,'pooler')
  except AssertionError:mutations.append(name)
  else:raise AssertionError('graphnegative accepted '+name)
 before={'a':{'name':'arena','values':[12,7,0,19]}}
 A.counter_check(before,{'a':{'name':'arena','values':[20,8,0,19]}},{'a':{'mode':'progress','offset':0,'changes':{0:8,1:1}}},1)
 for name,vals in [('missing_ticket',[12,8,0,19]),('dormant_word',[20,8,1,19]),('wrong_flag',[20,9,0,19])]:
  try:A.counter_check(before,{'a':{'name':'arena','values':vals}},{'a':{'mode':'progress','offset':0,'changes':{0:8,1:1}}},1)
  except AssertionError:counter_neg.append(name)
  else:raise AssertionError('counternegative accepted')
 return {'state':'CPU_LAB_PASS_GPU_UNRUN','sources':A.PINS,'GV':rows,'actual_trace_binding':trace_binding(),'JIT_negatives':mutations,'counter_negatives':counter_neg,'terminal_negatives':terminal,'fixed_plan_per_geometry':len(plan),'balanced_pairs':32,'fullpolicy_shapes':30,'scope_cases_per_policy':74,'GPU_API_calls':0,'production_imports':0}

def trace_binding():
 base=ROOT/'results/proj-rsi/scheduler/artifacts/pw38.hotpath.trace.02/collection-1793115060902266190/logs/scheduler/pw38-hotpath-trace-02/run'
 capture=json.loads((base/'capture.json').read_text());assert capture['status']=='VALID_PW38_HOTPATH_TRACE_ROOT_REVIEW_REQUIRED'
 actual=streams((base/'JIT01.gv20_7bb50cec.sass').read_text());ref=streams((CAND/'control-full.sass').read_text());rows=[]
 for name in ('decode1-pooler','verify1-pooler','decode16-pooler','decode16-unpooler'):
  pick=capture['cases'][name]['actual_selected_policy'];t=tuple(pick['variant']);assert t in TARGETS
  symbol='_Z2gvILi'+'ELi'.join(map(str,t))+'EEv1G';assert actual[symbol]==ref[symbol]
  rows.append({'case':name,'actual_trace_pick':pick,'symbol':symbol,'actual_PC_predicate_encoding':'EXACT_PW45','in_v2_scope':True})
 return {'source':'actual H100 PW38 trace02; GVCU source-equivalent to PW45','rows':rows,'limits':'Source/PC identity only; no sampled instruction hotspot or GPU performance forecast'}
