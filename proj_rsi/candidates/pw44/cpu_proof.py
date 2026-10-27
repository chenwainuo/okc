"""PW44 strict source, offline NVCC and physical closure. Never imports torch."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from proj_rsi.candidates.pw44.build_patch import apply,value,sha,PARENT_SOURCE,PARENT_MANIFEST,VARIANTS
FLAGS=['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def named(text):
 out={}
 for n in ast.parse(text).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):out[n.name]=ast.dump(n,include_attributes=False)
  elif isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name):out[t.id]=ast.dump(n,include_attributes=False)
 return out
# Reuse pure source extraction/stream helpers through AST, not their importing main.
env=globals()
for node in ast.parse((HERE.parent/'pw40/cpu_proof.py').read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ('productions','factories','device','tables','streams','resources'):
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-pure-offline-factory>','exec'),env)

def source_proof():
 old=(HERE.parent/'pw38/submission.py').read_text();new=(HERE/'submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
 assert sha(old)==PARENT_SOURCE and file_sha(HERE.parent/'pw38/manifest.json')==PARENT_MANIFEST
 for f,h in json.loads((HERE.parent/'pw38/manifest.json').read_text())['files'].items():assert file_sha(HERE.parent/'pw38'/f)==h
 assert apply(old,delta)==new and apply(new,delta,True)==old
 a,b=named(old),named(new);changed=sorted(k for k in a if a[k]!=b.get(k));assert changed==['RQCU','RQPROD','RQV','_rqbake'] and set(a)==set(b)
 for k in ['prepare','kernel_fn','_time_us','_rqcall','_rqsbake','_rqval','RQBAKE','RQSBAKE','RQKS','RQTOL','RQPROD2','_rqstart','MARGIN','PREPARE_BUDGET_S','RWCU']:
  assert a[k]==b[k],k
 assert '_rqconfirm' not in b
 cuold,cunew=value(old,'RQCU'),value(new,'RQCU')
 # Entire core through the pre-epilogue point is byte-exact after only layout/assert inverse.
 marker='  if (INS) pc[0] = clock64() - tq0;'
 prefix=cunew[:cunew.index(marker)]
 for edit in reversed(delta['edits'][:2]):prefix=prefix.replace(edit['after'],edit['before'],1)
 assert prefix==cuold[:cuold.index(marker)]
 private=delta['edits'][2]['after'].split('    } else if constexpr (FL & 65536) {\n')[0]
 assert private.count('EP = false')==1 and 'if (EP && issuer) zpoll();' in private
 for required,count in [('fence.proxy.async.shared::cta;',3),('cp.async.bulk.wait_group.read 0;',2),('if (!f) asm volatile("trap;");',1),('fence.proxy.async.global;',1),('bar.sync %0, 128;',3)]:assert private.count(required)==count,(required,private.count(required))
 # The original EH body remains a literal suffix in the new branch.
 start='    if constexpr (FL & 65536) {\n';end='    } else {\n    const unsigned long long ce0 = INS ? clock64() : 0;'
 body=cuold[cuold.index(start)+len(start):cuold.index(end,cuold.index(start))]
 assert '    } else if constexpr (FL & 65536) {\n'+body in cunew
 restored=private.replace('    if constexpr (FL & 262144) {\n','    if constexpr (FL & 65536) {\n',1)
 oldbar='    asm volatile("bar.sync 1, %0;\\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free'
 restored=restored.replace('    if constexpr (!(FL & 524288)) { '+oldbar.strip()+'\n    }',oldbar,1)
 restored=restored.replace('    const uint32_t slab = sbase + C::OUT + g * XB;','    const uint32_t slab = xring + g * XB;',1)
 assert restored=='    if constexpr (FL & 65536) {\n'+body,'private arm changed more than slab and crossWG-barrier conditional'

 assert value(new,'RQPROD')==value(old,'RQPROD')+[n for n,_ in VARIANTS]
 assert value(new,'RQV')[:len(value(old,'RQV'))]==value(old,'RQV') # existing EH rows are last
 bake=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='_rqbake')
 iterator=next(n.iter for n in ast.walk(bake) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='nm')
 choices=[]
 for layer in value(new,'RQLAYERS'):
  for m in [128,256]:
   got=eval(compile(ast.Expression(iterator),'<actual-bake-enumeration>','eval'),{'RQBAKE':value(new,'RQBAKE'),'layer':layer,'m':m})
   oldlist=value(old,'RQBAKE')[m];assert got[:len(oldlist)]==oldlist
   assert got[len(oldlist):]==([n for n,_ in VARIANTS] if layer in ('qkv_gate','pooler') and m==128 else [])
   choices.append(dict(layer=layer,m=m,candidates=got,KS=value(new,'RQKS')[layer]))
 neg=[]
 for e in delta['edits']:
  try:apply(old.replace(e['before'],e['before']+'\n'+e['before'],1),delta)
  except ValueError:neg.append(e['id'])
  else:raise AssertionError('duplicate anchor accepted')
 from proj_rsi.static_check import check_source
 lint=check_source(new,str(HERE/'submission.py'));assert lint['ok'],lint
 return {'status':'PASS_STRICT_FULL_PARENT_INVERSE','parent_source':PARENT_SOURCE,'source':sha(new),'changed_AST':changed,'optional_choices':choices,'negative_duplicate_anchors':neg,'old_EH_body_literal_exact':True,'math_producer_codec_zero_trap_counter_Dwait0_core':'BYTE_EXACT_AFTER_LAYOUT_ASSERT_INVERSE','lint':lint}

def physical(sass):
 result=[]
 for table in tables(sass):
  args=tuple(map(int,re.findall(r'Li(\d+)E',table.splitlines()[0])));fl=args[-1]
  if not fl&262144:continue
  rows=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
  stsm=[p for p,i in rows if 'STSM.' in i];reduce=[p for p,i in rows if 'UTMAREDG.' in i]
  wait=[p for p,i in rows if 'WARPGROUP.DEPBAR.LE' in i and '0x0' in i]
  depbar=[p for p,i in rows if i.startswith('DEPBAR.LE')]
  hgmma=[p for p,i in rows if 'HGMMA.' in i]
  assert len(stsm)==8 and len(reduce)==2 and wait and depbar and max(hgmma)<max(wait)<min(stsm)
  assert max(stsm[:4])<reduce[0]<min(stsm[4:])<reduce[1]<max(depbar)
  bars=[(p,i) for p,i in rows if i.startswith('BAR.SYNC')]
  pre=[(p,i) for p,i in bars if max(wait)<p<min(stsm)]
  assert len(pre)==(0 if fl&524288 else 1),(fl,pre)
  for group,issue in [(stsm[:4],reduce[0]),(stsm[4:],reduce[1])]:
   between=[i for p,i in rows if max(group)<p<issue]
   assert any('FENCE.VIEW.ASYNC.S' in i for i in between) and any(i.startswith('BAR.SYNC') for i in between)
  assert not re.search(r'\b(?:LDL|STL)\b',table)
  result.append(dict(flag=fl,prestore_crossWG_barrier=pre,STSM=stsm,TMA_reduce=reduce,Dwait0=max(wait),TMA_read_drain=max(depbar),all_poststore_publication_and_read_retirement='PASS'))
 assert len(result)==4
 return result

def main():
 r={'state':'CPU_PW44_STARTED_GPU_UNRUN'}
 try:
  r['source_proof']=source_proof();new=(HERE/'submission.py').read_text();old=(HERE.parent/'pw38/submission.py').read_text();res={}
  for stem,text in [('device',new),('control',old)]:
   p=HERE/(stem+'_source.cu');p.write_text(device(text));cmd=['/usr/local/cuda/bin/nvcc',*FLAGS,'-Xptxas=-v','-cubin',str(p),'-o',str(HERE/(stem+'.sm90a.cubin'))]
   x=subprocess.run(cmd,text=True,capture_output=True,timeout=180);log=x.stdout+x.stderr;(HERE/(stem+'-resource.log')).write_text(log);assert x.returncode==0,log
   assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log) and not re.search(r'C75|serializ',log,re.I)
   res[stem]=resources(log);assert len(res[stem])==len(productions(text)) and all(v==168 for v in res[stem].values())
   sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(HERE/(stem+'.sm90a.cubin'))],text=True);(HERE/('sass.txt' if stem=='device' else 'control-sass.txt')).write_text(sass)
   ru=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(HERE/(stem+'.sm90a.cubin'))],text=True);(HERE/(stem+'.resources')).write_text(ru)
  a,b=streams((HERE/'sass.txt').read_text()),streams((HERE/'control-sass.txt').read_text());assert len(a)==23 and len(b)==19
  for args,stream in b.items():assert a[args]==stream,('OLD_CONTROL_SASS_DRIFT',args)
  r['old_19_control_SASS_PC_predicate_instructions']='EXACT';r['physical']=physical((HERE/'sass.txt').read_text())
  r['resources']={stem:{','.join(map(str,k)):v for k,v in val.items()} for stem,val in res.items()}
  # Actual prelaunch guard is independently compiled as host code without torch or CUDA execution.
  guard=json.loads((HERE/'delta.json').read_text())['edits'][3]['after'].split('  k<<<')[0]
  kernel=device(new).split('void instantiate')[0]
  test=kernel+'\nint pw44_guard_compile(RqP& p,int KS){using C=RqC<128,5,1,0,0,0,851971>;auto k=rq<128,5,1,0,0,0,851971>;\n'+guard+'return 0;}\n'
  # Compile the extracted guard as a real template; bind FL without rewriting its body.
  test=kernel+'\ntemplate<int FL> int pw44_guard_compile(RqP& p,int KS){using C=RqC<128,5,1,0,0,0,FL>;auto k=rq<128,5,1,0,0,0,FL>;\n'+guard+'return 0;}\ntemplate int pw44_guard_compile<851971>(RqP&,int);\n'
  for flag in [327682,327683,851970,851971]:test+=f'static_assert(RqC<128,5,1,0,0,0,{flag}>::SMEM==178176);\n'
  test+='static_assert(RqC<128,5,1,0,0,0,65539>::SMEM==144520);\n'
  (HERE/'guard_compile.cu').write_text(test)
  x=subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-c',str(HERE/'guard_compile.cu'),'-o',str(HERE/'guard_compile.o')],text=True,capture_output=True,timeout=180)
  (HERE/'guard-compile.log').write_text(x.stdout+x.stderr);assert x.returncode==0,x.stderr
  r.update(state='CPU_PW44_SOURCE_CODEGEN_PASS_GPU_UNRUN_LIFETIME_REVIEW_REQUIRED',old_instances=19,new_instances=4,new_dynamic_shared=178176,old_dynamic_shared=144520,register_pool=64512,threads=384,guard_host_compiled=True,CUDA_calls=0,production_imports=0)
 except BaseException:
  import traceback;r['failure']=traceback.format_exc();(HERE/'cpu-proof.json').write_text(json.dumps(r,indent=2)+'\n');raise
 (HERE/'cpu-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(r['state'])
if __name__=='__main__':main()
