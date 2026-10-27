"""Derive bounded unpooler driver from reviewed fixed-RQ harness; CPU only."""
from pathlib import Path
import ast,json,hashlib,shutil
D=Path(__file__).resolve().parent;R=D.parents[2];BASE=D.parent/'rq_descriptor_offset_trial_v1';C=D.parent/'rq_compact256_half_v1'
def value(src,name):
 return next(ast.literal_eval(n.value)for n in ast.parse(src).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id==name)
parent=(D.parent/'gv_predxor_pw45_v2/submission.py').read_text();new=(C/'submission.py').read_text()
env={}
for n in ast.parse(parent).body:
 if isinstance(n,ast.FunctionDef)and n.name in ['_rqwg','_rqss']:exec(compile(ast.Module(body=[n],type_ignores=[]),'factory','exec'),env)
headers='';bodies=[]
for name,src,fl in [('control',parent,35),('oldhalf',parent,51),('compact',new,262195)]:
 cu=value(src,'RQCU');lines=cu.splitlines(True);headers=''.join(l for l in lines if l.startswith('#include'))
 cu=''.join(l for l in lines if not l.startswith('#include'))
 cu=cu.replace('@@WGMMA@@',env['_rqwg'](32)).replace('@@CASES@@',f'case 0: return rq_go<32,6,1,0,0,0,{fl}>(p,x,out,(int)KS,(int)info);')
 bodies.append(f'namespace rq_{name} {{\n{cu}\n}}\n')
binding='''
template<class K> int64_t query(K k,int64_t f,int block,int shared){
 auto rc=cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(rc!=cudaSuccess)return -1000-int(rc);
 cudaFuncAttributes a;rc=cudaFuncGetAttributes(&a,k);if(rc!=cudaSuccess)return -2000-int(rc);
 if(f==1)return a.numRegs;if(f==2)return a.localSizeBytes;if(f==3)return a.sharedSizeBytes;if(f==4)return a.maxDynamicSharedSizeBytes;if(f==5)return a.maxThreadsPerBlock;
 if(f==6){int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,block,shared);return rc==cudaSuccess?n:-3000-int(rc);}if(f==7)return block;if(f==8)return shared;return -9;
}
int64_t attrs(int64_t arm,int64_t f){
 if(arm==0)return query(rq_control::rq<32,6,1,0,0,0,35>,f,384,99488);
 if(arm==1)return query(rq_oldhalf::rq<32,6,1,0,0,0,51>,f,384,62624);
 if(arm==2)return query(rq_compact::rq<32,6,1,0,0,0,262195>,f,256,62624);return -90;
}
int64_t run(int64_t arm,torch::Tensor W,torch::Tensor x,torch::Tensor out,torch::Tensor esc,torch::Tensor cnt,int64_t KS,int64_t Tb,torch::Tensor prof,torch::Tensor sink){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.size(0)==32&&x.size(1)==4096&&out.size(1)==8192&&KS==2,"only unpoolerM32N8192K4096KS2");
 if(arm==0)return rq_control::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);
 if(arm==1)return rq_oldhalf::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);
 if(arm==2)return rq_compact::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);return -90;
}
'''
(D/'binding.cu').write_text(binding);(D/'trial.cu').write_text(headers+''.join(bodies)+binding);shutil.copyfile(BASE/'codec.py',D/'codec.py')
t=(BASE/'driver.py').read_text()
t=t.replace('Two scored-selected inc501d RQ32 cells, narrow descriptor-offset contrast.', 'One unpooler M32 cell: incumbent full, historical HALF, compact256 HALF.')
def rep(a,b):
 global t
 assert t.count(a)==1,(a,t.count(a));t=t.replace(a,b)
rep("LABELS=('control','offset')", "ALL_LABELS=('control','oldhalf','compact');LABELS=ALL_LABELS")
rep("CELLS={'pooler':(4096,8192,4),'unpooler':(8192,4096,2)}", "CELLS={'unpooler':(8192,4096,2)}")
start=t.index('def schedule():');end=t.index('\ndef boundary_columns',start)
t=t[:start]+'''def schedule(labels=ALL_LABELS):
 import itertools
 rows=[]
 for arm in labels:rows += [dict(arm=arm,seed=91000,mode='warmup')for _ in range(3)]
 blocks=('control','oldhalf','compact','oldhalf','control')if len(labels)==3 else('control','compact','control')
 for block,arm in enumerate(blocks):rows += [dict(arm=arm,seed=91100,mode='F1',block=block,sample=i)for i in range(7)]
 orders=list(itertools.permutations(labels))
 for i in range(12):rows += [dict(arm=arm,seed=91200+i,mode='paired',pair=i)for arm in orders[i%len(orders)]]
 return rows

def eligible(attributes,SMs):
 receipt={};accepted=[]
 for arm in ALL_LABELS:
  a=attributes[arm];grid=128 if arm=='control'else 256;capacity=SMs*a['occupancy']
  ok=capacity>=grid
  receipt[arm]={'state':'ELIGIBLE_RUNTIME_CAPACITY'if ok else'SKIPPED_PROTOCOL_CAPACITY_PRECONDITION','required_resident_CTAs':grid,'runtime_resident_capacity':capacity,'grid':[64 if arm=='control'else 128,2,1],'attributes':a,'performance_if_skipped':None,'skip_cause':'PROVEN actualoccupancy capacity below requiredfullgrid; no launch'if not ok else None}
  if ok:accepted.append(arm)
 return tuple(accepted),receipt
''' +t[end:]
start=t.index('def cpu_check():');end=t.index('\ndef run(a):',start)
t=t[:start]+'''def cpu_check():
 verify()
 import itertools
 for labels in [ALL_LABELS,('control','compact')]:
  rows=schedule(labels);f=[r for r in rows if r['mode']=='F1'];assert len({r['seed']for r in f})==1
  for a,b in itertools.combinations(labels,2):
   lead=0
   for i in range(12):
    block=[r['arm']for r in rows if r['mode']=='paired'and r['pair']==i];assert set(block)==set(labels)
    lead+=block.index(a)<block.index(b)
   assert lead==6
 attrs={a:{'occupancy':2 if a=='compact'else 1}for a in ALL_LABELS}
 labels,receipt=eligible(attrs,132);assert labels==('control','compact')and receipt['oldhalf']['performance_if_skipped']is None
 parent=ast.parse((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text());codec=ast.parse((D/'codec.py').read_text())
 named=lambda n:n.name if isinstance(n,ast.FunctionDef)else n.targets[0].id
 original={named(n):ast.dump(n,include_attributes=False)for n in parent.body if isinstance(n,(ast.FunctionDef,ast.Assign))and(isinstance(n,ast.FunctionDef)or isinstance(n.targets[0],ast.Name))}
 assert all(original[named(n)]==ast.dump(n,include_attributes=False)for n in codec.body)
 return {'state':'CPU_COMPACT_HARNESS_PASS_RUNTIME_UNRUN','cells':1,'arms_if_capacity_allows':list(ALL_LABELS),'fixed_F1':True,'twelve_samefixture_triples_or_pairs':'eachpair6/6leadorder','oldHALF_unsafe_skip_first_class':True,'codec_AST_nodes':9,'full_prepare':False,'full_basis':False,'profiler':False}
''' +t[end:]
rep('def run(a):\n started=', 'def run(a):\n global LABELS\n started=')
rep("'scope':'Scored-selected inc501d RQ32ST6FL35 pooler/unpooler fixed source contrast, no freshpolicy/full30/officialclaim'", "'scope':'OneunpoolerM32KS2 actualincumbentfull35/oldHALF51/compact262195; nofreshpolicy/full30/officialclaim'")
rep("phase('torch_and_exact_two_function_compile')", "phase('torch_and_exact_three_function_compile')")
rep("'rq_offset_fixed_'", "'rq_compact_half_'")
rep("'native_type':[32,6,1,0,0,0,35],'private_namespaces':['rq_control','rq_offset']", "'native_types':{'control':[32,6,1,0,0,0,35],'oldhalf':[32,6,1,0,0,0,51],'compact':[32,6,1,0,0,0,262195]},'private_namespaces':['rq_control','rq_oldhalf','rq_compact']")
rep("scoredselected.json; no added device kernel", "fixedexactsource; three functions only")
rep("fields=('registers','local','static','max_dynamic','max_threads','occupancy')", "fields=('registers','local','static','max_dynamic','max_threads','occupancy','launch_threads','launch_shared')")
rep("for i,arm in enumerate(LABELS)};save()", "for i,arm in enumerate(ALL_LABELS)};save()")
rep("for arm,x in r['attributes'].items():assert 0<x['registers']<=168 and x['local']==0 and x['static']==0 and x['max_dynamic']>=99488 and x['max_threads']>=384 and x['occupancy']>=1,(arm,x)", "for arm,x in r['attributes'].items():assert 0<x['registers']<=(128 if arm=='compact'else 168) and x['local']==0 and x['static']==0 and x['max_dynamic']>=x['launch_shared'] and x['max_threads']>=x['launch_threads'] and x['occupancy']>=1,(arm,x)\n  LABELS,r['arm_eligibility']=eligible(r['attributes'],props.multi_processor_count);r['measured_arms']=list(LABELS);save()\n  assert 'control'in LABELS and 'compact'in LABELS,r['arm_eligibility']")
rep("'geometry':[32,6,1,0,0,0,35]", "'geometry':r['build']['native_types']")
rep("for row in schedule()}}", "for row in schedule(LABELS)}}")
rep('2*(n//128)+16', '4*(n//128)+16')
rep('LABELS.index(arm)', 'ALL_LABELS.index(arm)')
rep('for row in schedule():', 'for row in schedule(LABELS):')
start=t.index("   F=[x for x in entry['raw']if x['mode']=='F1'];");end=t.index("   for kind,fresh in [('zero'",start)
t=t[:start]+'''   import itertools
   F=[x for x in entry['raw']if x['mode']=='F1'];blocks=sorted({x['block']for x in F})
   med=[{'block':b,'arm':next(x['arm']for x in F if x['block']==b),'median_us':statistics.median(x['us']for x in F if x['block']==b)}for b in blocks]
   pairs=[{x['arm']:x['us']for x in entry['raw']if x['mode']=='paired'and x['pair']==i}for i in range(12)]
   entry['F1_medians_us']=med;entry['paired']=pairs;entry['contrasts']={}
   for left,right in itertools.combinations(LABELS,2):
    lm=statistics.mean(x['median_us']for x in med if x['arm']==left);rm=statistics.mean(x['median_us']for x in med if x['arm']==right)
    entry['contrasts'][left+'_minus_'+right]={'paired_median_saving_us':statistics.median(x[left]-x[right]for x in pairs),'paired_median_ratio':statistics.median(x[left]/x[right]for x in pairs),'right_wins':sum(x[left]>x[right]for x in pairs),'ties':sum(x[left]==x[right]for x in pairs),'F1_bracket_average_saving_us':lm-rm}
   save()
''' +t[end:]
rep("'VALID_TWO_RQ32_SCORED_SELECTED_DESCRIPTOR_OFFSET_CELLS_ROOT_REVIEW'", "'VALID_ONE_UNPOOLER_COMPACT256_HALF_CAPACITY_SCREEN_ROOT_REVIEW'")
(D/'driver.py').write_text(t)
