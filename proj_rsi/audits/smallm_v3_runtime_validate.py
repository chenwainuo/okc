"""Independent CPU-only review of collected v3 evidence. Never imports torch/candidates."""
import ast,collections,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COLLECT=ROOT/'results/proj-rsi/scheduler/artifacts/smallm.cursor.v3.all5.01/collection-1793074281123122605'
OUT=COLLECT/'logs/scheduler/smallm-cursor-v3-01'
LABELS=('pw36','rs32','increment')
SHAPES={'qkv32':('qkv_gate',32,2304),'qkv16':('qkv_gate',16,2304),'pooler32':('pooler',32,4096),'unpooler32':('unpooler',32,8192),'o32':('o',32,8192)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ops(s):
 return {b.splitlines()[0].strip():collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',b)) for b in s.split('Function : ')[1:]}
def source_data(path):
 source=path.read_text();tree=ast.parse(source);ns={}
 needed={'RQV','RQCU','RQSBAKE'}
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id in needed:ns[target.id]=ast.literal_eval(node.value)
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in {'_rqwg','_rqss','_rqsrc'}:exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 ns['RQPROD2']=list(dict.fromkeys(n for cases in ns['RQSBAKE'].values() for n,_ in cases))
 start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and [x.id for x in t.elts if isinstance(x,ast.Name)]==['_0','_1','_01'] for t in n.targets))
 end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GVI' for t in n.targets))
 exec(compile(ast.Module(body=tree.body[start:end+1],type_ignores=[]),str(path),'exec'),ns)
 return ns

def run():
 inv=json.loads((COLLECT/'inventory.json').read_text());d=json.loads((OUT/'result.json').read_text());job=json.loads((ROOT/'proj_rsi/scheduling/jobs/smallm-cursor-v3-01.json').read_text())
 assert not inv['missing']
 for rel,h in inv['files'].items():assert sha(COLLECT/rel)==h,rel
 assert inv['dispatch']['job_id']==job['id'] and inv['dispatch']['source_pins']==dict(job['files'],**{job['review']['path']:job['review']['sha256']})
 for rel,h in inv['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
 assert d['review_sha256']==job['review']['sha256']
 clean=inv['cleanup_observation'];assert clean['terminal_exit']==0 and clean['cleanup_failed'] is False and clean['container_exists'] is False
 assert clean['guard_pids']==clean['compute_pids']==clean['containers']==[] and clean['gpu_memory_mib']==0
 host=(COLLECT/'logs/scheduler/smallm.cursor.v3.all5.01.host.log').read_text();assert 'TRIAL_EXIT okc-proj-smallm-cursor-v3-01 0' in host and '2026-10-27T04:11:07Z' in host
 assert d['scope']==list(SHAPES) and set(d['shapes'])==set(SHAPES) and 'failure' not in d and 'invariant_failure' not in d
 journal=[json.loads(s) for s in (OUT/'raw.jsonl').read_text().splitlines()];assert journal==d['raw'] and len(journal)==770
 for r in journal:assert r['shape'] in SHAPES and r['label'] in LABELS and math.isfinite(r['us']) and r['us']>0
 baselines=d['counter_prepare_baseline'];assert len(baselines)==73
 aliases=collections.defaultdict(list)
 for b in baselines:
  assert b['dtype']=='torch.int32' and b['stride']==[1] and b['storage_offset']==0
  assert b['storage'][2]==b['shape'][0]*4
  aliases[tuple(b['storage'])].append(b['name'])
 data={};jit={}
 before=ops((ROOT/'proj_rsi/candidates/smallm_increment/before.sass').read_text());after=ops((ROOT/'proj_rsi/candidates/smallm_increment/after.sass').read_text())
 for label,folder in zip(LABELS,('pw36','rs32','smallm_increment')):
  src=ROOT/'proj_rsi/candidates'/folder/'submission.py';ns=data[label]=source_data(src)
  assert sha(src)==d['source_pins'][label][1]
  runtime=(OUT/(label+'.rq32.cu')).read_text();vs=[next(v for v in ns['RQV'] if v[0]==name) for name in ns['RQPROD2']]
  assert ns['_rqsrc'](vs)==runtime and sha(OUT/(label+'.rq32.cu'))==d['jit'][label]['source_sha256']
  assert sha(OUT/(label+'.sass'))==d['jit'][label]['SASS_sha256']
  parsed=ops((OUT/(label+'.sass')).read_text());assert len(parsed)==11
  resource=(OUT/(label+'.resources')).read_text();assert 'arch = sm_90a' in resource
  for symbol in parsed:
   block=resource.split('Function '+symbol)[1].split('Function ')[0]
   assert re.search(r'REG:168\b',block) and re.search(r'STACK:0\b',block) and re.search(r'LOCAL:0\b',block)
  expected=after if label=='increment' else before
  for symbol,want in expected.items():assert parsed[symbol]==want,(label,symbol)
  jit[label]={'symbols':len(parsed),'regs':168,'stack':0,'local':0,'source_reconstructed':True,'expected_affected_opcode_counts':True}
 assert (OUT/'pw36.sass').read_bytes()==(OUT/'rs32.sass').read_bytes()
 control=ops((OUT/'rs32.sass').read_text());candidate=ops((OUT/'increment.sass').read_text());assert control.keys()==candidate.keys()
 changed=[]
 for symbol in control:
  nt,st,gs,prb,wm,ins,fl=map(int,re.findall(r'Li(\d+)E',symbol))
  affected=nt in (16,32) and gs==1 and prb==wm==ins==0 and fl==35
  if not affected:assert candidate[symbol]==control[symbol],symbol
  else:
   changed.append({'symbol':symbol,'NT':nt,'ST':st})
   for op in set(control[symbol])|set(candidate[symbol]):
    if op.startswith(('LDS','SYNCS','HGMMA','WARPGROUP')):assert candidate[symbol][op]==control[symbol][op],(symbol,op)
   assert candidate[symbol]['UIMAD.WIDE.U32']==control[symbol]['UIMAD.WIDE.U32']-1
 assert len(changed)==4
 walraw=(ROOT/'results/proj-rsi/scheduler/events.jsonl').read_bytes();assert walraw.endswith(b'\n')
 for line in walraw.splitlines():
  event=json.loads(line);payload=event['payload'];assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==event['sha256']
 live=payload['snapshot']['jobs'][job['id']];assert live['state']=='VALIDATING'
 collection=next(o['collection'] for o in reversed(live['observations']) if 'collection' in o)
 assert collection['sha256']==sha(COLLECT/'inventory.json') and (ROOT/collection['inventory']).resolve()==COLLECT/'inventory.json'
 cases=[];oracle_calls=0
 for shape,(layer,m,n) in SHAPES.items():
  entry=d['shapes'][shape];rows=[r for r in journal if r['shape']==shape];assert len(rows)==154
  expected=[]
  for l in LABELS:expected += [(l,20000,'warmup')]*3
  for l in LABELS[:-1]:expected += [(l,20001+i,'F1') for i in range(11)]
  expected += [('increment',20001+i,'F1') for i in range(11)]
  for l in reversed(LABELS[:-1]):expected += [(l,20001+i,'F1') for i in range(11)]
  orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)]
  for i in range(30):expected += [(LABELS[j],20040+i,'paired') for j in orders[i%6]]
  assert [(r['label'],r['seed'],r['mode']) for r in rows]==expected
  validations=entry['validation'];assert len(validations)==20
  for label in LABELS:
   basis=[v for v in validations if v.get('label')==label and 'full_K_basis_FP64' in v];assert len(basis)==1
   error=float(basis[0]['full_K_basis_FP64'].split()[1]);assert basis[0]['full_K_basis_FP64'].startswith('ok ') and error<.004
   mutations=[v for v in validations if v.get('label')==label and 'graph_mutation' in v]
   assert [v['graph_mutation'] for v in mutations]==['normal','outliers','zero','all_probes','fresh_normal']
   assert all(v['NaN_output_fresh_workspace']=='PASS' for v in mutations)
   pick=entry['picks'][label];prepared=d['prepared_policy'][label][shape];ns=data[label]
   assert pick['rq']==prepared['rq'] and pick['sel']==prepared['sel'] and pick['gv']==prepared['gv']
   if pick['rq']:
    rp=pick['rq'];assert rp['n']==n;name=ns['RQPROD2'][rp['i']];assert name==pick['rq_name']
    v=next(v for v in ns['RQV'] if v[0]==name);assert v[4]==0 and v[7]&2
    assert prepared['counter_path']=='RQ';assert pick['selected_attrs_regs_local_shared']==[168,0,99488 if v[2]==6 else 83080]
    active=next(b for b in baselines if b['name']==label+f"/rq/{(layer,m)}/cnt");assert active['nonzero']==0
    route={'path':'RQ','name':name,'KS':rp['KS'],'counter_storage':active['storage'],'active_reset0':True}
   else:
    assert pick['sel']=='None';i=int(re.match(r'\((\d+),',pick['gv']).group(1));g=ns['GVI'][i]
    assert g[6]==1 and not ((g[9]>>27)&5);assert prepared['counter_path']=='GV_no_counter'
    route={'path':'GV_no_counter','GVI_index':i,'GVI_tuple':g,'active_counter':None}
   cases.append({'shape':shape,'label':label,'reported_final_random_residual':error,'graph_mutations':5,'route':route,'counter_contract':'PASS_BY_SOURCE_BOUND_RUNTIME_ASSERTIONS','terminal_F1':'PASS','terminal_paired':'PASS'})
  for mode,seed in [('F1',20011),('paired',20069)]:
   terminal=[v for v in validations if v.get('immediate_final_timed_output')==mode];assert len(terminal)==1 and terminal[0]['PASS'] is True and terminal[0]['last_fixtures']==dict.fromkeys(LABELS,seed);oracle_calls+=3
 assert oracle_calls==30
 nonzero=[b for b in baselines if b['nonzero']];assert len(nonzero)==42 and sum('/p1cnt/' in b['name'] for b in nonzero)==39
 assert all('/p1cnt/' in b['name'] or b['name'].endswith('/rw_cnt') for b in nonzero)
 for b in nonzero:
  if '/p1cnt/' in b['name']:
   label=b['name'].split('/')[0];i=int(b['name'].split('/')[-1]);assert (data[label]['GVI'][i][9]>>27)&5
 return {'decision':'PASS_RUNTIME_VALIDITY','inventory_sha256':sha(COLLECT/'inventory.json'),'result_sha256':sha(OUT/'result.json'),'inventory_files_verified':len(inv['files']),'source_pins_verified':len(inv['dispatch']['source_pins']),'missing_artifacts':inv['missing'],'raw_records':770,'basis_checks':15,'graph_mutations':75,'strict_terminal_oracle_calls':30,'source_bound_counter_invariant_executions':125,'counter_baselines':73,'physical_counter_storages':len(aliases),'alias_groups':[{'storage':k,'names':v} for k,v in aliases.items() if len(v)>1],'nonzero_baselines':nonzero,'jit':jit,'all_changed_JIT_functions':changed,'collector_WAL_hash_matches':True,'cases':cases,'cleanup':clean,'limitation':'Successful final counter tensors/output arrays are not serialized; PASS relies on pinned executed fail-closed assertions and recorded post-check checkpoints, not independent remeasurement. No attribution of attempt01 exact failed tensor; no official/stack promotion.'}
if __name__=='__main__':
 result=run();(ROOT/'proj_rsi/audits/smallm-v3-runtime-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('nonzero_baselines','cases','cleanup')},indent=2))
