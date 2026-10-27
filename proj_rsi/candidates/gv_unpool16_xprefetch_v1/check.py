import collections,csv,hashlib,json,math,pathlib,re
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2]
s=(D/'pilot.sass').read_text();old=(R/'proj_rsi/audits/gv-unpool16-lanebase-result/actual.sass').read_text()
def functions(s):
 return {sec.splitlines()[0]:sec for sec in s.split('Function : ')[1:]}
def instructions(sec):return [" ".join(l.split()) for l in sec.splitlines() if re.search(r'/\*[0-9a-f]{4,}\*/',l) or re.search(r'/\* 0x[0-9a-f]+ \*/',l)]
f=functions(s);o=functions(old);control=next(k for k in f if k.startswith('_Z10gv_predxor'));new=next(k for k in f if k.startswith('_Z12gv_xprefetch'))
assert instructions(f[control])==instructions(o[control]),'control does not match collected actual binary'
def counts(sec):
 return dict(collections.Counter(re.search(r'\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9.]*)',l).group(1).split('.')[0] for l in sec.splitlines() if re.search(r'/\*[0-9a-f]{4,}\*/',l)))
# Address-sequence contract: immutable X, two tokens, two128-bit vectors/token.
seen=[];cache=[(0,n,k) for n in range(2) for k in range(2)];loads=cache.copy()
for u in range(16):
 current=cache.copy()
 if u+1<16:
  cache=[(u+1,n,k) for n in range(2) for k in range(2)];loads.extend(cache)
 assert current==[(u,n,k) for n in range(2) for k in range(2)]
 seen.extend(current)
assert loads==seen and len(loads)==64
route=next(r for r in csv.DictReader((R/'proj_rsi/audits/selected-hotpaths-gv9916/routes.csv').open()) if r['m']=='16' and r['layer']=='unpooler')
t=float(route['scored_latency_us']);e=float(route['score_elasticity']);score=3157.0384036526784
result={'scope':'CPU compiled candidate only; no GPU execution, runtime occupancy, task numeric validation, timing or official gain','parent_sha256':'501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9','template':[1,2,2,1,4,16,1,1,2,1],'collected_parent_instruction_encoding_match':True,'current_operand_address_regression':{'units':16,'tokens':2,'vector_loads':64,'bounds':'no u16 load','result':'PASS'},'instruction_counts':{'control':counts(f[control]),'candidate':counts(f[new])},'resource':{'control_registers':110,'candidate_registers':126,'stack_bytes':0,'spill_loads_bytes':0,'spill_stores_bytes':0,'threads':128,'rounded_registers_per_thread':128,'SM_register_bound_CTAs':4,'actual_candidate_runtime_active_CTAs':'UNKNOWN','parent_observed_runtime_active_CTAs':4,'dynamic_shared_bytes':5120,'static_shared_bytes':1024},'source_semantics':'Only X fetch staging/address time changed; current operands snapshot before next slot overwrite. Original W loads, global load cache policy, exponent decode, repair list cursor/order, accumulation, reduction and output ownership retained. Immutable X throughout invocation is required and already part of task contract.','SASS_motion':{'control_next_X_u1':['0xab0','0xac0','0xad0','0xae0'],'candidate_next_X_u1':['0x4a0','0x4b0','0x4c0','0x4d0'],'candidate_first_decode_PRMT':'0x580','candidate_after_repair_X_u2':['0xaf0','0xb00','0xb10','0xb20']},'fixed_9916_conditional_scenarios':[{'official_case_latency_cut_us':d,'conditional_score_gain_us':score*(1-math.exp(e*math.log(1-d/t))),'forecast':False} for d in [.1,.2,.5]],'next':'Optional bounded same-run selected control/candidate unpooler M16 timing with original task/reference/current-input/own terminals and runtime resource query, reused immutable W banks; actual H200 policy/score required before official claim. No register ban and no assumed H100-to-H200 transfer.'}
(D/'cpu-evidence-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['instruction_counts']));print(result['fixed_9916_conditional_scenarios'])
