"""Strict CPU reconciliation of collected thin-ABI validation; no GPU imports."""
from pathlib import Path
import hashlib,json,re,math,statistics,collections
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
C=R/'results/proj-rsi/scheduler/artifacts/research.gv53.thin.cabi.validation.01/collection-1793141233697823368'
I=json.loads((C/'inventory.json').read_text());assert I['job_id']=='research.gv53.thin.cabi.validation.01'and I['terminal_exit']==0 and not I['missing']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in I['files'].items():assert sha(C/p)==h,('collected hash',p)
for p,h in I['dispatch']['source_pins'].items():assert sha(R/p)==h,('dispatch pin',p)
P=next(C.glob('logs/scheduler/*/run/result.json'));data=json.loads(P.read_text());assert data['state']=='FOUR_INTERFACE_CELLS_COMPLETE_ROOT_REVIEW_REQUIRED'
prior=R/'results/proj-rsi/scheduler/artifacts/research.gv53.keyed.half.fixed.01/collection-1793137198211949643/logs/scheduler/gv53-keyed-half-01/run'
assert sha(P.parent/'thin.so')==data['build']['thin_SO_sha256'];assert sha(prior/'actual.so')==data['build']['prior_SO_sha256']
assert (P.parent/'thin.cu').read_text()==(R/'proj_rsi/candidates/gv53_keyed_half_v1/pilot.cu').read_text()+'\n'+(R/'proj_rsi/candidates/gv53_thin_cabi_validation_v1/bridge.cuh').read_text()

def native(text):
 result={}
 for section in re.split(r'Function\s*:\s*',text)[1:]:
  name=section.splitlines()[0].strip();rows=re.findall(r'/\*([0-9a-f]{4,})\*/\s*(.*?)\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*\n\s*/\*\s*(0x[0-9a-f]+)\s*\*/',section)
  result[name]=[list(x)for x in rows]
 return result
old=native((P.parent/'pybind.sass').read_text());new=native((P.parent/'thin.sass').read_text());identity={}
for name in data['build']['native_identity']:
 assert len(old[name])==360 and old[name]==new[name] and data['build']['native_identity'][name]
 identity[name]={'instruction_count':len(old[name]),'PC_text_encoding_control_exact':True}
for arm,attrs in data['raw_attributes'].items():
 assert attrs['pybind']==attrs['cabi'];assert attrs['cabi']=={'registers':64,'local':0,'user_static':0,'max_dynamic':17408,'max_threads':1024,'occupancy':1,'device_reserved':1024,'arm_echo':int(arm)}

def schedule():
 a=['pybind','cabi'];rows=[]
 for api in a:rows +=[{'arm':api,'seed':91000,'mode':'warmup'}for _ in range(5)]
 for api in ['pybind','cabi','pybind']:rows +=[{'arm':api,'seed':91100+i,'mode':'F1'}for i in range(11)]
 for i in range(24):rows +=[{'arm':api,'seed':91200+i,'mode':'paired','pair':i}for api in(a if i%2==0 else a[::-1])]
 return rows
journal=[json.loads(x)for x in(P.parent/'raw.jsonl').read_text().splitlines()];assert len(journal)==364
all_expected=[];contrasts=[];oracles=[];count={'task_validation':0,'basis_sets':0,'basis_columns':0,'own_terminals':0,'mutations':0,'timed':0,'warmup':0}
for arm in[0,1]:
 for m in[1,8]:
  cell=data['fixed_cells'][f'{arm}:{m}'];assert cell['native_arm']==arm and cell['state']=='VALID_INTERFACE_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and cell['cleanup']=='RESET_AFTER_DRAIN'
  rows=cell['raw'];assert len(rows)==91
  for actual,expected in zip(rows,schedule()):
   assert {k:v for k,v in actual.items()if k!='us'}==expected;assert math.isfinite(actual['us'])and actual['us']>0
   all_expected.append(dict(actual,m=m,native_arm=arm));count['warmup'if actual['mode']=='warmup'else'timed']+=1
  streams=cell['capture_streams'];assert set(streams)=={'pybind','cabi'}and all(isinstance(x,int)and x>0 for x in streams.values());assert cell['replay_stream']not in streams.values()
  assert cell['paired_terminal_interfaces_bitwise_equal']and cell['five_mutation_fixtures_interfaces_bitwise_equal']
  for api in['pybind','cabi']:
   v=cell['validation'][api];assert [x['kind']for x in v['task']]==['normal','outliers','zero'];assert v['basis']=='EXACT'and v['basis_columns']==[0,1,31,63,64,511,512,8191]
   count['task_validation']+=3;count['basis_sets']+=1;count['basis_columns']+=8;oracles +=[x['oracle']for x in v['task']]
  for phase,seed in[('F1',91110),('paired',91223)]:
   terms=cell['terminals'][phase];assert [t['arm']for t in terms]==['pybind','cabi'];assert all(t['seed']==seed and t['before_any_replay_or_mutation']for t in terms);count['own_terminals']+=2;oracles +=[t['oracle']for t in terms]
  muts=cell['mutations'];assert len(muts)==10
  assert [(x['kind'],x['arm'])for x in muts]==[(kind,api)for kind in['zero','normal','outliers','boundary_probes','fresh_normal']for api in['pybind','cabi']]
  assert all(x['NaN_output_overwritten']for x in muts);count['mutations']+=10;oracles +=[x['oracle']for x in muts]
  f1=rows[10:43];med=[statistics.median(x['us']for x in f1[i*11:(i+1)*11])for i in range(3)]
  pairs=[{r['arm']:r['us']for r in rows[43+i*2:45+i*2]}for i in range(24)];assert pairs==cell['paired']
  deltas=[x['pybind']-x['cabi']for x in pairs];paired=statistics.median(deltas);avg=(med[0]+med[2])/2-med[1];con=min(med[0],med[2])-med[1]
  assert paired==cell['contrasts']['paired_median_control_minus_new_us']and avg==cell['contrasts']['F1_endpoint_average_control_minus_new_us']
  contrasts.append({'native_arm':arm,'M':m,'paired_pybind_minus_cabi_us':paired,'F1_endpoint_average_pybind_minus_cabi_us':avg,'F1_conservative_pybind_minus_cabi_us':con,'pair_wins_ties_losses':[sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)],'all_signed_deltas_us':deltas,'claim':'Same kernel/interface replay variability, not a native/score gain.'})
assert all_expected==journal
for o in oracles:assert o['current_X']=='EXACT'and 0<=o['FP64_L2']<=.004 and 0<=o['worst_row']<=.02
life=data['lifetime_probe'];assert life=={'strong_owners_after_caller_refs_dropped':True,'nondefault_capture_and_different_replay_streams':True,'fresh_current_X_own_output_checks':2,'graph_reset_before_owner_release':True,'weak_owners_released_after_drain':True}
assert '-90'in data['ABI_errors']['invalid_native_arm'];assert data['cleanup']==[]
post=data['post_scope_cleanup'];assert post['allocated']==post['reserved']==0 and post['clear_error']is None and post['after_run_scope_and_drain'];assert post['blas_map_entries_workspace_bytes_after']==[1,33554432,0]
external=I['cleanup_observation'];assert external['terminal_exit']==0 and external['cleanup_failed']is False and external['gpu_memory_mib']==0 and not external['compute_pids']and not external['guard_pids']and not external['container_exists']and not external['containers']
stages=data['build']['stages'];assert [x['name']for x in stages]==['native_compile','host_link']and all(x['exit']==0 for x in stages)
report={'state':'VALID_THIN_ABI_SAME_KERNEL_TRANSPORT_DIAGNOSTIC','job_id':I['job_id'],'recommended_disposition':'VALID_NO_GAIN','hardware':data['hardware'],'counts':count,'lifetime_probe':life,'build_seconds':{'Torch_import':data['cost']['torch_import_seconds'],'collected_PyBind_import':data['cost']['prior_pybind_import_seconds'],'thin_native_compile':stages[0]['seconds'],'thin_link':stages[1]['seconds'],'thin_compile_link_sum':sum(x['seconds']for x in stages),'thin_dlopen':data['cost']['thin_dlopen_seconds'],'driver_total':data['cost']['run_elapsed_seconds'],'prior_PyBind_compile_and_load_descriptive_only':data['build']['prior_compile_and_load_seconds_descriptive_only'],'paired_build_savings':None},'native_identity':identity,'attributes':data['raw_attributes'],'stream_scope':'Explicit nondefault capture handles and differing replay handles recorded for all4cells; passed CABIhandle assertion and fresh-X graph replay/mutations/bitwise oracles jointly validate capture semantics for these cells.','errors':data['ABI_errors'],'cleanup':{'internal':post,'external':external},'max_saved_FP64_L2':max(x['FP64_L2']for x in oracles),'max_saved_worst_row':max(x['worst_row']for x in oracles),'all_signed_interface_timing':contrasts,'verified_artifact_hash_count':len(I['files']),'verified_dispatch_source_pin_count':len(I['dispatch']['source_pins']),'reuse_recommendation':'Reuse-ready seam for a next bounded fixed-kernel proposal with explicit parameters/current stream, strong graph-owner lifetime, pre-capture attrs, exact version-matched cleanup bridge and existing numerical/terminal/drain gates. No full policy migration or generic-kernel certification.','limits':['No speedup or official objective claim from identical-kernel event differences.','New compile/link1.658s is observed; old42.186s is a different trial lacking phase attribution. No paired40s savings claim.','Saved successful GPU assertions/source bindings reviewed; output tensor bytes were not separately archived for independent CPU numerical recomputation.','RawABI pointers require this validated Python ownership/metadata layer. Do not bypass it or infer other kernel structs, cluster launch protocols, images or architectures validated.','No unsafe asynchronous fault was injected; immediate invalid argument and retained synchronization error gates are the runtime error coverage.','Future kernel bodies/codegen/launch shape need their own ordinary focused validation; this result does not authorize new native designs.'],'files':{'inventory':{'path':str((C/'inventory.json').relative_to(R)),'sha256':sha(C/'inventory.json')},'result':{'path':str(P.relative_to(R)),'sha256':sha(P)},'thin_SO':{'path':str((P.parent/'thin.so').relative_to(R)),'sha256':sha(P.parent/'thin.so')}},'GPU_API_SSH_queue_git_actions':0,'contract':'proj_rsi/ITERATION.md'}
(D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
done={'job_id':I['job_id'],'recommended_state':'DONE','recommended_disposition':'VALID_NO_GAIN','scope':'Successful same-native-kernel PyBind/C ABI validation and measured thin build cost; not a native or official speedup.','observed_state':data['state'],'analysis_path':str((D/'analysis.json').relative_to(R)),'analysis_sha256':sha(D/'analysis.json'),'inventory_sha256':sha(C/'inventory.json'),'result_sha256':sha(P),'checks':count,'lifetime_probe':life,'native_SASS_attrs_exact':True,'thin_compile_link_seconds':sum(x['seconds']for x in stages),'driver_seconds':data['cost']['run_elapsed_seconds'],'internal_external_cleanup_valid':True,'source_seam_reuse_ready_scoped':True,'build_savings_seconds':None,'no_kernel_speed_gain_claim':True,'no_queue_mutation':True};(D/'done-evidence.json').write_text(json.dumps(done,indent=2)+'\n');print(json.dumps(done,indent=2))
