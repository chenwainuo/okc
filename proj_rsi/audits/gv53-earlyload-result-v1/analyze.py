"""CPU-only exact thin-ABI earlyload collection validation; no GPU/API/queue calls."""
from pathlib import Path
import hashlib,json,math,re,statistics,subprocess
D=Path(__file__).resolve().parent;R=D.parents[2];C=R/'results/proj-rsi/scheduler/artifacts/research.gv53.earlyload.fixed.01/collection-1793143210466227704';P=C/'logs/scheduler/gv53-earlyload-fixed-01/run';Q=R/'proj_rsi/candidates/gv53_earlyload_v1';LABELS=('control','earlyload');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(text):
 return {c.splitlines()[0].strip():re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',c)for c in text.split('Function : ')[1:]}
def med(x):return statistics.median(x)
def stats(x):return {'median':med(x),'minimum':min(x),'maximum':max(x),'mean':statistics.mean(x)}
def main():
 i=json.loads((C/'inventory.json').read_text());assert i['terminal_exit']==0 and not i['missing']
 for path,h in i['files'].items():assert sha(C/path)==h,path
 for path,h in i['dispatch']['source_pins'].items():assert sha(R/path)==h,path
 r=json.loads((P/'result.json').read_text());assert r['state']=='ONE_FIXED_M8_CELL_COMPLETE_ROOT_REVIEW_REQUIRED'and r['phase']=='complete'and not r['cleanup']
 assert r['source_sha256']==sha(Q/'pilot.cu') and sha(Q/'submission.py')=='81fdc825bc21a568db8082873892982fb16eff5fbbc894baed2211a58aef1791'and r['control_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert sha(P/'actual.cu')==r['build']['CUDA_sha256']and sha(P/'actual.so')==r['build']['SO_sha256'];assert(P/'actual.cu').read_text()==(Q/'pilot.cu').read_text()+'\n'+(Q/'bridge.cuh').read_text()
 dump=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(P/'actual.so')],text=True);actual=rows(dump);assert actual==rows((P/'actual.sass').read_text())==rows((Q/'thin.sass').read_text())
 resource=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(P/'actual.so')],text=True);(D/'actual.resources').write_text(resource)
 for stage in r['build']['stages']:assert stage['returncode']==0
 assert r['build']['CXX_Torch_compile_calls']==0 and r['build']['native_compile_calls']==1 and r['build']['actual_fullPC_predicate_encoding_equals_pilot']
 assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'in(P/'native_compile.stderr').read_text()
 for index,arm in enumerate(LABELS):
  a=r['raw_attributes'][arm];assert a['registers']==64 and a['local']==a['user_static']==0 and a['max_dynamic']==17408 and a['max_threads']==1024 and a['occupancy']==1 and a['arm_echo']==index
 cell=r['fixed_cells']['8'];assert cell['config']=={'layer':'pooler','m':8,'n':4096,'k':8192}and cell['state']=='VALID_FIXED_CELL_NO_POLICY_OR_OFFICIAL_ADMISSION'and cell['cleanup']=='RESET_AFTER_DRAIN'
 raw=cell['raw'];assert len(raw)==91 and sum(x['mode']!='warmup'for x in raw)==81 and all(math.isfinite(x['us'])and x['us']>0 for x in raw)
 assert[json.loads(x)for x in(P/'raw.jsonl').read_text().splitlines()]==[dict(x,m=8)for x in raw]
 for x,(jb,je)in zip(raw,cell['host_journal_spans_ns']):assert x['host_before_start_ns']<=x['host_after_end_enqueue_ns']<=x['host_after_sync_ns']<=jb<=je
 assert len(cell['host_journal_spans_ns'])==91
 for arm in LABELS:
  v=cell['validation'][arm];assert len(v['task'])==3 and v['basis']=='EXACT'and len(v['basis_columns'])==8
 for phase,end in [('F1',43),('paired',91)]:
  t=cell['terminals'][phase];assert len(t)==2 and {x['arm']for x in t}==set(LABELS)
  for x in t:assert x['before_any_replay_or_mutation']and x['oracle']['current_X']=='EXACT'and x['seed']==next(y['seed']for y in reversed(raw[:end])if y['arm']==x['arm']and y['mode']!='warmup')
 assert len(cell['mutations'])==10 and all(x['NaN_output_overwritten']for x in cell['mutations'])
 assert r['escape_contract']['both_arms_same_original_bank']and r['escape_contract']['transport_repack_calls']==0 and len(r['escape_contract']['SHA256'])==64 and r['pack']['actual_W_bit_reconstruction']=='EXACT'
 cl=r['post_scope_cleanup'];assert cl['allocated']==cl['reserved']==0 and cl['after_run_scope_and_drain']and cl['clear_error']is None and cl['blas_map_entries_workspace_bytes_after'][2]==0
 ex=i['cleanup_observation'];assert not ex['cleanup_failed']and not ex['compute_pids']and not ex['containers']and not ex['guard_pids']and not ex['container_exists']and ex['gpu_memory_mib']==0
 f1=raw[10:43];fm=[med(x['us']for x in f1[j*11:(j+1)*11])for j in range(3)];assert cell['F1_medians_us']==dict(zip(('control_before','earlyload','control_after'),fm))
 paired=[]
 for j in range(24):
  pair=raw[43+2*j:45+2*j];assert[x['arm']for x in pair]==list(LABELS if j%2==0 else LABELS[::-1])and len({x['seed']for x in pair})==1
  paired.append({x['arm']:x['us']for x in pair})
 assert paired==cell['paired'];delta=[x['control']-x['earlyload']for x in paired];point=med(delta);assert point==cell['contrasts']['paired_median_control_minus_new_us']
 meanF1=(fm[0]+fm[2])/2-fm[1];assert meanF1==cell['contrasts']['F1_endpoint_average_control_minus_new_us']
 host={}
 for tag,subset in [('all_clean',raw[10:]),('F1_control_before',f1[:11]),('F1_new',f1[11:22]),('F1_control_after',f1[22:])]+[(arm,[x for x in raw[43:]if x['arm']==arm])for arm in LABELS]:
  host[tag]={'submission_span_us':stats([(x['host_after_end_enqueue_ns']-x['host_before_start_ns'])/1000 for x in subset]),'end_enqueue_to_sync_us':stats([(x['host_after_sync_ns']-x['host_after_end_enqueue_ns'])/1000 for x in subset])}
 out={'job_id':'research.gv53.earlyload.fixed.01','state':'CPU_VALIDATED_ACTUAL_EARLYLOAD_COMPLETE','recommended_DONE_disposition':'VALID_GAIN'if point>0 else'VALID_REGRESSION'if point<0 else'VALID_NO_GAIN','native_pilot_sha256':r['source_sha256'],'submission_sha256':sha(Q/'submission.py'),'control_sha256':r['control_sha256'],'artifact_hashes_verified':len(i['files']),'source_pins_verified':len(i['dispatch']['source_pins']),'actual_CU_sha256':sha(P/'actual.cu'),'actual_SO_sha256':sha(P/'actual.so'),'actual_native_full_PC_predicate_128bitencoding_equals_CPU':True,'native_instruction_counts':{k:len(v)for k,v in actual.items()},'actual_resources':r['raw_attributes'],'hardware':r['hardware'],'pack':r['pack'],'escape_contract':r['escape_contract'],'timing':{'sign':'control_minus_earlyload_us,positivefavorsnew','paired_median_us':point,'paired_mean_us':statistics.mean(delta),'paired_median_control_over_new':med(x['control']/x['earlyload']for x in paired),'win_tie_loss':[sum(x>0 for x in delta),sum(x==0 for x in delta),sum(x<0 for x in delta)],'paired_each_delta_us':delta,'control_first_paired_median_us':med(delta[::2]),'new_first_paired_median_us':med(delta[1::2]),'F1_medians_us':cell['F1_medians_us'],'F1_before_minus_new_us':fm[0]-fm[1],'F1_after_minus_new_us':fm[2]-fm[1],'F1_endpoint_average_us':meanF1,'F1_control_drift_us':fm[2]-fm[0],'F1_matched_seed_endpoint_medians_us':[med(f1[j]['us']-f1[11+j]['us']for j in range(11)),med(f1[22+j]['us']-f1[11+j]['us']for j in range(11))]},'host_timestamps':host,'journal_spans_us':stats([(e-b)/1000 for b,e in cell['host_journal_spans_ns']]),'F1_first_start_to_last_sync_seconds':(f1[-1]['host_after_sync_ns']-f1[0]['host_before_start_ns'])/1e9,'build':r['build'],'cost':r['cost'],'validity':{'journal_rows':91,'clean_calls':81,'warmups':10,'initial_task_plus_FP64_oracles':6,'basis_columns_per_arm':8,'own_terminal_oracles':4,'mutations':10,'current_X_bytecheck':'PASS beforeanyreplay/mutation','same_original_escape_bank':'PASS','thin_currentstream_strongowners':'Source-bound success; owners retiredonlyafterallgraphsdrain/reset','post_scope_cleanup':cl,'external_cleanup':ex},'evidence':{str(x.relative_to(R)):sha(x)for x in[C/'inventory.json',P/'result.json',P/'raw.jsonl',P/'actual.sass',P/'native_compile.stderr']},'limits':['OnefreshWseed/fixedM8H100cell; nofullpolicy/officialH200gainclaim.','F1andpairedusedifferentXbanks; everylossandorderestimatepreserved.','EarliernativeissuePROVEN; hardwarewaitreductioncauseUNMEASURED here, priorNCUcountercontextnotthisbinary.','Reusableevents/hosttimestampsaremethodchange; same-windowarmsidenticalmethod, cross-jobcausalityunproven.','Noactualdeviceclocktelemetrythiscleanrun; timestampgapscannotdetermineeventGPUqueueunderflow.']}
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in['recommended_DONE_disposition','artifact_hashes_verified','source_pins_verified','timing','host_timestamps','journal_spans_us','F1_first_start_to_last_sync_seconds','cost']},indent=2))
if __name__=='__main__':main()
