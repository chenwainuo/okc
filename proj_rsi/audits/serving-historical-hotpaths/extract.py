"""Read retained JSON/CSV only; no torch, CUDA, remote, API or queue actions."""
import collections,csv,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def utc(seconds):return datetime.datetime.fromtimestamp(seconds,datetime.timezone.utc).isoformat()
def contains(span,event):return span['ts']-.002<=event['ts'] and event['ts']+event['dur']<=span['ts']+span['dur']+.002
def kernels(trace):return [e for e in trace['traceEvents']if e.get('cat')=='kernel']
def aggregate(events):
 grouped=collections.defaultdict(lambda:{'count':0,'GPU_duration_us':0.})
 for e in events:
  row=grouped[e['name']];row['count']+=1;row['GPU_duration_us']+=e['dur']
 return sorted([dict(name=name,**row)for name,row in grouped.items()],key=lambda x:-x['GPU_duration_us'])
def component(name,ordinal):
 if 'marlin_moe::Marlin<'in name:return ('gate_up' if ordinal==0 else 'down')
 if 'moe_align'in name or 'count_and_sort'in name:return 'alignment'
 if 'act_and_mul'in name:return 'swiglu'
 if 'post_expert_norm_weighted_sum'in name:return 'post_norm_router_weighted_sum'
 if 'FillFunctor<int>'in name:return 'workspace_zero'
 if 'index_'in name or'direct_copy'in name:return 'route_map_cast'
 return 'other'
def main():
 report_path=ROOT/'reports/2026-10-24-profiled.json';run_path=ROOT/'results/serving-profiled.json'
 report=json.loads(report_path.read_text());run=json.loads(run_path.read_text())
 for key in ('experiment_commit','candidate_sha256','baseline_sha256','serving_diff_sha256','gpu','torch','cuda'):assert report[key]==run[key]
 assert hashlib.sha256(run['serving_diff'].encode()).hexdigest()==run['serving_diff_sha256']
 result={'analysis_date':'2026-10-27','state':'HISTORICAL_RETAINED_CAPTURE_REEXTRACTION_NOT_FRESH_GPU_WORK','scope':'single-GPU synthetic local EP8 full-expert _forward_marlin; not full model serving/projection/official; six shapes, not original14','metadata':{k:report[k]for k in ('experiment_commit','gpu','torch','cuda','seed','pairs','candidate_sha256','baseline_sha256','serving_diff_sha256','measurement')},'source_identity_limit':'Adapter hashes and exact retained serving diff; actual classfile SHA/current861333f identity is not asserted. Defaults upstreama55 stockcontrol, not861333f anchor.','inputs':{str(p.relative_to(ROOT)):sha(p)for p in (report_path,run_path)},'graph_method':'Kernel dur sums grouped by actual GPU replay annotation; graphid0 cache flush excluded; five replays. First/second Marlin occurrence maps gate_up/down, corroborated by eager labels. Shares divide exclusive kernel sum, not wall/model time.','traces':[],'down_NCU':[]}
 for bs in (1,8,16,32,128,256):
  for arm in ('baseline','candidate'):
   path=ROOT/f'results/serving-profiled-profiles/bs{bs}-{arm}.trace.json';trace=json.loads(path.read_text());result['inputs'][str(path.relative_to(ROOT))]=sha(path)
   selected=[e for e in kernels(trace)if e.get('args',{}).get('graph id',0)]
   spans=sorted([e for e in trace['traceEvents']if e.get('cat')=='gpu_user_annotation'and e['name']==arm+'_graph_replay'],key=lambda e:e['ts']);assert len(spans)==5
   grouped=collections.defaultdict(lambda:{'count':0,'GPU_duration_us':0.});covered=[]
   for span in spans:
    events=sorted([e for e in selected if contains(span,e)],key=lambda e:e['ts']);ordinal=0
    mm=[i for i,e in enumerate(events)if 'marlin_moe::Marlin<'in e['name']]
    act=[i for i,e in enumerate(events)if 'act_and_mul'in e['name']]
    assert len(mm)==2 and len(act)==1 and mm[0]<act[0]<mm[1],'source up→activation→down order not established'
    assert len({e['args']['stream']for e in events})==1,'replay attribution assumes source-ordered single stream'
    for event in events:
     tag=component(event['name'],ordinal)
     if 'marlin_moe::Marlin<'in event['name']:ordinal+=1
     row=grouped[tag];row['count']+=1;row['GPU_duration_us']+=event['dur'];covered.append(id(event))
    assert ordinal==2,'expected exactly two source-bound Marlin calls per expert replay'
   assert len(set(covered))==len(covered)==len(selected)
   total=sum(e['dur']for e in selected)
   stages=sorted([dict(stage=stage,**row,mean_us_per_profiled_replay=row['GPU_duration_us']/5,share_percent_of_exclusive_GPU_kernel_sum=100*row['GPU_duration_us']/total)for stage,row in grouped.items()],key=lambda row:-row['GPU_duration_us'])
   kernel_rows=aggregate(selected)
   for row in kernel_rows:row.update(mean_us_per_profiled_replay=row['GPU_duration_us']/5,share_percent_of_exclusive_GPU_kernel_sum=100*row['GPU_duration_us']/total)
   offset=trace['baseTimeNanoseconds']/1e9
   eager_path=path.with_name(f'bs{bs}-{arm}-eager-stages.trace.json');eager=json.loads(eager_path.read_text());result['inputs'][str(eager_path.relative_to(ROOT))]=sha(eager_path)
   eager_stages=collections.defaultdict(lambda:{'count':0,'GPU_span_us':0.})
   for event in eager['traceEvents']:
    if event.get('cat')=='gpu_user_annotation'and event['name'].startswith('okc_stage/'):
     item=eager_stages[event['name']];item['count']+=1;item['GPU_span_us']+=event['dur']
   for row in eager_stages.values():assert row['count']==3;row['mean_GPU_span_us_per_eager_call']=row['GPU_span_us']/3
   result['traces'].append({'bs':bs,'arm':arm,'path':str(path.relative_to(ROOT)),'capture_UTC_first_kernel':utc(offset+min(e['ts']for e in selected)/1e6),'capture_UTC_last_kernel_end':utc(offset+max(e['ts']+e['dur']for e in selected)/1e6),'GPU':trace['deviceProperties'],'cuda_driver_version':trace.get('cuda_driver_version'),'cuda_runtime_version':trace.get('cuda_runtime_version'),'profiled_replays':5,'kernel_events':len(selected),'exclusive_kernel_sum_us':total,'mean_exclusive_kernel_sum_us_per_replay':total/5,'GPU_replay_span_us_mean':sum(s['dur']for s in spans)/5,'ranked_stage_components':stages,'ranked_physical_kernel_symbols':kernel_rows,'eager_calls':3,'eager_stage_GPU_spans':dict(eager_stages),'eager_kernel_events':len(kernels(eager))})
 for name,report_name in [('wgmma-down-profile-v1','2026-10-25-wgmma-down-profile.json'),('wgmma-coalesced-profile-v3','2026-10-25-wgmma-coalesced-profile.json'),('wgmma-pipeline-profile-v4','2026-10-25-wgmma-pipeline-profile.json')]:
  folder=ROOT/'results/rsi'/name;rp=ROOT/'reports'/report_name;j=json.loads(rp.read_text());setup_path=folder/'setup.json';setup=json.loads(setup_path.read_text());csv_path=folder/'down-m128.csv';raw=list(csv.reader(csv_path.open()));assert len(raw)==3
  d=dict(zip(raw[0],raw[2]));units=dict(zip(raw[0],raw[1]));value=float(d['gpu__time_duration.sum'].replace(',',''));unit=units['gpu__time_duration.sum'];assert unit in ('ms','us');duration_us=value*(1000 if unit=='ms' else 1)
  lease=j.get('lease')
  if not lease:
   lp=ROOT/'results/rsi/probes/pipeline-profile-v4-lease.json';lease=json.loads(lp.read_text());result['inputs'][str(lp.relative_to(ROOT))]=sha(lp)
  assert lease['status']=='complete'
  for p in (rp,setup_path,csv_path):result['inputs'][str(p.relative_to(ROOT))]=sha(p)
  ncu=folder/'down-m128.ncu-rep';result['inputs'][str(ncu.relative_to(ROOT))]=sha(ncu)
  result['down_NCU'].append({'version':name,'scope':'one logical M128 down projection; NCU kernel replay/cache-control-all; no clean benchmark or whole-MoE contribution share','reservation_start_UTC':utc(lease['started_at']),'exact_GPU_capture_timestamp':'not recorded in CSV/setup; reservation time is not capture timestamp','hardware':d['device__attribute_display_name'],'kernel_name':d['Kernel Name'],'logical_profile_rows':1,'NCU_replay_passes':float(d['profiler__replayer_passes'].replace(',','')),'raw_gpu__time_duration.sum':{'value':value,'unit':unit,'normalized_microseconds':duration_us,'limit':'Profiler-reported metric only; do not divide by replaypasses or infer causal speedup across unmatched fixtures'},'bundle_sha256':setup['bundle_sha256'],'source_pins':setup['source_pins'],'weight_seed':setup['weight_seed'],'input_seed':setup['input_seed'],'routing':{k:setup[k]for k in ('local_route_rows','total_ctas','useful_ctas','all_padding_ctas','nonlocal_or_beyond_post_ctas')},'selected_counters':{k:{'value':d[k],'unit':units[k]}for k in ('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_barrier_per_warp_active.pct','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')},'raw_ncu_retained':True})
 result['GPU_calls_by_extractor']=0;result['torch_imported']=False
 (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'state':result['state'],'trace_captures':len(result['traces']),'down_NCU_captures':len(result['down_NCU']),'inputs':len(result['inputs']),'GPU_calls':0},indent=2))
if __name__=='__main__':main()
