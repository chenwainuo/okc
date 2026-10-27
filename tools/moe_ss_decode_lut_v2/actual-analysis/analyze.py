"""CPU-only v2 down measurement readout; retains wins, losses and partial failures."""
from pathlib import Path
import argparse,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(result):
 a=json.loads(result.read_text());folder=result.parent
 before=json.loads((ROOT/'results/proj-rsi/scheduler/artifacts/research.moe.ss.decode.lut.stage.01/collection-1793130937294077992/logs/scheduler/moe-ss-decode-lut-stage-01/result.json').read_text());old=next(x for x in before['jit'][0]['resources']if x['DIV']==1)
 r={'state':a['state'],'result':str(result.relative_to(ROOT)),'result_sha256':sha(result),'wall_seconds':a.get('elapsed_seconds'),'error':a.get('error'),'raw_calls':len(a.get('raw',[])),'validated_direct_timings_completed':a.get('validated_direct_timings_completed',False),'correctness_records':len(a.get('correctness',[])),'stage_bitwise_checks':[x for x in a.get('correctness',[])if 'bitwise_equal'in x],'mixed_full_checks':[x for x in a.get('correctness',[])if 'bitwise_equal_v1_full_serving'in x],'F1_frozen_workload':a.get('F1_frozen_workload'),'F1_medians_us':a.get('F1_medians_us'),'profiles':a.get('profiles'),'v1_resources':old,'actual_v2_resources':None,'per_seed':[]}
 if a.get('jit'):
  r['actual_v2_resources']=a['jit'][0]['resources'];r['JIT']=a['jit'][0]
 if a.get('F1_raw_us'):
  f=a['F1_medians_us'];left=f['v1_full_lut0'];mid=f['v2_rolled_down1'];right=f['v1_full_lut2']
  r['fixed_workload_F1']={'v1_before_us':left,'v2_us':mid,'v1_after_us':right,'before_minus_v2_us':left-mid,'after_minus_v2_us':right-mid,'v2_wins_both_endpoints':mid<left and mid<right,'scope':'same frozen seed7904 record, same private graph buffers;9calls total, descriptive bracket'}
  first=next(x['routing']for x in a['raw']if x.get('kind')=='F1_fixed_B9');assert all(x['routing']==first for x in a['raw']if x.get('kind')=='F1_fixed_B9')
  assert first==a['F1_frozen_workload']['routing'];r['fixed_workload_F1']['routing_record_identical']=True
 pairs={}
 for row in a.get('raw',[]):
  if row.get('kind')=='paired':pairs.setdefault(row['seed'],{})[row['arm']]=row
 for seed,arms in sorted(pairs.items()):
  if set(arms)!={'v1_full_lut','v2_rolled_down'}:continue
  p,q=arms['v1_full_lut'],arms['v2_rolled_down'];assert p['routing']==q['routing'];b=p['routing']['active_local_blocks'];new=r['actual_v2_resources'][0];sms=a['runtime']['SMs']
  r['per_seed'].append({'seed':seed,'active_blocks':b,'v1_us':p['us'],'v2_us':q['us'],'signed_v1_minus_v2_us':p['us']-q['us'],'v1_over_v2':p['us']/q['us'],'valid_CTAs':b*64,'idealized_v1_capacity_waves':math.ceil(b*64/(sms*old['occupancy'])),'idealized_v2_capacity_waves':math.ceil(b*64/(sms*new['occupancy'])),'counts':p['routing']['counts']})
 if r['per_seed']:
  deltas=[x['signed_v1_minus_v2_us']for x in r['per_seed']]
  r['paired']={'complete_pairs':len(deltas),'signed_deltas_us':deltas,'median_v1_minus_v2_us':statistics.median(deltas),'median_v1_over_v2':statistics.median(x['v1_over_v2']for x in r['per_seed']),'v2_wins':sum(d>0 for d in deltas),'v2_losses':sum(d<0 for d in deltas),'v1_median_us':statistics.median(x['v1_us']for x in r['per_seed']),'v2_median_us':statistics.median(x['v2_us']for x in r['per_seed'])}
 if(folder/'actual_rolled.sass').exists():
  rows=[]
  for f in(folder/'actual_rolled.sass').read_text().split('Function : ')[1:]:
   if '7projectILi'not in f.splitlines()[0]:continue
   ins=re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)
   rows.append({'function':f.splitlines()[0],'instructions':len(ins),'LOP3':sum('LOP3'in x for _,x in ins),'LDS_U16':sum('LDS.U16'in x for _,x in ins),'HGMMA':sum('HGMMA'in x for _,x in ins)})
  r['actual_SASS']=rows
 v1_path=ROOT/'results/proj-rsi/scheduler/artifacts/research.moe.ss.decode.lut.stage.01/collection-1793130937294077992/logs/scheduler/moe-ss-decode-lut-stage-01/actual_lut.sass'
 def first_schedule(path):
  f=next(f for f in path.read_text().split('Function : ')[1:]if 'projectILi8ELi1E'in f.splitlines()[0]);rows=[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)]
  first=next(i for i,(_,ins)in enumerate(rows)if 'LDS.U16'in ins)
  events=[{'pc':hex(pc),'instruction':ins}for pc,ins in rows[first:first+90]if 'LDS'in ins or 'STS'in ins]
  return events
 if(folder/'actual_rolled.sass').exists():
  r['producer_schedule_observation']={'v1_first_memory_instructions':first_schedule(v1_path),'v2_first_memory_instructions':first_schedule(folder/'actual_rolled.sass'),'observed':'v1firststep issues8LDS_U16 before4packedSTS32; v2alternates8LDS_U16/dependentSTS_U16 beforebackedge0x2440_to_0x1f80. Same BF16payload but8halfstoreinstructions versus4wordstores.','interpretation':'Lookup-memory independent issue and packed-store formation were traded away together with registers; loss cannot be attributed to occupancy alone. Scheduling/loop criticalpath is a hypothesis without counters; no generic rolling/family rejection.'}
 probe=ROOT/'serving/experiments/moe_rsi_20261027/ss_decode_lut_v2/offline_unroll2/codegen.json'
 if probe.exists():r['offline_unroll2']=json.loads(probe.read_text())
 r['decision']='Retain measured v1fullLUTup/down. Reject only this fused+rolleddown measuredconfiguration. Parent alreadyregistered pair2structuralnext; no duplicateGPU or broadsweep. Offlineunroll2doesnotrestorepackedstores/parallelLDSshape.'
 r['wave_limit']='Idealized ceil(validCTAs/(SMs*queriedactiveCTAsSM)); not measured scheduling or unique causal attribution.'
 r['scope_limit']='Measured down stage plus mixed full numerical checks; mixed full-serving speed and anchor comparison are not timed here.'
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result',type=Path);args=p.parse_args();r=analyze(args.result.resolve());(D/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in ['state','actual_v2_resources','fixed_workload_F1','paired','per_seed','mixed_full_checks']},indent=2))
