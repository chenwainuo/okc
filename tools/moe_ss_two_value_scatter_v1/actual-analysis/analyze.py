"""Author CPU readout of the single three-arm scatter diagnostic; no new gates."""
from pathlib import Path
import argparse,hashlib,json,re,statistics
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result',type=Path);args=p.parse_args();rp=args.result.resolve();a=json.loads(rp.read_text());folder=rp.parent;invpath=rp.parents[3]/'inventory.json';inventory=json.loads(invpath.read_text())if invpath.exists()else{}
 labels=('v1_full_lut','v2_rolled_down','two_value_scatter');pairs={};comparisons={};workload=a.get('F1_frozen_workload');f1=a.get('F1_medians_us',{})
 for x in a.get('raw',[]):
  if x.get('kind')=='paired':pairs.setdefault(x['seed'],{})[x['arm']]=x
 rows=[]
 for seed,arms in sorted(pairs.items()):
  row={'seed':seed,'arms':{k:v['us']for k,v in arms.items()},'routing_equal':len(arms)==3 and all(v['routing']==next(iter(arms.values()))['routing']for v in arms.values()),'routing':next(iter(arms.values()))['routing']};rows.append(row)
 for control,variant in[(labels[0],labels[2]),(labels[1],labels[2]),(labels[0],labels[1])]:
  complete=[r for r in rows if control in r['arms']and variant in r['arms']];deltas=[r['arms'][control]-r['arms'][variant]for r in complete]
  comparisons[control+'_versus_'+variant]={'complete_pairs':len(complete),'signed_control_minus_variant_us':deltas,'median_control_minus_variant_us':statistics.median(deltas)if deltas else None,'median_control_over_variant':statistics.median(r['arms'][control]/r['arms'][variant]for r in complete)if complete else None,'variant_wins':sum(x>0 for x in deltas),'variant_losses':sum(x<0 for x in deltas)}
 f1result={'all_blocks':f1,'frozen_workload':workload}
 if len(f1)==5:
  new=f1['two_value_scatter2'];f1result.update(new_us=new,v1_before_minus_new_us=f1['v1_full_lut0']-new,v1_after_minus_new_us=f1['v1_full_lut4']-new,v2_before_minus_new_us=f1['v2_rolled_down1']-new,v2_after_minus_new_us=f1['v2_rolled_down3']-new)
 stage=[x for x in a.get('correctness',[])if 'bitwise_equal'in x];mixed=[x for x in a.get('correctness',[])if 'bitwise_equal_v1_full_serving'in x]
 report={'state':a['state'],'result_path':str(rp.relative_to(ROOT)),'result_sha256':sha(rp),'wall_seconds':a.get('elapsed_seconds'),'raw_calls':len(a.get('raw',[])),'F1':f1result,'comparisons':comparisons,'per_seed':rows,'actual_JIT':a.get('jit'),'correctness':{'total_records':len(a.get('correctness',[])),'stage_bitwise_records':len(stage),'mixed_full_records':len(mixed),'all_stage_bitwise_pass':all(x['bitwise_equal']for x in stage),'all_mixed_full_pass':all(x['reference_pass']and x['bitwise_equal_v1_full_serving']for x in mixed),'mixed_records':mixed,'validated_direct_completion':a.get('validated_direct_timings_completed',False)},'cleanup_observation':inventory.get('cleanup_observation'),'terminal_exit':inventory.get('terminal_exit'),'error':a.get('error'),'scope':'one fixedM8threearmdowndiagnostic; numericalmixedfullchecks only, no mixedfullspeed or broadfamilyclaim','iteration':'Root retainsbestmeasuredarm andcontinuesincumbentMarlin/projectionpriority; no broadSSrewrite/GPUaction byauthor'}
 if(folder/'actual_scatter.sass').exists():
  f=next(f for f in(folder/'actual_scatter.sass').read_text().split('Function : ')[1:]if 'projectILi8ELi1E'in f.splitlines()[0]);ins=re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)
  report['actual_SASS']={'instructions':len(ins),'LDS_U16':sum('LDS.U16'in s for _,s in ins),'LOP3':sum('LOP3'in s for _,s in ins),'HGMMA':sum('HGMMA'in s for _,s in ins),'source_sha256':sha(folder/'actual_scatter.cu'),'SO_sha256':sha(folder/'actual_scatter.so')}
 report['decision']='Retain actualv1LUTdown. Packed two-value scatter partiallyrescuesfailedv2 but loses8/8 tobestv1. StopbroadSSrewrite; parentprioritizesincumbentMarlin/projection.'
 report['mechanism_limit']='ActualtwoindependentLDS thenpackedSTS32 isrealized at48regs/5CTAsSM. Itsbenefitoverfailedv2 supports store/dependencycost beingmaterial, but neitheroccupancy norstorewidthalone matchesv1 eight-lookupparallelism. No uniquecyclecausalclaim withoutcounters.'
 (D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['state','wall_seconds','F1','comparisons','actual_JIT','correctness']},indent=2))
