"""Analyze only locally collected artifacts; preserve setup failures and direct/profile distinction."""
import hashlib,json,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2];JOB='research.moe.original.marlin.tiles.01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def tile(name):
 match=re.search(r'::marlin_moe::Marlin<([^>]+)>',name)
 if not match:return None
 a=[x.strip()for x in match[1].split(',')]
 assert len(a)==13,a
 return {'threads':int(a[3]),'thread_m_blocks':int(a[4]),'thread_n':16*int(a[5]),'thread_k':16*int(a[6]),'blockM8':a[7]=='true','stages':int(a[8]),'group_blocks':int(a[9]),'EP':a[11]=='true','bias':a[12]=='true'}
def main():
 base=ROOT/'results/proj-rsi/scheduler/artifacts'/JOB
 if not base.exists():print('NOT_COLLECTED');return
 collections=sorted(base.glob('collection-*'));assert collections
 C=collections[-1];I=json.loads((C/'inventory.json').read_text())
 for rel,h in I['files'].items():assert sha(C/rel)==h,rel
 candidates=list(C.glob('logs/scheduler/moe-original-marlin-tiles-01/result.json'))
 evidence=[pin(C/'inventory.json')]
 host=C/'logs/scheduler'/f'{JOB}.host.log'
 if host.exists():evidence.append(pin(host))
 r=json.loads(candidates[0].read_text())if candidates else {};evidence.extend(pin(p)for p in candidates)
 out={'job_id':JOB,'source_state':r.get('state','NO_RESULT_JSON'),'phase':r.get('phase'),'evidence':evidence,'inventory_file_hashes_validated':len(I['files']),'terminal_exit':I.get('terminal_exit'),'cleanup':{k:v for k,v in I.get('cleanup_observation',{}).items()if k!='host_log_tail'},'new_GPU_API_queue_calls':0}
 if r.get('state')!='VALID_M8_ORIGINAL_MARLIN_TILE_SCREEN_ROOT_REVIEW_REQUIRED':
  out.update(state='SETUP_OR_RUNTIME_FAILURE_PRESERVED_NO_PERFORMANCE_ADMISSION',error=r.get('error'),direct_timings_completed=r.get('validated_direct_timings_completed',False),raw=r.get('raw',[]),jit=r.get('jit',[]),compile_wall_seconds=r.get('compile_wall_seconds'))
  if host.exists():out['host_log_tail']=host.read_text()[-12000:]
 else:
  assert not I.get('missing')and I['terminal_exit']==0 and not r['cleanup']
  cleanup=I['cleanup_observation'];assert not cleanup['cleanup_failed']and not cleanup['container_exists']and not cleanup['compute_pids']and not cleanup['containers']and not cleanup['guard_pids']and cleanup['gpu_memory_mib']==0
  assert len(r['jit'])==1 and sha(candidates[0].parent/'two_tiles.so')==r['jit'][0]['so_sha256']
  assert sum(x['kind'].startswith('own_')for x in r['correctness'])==8
  for rel,h in I['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
  assert len(r['raw'])==61 and len(r['paired'])==12 and r['validated_direct_timings_completed']
  assert all(q.get('pass',q.get('exact_zero',False))for q in r['correctness'])
  raw=r['raw'];pairs=r['paired'];comparisons={}
  for a,b in(('anchor','default_shim'),('anchor','contrast'),('default_shim','contrast')):
   d=[x[a]-x[b]for x in pairs];summary={'median_us':statistics.median(d),'median_ratio':statistics.median(x[a]/x[b]for x in pairs),'b_wins':sum(x>0 for x in d)};assert summary==r['comparisons'][a+'_minus_'+b]
   comparisons[a+'_minus_'+b]=dict(summary,win_tie_loss=[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)])
  for i,order in enumerate(r['pair_orders']):
   group=[x for x in raw if x['kind']=='paired'][3*i:3*i+3];assert[x['arm']for x in group]==order and len({x['seed']for x in group})==1 and len({json.dumps(x['routing'],sort_keys=True)for x in group})==1
  f=r['F1_medians_us'];assert all(statistics.median(r['F1_raw_us'][k])==v for k,v in f.items())
  signed={a+'_minus_contrast':{'before_us':f[a+str(i)]-f['contrast2'],'after_us':f[a+str(j)]-f['contrast2']}for a,i,j in(('anchor',0,4),('default_shim',1,3))}
  receipts=r['capture_receipts'];default=receipts['default_shim']['selected'];other=receipts['contrast']['selected'];assert default==receipts['default_shim']['default']and other!=receipts['contrast']['default']and r['default_receipt']['before_any_forced_launch']and r['default_receipt']['default_output_bitwise_equal_original']
  profile={}
  for label,p in r['profiles'].items():
   kernels=p.get('kernels',[]);physical=[dict(tile=tile(x['name']),duration_us=x['duration_us'],resources=x['args'])for x in kernels if tile(x['name'])]
   profile[label]={'status':p['status'],'exclusive_kernel_sum_us':sum(x['duration_us']for x in kernels),'Marlin_calls':physical,'not_clean_latency':True}
   if p['status']=='COMPLETE':assert len(physical)==2
  original_profile_matches=None
  if profile.get('anchor',{}).get('status')=='COMPLETE':original_profile_matches=all(profile['anchor']['Marlin_calls'][0]['tile'][k]==default[k]for k in('threads','thread_n','thread_k'))
  out.update(state='VALID_COLLECTED_M8_RESULT_CPU_RECOMPUTED',disposition='VALID_REGRESSION',disposition_reason='Opposite existing tile loses every paired comparison versus original/default shim and both F1 brackets; retain original, no promotion. This is a tile-version result, not family rejection.',runtime_source_pins_validated=len(I['dispatch']['source_pins']),runtime=r['runtime'],compile_wall_seconds=r['compile_wall_seconds'],prepare_seconds=r['prepare_seconds'],default_receipt=r['default_receipt'],capture_receipts=receipts,paired=comparisons,F1_medians_us=f,F1_control_minus_contrast=signed,correctness_checks=len(r['correctness']),raw_clean_calls=61,own_timed_terminal_checks=8,profiling=profile,original_profile_gateup_matches_default_receipt=original_profile_matches,limits=['M8 normal EP8 only; no full14/model/official/3x result.','Profile sums are exclusive captured kernel durations, not clean event latency.','Actual original/default module differences are an explicit control; preserve all signed pair/F1 estimates.'])
 (D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('host_log_tail','profiling','raw')},indent=2))
if __name__=='__main__':main()
