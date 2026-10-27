"""Actual exported native/DP PC/role accounting. Local CPU only, no NCU/SSH/CUDA."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CAPTURE=ROOT/'results/proj-rsi/scheduler/artifacts/native.profile.v1.01/collection-1793077258598870796'
EXPORTS=ROOT/'results/proj-rsi/scheduler/artifacts/native.profile.export.v1.01'
TOKENS={'DP_pack':'rx_pack','DP_main':'rq_px_dp','NX_main':'rq_tma_x'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(value):
 if value is None or str(value).strip().lower() in ('','-','n/a','nan','not available'):return None
 try:x=float(str(value).replace(',',''))
 except ValueError:return None
 return x if math.isfinite(x) else None

def strict_sum(values):
 values=list(values)
 return sum(values) if all(v is not None for v in values) else None

def raw_csv(path):
 rows=list(csv.reader(io.StringIO(Path(path).read_text())));index=next(i for i,r in enumerate(rows) if 'Kernel Name'in r and'ID'in r)
 header=rows[index];units=dict(zip(header,rows[index+1]));data=[]
 for row in rows[index+2:]:
  if len(row)==len(header) and re.fullmatch(r'\d+',row[header.index('ID')]):data.append(dict(zip(header,row)))
 return units,data

def source_csv(path):
 tables=[];header=None
 for row in csv.reader(io.StringIO(Path(path).read_text())):
  if not row:continue
  if row[0]=='Kernel Name':tables.append({'kernel':row[1],'header':[],'rows':[]});header=None
  elif row[0]=='Address':header=row;tables[-1]['header']=row
  elif row[0].startswith('0x'):
   assert header and len(row)==len(header),'source CSV schema mismatch'
   tables[-1]['rows'].append(dict(zip(header,row)))
 return tables

def instruction_rows(sass,token):
 tables=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if token in t.splitlines()[0]]
 assert len(tables)==1,(token,'ambiguous actual SASS symbol')
 return {int(m[1],16):m[2].strip() for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',tables[0])}

def bind_source(table,instructions):
 base=min(int(row['Address'],16) for row in table['rows'])
 rows={int(row['Address'],16)-base:row for row in table['rows']}
 assert len(rows)==len(table['rows']) and set(rows)==set(instructions),'PC-only tamper/coverage mismatch'
 def normalize(text):
  text=re.sub(r'0x[0-9a-f]+',lambda m:hex(int(m[0],16)-base) if base<=int(m[0],16)<=base+max(instructions)+16 else m[0],text)
  return re.sub(r'\s+','',text.replace('.reuse','').strip().rstrip(';'))
 for pc,row in rows.items():assert normalize(row['Source'])==normalize(instructions[pc]),('NCU source/SASS PC-predicate-instruction mismatch',hex(pc),row['Source'],instructions[pc])
 return rows,base

def aggregate(rows,pcs,header):
 rs=[rows[pc] for pc in sorted(pcs)]
 sumkey=lambda key:strict_sum(number(r.get(key)) for r in rs) if key in header else None
 reasons=[key for key in header if key.startswith('stall_') and '(Not Issued)'not in key]
 ni=[key for key in header if key.startswith('stall_') and '(Not Issued)'in key]
 values={key:sumkey(key) for key in ['# Samples','Warp Stall Sampling (All Samples)','Warp Stall Sampling (Not-issued Samples)','Instructions Executed','Thread Instructions Executed','Predicated-On Thread Instructions Executed',*reasons,*ni]}
 total=values['# Samples'];exported=strict_sum(values[key] for key in reasons)
 issued_exported=strict_sum(values[key] for key in ni)
 values.update(PC_count=len(pcs),unexported_reason_residual=None if total is None or exported is None else total-exported,
  not_issued_unexported_reason_residual=None if values['Warp Stall Sampling (Not-issued Samples)'] is None or issued_exported is None else values['Warp Stall Sampling (Not-issued Samples)']-issued_exported)
 return values

def phase_sets(mapping,instructions):
 roles={name:{int(pc,16) for pc in mapping['roles'][key]} for name,key in [('math','math_PCs'),('producer','producer_PCs'),('common','common_setup_unreachable_PCs')]}
 assert set.union(*roles.values())==set(instructions) and sum(map(len,roles.values()))==len(instructions)
 phases=collections.defaultdict(set)
 for wait in mapping['waits']:
  if wait['wait_path']!='fast_entry_to_outlined':continue
  kind=wait['classification']
  if kind=='math_Xfull':kind+='_'+('loop'if wait['placement']=='loop'else'startup')
  pcs={int(wait['PC'],16),int(wait['failed_branch']['PC'],16)}
  target=int(wait['failed_branch']['target_PC'],16);ready=int(wait['ready_return']['PC'],16)
  assert 0<=ready-target<=16*16
  pcs|={pc for pc in instructions if target<=pc<=ready}
  phases[kind]|=pcs
  fast=int(wait['PC'],16);branch=int(wait['failed_branch']['PC'],16)
  assert fast<branch and branch-fast<=16*32
  phases[kind+'_interleaved_fast_window']|={pc for pc in instructions if fast<pc<branch}
 assert {int(w['PC'],16) for w in mapping['waits']}<=set.union(*phases.values())
 used=set()
 for name,pcs in phases.items():assert not used&pcs,(name,'wait phase overlap');used|=pcs
 names={'math_bar5':'math_bar5','math_LDSM':'math_LDSM_X','math_TMA':'math_X_TMA_issue','math_LDG128':'math_direct_X_LDG128',
        'math_fence':'math_WARPGROUP_ARRIVE','math_wait1':'math_WGMMA_wait1','math_wait0':'math_WGMMA_finalwait0',
        'math_proxy':'math_shared_proxy','producer_owner_barriers':'producer_owner_bar3_bar4','common_setup_LDG128':'common_predicated_X_LDG128'}
 for site,label in names.items():
  pcs={int(v['PC'],16) for v in mapping['sites'].get(site,[])}
  assert not pcs&used,(label,'site overlap');phases[label]=pcs;used|=pcs
 for label,pattern in [('math_HGMMA',r'\bHGMMA\.'),('math_output_shared_STS',r'\bSTS(?:\.| )')]:
  pcs={pc for pc in roles['math']-used if re.search(pattern,instructions[pc])};phases[label]=pcs;used|=pcs
 for label,pattern in [('producer_scalar_STS',r'\bSTS(?:\.| )'),('producer_W_shared_LDS',r'\bLDS(?:\.| )'),('producer_escape_global_LDG',r'\bLDG(?:\.| )'),('producer_proxy',r'\bFENCE\.VIEW\.ASYNC'),('producer_global_async_issue',r'\bUTMA')]:
  pcs={pc for pc in roles['producer']-used if re.search(pattern,instructions[pc])};phases[label]=pcs;used|=pcs
 for role,pcs in roles.items():phases[role+'_other_residual']=pcs-used
 covered=set()
 for name,pcs in phases.items():assert not covered&pcs,(name,'phase overlap');covered|=pcs
 assert covered==set(instructions)
 return roles,dict(phases)

def lane_records(mapping,rows,header):
 records=[]
 slow={int(w['PC'],16):w for w in mapping['waits'] if w['wait_path']=='outlined_retry'}
 for wait in mapping['waits']:
  if wait['wait_path']!='fast_entry_to_outlined':continue
  retry=slow[int(wait['outlined_try_PC'],16)]
  pcs={'fast_try':wait['PC'],'failed_entry_branch':wait['failed_branch']['PC'],'slow_try':retry['PC'],'failed_retry_branch':retry['failed_branch']['PC']}
  values={name:{'PC':pc,**aggregate(rows,{int(pc,16)},header)}for name,pc in pcs.items()}
  entry=values['failed_entry_branch'];ret=values['failed_retry_branch'];slowtest=values['slow_try']
  failed=entry['Predicated-On Thread Instructions Executed'];tested=entry['Thread Instructions Executed']
  expected=None if failed is None or ret['Predicated-On Thread Instructions Executed'] is None else failed+ret['Predicated-On Thread Instructions Executed']
  records.append({'classification':wait['classification'],'placement':wait['placement'],'sites':values,
   'failed_entry_lane_fraction':failed/tested if failed is not None and tested else None,
   'slow_test_lanes_vs_entry_plus_retry':{'slow':slowtest['Thread Instructions Executed'],'entry_plus_retry':expected,'equal':expected==slowtest['Thread Instructions Executed'] if expected is not None else None},
   'interpretation':'Predicated-on lanes, not initially-ready probability, warp retries or latency. TRYWAIT can block; no divide-by32 uniformity assumption.'})
 return records


def metric(raw,units,key):
 value=number(raw.get(key));unit=units.get(key);base=value
 scale={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'sector':1,'request':1,'sample':1,'inst':1,'Kinst':1e3,'Minst':1e6,'Ginst':1e9}
 if unit in scale and value is not None:base=value*scale[unit]
 return {'value':value,'unit':unit,'base_value':base,'base_unit':'byte' if unit in ('byte','Kbyte','Mbyte','Gbyte')else 'inst'if unit in ('inst','Kinst','Minst','Ginst')else unit}

def analyze(export_collection):
 inventory=json.loads((export_collection/'inventory.json').read_text())
 for rel,pin in inventory['files'].items():assert sha(export_collection/rel)==pin,rel
 assert inventory['terminal_exit']==0
 clean=inventory['cleanup_observation'];assert clean['container_exists'] is False and clean['guard_pids']==clean['compute_pids']==clean['containers']==[] and clean['gpu_memory_mib']==0 and clean['cleanup_failed'] is False
 directory=export_collection/'logs/scheduler/native-profile-export-01/run';export=json.loads((directory/'result.json').read_text())
 assert export['status']=='OFFLINE_EXPORT_COMPLETE_SOURCE_COUNTER_ANALYSIS_PENDING' and export['input_immutable']=='PASS'
 assert export['input_sha256_expected']=='4bbde4f4d2a157dde8504dee2764c1a80706dd4ae1ff735725703681b39c6ddf'
 for name,pin in export['artifact_hashes'].items():assert sha(directory/name)==pin
 assert len(export['commands'])==2 and all(c['returncode']==0 and c['argv'][1]=='--import'for c in export['commands'])
 units,raw=raw_csv(directory/'raw.csv');tables=source_csv(directory/'source.csv');assert len(raw)==len(tables)==3
 assert [row['ID']for row in raw]==['0','1','2']
 assert all(token in row['Kernel Name']for token,row in zip(TOKENS.values(),raw))
 assert all(any(':profile:'in value for key,value in row.items()if'Domain:'in key)for row in raw)
 driver=json.loads((CAPTURE/'logs/scheduler/native-profile-01/driver.json').read_text())
 capture=json.loads((HERE/'capture-validation.json').read_text());assert capture['cleanup_verified'] and capture['status'].startswith('CAPTURE_VALIDITY_PASS')
 spec=importlib.util.spec_from_file_location('profile_actual_map',ROOT/'proj_rsi/tools/native_x_compare_ncu.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 support=M.support_module();results={}
 selected_metrics=['launch__grid_dim_x','launch__grid_dim_y','launch__grid_dim_z','launch__block_dim_x','launch__block_dim_y','launch__block_dim_z','launch__registers_per_thread','launch__shared_mem_per_block','launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_driver','dram__bytes_read.sum','dram__bytes_write.sum','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','lts__t_sector_hit_rate.pct','lts__t_sector_op_read_hit_rate.pct','l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__inst_executed.sum']
 for name,token in TOKENS.items():
  rr=[r for r in raw if token in r['Kernel Name']];tt=[t for t in tables if token in t['kernel']];assert len(rr)==len(tt)==1
  rr,table=rr[0],tt[0];role='NX'if name=='NX_main'else'DP'
  sass=(CAPTURE/'logs/scheduler/native-profile-01'/('driver.'+role+'.jit.sass')).read_text()
  assert hashlib.sha256(sass.encode()).hexdigest()==driver['variants'][role]['JIT_sha256']
  instructions=instruction_rows(sass,token);rows,base=bind_source(table,instructions)
  header=table['header'];total=aggregate(rows,set(rows),header)
  raw_samples=number(rr.get('smsp__pcsamp_sample_count'));assert total['# Samples']==raw_samples and raw_samples is not None
  assert total['Warp Stall Sampling (All Samples)']==raw_samples
  metadata={key:metric(rr,units,key)for key in rr if key.startswith('smsp__pcsamp_')}
  assert number(rr['smsp__pcsamp_dropped_bytes'])==0
  assert number(rr['smsp__pcsamp_aggregated_passes'])==2 and number(rr['smsp__pcsamp_interval_cycles'])==2048
  assert all(number(rr['launch__'+part+'_dim_'+axis])==1 for part in ('grid','block')for axis in ('y','z'))
  assert number(rr['launch__grid_dim_x'])==(256 if name=='DP_pack'else 128)
  assert number(rr['launch__block_dim_x'])==(256 if name=='DP_pack'else 512)
  assert number(rr['launch__registers_per_thread'])==(22 if name=='DP_pack'else 122)
  if name=='DP_pack':
   roles={'pack':set(rows)};phases={'pack_all':set(rows)};lanes=[];mapping=None
  else:
   _,mod,checker=M.bind(role);gate=checker.check(sass)
   assert gate==driver['variants'][role]['physical_gate']
   mapping=json.loads(json.dumps(M.pc_map(sass,gate,role)[0]));assert mapping==driver['variants'][role]['PC_map'][0]
   roles,phases=phase_sets(mapping,instructions);lanes=lane_records(mapping,rows,header)
   assert json.loads(json.dumps(support.actual_binary_gate('dp'if role=='DP'else'new',sass)))==driver['variants'][role]['canonical_gate']
  role_stats={key:aggregate(rows,pcs,header)for key,pcs in roles.items()}
  assert sum(v['# Samples']for v in role_stats.values())==raw_samples
  phase_stats={key:aggregate(rows,pcs,header)for key,pcs in phases.items()}
  assert sum(v['# Samples']for v in phase_stats.values())==raw_samples
  for key,v in phase_stats.items():
   pcs=phases[key];role_names=[r for r,rpcs in roles.items()if pcs and pcs<=rpcs]
   phase_role=role_names[0]if len(role_names)==1 else None
   v['role']=phase_role;denom=role_stats[phase_role]['# Samples']if phase_role else None
   v['fraction_of_role_samples']=v['# Samples']/denom if denom else None
   v['leading_phase_sample_sufficiency']=v['# Samples']>=30
  wait_windows={}
  for key in phase_stats:
   if key.endswith('_interleaved_fast_window'):
    basekey=key[:-len('_interleaved_fast_window')]
    if basekey in phase_stats:
     count=phase_stats[basekey]['# Samples']+phase_stats[key]['# Samples'];role_=phase_stats[basekey]['role'];den=role_stats[role_]['# Samples'] if role_ else None
     wait_windows[basekey]={'exact_test_predicate_slow_path_samples':phase_stats[basekey]['# Samples'],'interleaved_fast_window_samples':phase_stats[key]['# Samples'],'combined_samples':count,'fraction_of_role_samples':count/den if den else None,'limit':'Qualified code window includes interleaved arithmetic; not isolated mbarrier latency.'}
  role_counts={key:v['# Samples']for key,v in role_stats.items()}
  sufficient=name=='DP_pack'or all(role_counts[r]>=200 for r in ('math','producer'))
  wg=number(rr.get('smsp__pcsamp_warps_issue_stalled_warpgroup_arrive'))
  wgni=number(rr.get('smsp__pcsamp_warps_issue_stalled_warpgroup_arrive_not_issued'))
  bypc={pc:key for key,pcs in phases.items()for pc in pcs}
  top=[{'PC':hex(pc),'instruction':instructions[pc],'phase':bypc[pc],'stats':aggregate(rows,{pc},header)}for pc in sorted(rows,key=lambda p:number(rows[p].get('# Samples'))or 0,reverse=True)[:20]]
  results[name]={'kernel':rr['Kernel Name'],'source_and_actual_SASS_full_PC_predicate_instruction_binding':'PASS','virtual_base':hex(base),'PC_count':len(rows),
   'raw_source_sample_count_reconciliation':'PASS','sample_denominators':{**{key:value['# Samples']for key,value in role_stats.items()},'unmapped':0,'total':raw_samples},'total':total,'roles':role_stats,'phases':phase_stats,'top20_PCs':top,'lane_tests':lanes,
   'sample_sufficiency':sufficient,'wait_window_summaries':wait_windows,'sampling_metadata':metadata,'metrics':{key:metric(rr,units,key)for key in selected_metrics},
   'unexported_reason_reconciliation':{'source_residual':total['unexported_reason_residual'],'raw_warpgroup_arrive':wg,'matches':total['unexported_reason_residual']==wg,
     'not_issued_residual':total['not_issued_unexported_reason_residual'],'raw_not_issued_warpgroup_arrive':wgni,'not_issued_matches':total['not_issued_unexported_reason_residual']==wgni,
     'limit':'Residual is not permission to invent per-PC warpgroup_arrive attribution.'}}
 return {'status':'ACTUAL_COUNTER_SOURCE_ANALYSIS_PASS_NO_TIME_OR_SPEED_CLAIM','capture_job':'native.profile.v1.01','export_job':'native.profile.export.v1.01',
  'export_collection':str(export_collection.relative_to(ROOT)),'export_inventory_sha256':sha(export_collection/'inventory.json'),'export_source_sha256':export['source_sha256'],
  'capture_report_sha256':export['input_sha256_expected'],'export_client':export['export_client'],'capture_binary_sha256':'UNKNOWN; export hash is not capture provenance',
  'capture_post_range_oracles_and_cleanup':'PASS; see capture-validation.json','three_nodes_in_order':list(results),'variants':results,
  'unavailable_metrics':capture['unavailable_capture_metrics'],'interpretation_limits':capture['interpretation_limits'],'remaining':'Generate source-scoped implementation hypotheses; separately reviewed unprofiled comparisons required for any speed claim.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--collection',type=Path);args=ap.parse_args()
 collections_=sorted(EXPORTS.glob('collection-*')) if EXPORTS.exists() else []
 if not args.collection and not collections_:
  result={'status':'QUEUED_CPU_EXPORT','sample_denominators':None,'reason':'No actual collected exports available; no fabricated counts.'}
 else:result=analyze(args.collection or collections_[-1])
 (HERE/'counter-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 summary={'status':result['status']}
 if 'variants'in result:
  summary['samples']={name:{'total':v['total']['# Samples'],'roles':{key:x['# Samples']for key,x in v['roles'].items()},'leading_phases':{key:{'samples':x['# Samples'],'role_fraction':x['fraction_of_role_samples']}for key,x in v['phases'].items()if x['# Samples']>=30},'residual':v['unexported_reason_reconciliation']}for name,v in result['variants'].items()}
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
