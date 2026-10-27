import collections,hashlib,importlib.util,json,statistics
from pathlib import Path
root=Path('/home/ubuntu/okc');base=root/'proj_rsi/candidates';arc=root/'proj_rsi/diagnostics/group8-stmatrix-01';d=json.loads((arc/'group8-stmatrix-01.json').read_text());t=d['timing'];labels=('incumbent','ds','g8','new')
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r={};store=mod(base/'operand_reverse_dual_stmatrix/liveness.py','sol03_runtime_store')
for l,folder,token in [('ds','operand_reverse_dual_stmatrix','rq_px_dp_stmx'),('g8','operand_reverse_group8','rq_px_g8'),('new','operand_reverse_group8_stmatrix','rq_px_g8_stmx')]:
 s=(arc/f'group8-stmatrix-01.{l}.jit.sass').read_text();assert hashlib.sha256(s.encode()).hexdigest()==d['actual_JIT_SASS'][l]['sha256'];c=mod(base/folder/'liveness.py','sol03_runtime_'+l);r[l]={'physical':c.check(s),'store':store.store_form(s,token,l!='g8',dual=True)}
 attrs=d['attrs'][l];assert attrs['regs']==122 and attrs['local']==0 and attrs['smem']==103600 and attrs['threads']==512 and attrs['rounded_register_pool']==65536 and attrs['occupancy_after_optin']==1 and attrs['max_dynamic_after']>=103600
 assert d['packer_attrs'][l]['regs']==22 and d['packer_attrs'][l]['local']==0
 assert d['GPU_pack_all65536_BF16_storage_bits'][l]=='PASS_1MiB_EXACT_INPUT_IMMUTABLE_STORAGE_ONLY'
 assert d['fresh_workspace_and_output'][l]['status']=='PASS'
 ev=d['profile'][l];assert len(ev)==2 and sum('rx_pack' in x['name'] for x in ev)==1 and sum(token+'<' in x['name'] for x in ev)==1
 assert len(d['validation'][l])==6
assert len(d['validation']['incumbent'])==3
calls=t['all_timed_and_warmup_calls'];assert len(calls)==295
assert sum(c['warmup'] for c in calls)==20
f1calls=[c for c in calls if c['mode']=='F1'];pcalls=[c for c in calls if c['mode']=='paired'];assert len(f1calls)==147 and len(pcalls)==128
raw=t['F1_samples_us'];assert len(raw)==7 and all(len(x)==21 for x in raw.values());med={k:statistics.median(x) for k,x in raw.items()};assert med==t['F1_medians_us']
expected_blocks=['incumbent_before','ds_before','g8_before','new','g8_after','ds_after','incumbent_after']
for j,k in enumerate(expected_blocks):
 cs=f1calls[21*j:21*(j+1)];label=k.split('_')[0];assert all(c['label']==label and c['input_seed']==240000+21*j+i for i,c in enumerate(cs));assert [c['us'] for c in cs]==raw[k]
pairs=t['paired_graph_us'];assert len(pairs)==32
orders=[]
for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
for j,row in enumerate(pairs):
 cs=pcalls[4*j:4*j+4];assert [c['label'] for c in cs]==[labels[i] for i in orders[j%8]];assert all(c['input_seed']==250000+j for c in cs);assert {c['label']:c['us'] for c in cs}==row
f1={l:min(med[l+'_before'],med[l+'_after'])-med['new'] for l in labels[:3]};pg={l:statistics.median(x[l]-x['new'] for x in pairs) for l in labels[:3]};wins={l:sum(x[l]>x['new'] for x in pairs) for l in labels[:3]};assert f1==t['F1_new_gain_us'] and pg==t['paired_new_median_gain_us'] and wins==t['paired_new_wins']
for phase,cs in [('F1',f1calls),('paired',pcalls)]:
 gate=t['strict_immediate_actual_timed_output_checks'][phase];last={l:next(c['input_seed'] for c in reversed(cs) if c['label']==l) for l in labels};assert len(gate['checks'])==4
 for c in gate['checks']:assert c['actual_timed_output_before_any_replay_or_mutation']=='PASS' and c['input_seed']==last[c['graph']] and c['XP_current_recorded_fixture_exact']==(c['graph']!='incumbent') and c['oracle']
 assert gate['current_X_matches_last_actual_call_seed']==cs[-1]['input_seed'] and gate['W_packed_escape_immutable_counters_zero'] and gate['no_output_repair_replay']
assert t['terminal_last_timed_input_seed_per_policy']=={l:250031 for l in labels}
mut=d['actual_timed_graph_checks'];assert len(mut)==20 and sum(c['packed_poison_overwritten_byte_exact'] for c in mut)==15 and all(c['NaN_poison_overwritten'] and c['input_immutable'] for c in mut)
assert d['weights_inputs_counters']=='UNCHANGED_OR_RESET' and d['graph_cleanup']=='RESET_ALL_FOUR'
r.update(status='VALID_REGRESSION_NO_INCUMBENT_GAIN',F1=med,F1_gains=f1,paired_gains=pg,wins=wins,calls=len(calls),strict_terminal_oracles=8,strict_XP_checks=6,paired_orders={','.join(labels[i] for i in o):4 for o in orders},G8_paired_gain_vs_DS=statistics.median(p['ds']-p['g8'] for p in pairs),G8_wins_vs_DS=sum(p['ds']>p['g8'] for p in pairs))
(root/'proj_rsi/audits/sol03-kernel-validation-data/strict-runtime-review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('ds','g8','new')},indent=2))
