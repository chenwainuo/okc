"""Read-only exact exposure receipt classification; no timing inference."""
from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[4];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stats=a.collection/'logs/scheduler/rq-pooler32-escape-exposure-01/stats.json';inv=a.collection/'inventory.json';r=json.loads(stats.read_text());i=json.loads(inv.read_text())
for f,h in i.get('dispatch',{}).get('source_pins',{}).items():assert sha(ROOT/f)==h,f
valid=r['state']=='VALID_EXACT_POOLER32_ESCAPE_WINDOW_EXPOSURE'
out={'state':'INVALID_EXPOSURE_NO_MECHANISM_DECISION','actual_stats_state':r['state'],'stats_path':str(stats),'stats_sha256':sha(stats),'inventory_sha256':sha(inv),'elapsed_seconds':r.get('elapsed_seconds'),'speed_gain_us':None,'family_rejection':False}
if valid:
 e=r['exposure'];assert len(e['lengths'])==1024 and max(e['lengths'])==e['maximum_records_per_slot']and sum(v//32 for v in e['lengths'])==e['required_subsequent_bank_loads']
 assert r['cleanup']['allocated']==0 and r['cleanup']['reserved']==0
 cleanup=i['cleanup_observation'];assert cleanup['terminal_exit']==0 and not cleanup['cleanup_failed']and cleanup['gpu_memory_mib']==0
 active=e['required_subsequent_bank_loads']>0
 out.update(state='VALID_EXPOSED'if active else'VALID_NOT_EXPOSED',exposure=e,cleanup=cleanup,next='Root may consider frozen two-arm lookahead trial with actual exposure count; no assumed latency gain.'if active else'Park current lookahead version for this actual fixture. No43sJIT/timing needed to test a nonexistent subsequent-window path. Preserve CPU correctness and reopen on actual active-window evidence; no synthetic tail benchmark.',cause_status={'exact_fixture_and_count':'PROVEN','subsequent_refill_exposure':'PROVEN active'if active else'PROVEN zero','initial_LDG_dependency':'Different path; candidate does not target it','performance_gain':'UNKNOWN; no performance measurement'},contract='proj_rsi/ITERATION.md')
s=json.dumps(out,indent=2)+'\n';assert not a.output.exists()or a.output.read_text()==s;a.output.write_text(s);print(s)
