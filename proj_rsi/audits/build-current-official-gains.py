"""Read-only local evidence extraction; writes only this audit's JSON refs."""
import hashlib,json,math,re
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent;JOBS=ROOT/'results/proj-rsi/jobs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
ledger=json.loads((ROOT/'results/proj-rsi/ledger.json').read_text())
r={'scope':'Archived official job/source evidence; no live API or GPU invocation','incumbent':ledger['incumbent'],'jobs':{},'source_refs':{},'cautions':['Same-job prepare bake is distinct from scored timing; cross-job differences are descriptive only.','No parent_ns field is presumed to be a timed source-predecessor control.','Aggregate confidence intervals retained; point estimates alone do not prove each incremental mechanism.']}
for job,entry in ledger['jobs'].items():
 p=JOBS/(job+'.json');log=JOBS/(job+'.log')
 if not p.exists():continue
 d=json.loads(p.read_text());benches=d.get('benchmarks',[]);su=d.get('summary') or {};loads={}
 for b in benches:
  c=b.get('case',{});load=c.get('load')
  if load is None:continue
  z=loads.setdefault(load,{'ours_us':0.,'sglang_us':0.});z['ours_us']+=b['median_ns']/1000;z['sglang_us']+=b['baseline_ns']/1000
 recompute=4830.8/math.exp(sum(math.log(v['sglang_us']/v['ours_us']) for v in loads.values())/len(loads)) if len(loads)==6 else None
 if recompute is not None and su.get('score_us') is not None:assert abs(recompute-su['score_us'])<1e-7,(job,recompute,su)
 obj={'experiment':entry.get('experiment'),'kind':entry.get('kind'),'summary':su,'json_ref':ref(p),'load_sums':loads,'recomputed_score_us':recompute,'selected_positive_bakes':[],'gv_best_bakes':[],'pick_lines':[]}
 if log.exists():
  obj['stdout_ref']=ref(log);lines=log.read_text().splitlines();base={}
  for no,line in enumerate(lines,1):
   m=re.match(r'BAKE (\w+) m=(\d+) r6 (\d+\.\d+)',line)
   if m:base[m[1],int(m[2])]=(float(m[3]),no,line)
   m=re.match(r'BAKE (\w+) m=(\d+) r6 F1 pre .* base (\d+\.\d+)',line)
   if m:base[m[1],int(m[2])]=(float(m[3]),no,line)
   m=re.match(r'PICK (\w+) m=(\d+) (\S+) (\d+\.\d+)',line)
   if m:
    layer,n,path,t=m[1],int(m[2]),m[3],float(m[4]);obj['pick_lines'].append({'line':no,'text':line})
    if path!='r6' and (layer,n) in base:
     old,oldno,oldline=base[layer,n]
     if old>t:obj['selected_positive_bakes'].append({'layer':layer,'m':n,'selected':path,'incumbent_bake_us':old,'selected_bake_us':t,'observed_gain_us':round(old-t,6),'baseline_line':oldno,'pick_line':no,'caution':'Selected path vs same-job r6 bake, not necessarily incremental vs source predecessor; prepare context only.'})
   m=re.match(r'gv (\w+) m(\d+) v12 ([\d.]+) best (\S+) ([\d.]+)',line)
   if m:obj['gv_best_bakes'].append({'line':no,'layer':m[1],'m':int(m[2]),'baseline':m[3],'selected':m[4],'selected_us':m[5],'text':line,'caution':'Inherits tuning/search; not an isolated delta for newly added geometry.'})
 obj['scored_rows']=[{'layer':b['case']['layer'],'m':b['case']['m'],'load':b['case']['load'],'yours_us':b['median_ns']/1000,'sglang_us':b['baseline_ns']/1000,'runs':b.get('runs'),'status':b.get('status')} for b in benches]
 if entry.get('kind')=='scored':obj['profiles']=[{'layer':x['case']['layer'],'m':x['case']['m'],'report':x['report']} for x in d.get('profiles',[])]
 r['jobs'][job]=obj
for name in ['r6-c12','pw23','pw27h','pw29h','pw31','pw33','pw34','pw35','rs32','pw36']:
 p=BASE/'candidates'/name/'submission.py'
 if p.exists():r['source_refs'][name]=ref(p)
for name in ['01-state-board-score.md','05-track-rs32.md','06-track-router-pw35.md','10-closed-tracks.md']:
 p=ROOT/'handover_folder'/name;r.setdefault('handover_refs',[]).append(ref(p))
r['independent_followups']=[]
for name in ['current-leaderboard-stack-source.md','current-leaderboard-stack-source.json','current-leaderboard-gains-completeness.md','current-leaderboard-gains-completeness.json','eh-backport.md','pw38-independent.md']:
 p=BASE/'audits'/name
 if p.exists():r['independent_followups'].append(ref(p))
r['omitted_small_local_refinement']={'feature':'rs128b EH early-half epilogue bit65536','included_in_rs32':False,'different_from_EARLY_bit':32,'historical_reported_kernel_gain_us':[.13,.16],'recovered_single_qkv_bake_us':{'control':17.25,'EH':16.96},'limits':'Historical pooler/unpooler bake losses; raw CUPTI repetition distributions absent; not established current selected-incumbent/official gain. New optional PW38 remains runtime-unvalidated.'}
out=BASE/'audits/current-leaderboard-gains-official.refs.json';out.write_text(json.dumps(r,indent=2)+'\n');print('AUDITED_JOBS',len(r['jobs']),'SCORed',sum(o['kind']=='scored' for o in r['jobs'].values()),'POSITIVE_PICK_BAKES',sum(len(o['selected_positive_bakes']) for o in r['jobs'].values()))
for j in ['fb548b5dc6f04696','e22a4b9d693d4902']:
 o=r['jobs'][j];print(j,o['recomputed_score_us'],o['load_sums'])
