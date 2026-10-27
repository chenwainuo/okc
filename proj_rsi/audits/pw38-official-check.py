"""CPU-only independent reproduction of the PW38 official check gate."""
import hashlib,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=ROOT/'results/proj-rsi/releases/pw38-v1-29e557ea53f3';p=ROOT/'results/proj-rsi/jobs/a3ed4ce5886a4415.json';c=ROOT/'proj_rsi/candidates/pw38';d=json.loads(p.read_text());s=d['summary'];source='29e557ea53f3ce73cacfe5d22df4a5e405a822d195abd6311be3f4809dedad0e'
 assert sha(r/'submission.py')==sha(c/'submission.py')==source
 assert sha(r/'manifest.json')==sha(c/'manifest.json')=='48b697b288e71617bdf5e688325c17e90d6541d140153acfb73a464e01dbc89b'
 m=json.loads((r/'manifest.json').read_text())
 for f,h in m['files'].items():assert sha(c/f)==h,(f,h)
 a=json.loads((r/'check-completed-attestation.json').read_text());assert a['normalized_result']['sha256']==sha(p)
 archived=json.loads((r/'profiles/submission.a3ed4ce5886a4415.json').read_text());assert archived==d
 assert s['passed'] is True and s['error'] is None and s['status']=='ok' and s['tests']==s['tests_passed']==9
 b=d['benchmarks'];ps=d['profiles'];assert len(b)==len(ps)==30
 loads={};rows=[]
 for i,(x,y) in enumerate(zip(b,ps)):
  assert x['status']=='pass' and x['runs']==60 and len(x['runs_ns'])==len(x['baseline_runs_ns'])==60
  assert x['index']==y['index']==i and x['case']==y['case']
  for k in ['median_ns','baseline_ns','mean_ns','best_ns','worst_ns']:assert math.isfinite(x[k]) and x[k]>0
  for k in ['runs_ns','baseline_runs_ns']:assert all(math.isfinite(v) and v>0 for v in x[k])
  ls=loads.setdefault(x['case']['load'],{'ours_us':0.,'baseline_us':0.});ls['ours_us']+=x['median_ns']/1000;ls['baseline_us']+=x['baseline_ns']/1000
  lines=y['report'].splitlines();assert len(lines)==3 and re.search(r'100\.0%\s+1$',lines[1]),lines
  rows.append({'index':i,'case':x['case'],'calibrated_us':x['median_ns']/1000,'profile':lines[1]})
 assert len(loads)==6 and all(sum(x['case']['load']==k for x in b)==5 for k in loads)
 speed=math.exp(sum(math.log(v['baseline_us']/v['ours_us']) for v in loads.values())/6);score=4830.8/speed
 assert abs(speed-s['speedup'])<1e-12 and abs(score-s['score_us'])<1e-7
 log=(r/'profiles/submission.a3ed4ce5886a4415.log').read_text()
 assert 'PREP done in 161s' in log and 'PREP rq thread ready at 41s' in log and 'PREP rq32 thread ready at 41s' in log
 assert not re.search(r'(?i)traceback|\berror\b|\bfailed\b|\bskipped\b',log)
 for frag in ['PICK pooler m=128 rq:s4s5hzEH 22.85','PICK qkv_gate m=128 rq:s4s5hzEH 16.80','PICK router m=128 rt64k32b3x2z 11.49','PICK router m=256 rt64k32b3x2z 13.18']:
  assert frag in log
 for i in [20,22]:assert '65539>' in ps[i]['report']
 for i in [24,29]:assert 'void rtk<' in ps[i]['report']
 for i in [5,10,11,12,13]:assert 'void rq<' in ps[i]['report']
 old=json.loads((ROOT/'results/proj-rsi/jobs/fb548b5dc6f04696.json').read_text())['summary'];ci=[4830.8/s['speedup_ci'][1],4830.8/s['speedup_ci'][0]];oldci=[4830.8/old['speedup_ci'][1],4830.8/old['speedup_ci'][0]]
 out={'decision':'GO_SCORING_EXACT_RELEASE','job':'a3ed4ce5886a4415','mode':'unscored check','source_sha256':source,'summary':s,'recomputed_score_us':score,'score_ci_us':ci,'reference_scored_job':'fb548b5dc6f04696','reference_summary':old,'reference_score_ci_us':oldci,'score_point_delta_us':score-old['score_us'],'score_point_delta_percent':100*(score/old['score_us']-1),'ci_overlap':max(ci[0],oldci[0])<=min(ci[1],oldci[1]),'load_sums':loads,'profiles':rows,'checks':{'tests_passed':9,'finite_benchmarks':30,'runs_per_case':60,'one_kernel_profiles':30,'manifest_files_verified':len(m['files']),'cold_prepare_seconds':161,'runtime_errors_or_budget_skips':False},'refs':{str(z.relative_to(ROOT)):sha(z) for z in [p,r/'submission.py',r/'manifest.json',r/'profiles/submission.a3ed4ce5886a4415.log',r/'profiles/submission.a3ed4ce5886a4415.json',r/'attestation.json',r/'check-completed-attestation.json']}}
 (ROOT/'proj_rsi/audits/pw38-official-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['decision','recomputed_score_us','score_ci_us','score_point_delta_us','score_point_delta_percent','ci_overlap','checks']},indent=2))
if __name__=='__main__':main()
