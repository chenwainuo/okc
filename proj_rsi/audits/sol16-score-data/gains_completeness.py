#!/usr/bin/env python3
"""Read-only local corpus audit; only emits this agent's audit files."""
import collections,hashlib,json,pathlib,re,statistics
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=ROOT/'proj_rsi/audits/current-leaderboard-gains-completeness.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):
 p=ROOT/p if isinstance(p,str) else p
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p),'bytes':p.stat().st_size} if p.exists() else {'path':str(p),'missing':True}
roots=['proj_rsi/candidates','proj_rsi/audits','proj_rsi/cleanroom','results/proj-rsi','handover_folder','proj_rsi/scheduling/jobs']
files=set()
for root in roots:
 files.update(p for p in (ROOT/root).rglob('*') if p.is_file() and p.suffix in ('.md','.json','.log','.txt') and 'current-leaderboard-gains-completeness' not in p.name)
files.update(ROOT/p for p in ['proj_rsi/LOOP.md','proj_rsi/ITERATION.md','proj_rsi/queue.json','proj_rsi/protocol.json'])
memory=pathlib.Path('/home/ubuntu/.claude/projects/-home-ubuntu-okc/memory/muse-bf16-proj-loop.md')
if memory.exists():files.add(memory)
pattern=re.compile(r'\b(?:gain|wins?|faster|improv\w*|positive|merge\w*|includ\w*|CLOSED|NO_RETRY|kill)\b',re.I)
inventory=[]; claim_index=[]
for p in sorted(files):
 try:txt=p.read_text()
 except (UnicodeError,OSError):continue
 hits=[{'line':i,'text':line[:2000]} for i,line in enumerate(txt.splitlines(),1) if pattern.search(line)]
 rec=ref(p);rec.update(lines=len(txt.splitlines()),claim_search_hits=len(hits));inventory.append(rec)
 if hits:claim_index.append({'path':rec['path'],'hits':hits})
ledger=json.loads((ROOT/'results/proj-rsi/ledger.json').read_text())
official=[];bake_comparisons=[]
for p in sorted((ROOT/'proj_rsi/candidates').glob('*/profiles/*.json')):
 try:x=json.loads(p.read_text())
 except (ValueError,OSError):continue
 if not isinstance(x,dict) or 'benchmarks' not in x:continue
 official.append({'candidate':p.parents[1].name,'job':x.get('job_id'),'summary':x.get('summary'),'component_count':len(x['benchmarks']),'raw':ref(p),'evidence_type':'Official observed score, no cross-run causal subtraction'})
 log=p.with_suffix('.log')
 if not log.exists():continue
 bake=[];picks={}
 for i,line in enumerate(log.read_text().splitlines(),1):
  m=re.match(r'BAKE (\w+) m=(\d+) (\S+) ([0-9.]+)',line)
  if m:bake.append((m[1],int(m[2]),m[3],float(m[4]),i,line))
  m=re.match(r'PICK (\w+) m=(\d+) (\S+) ([0-9.]+)',line)
  if m:picks[(m[1],int(m[2]))]=(m[3],float(m[4]))
 for layer,m,mode,v,line_no,line in bake:
  if mode=='r6':continue
  control=[r for r in bake if r[0]==layer and r[1]==m and r[2]=='r6']
  if not control:continue
  # Same preparation can contain two phases; use nearest logged control.
  c=min(control,key=lambda r:abs(r[4]-line_no));gain=c[3]-v
  if gain>0:
   bake_comparisons.append({'candidate':p.parents[1].name,'job':x.get('job_id'),'layer':layer,'m':m,'mode':mode,'candidate_bake_us':v,'logged_r6_control_us':c[3],'positive_bake_difference_us':gain,'below_legacy_0_3us':gain<.3,'below_legacy_0_5us':gain<.5,'actual_final_pick':picks.get((layer,m)),'line':line_no,'control_line':c[4],'log':str(log.relative_to(ROOT)),'classification':'Positive rounded same-job bake sample vs named r6 control; not necessarily current-selected rs32 or independent repeat; do not promote from this index'})
# Recompute all strict positive cleanroom shape differences, without a kill bar.
cp=ROOT/'results/proj-rsi/takeover/20261026/cleanroom'
clean={n:{(r['layer'],r['m']):r['yours_us'] for r in json.loads((cp/(n+'.json')).read_text()) if 'layer' in r} for n in ['pw36','pw36_after','bandwidth','contrarian','latency','tensorcore']}
clean_result={}
for n in ('bandwidth','contrarian','latency','tensorcore'):
 diffs=[{'layer':k[0],'m':k[1],'candidate_us':v,'before_us':clean['pw36'][k],'after_us':clean['pw36_after'][k],'gain_vs_faster_bracket_us':min(clean['pw36'][k],clean['pw36_after'][k])-v} for k,v in clean[n].items()]
 clean_result[n]={'raw':ref(cp/(n+'.json')),'shapes':diffs,'strict_positive_count':sum(r['gain_vs_faster_bracket_us']>0 for r in diffs),'best_gain_us':max(r['gain_vs_faster_bracket_us'] for r in diffs)}
rtkpath=ROOT/'proj_rsi/candidates/rtk_early/results/rtk-early01.json';rtk=json.loads(rtkpath.read_text());rtkcalc={}
for m,r in rtk['shapes'].items():
 bs=[str(z) for z in r['baseline_versions']];c=str(r['candidate']);ss=r['graph_us']['samples'];gains={b:[] for b in bs}
 for rep in range(40):
  for pos,b in enumerate(bs if rep%2==0 else reversed(bs)):
   gains[b].append(ss[b][rep]-ss[c][2*rep+pos])
 med={k:statistics.median(v) for k,v in ss.items()};paired={k:statistics.median(v) for k,v in gains.items()}
 assert paired==r['graph_us']['paired_gain_median_us']
 assert med==r['graph_us']['median']
 f1=r['prepare_style_us'];fg=min(*f1['baseline_pre'].values(),*f1['baseline_post'].values())-f1['candidate']
 assert fg==f1['gain_us']
 rtkcalc[m]={'raw':ref(rtkpath),'candidate':c,'baseline_versions':bs,'medians_us':med,'paired_gains_us':paired,'F1_gain_us':fg,'gain_vs_fastest_shipped_medians_us':min(med[b] for b in bs)-med[c],'scope':'router hi+lo FP32; baseline shipped pw35 rtk variants, not rs32 or pw36 complete kernel_fn','precision_and_retained_graph_checks_present':all(v.get('probe_exact_hi_plus_lo',False) for g in r['actual_timed_graph_validation_after_replays'].values() for v in g['checks'])}
M=[]
def add(name,classification,finding,paths,in_rs32='No',scope='H100 local',certainty='reported audit; raw attached where present'):
 M.append({'id':name,'classification':classification,'in_rs32':in_rs32,'scope':scope,'finding':finding,'verification':certainty,'evidence':[ref(p) for p in paths]})
add('r6 lossless12bit/direct GV +Lt selection','SHIPPED_INCLUDED_OFFICIAL','Core lossless codec, direct-register small shapes and tuned Lt baseline remain in rs32; actual source ancestry and30 measured shapes independently mapped by Sol11. Older r6/e4/e5 raw jobs preserve progress, not additive gains.',['proj_rsi/candidates/r6-c12/profiles/submission.23cecdef1bb54c9e.json','proj_rsi/candidates/rs32/submission.py'],'Yes','Official H200','official raw + source map')
add('P1 small router/unpooler self-zero+RR6','SHIPPED_INCLUDED_OFFICIAL','P1 removes memset/cluster edges; router m8/16/32 and sliced unpooler m1/8 remain selected; router m1 RR6 retained. Includes narrow rr6 ~0.22us local positive, not dropped by0.3 bar.',['proj_rsi/audits/p1-h100.md','proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.log'],'Yes','Official H200 + historical H100')
add('P2 qkv shortened dependency chains','SHIPPED_INCLUDED_OFFICIAL','P2 qkv small geometry choices included through pw33 ancestry; actual rq overrides qkv16 and qkv32 in best job. Historical microvariants with0.08/0.19/0.24/0.32us observations are recorded, not independent omitted end-to-end gains.',['proj_rsi/audits/p2-h100.md','proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.log'],'Yes, some exact-M overrides','Official H200 + historical H100')
add('pw22/pw23/pw29h router128 progression','SHIPPED_INCLUDED_OFFICIAL','Split-K/pipelined/self-zeroing router lineage retained through old RWCU; rs32 selects rw64k32s8z at128. Earlier router variants are selectable alternatives or superseded, not additional additive gains.',['proj_rsi/audits/p3-h100.md','proj_rsi/candidates/pw22/profiles/submission.cdb3e04eda1c4d13.json','proj_rsi/candidates/pw23/profiles/submission.17f4484b97644673.json'],'Yes','Official H200')
add('RS128 packed register-A','SHIPPED_INCLUDED_OFFICIAL','Best job selects qkv128 s4s5hz and pooler128 s4s5z. Full source also offers other shapes; unpooler128 andM256 retain r6/Lt in this job. A local unpooler gain is not selected official inclusion.',['proj_rsi/candidates/pw34/profiles/submission.846e25d9df09433f.log','proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.log'],'Yes, qkv/pooler128 selected','Official H200')
add('RS32 five actual rq small-M picks','SHIPPED_INCLUDED_OFFICIAL','qkv16 +qkv32/o32/pooler32/unpooler32 all select rq in actual best official log. Narrow o32 bake improvement0.44us (r610.94→10.50) is included; source/box NOTES saying o32 kept r6 or all16kept r6 are not actual best dispatch.',['proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.log'],'Yes','Official H200','raw PICK + actual timed profile')
add('PW35 one-node rtk router128/256','COMPONENT_WIN_FAILED_COMBINED_STACK','Genuine official within-prepare rtk wins, absent best rs32. Included in reviewed pw36 combined source; pw36 official3212.415 loses best3200.615 aggregate with overlappingCI despite rtk selections. This is the concrete omitted feature; not proof of a better combined score or license to add crossrun minima.',['proj_rsi/candidates/pw35/profiles/submission.5d880cd0aa504209.json','proj_rsi/candidates/pw36/profiles/submission.e22a4b9d693d4902.json','proj_rsi/audits/pw36-official.md','proj_rsi/audits/pw36-takeover.md'],'No; PW36 includes','Official H200','raw official + byte-exact ancestry confirmedSol11')
add('RS128b early-half epilogue EH','MEASURED_SMALL_COMPONENT_GAIN_OMITTED_UNCONFIRMED_FULL_PATH','Valid/probe-exact CUPTI0.13–0.16us report versus R4, below historical bar; qkv single-bake gain0.29. Complete actual-kernel same-run control/paired/F1 proof absent locally. LOOP says not packaged; Sol11 source audit confirms EHbit16=65536/EPbit17 absent; current rs32 HALFbit4=16/EARLYbit5=32 is different. Do not discard or count as a proven current incumbent win.',['proj_rsi/audits/mx-h100.md','proj_rsi/LOOP.md','proj_rsi/candidates/rs128b/lab.py'],'No, EHbit65536 absent; distinct from EARLYbit32','Historical H100 component/kernel','audit claim verified present; original remote raw not local')
add('RS128b separate-W ring qkv x6w4','MEASURED_SMALL_COMPONENT_GAIN_UNCONFIRMED','qkv12814.85 vs15.23 (~0.38us) CUPTI near-noise result; equal depth neutral/deeper worse. Not paired full serving win vs actual selected rs32; preserve as specific unresolved historical mechanism.',['proj_rsi/audits/mx-h100.md'],'No evidence of selected inclusion','Historical H100 kernel')
add('MX exact qkv/pooler128','HISTORICAL_BASELINE_WIN_SUPERSEDED','Lossless MX table qkv18.50 vsr619.49 (~0.99us),pooler26.53vs27.10 (~0.57us); later RS128 is faster and shipped. These are not omitted wins versus current selected rq. No combined production/H200 win measured for MX.',['proj_rsi/audits/mx-h100.md','proj_rsi/LOOP.md'],'No, superseded by faster RS128','Historical H100; local audit has remote raw-path gap')
add('MX queued2groups','PARENT_ONLY_LEARNING','Two groups in flight reports0.5us improvement within MX lineage; not actual incumbent/official gain. Selfzero/one-lane reduce and header are mechanisms, not additive measured board gains.',['proj_rsi/audits/mx-h100.md'],'No direct inclusion claim')
add('bf16 P4 k128/k2x','HISTORICAL_UNTUNED_CONTROL_WIN','k128/merged barriers reports~2.4us pooler256 local gain versus priorbf16 version or torch; pw32 actual tuned-Lt comparison loses every path. Avoid treating torch-only parity or timing-only rungs as omitted rq wins.',['proj_rsi/audits/p4-h100.md','proj_rsi/candidates/pw32/hypothesis.json','proj_rsi/LOOP.md'],'Not selected')
add('RTK stage0 early','VALID_REGRESSION_SIGN_CORRECTION','+.480/+.096 are candidate-minus-fastest LOSSES; raw recomputed paired gains M128−.480/−.528,M256−.144/−.096; F1−.544/−.192. No small end-to-end positive omitted. First-full improves0.320/0.384us but tail regresses; timestamp only.',['proj_rsi/candidates/rtk_early/results/rtk-early01.json','proj_rsi/candidates/rtk_early/RESULTS.md'],'No','H100 router pw35 variants','all raw sample medians +80 candidate pair mapping recomputed')
add('cleanroom latency DBG512/b8','PARENT_ONLY_LEARNING','Reported1.04971x paired raw graph/1.06947x kernel vs own DBG0 control across15shapes. No actual incumbent comparison in b8. Later full-school trial loses all30shapes against both pw36 brackets at threshold0, not merely0.3.',['proj_rsi/audits/latency-b8-preflight.md','proj_rsi/LOOP.md','proj_rsi/audits/cleanroom-measure.md'],'No','H100 own-control','b8 raw absent locally; terminal full-school raw recomputed')
add('cleanroom BW/CR/LAT/TC','VALID_FULL_POLICY_REGRESSION','120school/shape comparisons: zero strict positive gains vsfaster opening/closing pw36, no minimum threshold. CR o32 opening-only0.368us disappears closingbaseline; closing loss0.112us and chosen pathtorch fallback.',['proj_rsi/audits/cleanroom-measure.md','results/proj-rsi/takeover/20261026/cleanroom/contrarian.json'],'No','H100 full policies','all120 bracket differences recomputed from6raw JSONs')
add('hw native allocation tiny paired positives','CONFLICTING_SMALL_PARENT_ONLY_ESTIMATORS','GENERIC paired+0.096001us vsordinary/NONE Lt, but F1 worse vsall; versus actual RQ paired−2.623999. Explicitly retain tiny positive estimator, no robust incumbent win.',['proj_rsi/candidates/hw_native/RESULTS.md','results/proj-rsi/takeover/20261026/hw-native01.json'],'No')
add('hardware compression original','INVALID_NO_GAIN_ADMISSION','Original cleanup failure invalidates whole timing run; NONE-vsGENERIC paired+0.032us cannot be promoted. Owner repair validates lifetime only, not retrospective speed.',['proj_rsi/candidates/hw_compression/RESULTS.md','proj_rsi/candidates/hw_compression_owner/RESULTS.md'],'No')
add('hardware compression knownzero controls','DIAGNOSTIC_RESPONSE_NOT_BENCHMARK_GAIN','Knownzero compressed allocation shows73.827% write-sector reduction; later64MiB zero controls show readresponse. Normal native/byteplane/signlow representations do not demonstrate real-weight matmul gain; no timing benefit inferred from counters.',['proj_rsi/candidates/hw_compression_zero_control/RESULTS.md','proj_rsi/audits/hw-compression-byteplanes-result.md','proj_rsi/audits/hw-compression-signlow-zero-result.md'],'No')
add('o direct W /Lt graph /router small','VALID_ACTUAL_INCUMBENT_REGRESSIONS','oDirect M32F1/paired−2.240/−2.208,M128−4.064/−3.936; Lt layouto32−.864/−.864; routerSmallM8−2.656/−2.592. All losing; no subbar positives hidden.',['proj_rsi/candidates/o_direct_w/RESULTS.md','proj_rsi/candidates/lt_graph/RESULTS.md','proj_rsi/candidates/router_small/RESULTS.md'],'No')
for name,desc,paths in [
 ('reverse V3','Paired+2.176us vsV2; −7.520vsV1,−30.368vsactual.',['proj_rsi/candidates/operand_reverse_v3/RESULTS.md']),
 ('reverse V4','Paired+4.608us vsV3,F1+2.688; −4.960vsV1,−28.416vsactual.',['proj_rsi/candidates/operand_reverse_v4/RESULTS.md']),
 ('fresh packed-X','F1+7.520/paired+7.904 vsV1; −15.424/−15.840vsactual.',['proj_rsi/candidates/operand_reverse_packed_x/RESULTS.json']),
 ('STMatrix packed','F1+1.184/paired+1.440 vsPX; −13.856/−14.400vsactual.',['proj_rsi/candidates/operand_reverse_stmatrix/RESULTS.json']),
 ('dual producer','F1+3.680/paired+3.904 vsSTMX; −10.400/−10.592vsactual.',['proj_rsi/candidates/operand_reverse_dual_producer/RESULTS.json']),
 ('dual STMatrix01','F1+0.832vsDP but paired−.256/10of21wins; +4.288pairedvsSTMX; actual−10.592.',['proj_rsi/audits/dual-stmatrix-result.md']),
 ('dual STMatrix balanced diagnostic','FixedDP/DS+1.168paired48of64wins, both orders/brackets positive; no actual incumbent arm and no official comparison.',['proj_rsi/audits/dual-stmatrix-measure-result.md']),
 ('native-X','F1/paired vsDP+1.152/+2.208,vsDS+2.752/+3.472; actual−7.552/−7.952,0of32wins.',['proj_rsi/audits/native-x-01/report.md','proj_rsi/audits/native-x-01/analysis.json'])]: add(name,'PARENT_ONLY_LEARNING_NO_ACTUAL_WIN',desc,paths,'No')
add('reverse V1/V2','VALID_ACTUAL_INCUMBENT_REGRESSIONS','V1 F1/paired−23.520vsactual;V2−33.152pairedvsactual and−9.568vsV1; later gains versusV2 cannot be attributed to shipping baseline.',['proj_rsi/candidates/operand_reverse/RESULTS.md','proj_rsi/candidates/operand_reverse_v2/RESULTS.md'],'No')
add('group8 original attempt','INVALID_STRICT_TIMING_ADMISSION','All source/runtime oracles pass but final actual timed output overwritten before independent terminal check. Preserve provisional losing tendency; corrected GS trial supplies strictG8 control evidence only, not retroactive admission.',['proj_rsi/audits/sol03-kernel-validation.md'],'No')
add('group8+STMatrix strict','VALID_REGRESSION','Strict20warm+147F1+128paired+terminal outputs; GS vsG8 F1−.576/paired−.704, vsactual−11.040/−11.296. No gain overactual; G8 also losesDS in strictsame-run rows.',['proj_rsi/audits/sol03-kernel-validation.md','results/proj-rsi/takeover/20261027/group8-stmatrix-01/group8-stmatrix-01.json'],'No')
add('native-X async refill','VALID_REGRESSION','Parent comparison F1−4.192/paired−3.760; actual−11.296/−11.488 and0of32wins; no current actual gain.',['proj_rsi/audits/async-refill-01/report.md','proj_rsi/audits/async-refill-01/analysis.json'],'No')
add('smallm cursor attempt01','INVALID_NO_PERFORMANCE_ADMISSION','Raw result statusINVALID_NO_PERFORMANCE_ADMISSION, timingraw empty; failed a counter assertion during qkv32 basis/FP64 phase; exact counter name UNKNOWN because assertion had no payload. Retain CPU codegen and repaired driver versions separately; no measured benefit.',['results/proj-rsi/takeover/20261027/smallm-increment01/result.json','proj_rsi/audits/sol04-smallm-increment.md'],'No')
add('CPU-only remaining ports/designs','CPU_ONLY_UNPROVEN','X4/nativeSTMatrix/X4STMatrix, QKVsplit4 wrapper, group8M256, rq11 codec, aligned epilogue, prepared footer andgroup16 forecasts remain source/CPU evidence unless later archived trial supplies validity. No speed from static instructions/bytes. Candidate inventory below tracks each distinct directory.',['proj_rsi/audits/sol15-qkv-runtime.md','proj_rsi/audits/sol17-rq11-feasibility.md','proj_rsi/audits/group16-feasibility.md'],'No')
# Every candidate directory is listed; no result-less directory is silently treated as a losing benchmark.
candidates=[]
for d in sorted((ROOT/'proj_rsi/candidates').iterdir()):
 if not d.is_dir():continue
 ps=[p for p in d.rglob('*') if p.is_file() and p.suffix in ('.md','.json','.log','.txt')]
 key=[p for p in ps if p.name.startswith(('RESULTS','STATUS','NOTES','README','hypothesis','decision'))]
 candidate_jobs=[r for r in official if r['candidate']==d.name]
 candidates.append({'candidate':d.name,'text_artifacts_scanned':len(ps),'summary_paths':[str(p.relative_to(ROOT)) for p in key],'official_jobs':[r['job'] for r in candidate_jobs],'classification':'OFFICIAL_RAW_PRESENT_SEE_JOB_AND_BAKE_MATRIX' if candidate_jobs else 'LOCAL_OR_CPU_EVIDENCE_SEE_NAMED_FINDINGS; no automatic gain admission','source_files':[ref(p) for p in d.glob('submission*.py')]})
ranked=[
 {'rank':1,'opportunity':'PW35 router benefit omitted from best artifact; preserve/review selective fusion with rs32','basis':'Only identified latest official component-positive feature absent best; combinedpw36 alreadytested and aggregate loses. Next action is exact current/pw36 prepare-pick regression analysis and controlled selective integration, not unchanged resubmission or crossrun min stitch.','speedup_forecast':'UNKNOWN','evidence':['PW35 one-node rtk router128/256']},
 {'rank':2,'opportunity':'EH half-epilogue historical smallgain provenance/inclusion','basis':'0.13–0.16us component observation retained below oldbar; source map confirms absent; first retrieve archived raw and verify exact source/full-path controls. Only then consider bounded full-kernel comparison against actual selected shape.','speedup_forecast':'UNKNOWN'},
 {'rank':3,'opportunity':'Repair smallm cursor numerical/driver gate and complete actual-policy evidence','basis':'Multi-load reach but attempt01 invalid with zero timings; no gain can be admitted. Existing concrete repaired source preserves earlier failure.','speedup_forecast':'UNKNOWN'},
 {'rank':4,'opportunity':'Existing native-X bottleneck/X4/store alternatives','basis':'Seed wins versusfailedparents butstill~8us behind actual U128; queueddiagnostics resolve a decision, not an omitted speedup. BroaderQKV/M256 only after mechanism evidence.','speedup_forecast':'UNKNOWN'}]
result={'authority':{'best_believed_job':'fb548b5dc6f04696','score_us':3200.6150084373335,'source':ref('proj_rsi/candidates/rs32/submission.py'),'official_raw':ref('proj_rsi/candidates/rs32/profiles/submission.fb548b5dc6f04696.json'),'live_API_verification':'Root owns live API verification; this worker did not contact service','rule':'family != version != attempt; legacy killbars do not erase any positive observation'},'coverage':{'roots':roots,'files_read':len(inventory),'lines_searched':sum(x['lines'] for x in inventory),'files_with_claim_hits':len(claim_index),'candidate_directories':len(candidates),'ledger_experiments':len(ledger['experiments']),'ledger_jobs':len(ledger['jobs']),'official_raw_jobs':len(official),'official_components':sum(r['component_count'] for r in official),'zero_threshold_positive_bake_samples':len(bake_comparisons),'below_0_3_positive_bake_samples':sum(r['below_legacy_0_3us'] for r in bake_comparisons),'below_0_5_positive_bake_samples':sum(r['below_legacy_0_5us'] for r in bake_comparisons),'gaps':['Remote historical lab logs named inP1/P2/P3/P4/MX/latencyb8 reports are not all locally archived; reported component effects remainclaims unlessrawverified.','Search indexes identify claims and exact tiny bake differences; they do not independently establish validity/repeatability of every old microvariant.','Early r1/r2/r3/r4/r5 source-only hypotheses have no local official profiles; ledger.v1 empty. Absence oflocalresult isnotzeroorloss.','New jobs may complete after this inventory snapshot; root must attach later immutable artifacts.','Source inclusion follows Sol11 independent AST map; official chronological component detail follows Sol07.']},'independent_sibling_audits':[ref(p) for p in ['proj_rsi/audits/current-leaderboard-gains-official.md','proj_rsi/audits/current-leaderboard-gains-official.refs.json','proj_rsi/audits/current-leaderboard-stack-source.json']],'findings':M,'ranked_reopen_candidates':ranked,'rtk_recomputed':rtkcalc,'cleanroom_zero_threshold_recomputed':clean_result,'official_jobs':official,'positive_bake_samples_no_killbar_filter':bake_comparisons,'ledger_experiment_decisions':[{'id':k,'candidate':v.get('candidate'),'decision':v.get('decision'),'hypothesis':v.get('hypothesis')} for k,v in ledger['experiments'].items()],'candidate_inventory':candidates,'searched_artifact_inventory':inventory,'claim_search_index':claim_index}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['coverage'],indent=2));print('findings',len(M),'json bytes',OUT.stat().st_size)
