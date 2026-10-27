"""Build private unsealed records only; never writes live registry."""
import copy, hashlib, json
from pathlib import Path
OUT=Path(__file__).parent
BASE=Path('proj_rsi/learnings.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base_sha=sha(BASE)
assert base_sha=='75966e64cf37677978e17d965ac6b9e092dfd3ffba720caf925fa7da9079560e'
registry=json.loads(BASE.read_text()); latest={x['id']:x for x in registry['lessons']}
paths={
 'thin':'proj_rsi/audits/gv53-thin-cabi-validation-result-v1/analysis.json',
 'top8':'proj_rsi/candidates/gv_top8_exponent_stats_v1/actual-analysis/analysis-v1.json',
 'layout':'proj_rsi/audits/gv53-payload-layout-feasibility-v1/accounting.json',
 'quarter':'proj_rsi/audits/gv53-quarter-stream-result-v1/analysis.json',
 'sparse':'proj_rsi/audits/gv53-quarter-sparse-result-v1/analysis.json',
 'profile':'proj_rsi/audits/gv53-exactso-counter-result-v1/analysis.json',
 'quarter_native':'proj_rsi/audits/gv53-quarter-stream-result-v1/native-analysis.json',
 'sparse_native':'proj_rsi/audits/gv53-quarter-sparse-result-v1/native-analysis.json'}
docs={k:json.loads(Path(p).read_text()) for k,p in paths.items()}
def ptr(d,p):
 for k in p.strip('/').split('/'):
  k=k.replace('~1','/').replace('~0','~');d=d[int(k)] if isinstance(d,list) else d[k]
 return d
def obs(r,id,e,p,s,outcome='neutral'):
 r['observations'].append(dict(id=id,statement=s,outcome=outcome,evidence=e,selector={'json_pointer':p,'equals':ptr(docs[e],p)}))
def evidence(r,keys):
 for k in keys:r['evidence'].append(dict(id=k,path=paths[k],sha256=sha(paths[k])))
def source(r,p,role):r['context']['sources'].append(dict(path=p,sha256=sha(p),role=role))
def record(id,title,keys,existing=False):
 if existing:
  r=copy.deepcopy(latest[id]);r.update(version=r['version']+1,supersedes=r['record_sha256'],record_sha256=None,title=title)
 else:
  r=copy.deepcopy(latest['exact-so-reuse-bounds-robustness-cost']);r.update(id=id,version=1,supersedes=None,record_sha256=None,title=title,evidence=[],observations=[],hypotheses=[])
 r['context']=dict(experiment_id='; '.join(docs[k].get('job_id',k) for k in keys),family='muse_bf16_proj',parents=['incumbent projection 501d lineage'],comparators={'thin':['same native kernel PyBind versus thin C ABI','earlier build cost descriptive only'],'top8':['old aligned range versus best affine8 versus global top8'],'layout':['planar versus direct AoS48 versus cooperative flat AoS under fixed ownership'],'quarter':['control versus quarter versus sparse; clean timing separate from profiling']}[keys[0]],hardware={'kind':'GPU','model':'H100; CPU accounting explicitly separated'},validity='VALIDATED_DIAGNOSIS',disposition='CPU_ACCOUNTING',sources=[])
 r['related_queue_ids']=[docs[k]['job_id'] for k in keys if 'job_id' in docs[k]] or ['CPU.gv53.payload.layout.feasibility.v1']
 r['tags']=['projection','learning-v9']
 r['conditional_benefit']={'kind':'conditional_scenario_not_measured','value':None,'unit':'decision','conditions':['No kernel or official score gain is forecast; root owns any follow-up admission.']}
 r['confidence']={'observation':'high','causality':'low','transfer':'low'}
 evidence(r,keys)
 return r
def finish(r,scope,limits,falsifier,cost):
 r.update(scope=scope,transfer_limits=limits,falsification=falsifier,cost={'basis':cost,'cpu_minutes':None,'gpu_seconds':None})
 r['next_test']={'question':falsifier,'controls':['Reuse unchanged proofs and frozen evidence within their assumptions.'],'discriminator':falsifier,'stop_rule':'No unchanged rerun or new broad campaign solely to populate learning.','required_gates':['Reuse existing scope-appropriate validity requirements; this learning proposal adds no gate.']}
 return r
records=[]
r=record('exact-so-reuse-bounds-robustness-cost','Exact-binary reuse and a thin ABI bound diagnostic cost, not kernel gains',['thin'],True)
source(r,'proj_rsi/candidates/gv53_thin_cabi_validation_v1/driver.py','thin ABI driver')
obs(r,'thin-build','thin','/build_seconds','Thin native compile 1.609522411s + link 0.048527525s = 1.658049936s; total driver 5.985287921s. Earlier PyBind 42.186037737s is descriptive, not a controlled 40s saving.')
obs(r,'thin-lifetime','thin','/lifetime_probe','Strong owners survive dropped caller references and nondefault capture/different replay streams; graph reset/drain precedes owner release.')
for i in range(4):
 obs(r,f'interface-{i}','thin',f'/all_signed_interface_timing/{i}', 'Identical native kernel/interface timings retain both signs and all F1 endpoints; these are interface replay variability, not native or official score gains.','mixed')
finish(r,'Extends prior exact-SO three-weight cost evidence; thin ABI is one same-kernel transport diagnostic, not a controlled PyBind build-cost experiment.',['Raw-pointer ABI transfer requires validated metadata, strong ownership and correct current stream.','Prior robust aggregate paired -0.063999556us remains a loss; cheaper testing does not improve kernels.','Do not extrapolate this ABI to different structs, clusters or binaries.'],'Reopen only for a new causal comparison whose exact SO can be reused, or a changed ABI requiring a bounded lifetime check.','Measured thin compile/link 1.658049936s, total 5.985287921s; paired build savings unknown.')
records.append(r)
r=record('rq11-full-residency-ledger','Charge escape capacity and complete residency before claiming narrow-payload savings',['top8'],True)
source(r,'proj_rsi/candidates/gv_top8_exponent_stats_v1/stats.py','weight-only diagnostic')
for i,row in enumerate(docs['top8']['rows']):
 label=f"seed {row['seed']} {row['layer']}"
 obs(r,f'top8-affine-{i}','top8',f'/rows/{i}/dictionary_coverage_gain_vs_best_affine',label+': arbitrary global top8 equals best affine8 coverage; additional dictionary coverage is zero.')
 obs(r,f'top8-capacity-{i}','top8',f'/rows/{i}/codebooks/global_top8_exp8',label+': every group exceeds both 32 and 64 repairs. Nominal 11-bit fixed byte total is UNREPRESENTABLE in original 128B packet; extra overflow byte estimate excludes indices/alignment.','negative')
 if row['layer']=='unpooler':obs(r,f'top8-reduction-{i}','top8',f'/rows/{i}/old_aligned_to_best_affine_repair_reduction',label+': odd affine base115 reduces repairs by this count, yet every group still exceeds capacity.','mixed')
finish(r,'Preserves prior CPU RQ11 +1069056 packed bytes saved / -11085360 residency bytes saved with stock cache and unknown CUDA bit identity. New actual weight-only statistics are separate: seeds5151/60631, two old GV geometries; no decode/prepare/output test.',['Pooler nominal46661640B and unpooler46399496B are not representable streams or measured savings.','Top8 affine equality is observed for four fixtures, not universal.','Capacity failure rejects this encoding-only fixed-packet version; no whole-family or runtime rejection.'],'A new representation must account for overflow/index/padding/transport and retained metadata; reopen only if complete actual-fixture capacity and byte accounting become feasible.','Actual weight-only diagnostic elapsed '+str(docs['top8']['elapsed_seconds'])+'s; no JIT, prepare, activations or performance claim.')
records.append(r)
r=record('gv53-payload-layout-ownership-lower-bound','Planar payload already meets the current ownership/load lower bound',['layout'])
r['context']['hardware']={'kind':'CPU','model':'CPU address and operand ownership accounting'};r['context']['validity']='CPU_VALIDATED'
source(r,'proj_rsi/audits/gv53-payload-layout-feasibility-v1/address_accounting.py','CPU address model')
obs(r,'planar-aligned','layout','/address_models_by_base_mod32/0/planar','Aligned planar1536B payload uses three warp LDG128 and48 sector instruction incidences; these are modeled addresses, not hardware transactions.')
obs(r,'aos-aligned','layout','/address_models_by_base_mod32/0/AoS48','Direct per-lane48B AoS retains1536B and3 loads but doubles aligned sector instruction incidences48 to96.','negative')
obs(r,'misaligned-planar','layout','/address_models_by_base_mod32/16/planar','At base mod32=16 planar has51 sector incidences and49 unique sectors; actual runtime alignment is not established.')
obs(r,'ownership','layout','/concrete_alternative/minimum_scalar_SHFL_instructions_given_original_operand_ownership','Cooperative flat AoS would require at least12 scalar SHFL under original ownership, versus planar0; lower bound, not compiled instruction count.','negative')
r['hypotheses']=[{'statement':'An AoS coalescing-only rewrite improves the payload mover under unchanged ownership.','status':'refuted','observation_ids':['planar-aligned','aos-aligned','ownership']}]
finish(r,'Exact M8 pooler tuple under current16B/lane primitive and original operand ownership. Distinct from generic codec-residency ledgers.',['No measured cache requests, replays, DRAM traffic or timing.','Changing mover primitive or operand ownership can invalidate the bound.'],'Reopen only if actual per-PC requests/alignment contradict the model or a concrete ownership/mover change removes a cost absent here.','CPU-only accounting; no native compile justified solely by coalescing hypothesis.')
records.append(r)
r=record('gv53-quarter-pipeline-control-cost','Earlier first MMA and fewer sparse headers did not improve whole-kernel timing',['quarter','sparse','profile'])
r['context']['disposition']='VALID_MIXED'
evidence(r,['quarter_native','sparse_native'])
for e in ['quarter_native','sparse_native']:
 for symbol,v in docs[e]['functions'].items():
  if 'predxor' not in symbol:
   obs(r,e,e,'/functions/'+symbol,'Actual collected native target matches CPU prototype; first-MMA PRMT4 and quarter462/sparse455 nonNOP establish the intended static schedule, not a timing gain.')
source(r,'proj_rsi/candidates/gv53_quarter_stream_v1/pilot.cu','quarter source')
source(r,'proj_rsi/candidates/gv53_quarter_sparse_header_v1/pilot.cu','sparse source')
for key,msg in [('paired_median_control_minus_new_us','Quarter initial paired control-new -0.144000165us: a loss.'),('F1_endpoint_average_control_minus_new_us','Quarter F1 endpoint-average control-new +0.256000087us; retain mixed sign and endpoint drift.'),('F1_conservative_min_endpoint_control_minus_new_us','Quarter conservative endpoint control-new -0.320000574us.')]:obs(r,'quarter-'+key,'quarter','/rows/0/'+key,msg,'mixed' if key.startswith('F1_endpoint') else 'negative')
for i in range(3):
 obs(r,f'retest-paired-{i}','sparse',f'/comparisons/{i}/paired_median_first_minus_second_us',['Control-quarter -0.191999599us retest loss.','Control-sparse -0.223999843us loss.','Quarter-sparse -0.063999556us loss.'][i]+' Positive first-minus-second favors second; comparisons share24 triples and are not additive.','negative')
 obs(r,f'retest-blocked-{i}','sparse',f'/comparisons/{i}/blocked_endpoint_average_first_minus_second_us','Blocked endpoint-average retains the signed first-minus-second effect from the same trial.','negative')
for i in range(3):
 for metric in ['gpu__time_duration.sum','dram__bytes_read.sum','smsp__inst_executed.sum','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']:
  obs(r,f'profile-{i}-{metric}','profile',f'/arms/{i}/metrics/{metric}','Exact-SO '+docs['profile']['arms'][i]['arm']+': profiled duration is not clean timing gain; bytes/instructions are counters, stall percentage is sampled warp activity, not wall-time or repair-time fraction.')
obs(r,'cause','profile','/cause_confidence/clean_loss_cause','Profile durations rank quarter/sparse/control differently from clean timings; unchanged traffic plus increased instructions establishes a tradeoff, not causal allocation of loss.')
obs(r,'quarter-build','quarter','/cost','Quarter measured compile/load41.838135824s and run45.180085850s; do not substitute this cost into a controlled thin-ABI savings claim.')
r['hypotheses']=[{'statement':'Moving first MMA earlier or skipping empty-unit headers is sufficient to improve this complete kernel.','status':'refuted','observation_ids':['quarter-paired_median_control_minus_new_us','retest-paired-1']},{'statement':'More executed instructions contribute to the clean loss.','status':'unproven','observation_ids':['cause']}]
finish(r,'Measured fixed M8 pooler tuple and weights, clean quarter/sparse trials plus separately profiled exact binaries; no policy/official score win. Earlier first-MMA schedule is realized, whole-kernel benefit is not.',['Quarter native first-MMA PRMT16 to4; sparse nonNOP462 to455 are static changes, not runtime savings.','Profile durations22400/21664/21984ns are not gains; profile ranking opposes clean timings.','Counters do not identify repair PC time, wall-time fractions or a proven bottleneck.','A losing version does not reject quarter/pipeline family; preserve mixed F1 and negative paired estimates.'],'Reopen with a changed load-stage/dependency mechanism that can separate added control cost from useful overlap; reuse exact binaries/proofs where valid, not unchanged broad profiling.','Quarter actual compile/load41.838135824s/run45.180085850s; exact-SO profile native builds0; no new execution requested.')
records.append(r)
proposal=dict(status='PRIVATE_UNSEALED_PENDING_ROOT_REVIEW',base_registry_path=str(BASE),base_registry_sha256=base_sha,base_registry_version=8,proposed_registry_version=9,append_lessons=records,append_ideas=[],live_registry_mutated=False,seal_policy='Root reviews/applies; record_sha256 remains null. No new gate or GPU work requested.')
(OUT/'records.json').write_text(json.dumps(proposal,indent=2,ensure_ascii=False)+'\n')
checks=[]
for r in records:
 em={e['id']:e for e in r['evidence']}
 for e in r['evidence']+r['context']['sources']:assert sha(e['path'])==e['sha256']
 for o in r['observations']:
  e=em[o['evidence']];s=o['selector']; raw=Path(e['path']).read_text()
  assert ptr(json.loads(raw),s['json_pointer'])==s['equals'] if 'json_pointer' in s else s['contains'] in raw
  checks.append(r['id']+'/'+o['id'])
 assert r['record_sha256'] is None
assert sha(BASE)==base_sha
(OUT/'checks.json').write_text(json.dumps(dict(status='FACT_SELECTORS_AND_FILE_HASHES_CHECKED_NOT_REGISTRY_VALIDATION',observations_checked=len(checks),observations=checks,base_unchanged_sha256=base_sha,record_hashes_unsealed=True,live_mutations=0,GPU_API_SSH_queue_git_actions=0),indent=2)+'\n')
print('records',len(records),'observations',len(checks),'recordsSHA',sha(OUT/'records.json'))
