"""CPU-only unsealed v7 draft; never updates the live registry or v6 archive."""
import hashlib,json
from copy import deepcopy
from pathlib import Path
from proj_rsi import learning as L
R=L.ROOT
D=R/'proj_rsi/audits/learning-v7-draft'
D.mkdir(exist_ok=True)
base=R/'proj_rsi/learning_archive/registry-v6-18f19bb1.json'
data=json.loads(base.read_text());data['registry_version']=7
template=deepcopy(data['lessons'][-1])
def pin(path,role):
 p=R/path;raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();out=D/'evidence'/(p.parent.name+'-'+p.stem+'-'+sha[:12]+p.suffix);out.parent.mkdir(exist_ok=True)
 if out.exists(): assert out.read_bytes()==raw
 else: out.write_bytes(raw)
 return {'path':str(out.relative_to(R)),'sha256':sha,'role':role}
def add(id,title,jobs,sources,evidence,facts,scope,hypothesis,limits,falsifier,question,discriminator,roi,validity,disposition):
 x=deepcopy(template);x.update(id=id,title=title,tags=[id,'scope','counterevidence'],related_queue_ids=jobs,scope=scope,record_sha256=None)
 ss=[pin(p,r)for p,r in sources];x['context'].update(experiment_id=jobs[0],family=id,parents=[s['role']+'@'+s['sha256']for s in ss],sources=ss,comparators=[r for p,r in sources],hardware={'kind':'GPU','model':'Archived H100 local attempts only; CPU evidence backfill'},validity=validity,disposition=disposition)
 x['evidence']=[];payload={}
 for label,path in evidence:
  e=pin(path,label);e.pop('role');e['id']=label;x['evidence'].append(e);payload[label]=json.loads((R/e['path']).read_text())
 x['observations']=[{'id':n,'statement':s,'outcome':o,'evidence':e,'selector':{'json_pointer':p,'equals':L.pointer(payload[e],p)}}for n,s,o,e,p in facts]
 x['hypotheses']=[{'statement':hypothesis,'status':'unproven','observation_ids':[f[0]for f in facts]}];x['transfer_limits']=limits;x['falsification']=falsifier
 x['next_test'].update(question=question,controls=x['context']['comparators'],discriminator=discriminator,required_gates=['Preserve the exact named source/fixture and attempt evidence relevant to this diagnostic; no general admission gate is introduced.'])
 x['conditional_benefit']['conditions']=[roi,'No new GPU reservation, queue action, quota assumption or measured speed forecast follows.'];data['lessons'].append(x)
add('empty-oracle-domain-before-optional-counters','An empty routing fixture needs its defined zero oracle, not a rejecting reference',['research.moe.ss.v1.up.counters.01','research.moe.ss.v1.up.counters.v2.01'],[('tools/moe_ss_v1_up_counters_v2/probe.py','separate repaired probe; same numerical kernels')],[('invalid','tools/moe_ss_v1_up_counters/actual-analysis/analysis.json'),('repair','tools/moe_ss_v1_up_counters_v2/actual-analysis/analysis.json')],[
 ('no_metrics','Original probe failed before counter collection; completed nonempty checks do not create hardware metrics.','negative','invalid','/counter_metrics_collected'),
 ('cause','Original reference intentionally rejects an all-nonlocal fixture; exact-zero branch is the defined repair.','negative','invalid','/cause'),
 ('empty_zero','Repaired seed7779 validates actual serving output exact zero without weakening the nonempty reference.','positive','repair','/correctness/2/message'),
 ('same_fixtures','Recorded tensor hashes match the original G6/G7 fixtures.','positive','repair','/fixtures_tensor_hashes_equal_original_attempt'),
 ('metrics','Repaired attempt independently verifies two actual nodes and eight metrics/units.','positive','repair','/exact2nodes_all8metrics_units_independently_verified'),
 ('cleanup','Repaired attempt externally exits cleanly.','positive','repair','/cleanup/terminal_exit')],
 'One optional M8 SS up counter probe and separately frozen empty-oracle repair; no kernel performance comparison. Nonempty task and bitwise stage checks remain unchanged.',
 'Fixture/reference domain mismatch caused the setup failure, rather than unsupported counters or numerical kernel failure. A domain-specific empty oracle repairs this exact mismatch.',
 ['Original attempt remains INVALID_SETUP with no counters; later repair does not retroactively validate it.','Counter replay durations197.472/312.320us are instrumented different workloads, not clean speedup evidence.','Low DRAM/tensor percentages do not identify producer/consumer or source stall causes.'],
 'A fixture with local routes enters the zero-only branch, or the empty output contains any nonzero byte; then the narrow repair is invalid.',
 'Does the next existing probe select its oracle from the actual fixture domain?',
 'Reuse CPU fixture-branch checks and current exact runtime output assertions in the already justified next probe; retain nonempty reference and empty byte-zero checks separately. No repeat profiler run solely for registry learning.',
 'Avoids a repeated invalid optional probe; no speed gain is measured. Existing performance work retains priority.', 'VALIDATED_DIAGNOSIS','VALID_MIXED')
add('cleanup-residual-needs-origin-proof','A 32MiB post-drain residual is not proven cuBLAS workspace ownership',['research.gv.codec.frontier.two-cell.01'],[('proj_rsi/candidates/gv_codec_frontier_v1/native_v2_capacity32_48/driver.py','actual failed cleanup driver')],[('attempt','proj_rsi/candidates/gv_codec_frontier_v1/attempt01-analysis/report.json'),('header','proj_rsi/audits/coordinator-observations-20261027/torch28-cublas-header-inspection.json')],[
 ('api','Actual targetimage header declares supported workspace map and clear APIs; this proves availability only.','neutral','header','/declarations'),
 ('invalid','Completed attempt remains INVALID_CLEANUP.','negative','attempt','/disposition'),
 ('residual','Internal post-scope allocation is exactly33554432bytes despite released run aliases.','negative','attempt','/internal_cleanup'),
 ('external','External process/container cleanup succeeds while trial exits1.','neutral','attempt','/external_cleanup/terminal_exit'),
 ('unproven','Version-matched workspace explanation is consistent but allocation origin was not directly measured.','neutral','attempt','/causal_status'),
 ('loss_pooler','Preserve unadmitted pooler paired loss-5.408000us without performance admission.','negative','attempt','/cells/pooler/unadmitted_paired_control_minus_new_us'),
 ('loss_unpooler','Preserve unadmitted unpooler paired loss-12.656000us without performance admission.','negative','attempt','/cells/unpooler/unadmitted_paired_control_minus_new_us')],
 'One C11 two-cell attempt: completed numerics/timings, failed internal allocation assertion, successful independent external cleanup. No direct workspace map/byte attribution was captured.',
 'Oracle BLAS calls may leave a version-specific cached workspace. Size agreement and an available clear API are not allocation-origin proof; graph, reference and allocator ownership must be observed separately.',
 ['Immutable actual Torch2.8 header-inspection receipt establishes API declarations only, not execution, allocation ownership or successful cleanup.','External GPU0MiB after process exit does not satisfy the failed internal assertion.','New post-drain cleanup may validate a new attempt only; frozen old data/disposition stay unchanged.','Do not clear workspace during live graph use or alter numerical tolerances to hide retained storage.'],
 'The actual workspace map has different/no bytes or33554432bytes remain after a measured safe post-drain clear; then cached cuBLAS workspace alone cannot explain the residual.',
 'What actual allocation owns the32MiB, and does safe post-drain release remove exactly that residual?',
 'In the next meaningful source comparison, record actual version-matched workspace count/bytes and Torch allocated/reserved before/after graph drain and supported clear. Match the byte delta and retain failures; no identical full rerun solely to clean up history.',
 'Information value is a valid lifetime contract for the next meaningful experiment; neither a speed gain nor a codec-family rejection follows.', 'INVALID_CLEANUP','INVALID_CLEANUP')
add('marlin-four-blocks-cache-and-mixed-estimators','Marlin four-block paired progress coexists with negative brackets and a cache-policy change',['research.moe.original.marlin.fourblocks.01'],[('tools/moe_marlin_four_blocks_v1/pilot.py','measured original/defaultshim/forced4 driver')],[('result','proj_rsi/audits/moe-original-marlin-fourblocks-result/analysis.json'),('routes','proj_rsi/audits/moe-original-marlin-fourblocks-result/route-signs.json')],[
 ('paired','Actual original-minus4block paired median is+0.591997us; retain10wins/2losses.','positive','result','/paired/anchor_minus_contrast'),
 ('shim','Defaultshim-minus4block median is+0.463996us; module/control difference remains explicit.','positive','result','/paired/default_shim_minus_contrast'),
 ('F1','Both original F1 endpoints lose-1.535997/-2.560005us.','negative','result','/F1_control_minus_contrast/anchor_minus_contrast'),
 ('capacity','Actual query changes maximum active CTA capacity3to4 at different shared allocation; not measured scheduling.','neutral','result','/four_block_feasibility/queries'),
 ('cache','Source A-cache capacity changes444to293rows and13to9stagegroups; dynamic reload/stall fractions are unmeasured.','neutral','result','/source_Acache_capacity_model_not_dynamic_load_counts'),
 ('validity','All22correctness checks pass.','positive','result','/correctness_checks'),
 ('cleanup','Trial exits0 with independently clean external scope.','positive','result','/cleanup')],
 'Local original Marlin M8 whole-serving screen versus exact original and default shim,12paired seeds; bracketed medians and single-replay profiles remain separate. No full14/H200/model stack result.',
 'Scheduling and smaller A-cache interact with routing/schedule context. Positive paired point supports a useful measured version, but opposite brackets and module difference prevent unique occupancy or route-causal attribution.',
 ['Query3to4 is maximum active CTA capacity, not measured residency or achieved scheduling.','Shared76458to57088bytes changes cache policy as well as capacity, so this is not an isolated occupancy intervention.','Each F1 block reuses seeds7800–7804, so internal brackets use the same bank. Paired seeds7900–7911 and blocked/interleaved schedules differ; route histograms alone do not establish why estimators differ.','Do not add isolated gains, transfer profiles into clean event medians, or reject the family because brackets lose.'],
 'On the same expanded frozen fixture bank under both schedules the paired positive disappears, or gains do not track the proposed routing context; retain each sign instead of inventing a selector.',
 'Does the four-block progress persist when workload bank and scheduling context are matched?',
 'If root prioritizes it, reuse exact binaries in one expanded normal/skewed seed bank with matched bracket fixture identity and balanced interleaving; record current routes/own terminals and all signed estimates. No unchanged full screen or forced policy promotion.',
 'Retain the valid small paired progress without a significance/minimum-size gate. One workload/schedule diagnostic can price broader integration; transferable package gain remains unknown.', 'VALID_LOCAL','VALID_MIXED')
(D/'registry-unsealed.json').write_text(json.dumps(data,indent=2)+'\n')
(D/'status.json').write_text(json.dumps({'state':'DRAFT_UNSEALED_NOT_LIVE_NOT_REVIEWED','base':str(base.relative_to(R)),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'new_lessons':[x['id']for x in data['lessons'][20:]],'preserved_v6_prefix':True,'new_record_digests':None,'new_ideas':0,'tests':'No existing tests rerun; draft evidence pointers resolved during build.','runtime_header_proof':'Actual targetimage CPU header receipt pinned for API availability only; no runtime ownership attribution.','GPU_SSH_API_queue_git_actions':0,'contract':'proj_rsi/ITERATION.md'},indent=2)+'\n')
print(D/'registry-unsealed.json')
