"""Append reviewed evidence-bound lessons to v5; CPU-only, immutable snapshots."""
from copy import deepcopy
import argparse,hashlib,json
from pathlib import Path
from proj_rsi import learning as L
R=L.ROOT;BASE='proj_rsi/learning_archive/registry-v5-79674136.json';A=R/'proj_rsi/learning_archive/evidence-v6';DRAFT=R/'proj_rsi/audits/learning-v6-draft.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(label,path):
    p=R/path;raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();A.mkdir(parents=True,exist_ok=True)
    out=A/(label+'-'+digest[:12]+p.suffix)
    if out.exists():assert out.read_bytes()==raw,'immutable evidence conflict'
    else:out.write_bytes(raw)
    return str(out.relative_to(R))
def build():
    data=json.loads((R/BASE).read_text());data['registry_version']=6
    template=deepcopy(data['lessons'][0])
    def lesson(id,title,family,job,validity,disposition,hardware,source_paths,evidence,facts,scope,hypothesis,limits,falsifier,nextquestion,controls,discriminator,roi):
        row=deepcopy(template);row.update(id=id,version=1,supersedes=None,title=title,tags=[family,'actual-resources','matched-controls','signed-outcomes'],scope=scope,related_queue_ids=job)
        row['context']={'experiment_id':job[0],'family':family,'parents':[role+'@'+sha(R/path)for role,path in source_paths],'comparators':controls,'hardware':hardware,'validity':validity,'disposition':disposition,'sources':[{'path':snapshot(id+'-source'+str(i),path),'sha256':sha(R/path),'role':role}for i,(role,path)in enumerate(source_paths)]}
        row['evidence']=[];payload={}
        for label,path in evidence:
            archived=snapshot(id+'-'+label,path);row['evidence'].append({'id':label,'path':archived,'sha256':sha(R/archived)});payload[label]=json.loads((R/archived).read_text())
        row['observations']=[{'id':name,'statement':text,'outcome':outcome,'evidence':label,'selector':{'json_pointer':ptr,'equals':L.pointer(payload[label],ptr)}}for name,text,outcome,label,ptr in facts]
        row['hypotheses']=[{'statement':hypothesis,'status':'unproven','observation_ids':[f[0]for f in facts]}];row['transfer_limits']=limits;row['confidence']={'observation':'high','causality':'low','transfer':'low'};row['falsification']=falsifier
        row['next_test']={'question':nextquestion,'controls':controls,'discriminator':discriminator,'stop_rule':'Record the named contrast and all signed outcomes. Missing required validity invalidates only that attempt; neither a loss nor an estimator disagreement imposes a minimum gain/significance gate or rejects a family.','required_gates':['Use the exact source, workload and comparator named for this discriminator.','Check the actual observed state or selected native path; preserve current-input/terminal/cleanup evidence relevant to the measured scope.']}
        row['conditional_benefit']={'kind':'conditional_scenario_not_measured','value':None,'unit':'decision','conditions':[roi,'No benefit forecast, new GPU reservation, quota assumption or promotion follows from this registry update.']};row['cost']={'basis':'Evidence backfill and scoped next question only; future cost must be priced separately by root.','cpu_minutes':None,'gpu_seconds':None};row['record_sha256']=None;return row
    v1='tools/moe_ss_decode_lut_v1/actual-analysis/analysis.json';v2='tools/moe_ss_decode_lut_v2/actual-analysis/analysis.json';scatter='tools/moe_ss_two_value_scatter_v1/actual-analysis/analysis.json';pair='tools/moe_ss_pair2_v1/actual-analysis/analysis.json';whole='tools/moe_ss_lut_whole_v1/actual-analysis/analysis.json'
    src1='serving/experiments/moe_rsi_20261027/ss_decode_lut_v1/full_table/extension_source.cu';src2='serving/experiments/moe_rsi_20261027/ss_decode_lut_v2/down_only/extension_source.cu';srcs='serving/experiments/moe_rsi_20261027/ss_decode_two_value_scatter_v1/extension_source.cu';srcp='serving/experiments/moe_rsi_20261027/ss_pipeline_pair2_v1/extension_source.cu'
    gpu={'kind':'GPU','model':'H100 local132SM, fixed M8 MoE stage scope'}
    rows=[]
    rows.append(lesson('occupancy-versus-producer-ilp-packed-stores','Higher queried CTA capacity can lose when producer ILP and packed stores change','moe-SS-producer-schedule',['research.moe.ss.decode.lut.rolled.down.01','research.moe.ss.two.value.scatter.down.01'],'VALID_LOCAL','VALID_MIXED',gpu,[('exact v1 LUT control',src1),('measured rolled v2',src2),('two-value scatter derivative',srcs)],[('v1',v1),('v2',v2),('scatter',scatter)],[
      ('v1_capacity','The actual v1 down function query returns a maximum active capacity of four CTAs/SM.','neutral','v2','/v1_resources/occupancy'),
      ('v2_capacity','Rolled v2 raises the queried maximum active down capacity to five CTAs/SM.','positive','v2','/actual_v2_resources/0/occupancy'),
      ('v2_loss','Despite 59-to-48 register reduction and higher queried CTA capacity, v2 loses all eight paired down seeds.','negative','v2','/paired/median_v1_minus_v2_us'),
      ('schedule','Actual code changes from eight independent LDS.U16 lookups then four packed word stores to alternating dependent lookup/half-store instructions.','neutral','v2','/producer_schedule_observation/observed'),
      ('scatter_rescue','Restoring two-value packed scatter improves all eight seeds over failed v2.','positive','scatter','/comparisons/v2_rolled_down_versus_two_value_scatter/median_control_minus_variant_us'),
      ('scatter_still_loses','The packed scatter variant still loses all eight seeds to actual v1 LUT.','negative','scatter','/comparisons/v1_full_lut_versus_two_value_scatter/median_control_minus_variant_us'),
      ('counterexample','Earlier exact v1 LUT improved down over original while reducing queried maximum active CTA capacity from five to four CTAs/SM.','positive','v1','/stage/down/median_paired_saving_us')],
      'One actual M8 down-stage v1/v2/scatter sequence with bitwise current-input and mixed full numerical checks. Full-serving speed and official incumbent comparisons were not timed in these stage screens.',
      'Register/liveness reduction traded away independent lookup issue and packed stores. Scatter rescue supports a material schedule/store cost, but cannot assign unique cycles or show occupancy alone caused either result.',
      ['Actual SASS and resource query returns are facts; queried maximum active CTA capacity is not measured residency/scheduling, and causal ILP/latency attribution remains unproven.','A shared producer version loss does not reject rolling, lookup tables, occupancy changes or the SS family.','Do not transfer H100 stage deltas to whole serving, Marlin or H200.'],
      'An equal-resource producer schedule retaining v1 lookup independence and packed stores does not change the measured contrast, or an independently attributed consumer bottleneck dominates.',
      'Does producer dependency/store formation explain the remaining v1-versus-scatter gap?',
      ['exact measured v1 LUT down','failed rolled v2 down','measured two-value scatter down'],
      'At one fixed M8 workload, distinguish the retained producer schedule from consumer/startup costs using only an already justified scoped contrast. Query actual resources and inspect realized stores; do not launch a broad occupancy sweep or infer elapsed-time savings from opcode counts.',
      'Retain the measured scatter rescue and v1 counterwin. Information value is choosing producer versus consumer work before another rewrite; attainable gain remains unknown.'))
    rows.append(lesson('grouped-stages-sync-versus-shared-residency','Halving sync rounds can lose with a larger ring; retain both B7 gains','moe-SS-grouped-stages',['research.moe.ss.lut.pair2.stage.01'],'VALID_LOCAL','VALID_MIXED',gpu,[('exact v1 LUT control',src1),('measured pair2 source',srcp)],[('pair',pair),('v1',v1)],[
      ('realized','Actual eight MMAs/group and 32 logical rounds were verified; reduced sync rounds did not produce aggregate stage gain.','neutral','pair','/causal_interpretation/0'),
      ('up_loss','Pair2 up paired median saving is negative.','negative','pair','/stage/up/paired_median_saving_us'),
      ('down_loss','Pair2 down paired median saving is negative despite unchanged queried maximum capacity of four CTAs/SM.','negative','pair','/stage/down/paired_median_saving_us'),
      ('up_capacity','Pair2 up queries a maximum active capacity of four CTAs/SM versus v1 six.','negative','pair','/stage/up/actual_candidate_resources/occupancy'),
      ('B7a','The first B7 up context retains a positive 51.232010us paired saving.','positive','pair','/stage/up/pairs/3/saving_us'),
      ('B7b','The second B7 up context retains a positive 51.551998us paired saving under opposite arm order.','positive','pair','/stage/up/pairs/6/saving_us'),
      ('shared','Measured pair2 dynamic shared allocation is 46112 bytes.','neutral','pair','/stage/up/actual_candidate_resources/dynamic_shared')],
      'Fixed M8 EP8 normal up/down stages versus exact v1 LUT, eight routing seeds with frozen-record F1. Mixed full serving was checked numerically, not timed. B7 identifies these two observed routing contexts, not a validated dispatch policy.',
      'A larger shared ring reduces queried maximum active up CTA capacity, while actual scheduling, batching/startup and producer ILP remain confounded; down losses at equal queried capacity show this capacity alone is insufficient. B7 gains may reflect schedule/wave context rather than a general group-size win.',
      ['Idealized wave counts are a source/resource model, not measured CTA scheduling.','Do not average away B7 positives or declare B7-selectable whole-serving coverage.','Fewer barriers/sync rounds is not a speed forecast; one pair2 loss is not a grouped-stage family rejection.'],
      'An equal-queried-capacity grouping contrast does not preserve B7 gains or improve the named stage; actual scheduling attribution contradicts the wave hypothesis.',
      'Can grouped-stage work preserve useful B7 progress without losing queried CTA capacity or equal-capacity down performance?',
      ['exact v1 LUT up/down','measured pair2 up/down','both observed B7 contexts plus B6/B9/B11 countercontexts'],
      'Preserve both signed B7 outcomes and compare only a resource-accounted grouping refinement or existing producer/consumer diagnostic. Keep the same frozen workload per F1 bracket; do not hardcode a B7 winner or repeat a broad group sweep.',
      'B7 progress remains evidence for a specific future causal question. Full-package gain and useful runtime coverage are unknown, so next work needs its own bounded ROI price.'))
    rows.append(lesson('f1-exact-workload-not-distinct-seeds','An F1 bracket needs the same workload; distinct routing seeds are confounded','measurement-workload-identity',['research.moe.ss.lut.whole.01','research.moe.ss.decode.lut.rolled.down.01'],'VALIDATED_DIAGNOSIS','VALID_MIXED',gpu,[('whole-serving measurement driver','tools/moe_ss_lut_whole_v1/pilot.py'),('frozen-record stage driver','tools/moe_ss_decode_lut_v2/pilot.py')],[('whole',whole),('fixed',v2)],[
      ('unmatched','The whole-serving descriptive F1 groups used distinct input/routing fixtures.','negative','whole','/F1_design/matched_fixtures'),
      ('identity_code','Its block-specific seed formula changed the workload between F1 arms.','negative','whole','/F1_design/code'),
      ('paired_valid','Balanced paired whole-serving comparisons retained identical routing across all arms.','positive','whole','/paired_fixture_routing_equal_all_arms'),
      ('repair','The later stage F1 bracket reuses one frozen seed7904 record and private graph buffers.','positive','fixed','/fixed_workload_F1/routing_record_identical'),
      ('frozen_hashes','The corrected bracket retains byte hashes for activation, sorted routes, experts and post metadata.','neutral','fixed','/F1_frozen_workload/tensor_byte_sha256'),
      ('real_loss','Both corrected fixed-workload F1 endpoints lose to v1; preserving workload identity does not imply candidate success.','negative','fixed','/fixed_workload_F1/after_minus_v2_us')],
      'MoE whole-serving descriptive groups and subsequent fixed M8 down-stage F1 repair. Routing changes active local expert blocks and therefore actual kernel work. Paired evidence remains valid; unmatched F1 groups are not a controlled stack-effect estimator.',
      'Different active-block and input/routing workloads confound arm effects with workload effects; identical fixtures and byte hashes remove this particular confound while leaving clock/context and finite-sample uncertainty.',
      ['Do not call unmatched-block signs estimator disagreement with matched paired effects.','A common seed label alone is weaker than exact workload/metadata identity; keep hashes or immutable identity receipts.','The later corrected stage bracket neither repairs old measurements retroactively nor establishes a whole-serving result.'],
      'After fixing exact workload identity, the former arm contrast persists in matched pairs and both endpoints; then workload changes alone cannot explain it.',
      'Does a proposed F1 comparison preserve exact activation and routing work across all arms/endpoints?',
      ['same immutable record copied into private arm graph buffers','balanced per-seed paired comparisons','original unmatched groups retained as descriptive context'],
      'For the next named comparison, freeze one actual record before the bracket and retain its tensor/route metadata hashes for every arm. Compare current outputs before mutation. Do not rerun a completed experiment solely to relabel old unmatched medians.',
      'This is measurement-design learning, not a speed gain. It protects interpretation of the already authorized next comparison without creating a new broad proof or GPU gate.'))
    official='proj_rsi/audits/gv53-splitcta-m1-official-results/analysis.json';local='proj_rsi/audits/gv53-splitcta-m1-composed-result/analysis.json';fixedgv='proj_rsi/audits/gv53-splitcta-fixed-result/analysis.json'
    rows.append(lesson('normal-chooser-needs-actual-route-evidence','Source inclusion and fixed wins do not establish selected execution or incumbent gain','prepare-selection-actual-route',['research.complete.splitcta.m1.incremental.01','research.gv53.splitcta.m1.fullpolicy.01'],'VALID_OFFICIAL_CHECK','VALID_MIXED',{'kind':'GPU','model':'H100 local and H200 official archives; separate scoped observations'},[('current GV incumbent','proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),('M1-only optional standalone','proj_rsi/candidates/gv53_splitcta_m1_selection_v1/submission.py'),('exact composed M1 source','proj_rsi/candidates/complete_gv_rq_eh_o32_splitcta_m1_v1/submission.py')],[('official',official),('local',local),('fixed',fixedgv)],[
      ('fixed_win','Matched fixed M1 split-CTA paired improvement remains positive.','positive','fixed','/cells/1/signed_gain_positive_favors_splitcta/paired_median_us'),
      ('local_unselected','The new tuple is unselected in the valid local composed full30 run.','negative','local','/new_tuple_selected_shapes'),
      ('official_unselected','Composed scored ae061d contains the source but selects no new split tuple.','negative','official','/new_tuple_selected_routes/ae061d458eaa420a'),
      ('standalone_selected','Standalone M1 CHECK selects the new tuple only for decode1/pooler.','positive','official','/new_tuple_selected_routes/53e707b4e7ed470e'),
      ('parent_progress','Composed official parent-relative score point improves by 0.397884us; retain it.','positive','official','/comparisons/composed_vs_exact_d1ca/cross_job_score_gain_us_positive_favors_new'),
      ('incumbent_loss','The same composed scored source remains worse than actual best GV.','negative','official','/comparisons/composed_vs_best_GV/cross_job_score_gain_us_positive_favors_new'),
      ('mixed_F1','Local incremental conservative F1 opposes its paired and endpoint-average gains.','negative','local','/actual_policy_geometric_load_speedup/F1_conservative')],
      'Exact GV53 split-CTA lineage: H100 fixed M1, local actual full30 incremental versus d1ca, H200 composed scored ae061d and standalone M1 CHECK53e707. These contexts remain separate; labels identify actual selected profiles, not inferred source execution.',
      'Normal short-screen/finalist selection and preparation context can alter old and new routes. Nonselection can explain why a fixed mechanism is absent, but not attribute whole-package deltas; matched contexts are needed to isolate any cause.',
      ['No cross-job, H100-to-H200, source-commutation-to-additivity or parent-to-incumbent gain transfer.','Official raw events, calibrated objective medians and separate profile durations are distinct; preserve all values.','Normal chooser nonselection is not a legality bug or a reason to force a measured winner; no minimum-size/significance cutoff applies.'],
      'The exact new tuple is selected in a same-context actual-policy test yet the proposed mechanism contrast is absent, or a matched old/new finalist contrast refutes the rank explanation.',
      'Does the new option beat the actual old H100 winner, rather than only the earlier GV53 comparator?',
      ['actual selected old H100 tuple0 WR1/WK8/KU16','new tuple8195','actual incumbent and exact parent package separately'],
      'If still useful after root reprioritization, use the already proposed single-M1 matched old-winner versus new contrast and shadow screen/retest ranks without changing serving choices. Do not make another unchanged full30 selector copy or force promotion.',
      'Keep the tiny official parent-relative positive and all losses. One-cell diagnosis may choose whether selector work has value; neither speed benefit nor remaining quota is inferred.'))
    held='proj_rsi/audits/coordinator-observations-20261027/lut-v2-register-held.json'
    rows.append(lesson('register-go-needs-actual-state-verification','A successful GO registration can remain HELD; verify state before claiming readiness','scheduler-admission-postcondition',['research.moe.ss.decode.lut.rolled.down.01'],'VALIDATED_DIAGNOSIS','VALID_MIXED',{'kind':'CPU','model':'CPU archived root registration receipt and source inspection'},[('registration/rebalance source snapshot, not a runtime binary attestation','proj_rsi/scheduling/store.py')],[('held',held)],[
      ('go','The actual archived register response contains independent GO.','positive','held','/review/decision'),
      ('held','The same returned descriptor is HELD, so registration plus GO did not make it READY.','negative','held','/state'),
      ('id','The observation concerns the actual rolled-down GPU job, not a synthetic queue example.','neutral','held','/id'),
      ('dependency','The response also retains an explicit valid-result dependency, which must be checked separately.','neutral','held','/depends_on')],
      'Byte-exact root register stdout archived for one actual GPU descriptor. CPU diagnosis uses that returned state and current register/rebalance source semantics; no live WAL or mutable checkpoint is pinned and no queue action is performed.',
      'Registration preserves descriptor intent and eligibility/state transitions are separate from independent source GO. Explicit HELD state and dependency/eligibility can block readiness; the receipt alone does not identify every later transition.',
      ['GO is source/launch review evidence, not READY/RUNNING or physical GPU utilization.','Do not generalize one HELD receipt into an automatic release policy; root authority and actual eligibility reasons still apply.','The archived response proves the returned state at registration, not current live state or successful subsequent launch.'],
      'A pinned post-action readback shows READY with satisfied dependencies and eligible source/launch bindings; then this particular HELD blocker has been resolved.',
      'Did the authorized registration/transition actually produce the expected eligible state?',
      ['exact registered descriptor and GO binding','actual post-action state/reasons','dependency disposition and exclusive worker state'],
      'After an already authorized root mutation, inspect its completion and actual returned/readback state and reason. If HELD remains, diagnose that reason and apply only the authorized explicit release or repair; verify again before claiming queued/active work. This registry neither registers nor transitions jobs.',
      'This avoids idle eligible-queue mistakes; no measured GPU speedup, automatic admission rule or new broad checklist is claimed.'))
    data['lessons']+=rows
    for row in rows:
        idea=deepcopy(data['ideas'][0]);idea.update(id='idea.'+row['id'],version=1,supersedes=None,title=row['next_test']['question'],tags=row['tags'],lesson_refs=[row['id']+'@1'],existing_roi_ids=row['related_queue_ids'],mechanism=row['hypotheses'][0]['statement'],discriminator=row['next_test']['discriminator'],conditional_benefit=row['conditional_benefit'],cost=row['cost'],record_sha256=None);data['ideas'].append(idea)
    return data
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal-review',type=Path);args=p.parse_args();data=build()
    if not args.seal_review:
        DRAFT.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'state':'DRAFT_UNSEALED_INDEPENDENT_REVIEW_PENDING','path':str(DRAFT.relative_to(R)),'new_lessons':5,'registry_version':6},indent=2))
    else:
        assert args.seal_review.is_file(),'independent review receipt required before new digests'
        review=json.loads(args.seal_review.read_text());assert review['decision']=='GO'
        assert review['draft_sha256']==hashlib.sha256((json.dumps(data,indent=2)+'\n').encode()).hexdigest(),'reviewed draft drift'
        for row in data['lessons']+data['ideas']:
            if row['record_sha256']is None:row['record_sha256']=L.digest(row)
        L.validate(data);raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode();digest=hashlib.sha256(raw).hexdigest();archive=R/('proj_rsi/learning_archive/registry-v6-'+digest[:8]+'.json')
        if archive.exists():assert archive.read_bytes()==raw,'immutable archive conflict'
        else:archive.write_bytes(raw)
        Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n');L.DEFAULT.write_bytes(raw);print(json.dumps({'state':'SEALED_VALID_CPU_ONLY','archive':str(archive.relative_to(R)),'sha256':digest,'review_sha256':sha(args.seal_review)},indent=2))
