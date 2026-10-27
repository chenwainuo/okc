"""CPU-only actual LUT stage timing/resource/routing analysis. No new proof."""
from pathlib import Path
import hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
lut=ROOT/'results/proj-rsi/scheduler/artifacts/research.moe.ss.decode.lut.stage.01/collection-1793130937294077992/logs/scheduler/moe-ss-decode-lut-stage-01'
warm=ROOT/'results/proj-rsi/scheduler/artifacts/research.moe.ss.mapped.grid.diag.01/collection-1793130372740550883/logs/scheduler/moe-ss-mapped-grid-diag-01'
anchor=ROOT/'results/proj-rsi/scheduler/artifacts/research.moe.ss.true.serving.pilot.01/collection-1793128334575874073/logs/scheduler/moe-ss-true-serving-pilot-01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(lut/'result.json')=='d953d950c4a14bec0253d4086a48a2766060726efa0b2cb0bdbacd48a296abf9'
a=json.loads((lut/'result.json').read_text());w=json.loads((warm/'result.json').read_text());orig=json.loads((anchor/'result.json').read_text());assert a['validated_direct_timings_completed']and a['state']=='VALID_M8_SEPARATE_DECODE_LUT_STAGE_SCREEN_ROOT_REVIEW_REQUIRED'
SMs=orig['device']['SMs'];assert SMs==132
hist={}
for r in w['raw']:
 if r['kind']=='paired':
  if r['seed']in hist:assert hist[r['seed']]==r['routing']
  hist[r['seed']]=r['routing']
resources={r['DIV']:r for r in a['jit'][0]['resources']}
original_resources={r['DIV']:r for r in orig['jit'][0]['actual_resources']if r['T']==8}
report={'state':'ACTUAL_VALID_LUT_STAGE_WINS_BOTH_NO_WHOLE_SERVING_ADMISSION','result':str((lut/'result.json').relative_to(ROOT)),'result_sha256':sha(lut/'result.json'),'wall_seconds':a['elapsed_seconds'],'correctness_records':len(a['correctness']),'own_stage_bitwise_records':sum('bitwise_equal'in x for x in a['correctness']),'own_terminal_bitwise_records':sum(x.get('kind')=='own_terminal_before_later_stage'and x.get('bitwise_equal')for x in a['correctness']),'full_serving_reference_records':sum('actual_serving_capture'==x.get('kind')for x in a['correctness']),'current_input_and_immutable_checks':'driver mandatory all timed calls + source/bank terminal PASS','raw_event_calls':len(a['raw']),'F1':{'collected':False,'reason':'This bounded driver records balanced paired stage timings only;3warmups/arm are untimed. Do not substitute another job F1.'},'profiles':a['profiles'],'stage':{},'per_seed':[]}
assert all(x.get('pass',True)and x.get('bitwise_equal',True)for x in a['correctness'])
for stage,div,n in [('up',8,8192),('down',1,4096)]:
 pairs=a['paired'][stage]['raw'];old=[p['original']for p in pairs];new=[p['full_lut']for p in pairs];delta=[x-y for x,y in zip(old,new)]
 report['stage'][stage]={'DIV':div,'N':n,'rounds':len(pairs),'original_median_us':statistics.median(old),'LUT_median_us':statistics.median(new),'median_paired_saving_us':statistics.median(delta),'signed_paired_savings_us':delta,'median_paired_original_over_LUT':statistics.median(x/y for x,y in zip(old,new)),'wins':sum(d>0 for d in delta),'losses':sum(d<0 for d in delta),'original_resources':original_resources[div],'LUT_resources':resources[div]}
for i,seed in enumerate(range(7900,7908)):
 h=hist[seed];b=h['active_local_blocks'];assert b==sum((count+7)//8 for expert,count in h['counts'].items()if int(expert)>=0)
 row={'seed':seed,'active_local_expert_blocks':b,'local_routes':sum(c for e,c in h['counts'].items()if int(e)>=0),'counts':h['counts'],'histogram_source':'same CUDA13 pinned reference generate_input seed and case, prior warmgrid raw routing; not recaptured in LUT result','stage':{}}
 for stage,div,tile_count in [('up',8,128),('down',1,64)]:
  ct=b*tile_count;old=original_resources[div]['occupancy']*SMs;new=resources[div]['occupancy']*SMs;pair=a['paired'][stage]['raw'][i]
  row['stage'][stage]={'valid_CTAs':ct,'original_CTA_capacity_all_SMs':old,'LUT_CTA_capacity_all_SMs':new,'idealized_original_waves':math.ceil(ct/old),'idealized_LUT_waves':math.ceil(ct/new),'original_us':pair['original'],'LUT_us':pair['full_lut'],'signed_saving_us':pair['original']-pair['full_lut']}
 report['per_seed'].append(row)
report['wave_model_limit']='ceil(validCTAs/(132SM*queriedactiveCTAperSM)) is an idealized capacity bound, not measured scheduling. Actual grid also includes short invalid buckets/tail CTAs. Timing bands are consistent with thresholds but do not uniquely prove cause.'
sums=[]
for i in range(8):
 up=a['paired']['up']['raw'][i];down=a['paired']['down']['raw'][i]
 sums.append({'seed':7900+i,'original_stage_sum_us':up['original']+down['original'],'LUT_stage_sum_us':up['full_lut']+down['full_lut'],'both_LUT_saving_us':up['original']+down['original']-up['full_lut']-down['full_lut'],'LUT_up_original_down_saving_us':up['original']-up['full_lut'],'original_up_LUT_down_saving_us':down['original']-down['full_lut']})
report['mixed_stage_estimates']={'scope':'stage sums only; up/down measured in separate phases, not full-serving measurements','per_seed':sums,'median_both_LUT_stage_saving_us':statistics.median(p['both_LUT_saving_us']for p in sums),'median_LUT_up_only_saving_us':report['stage']['up']['median_paired_saving_us'],'median_LUT_down_only_saving_us':report['stage']['down']['median_paired_saving_us'],'decision':'Both LUT stages win every pair; observed evidence favors both, with no useful performance reason to retain either original stage. Root is preparing same-binary whole-serving anchor/SS/LUT/LUT+bound comparison.'}
production=(lut/'actual_lut.cu').read_text();kernel=(ROOT/'serving/experiments/moe_rsi_20261027/ss_decode_lut_v1/full_table/kernel.cuh').read_text();assert production.startswith(kernel)and 'return lut[s*16u+q];'in production
sass=[]
for f in (lut/'actual_lut.sass').read_text().split('Function : ')[1:]:
 if '7projectILi'not in f.splitlines()[0]:continue
 div=int(re.search(r'projectILi8ELi(\d+)E',f)[1]);rows=re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)
 sass.append({'DIV':div,'instructions':len(rows),'LOP3':sum('LOP3'in ins for _,ins in rows),'LDS_U16':sum('LDS.U16'in ins for _,ins in rows),'HGMMA':sum('HGMMA'in ins for _,ins in rows),'decoder_LDS_offset_0x4820':sum('LDS.U16'in ins and '+0x4820]'in ins for _,ins in rows)})
report['actual_LUT_path']={'production_source_matches_candidate_prefix':True,'kernel_sha256':sha(ROOT/'serving/experiments/moe_rsi_20261027/ss_decode_lut_v1/full_table/kernel.cuh'),'augmented_source_sha256':sha(lut/'actual_lut.cu'),'production_source_sha256':a['jit'][0]['production_sha256'],'SO_sha256':sha(lut/'actual_lut.so'),'PyInit_name':'ss_lut_stage_cu130_dd3f2f935a82b8f1ee30','actual_SASS':sass,'shared_table':'8192B full signed FP4code/FP8scale; unchanged global after projection BF16 cast','hot_load_address':'shared byte offset0x4820=18464 after18432Boperandring+32Bbarriers','no_spills':all(r['local']==0 and r['binary_resources']['STACK']==0 for r in resources.values())}
report['next_iteration']={'immediate':'Reuse exact measured SO in whole-serving anchor/SS/LUT/LUT+bound warm comparison; no recompile or broad suite needed to establish transfer. Already coordinated with moe_result_repair.','next_kernel_hypothesis':'Fuse per-value lookup directly into its existing shared scatter to avoid live values[8]; target up<=32 and down<=48 registers to restore queried occupancy8/5, preserving8KBtable, offsets, all waits and matrix instructions. If insufficient, bounded producer step-loop unroll1 changes register lifetime/loop overhead; inspect two-function offline codegen before one stage screen.','why':'B7/9 wave thresholds show the new register footprint can erase a substantial part of otherwise consistent LUT benefit. This is a resource/codegen hypothesis, not proven cycle attribution.','do_not':'Do not call this a3x-anchor/full14/whole-serving gain; do not reject entire LUT family or assume lowerinstructioncount alone proves speed.'}
next_codegen=ROOT/'serving/experiments/moe_rsi_20261027/ss_decode_lut_v2/codegen.json'
if next_codegen.exists():
 report['offline_next_iteration']={'codegen_path':str(next_codegen.relative_to(ROOT)),'variants':json.loads(next_codegen.read_text()),'candidate':'v1fullLUTup plusv2fused_rolled_stepdown; noGPU timingclaim','interpretation':'Purefusion fails target; producerstep unroll1 reduces offline down to48registers, potentially restoring5activeCTAsSM. Must gate actualCUDA13/resources and measure loop-overhead/ILP tradeoff; staticinstructioncompaction does not reduce dynamic lookup traffic.'}
(D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k]for k in ['state','stage','mixed_stage_estimates']},indent=2))
