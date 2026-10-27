"""Focused CPU specialization/resource/readiness evidence; no CUDA work."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;C=R/'serving/experiments/moe_rsi_20261027/ss_lut_t64_m3012_v1';P=C.parent/'ss_decode_lut_v1/full_table'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('t64_readiness_driver',D/'pilot.py');pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot);assert 'torch'not in sys.modules
assert (C/'kernel.cuh').read_text().replace('moe_ss_lut_t64_m3012_v1','moe_ss_lut_full_v1')==(P/'kernel.cuh').read_text()
def method(p):return next(n for n in ast.walk(ast.parse(p.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='_marlin_expert_outputs')
assert ast.dump(method(C/'serving_both.py'))==ast.dump(method(P/'serving_both.py'))
source=(R/'serving/sglang/python/sglang/srt/layers/muse_spark_fp4_moe.py').read_text();tree=ast.parse(source)
candidates=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_MARLIN_BLOCK_SIZE_M_CANDIDATES'for t in n.targets))
for block_m in candidates:
 if 3012*8/32/block_m<.9:break
assert block_m==64
cap=((3012*8+(256+1)*(64-1)+3)//4)*4;rows=(cap+63)//64;assert(cap,rows)==(40288,630)
sass=(C/'sass.txt').read_text();proof=pilot.physical(sass);neg=[]
for name,bad in [('missing_wait0',sass.replace('WARPGROUP.DEPBAR.LE','NO_DRAIN',1)),('missing_reader_barrier',sass.replace('BAR.SYNC.DEFER_BLOCKING 0x3, 0x80','NO_READER_BARRIER',1))]:
 try:pilot.physical(bad)
 except AssertionError:neg.append(name)
 else:raise AssertionError('gate admitted '+name)
log=(C/'resource.log').read_text();assert not re.search(r'C75|serializ|[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',log,re.I);regs=list(map(int,re.findall(r'Used (\d+) registers',log)));assert regs==[62,64]
slots=3012*8
mem={'all256_raw_packed_plus_FP8_scales_bytes':256*3*4096*4096*9//16,'local32_additional_SS_banks_bytes':32*3*4096*4096*9//16,'per_arm_cache13_bytes':slots*8192*2,'per_arm_cache2_bytes':slots*4096*2,'per_arm_output_bytes':3012*4096*2,'two_arm_graph_caches_plus_outputs_bytes':2*(slots*(8192+4096)*2+3012*4096*2),'flush_bytes':128<<20,'actual_memory':'Runtimefree/allocated/reserved/peak recorded; strictfree>=32GiBbeforeprepare, >=8GiBaftercapture andallocated<48GiB; totaldevice>=70GB. This is not exhaustive allocatorpeak proof.'}
out={'state':'CPU_ONLY_M3012_T64_FEASIBLE_RUNTIME_UNMEASURED','source_delta':'Kernel namespace only relative to measured fullLUTv1; decoder/table/descriptor/math/K64producer-consumerprotocol/epilogue byteexact inverse. Host factory only selects64 and guards exact M3012EP8dimensions. Whole-serving method AST identical.','block_selection':{'candidates':candidates,'selected_block_m':block_m,'heuristic_unmapped_Mtopk_per_32local_experts':753.0,'uniform_mapped_local_average_before_busiest_rotation':94.125,'actual_mapped_average_and_histogram':'UNKNOWN until original GPUfixture; driver saves all256expertcounts,32localcounts,localrouteutilization andblocks separately for every seed. Heuristic753 is not actualmappedlocalaverage.'},'alignment':{'source_capacity':cap,'grid_rows':rows,'up_grid':[128,rows,1],'down_grid':[64,rows,1],'tail':'capacity not64aligned; gridceil plus existingposition/tokencapacityguards retained','actual_post':'UNKNOWN until routingfixture; driver derives from actualhistogram without host-sync changing timedmethod'},'T64_resources':{'offline_CUDA':'12.8 sm90a','registers_up_down':regs,'spill_stack_bytes':0,'dynamic_shared':42016,'physical_gate':proof,'negative_rejections':neg,'capacity_model':'rounded64registers with256threads permits at most4residentCTAs on65536regs; actualCUDA13occupancy+sharedcarveout queried beforemath. No actualresource/speedclaim.'},'memory_model':mem,'timing_contract':{'case':'originalM3012EP8normal/busiest,weights5252','arms':['exact05ec9a8anchor','guardedT64LUTwhole-serving'],'untimed_control':'oneoriginalcanonicalSS T64bitwise check, samealreadycompiledSSSO/banks; nooldSStiming arm','F1':'oneimmutableactualseed7904fixture,3anchor/3LUT/3anchor calls, everycurrentoutput andownblockterminal checked','paired':'8matchedseedpairs7900..7907, alternatingAB/BA4each, everycurrentoutput andownarmterminal checked','total_timed_calls':25,'new_compiles':1,'new_functions':2,'graph_and_oracles':'originalactualmethod/aligner/SwiGLU/finalizer+originaltaskcheck; freshX/routing replay, valid-to-emptyexactzero, inputs/weights/banks immutable; sourcepins andcleanup','bounds_inner_outer_seconds':[180,240],'forecast':'UNKNOWN; generalization test only, nofull14/3x/officialclaim'},'loop_contract':'profile->measure->hypothesis->optimize/reimplement->test->validate->stack->submit->learn->reprioritize->repeat; idle<=10s; proj_rsi/ITERATION.md'}
(D/'feasibility.json').write_text(json.dumps(out,indent=2)+'\n')
# Exact factory test is mocked; importing torch must not initialize CUDA.
import torch
import torch.utils.cpp_extension as cpp
builder=cpp.load_inline;calls=[]
def mock(*a,**kw):calls.append(kw);return object()
try:
 cpp.load_inline=mock;s=importlib.util.spec_from_file_location('t64_mock_factory',C/'adapter.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.module()
finally:cpp.load_inline=builder
assert len(calls)==1 and calls[0]['cuda_sources']==(C/'extension_source.cu').read_text()and calls[0]['functions']==['run']
assert not torch.cuda._initialized
readiness={'state':'CPU_RUNNABLE_GPU_UNRUN','driver_sha256':sha(D/'pilot.py'),'kernel_sha256':sha(C/'kernel.cuh'),'factory_source_exact':True,'kernel_inverse_delta_namespace_only':True,'whole_serving_method_AST_identical':True,'driver_import_without_torch':True,'physical_gate_and_negatives_PASS':True,'CUDA_context_initialized':False,'feasibility_sha256':sha(D/'feasibility.json')}
(D/'cpu-readiness.json').write_text(json.dumps(readiness,indent=2)+'\n');print(json.dumps(readiness,indent=2))
