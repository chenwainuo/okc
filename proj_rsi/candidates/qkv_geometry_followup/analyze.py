"""CPU-only source-bound QKV geometry feasibility; no performance forecast or GPU import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/research.qkv.split4.harness.gpu.01/collection-1793077707233847694'
RUN=COLLECTION/'logs/scheduler/qkv-runtime-01/run'
SOURCE=ROOT/'proj_rsi/candidates/group8_qkv_trial/extension_source.cu'
RUNTIME=ROOT/'proj_rsi/candidates/group8_qkv_trial/runtime.py'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def partition(sk):return [s*128//sk for s in range(sk+1)]

def schedule(kcs):
    # Necessary conditions from exact unchanged startup/pair/source loop.
    assert kcs>=6,'six unconditional W startup reads exceed local K slice'
    assert kcs>=4,'four unconditional A startup prefetches exceed slice'
    assert kcs%4==0,'unconditional pair(pidx)+pair(pidx+1) needs four-stage multiples'
    stages=[];a_banks={0:[0,1],1:[2,3]};retired=[]
    for pidx in range(kcs//2):
        bank=pidx%2
        assert a_banks[bank]==[2*pidx,2*pidx+1]
        stages.extend(a_banks[bank])
        if pidx>0:
            prev=(pidx-1)%2;retired.extend([2*pidx-2,2*pidx-1])
            if pidx+1<kcs//2:a_banks[prev]=[2*pidx+2,2*pidx+3]
    retired.extend([kcs-2,kcs-1])
    assert stages==retired==list(range(kcs))
    deps={}
    def edge(n,d):deps.setdefault(n,set()).add(d);deps.setdefault(d,set())
    for s in range(kcs):
        deps.setdefault(f'P{s}',set())
        if s>=2:edge(f'P{s}',f'P{s-2}')
        if s>=6:edge(f'P{s}',f'P{s-6}')
        if s>=8:edge(f'P{s}',f'R{s-8}')
        edge(f'R{s}',f'G{min(s//2+1,kcs//2-1)}')
    for g in range(kcs//2):
        edge(f'G{g}',f'P{2*g}');edge(f'G{g}',f'P{2*g+1}')
        if g:edge(f'G{g}',f'G{g-1}')
    done=set()
    while len(done)<len(deps):
        ready={n for n,d in deps.items() if n not in done and d<=done};assert ready,'publication/retirement dependency cycle';done|=ready
    return {'stages_consumed_once':len(stages),'retired_after_use':len(retired),'DAG_nodes':len(done),
            'limits':'Necessary arithmetic and finite source dependency model, not compiled physical or runtime validity.'}

def csr_coordinate_proof(sk):
    boundaries=partition(sk);count=0
    for tile in range(144):
        for ch in range(128):
            split=next(i for i in range(sk) if boundaries[i]<=ch<boundaries[i+1])
            seq=ch-boundaries[split];group=(split*18+tile//8)*8+tile%8
            assert 0<=group<sk*18*8 and 0<=seq<=32766
            recovered_split=group//144;recovered_tile=((group%144)//8)*8+group%8
            assert recovered_split==split and recovered_tile==tile and boundaries[recovered_split]+seq==ch
            for lane,reg,half,xor in [(0,0,0,1),(31,15,1,127)]:
                entry=(seq<<17)|(lane<<12)|(reg<<8)|(half<<7)|xor
                assert entry!=0xffffffff and entry>>17==seq and (entry>>12)&31==lane and (entry>>8)&15==reg and (entry>>7)&1==half and entry&127==xor
            count+=1
    return count


def analyze():
    inv=json.loads((COLLECTION/'inventory.json').read_text());d=json.loads((RUN/'result.json').read_text());source=SOURCE.read_text();runtime=RUNTIME.read_text()
    assert inv['terminal_exit']==0 and d['status']=='VALID_LOCAL_COMPARISON_REVIEW_REQUIRED_NO_OFFICIAL_GAIN'
    for name,pin in inv['files'].items():assert sha(COLLECTION/name)==pin,name
    for name,pin in inv['dispatch']['source_pins'].items():assert sha(ROOT/name)==pin,name
    assert sha(SOURCE)==d['JIT']['new']['source_sha256']=='29f7f11bdf3246e66b9419fce4003cb27c22a4cf263a3d1f15c0f3770371f1a2'
    anchors=['rx_pack<<<512,256,0,stream>>>','rq_qkv_g8_sk4<128,8,1,0,0,0,20><<<dim3(36,4),512,',
        'qkv_reduce<<<1152,256,0,stream>>>','constexpr int KC=32,WST=6,BST=8,WB=6144,BB=8192;',
        'int nb=blockIdx.x/2,half=blockIdx.x&1,cb=blockIdx.x*64;', 'const int ks=blockIdx.y,k0=32*ks;',
        'for(int s=0;s<WST;++s)', 'for(int pidx=0;pidx<KC/2;pidx+=2)', 'for(int s=1;s<4;++s)d+=partial[s*128u*2304u+i];',
        'qcheck(partial,x,torch::kFloat32,4*128*2304,"partial");', 'wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16']
    for anchor in anchors:assert anchor in source,anchor
    assert 'self.partial=torch.empty((4,128,2304)' in runtime and "self.esc,_=c['_rqesc'](esc6,18,128,4,w.device)" in runtime
    parent=(RUN/'rs32.selected.cu').read_text()
    assert 'k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(384)' in parent
    assert d['actual_prepared_picks']['rs32']['selected_rq']['KS']==7 and 'rq<128, 5, 1, 0, 0, 0, 3>' in d['actual_profile_kernels']['rs32'][0]
    assert d['JIT']['new']['occupancy_after_optin']==1
    assert all(isinstance(name,str) for names in d['actual_profile_kernels'].values() for name in names)
    current_proof_path=ROOT/'proj_rsi/candidates/group8_qkv/proofs.py'
    spec=importlib.util.spec_from_file_location('qkv_geometry_parent_proof',current_proof_path)
    proofmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(proofmod)
    current_proof=proofmod.proof()
    assert current_proof['XP_u32_words']==524288 and current_proof['output_values']==294912
    assert d['actual_W_codec']['CSR_CAP']==32
    csr_cases={str(sk):csr_coordinate_proof(sk) for sk in (1,2,4,8,16,3,5,6,7)}
    geometries=[]
    for sk in [v for v in range(1,129) if 128%v==0]:
        kcs=128//sk;b=partition(sk);assert b==list(range(0,129,kcs))
        spans=[list(range(b[s],b[s+1])) for s in range(sk)];assert sum(spans,[])==list(range(128))
        try:proof=schedule(kcs);eligible=True;failure=None
        except AssertionError as e:proof=None;eligible=False;failure=str(e)
        groups=sk*18*8;cap=((kcs*1024+32)//32)*32
        geometries.append({'SK':sk,'KCS':kcs,'boundaries':b,'main_grid_source_if_coherently_rewritten':[36,sk,1],
            'main_CTA_count_arithmetic':36*sk,'necessary_unchanged_schedule_eligible':eligible,'failure':failure,'schedule':proof,
            'CSR_groups':groups,'CSR_universal_CAP_including_sentinel':cap,'CSR_universal_allocation_bytes':4*(groups*cap+32),
            'partial_planes':sk,'partial_bytes':sk*128*2304*4,'source_logical_X_main_bytes':36*128*16384,
            'performance_or_runtime_admission':False})
    ragged=[]
    for sk in (3,5,6,7):
        b=partition(sk);lengths=[b[i+1]-b[i] for i in range(sk)]
        assert sum(lengths)==128
        ragged.append({'SK':sk,'boundaries':b,'lengths':lengths,'all_lengths_multiple4_and_at_least8':all(n>=8 and n%4==0 for n in lengths),
            'unchanged_source_admissible':False,'required_changes':['floor split boundaries for main k0/KCS and host CSR grouping','stage-count-aware initial W/A prefetch bounds','tail/parity-safe MMA/A-retirement and Bempty drain','all SK partial planes plus ordered FP32 reduction and workspace extents','new exact coverage, numerical and actual JIT/resource/protocol gates']})
    negatives={}
    for kcs in (1,2,4,18,19):
        try:schedule(kcs)
        except AssertionError:negatives[str(kcs)]='REJECTED'
        else:raise AssertionError(('invalid KCS admitted',kcs))
    original=json.loads((ROOT/'proj_rsi/audits/qkv-runtime-01/analysis.json').read_text())
    return {'state':'ANALYTIC_HOLD_NO_PERFORMANCE_VARIANT','decision':'Do not manufacture a geometry-only speed candidate from unknown wave/node costs. Preserve exact failure and require a new bounded diagnostic decision.',
        'evidence':{'result':{'path':str((RUN/'result.json').relative_to(ROOT)),'sha256':sha(RUN/'result.json')},'source':{'path':str(SOURCE.relative_to(ROOT)),'sha256':sha(SOURCE)},'runtime':{'path':str(RUNTIME.relative_to(ROOT)),'sha256':sha(RUNTIME)},'prior_analysis':{'path':'proj_rsi/audits/qkv-runtime-01/analysis.json','sha256':sha(ROOT/'proj_rsi/audits/qkv-runtime-01/analysis.json')},'inventory_files_verified':len(inv['files']),'dispatch_source_pins_verified':len(inv['dispatch']['source_pins'])},
        'applied_lessons':['wring-mechanism-policy-separation@1','prepare-selection-is-experiment@2','composition-not-additivity@1','rq11-full-residency-ledger@2','proj_rsi/audits/qkv-runtime-01/lessons.json'],
        'recorded_timing':{'candidate_paired_median_us':original['paired_policy_medians_us']['new'],'rs32_paired_median_us':original['paired_policy_medians_us']['rs32'],'pw36_paired_median_us':original['paired_policy_medians_us']['pw36']},
        'source_bound_launches':{'pack':{'grid':[512,1,1],'threads':256,'profile_duration_us':None,'occupancy':None},'main':{'grid':[36,4,1],'threads':512,'CTA_count':144,'output_tile_rows':128,'output_tile_columns':64,'K64_stages_per_split':32,'SK':4,'actual_occupancy_after_optin':1,'profile_duration_us':None},'reduce':{'grid':[1152,1,1],'threads':256,'partial_planes':4,'profile_duration_us':None,'occupancy':None},'selected_rs32':{'grid_expression':'(p.NB*((FL&16)?2:1),KS)','source_bound_p_NB':18,'FL':3,'KS':7,'source_bound_grid':[18,7,1],'CTA_count':126,'threads':384,'actual_occupancy':None,'profile_duration_us':None},'scope':'Launch expressions bound to executed source/shape/profile, not captured per-node launch telemetry.'},
        'SM_count_recorded':None,'actual_wave_counts':None,'node_duration_recorded':False,'wave_formula_only':'ceil(CTA_count/(runtime_SM_count*per_kernel_actual_occupancy)); no value admitted when required observations absent.',
        'wave_hypothesis':{'status':'UNRESOLVED','statement':'Launch-wave quantization, tile/split geometry may contribute; missing SMcount and node durations prevent testing this causal explanation from this archive.','speed_forecast_us':None,'whole_score_forecast':None},
        'current_layout_XP_output_physical_proof':current_proof,'CSR_coordinate_cases_by_SK':csr_cases,'equal_split_divisors':geometries,'ragged_splits':ragged,'negative_schedule_controls':negatives,
        'actual_recorded_CSR_CAP':32,'SK8_CSR_bound':{'logical_CAP':32,'status':'SOURCE_DERIVED_SUBSET_BOUND_NOT_NEW_GPU_MEASUREMENT','proof':'Each16chunk SK8 list is a subset of an observed32chunk SK4 list whose length<32, so every new list length<32 and minimum strict CAP remains32.','groups':1152,'allocation_bytes':147584,'current_allocation_bytes':73856},'SK8_assessment':{'necessary_schedule_coverage':'PASS_KCS16','main_CTA_count':288,'partial_bytes':9437184,'current_partial_bytes':4718592,'source_logical_X_bytes_change':0,'required_coupled_edits':['main KC/k0/symbol and grid','CSR split grouping/G/CAP/allocation guards','partial shape/qcheck/ownership extents','ordered FP32 reduction visits8planes instead4','terminal reconstruction oracle visits8planes','new numerical/physical/JIT/resource proofs'],'ROI':'UNKNOWN; more CTAs plus extra reduction/partial cost, unknown node decomposition. No variant admitted.'},
        'TT128_assessment':{'status':'NOT_A_PARAMETER_FLIP','TT_symbol_in_frozen_source':False,'NT_already':128,'if_means_N_tile128':['new m64n128 WGMMA/D64 mapping','decoded W Btile doubles and producer ownership changes','new N tile/output/partial coverage','new shared regions/descriptor bounds and resource admission','new physical A/D/fence/JIT proof'],'resource_result':'UNKNOWN_UNCOMPILED','ROI':'UNKNOWN'},
        'falsifier_and_next_decision':'Obtain a separately authorized source-frozen diagnostic recording exact device SMcount, actual grid/block/shared for all nodes, post-optin occupancy, and pack/main/reduce GPU durations in a real current-X full pipeline with actual selected PW36/RS32 controls. A dominant main and repeatable wave-tail exposure supports a narrow geometry source version; dominant auxiliaries or unchanged main cost across matched geometry refutes a wave-only recovery. Retain all outcomes.',
        'offline_compile':'NOT_RUN_NO_CHANGED_KERNEL_OR_ADMITTED_VARIANT','GPU_calls':0,'prepare_calls':0,'API_calls':0,'queue_changes':0,'family_status':'NOT_RETIRED'}

if __name__=='__main__':
    result=analyze();(HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('state','recorded_timing','SM_count_recorded','actual_wave_counts','node_duration_recorded','negative_schedule_controls','offline_compile','GPU_calls')},indent=2))
