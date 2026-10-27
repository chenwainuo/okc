"""Read-only CPU audit of collected actual EH same-geometry artifacts."""
import ast,collections,hashlib,importlib.util,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COLLECTION=ROOT/'results/proj-rsi/scheduler/artifacts/research.eh.samegeometry.gpu.01/collection-1793076695219438309'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(collection=COLLECTION):
    inventory=json.loads((collection/'inventory.json').read_text())
    for rel,h in inventory['files'].items():assert sha(collection/rel)==h,rel
    for rel,h in inventory['dispatch']['source_pins'].items():assert sha(ROOT/rel)==h,rel
    assert len(inventory['files'])==4 and inventory['missing']==[] and inventory['terminal_exit']==0
    cleanup=inventory['cleanup_observation'];assert cleanup['terminal_exit']==0 and not cleanup['cleanup_failed'] and not cleanup['container_exists'] and cleanup['compute_pids']==cleanup['guard_pids']==cleanup['containers']==[] and cleanup['gpu_memory_mib']==0
    folder=collection/'logs/scheduler/eh-same-geometry-01';result=folder/'result.json';d=json.loads(result.read_text())
    path=ROOT/'proj_rsi/candidates/eh_same_geometry_trial/support.py';spec=importlib.util.spec_from_file_location('eh_result_support',path);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);S.verify_frozen();S.verify_lab()
    assert d['status']=='VALID_EH_SAME_GEOMETRY_MECHANISM_ONLY' and d['scope']=='EH_FIXED_GEOMETRY_MECHANISM_ONLY_NOT_SELECTED_SERVING_POLICY'
    assert d['source_pins']==S.PINS and d['source_roles']==S.SOURCES and d['all_candidate_JIT_and_resources_before_first_math']=='PASS' and not d['full_stack_or_actual_selected_incumbent_win_claim']
    labels=S.LABELS;assert set(d['attrs'])==set(labels) and set(d['JIT'])=={'parent','eh'}
    sass=[]
    for role,rec in d['JIT'].items():
        file=folder/Path(rec['artifact']).name;assert sha(file)==rec['sha256'] and len(rec['module_sha256'])==64
        assert S.binary_gate(role,file.read_text())==rec['gate'];sass.append(file.read_text())
        assert S.cuda_source(role)==(ROOT/f'proj_rsi/candidates/eh_same_geometry_trial/{role}.extension.cu').read_text()
    env={'re':re}
    for n in ast.parse((ROOT/'proj_rsi/candidates/rs32_eh_backport/preflight.py').read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('tables','physical','bitmap'):exec(compile(ast.Module(body=[n],type_ignores=[]),'<pure-EH-review>','exec'),env)
    physical=env['physical']('\n'.join(sass));bitmap=env['bitmap']()
    for label,attrs in d['attrs'].items():assert attrs==dict(regs=168,local=0,shared=144520,occupancy=1,max_threads=384,threads=384,static_rounded_register_pool=64512,dynamic_register_pool=64512)
    assert list(d['shapes'])==list(S.SHAPES)
    comparisons={};nraw=nterminal=nmutation=0
    orders=[]
    for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend(seq[j:]+seq[:j] for j in range(4))
    for layer,(n,k,ks) in S.SHAPES.items():
        r=d['shapes'][layer];assert r['status']=='VALID_FIXED_GEOMETRY_RESULT_NO_PRODUCTION_ADMISSION' and r['graph_cleanup']=='DRAINED_RESET_ALL_FOUR'
        assert r['geometry']==dict(m=128,n=n,k=k,KS=ks,NT=128,ST=5,GS=1,PRB=0,WM=0,INS=0)
        assert set(r['validation'])==set(labels)
        for label,checks in r['validation'].items():
            assert len(checks)==3 and checks[0]['all_K_basis']=='EXACT' and checks[0]['random_FP64']=='PASS' and checks[0]['tolerance']==S.value(S.source_for('eh'),'RQTOL')[ks]
            assert [x['task_dist'] for x in checks[1:]]==['normal','outliers'] and all(x['oracle'] for x in checks[1:])
        t=r['timing'];calls=t['all_raw_calls'];assert len(calls)==274 and all(math.isfinite(x['us']) and x['us']>0 for x in calls)
        assert [(x['label'],x['input_seed'],x['mode'],x['warmup']) for x in calls[:20]]==[(label,239999,'warmup',True) for label in labels for _ in range(5)]
        blocks={};last={};at=20;blockno=0
        for parent,new in [('parent2','eh2'),('parent3','eh3')]:
            for key,label in [(parent+'_before',parent),(new,new),(parent+'_after',parent)]:
                rows=calls[at:at+21];at+=21
                assert [(x['label'],x['input_seed'],x['mode'],x['warmup']) for x in rows]==[(label,240000+21*blockno+i,'F1',False) for i in range(21)]
                blocks[key]=[x['us'] for x in rows];last[label]=rows[-1]['input_seed'];blockno+=1
        assert at==146 and blocks==t['F1_samples_us']
        med={key:statistics.median(values) for key,values in blocks.items()};assert med==t['F1_medians_us']
        pairs=[]
        for i in range(32):
            rows=calls[146+4*i:146+4*i+4]
            assert [(x['label'],x['input_seed'],x['mode'],x['warmup']) for x in rows]==[(labels[j],250000+i,'paired',False) for j in orders[i%8]]
            pairs.append({x['label']:x['us'] for x in rows})
        assert pairs==t['paired_samples_us'] and t['terminal_input_seed_per_policy']==dict.fromkeys(labels,250031)
        for phase,expected in [('F1',last),('paired',dict.fromkeys(labels,250031))]:
            checks=t['strict_terminal_checks'][phase];assert len(checks)==4 and {x['label'] for x in checks}==set(labels)
            assert all(x['input_seed']==expected[x['label']] and x['actual_terminal_output_before_any_replay']=='PASS' and x['oracle'] for x in checks);nterminal+=len(checks)
        compare={}
        for parent,new in [('parent2','eh2'),('parent3','eh3')]:
            c=t['comparisons'][new];g=[p[parent]-p[new] for p in pairs]
            assert c['control']==parent and c['FL_control']==S.FLAGS[parent] and c['FL_EH']==S.FLAGS[new]
            assert c['F1_gain_us']==min(med[parent+'_before'],med[parent+'_after'])-med[new]
            assert c['paired_median_gain_us']==statistics.median(g) and c['paired_all_gains_us']==g and c['paired_wins']==sum(v>0 for v in g)
            assert c['paired_median_ratio']==statistics.median(p[parent]/p[new] for p in pairs) and c['gain_threshold_us']==0 and not c['serving_incumbent_win_claim']
            compare[new]={**c,'paired_win_tie_loss':[sum(v>0 for v in g),sum(v==0 for v in g),sum(v<0 for v in g)],'paired_delta_MAD_us':statistics.median(abs(v-statistics.median(g)) for v in g),'paired_delta_range_us':[min(g),max(g)],'paired_policy_medians_us':{label:statistics.median(p[label] for p in pairs) for label in [parent,new]}}
        mutations=r['mutations'];assert len(mutations)==20
        assert {(x['label'],x['input']) for x in mutations}=={(label,kind) for label in labels for kind in ['original','zero','fresh_normal','fresh_outliers','probes']}
        assert all(x['NaN_poison_overwritten'] and x['oracle'] for x in mutations)
        assert set(r['graph_nodes'])==set(labels)
        for label,nodes in r['graph_nodes'].items():assert len(nodes)==1 and 'rq<128, 5, 1, 0, 0, 0, '+str(S.FLAGS[label])+'>' in nodes[0]
        comparisons[layer]=compare;nraw+=len(calls);nmutation+=len(mutations)
    assert (nraw,nterminal,nmutation)==(1096,32,80)
    return dict(status='INDEPENDENT_FIXED_EH_RUNTIME_VALIDITY_PASS',job_id='research.eh.samegeometry.gpu.01',inventory_files=len(inventory['files']),source_pins=len(inventory['dispatch']['source_pins']),inventory_sha256=sha(collection/'inventory.json'),result_sha256=sha(result),raw_calls=nraw,terminal_oracles=nterminal,mutations=nmutation,all_K_FP64_arm_shapes=16,JIT_roles_regated=2,actual_combined_physical=physical,output_bitmap=bitmap,comparison=comparisons,cleanup={key:value for key,value in cleanup.items() if key!='host_log_tail'},scope='Matched forced variants only; no actual selected policy, cumulative-policy, official-score or H200 gain claim.')
if __name__=='__main__':
    r=analyze();(ROOT/'proj_rsi/audits/eh-same-geometry-01-independent.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','inventory_files','source_pins','raw_calls','terminal_oracles','mutations']},indent=2))
