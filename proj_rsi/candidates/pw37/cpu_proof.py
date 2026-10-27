"""CPU-only PW37 cumulative composition gate. Never imports production or torch."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent/'pw36/submission.py'
CURSOR=HERE.parent/'smallm_increment'
PINS={'base':'2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf',
      'rs32':'a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad',
      'cursor':'0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737',
      'delta':'9513bc03fc5288d670b8b3c2a519ef63e3d03b90391e5d8ebaf7419276ad4d17'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def literal(source,name):
    matches=[node.value for node in ast.parse(source).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)]
    assert len(matches)==1,name
    return ast.literal_eval(matches[0])
def named(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):result[target.id]=ast.dump(node,include_attributes=False)
    return result

def check():
    assert sha(BASE)==PINS['base']
    assert sha(HERE.parent/'rs32/submission.py')==PINS['rs32']
    assert sha(CURSOR/'submission.py')==PINS['cursor']
    assert sha(CURSOR/'delta.json')==PINS['delta']
    base=BASE.read_text();candidate=(HERE/'submission.py').read_text();cursor=(CURSOR/'submission.py').read_text()
    changes=json.loads((HERE/'delta.json').read_text())
    assert changes==json.loads((CURSOR/'delta.json').read_text()) and len(changes)==4
    expected=base
    for old,new in changes:
        assert expected.count(old)==1 and old!=new
        expected=expected.replace(old,new)
    assert expected==candidate,'production cumulative source differs from exact four replacements'
    restored=candidate
    for old,new in reversed(changes):
        assert restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored.encode()==BASE.read_bytes(),'reverse delta must restore entire PW36 byte-for-byte'
    assert literal(candidate,'RQCU')==literal(cursor,'RQCU'),'RQCU cursor differs'
    assert literal(base,'RQCU')==literal((HERE.parent/'rs32/submission.py').read_text(),'RQCU')
    before,after=named(base),named(candidate)
    changed=[key for key in before if before[key]!=after.get(key)]
    assert changed==['RQCU'] and set(before)==set(after),changed
    preserved=['RWCU','_rwsrc','_rwext','RSPALL','RSP','_rsbake','prepare','kernel_fn','_r6','_v12','_rqbake','_rqsbake','RQPROD','RQPROD2','RQSBAKE','RQBAKE','RQFLAGS','_rqsrc','_rqwg','_rqss']
    assert all(before[key]==after[key] for key in preserved)
    router=['RWCU','_rwsrc','_rwext','RSPALL','RSP','_rsbake']
    pw35=named((HERE.parent/'pw35/submission.py').read_text())
    assert all(after[key]==pw35[key] for key in router),'PW35 RTK branch changed'
    assert len(candidate.encode())<1_000_000
    tree=ast.parse(candidate)
    forbidden={'subprocess','socket','requests','urllib','http','reference'}
    imports=[alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    imports += [node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module]
    assert not forbidden.intersection(imports)
    # Full sequential-slot model including ragged split-K bounds and all possible rings.
    steps=0
    for st in range(1,65):
        for kcs in range(1,129):
            slot=phase=0
            for stage in range(kcs):
                assert (slot,phase)==(stage%st,(stage//st)&1)
                steps+=1;slot+=1
                if slot==st:slot=0;phase^=1
    # Reconstruct exactly the existing offline compile-only translation unit.
    env={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rqwg','_rqss'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure-source-generator>','exec'),env)
    cu=literal(candidate,'RQCU').split('typedef CUresult')[0].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
    cu=cu.replace('@@WGMMA@@',''.join(env['_rqwg'](n) for n in (16,32,128,256))+''.join(env['_rqss'](n) for n in (128,256)))
    for nt,st in ((32,5),(32,6),(16,5)):
        cu+=f'\ntemplate __global__ void rq<{nt},{st},1,0,0,0,35>(const RqP,__grid_constant__ const RqTmap,__grid_constant__ const RqTmap);\n'
    assert cu==(CURSOR/'after.cu').read_text()
    manifest=json.loads((CURSOR/'manifest.json').read_text())
    assert literal(candidate,'RQFLAGS')==manifest['flags']
    for name in ('after.cu','after.cubin','after.sass','delta.json','submission.py'):
        assert sha(CURSOR/name)==manifest['files'][name]
    from proj_rsi.static_check import check_source
    lint=check_source(candidate,str(HERE/'submission.py'))
    assert lint['ok'],lint
    # Negative: an unintended RTK change must break whole-file restoration.
    changed_router=candidate.replace('RWCU = r"""','RWCU = r"""\n// unauthorized router edit',1)
    assert changed_router!=candidate
    trial=changed_router
    for old,new in reversed(changes):trial=trial.replace(new,old)
    assert trial!=base
    return {'state':'CPU_CUMULATIVE_COMPOSITION_PASS_GPU_UNRUN','candidate_sha256':digest(candidate),'base_sha256':PINS['base'],
            'cursor_source_sha256':PINS['cursor'],'RQCU_sha256':digest(literal(candidate,'RQCU')),
            'delta_count':4,'reverse_restoration':'ENTIRE_PW36_BYTES_EXACT','changed_named_AST_nodes':changed,
            'preserved_named_AST_nodes':preserved,'PW35_PW36_router':'ALL_SIX_NAMED_NODES_EXACT',
            'cuda_API_calls':0,'extension_builds':0,'production_imports_executed':0,'prepare_calls':0,
            'cursor_model':{'ST':[1,64],'KCS':[1,128],'stage_steps':steps,'status':'PASS'},
            'negative_unintended_router_edit':'CAUGHT_BY_REVERSE_RESTORATION',
            'offline_compilation_reuse':{'state':'EXACT_EXISTING_TRANSLATION_UNIT_AND_FLAGS_NO_NEW_COMPILATION',
                'manifest':str(CURSOR/'manifest.json'),'manifest_sha256':sha(CURSOR/'manifest.json'),
                'after_cu_sha256':sha(CURSOR/'after.cu'),'after_cubin_sha256':sha(CURSOR/'after.cubin'),'after_sass_sha256':sha(CURSOR/'after.sass'),
                'covered_instances':[[32,5,1,0,0,0,35],[32,6,1,0,0,0,35],[16,5,1,0,0,0,35]],
                'limits':'Offline cubin only for these three instances; no actual JIT/full-production resource or GPU correctness/performance claim.'},
            'lint':lint,'performance':'UNMEASURED_PW37_IS_NOT_SHIPPED_OR_PROMOTED'}

if __name__=='__main__':
    result=check();(HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'state':result['state'],'candidate_sha256':result['candidate_sha256'],'cuda_API_calls':0}))
