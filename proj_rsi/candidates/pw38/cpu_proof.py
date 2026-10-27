"""CPU-only PW38 composition; no production/torch import, compilation or GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
EH=HERE.parent/'rs32_eh_backport'
BASE=HERE.parent/'pw36/submission.py'
EH_MANIFEST='a4ccf23b7ff6b9a383ad0d6299d7de2d0d0d18b60d85e287150a0e465aac9869'
BASE_SHA='2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def value(text,name):
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise KeyError(name)
def named(text):
    result={}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):result[target.id]=ast.dump(node,include_attributes=False)
    return result
def apply(text,delta,reverse=False):
    for entry in reversed(delta['edits']) if reverse else delta['edits']:
        old,new=(entry['after'],entry['before']) if reverse else (entry['before'],entry['after'])
        assert text.count(old)==1,entry['id'];text=text.replace(old,new,1)
    return text
def streams(sass):
    result={}
    for table in re.split(r'\n\s*Function : ',sass)[1:]:
        flag=int(re.search(r'Li(\d+)EEv3RqP',table.splitlines()[0])[1])
        result[flag]=[(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]
    return result

def check():
    assert sha(BASE)==BASE_SHA and sha(EH/'manifest.json')==EH_MANIFEST
    manifest=json.loads((EH/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(EH/name)==expected,name
    assert sha(EH/'submission.py')=='9067795eb0d01abe576744a874e3504d6371a92ebc05543708c500b8d83a4819'
    candidate=(HERE/'submission.py').read_text();base=BASE.read_text();isolated=(EH/'submission.py').read_text()
    delta=json.loads((HERE/'delta.json').read_text())
    assert delta==json.loads((EH/'delta.json').read_text()) and len(delta['edits'])==6
    assert apply(base,delta)==candidate and apply(candidate,delta,True).encode()==BASE.read_bytes()
    assert value(candidate,'RQCU')==value(isolated,'RQCU')
    old,new=named(base),named(candidate)
    changed=sorted(key for key in old if old[key]!=new.get(key))
    assert changed==['RQBAKE','RQCU','RQPROD','RQV'] and set(old)==set(new),changed
    unchanged=['prepare','kernel_fn','_r6','_v12','_rqbake','_rqsbake','RQSBAKE','RQPROD2','RQFLAGS','_rqsrc','_rqwg','_rqss',
               'RWCU','_rwsrc','_rwext','RSPALL','RSP','_rsbake']
    assert all(old[key]==new[key] for key in unchanged)
    pw35=named((HERE.parent/'pw35/submission.py').read_text())
    assert all(new[key]==pw35[key] for key in ('RWCU','_rwsrc','_rwext','RSPALL','RSP','_rsbake'))
    assert 'LDS_CURSOR' not in candidate and len(candidate.encode())<1_000_000
    assert value(candidate,'PREPARE_BUDGET_S')==240 and value(candidate,'MARGIN')==.99
    # Old production candidates and exact geometry tuples remain; EH only appends.
    original_variants=value(base,'RQV');variants=value(candidate,'RQV')
    assert variants[:len(original_variants)]==original_variants
    assert variants[len(original_variants):]==[('s4s5zEH',128,5,1,0,0,0,65538),('s4s5hzEH',128,5,1,0,0,0,65539)]
    assert value(candidate,'RQBAKE')[256]==value(base,'RQBAKE')[256]
    assert value(candidate,'RQBAKE')[128]==value(base,'RQBAKE')[128]+['s4s5zEH','s4s5hzEH']
    # Duplicate strict anchors fail closed rather than partially composing.
    for entry in delta['edits']:
        try:apply(base.replace(entry['before'],entry['before']+'\n'+entry['before'],1),delta)
        except AssertionError:pass
        else:raise AssertionError('duplicate anchor accepted: '+entry['id'])
    # Execute only pure device-text/bitmap/SASS routines from pinned CPU source.
    env={'ast':ast,'re':re,'value':value}
    for node in ast.parse((EH/'preflight.py').read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('device','bitmap','tables','physical'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-pure-EH-proof>','exec'),env)
    device=env['device'](candidate)
    assert device==(EH/'device_source.cu').read_text()
    assert value(candidate,'RQFLAGS')==['-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr']
    bitmap=env['bitmap']();physical=env['physical']((EH/'sass.txt').read_text())
    controls,combined=streams((EH/'control-sass.txt').read_text()),streams((EH/'sass.txt').read_text())
    assert all(controls[fl]==combined[fl] for fl in (2,3))
    from proj_rsi.static_check import check_source
    lint=check_source(candidate,str(HERE/'submission.py'));assert lint['ok'],lint
    return {'state':'CPU_PW38_CUMULATIVE_OPTIONAL_EH_PASS_GPU_UNRUN','candidate_sha256':digest(candidate),
            'base_sha256':BASE_SHA,'EH_manifest_sha256':EH_MANIFEST,'EH_source_sha256':sha(EH/'submission.py'),
            'exact_text_replacements':6,'reverse_restoration':'ENTIRE_PW36_BYTES_EXACT',
            'changed_named_AST_nodes':changed,'unchanged_named_AST_nodes':unchanged,'PW35_PW36_router':'ALL_SIX_NAMED_NODES_EXACT',
            'cursor':'EXCLUDED_NO_LDS_CURSOR','EH_flags':[65538,65539],'EH_flag_bit':65536,'HALF_bit':16,'EARLY_bit':32,'EP_ported':False,
            'existing_controls':'ALL_VARIANTS_RETAINED_NON_EH_FL2_FL3_OFFLINE_SASS_EXACT',
            'control_instruction_counts':{str(fl):len(controls[fl]) for fl in (2,3)},
            'bitmap':bitmap,'offline_physical':physical,'offline_reuse':{'translation_unit_sha256':sha(EH/'device_source.cu'),
                'cubin_sha256':sha(EH/'device.sm90a.cubin'),'SASS_sha256':sha(EH/'sass.txt'),
                'covered_instances':['NT128/ST5/GS1/PRB0/WM0/INS0/FL'+str(fl) for fl in (2,3,65538,65539)],
                'limits':'CPU offline exact-source reuse only; actual JIT/all production resources/GPU correctness/performance remain unrun.'},
            'selection':{'mechanism':'existing validated RQ bake with MARGIN0.99, unchanged R6 F1 incumbent gate',
                'EH_is_optional':True,'EH_ACTUALLY_SELECTED':'UNRUN','small_historical_gain_can_be_suppressed':True,
                'next_diagnostic':'same-geometry controls FL2vs65538 and FL3vs65539 on current cumulative source before changing selection thresholds'},
            'production_imports_executed':0,'cuda_API_calls':0,'prepare_calls':0,'extension_builds':0,'new_compilations':0,'lint':lint,
            'performance':'UNMEASURED_NOT_SHIPPED_NOT_PROMOTED'}
if __name__=='__main__':
    result=check();(HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('state','candidate_sha256','changed_named_AST_nodes','cuda_API_calls')}))
