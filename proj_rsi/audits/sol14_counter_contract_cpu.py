"""CPU-only counter policy negative controls; no candidate import or torch."""
import ast, hashlib, json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]

def check(mode, baseline, current, splits=0, launches=0):
    if mode=='unchanged': return current==baseline
    if mode=='reset0': return all(v==0 for v in current)
    if mode=='p1':
        if len(current)!=len(baseline) or len(current)%2: return False
        expected=[v+launches*(splits if i%2==0 else 1) for i,v in enumerate(baseline)]
        return current==expected
    return False  # HOLD for any unclassified active protocol

def overlap(a,b):
    # Allocation identity plus half-open byte offsets, independent of first-element data_ptr.
    return a[0]==b[0] and max(a[1],b[1])<min(a[2],b[2])

def run():
    results={}
    def expect(name, value): assert value,name; results[name]=True
    expect('dormant_nonzero_valid_unchanged',check('unchanged',[7,2],[7,2]))
    expect('dormant_any_mutation_rejected',not check('unchanged',[7,2],[7,3]))
    expect('active_RQ_reset0_accepted',check('reset0',[0,0],[0,0]))
    expect('active_RQ_nonzero_rejected',not check('reset0',[0,0],[1,0]))
    expect('active_P1_expected_transition_accepted',check('p1',[12,3,12,3],[20,5,20,5],4,2))
    expect('active_P1_unchanged_rejected',not check('p1',[12,3],[12,3],4,1))
    expect('active_P1_wrong_flag_rejected',not check('p1',[12,3],[16,3],4,1))
    expect('active_P1_wrong_ticket_rejected',not check('p1',[12,3],[15,4],4,1))
    expect('unknown_protocol_HOLD',not check('unknown',[0],[0]))
    expect('offset_alias_detected',overlap(('allocation0',0,16),('allocation0',8,24)))
    expect('disjoint_views_not_alias',not overlap(('allocation0',0,8),('allocation0',8,24)))
    active_storages={'shared_RQ'}
    aliases={'rq32':'shared_RQ','rq16':'shared_RQ','dormant_p1':'p1'}
    expect('shared_RQ_aliases_both_receive_active_reset',all(('reset0' if aliases[name] in active_storages else 'unchanged')=='reset0' for name in ('rq32','rq16')))
    expect('unrelated_storage_keeps_dormant_contract',aliases['dormant_p1'] not in active_storages)
    functions=['kernel_fn','_r6','_v12','_gvcall','_bk','_cublas','_rqcall']
    source_bindings={}
    for folder in ['pw36','rs32','smallm_increment']:
        path=BASE/'candidates'/folder/'submission.py'; source=path.read_text(); tree=ast.parse(source)
        bound={node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
               for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in functions}
        assert set(bound)==set(functions)
        source_bindings[folder]={'sha256':hashlib.sha256(source.encode()).hexdigest(),'dispatch_AST':bound}
    expect('all_comparators_same_dispatch_AST',all(source_bindings[f]['dispatch_AST']==source_bindings['pw36']['dispatch_AST'] for f in source_bindings))
    return {'CPU_ONLY':True,'checks':results,'source_bindings':source_bindings,'limitations':'Policy-model tests, not GPU execution or proof of a recorded failed tensor. P1 model applies only live tile words, no integer overflow, exact launch accounting; unsupported runtime paths HOLD.'}
if __name__=='__main__': print(json.dumps(run(),indent=2))
