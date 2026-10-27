"""Execute only pure instance/choice metadata AST, never production imports."""
import ast
import json
import torch
from pathlib import Path
HERE=Path(__file__).resolve().parent

def target_names(node):
    if isinstance(node,ast.Name):return [node.id]
    if isinstance(node,(ast.Tuple,ast.List)):return [name for n in node.elts for name in target_names(n)]
    return []

def metadata(text):
    tree=ast.parse(text);start=end=None
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.Assign):
            names=[name for t in node.targets for name in target_names(t)]
            if '_0' in names:start=i
            if '_gvmod' in names and start is not None:end=i+1;break
    assert start is not None and end is not None
    nodes=tree.body[start:end]
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.With)) for node in nodes for n in ast.walk(node))
    assert not any(isinstance(n,ast.Name) and n.id in ('torch','time','load_inline','prepare') for node in nodes for n in ast.walk(node))
    env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-instance-metadata>','exec'),env)
    return env

def proof():
    old=(HERE.parent/'pw38/submission.py').read_text();source=(HERE/'submission.py').read_text();a,b=metadata(old),metadata(source)
    assert b['GVI'][:len(a['GVI'])]==a['GVI']
    assert b['GVI'][len(a['GVI']):]==[tuple(c[:-1])+(c[-1]|(1<<30),) for c in b['GV_EMPTY_BASES']]
    assert len(b['GVI'])<1000
    assert all(not (args[-1] & (1<<30)) for args in a['GVI'])
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_gvee_choices')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pure-optional-choice>','exec'),b)
    extra=[]
    for (layer,mt),geos in a['GVC'].items():
        for bm in (1,8,16,32):
            if a['GVB'][bm]!=mt:continue
            for geo,vs in geos:
                empty=torch.full((64,),-1,dtype=torch.int32,device="cpu");nonempty=torch.ones(64,dtype=torch.int32,device="cpu");mixed=nonempty.clone();mixed[32:]=-1
                choices=b['_gvee_choices'](layer,bm,geo,vs,True,empty)
                assert b['_gvee_choices'](layer,bm,geo,vs,True,nonempty)==tuple(vs)
                assert b['_gvee_choices'](layer,bm,geo,vs,True,None)==tuple(vs)
                assert b['_gvee_choices'](layer,bm,geo,vs,True,mixed)==choices
                assert choices[:len(vs)]==tuple(vs)
                for v in choices[len(vs):]:
                    args=(1,mt,b['_gvst'](geo))+geo[:6]+(v,)
                    assert layer=='qkv_gate' and bm in (1,8) and args in b['GVI']
                    assert args[:-1]+(v&~(1<<30),) in b['GV_EMPTY_BASES']
                    extra.append(dict(layer=layer,M=bm,args=args))
                assert b['_gvee_choices'](layer,bm,geo,vs,False,empty)==(0,)
    assert len(extra)==4
    return dict(state='PASS_GENERIC_IDS_PREFIX_AND_ACTUAL_CHOICES',original_GVI=len(a['GVI']),new_GVI=len(b['GVI']),extra_choices=extra,
                generic_fallback='Every old choice retained in oldorder; noC12 pack retains oldraw0choice; nonQKV/M>8 receive noextras',
                host_eligibility='Exact actual prepared firstentries.any sentinel; zeroempty orNone retains oldchoices',actual_weight_gate='Only perwarp actualfirstsentinel enters emptycompute; optional trait is guarded, not a blanket no-escape declaration')
if __name__=='__main__':
    result=proof();(HERE/'metadata-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
