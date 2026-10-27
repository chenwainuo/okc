"""Bounded CPU negative controls; never edits a candidate or starts CUDA."""
import json
from pathlib import Path
import re
import actual_binary_gate as gate
import proofs
HERE=Path(__file__).resolve().parent

def inverse(body):
    delta=json.loads((HERE/'delta.json').read_text())
    for e in reversed(delta['edits']):
        assert body.count(e['after'])==1
        body=body.replace(e['after'],e['before'],1)
    assert body==(proofs.PARENT/'kernel_body.cuh').read_text()
    return body

def proof():
    body=(HERE/'kernel_body.cuh').read_text();inverse(body);rejected={}
    for name,a,b in [('missing_extra_tile','if(warp==12)decode_store(s,3','if(warp==13)decode_store(s,3'),
                     ('wrong_odd_barrier_count','bar.sync 4,96','bar.sync 4,128'),
                     ('wrong_Bempty_readers','[%0],8;\\n"::"r"(bempty','[%0],7;\\n"::"r"(bempty'),
                     ('wrong_Xempty_readers','[%0],8;\\n"::"r"(xempty','[%0],7;\\n"::"r"(xempty')]:
        assert a in body,name
        bad=body.replace(a,b,1)
        try:inverse(bad)
        except AssertionError:rejected[name]='REJECTED_STRICT_PARENT_INVERSE'
        else:raise AssertionError('inverse admitted '+name)
    sass=(HERE/'sass.txt').read_text();gate.check(sass)
    for name,bad in [('wrong_bar4_count',sass.replace('0x4, 0x60','0x4, 0x80',1)),
                     ('wrong_role_branch',sass.replace('@!P0 BRA 0x4360','@!P0 BRA 0x2360',1)),
                     ('wrong_instruction',sass.replace('HGMMA.','INVALID_HGMMA.',1))]:
        assert bad!=sass
        try:gate.check(bad)
        except AssertionError:rejected[name]='REJECTED_CANONICAL'
        else:raise AssertionError('actual gate admitted '+name)
    # Independent physical gate must reject wrong publication count even without
    # canonical equality; proves the count is checked rather than only inherited.
    try:proofs.physical(sass.replace('0x4, 0x60','0x4, 0x80',1))
    except AssertionError:rejected['physical_wrong_bar4_count']='REJECTED_PHYSICAL'
    else:raise AssertionError('physical count mutation admitted')
    return dict(status='CPU_NEGATIVE_GATES_PASS',controls=rejected,source_changed=False,GPU='UNRUN')

if __name__=='__main__':
    r=proof();(HERE/'negative-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'])
