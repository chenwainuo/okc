"""CPU-only identity and exact JIT main-semantic gates. Never loads an extension."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent
CANDIDATES=HERE.parent
PINS={
 'reverse_tma_x':'025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc',
 'operand_reverse_dual_producer':'281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241',
 'operand_reverse_dual_stmatrix':'6785da2ec9881ce13148951daff2f8802d8e7bbe6c83abf588607092b3f97072'}
TOKENS={'new':'rq_tma_x','dp':'rq_px_dp','ds':'rq_px_dp_stmx'}
FOLDERS={'new':'reverse_tma_x','dp':'operand_reverse_dual_producer','ds':'operand_reverse_dual_stmatrix'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(folder,name,label):
    path=CANDIDATES/folder/(name+'.py')
    spec=importlib.util.spec_from_file_location('native_x_'+label+'_'+name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def verify_frozen():
    result={}
    for folder,pin in PINS.items():
        directory=CANDIDATES/folder;assert sha(directory/'manifest.json')==pin,(folder,'manifest drift')
        manifest=json.loads((directory/'manifest.json').read_text())
        for name,expected in manifest['files'].items():assert sha(directory/name)==expected,(folder,name,'drift')
        result[folder]={'manifest_sha256':pin,'files':len(manifest['files'])}
    source=(CANDIDATES/'reverse_tma_x/extension_source.cu').read_text()
    assert 'int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);' in source
    assert 'p.XP =' not in source and 'xp.data_ptr' not in source and 'rx_pack' not in source
    assert source.count('k<<<dim3(')==1
    assert not re.search(r'(?:cudaMemcpy|\.item\(|\.cpu\()',source)
    return result

def main_table(sass,token):
    tables=[t for t in re.split(r'\n\s*Function : ',sass)[1:] if token in t.splitlines()[0]]
    assert len(tables)==1,('expected exactly one main',token,len(tables))
    return tables[0]

def instructions(table):
    # Full PC+predicate+instruction equality, including control targets. Host
    # symbols, addresses, scheduling encodings and ELF wrapper are excluded.
    return [(int(m[1],16),m[2].strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',table)]

def actual_binary_gate(label,sass):
    folder=FOLDERS[label];token=TOKENS[label]
    local=(CANDIDATES/folder/'sass.txt').read_text()
    actual=main_table(sass,token);canonical=main_table(local,token)
    ai,ci=instructions(actual),instructions(canonical)
    assert ai and ai==ci,(label,'JIT main differs from frozen CPU cubin semantic instruction stream')
    checker=load(folder,'liveness',label)
    record={'canonical_main_semantic_equality':'PASS_ALL_PC_PREDICATE_INSTRUCTIONS_EXACT',
            'canonical_main_instructions':len(ci),'retirement_and_fence_gate':checker.check(sass),
            'canonical_sass_sha256':sha(CANDIDATES/folder/'sass.txt')}
    if label=='new':
        # proofs imports its liveness dependency by name; caller already binds
        # the native-X immutable directory first on sys.path.
        proof=load(folder,'proofs',label)
        record['X_layout_protocol_physical']=dict(layout=proof.layout(),protocol=proof.protocol(),physical=proof.physical(sass))
        # This exact main equality binds producer/output owner and proxy fences
        # to frozen proofs; does not repurpose DP's mandatory 32 LDG assertion.
        record['producer_output_owner_proxy']='EXACT_CANONICAL_MAIN_AND_FROZEN_PRODUCER_SOURCE_PROOF'
    else:
        store=load('operand_reverse_dual_stmatrix','liveness','store_'+label)
        record['store_form_gate']=store.store_form(sass,token,label=='ds',dual=True)
    return record


def verify_lab():
    manifest=json.loads((HERE/'manifest.json').read_text())
    root=HERE.parents[1]
    for name,expected in manifest['files'].items():assert sha(root/name)==expected,('lab drift',name)
    return manifest
