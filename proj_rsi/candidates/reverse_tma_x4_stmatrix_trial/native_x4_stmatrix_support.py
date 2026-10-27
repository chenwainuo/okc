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
 'reverse_tma_x4':'d47ed1663eac87f5bcc05b5f17df8fa62e7c9e91780a13d02ce9b95ac0f05c8a',
 'reverse_tma_x_stmatrix':'2c132b82a854e476f433335462b2122f21bc3cb490dec6f2b691c01cd9775045',
 'reverse_tma_x4_stmatrix':'11892b99e5e7b87c6e7413202701f18f1a2fe0407596d8e7dea184881981b22c'}
TOKENS={'native':'rq_tma_x','x4':'rq_tma_x4','stmx':'rq_tma_x_stmx','new':'rq_tma_x4_stmx'}
FOLDERS={'native':'reverse_tma_x','x4':'reverse_tma_x4','stmx':'reverse_tma_x_stmatrix','new':'reverse_tma_x4_stmatrix'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(folder,name,label):
    path=CANDIDATES/folder/(name+'.py')
    spec=importlib.util.spec_from_file_location('native_x4_stmatrix_'+label+'_'+name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def verify_frozen():
    result={}
    for folder,pin in PINS.items():
        directory=CANDIDATES/folder;assert sha(directory/'manifest.json')==pin,(folder,'manifest drift')
        manifest=json.loads((directory/'manifest.json').read_text())
        for name,expected in manifest['files'].items():assert sha(directory/name)==expected,(folder,name,'drift')
        result[folder]={'manifest_sha256':pin,'files':len(manifest['files'])}
    for folder in FOLDERS.values():
        source=(CANDIDATES/folder/'extension_source.cu').read_text()
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
    # Bind exact version dependencies; no cross-version checker/proof imports.
    prior={name:sys.modules.get(name) for name in ('liveness','proofs')}
    sys.modules['liveness']=checker
    try:
        proof=load(folder,'proofs',label)
        sys.modules['proofs']=proof
        detail=dict(layout=proof.layout(),protocol=proof.protocol(),physical=proof.physical(sass))
        if label=='x4':detail['source_restore']=proof.source_restore()
        if label=='new':detail['composition']=load(folder,'composition_proof',label).proof()
        if label in ('stmx','new'):
            gate=load(folder,'actual_binary_gate',label)
            detail['candidate_actual_binary_gate']=gate.check(sass)
        record['X_layout_protocol_physical']=detail
    finally:
        for name,mod in prior.items():
            if mod is None:sys.modules.pop(name,None)
            else:sys.modules[name]=mod
    record['producer_output_owner_proxy']='EXACT_CANONICAL_MAIN_AND_FROZEN_SOURCE_PROOF'
    return record


def verify_lab():
    manifest=json.loads((HERE/'manifest.json').read_text())
    root=HERE.parents[1]
    for name,expected in manifest['files'].items():assert sha(root/name)==expected,('lab drift',name)
    return manifest
