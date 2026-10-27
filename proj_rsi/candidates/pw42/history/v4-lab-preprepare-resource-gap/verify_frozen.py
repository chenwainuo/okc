"""Read-only frozen source, codegen, CSR and lab verification; no CUDA context."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import preflight as G
import csr_proof
import physical_proof
import metadata_proof
HERE=Path(__file__).resolve().parent

def main():
    m=json.loads((HERE/'manifest.json').read_text())
    for name,digest in m['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    from proj_rsi import learning
    registry=json.loads((HERE.parents[2]/'proj_rsi/learnings.json').read_text());retrieved=json.loads((HERE/'learning-proof.json').read_text())
    for row in retrieved['applied']:
        name,version=row['ref'].split('@');record=next(r for r in registry['lessons'] if r['id']==name and r['version']==int(version));assert record['record_sha256']==learning.digest(record)==row['record_sha256']
    source=G.source_proof();assert G.device((HERE/'submission.py').read_text(),True)==(HERE/'device_source.cu').read_text()
    a,b=G.tables((HERE/'sass.txt').read_text()),G.tables((HERE/'control-sass.txt').read_text())
    for args in G.BASES:assert a[('generic',args)]==b[('generic',args)]
    assert all(n==64 for n in G.resource((HERE/'device-resource.log').read_text()).values())
    physical=physical_proof.proof((HERE/'sass.txt').read_text());assert json.dumps(physical,sort_keys=True)==json.dumps(json.loads((HERE/'physical-proof.json').read_text()),sort_keys=True)
    csr=csr_proof.proof();assert json.dumps(csr,sort_keys=True)==json.dumps(json.loads((HERE/'csr-proof.json').read_text()),sort_keys=True)
    metadata=metadata_proof.proof();assert json.dumps(metadata,sort_keys=True)==json.dumps(json.loads((HERE/'metadata-proof.json').read_text()),sort_keys=True)
    spec=importlib.util.spec_from_file_location('pw42_frozen_lab_source',HERE/'lab.py');lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
    assert lab.source()==(HERE/'lab_source.cu').read_text();lab.actual_binary_gate((HERE/'sass.txt').read_text())
    print(json.dumps(dict(state='PASS_READONLY_PW42_FROZEN',source_sha256=source['candidate_sha256'],generic_controls_exact=2,guarded_64reg_zero_spill=2,CSR='PASS',metadata='ACTUAL_FIRSTSENTINEL_ELIGIBILITY',GPU='UNRUN'),indent=2))
if __name__=='__main__':main()
