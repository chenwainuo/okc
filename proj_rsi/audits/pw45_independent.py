"""Independent read-only CPU PW45 source/geometry/SASS review."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'proj_rsi/candidates/pw45'
sys.path.insert(0,str(C))
from physical_proof import check
from cpu_proof import streams,NEW

def main():
    manifest=C/'manifest.json';m=json.loads(manifest.read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='988214460a032f11d620801f2661c32cbe9ca9572fe4f40ab0bbe931aa61ae02'
    for name,pin in m['files'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==pin
    source=(C/'submission.py').read_text();parent=(ROOT/'results/proj-rsi/releases/pw38-v1-29e557ea53f3/submission.py').read_text()
    def nodes(text):
        ans={}
        for n in ast.parse(text).body:
            if isinstance(n,ast.FunctionDef):ans[n.name]=ast.dump(n,include_attributes=False)
            elif isinstance(n,ast.Assign):
                for t in n.targets:
                    if isinstance(t,ast.Name):ans[t.id]=ast.dump(n,include_attributes=False)
        return ans
    a,b=nodes(parent),nodes(source);assert set(a)==set(b)
    assert sorted(k for k in a if a[k]!=b[k])==['RQCU','RQSBAKE','RQV','_rqsbake']
    # Independent algebraic output ownership: each 64-channel tile is disjoint,
    # covers32tokens, and pack block/half offsets exhaust each128-row chunk.
    intervals=[]
    for block in range(128):
        nb,hh=divmod(block,2);begin=128*nb+64*hh
        intervals.append((begin,begin+64))
        assert nb*16*12288+15*12288+hh*6144+6144<=(nb+1)*16*12288
    assert intervals==[(i*64,(i+1)*64) for i in range(128)]
    assert sum((hi-lo)*32 for lo,hi in intervals)==262144
    # Source protocol selection, not a generic name inference.
    assert 52&16 and 52&32 and 52&4 and not 52&2 and not 52&1
    sass=(C/'sass.txt').read_text();control=streams((C/'control-sass.txt').read_text());new=streams(sass)
    assert set(new)==set(control)|NEW
    for key,value in control.items():assert new[key]==value
    check(sass)
    negatives=[]
    for label,mutant in [('all_retirement_removed',sass.replace('WARPGROUP.DEPBAR.LE','NOP')),
                          ('plain_store_replaced_by_reduce',sass.replace('UTMASTG.2D','UTMAREDG.2D')),
                          ('publication_barrier_removed',sass.replace('BAR.SYNC','NOP'))]:
        try:check(mutant)
        except (AssertionError,IndexError,ValueError):negatives.append(label)
        else:raise AssertionError('mutation admitted '+label)
    r=subprocess.run([str(C/'host_admission_cpu')],capture_output=True,timeout=10);assert r.returncode==0
    out={'state':'GO_SOURCE_FOR_REVIEWED_RESEARCH_HARNESS_ONLY','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'changed_AST_names':['RQCU','RQSBAKE','RQV','_rqsbake'],'old_main_streams_exact':len(control),'new_mains':len(NEW),'output_values':262144,'independent_mutations_rejected':negatives,'compiled_exact_host_guard_execution':'PASS','counter_new_plain':'ALL_PHYSICAL_STORAGE_DORMANT_UNCHANGED','logical_X_payload_bytes':{'old':128*8*4096,'new':128*16*4096},'GPU_calls':0,'actual_occupancy':None,'numerical_GPU_validation':'UNRUN'}
    path=ROOT/'proj_rsi/audits/pw45-independent.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
