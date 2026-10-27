"""Compile and run exact isolated C++ host admission expressions on CPU; no CUDA."""
import json,subprocess
from pathlib import Path
from build_patch import value
HERE=Path(__file__).resolve().parent

def proof():
    cu=value((HERE/'submission.py').read_text(),'RQCU')
    begin=cu.index('  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;')
    end=cu.index('  if (p.ncnt <',begin);guard=cu[begin:end]
    source='#include <cassert>\nstruct P{int m,N,K,KC,walias;};\ntemplate<int NT,int ST,int GS,int FL> int admission(P p,int KS){\n'+guard+'  return 0;\n}\n'
    source+='int main(){P p{32,8192,1024,16,0};assert((admission<32,4,1,52>(p,1)==0));assert((admission<32,5,1,52>(p,1)==0));assert((admission<32,4,1,52>(p,2)==-45));assert((admission<32,6,1,35>(p,2)==0));\n'
    source+='for(int m:{16,31,33}){P q=p;q.m=m;assert((admission<32,4,1,52>(q,1)!=0));}\n'
    source+='P q=p;q.N=4096;assert((admission<32,5,1,52>(q,1)==-45));q=p;q.K=4096;q.KC=64;assert((admission<32,5,1,52>(q,1)==-45));q=p;q.walias=1;assert((admission<32,5,1,52>(q,1)==-45));assert((admission<32,5,1,52>(p,0)==-4));assert((admission<32,5,1,52>(p,65)==-4));}\n'
    source=source.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>')
    path=HERE/'host_admission.cpp';binary=HERE/'host_admission_cpu';path.write_text(source)
    r=subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(binary)],text=True,capture_output=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([str(binary)],text=True,capture_output=True,timeout=10);assert r.returncode==0,r.stdout+r.stderr
    return {'state':'PASS_ACTUAL_ISOLATED_CXX_GUARD','new_legal_types':2,'negative_cases':9,'old_selected_ST6_KS2_admission_unchanged':True,
        'scope':'Exact isolated host guard compiled and executed on CPU; full Torch/CUDA wrapper linking and runtime attributes remain unrun.','GPU_calls':0}

if __name__=='__main__':
    r=proof();(HERE/'host-admission-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
