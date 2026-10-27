"""Read-only frozen PW45 verifier; no compiler, torch, context, prepare or writes."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]))
import cpu_proof as G
import physical_proof
from protocol_proof import layout,proof
from learning_proof import proof as lessons

def verify():
    m=json.loads((HERE/'manifest.json').read_text())
    for name,pin in m['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin,name
    for name,pin in m.get('external_files',{}).items():assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()==pin,name
    source=G.source_proof();a=G.streams((HERE/'sass.txt').read_text());b=G.streams((HERE/'control-sass.txt').read_text())
    assert set(a)==set(b)|G.NEW
    for args,rows in b.items():assert a[args]==rows,args
    regs=G.resources((HERE/'device-resource.log').read_text())
    assert all(regs[args]==168 for args in G.NEW)
    p={'layout':layout(),'protocol':proof()}
    assert json.dumps(p,sort_keys=True)==json.dumps(json.loads((HERE/'protocol-proof.json').read_text()),sort_keys=True)
    physical=physical_proof.check((HERE/'sass.txt').read_text())
    assert json.dumps(physical,sort_keys=True)==json.dumps(json.loads((HERE/'physical-proof.json').read_text()),sort_keys=True)
    learning=lessons();assert learning==json.loads((HERE/'learning-proof.json').read_text())
    return {'state':'PASS_READONLY_FROZEN_PW45','source_sha256':source['new_sha256'],'old_control_main_streams':len(b),'new_mains':len(G.NEW),'full_weight_BF16_values':p['layout']['weight_BF16_values'],'output_values':p['layout']['output_BF16_values'],'protocol_states':p['protocol']['total_states'],'actual_GPU_occupancy':None,'GPU_calls':0}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
