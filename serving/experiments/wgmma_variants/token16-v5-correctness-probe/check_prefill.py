"""Bounded official-reference prefill sentinel checks; no timing/graph claim."""
import hashlib,importlib.util,json
from pathlib import Path
from rsi_worker import setup_paths,fingerprint
setup_paths(Path('/baseline'))
import torch
import reference
spec=importlib.util.spec_from_file_location('_token16_prefill_candidate','/probe/serving_wgmma_down.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
seed=963019
weights=reference.generate_weights(seed)
state=candidate.prepare({k:v.clone() for k,v in weights.items()})
records=[]
for index,batch in enumerate((3012,16384)):
    data=reference.generate_input(batch,'ep8',seed+100+index)
    before=fingerprint(data);out=candidate.kernel_fn(data,state)
    assert fingerprint(data)==before,'candidate mutated prefill input before oracle'
    ok,detail=reference.check_implementation(data,out,weights);assert ok,detail
    assert fingerprint(data)==before,'prefill input changed'
    torch.cuda.synchronize()
    records.append({'batch':batch,'input_seed':seed+100+index,'input_sha256':before,'official_reference_pass':True,'detail':str(detail)})
    Path('/output/prefill-progress.json').write_text(json.dumps({'status':'incomplete','cases':records},indent=2)+'\n')
    del data,out
report={'status':'complete','scope':'two eager full-expert prefill correctness sentinels only; not full-suite validation or timing','weight_seed':seed,'cases':records,'reference_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()}
Path('/output/prefill-correctness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
