"""Root-only observational probe bootstrap; use exclusive GPU lease and timeout."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from rsi_worker import setup_paths

root = Path('/baseline')
setup_paths(root)
sys.path.insert(0, '/probe')
runpy.run_path('/probe/check_routed.py', run_name='__main__')
import torch

torch.cuda.synchronize()
imports = {}
for name, module in list(sys.modules.items()):
    if name == 'sglang' or name.startswith('sglang.'):
        location = getattr(module, '__file__', None)
        if location:
            path = Path(location).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f'Import escaped frozen tree: {name}: {path}')
            imports[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
libraries = {}
for line in Path('/proc/self/maps').read_text().splitlines():
    path = Path(line.split()[-1])
    if path.is_absolute() and path.suffix == '.so' and path.is_relative_to('/cache') and path.is_file():
        libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
if not libraries:
    raise RuntimeError('No private-cache JIT shared library recorded')
print(json.dumps({'status': 'complete', 'scope': 'correctness-only routed down projection',
                  'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name()},
                  'sglang_imports': imports, 'jit_libraries': libraries,
                  'source_hashes': {n: hashlib.sha256(Path('/probe', n).read_bytes()).hexdigest()
                                    for n in ('routed_down.cuh', 'routed_adapter.py', 'check_routed.py', 'run_routed_probe.py')}}))
