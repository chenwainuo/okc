"""Load exact preserved legacy code for isolated compatibility/perf tests, not live queue."""
import hashlib
from pathlib import Path
import sys
import types
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'baseline-store-fbc90c7.py'
SHA256='e1b82e5472a41dfbec7e1ee3b547c345de201e441f424557f9c9fef6b6c1cf1f'
def load():
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA256,'preserved baseline drift'
    name='proj_rsi.scheduling._legacy_fbc90c7_review'
    module=types.ModuleType(name);module.__package__='proj_rsi.scheduling'
    # Preserve original default repo-root calculation without importing live store.
    module.__file__=str(HERE.parents[1]/'scheduling/store.py')
    module._review_source_path=SOURCE;sys.modules[name]=module
    exec(compile(raw,str(SOURCE),'exec'),module.__dict__)
    return module
