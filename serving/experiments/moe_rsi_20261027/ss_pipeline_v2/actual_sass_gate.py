"""Future actual-JIT gate; consumes root-collected SASS, performs no CUDA calls."""
import importlib.util,re
from pathlib import Path
D=Path(__file__).resolve().parent
def streams(raw):return {t.splitlines()[0].strip():[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',t)]for t in raw.split('Function : ')[1:]if '7projectILi' in t.splitlines()[0]}
def check(actual):
 assert streams(actual)==streams((D/'sass.txt').read_text()),'actual eight-kernel PC/predicate/opcode drift'
 sp=importlib.util.spec_from_file_location('ss_actual_physical',D/'cpu_proof.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 return m.physical(actual)
