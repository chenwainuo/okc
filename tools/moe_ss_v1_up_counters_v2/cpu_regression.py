"""Exercise the exact repaired branch without Torch, CUDA or source imports."""
import ast,json,types
from pathlib import Path
tree=ast.parse(Path(__file__).with_name('probe.py').read_text())
branch=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=='expected_blocks == 0'and isinstance(n.body[0],ast.Assign))
check=ast.Assert(test=ast.Name(id='ok',ctx=ast.Load()),msg=ast.Name(id='msg',ctx=ast.Load()))
body=ast.Module(body=[branch,check],type_ignores=[]);ast.fix_missing_locations(body)
calls=[]
def checker(*args):
 calls.append(args)
 return True,'original task checker called'
fake_torch=types.SimpleNamespace(uint8='uint8',count_nonzero=lambda x:x)
class Output:
 def __init__(self,nonzero):self.nonzero=nonzero
 def view(self,dtype):assert dtype=='uint8';return self.nonzero
def run(blocks,nonzero):
 ns={'expected_blocks':blocks,'answer':Output(nonzero),'torch':fake_torch,'reference':types.SimpleNamespace(check_implementation=checker),'data':'data','weights':'weights'}
 exec(compile(body,'<exact driver validation branch>','exec'),ns)
run(0,0);assert not calls
try:run(0,1)
except AssertionError:pass
else:raise AssertionError('nonzero empty output accepted')
assert not calls
run(6,1);run(7,1);assert len(calls)==2
print(json.dumps({'state':'PASS','empty_zero_no_original_checker':True,'empty_nonzero_rejected_no_original_checker':True,'G6_G7_original_checker_called':True,'GPU_calls':0}))
