"""CPU-only schedule/source helpers for the fixed-geometry cursor diagnostic."""
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LABELS=('parent5','cursor5','parent6','cursor6','actual_pw36','actual_rs32','actual_cursor')
SHAPES={'qkv32':('qkv_gate',32,2304,8192,6),'pooler32':('pooler',32,4096,8192,4),'unpooler32':('unpooler',32,8192,4096,2)}
COMPARISONS={'ST5':('parent5','cursor5'),'ST6':('parent6','cursor6'),'actual_pw36':('actual_pw36','actual_cursor'),'actual_rs32':('actual_rs32','actual_cursor')}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def driver():return module(ROOT/'proj_rsi/tools/smallm_increment_lab_v3.py','frozen_cursor_v3')
def schedule():
 events=[]
 for label in LABELS:
  for _ in range(5):events.append(('warmup',label,239999,'warmup'))
 blocks=[('actual_pw36_before','actual_pw36'),('actual_rs32_before','actual_rs32'),('actual_cursor','actual_cursor'),
         ('parent5_before','parent5'),('cursor5','cursor5'),('parent5_after','parent5'),
         ('parent6_before','parent6'),('cursor6','cursor6'),('parent6_after','parent6'),
         ('actual_rs32_after','actual_rs32'),('actual_pw36_after','actual_pw36')]
 for block,(key,label) in enumerate(blocks):
  events.extend(('F1',label,240000+block*21+i,key) for i in range(21))
 events.append(('terminal','F1',None,None))
 orders=[]
 for sequence in (tuple(range(7)),tuple(reversed(range(7)))):
  orders.extend(sequence[j:]+sequence[:j] for j in range(7))
 for row in range(28):
  events.extend(('paired',LABELS[j],250000+row,'paired') for j in orders[row%14])
 events.append(('terminal','paired',None,None))
 return events

def verify():
 m=json.loads((HERE/'manifest.json').read_text())
 for rel,pin in m['files'].items():assert sha(ROOT/rel)==pin,('matched diagnostic source drift',rel)
 return m
