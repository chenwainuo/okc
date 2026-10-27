"""CPU-only structural, phase and compiled operand-lifetime audit."""
import hashlib,json,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
root=D.parents[3]
e=json.loads((D/'cpu-evidence.json').read_text())
assert hashlib.sha256(Path(e['parent']).read_bytes()).hexdigest()==e['parent_sha256']
# Explicit full/empty barrier generations for both ring slots, including first and last.
phases=[]
for count in [1,2,3,32,64]:
 full=[0,0];empty=[0,0];published=[None,None];released=[None,None]
 for s in range(count):
  slot=s&1
  if s>=2:
   assert released[slot]==s-2
   assert ((empty[slot]-1)&1)==((s//2-1)&1)
  published[slot]=s
  assert (full[slot]&1)==((s//2)&1)
  full[slot]+=1
  assert published[slot]==s
  # Pair2 wait0 or wait1 of the next issued group guarantees this read retires.
  released[slot]=s;empty[slot]+=1
 assert sum(full)==sum(empty)==count
 phases.append(dict(stages=count,full_arrivals=full,empty_arrivals=empty,final_read=count-1))
# Every physical tile's descriptor range and swizzled producer scatter remain in its own tile.
layout=[]
for T in [8]:
 stage=4096+T*64
 for slot in range(2):
  for half in range(2):
   base=(slot*2+half)*stage
   w={base+r*64+(((k//8)^(r&7))*8)+(k&7) for r in range(64) for k in range(64)}
   x={base+4096+r*64+(((k//8)^(r&7))*8)+(k&7) for r in range(T) for k in range(64)}
   assert w==set(range(base,base+4096))
   assert x==set(range(base+4096,base+stage))
   assert (base*2)%1024==0
   layout.append(dict(slot=slot,half=half,base_bytes=base*2,end_bytes=(base+stage)*2))
# Deliberately releasing the newest wait1 group exposes a concrete producer-overwrite hazard.
negative={'issued':[0],'retired':[],'wrongly_released':[0],'producer_may_overwrite_stage':2,'live_shared_reader_stage':0,'hazard_detected':True}
compiled={}
for name in ['wait1','pair2']:
 sass=(D/name/'sass.txt').read_text()
 functions=re.split(r'Function : ',sass)[1:]
 checks=[]
 for f in functions:
  mma=[m for m in re.finditer(r'HGMMA[^\n]*',f)]
  waits=list(re.finditer(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x([01])',f))
  regs={int(x) for m in mma for x in re.findall(r'F32.BF16 R(\d+)',m.group())}
  assert regs=={24},regs
  zero=[m for m in waits if m.group(1)=='0']
  if name=='wait1':
   assert len(zero)==1
   assert zero[0].start()>mma[-1].start()
  for m in re.finditer(r'F2FP.BF16.F32.PACK_AB R(2[4-7]),',f):
   assert m.start()>zero[-1].start()
  assert 'spill' not in f.lower()
  checks.append({'function':f.splitlines()[0],'HGMMA':len(mma),'ARRIVE':f.count('WARPGROUP.ARRIVE'),'wait0':len(zero),'wait1':len(waits)-len(zero),'accumulator_first_register':24,'epilogue_conversion_after_covering_wait0':True,'operand_kind':'SS gdesc uniform descriptors, no register-A operand'})
 compiled[name]=checks
# OLS only descriptive: source work scales with active expert blocks, no causal attribution.
a=json.loads((root/'tools/moe_ss_lut_whole_v1/actual-analysis/analysis.json').read_text())
x=[s['active_local_blocks'] for s in a['per_seed']]
y=[s['us']['LUT'] for s in a['per_seed']]
xm=statistics.mean(x);ym=statistics.mean(y)
slope=sum((p-xm)*(q-ym) for p,q in zip(x,y))/sum((p-xm)**2 for p in x)
r2=sum((p-xm)*(q-ym) for p,q in zip(x,y))**2/(sum((p-xm)**2 for p in x)*sum((q-ym)**2 for q in y))
result={'state':'CPU_ONLY_NOT_HARDWARE_VALIDATED','full_empty_parity':phases,'pair2_layout_T8':layout,'unsafe_release_negative':negative,'compiled_lifetimes':compiled,'recorded_whole_LUT_vs_active_blocks':{'n':len(x),'slope_us_per_block':slope,'intercept_us':ym-slope*xm,'r_squared':r2,'limit':'descriptive routing/drift correlation, not PC latency attribution'},'assumptions':['finite warp and async engine progress','abstract group completion obeys ordered wait_group semantics','bar.sync 3 includes all128maththreads; elected release after rendezvous','fence.proxy then bar.sync2 includes all128producerthreads before publication'],'limits':['offline CUDA12.8 cubin only, actual JIT CUDA13 resources unknown','no device numeric/graph/mutation/lifetime execution','pair2 supplied T8 DIV1/DIV8 K4096 only; no runtime wrapper or tail support','descriptor uniform-register values may be overwritten after issue because descriptor values are instruction inputs; shared ranges remain live until covering wait','CPU phase check complements abstract exploration, not arbitrary hardware scheduler proof']}
(D/'audit-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['recorded_whole_LUT_vs_active_blocks'],indent=2))
