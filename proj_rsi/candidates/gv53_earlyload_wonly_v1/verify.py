"""Focused CPU semantics/source/native receipts; no production import or GPU."""
import ast,hashlib,json,re,collections,random
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nodes(s):return ast.parse(s).body
def native(p):
 result={}
 for sec in p.read_text().split('Function : ')[1:]:
  name=sec.splitlines()[0].strip();result[name]=re.findall(r'/\*([0-9a-f]+)\*/\s+(?:(@!?P\d)\s+)?([^;]+);\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',sec)
 return result
def verify():
 base=(R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text();source=(D/'submission.py').read_text();delta=json.loads((D/'delta.json').read_text());assert sha(D/'submission.py')==delta['candidate_sha256']
 old,new=nodes(base),nodes(source);assert len(old)==len(new)==176
 changed=[getattr(a,'name',None)or a.targets[0].id for a,b in zip(old,new)if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False)];assert changed==['GVCU','_gvext']
 kernel=delta['native_kernel'];assert 'early_x'not in kernel and 'if (u+ST<KU) ldw(s,u+ST);'in kernel and 'if (u + ST < KU) ldx(s, u + ST);'in kernel
 # Each trace uses arbitrary raw W/X word identities. Current W snapshot is
 # independent of next W overwrite; current X is consumed before late refill.
 rng=random.Random(4703);traces=0
 for _ in range(256):
  W=[[rng.getrandbits(32)for _ in range(12)]for u in range(8)];X=[[rng.getrandbits(32)for _ in range(8)]for u in range(8)]
  wb=W[0][:];xb=X[0][:];wreads=[0];xreads=[0];seen=[]
  for u in range(8):
   early_w=wb[:]
   if u+1<8:wb=W[u+1][:];wreads.append(u+1)
   current_w=early_w;current_x=xb[:]
   if u+1<8:xb=X[u+1][:];xreads.append(u+1)
   seen.append((current_w,current_x));assert seen[-1]==(W[u],X[u])
  assert wreads==xreads==list(range(8));traces+=1
 S=native(D/'pilot.sass');prior=native(R/'proj_rsi/candidates/gv53_earlyload_v1/thin.sass');parent='_Z10gv_predxorILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G';wx='_Z12gv_earlyloadILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G';wo='_Z18gv_earlyload_wonlyILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G'
 assert S[parent]==prior[parent]and S[wx]==prior[wx]
 info={}
 for label,name in [('parent',parent),('W_X',wx),('W_only',wo)]:
  rows=S[name];first=min(int(a,16)for a,b,c,d,e in rows if c.startswith('PRMT '));w=[int(a,16)for a,b,c,d,e in rows if b and c.startswith('LDG.E.NA.128')];x=[int(a,16)for a,b,c,d,e in rows if b and c.startswith('LDG.E.128')];assert len(w)==3 and len(x)==2
  count=collections.Counter(c.split()[0]for a,b,c,d,e in rows);info[label]={'instructions':len(rows),'non_NOP':len(rows)-count['NOP'],'opcode_counts':dict(count),'first_decode_PRMT_PC':hex(first),'next_W_PCs':[hex(p)for p in w],'next_X_PCs':[hex(p)for p in x],'next_W_all_before_first_decode':max(w)<first,'next_X_all_before_first_decode':max(x)<first}
  for op,n in [('LDG.E.NA.128.CONSTANT',6),('LDG.E.128.CONSTANT',4),('SHFL.IDX',2),('P2R',1),('HMMA.16816.F32.BF16',4)]:assert count[op]==n
 assert info['W_only']['next_W_all_before_first_decode']and not info['W_only']['next_X_all_before_first_decode']
 log=(D/'compile.stderr').read_text();assert log.count('Used 64 registers')==3 and log.count('0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads')==3
 return {'state':'CPU_SOURCE_AND_CODEGEN_PASS_GPU_UNRUN','source_sha256':sha(D/'submission.py'),'parent_source_sha256':delta['parent_sha256'],'W_X_parent_sha256':delta['W_X_parent_sha256'],'changed_AST_nodes':changed,'other_original_AST_nodes_exact':174,'whole_source_inverse_to_501d':delta['whole_file_inverse_to_501d'],'native_inverse_to_W_X_except_explicit_W_X_refill_split':delta['native_inverse_to_W_X'],'raw_operand_traces':traces,'units_each':8,'W_X_address_multisets_and_order_exact':True,'last_unit_no_refill':True,'escape_cursor_HMMA_FP32_reduction_output_source_order':'Preserved; only raw operand capture/refill source altered.','native':info,'registers_each':64,'stack_spills_each':0,'dynamic_shared_bytes':17408,'runtime_occupancy':'UNQUERIED; same64registers/1024threads modeloneCTA, not measured scheduling.','hypothesis':'W-only issue can retain overlap with less X staging; not yet measured.','counterevidence':'W-only392total/378nonNOP vs W_X384/374, so removing sourceXsnapshot did NOT yield cheaper static code. X refills partly hoisted within decode; not isolated exact-parent X latency.','GPU_gain_us':None,'limits':['No GPU numerics/performance or actualselection validated.','Exact source math/order and raw operand traces are CPU evidence, not runtime-output proof.','Do not promote or impose a minimum gain gate; retain all signed three-arm results.']}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
