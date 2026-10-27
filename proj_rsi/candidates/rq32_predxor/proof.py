"""Read-only CPU source/inline-PTX truth and native physical register predicate gates."""
import ast,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import apply,PARENT_SHA,PARENT_MANIFEST
from compile import streams,ops,registers,TARGET,device
ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def named(text):
 out={}
 for node in ast.parse(text).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):out[node.name]=ast.dump(node,include_attributes=False)
  elif isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name):out[target.id]=ast.dump(node,include_attributes=False)
 return out

def native_gate(rows):
 starts=[idx for idx,(pc,ins) in enumerate(rows) if ins=='ISETP.NE.U32.AND P0, PT, R2, R5, PT'];assert len(starts)==4
 result=[]
 for block,start in enumerate(starts):
  stop=starts[block+1] if block+1<len(starts) else len(rows)
  part=rows[start:stop];pred={};dest={};mask_regs=set();last_xor=None
  for pc,ins in part:
   eq=re.fullmatch(r'ISETP.EQ.U32.AND (P\d+), PT, R\d+(?:\.reuse)?, (0x[0-9a-f]+|RZ), !P0',ins)
   if eq:
    index=0 if eq[2]=='RZ' else int(eq[2],16);assert 0<=index<16;pred[eq[1]]=index
   xor=re.fullmatch(r'@(P\d+) LOP3.LUT R(\d+), R(\d+)(?:\.reuse)?, R(\d+)(?:\.reuse)?, RZ, 0x3c, !PT',ins)
   if xor:
    assert xor[1] in pred and xor[2]==xor[3]
    index=pred[xor[1]];assert index not in dest;dest[index]=int(xor[2]);mask_regs.add(xor[4]);last_xor=pc
    if len(dest)==16:break
  assert set(dest)==set(range(16)) and len(mask_regs)==1
  hgmma=[(pc,ins) for pc,ins in part if 'HGMMA.' in ins][:4];assert len(hgmma)==4 and last_xor<hgmma[0][0]
  for q,(pc,ins) in enumerate(hgmma):
   base=int(re.match(r'HGMMA\.64x32x16\.F32\.BF16 R\d+, R(\d+),',ins)[1])
   assert [dest[i] for i in range(4*q,4*q+4)]==list(range(base,base+4)),('predicate-register index differs physical MMA A',block,q,dest,ins)
  result.append({'escape_copy':block,'target_lane_NE_pc':hex(rows[start][0]),'logical_Aword_to_physical_GPR':dest,'last_patch_pc':hex(last_xor),'first_HGMMA_pc':hex(hgmma[0][0]),'predicate_XORs':16})
 return result

def inline_gate(source):
 cu=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RQCU' for t in n.targets))
 helper=cu.split('void rq_fix_pred(',1)[1].split('__device__ __forceinline__ unsigned long long rq_gt()',1)[0]
 lines=[tuple(map(int,m)) for m in re.findall(r'setp.eq.and.u32 pr, ri, (\d+), pl; @pr xor.b32 %(\d+), %(\d+), mask;',helper)]
 assert lines==[(i,i,i) for i in range(16)]
 assert 'mov.u32 mask, %16;' in helper and 'mov.u32 ri, %19;' in helper and 'setp.eq.u32 pl, %17, %18;' in helper
 assert re.findall(r'"\+r"\(A\[(\d+)\]\)',helper)==list(map(str,range(16)))
 assert ': "r"(xr), "r"(lane), "r"(target), "r"(reg)' in helper
 assert 'target = (nx >> 12) & 31, reg = (nx >> 8) & 15' in helper
 assert 'xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16)' in helper
 old=cu.split('void rq_fix(',1)[1].split('__device__ __forceinline__ void rq_fix_pred',1)[0]
 tail='    ++ei;\n    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);\n    nx = __shfl_sync(0xffffffffu, ec, ei & 31);\n    nxs = nx >> 17;'
 assert tail in old and tail in helper
 return lines

def truth(lines):
 cases=0;targets=0
 for target in range(32):
  for reg in range(16):
   for lane in range(32):
    for i,a,b in lines:assert ((lane==target) and (reg==i))==((lane==target) and (reg==a));targets+=1
   for half in range(2):
    for xor7 in range(128):
     entry=(17<<17)|(target<<12)|(reg<<8)|(half<<7)|xor7
     original=[(0xa5a5a5a5^(i*0x11111111))&0xffffffff for i in range(16)];want=list(original);got=list(original)
     mask=(entry&127)<<(8+((entry>>7)&1)*16)
     want[(entry>>8)&15]^=mask
     for selector,output,input_index in lines:
      if (entry>>12)&31==target and (entry>>8)&15==selector:got[output]=got[input_index]^mask
     assert got==want;cases+=1
 return {'encoded_entries':cases,'lane_register_predicate_truth_cases':targets,'includes_xor0_and127_both_halves_all_registers_all_lanes':True,'interpretation':'CPU exact emitted-inline-PTX instruction-template truth only; not GPU data or timing'}

def verify():
 parent=HERE.parent/'pw45';grand=HERE.parent/'pw38';old=(parent/'submission.py').read_text();new=(HERE/'submission.py').read_text();delta=json.loads((HERE/'delta.json').read_text())
 assert sha(parent/'submission.py')==PARENT_SHA and sha(parent/'manifest.json')==PARENT_MANIFEST
 assert apply(old,delta['edits'])==new and apply(new,delta['edits'],True)==old
 a,b=named(old),named(new);assert a.keys()==b.keys();changed=[k for k in a if a[k]!=b[k]];assert changed==['RQCU'],changed
 # Device body pre-host admission is exact PW38 ancestor before this helper edit.
 assert device(old).split('void instantiate(')[0]==device((grand/'submission.py').read_text()).split('void instantiate(')[0]
 assert device(old)==(parent/'device_source.cu').read_text()
 pa=streams((parent/'sass.txt').read_text());ch=streams((HERE/'sass.txt').read_text());ancestor=streams((parent/'control-sass.txt').read_text())
 for key in TARGET:assert pa[key]==ancestor[key]
 for key in set(pa)-TARGET:assert pa[key]==ch[key]
 native={','.join(map(str,key)):native_gate(ch[key]) for key in sorted(TARGET)}
 regs=registers((HERE/'device-resource.log').read_text());assert all(regs[k]==168 for k in TARGET)
 for key in TARGET:
  before,after=ops(pa[key]),ops(ch[key])
  for opcode in set(before)|set(after):
   if opcode.startswith(('HGMMA','WARPGROUP','SYNCS','LDS','UTMA','UBLKCP','DEPBAR','STSM','STG','ATOM','RED','TRAP','BAR','FENCE','USETMAXREG')):assert before[opcode]==after[opcode],(key,opcode)
  assert after['SEL']==before['SEL']-64 and after['BSSY']==before['BSSY']-4 and after['BSYNC']==before['BSYNC']-4
 assert not re.search(r'\b[1-9]\d* bytes (?:stack frame|spill stores|spill loads)',(HERE/'device-resource.log').read_text())
 negative=[]
 for label,source in [('wrong_selector',new.replace('setp.eq.and.u32 pr, ri, 7, pl;','setp.eq.and.u32 pr, ri, 6, pl;',1)),('wrong_input_word',new.replace('xor.b32 %7, %7, mask;','xor.b32 %7, %6, mask;',1)),('wrong_half','8 + ((nx >> 7) & 1) * 8'.join(new.rsplit('8 + ((nx >> 7) & 1) * 16',1))),('swapped_constraints',new.replace('"r"(lane), "r"(target)','"r"(target), "r"(lane)',1))]:
  try:inline_gate(source)
  except AssertionError:negative.append(label)
  else:raise AssertionError('negative passed '+label)
 altered=list(ch[(32,6,1,0,0,0,35)]);i=next(i for i,(pc,ins) in enumerate(altered) if re.match(r'@P\d+ LOP3.LUT',ins) and ', 0x3c,' in ins);pc,ins=altered[i];altered[i]=(pc,re.sub(r'LOP3.LUT R\d+,','LOP3.LUT R199,',ins,1))
 try:native_gate(altered)
 except AssertionError:negative.append('wrong_native_XOR_destination')
 else:raise AssertionError('native negative passed')
 return {'state':'CPU_TRUTH_SOURCE_NATIVE_REGISTER_GATES_PASS_GPU_UNRUN','parent_source_sha256':PARENT_SHA,'new_source_sha256':sha(HERE/'submission.py'),'changed_top_level_AST':changed,'exact_inverse_to_PW45':True,'ancestor_PW38_device_body_and_two_parent_PC_streams_exact':True,'unrelated_exact_PC_streams':19,'inline_truth':truth(inline_gate(new)),'actual_compiled_predicate_to_HGMMA_A_register_gate':native,'negative_controls':negative,'registers':168,'rounded_uniform_pool':168*384,'dynamic_pool':232*256+40*128,'threads':384,'source_sync_codec_payload_output_counter_unchanged':True,'GPU_contexts':0,'actual_GPU_occupancy':None,'measured_speed_us':None}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
