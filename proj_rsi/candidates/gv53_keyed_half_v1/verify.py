"""Read-only exact-field/source/native check for one keyed-half change. CPU only."""
import ast,hashlib,importlib.util,json,random,re
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
spec=importlib.util.spec_from_file_location('keybuild',D/'build.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
def streams(s):
 return {sym:[(int(m[1],16),re.sub(r'\s+',' ',m[2].strip()),m[3],m[4])for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/\s*/\* (0x[0-9a-f]+) \*/',body)]for sym,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',s,re.S)}
def main():
 old=B.P.read_text();new=(D/'submission.py').read_text();assert B.sha(old)==B.PARENT_SHA
 cu=B.value(old);nc=B.value(new);helper,kernel,nh,nk,dispatch,at=B.parts(cu);added='\n'+nh+'\n'+nk+'\n'
 assert nc.replace(added,'',1).replace(dispatch,'',1)==cu
 restored=new.replace('load_inline("gv53_keyed_half_v1_%x"','load_inline("gvp45_predxor_v2_%x"',1).replace(nc,cu,1);assert restored==old
 aa,bb=ast.parse(old).body,ast.parse(new).body;assert len(aa)==len(bb)
 diffs=[i for i,(a,b)in enumerate(zip(aa,bb))if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False)];assert len(diffs)==2
 assert [getattr(aa[i],'name',None)or aa[i].targets[0].id for i in diffs]==['GVCU','_gvext']
 constants=list(map(int,re.findall(r'pr, ri, (\d+), pl;',nh)));assert constants==[i*4 for i in range(16)]
 assert '.reg .pred pl, pr;'in nh and '%20'not in nh and '"r"(hm)'not in nh
 first=nh.index('@pr xor.b32');assert all(nh.index(x)<first for x in ['mov.b32 ri, %16','mov.b32 mask, %17','setp.eq.u32 pl, %18, %19'])
 cases=0
 for r in range(16):
  for hm in range(2):
   for half in range(2):
    for lane in range(32):
     for target in range(32):
      word=(r<<11)|(hm<<9)|(half<<10)|(target<<15)|127
      key=(word>>9)&61;assert key==4*r+hm
      old_hits=[i for i in range(16)if lane==target and hm==0 and r==i]
      new_hits=[i for i,c in enumerate(constants)if lane==target and key==c]
      assert old_hits==new_hits;cases+=1
 rng=random.Random(3801);chains=0
 for length in [0,1,2,15,31,32]:
  for rep in range(24):
   words=sorted([(rng.randrange(8)<<20)|(rng.randrange(32)<<15)|(rng.randrange(16)<<11)|(rng.randrange(2)<<10)|(rng.randrange(2)<<9)|rng.randrange(128)for _ in range(length)],key=lambda v:v>>20)
   oldA=[[rng.getrandbits(32)for i in range(16)]for lane in range(32)];newA=[x[:]for x in oldA];ep=0;nx=words[0]if words else 0xffffffff
   for u in range(8):
    while nx>>20==u:
     r=(nx>>11)&15;hm=(nx>>9)&1;key=(nx>>9)&61;target=(nx>>15)&31;xr=(nx&127)<<(8+((nx>>10)&1)*16)
     for lane in range(32):
      for i in range(16):
       if lane==target and hm==0 and r==i:oldA[lane][i]^=xr
       if lane==target and key==4*i:newA[lane][i]^=xr
     ep+=1;nx=words[ep]if ep<length and ep<32 else 0xffffffff
   assert ep==length and oldA==newA;chains+=1
 actual=streams((D/'pilot.sass').read_text());archive=streams((R/'proj_rsi/candidates/gv_predxor_pw45_v2/candidate-full.sass').read_text())
 ctrl=next(k for k in actual if'gv_predxor'in k);cand=next(k for k in actual if'gv_keyed'in k);assert actual[ctrl]==archive[ctrl]
 def metrics(rows):
  ins=[r[1]for r in rows];return {'slots':len(ins),'nonNOP':sum(x!='NOP'for x in ins),**{op:sum(op in x for x in ins)for op in ['USHF','ULOP3','ISETP','P2R','SHFL','HMMA','LDG']}}
 a,b=metrics(actual[ctrl]),metrics(actual[cand]);assert a['nonNOP']-b['nonNOP']==3
 for op in ['P2R','SHFL','HMMA','LDG']:assert a[op]==b[op]
 res=(D/'pilot.resources').read_text();assert res.count('REG:64 STACK:0 SHARED:1024 LOCAL:0')==2
 return {'state':'CPU_FIELD_SOURCE_NATIVE_PASS_GPU_UNRUN','source_sha256':B.sha(new),'parent_sha256':B.PARENT_SHA,'source_inverse':True,'unchanged_AST_nodes':len(aa)-2,'field_cases':cases,'cursor_chains_0_through_32_entries':chains,'all_ignored_hm1_preserved':True,'no_33rd_read':True,'inputs_copied_before_first_XOR':True,'actual_control_PC_predicate_encoding_exact':True,'control_native':a,'candidate_native':b,'target_registers':64,'stack_local_spills':0,'source_dynamic_shared_bytes':17408,'extra_static_p1zero_helper':'Unchanged unused zeroing helper also emitted; only two target repair functions','GPU_calls':0,'performance_measured':False}
if __name__=='__main__':print(json.dumps(main(),indent=2))
