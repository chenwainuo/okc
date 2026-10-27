import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=json.loads((D/'delta.json').read_text());old=(R/d['parent_path']).read_text();new=(D/'submission.py').read_text();assert hashlib.sha256(old.encode()).hexdigest()==d['parent_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert new.replace(d['new_GVCU_literal'],d['old_GVCU_literal'],1).replace(d['new_prefix'],d['old_prefix'],1)==old
 changed=[getattr(a,'name',None)or ast.unparse(a.targets)for a,b in zip(ast.parse(old).body,ast.parse(new).body)if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False)];assert changed==['GVCU','_gvext']
 def remove_scope(k):
  a=k.index('  static_assert(');b=k.index('  // pw14 probes only',a);return k[:a]+k[b:]
 restored=d['new_kernel'].replace(' gv_unpool16_deltakey(G p) {',' gv_predxor(G p) {',1).replace(d['new_repair'],d['old_repair'],1);assert remove_scope(restored)==remove_scope(d['old_kernel'])
 helper=(D/'helper.cuh').read_text();parsed=[(int(a),int(b),int(c))for a,b,c in re.findall(r'setp.eq.u32 pr, k, (\d+); @pr xor.b32 %(\d+), %(\d+), mask;',helper)];assert parsed==[(4*i,i,i)for i in range(16)]
 constants=','.join(str(a)for a,b,c in parsed)
 cpp='''#include <cstdint>\n#include <cstdio>\nint main(){const uint32_t C[]={'''+constants+'''};uint64_t cases=0,chains=0;for(uint32_t target=0;target<32;target++)for(uint32_t lane=0;lane<32;lane++)for(uint32_t r=0;r<16;r++)for(uint32_t hm=0;hm<2;hm++){const uint32_t nx=(target<<15)|(r<<11)|(hm<<9);const uint32_t key=((nx>>9)&0x7fdu)-(lane<<6);for(int i=0;i<16;i++){bool ref=lane==target&&hm==0&&r==(uint32_t)i;if(ref!=(key==C[i]))return 1;cases++;}}for(int n:{0,1,31,32})for(uint32_t seed=0;seed<64;seed++){uint32_t a[32][16],b[32][16];for(int l=0;l<32;l++)for(int i=0;i<16;i++)a[l][i]=b[l][i]=0x9e3779b9u*(1+l*16+i);for(int j=0;j<n;j++){uint32_t nx=(((j+seed)%32)<<15)|(((j/3+seed)%16)<<11)|((j%2)<<10)|(((j+seed)%3==0)<<9)|((j*13+seed)%128);uint32_t xr=(nx&127u)<<(8+((nx>>10)&1)*16);for(uint32_t lane=0;lane<32;lane++){uint32_t key=((nx>>9)&0x7fdu)-(lane<<6);for(int i=0;i<16;i++){if(lane==((nx>>15)&31)&&((nx>>9)&1)==0&&i==((nx>>11)&15))a[lane][i]^=xr;if(key==C[i])b[lane][i]^=xr;}}}for(int l=0;l<32;l++)for(int i=0;i<16;i++)if(a[l][i]!=b[l][i])return 2;chains++;}printf("%llu %llu\\n",(unsigned long long)cases,(unsigned long long)chains);}\n'''
 cpp=cpp.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>');(D/'semantic.cpp').write_text(cpp);subprocess.run(['g++','-std=c++17','-O2',str(D/'semantic.cpp'),'-o',str(D/'semantic')],check=True);cases,chains=map(int,subprocess.check_output([str(D/'semantic')],text=True).split());assert cases==524288 and chains==256
 text=(D/'pilot.sass').read_text();resource=(D/'pilot.resources').read_text();summaries={}
 for symbol,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S):
  rows=re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',body);ops=collections.Counter(re.sub(r'^@!?[UP]*P\d+\s+','',x[1].strip()).split()[0]for x in rows);resblock=resource.split('Function '+symbol+':')[1].split('Function ')[0];res={k:int(v)for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',resblock)}
  summaries['new'if'gv_unpool16'in symbol else'control']={'symbol':symbol,'instructions':len(rows),'registers':res['REG'],'stack_B':res['STACK'],'local_B':res['LOCAL'],'P2R':ops['P2R'],'SHFL':ops['SHFL.IDX'],'HMMA':ops['HMMA.16816.F32.BF16'],'LDG':sum(v for k,v in ops.items()if k.startswith('LDG'))}
 assert len(summaries)==2 and summaries['new']['stack_B']==summaries['new']['local_B']==0
 assert all(summaries['new'][k]==summaries['control'][k]for k in('P2R','SHFL','HMMA','LDG'))
 out={'state':'CPU_FOCUSED_SOURCE_SEMANTIC_TWO_SYMBOL_CODEGEN_PASS_GPU_UNRUN','source_sha256':sha(D/'submission.py'),'changed_AST_nodes':changed,'all_other174_parent_nodes_exact':True,'whole_file_inverse':True,'copied_kernel_inverse_except_name_scope_repair_decode':True,'unchanged_cursor_shuffle_load_HMMA_reduction_output_source':True,'predicate_domain_cases':cases,'boundary_chains_0_1_31_32':chains,'ignored_half1_and_wronglanes_exact':True,'unsigned_delta_domain':'eligible iff target==lane,hm==0,index==r; offlane integer difference lies outside0..60 modulo2^32','key_mask_excludes_payload_halfshift_chunk_bits':True,'native':summaries,'performance_claim':False,'excluded_pooler_spill_version':'proj_rsi/candidates/gv16_deltakey_v1','GPU_calls':0}
 (D/'proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
