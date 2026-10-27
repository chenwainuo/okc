"""Exact byte-address model; sector sets are not measured GPU transactions."""
from pathlib import Path
import json,hashlib,re
D=Path(__file__).resolve().parent;R=D.parents[2]
SRC=R/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'
SASS=R/'proj_rsi/audits/gv53-quarter-sparse-result-v1/actual.sass'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=SASS.read_text();symbol='_Z10gv_predxorILi1ELi1ELi1ELi2ELi16ELi8ELi1ELi1ELi0ELi3EEv1G'
chunk=next(c for c in s.split('Function : ')[1:] if c.splitlines()[0].strip()==symbol)
rows=[(int(m[1],16),m[2].strip())for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',chunk)]
by_pc=dict(rows)
Wpc=[0x3c0,0x3d0,0x3e0,0x770,0x830,0x870]
assert all('LDG.E.NA.128.CONSTANT' in by_pc[p]for p in Wpc)
def accesses(base,layout):
 return [[set(range(base+(plane*512+lane*16 if layout=='planar'else lane*48+plane*16),base+(plane*512+lane*16 if layout=='planar'else lane*48+plane*16)+16))for lane in range(32)]for plane in range(3)]
def account(base,layout):
 a=accesses(base,layout);planes=[set.union(*p)for p in a];sectors=[{b//32 for b in p}for p in planes]
 assert all(len(p)==512 for p in planes)
 assert len(set.union(*planes))==1536 and sum(map(len,planes))==1536
 return {'bytes':1536,'unique_32byte_sector_set_size':len(set.union(*sectors)),'sector_set_size_per_warp_load':list(map(len,sectors)),'sector_instruction_incidences':sum(map(len,sectors)),'payload_warp_LDG128_instructions':3,'note':'Sector address incidences only, not measured L1 requests/replays/L2/DRAM transactions.'}
models={str(b):{layout:account(b,layout)for layout in ['planar','AoS48']}for b in [0,16]}
# Exact permutation: same three uint4s per logical lane, codec/HMMA operands untouched.
planar=bytes((i*19+7)%251 for i in range(1536));aos=bytearray(1536)
for lane in range(32):
 for plane in range(3):aos[lane*48+plane*16:lane*48+(plane+1)*16]=planar[plane*512+lane*16:plane*512+(lane+1)*16]
restored=bytearray(1536)
for lane in range(32):
 for plane in range(3):restored[plane*512+lane*16:plane*512+(lane+1)*16]=aos[lane*48+plane*16:lane*48+(plane+1)*16]
assert restored==planar
# Cooperative flat reading of the AoS buffer restores minimum3 coalesced loads,
# but each destination needs its original three vectors from distributed registers.
mapping=[{'destination_lane':l,'plane':i,'load_round':(3*l+i)//32,'source_lane':(3*l+i)%32}for l in range(32)for i in range(3)]
remote=[x for x in mapping if x['source_lane']!=x['destination_lane']]
assert len(remote)==92
per_lane=[sum(x['destination_lane']==l for x in remote)*4 for l in range(32)]
assert max(per_lane)==12
# Source-selected grid covers each rowtile/Kchunk once: no W reuse is discarded.
coverage={(bx*2+wr,wk*8+u)for bx in range(128)for wr in range(2)for wk in range(16)for u in range(8)}
assert len(coverage)==256*128
out={'state':'CPU_ADDRESS_ACCOUNTING_COMPLETE_NO_LAYOUT_COMPILE_JUSTIFIED_YET','target':{'source':str(SRC.relative_to(R)),'source_sha256':sha(SRC),'kernel':symbol,'m':8,'N':4096,'K':8192,'grid':[1,128,1],'block':1024,'WK':16,'WR':2,'KU':8,'ST':1,'codec_shape':'[N/16,K/64,3,32,16] uint8; p.W cast uint4','codec_bits_per_value':12},'source_native':{'SASS_path':str(SASS.relative_to(R)),'SASS_sha256':sha(SASS),'payload_load_PCs':[{'pc':hex(p),'instruction':by_pc[p]}for p in Wpc],'prologue_plane_offsets_bytes':[0,512,1024],'dynamic_per_warp':'3prologue+7iterations*3prefetch=24 payloadLDG128 perKU8; finalprefetch predicatedoff','separate_X_PCs':['0x370','0x3a0','0x8a0','0x8b0'],'escape_header_scalar_PC':'0x1e0'},'address_models_by_base_mod32':models,'whole_payload':{'W_scalar_values':4096*8192,'compressed_bytes':256*128*1536,'unique_rowtile_chunk_pairs':len(coverage),'unique_payload_vector_addresses':len(coverage)*96,'payload_warp_LDG128_instructions':128*32*8*3,'32byte_sector_incidences_aligned32':128*32*8*48,'alignment_limit':'ActualWbase pointer not logged here. Relative offsets are exact; showboth0/16mod32 consistentwith16Bvectoralignment. No hiddenalignmentassumption or hardwaretransactionclaim.'},'concrete_alternative':{'name':'Per-lane48B AoS repack, directvectorread or cooperativeflatread','permutation_inverse_exact':True,'direct_read':'Three16Breads perlane at48*lane+16*plane. Same1536unique bytes; aligned32 sector-incidences96 instead48. Repackingthisway is worse in this addressmodel.','cooperative_flat_read':'Threecontiguous512Bwarp loads retain48alignedsector-incidences, but destination(lane,plane) mustgather vectorindex3*lane+plane fromround=floor(index/32), sourcelane=index%32.','cross_lane_vectors':len(remote),'cross_lane_scalar_words':len(remote)*4,'minimum_scalar_SHFL_instructions_given_original_operand_ownership':max(per_lane),'shuffle_bound_limit':'Atleastone destinationrequires12remote32bitwords, eachscalarSHFLcanprovideone. Inputround multiplexing mayrequire moreSHFL/selects;12is lowerbound, notcompiledcount.','existing_planar_payload_gather_SHFL':0,'mapping':mapping},'minimum':'Underexisting16B/lanevectorloadprimitive,1536B/(32lanes*16B)=3fullwarpLDG128 minimum. Existingplanarlayout already meetsbyte/sector/load-instruction lowerbounds; unchangedcodec/HMMA offers no payload-layout saving in this model.','not_established':['ActualL1 request/replay behavior or cache miss merging.','Actualbasepointer alignment and runtimebandwidth/issue bottleneck.','Load-address dependencies or header/X/escape traffic criticalpath.','Transportfamily-wide impossibility; dynamicprofile may expose a different cause.'],'smallest_falsifier':'CPUaddress/layout inverse and exactexistingSASS already falsify the proposed12-byte-stride/redundant-plane-load premise. Do notcompileanAoS/shufflevariant solelyforpayloadcoalescing. If dynamicprofile contradicts minimumsector expectations, firstbindper-PC W requests and actualbasealignment; thenconstruct exactlyonechangedload-stage prototype for thatcause.','GPU_API_SSH_queue_git_actions':0,'contract':'proj_rsi/ITERATION.md'}
p=D/'accounting.json';s=json.dumps(out,indent=2)+'\n';assert not p.exists()or p.read_text()==s;p.write_text(s);print(json.dumps({k:v for k,v in out.items()if k in ['state','address_models_by_base_mod32','whole_payload','minimum']},indent=2))
