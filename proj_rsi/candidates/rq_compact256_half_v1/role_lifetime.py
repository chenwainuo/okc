"""Finite source-role proof, not a CUDA scheduling or numerical execution test."""
from pathlib import Path
import json,ast,hashlib
D=Path(__file__).resolve().parent
def val(t,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(t).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id==name)
src=(D/'submission.py').read_text();cu=val(src,'RQCU')
assert 'if (warp >= 8)' in cu and cu.index('if (warp >= 8)')<cu.index('if (SELF || (HALF && warp >= 4)) return;')
assert 'rq_expect(full + 8 * s, WB)' in cu and 'rq_expect(full + 8 * s, XB)' in cu
assert '"r"(NW)' in cu and 'bar.sync 4, 64' in cu
assert 'const int t = tid - 320' in cu
physical=lambda tid:tid if tid<128 else tid-128
logical=lambda tid:tid if tid<128 else tid+128
oldactive=list(range(128))+list(range(256,384));newactive=list(range(256))
assert [logical(p)for p in newactive]==oldactive
def role(l):
 w=l//32
 return 'math' if w<4 else 'W' if w==8 else 'X' if w==9 else 'zero'
assert all(role(logical(p))==role(l)for p,l in zip(newactive,oldactive))
assert physical(256)==128 and physical(288)==160 and physical(320)==192
# Math el/lane/shared-decode coordinates identical for both halves/ring slots.
for hh in range(2):
 for s in range(6):
  oldcoords=[(4*hh+(t//32&3),t&31,s*6144+(t//32)*1536+(t&31)*16)for t in range(128)]
  newcoords=[(4*hh+(logical(t)//32&3),logical(t)&31,s*6144+(logical(t)//32)*1536+(logical(t)&31)*16)for t in range(128)]
  assert oldcoords==newcoords
# Each CTA zeroes its exact16x64 region, with8-BF16 vector stores, once.
allcells=set()
for ks in range(2):
 for hh in range(2):
  stores=[]
  for p in range(192,256):
   t=logical(p)-320
   for e in range(t,16*8,64):
    stores.extend((ks*16+e//8,hh*64+(e%8)*8+j)for j in range(8))
  expected={(r,c)for r in range(ks*16,(ks+1)*16)for c in range(hh*64,(hh+1)*64)}
  assert len(stores)==len(set(stores))==1024 and set(stores)==expected
  assert not allcells&set(stores);allcells.update(stores)
assert len(allcells)==32*128
# PerWG register transfer is exact and requires no otherCTA's released registers.
assert (216-128)*128==(128-40)*128==11264
assert 128*216+128*40==256*128==32768
out={'state':'CPU_ROLE_AND_ADDRESS_MAPPING_PASS_RUNTIME_UNRUN','source_sha256':hashlib.sha256(src.encode()).hexdigest(),'roles_equal_256_threads':True,'physical_EARLY_W_X_prefetch_zero_roles':[128,128,160,192,[192,255]],'producer_branch_before_half_math_return':True,'math_escape_and_LDS_coordinates_equal_both_halves_six_slots':True,'zero_unique_cells_per128channel_block':len(allcells),'mbarrier_full_expected_arrivals':2,'Wempty_Xempty_math_warp_arrivals':4,'zero_named_barrier_threads':64,'CTA_initial_barrier_threads':256,'register_transfer_each_direction':11264,'shared_request':62624,'limits':['Initialbarrierparticipantset is256ratherthan384; absentoldHALFmathWG1 did no ring/math/epilogue work after earlyreturn.','No changedasyncordering/lifetime expressions; this finite proof does not establish CUDA co-residency, liveness, numerical outputs or latency.','KS2 halfgrid256CTA requires runtimeoccupancy and cooperativezero-handshake validity before any timing admitted.']}
(D/'role-lifetime-proof.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
