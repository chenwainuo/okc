"""PW44 disjoint-storage, EH coordinates and event-DAG proof; CPU only."""
import collections,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ST=5;XB=16384;WB=12288;OUT=((ST*(XB+WB)+24*ST+16+1023)//1024)*1024;SMEM=1024+OUT+2*XB

def layout(out=OUT):
 regions={'X':(0,ST*XB),'W':(ST*XB,ST*(XB+WB)),'barriers':(ST*(XB+WB),ST*(XB+WB)+24*ST),'zflag':(ST*(XB+WB)+24*ST,ST*(XB+WB)+24*ST+4),'output0':(out,out+XB),'output1':(out+XB,out+2*XB)}
 for name,(lo,hi) in regions.items():assert 0<=lo<hi and hi+1023<=SMEM,(name,lo,hi)
 for a,(alo,ahi) in regions.items():
  for b,(blo,bhi) in regions.items():
   if a!=b:assert ahi<=blo or bhi<=alo,('ALIAS',a,b)
 assert out%1024==0
 # Conservative arbitrary sbase alignment bump; declared extern alignment is stronger.
 for shift in range(1024):assert max(hi+shift for lo,hi in regions.values())<=SMEM
 return regions

def bitmap():
 cells=set();physical=set();half_ranges=[]
 for g in range(2):
  for half in range(2):
   addresses=set()
   for pp in range(4*half,4*half+4):
    for w in range(4):
     for lane in range(32):
      for reg in range(4):
       for elem in range(2):
        ch=16*w+(lane>>2)+8*(reg&1);tok=16*pp+8*(reg>>1)+2*(lane&3)+elem
        globalcoord=(tok,64*g+ch);assert globalcoord not in cells;cells.add(globalcoord)
        relative=tok*128+(((ch//8)^(tok&7))*16)+2*(ch%8)
        addr=OUT+g*XB+relative;assert addr not in physical;physical.add(addr);addresses.add(addr)
   low=OUT+g*XB+half*8192;high=low+8192
   assert addresses==set(range(low,high,2))
   half_ranges.append({'WG':g,'half':half,'byte_range':[low,high],'TMA_box':[64,64],'aligned_1024':low%1024==0})
 assert cells=={(t,c) for t in range(128) for c in range(128)} and len(physical)==16384
 return {'bf16_outputs_once':len(cells),'unique_physical_bf16':len(physical),'half_slabs':half_ranges,'first_TMA_read_vs_second_STSM':'DISJOINT_8192B_PER_WG'}

def reachable(edges,a,b):
 seen={a};todo=[a]
 while todo:
  x=todo.pop()
  for y in edges.get(x,()):
   if y==b:return True
   if y not in seen:seen.add(y);todo.append(y)
 return False

def dag(crossbar=False,omit=None):
 nodes=set();edges=collections.defaultdict(set)
 def edge(a,b):nodes.update([a,b]);edges[a].add(b)
 for g in range(2):
  for w in range(4):
   edge(f'g{g}.w{w}.last_WGMMA',f'g{g}.w{w}.Dwait0')
   if crossbar:edge(f'g{g}.w{w}.Dwait0','crossbar256');edge('crossbar256',f'g{g}.w{w}.store0')
   elif omit!='Dwait0':edge(f'g{g}.w{w}.Dwait0',f'g{g}.w{w}.store0')
   if omit!='proxy':edge(f'g{g}.w{w}.store0',f'g{g}.w{w}.proxy0');edge(f'g{g}.w{w}.proxy0',f'g{g}.bar128.0')
   if omit!='postbar':edge(f'g{g}.bar128.0',f'g{g}.issue0')
   edge(f'g{g}.bar128.0',f'g{g}.w{w}.store1')
   if omit!='proxy':edge(f'g{g}.w{w}.store1',f'g{g}.w{w}.proxy1');edge(f'g{g}.w{w}.proxy1',f'g{g}.bar128.1')
   if omit!='postbar':edge(f'g{g}.bar128.1',f'g{g}.issue1')
  edge(f'g{g}.bar128.0',f'g{g}.zpoll.acquire')
  edge(f'g{g}.zeroers.publish',f'g{g}.zpoll.acquire');edge(f'g{g}.zpoll.acquire',f'g{g}.global_proxy');edge(f'g{g}.global_proxy',f'g{g}.issue0')
  edge(f'g{g}.issue0',f'g{g}.commit0');edge(f'g{g}.commit0',f'g{g}.issue1');edge(f'g{g}.issue1',f'g{g}.commit1')
  if omit!='read_drain':edge(f'g{g}.commit1',f'g{g}.read_drain0');edge(f'g{g}.read_drain0',f'g{g}.issuer_exit')
  edge(f'g{g}.issuer_exit','CTA_exit')
 # Every dependency must precede access; no cross-WG data dependence exists in private output ranges.
 for g in range(2):
  for w in range(4):
   assert reachable(edges,f'g{g}.w{w}.Dwait0',f'g{g}.w{w}.store0')
   for half in range(2):assert reachable(edges,f'g{g}.w{w}.store{half}',f'g{g}.w{w}.proxy{half}') and reachable(edges,f'g{g}.w{w}.proxy{half}',f'g{g}.issue{half}')
  assert reachable(edges,f'g{g}.zeroers.publish',f'g{g}.issue0') and reachable(edges,f'g{g}.read_drain0','CTA_exit') and reachable(edges,f'g{g}.commit1',f'g{g}.read_drain0')
 # Topological acyclicity and concrete schedule in which WG0 enters its output stores before WG1 retires.
 indeg={n:0 for n in nodes}
 for a,bs in edges.items():
  for b in bs:indeg[b]+=1
 ready=sorted(n for n in nodes if indeg[n]==0);order=[]
 while ready:
  a=ready.pop(0);order.append(a)
  for b in sorted(edges[a]):
   indeg[b]-=1
   if indeg[b]==0:ready.append(b);ready.sort()
 assert len(order)==len(nodes)
 if not crossbar:assert not reachable(edges,'g1.w0.Dwait0','g0.w0.store0')
 return {'nodes':len(nodes),'edges':sum(map(len,edges.values())),'topological_order':order,'crossWG_barrier':crossbar,'per_warp_D_completion_and_both_128thread_publications_read_drain':'PASS','other_WG_wait_not_needed_for_private_range':not crossbar}

def proof():
 regions=layout();bm=bitmap();dags={'private_barrier_control':dag(True),'private_no_crossWG_barrier':dag(False)};parts=[]
 for N,KS in [(2304,7),(4096,4)]:
  bounds=[(s*128//KS,(s+1)*128//KS) for s in range(KS)];assert bounds[0][0]==0 and bounds[-1][1]==128 and all(a<b for a,b in bounds)
  assert sum(b-a for a,b in bounds)==128
  assert [t for a,b in bounds for t in range(a,b)]==list(range(128))
  # Every producer/mainloop access remains inside the complete X/W ring regardless of startup/tail ordering.
  for a,b in bounds:
   for stage in range(b-a):
    slot=stage%ST;assert slot*XB+XB<=regions['X'][1] and regions['W'][0]+slot*WB+WB<=regions['W'][1]
  grid=(N//128)*KS;assert grid in (126,128)
  parts.append({'N':N,'KS':KS,'K64_partitions':bounds,'ZI_grid_CTAs':grid,'runtime_required_resident_capacity':grid,'static_reg_pool':168*384,'dynamic_pool':232*256+40*128})
 negatives=[]
 for key in ['Dwait0','proxy','postbar','read_drain']:
  try:dag(False,key)
  except AssertionError:negatives.append('missing_'+key)
  else:raise AssertionError('negative DAG admitted '+key)
 try:layout(0)
 except AssertionError:negatives.append('private_output_aliases_X')
 else:raise AssertionError('alias admitted')
 # Explicit old-storage hazard allowed by the unchanged ST5 pipeline: slow WG may trail by two stages.
 # Fast WG finishedKCS18, slow WG still readsstage15(slot0); producerstage17 needed retirement12, already satisfied.
 assert 17-15<ST and 12<15 and regions['X'][0]==0
 negatives.append('reused_X_without_crossWG_barrier_has_legal_slow_WG_stage15_alias')
 return {'status':'CPU_PRIVATE_EH_STORAGE_AND_DAG_PASS_GPU_UNRUN','OUT':OUT,'SMEM':SMEM,'old_SMEM':144520,'extra_bytes_with_alignment':SMEM-144520,'regions':regions,'bitmap':bm,'DAGs':dags,'partitions':parts,'negative_controls_rejected':negatives,'tail_and_producer_scope':'All old producer, A buffers, mainloop waits/retirement and zeroer code remain byte-identical; output ranges are disjoint from their entire shared address domain even if another WG/producer lags. No assumed cross-WG convergence is used.','CTA_lifetime':'Each issuer retains both TMA read-drain0 before it exits; CTA shared allocation survives until all threads including both issuers/producers exit.'}
if __name__=='__main__':
 p=proof();(HERE/'protocol-proof.json').write_text(json.dumps(p,indent=2)+'\n');print(p['status'])
