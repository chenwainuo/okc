"""Bounded local CPU reference-fixture RQ12/RQ11 accounting; never CUDA or benchmark.
Run with external timeout and single-thread environment; --seconds also bounds loops.
CPU reference RNG is explicitly NOT CUDA reference RNG for the same seed.
"""
import argparse,ast,collections,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RS32=ROOT/'proj_rsi/candidates/rs32/submission.py'
REFERENCE=ROOT/'muse_bf16_proj/reference.py'
OUT=ROOT/'proj_rsi/candidates/rq11_actualescape'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fixture_source(torch):
 tree=ast.parse(REFERENCE.read_text());nodes=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('LAYERS','WEIGHTS_SEED','FP32_LAYERS') for t in node.targets):nodes.append(node)
  if isinstance(node,ast.FunctionDef) and node.name=='generate_weights':nodes.append(node)
 ns={'torch':torch,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REFERENCE),'exec'),ns)
 return ns

def stock_cpu_functions(torch):
 tree=ast.parse(RS32.read_text());ns={'torch':torch}
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for t in node.targets:
    if isinstance(t,ast.Name) and t.id in ('_OS','_SS'):ns[t.id]=ast.literal_eval(node.value)
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in ('_pack_nat','_c12','_unc12'):
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(RS32),'exec'),ns)
 return ns

def choose_window(hist,width):
 import numpy as np
 scores=np.convolve(np.asarray(hist,dtype=np.int64),np.ones(width,dtype=np.int64),'valid')
 return int(scores.argmax()),scores

def fragments(bits):
 # Exact pinned _pack_nat transform, not an arbitrary flat chunk grouping.
 n,k=bits.shape
 return bits.reshape(n//16,2,8,k//64,4,2,4,2).transpose(0,3,4,2,6,5,1,7).reshape(n//16,k//64,4,32,8)

def encode_decode(fr,base,width,os12,ss12):
 import numpy as np
 t,kc=fr.shape[:2];unitbytes=1536 if width==8 else 1408
 hi=((fr>>8)&127).astype(np.uint32);sign=(fr>>15).astype(np.uint32);bad=(hi<base)|(hi>=base+width)
 offset=np.where(bad,0,hi-base).astype(np.uint32);lo=(fr&255).astype(np.uint8)
 payload=np.empty((t,kc,unitbytes),np.uint8)
 payload[...,:512]=lo[:,:,:2].transpose(0,1,3,2,4).reshape(t,kc,512)
 payload[...,512:1024]=lo[:,:,2:].transpose(0,1,3,2,4).reshape(t,kc,512)
 if width==8:
  codes=(np.left_shift(offset,np.asarray(os12,dtype=np.uint32))|np.left_shift(sign,np.asarray(ss12,dtype=np.uint32))).sum(-1,dtype=np.uint64).astype(np.uint32)
  heads=codes.transpose(0,1,3,2).copy().astype('<u4');payload[...,1024:]=heads.view(np.uint8).reshape(t,kc,512)
  c=payload[...,1024:].copy().view('<u4').reshape(t,kc,32,4).transpose(0,1,3,2)
  off=np.right_shift(c[...,None],np.asarray(os12,dtype=np.uint32))&7
  sg=np.right_shift(c[...,None],np.asarray(ss12,dtype=np.uint32))&1
 else:
  code=offset|(sign<<2);heads=np.zeros((t,kc,3,32),np.uint32)
  for q in range(4):
   for v in range(8):
    bit=(q*8+v)*3;word,shift=divmod(bit,32);c=code[:,:,q,:,v]
    heads[:,:,word,:]|=c<<shift
    if shift>29:heads[:,:,word+1,:]|=c>>(32-shift)
  payload[...,1024:]=heads.astype('<u4').view(np.uint8).reshape(t,kc,384)
  heads=payload[...,1024:].copy().view('<u4').reshape(t,kc,3,32)
  code=np.empty((t,kc,4,32,8),np.uint32)
  for q in range(4):
   for v in range(8):
    bit=(q*8+v)*3;word,shift=divmod(bit,32);c=heads[:,:,word,:]>>shift
    if shift>29:c|=heads[:,:,word+1,:]<<(32-shift)
    code[:,:,q,:,v]=c&7
  off=code&3;sg=code>>2
 low=np.empty(fr.shape,np.uint16)
 low[:,:,:2]=payload[...,:512].reshape(t,kc,32,2,8).transpose(0,1,3,2,4)
 low[:,:,2:]=payload[...,512:1024].reshape(t,kc,32,2,8).transpose(0,1,3,2,4)
 decoded=low|(((base+off)|(sg<<7)).astype(np.uint16)<<8)
 # Apply actual XOR7 escapes at their exact fragment coordinates.
 coordinates=np.nonzero(bad);xor=(hi[coordinates]^base).astype(np.uint16)
 assert bool(np.all(xor>0)) and bool(np.all(xor<128))
 decoded[coordinates]^=xor<<8
 assert np.array_equal(decoded,fr),'actual weight storage roundtrip mismatch'
 return payload,bad,coordinates,xor

def entry_proof(coords,xors,KC,KS):
 import numpy as np
 tile,ch,q,lane,v=coords;boundaries=np.arange(KS+1,dtype=np.int64)*KC//KS
 split=np.searchsorted(boundaries,ch,side='right')-1;seq=ch-boundaries[split]
 assert bool(np.all(seq<=32766))
 entry=(seq.astype(np.uint32)<<17)|(lane.astype(np.uint32)<<12)|((q*4+v//2).astype(np.uint32)<<8)|((v%2).astype(np.uint32)<<7)|xors.astype(np.uint32)
 assert bool(np.all(entry!=0xffffffff))
 assert np.array_equal(entry>>17,seq) and np.array_equal((entry>>12)&31,lane)
 assert np.array_equal((entry>>8)&15,q*4+v//2) and np.array_equal((entry>>7)&1,v%2) and np.array_equal(entry&127,xors)
 return len(entry),np.bincount(split*8+tile,minlength=KS*8).reshape(KS,8)

def allocation(counts,KS,bits):
 import numpy as np
 t,kc=counts.shape;assert t%8==0;nb=t//8
 boundaries=np.arange(KS+1,dtype=np.int64)*kc//KS
 assert bool(np.all(np.diff(boundaries)>0))
 physical=counts.reshape(nb,8,kc)
 groups=np.stack([physical[:,:,boundaries[s]:boundaries[s+1]].sum(2,dtype=np.int64) for s in range(KS)])
 g=KS*nb*8;maximum=int(groups.max());cap=((maximum+1+31)//32)*32
 assert cap>maximum and cap%32==0
 actual=int(groups.sum());fixed_words=g*cap;pad=fixed_words-actual
 payload=t*kc*(1536 if bits==12 else 1408)
 esc=4*(fixed_words+32)
 return {'KS':KS,'NB':nb,'KC':kc,'groups':g,'split_chunk_boundaries':boundaries.tolist(),'max_list':maximum,'CAP_entries':cap,'group_lengths':groups.reshape(-1).tolist(),
  'actual_escape_entries':actual,'actual_escape_bytes':actual*4,'fixed_stride_unused_words_including_per_group_sentinels':pad,'fixed_stride_unused_bytes':pad*4,'trailing_guard_words':32,'trailing_guard_bytes':128,
  'escape_allocation_bytes':esc,'payload_bytes':payload,'low_bytes':t*kc*1024,'selector_header_bytes':t*kc*(512 if bits==12 else 384),'selector_header_padding_bits':0,'stage_alignment_padding_bytes':0,
  'device_codec_total_bytes':payload+esc,'escape_sink_bytes':0,'discarded_rqesc_dummy_tensor_bytes':4,'stock_style_rq_pk_cached_escape_tuple_bytes':48*actual,'device_pack_plus_escape_cache_bytes':payload+esc+48*actual,'cache_retention_assumption':'RQ12 matches pinned rq_pk six-int64 esc6 cache; RQ11 conditional on reusing that prepare-cache representation; no RQ11 GPU integration exists','common_counter_bytes_full_grid':8*nb,'common_counter_bytes_half_grid':16*nb,'decode_table_logical_bytes':4,'decode_table_host_ABI_argument_bytes':8,'decode_table_device_allocation_bytes':0,
  'full128_stage_bytes':12288 if bits==12 else 11264,'HALF64_stage_bytes':6144 if bits==12 else 5632,
  'max_prefetch_windows_per_list':(maximum+31)//32,'capacity_bounds_and_sentinel_guard':'PASS_EXACT_NO_TRUNCATION'}

def unit_tests():
 import numpy as np
 hist=np.zeros(128,dtype=np.int64);hist[20:28]=1
 assert choose_window(hist,8)[0]==20 and choose_window(hist,4)[0]==20
 for n in (0,1,31,32,33,63,64,1024):
  counts=np.zeros((8,2),dtype=np.int64);counts[0,0]=n;a=allocation(counts,1,11)
  assert a['CAP_entries']>n and a['CAP_entries']%32==0 and a['actual_escape_entries']==n
  assert a['device_codec_total_bytes']==a['payload_bytes']+a['actual_escape_bytes']+a['fixed_stride_unused_bytes']+128
 return {'window_tie_uses_first_argmax':True,'CAP_boundary_controls':8,'full_cost_identity':True}

def run(args):
 os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
 import numpy as np
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert not torch.cuda.is_initialized()
 clock=time.monotonic();deadline=clock+args.seconds;OUT.mkdir(exist_ok=True)
 result={'status':'RUNNING','fixture_kind':'EXACT_REFERENCE_FUNCTION_CPU_DEVICE_NOT_CUDA_RNG','GPU_fixture_bit_identity':'UNKNOWN_NOT_CLAIMED','seed':5151,'torch_version':torch.__version__,'numpy_version':np.__version__,'CPU_threads':1,'seconds_limit':args.seconds,'sources':{str(p.relative_to(ROOT)):sha(p) for p in (REFERENCE,RS32,ROOT/'proj_rsi/candidates/rq11_codec/codec.py',Path(__file__))},'tests':unit_tests(),'layers':{},'errors':[]}
 def save():args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 def bounded():
  if time.monotonic()>deadline:raise TimeoutError('internal CPU wall bound reached')
 try:
  fixture=fixture_source(torch);assert fixture['WEIGHTS_SEED']==5151
  weights=fixture['generate_weights'](fixture['WEIGHTS_SEED'],device='cpu');assert all(not w.is_cuda for w in weights.values())
  result['generation_seconds']=time.monotonic()-clock;bounded();stock=stock_cpu_functions(torch)
  spec=importlib.util.spec_from_file_location('frozen_rq11_codec',ROOT/'proj_rsi/candidates/rq11_codec/codec.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
  for layer,w in weights.items():
   bounded();raw=w.view(torch.int16).numpy().view(np.uint16) if w.dtype==torch.bfloat16 else w.numpy().view(np.uint32)
   if layer=='router':
    result['router']={'dtype':'float32','shape':list(w.shape),'sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'excluded_from_BF16_codec':True,'reason':'Do not truncate FP32 router into BF16 representation'};continue
   n,k=w.shape;t=n//16;kc=k//64;h=(raw>>8)&127;hist=np.bincount(h.reshape(-1).astype(np.int64),minlength=128);finite=bool(torch.isfinite(w).all());assert finite
   row={'shape':[n,k],'dtype':'bfloat16','values':n*k,'source_weight_bytes':n*k*2,'sha256':hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest(),'h7_histogram':hist.tolist(),'all_finite':finite,'formats':{}}
   states={}
   for bits,width in ((12,8),(11,4)):
    b,scores=choose_window(hist,width);badcount=int(hist.sum()-scores[b]);states[bits]={'base':b,'width':width,'counts':np.zeros((t,kc),np.int64),'unit_hist':np.zeros(1025,np.int64),'max_lane_hist':np.zeros(33,np.int64),'walk_entries_checked':0,'payload_sha':hashlib.sha256(),'coordinate_group_counts':{ks:np.zeros((ks,n//128,8),np.int64) for ks in (1,2,4,6,7,8)}}
    row['formats'][str(bits)]={'window_width':width,'base':b,'window_candidate_coverage_counts':scores.tolist(),'escape_count':badcount,'escape_fraction':badcount/(n*k),'raw_payload_bits_per_weight':bits}
   for start in range(0,n,128):
    bounded();fr=fragments(raw[start:start+128]);assert fr.shape==(8,kc,4,32,8)
    if start==0:
     sample=torch.from_numpy(raw[:16,:64].copy()).view(torch.bfloat16)
     nat=stock['_pack_nat'](sample).view(torch.int16).numpy().view(np.uint16);assert np.array_equal(nat[0,0],fr[0,0])
     stockpacked,tab,_=stock['_c12'](stock['_pack_nat'](sample),1)
     check,_,_,_=encode_decode(fr[:1,:1],int(tab)&255,8,stock['_OS'][1],stock['_SS'][1]);assert np.array_equal(check.reshape(-1),stockpacked.numpy().reshape(-1))
    for bits,state in states.items():
     payload,bad,coords,xor=encode_decode(fr,state['base'],state['width'],stock['_OS'][1],stock['_SS'][1]);state['payload_sha'].update(payload.transpose(1,0,2).copy().tobytes())
     count=bad.sum((2,3,4),dtype=np.int64);state['counts'][start//16:start//16+8]=count
     state['unit_hist']+=np.bincount(count.reshape(-1),minlength=1025)
     lane=bad.sum((2,4),dtype=np.int64).max(2);state['max_lane_hist']+=np.bincount(lane.reshape(-1),minlength=33)
     for ks in (1,2,4,6,7,8):
      entry_count,coordinate_counts=entry_proof(coords,xor,kc,ks);state['walk_entries_checked']+=entry_count;state['coordinate_group_counts'][ks][:,start//128,:]=coordinate_counts
     if bits==11 and start==0:
      oldp,olde=codec.pack(fr[0,0],state['base']);assert np.array_equal(oldp,payload[0,0]) and np.array_equal(codec.unpack(oldp,state['base'],olde),fr[0,0])
   for bits,state in states.items():
    f=row['formats'][str(bits)];assert int(state['counts'].sum())==f['escape_count']
    for ks in (1,2,4,6,7,8):
     independent=allocation(state['counts'],ks,bits);assert np.array_equal(state['coordinate_group_counts'][ks].reshape(-1),independent['group_lengths'])
    f.update(independent_escape_coordinate_group_accounting='PASS_ALL_KS',payload_SHA256=state['payload_sha'].hexdigest(),whole_matrix_bitexact_roundtrip='PASS',exact_stock_layout_or_frozen11_match='PASS_REAL_FIRST_UNIT',entries_field_roundtrips_across_6_KS=state['walk_entries_checked'],escapes_per_warp_unit_histogram=state['unit_hist'].tolist(),max_lane_escapes_per_unit_histogram=state['max_lane_hist'].tolist(),geometry=[allocation(state['counts'],ks,bits) for ks in (1,2,4,6,7,8)])
   row['comparison']=[]
   for old,new in zip(row['formats']['12']['geometry'],row['formats']['11']['geometry']):
    a,b=old['device_codec_total_bytes'],new['device_codec_total_bytes'];row['comparison'].append({'KS':old['KS'],'RQ12_total_bytes':a,'RQ11_total_bytes':b,'net_bytes_saved':a-b,'fraction_saved':(a-b)/a,'RQ12_CAP':old['CAP_entries'],'RQ11_CAP':new['CAP_entries'],'RQ12_max_list':old['max_list'],'RQ11_max_list':new['max_list'],'RQ12_with_esc6_cache_bytes':old['device_pack_plus_escape_cache_bytes'],'RQ11_with_esc6_cache_bytes':new['device_pack_plus_escape_cache_bytes'],'net_residency_bytes_saved_with_stock_cache':old['device_pack_plus_escape_cache_bytes']-new['device_pack_plus_escape_cache_bytes']})
   result['layers'][layer]=row;save();print('COMPLETE',layer,'weights',n*k,'windows',{bits:f['base'] for bits,f in row['formats'].items()},flush=True)
  result['status']='PASS_CPU_REFERENCE_FIXTURE_ACCOUNTING_GPU_DISTRIBUTION_UNKNOWN'
 except Exception as error:
  result['status']='INCOMPLETE_NO_GPU_OR_SPEED_CLAIM';result['errors'].append(repr(error));raise
 finally:
  result['elapsed_seconds']=time.monotonic()-clock;result['CUDA_initialized_after']=torch.cuda.is_initialized();assert not result['CUDA_initialized_after'];save()
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=180);p.add_argument('--output',type=Path,default=OUT/'actual-cpu-reference.json');args=p.parse_args();assert 1<=args.seconds<=300;run(args)
