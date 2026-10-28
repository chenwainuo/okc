"""Exact task CUDA weights and original RQ packing; no native build or timing."""
from pathlib import Path
import argparse,ast,gc,hashlib,json,sys,time,traceback
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 m=json.loads((D/'manifest.json').read_text())
 for p,h in m['files'].items():assert sha(ROOT/p)==h,p
 return m
def summarize(lengths,cap):
 assert len(lengths)==1024 and cap>=32 and cap%32==0 and all(isinstance(v,int)and 0<=v<cap for v in lengths)
 hist={str(v):lengths.count(v)for v in sorted(set(lengths))};refills=sum(v//32 for v in lengths)
 return {'cap':cap,'slots':len(lengths),'lengths':lengths,'length_histogram':hist,'maximum_records_per_slot':max(lengths),'total_escape_records':sum(lengths),'slots_with_refills':sum(v>=32 for v in lengths),'required_subsequent_bank_loads':refills,'original_total_window_loads':1024+refills,'lookahead_mechanism_exposed':refills>0,'interpretation':'This counts subsequent32-entry window switches, not initialLDG/firstSHFL. Zeroexposure doesnot testlookahead benefit; keep family/proof andpark thisfixture/version.'}
def cpu_check():
 verify();assert summarize([0]*1024,32)['required_subsequent_bank_loads']==0
 a=[0]*1024;a[:4]=[31,32,33,64];s=summarize(a,96);assert s['required_subsequent_bank_loads']==4 and s['slots_with_refills']==3
 parent=ast.parse((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text());codec=ast.parse((ROOT/'proj_rsi/candidates/rq_descriptor_offset_trial_v1/codec.py').read_text());key=lambda n:n.name if isinstance(n,ast.FunctionDef)else n.targets[0].id
 original={key(n):ast.dump(n,include_attributes=False)for n in parent.body if isinstance(n,(ast.FunctionDef,ast.Assign))and(isinstance(n,ast.FunctionDef)or isinstance(n.targets[0],ast.Name))}
 assert all(original[key(n)]==ast.dump(n,include_attributes=False)for n in codec.body)
 return {'state':'CPU_EXPOSURE_STATS_PASS_RUNTIME_UNRUN','original_codec_AST_nodes':9,'summary_zero_and_boundary_cases':'PASS','JIT_or_native_RQ_calls':0,'CPU_GPU_proxy':False}
def work(torch):
 sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
 weights=reference.generate_weights(5151,device='cuda');w=weights['pooler'];assert tuple(w.shape)==(4096,8192)
 env={'torch':torch};exec(compile((ROOT/'proj_rsi/candidates/rq_descriptor_offset_trial_v1/codec.py').read_text(),'<exactRQcodec>','exec'),env)
 packed=env['_rqpack'](w);assert packed is not None
 lay,table,esc6=packed;esc,_=env['_rqesc'](esc6,32,128,4,w.device);cap=(esc.numel()-32)//1024
 bank=esc[:-32].view(1024,cap);lengths=(bank!=-1).sum(1).cpu().tolist();s=summarize(lengths,cap)
 assert sum(lengths)==len(esc6)and bool((esc[-32:]==-1).all())
 # Original generator/packing is the exact intended fixture, including all earlier layer RNG draws.
 raw=hashlib.sha256(w.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
 s.update(W_seed=5151,shape=[4096,8192],KS=4,KC=128,raw_weight_sha256=raw,known_prior_5151_weight_match=raw=='b56f76d32015a8552987feb0cee8a3ffe9de968f8d87604ad89c4325d1aa2b72',packed_bytes=lay.numel(),packed_sha256=hashlib.sha256(lay.cpu().numpy().tobytes()).hexdigest(),escape_words=esc.numel(),escape_sha256=hashlib.sha256(esc.cpu().numpy().tobytes()).hexdigest(),packing='Exact original _rqpack (its own _unc12 equality) and _rqesc AST; no native function')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--review-sha256');a=p.parse_args()
 if a.cpu_check:print(json.dumps(cpu_check(),indent=2));return
 verify();assert sha(a.review)==a.review_sha256 and not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();r={'state':'RUNNING_EXACT_ESCAPE_EXPOSURE','scope':'Only exacttask5151CUDA generation/originalpack/count; noactivation, output, nativeRQ/JIT/timing/profiler','review_sha256':a.review_sha256};torch=None
 def save():a.output.write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);r['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name()}
  r['exposure']=work(torch);torch.cuda.synchronize();assert time.monotonic()-start<25,'25sstatsbound';r['state']='VALID_EXACT_POOLER32_ESCAPE_WINDOW_EXPOSURE'
 except BaseException:r['state']='INVALID_EXPOSURE_SETUP_OR_RUNTIME';r['error']=traceback.format_exc();save();raise
 finally:
  gc.collect()
  if torch is not None and torch.cuda.is_initialized():
   torch.cuda.synchronize();torch.cuda.empty_cache();r['cleanup']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
   if r['cleanup']['allocated']!=0:r['state']='INVALID_EXPOSURE_CLEANUP'
  r['elapsed_seconds']=time.monotonic()-start;save()
if __name__=='__main__':main()
