"""Weight-only full-exponent codebook/capacity statistics. No JIT or model prepare."""
import argparse,gc,hashlib,importlib.util,json,math,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def choose(hist):
 assert len(hist)==256 and all(isinstance(x,int)and x>=0 for x in hist)
 fullbase=max(range(249),key=lambda b:sum(hist[b:b+8]))
 high7=[hist[2*i]+hist[2*i+1]for i in range(128)]
 aligned=2*max(range(125),key=lambda b:sum(high7[b:b+4]))
 top=sorted(range(256),key=lambda i:(-hist[i],i))[:8]
 return {'aligned_C11_range':list(range(aligned,aligned+8)),
         'best_contiguous_exp8':list(range(fullbase,fullbase+8)),
         'global_top8_exp8':top}

def summarize_groups(torch,counts,weight_count,table):
 groups=counts.numel();total=int(counts.sum());hist=torch.bincount(counts.flatten().long()).tolist()
 return {'exponent_table':table,'repairs':total,'repair_fraction':total/weight_count,
  'group_count':groups,'repair_count_histogram':hist,'max_group_repairs':int(counts.max()),
  'mean_group_repairs':total/groups,'groups_over32':int((counts>32).sum()),
  'groups_over64_optimistic_no_metadata':int((counts>64).sum()),
  'necessary_total_budget32':groups*32,'all_groups_fit_existing32_records':bool((counts<=32).all()),
  'prospective_11bit_fixed_payload_bytes':weight_count*11//8,
  'existing128B_group_bytes':groups*128,'global_table_bytes':8,
  'prospective_total_fixed_bytes_if_representable':weight_count*11//8+groups*128+8,
  'byte_scope':'Static candidate budget only, not an implemented packed stream or measured traffic; any overflow keeps original fallback, never truncates.'}

def analyze(torch,helper,ns,w,geo):
 cpu=w.cpu();fr=ns['_pack'](cpu).reshape(cpu.shape[0]//16,cpu.shape[1]//64,4,32,8)
 bits=fr.view(torch.int16).int()&65535;e8=(bits>>7)&255
 hist=torch.bincount(e8.flatten().long(),minlength=256).tolist();tables=choose(hist)
 result={'shape':list(cpu.shape),'raw_weight_sha256':hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
  'fragment_sha256':hashlib.sha256(fr.view(torch.uint8).numpy().tobytes()).hexdigest(),
  'geometry':list(geo),'full_exp8_histogram':hist,'nonfinite_exponent_count':hist[255],
  'packing':'Exact original _pack; group_counts from reviewed production-GV stats driver','formats':{}}
 for name,table in tables.items():
  lookup=torch.zeros(256,dtype=torch.bool);lookup[table]=True
  bad=~lookup[e8.long()];units=bad.sum((2,3,4),dtype=torch.int64)
  counts=helper.group_counts(units,geo)
  result['formats'][name]=summarize_groups(torch,counts,bits.numel(),table)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args()
 assert sha(a.review)==a.review_sha256
 manifest=json.loads((D/'manifest.json').read_text())
 for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 started=time.monotonic();r={'state':'RUNNING_WEIGHT_ONLY_STATS','seeds':manifest['seeds'],
  'scope':'Only reference CUDA W generation, CPU exact production-pack histograms/capacity. No activation/output, JIT, model prepare, native codec or speed claim.',
  'manifest_sha256':sha(D/'manifest.json'),'review_sha256':a.review_sha256,'records':[]};torch=None;weights=None
 def save():a.output.write_text(json.dumps(r,indent=2)+'\n')
 def bounded():
  if time.monotonic()-started>75:raise TimeoutError('75s internal stats bound')
 save()
 try:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  spec=importlib.util.spec_from_file_location('top8_original_stats_helper',ROOT/manifest['helper']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
  ns={'torch':torch,'math':math}
  exec(compile(helper.nodes(ROOT/manifest['reference'],{'LAYERS','FP32_LAYERS','WEIGHTS_SEED'},{'generate_weights'}),'<exact reference generator>','exec'),ns)
  exec(compile(helper.nodes(ROOT/manifest['source'],set(),{'_pack'}),'<exact production pack>','exec'),ns)
  r['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'generation_order':list(ns['LAYERS'])};save()
  for seed in manifest['seeds']:
   bounded();weights=ns['generate_weights'](seed,device='cuda');torch.cuda.synchronize()
   for layer,geo in manifest['geometries'].items():
    bounded();row=analyze(torch,helper,ns,weights[layer],tuple(geo));row.update(seed=seed,layer=layer)
    if seed==5151:row['matches_old_fixture_sha256']=row['raw_weight_sha256']==manifest['known_seed5151_weight_sha256'][layer]
    r['records'].append(row);r['elapsed_seconds']=time.monotonic()-started;save()
   weights=None;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
  bounded();r['state']='ACTUAL_CUDA_WEIGHT_ONLY_TOP8_STATS_COMPLETE'
 except BaseException:r.update(state='INVALID_WEIGHT_STATS_SETUP_OR_RUNTIME',error=traceback.format_exc());save();raise
 finally:
  weights=None;gc.collect()
  if torch is not None and torch.cuda.is_initialized():
   try:
    torch.cuda.synchronize();torch.cuda.empty_cache();r['cleanup']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
    if r['cleanup']['allocated']!=0:r['state']='INVALID_STATS_CLEANUP'
   except BaseException:r['state']='INVALID_STATS_CLEANUP';r['cleanup_error']=traceback.format_exc()
  r['elapsed_seconds']=time.monotonic()-started;save()
if __name__=='__main__':main()
