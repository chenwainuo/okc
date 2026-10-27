"""CPU-only full old-code identity and narrowly qualified new-core diagnostics."""
import collections,importlib.util,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pw44_offline_checker',HERE/'cpu_proof.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
def proof():
 sass=(HERE/'sass.txt').read_text();actual,old=C.streams(sass),C.streams((HERE/'control-sass.txt').read_text())
 for k,v in old.items():assert actual[k]==v
 def resource_pages(text):return {p.splitlines()[0].strip():p.splitlines()[1].strip() for p in text.split(' Function ')[1:]}
 ra,rb=resource_pages((HERE/'device.resources').read_text()),resource_pages((HERE/'control.resources').read_text());assert all(ra[k]==v for k,v in rb.items())
 private=next(t for t in C.tables(sass) if 'Li851971E' in t.splitlines()[0]);neg=[]
 for name,changed in [('missing_shared_proxy',private.replace('FENCE.VIEW.ASYNC.S','NO_PROXY')),('missing_poststore_bar',re.sub(r'BAR.SYNC.DEFER_BLOCKING R3, 0x80','NO_BAR',private)),('missing_read_drain',re.sub(r'(?<![A-Z.])DEPBAR.LE','NO_DRAIN',private))]:
  try:C.physical(sass.replace(private,changed,1))
  except (AssertionError,ValueError):neg.append(name)
  else:raise AssertionError('physical negative accepted '+name)
 core={}
 for fl in [327682,327683,851970,851971]:
  symbol=next(k for k in ra if 'Li'+str(fl)+'EE' in k);assert 'SHARED:1024' in ra[symbol] and 'STACK:0' in ra[symbol] and 'LOCAL:0' in ra[symbol]
  a=actual[(128,5,1,0,0,0,fl)];b=actual[(128,5,1,0,0,0,65538+(fl&1))]
  ops=lambda rows:collections.Counter(i.split()[0] for _,i in rows if i.startswith(('HGMMA.','WARPGROUP.','LDSM.','LDS.','SYNCS.')))
  assert ops(a)==ops(b)
  end=max(pc for pc,i in a if 'WARPGROUP.DEPBAR.LE' in i and '0x0' in i)
  aa=[(pc,i) for pc,i in a if pc<=end];bb=[(pc,i) for pc,i in b if pc<=end];assert len(aa)==len(bb)
  reloc=[]
  for (pc,i),(qc,j) in zip(aa,bb):
   assert pc==qc
   if i==j:continue
   x=re.fullmatch(r'(.*\b(?:CALL.REL.NOINC|BRA)) (0x[0-9a-f]+)',i);y=re.fullmatch(r'(.*\b(?:CALL.REL.NOINC|BRA)) (0x[0-9a-f]+)',j)
   assert x and y and x[1]==y[1] and int(x[2],16)>end and int(y[2],16)>end,('unexplained new math-prefix drift',fl,pc,i,j)
   reloc.append({'PC':pc,'old':j,'new':i,'scope':'target outside pre-Dwait0 core prefix'})
  core[str(fl)]={'math_LDS_barrier_opcode_counts_exact':True,'pre_Dwait0_instructions_exact_except_listed_external_targets':True,'listed_external_control_targets':reloc,'new_smem_dynamic':178176,'compiler_static_shared':1024,'total_shared':179200,'regs':168,'local':0,'stack':0}
 return {'old19_resources_exact':True,'old19_full_PC_predicate_instruction_streams_exact_no_normalization':True,'new_core':core,'new_physical_negatives_rejected':neg,'limit':'The listed external-target diagnostic is only new-vs-old source-scope support; it is not whole-new-kernel canonical identity. New epilogue scheduling differs and is not isolated intrinsic barrier latency. Future actual JIT must exactly match all new canonical streams.'}
if __name__=='__main__':
 p=proof();(HERE/'codegen-proof.json').write_text(json.dumps(p,indent=2)+'\n');print('PW44_CODEGEN_IDENTITY_AND_PHYSICAL_NEGATIVES_PASS')
