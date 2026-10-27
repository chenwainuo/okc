import ast,hashlib,json,pathlib,re,subprocess,collections
D=pathlib.Path(__file__).resolve().parent
P=D.parent/'rs32/submission.py'; original=P.read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'
changes=[('  auto ldr = [&](int G) {  // full waits, LDS of this warp\'s 48 B per lane per stage, W slots released', '''  // smallm_increment v1: ldr is called once for each consecutive stage in WM0/GS1.
  constexpr bool LDS_CURSOR = (NT == 16 || NT == 32) && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && FL == 35;
  int ldslot = 0, ldphase = 0;
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released'''),
('        const int slot = RES ? 0 : s % ST;\n        if (!RES) rq_wait(full + 8 * slot, (s / ST) & 1);','''        const int slot = LDS_CURSOR ? ldslot : (RES ? 0 : s % ST);
        if (!RES) rq_wait(full + 8 * slot, LDS_CURSOR ? ldphase : ((s / ST) & 1));'''),
('for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(wempty + 8 * ((GS * G + j) % ST));', 'for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (LDS_CURSOR ? ldslot : ((GS * G + j) % ST)));'),
('    RQ_T1(2);\n  };\n  auto xrel', '''    RQ_T1(2);
    if (LDS_CURSOR && ++ldslot == ST) { ldslot = 0; ldphase ^= 1; }
  };
  auto xrel''')]
s=original
for old,new in changes:
 assert s.count(old)==1,old
 s=s.replace(old,new)
restored=s
for old,new in reversed(changes):
 assert restored.count(new)==1
 restored=restored.replace(new,old)
assert restored==original
(D/'submission.py').write_text(s)
(D/'delta.json').write_text(json.dumps(changes,indent=2)+'\n')
# Exhaustive sequential-loader equivalence over every supported KCS bound and rings (includes ragged splitK).
for st in (5,6):
 for kcs in range(1,129):
  slot=phase=0
  for G in range(kcs):
   assert (slot,phase)==(G%st,(G//st)&1)
   slot+=1
   if slot==st:slot=0;phase^=1
for label,src in [('before',original),('after',s)]:
 tree=ast.parse(src);env={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RQCU' for x in n.targets):env['RQCU']=ast.literal_eval(n.value)
  if isinstance(n,ast.FunctionDef) and n.name in ('_rqwg','_rqss'):exec(compile(ast.Module(body=[n],type_ignores=[]),str(P),'exec'),env)
 c=env['RQCU'].split('typedef CUresult')[0].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
 c=c.replace('@@WGMMA@@',''.join(env['_rqwg'](n) for n in (16,32,128,256))+''.join(env['_rqss'](n) for n in (128,256)))
 for nt,st in ((32,5),(32,6),(16,5)):c+=f'\ntemplate __global__ void rq<{nt},{st},1,0,0,0,35>(const RqP,__grid_constant__ const RqTmap,__grid_constant__ const RqTmap);\n'
 (D/f'{label}.cu').write_text(c)
 cmd=['/usr/local/cuda/bin/nvcc','-gencode=arch=compute_90a,code=sm_90a','-O3','-DNDEBUG','--expt-relaxed-constexpr','-cubin','-Xptxas=-v',str(D/f'{label}.cu'),'-o',str(D/f'{label}.cubin')]
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(D/f'{label}.log').write_text(r.stdout);assert r.returncode==0,r.stdout
 sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(D/f'{label}.cubin')],text=True);(D/f'{label}.sass').write_text(sass)
 summary={}
 for sect in sass.split('Function : ')[1:]:
  name=sect.splitlines()[0].strip();ops=[]
  for l in sect.splitlines():
   if not re.search(r'/\*[0-9a-f]+\*/',l):continue
   ins=l.split('*/',1)[1].strip()
   if ins.startswith('@'):ins=ins.split(None,1)[1]
   ops.append(ins.split()[0].rstrip(';'))
  summary[name]={'total':len(ops),'non_NOP':sum(x!='NOP' for x in ops),'ops':dict(collections.Counter(ops))}
 (D/f'{label}.counts.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(label,r.stdout)
print('Exact restoration and cursor invariants PASS. Candidate SHA256',hashlib.sha256(s.encode()).hexdigest())
