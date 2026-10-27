"""Exact scored-selected type only: parent/new RQ source in private namespaces."""
from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
def cuda(src):
 tree=ast.parse(src);env={}
 for n in tree.body:
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='RQCU'for t in n.targets):env['RQCU']=ast.literal_eval(n.value)
  if isinstance(n,ast.FunctionDef)and n.name in['_rqwg','_rqss','_rqsrc']:exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact-source-generator>','exec'),env)
 return env['_rqsrc']([('scored_selected',32,6,1,0,0,0,35)])
headers=[];bodies=[]
for ns,path in[('rq_control',ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py'),('rq_offset',ROOT/'proj_rsi/candidates/rq_mma_descriptor_offset_inc501d_v2/submission.py')]:
 cu=cuda(path.read_text());body=[]
 for line in cu.splitlines():
  if line.startswith('#include'):
   if line not in headers:headers.append(line)
  else:body.append(line)
 bodies.append('namespace '+ns+' {\n'+'\n'.join(body)+'\n}\n')
binding=r'''
template<class K> int64_t query(K k,int64_t f){
 auto rc=cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,99488);if(rc!=cudaSuccess)return -1000-int(rc);
 cudaFuncAttributes a;rc=cudaFuncGetAttributes(&a,k);if(rc!=cudaSuccess)return -2000-int(rc);
 if(f==1)return a.numRegs;if(f==2)return a.localSizeBytes;if(f==3)return a.sharedSizeBytes;if(f==4)return a.maxDynamicSharedSizeBytes;if(f==5)return a.maxThreadsPerBlock;
 if(f==6){int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,384,99488);return rc==cudaSuccess?n:-3000-int(rc);}return -9;
}
int64_t attrs(int64_t arm,int64_t f){if(arm==0)return query(rq_control::rq<32,6,1,0,0,0,35>,f);if(arm==1)return query(rq_offset::rq<32,6,1,0,0,0,35>,f);return -90;}
int64_t run(int64_t arm,torch::Tensor W,torch::Tensor x,torch::Tensor out,torch::Tensor esc,torch::Tensor cnt,int64_t KS,int64_t Tb,torch::Tensor prof,torch::Tensor sink){
 TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.size(0)==32,"M32 CUDA input required");
 TORCH_CHECK((out.size(1)==4096&&x.size(1)==8192&&KS==4)||(out.size(1)==8192&&x.size(1)==4096&&KS==2),"only scored selected pooler/unpooler cells");
 if(arm==0)return rq_control::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);
 if(arm==1)return rq_offset::rqrun(0,W,x,out,esc,cnt,KS,Tb,0,prof,sink,0);return -90;
}
'''
(D/'binding.cu').write_text(binding);(D/'trial.cu').write_text('\n'.join(headers)+'\n'+''.join(bodies)+binding)
# Extract only original lossless packing helpers; do not import submission/prepare.
names={'_pack_nat','_c12','_unc12','_frag_bits','_rqpack','_rqunpack','_rqesc'};nodes=[]
for n in ast.parse((ROOT/'proj_rsi/candidates/gv_predxor_pw45_v2/submission.py').read_text()).body:
 if isinstance(n,ast.FunctionDef)and n.name in names:nodes.append(n)
 elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in['_OS','_SS']for t in n.targets):nodes.append(n)
assert len(nodes)==9
(D/'codec.py').write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n')
job=json.loads((ROOT/'results/proj-rsi/jobs/9916d34357b745cf.json').read_text());selected=[]
for p in job['profiles']:
 if p['case']['m']==32 and p['case']['layer']in['pooler','unpooler']:
  assert 'void rq<32, 6, 1, 0, 0, 0, 35>'in p['report'];selected.append(p)
assert len(selected)==2
(D/'scored-selected.json').write_text(json.dumps({'source_sha256':'501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9','job':'9916d34357b745cf','profiles':selected,'scope':'Scored-selected geometry fixed-cell contrast. No fresh prepare or selector/policy win claimed.'},indent=2)+'\n')
print(hashlib.sha256((D/'trial.cu').read_bytes()).hexdigest())
