"""Small source/dispatch/composition checks only; no torch/compiler/GPU import."""
import ast,hashlib,json,re,types,importlib.util
from pathlib import Path
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
TRACE='results/proj-rsi/scheduler/artifacts/research.gv.direct.policy.01/collection-1793124217631420611/logs/scheduler/gv-direct-policy-01/run/pw45.gvpred_lab_pw45_6c567ec89ba176b11d32.original.cu'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def edits(s,rows,reverse=False):
 for e in reversed(rows)if reverse else rows:
  a,b=(e['after'],e['before'])if reverse else(e['before'],e['after']);assert s.count(a)==1,e['name'];s=s.replace(a,b,1)
 return s
def get(s,name):return next(n for n in ast.parse(s).body if getattr(n,'name',None)==name or isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
def check():
 delta=json.loads((D/'delta.json').read_text());old=(ROOT/delta['parent_path']).read_text();new=(D/'submission.py').read_text();assert sha(old)==delta['parent_sha256']=='501d901db8eeb0a7547191fbd1e2df2e7c3132b31ac974355dd5bf811d232ab9'
 assert edits(old,delta['edits'])==new and edits(new,delta['edits'],True)==old
 a=ast.parse(old).body;b=[n for n in ast.parse(new).body if getattr(n,'name',None)!='_gvpred_confirm'];assert len(a)==len(b)==176
 changed=[getattr(x,'name',None)or ast.unparse(x.targets)for x,y in zip(a,b)if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False)]
 assert changed==['GVCU','_gvext','_gvcall','prepare'],changed
 oldcu=ast.literal_eval(get(old,'GVCU').value);newcu=ast.literal_eval(get(new,'GVCU').value);assert newcu.replace('int64_t gvpredrun(', 'int64_t gvrun(',1).startswith(oldcu)
 tuples=list(dict.fromkeys(tuple(map(int,s.split(',')))for s in re.findall(r'return gvl<([0-9,]+)>',(ROOT/TRACE).read_text())))
 assert len(tuples)==95
 captured={};env={'GVCU':newcu,'GVI':tuples,'_gvmod':None,'load_inline':lambda *a,**kw:captured.update(kw)or object()}
 exec(compile(ast.Module(body=[get(new,'_gvext')],type_ignores=[]),'<actual-source-factory>','exec'),env);env['_gvext']()
 cu=captured['cuda_sources'];assert '@@'not in cu
 oldgo=cu[cu.index('static int64_t gv_keepoldgo('):cu.index('\nint64_t gvrun(',cu.index('static int64_t gv_keepoldgo('))]
 assert oldgo.count('return gv_keepoldlaunch<')==95 and 'return gvl<'not in oldgo
 assert 'gvrun'in captured['functions'] and 'gvpredrun'in captured['functions'] and 'gvpredrun'in captured['cpp_sources']
 routes=[]
 ext=types.SimpleNamespace(gvpredrun=lambda *a:routes.append('new')or 0,gvrun=lambda *a:routes.append('old')or 0)
 env={'GVI':tuples,'_gvext':lambda:ext};exec(compile(ast.Module(body=[get(new,'_gvcall')],type_ignores=[]),'<actual-call-route>','exec'),env)
 t=(1,1,2,1,8,16,1,1,0,1);p=(tuples.index(t),1,2,3,4096,4,5);st={'gvws':6,'gvcnt':7}
 env['_gvcall'](st,p,object(),object());env['_gvcall'](st,p+('gvpredxor',),object(),object());assert routes==['old','new']
 # Actual helper declines unrelated/fallback shapes and an exhausted original clock before any torch call.
 helper=get(new,'_gvpred_confirm');env={'_T0':0,'PREPARE_BUDGET_S':240,'time':types.SimpleNamespace(time=lambda:241),'GVI':tuples,'_bk':lambda x:x}
 exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual-confirm-skip>','exec'),env);env['_gvpred_confirm']({},None)
 env['time']=types.SimpleNamespace(time=lambda:1);state={'sel':{},'gv':{}};env['_gvpred_confirm'](state,None);assert state=={'sel':{},'gv':{}}
 assert any(isinstance(n,ast.Compare)and isinstance(n.left,ast.Name)and n.left.id=='effect'and isinstance(n.ops[0],ast.Gt)and isinstance(n.comparators[0],ast.Constant)and n.comparators[0].value==0 for n in ast.walk(helper))
 # Exact commutativity with RQ source edits and separately allowlisted .99 removal.
 rq=json.loads((ROOT/'proj_rsi/candidates/rq32_predxor/delta.json').read_text())['edits'];rq=[dict(e,name=e['id'])for e in rq]
 combined=(ROOT/'proj_rsi/candidates/rq32_gv_predxor_pw45_v1/submission.py').read_text();assert edits(new,rq)==edits(combined,delta['edits'])
 spec=importlib.util.spec_from_file_location('positive_margin_source_only',ROOT/'proj_rsi/candidates/combined_positive_selection_v1/verify.py');positive=importlib.util.module_from_spec(spec);spec.loader.exec_module(positive)
 positive_parent=(ROOT/'proj_rsi/candidates/combined_positive_selection_v1/submission.py').read_text()
 stacked=positive.transform(edits(new,rq));assert stacked==edits(positive_parent,delta['edits'])
 m=D/'manifest.json'
 if m.exists():
  for path,pin in json.loads(m.read_text())['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==pin,path
 return {'state':'CPU_SOURCE_KEEPCHOICE_PASS_GPU_UNRUN','source_sha256':sha(new),'parent_sha256':sha(old),'changed_parent_nodes':changed,'new_helper':'_gvpred_confirm','all_other_172_parent_nodes_exact':True,'strict_forward_inverse':True,'original_device_source_exact_only_original_gvrun_export_renamed':True,'default_gvrun_for_all_existing_direct_bake_callers_is_original':True,'actual_old_dispatch_cases':95,'actual_call_old_new_marker_routes':routes,'budget_and_unselected_skip_fixtures':True,'balanced_pairs':4,'minimum_gain_threshold_us':0,'RQ_source_patch_commutes':True,'positive_margin_patch_commutes':True,'ephemeral_fully_stacked_source_sha256':sha(stacked),'compile_or_GPU_validation':False}
if __name__=='__main__':print(json.dumps(check(),indent=2))
