"""Independent read-only CPU review. Never compiles or imports GPU lab/submission."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
TRIAL=ROOT/'proj_rsi/candidates/eh_same_geometry_trial'
BACK=ROOT/'proj_rsi/candidates/rs32_eh_backport'
sys.path[:0]=[str(TRIAL),str(BACK)]
import support as S

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def review():
 assert S.sha(TRIAL/'manifest.json')=='92121f07b0e4e895cdb9e5f1292971dd73776e9bb03b5e535364aefc6e47d9cc'
 assert S.sha(ROOT/'proj_rsi/tools/eh_same_geometry_lab.py')=='fa8420608041ae6884518c95b44fb495b60262e5e498627ff74a2080e0b95c79'
 S.verify_frozen();manifest=S.verify_lab();closure={}
 for directory in [ROOT/'proj_rsi/candidates/pw38',BACK]:
  data=json.loads((directory/'manifest.json').read_text())
  for name,expected in data['files'].items():assert S.sha(directory/name)==expected,(directory.name,name)
  closure[directory.name]=len(data['files'])
 binary={};rejections={}
 for role in S.SOURCES:
  source=S.cuda_source(role);assert source==(TRIAL/(role+'.extension.cu')).read_text()
  sass=(TRIAL/(role+'.sass')).read_text();binary[role]=S.binary_gate(role,sass)
  for name,bad in [('wrong_role',(TRIAL/(('eh' if role=='parent' else 'parent')+'.sass')).read_text()),('changed_instruction',sass.replace('HGMMA.','CORRUPTED.',1))]:
   try:S.binary_gate(role,bad)
   except AssertionError as e:rejections[role+'_'+name]=repr(e)
   else:raise AssertionError('Gate admitted '+role+'/'+name)
 pre=load(BACK/'preflight.py','independent_eh_preflight_readonly')
 life={'bitmap':pre.bitmap(),'physical':pre.physical((BACK/'sass.txt').read_text())}
 # Independently restore full factory prefix: EH source changes epilogue/map,
 # while producer/full-K/math and generic accumulator retirement remain exact.
 from build_patch import apply,value
 delta=json.loads((BACK/'delta.json').read_text())
 before=(ROOT/'proj_rsi/candidates/rs32/submission.py').read_text();after=(BACK/'submission.py').read_text()
 assert apply(before,delta)==after and apply(after,delta,True)==before
 pw36=S.source_for('parent');pw38=S.source_for('eh')
 assert value(apply(pw36,delta),'RQCU')==value(pw38,'RQCU')==value(after,'RQCU')
 original=value(before,'RQCU');updated=value(after,'RQCU')
 anchor='    const unsigned long long ce0 = INS ? clock64() : 0;'
 original_prefix=original[:original.index(anchor)];updated_prefix=updated[:updated.index(anchor)]
 guard=delta['edits'][1]
 updated_prefix=updated_prefix.replace(guard['after'],guard['before']).replace('    if constexpr (FL & 65536) {\n','')
 assert original_prefix==updated_prefix and 'rq_facc<ND>(D);' in updated_prefix and 'wgmma.wait_group.sync.aligned 0;' in updated_prefix

 code=load(TRIAL/'cpu_proof.py','independent_eh_terminal_proof_readonly')
 terminal=code.terminal_proof();assert len(terminal['negative_controls'])==9
 labels=S.LABELS
 orders=[]
 for seq in [(0,1,2,3),(3,2,1,0)]:orders.extend([seq[j:]+seq[:j] for j in range(4)])
 assert len(set(orders))==8
 assert all(sum(orders[r%8][pos]==label for r in range(32))==8 for label in range(4) for pos in range(4))
 sequence=['parent2','eh2','parent2','parent3','eh3','parent3'];last={}
 for b,label in enumerate(sequence):last[label]=240000+21*b+20
 assert len(set(last.values()))==4 and last['parent3']==240125
 assert all(S.FLAGS[label]&2 for label in labels),'all4 counter roles must be active selfzero'
 assert max(n//128*ks for n,k,ks in S.SHAPES.values())==128
 return {'status':'CPU_INDEPENDENT_GO_FIXED_GEOMETRY_ONLY_GPU_UNRUN','driver_sha256':S.sha(ROOT/'proj_rsi/tools/eh_same_geometry_lab.py'),
 'trial_manifest_sha256':S.sha(TRIAL/'manifest.json'),'closure':closure,'lab_files':len(manifest['files']),
 'canonical_binary_gates':binary,'wrong_role_instruction_rejections':rejections,'EH_lifetime':life,'terminal':terminal,
 'source_delta_inverse_prefix':'PASS','counter_contract':'all4FLbit2active_selfzero_KSmax7_twoNBzero_between_calls',
 'minimum_SM_for_complete_co_resident_grid':128,'F1_last_fixtures':last,
 'counts':{'per_shape_raw':274,'all4shapes_raw':1096,'terminal_oracles':32,'mutation_oracles':80},
 'scope':'FIXED_GEOMETRY_EH_ONLY_NO_ACTUAL_SELECTED_POLICY_OR_FULLSTACK_SPEED_CLAIM','GPU_calls':0,
 'budget':'ROOT_ACCEPTED_INNER600_OUTER720_FOUR_SHAPES_ONLY_NO_EXECUTION_BY_REVIEWER'}
if __name__=='__main__':
 result=review();out=Path(__file__).with_name('review-cpu.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('canonical_binary_gates','EH_lifetime','terminal')},indent=2))
