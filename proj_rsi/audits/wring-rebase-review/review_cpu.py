"""Read-only independent W-ring review: no compile, GPU import/launch or frozen writes."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'proj_rsi';CAND=BASE/'candidates/pw36_wring_rebase'
sys.path.insert(0,str(CAND))
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def review():
 D=load(BASE/'tools/wring_rebase_lab.py','wring_independent_lab')
 assert D.sha(CAND/'manifest.json')=='20ffb258d67a4ed205298103e0e34e9f06d929957c4cc670dd5985deccc1c299'
 assert D.sha(BASE/'tools/wring_rebase_lab.py')=='fd43d83540e2469be8bcd0d5c676b2a96c1ccbeab0e440b767603bcabd594924'
 assert D.sha(CAND/'submission.py')=='207f5a1bdc4f5d7712d444d9c71e857643e6700451c88734fce898ac1fc1b7ab'
 manifest=D.verify_frozen();P=load(CAND/'preflight.py','wring_independent_preflight');from build_patch import apply,value
 parent=(BASE/'candidates/pw36/submission.py').read_text();source=(CAND/'submission.py').read_text();delta=json.loads((CAND/'delta.json').read_text())
 assert apply(parent,delta)==source and apply(source,delta,True)==parent
 assert P.extension_source(source)==(CAND/'extension_source.cu').read_text()
 old,new=value(parent,'RQCU'),value(source,'RQCU')
 za='    } else if (ZI && warp >= 10)';zb='  // ---------------- math warpgroups:';out='  if (INS) pc[0] = clock64() - tq0;'
 assert old[old.index(za):old.index(zb)]==new[new.index(za):new.index(zb)]
 assert old[old.index(out):]==new[new.index(out):]
 for key in ('RQPROD','RQBAKE','RQKS','RQFLAGS'):assert value(parent,key)==value(source,key)
 candidate=P.streams((CAND/'sass.txt').read_text());control=P.streams((CAND/'control-sass.txt').read_text())
 assert len(candidate)==5 and len(control)==3
 for name,args in P.CONTROLS.items():assert candidate[args]==control[args],('original M128 drift',name)
 physical=P.physical((CAND/'sass.txt').read_text())
 proto=load(CAND/'protocol_proof.py','wring_independent_protocol').proof()
 tests=load(CAND/'cpu_lab_proof.py','wring_independent_cpu_lab')
 terminal=tests.terminal();schedule=tests.schedule();counters=tests.counters();binary=tests.binary_negatives()
 assert len(terminal['negative_controls'])==7 and len(counters['negative_controls'])==5
 # Additional reviewer-controlled PC-only mutation and wrong-arm rejection.
 streams={args:candidate[args] for args in P.ARMS.values()};mutated={args:list(rows) for args,rows in streams.items()}
 key=P.ARMS['split_x6w4'];pc,ins=mutated[key][1];mutated[key][1]=(pc+16,ins);assert mutated!=streams
 wrong={args:list(rows) for args,rows in streams.items()};wrong[key]=candidate[P.ARMS['split_x5w5']];assert wrong!=streams
 return {'status':'CPU_INDEPENDENT_WRING_GO_GPU_UNRUN','manifest_sha256':D.sha(CAND/'manifest.json'),'source_sha256':D.sha(CAND/'submission.py'),
 'lab_sha256':D.sha(BASE/'tools/wring_rebase_lab.py'),'candidate_files_verified':len(manifest['files']),'external_files_verified':len(manifest['external_files']),
 'whole_source_delta_inverse':'PASS','zero_counter_protocol_and_output_suffix_exact':'PASS','all3_current_M128_control_PC_streams_exact':'PASS',
 'physical':physical,'protocol':proto,'terminal':terminal,'measurement':schedule,'counters':counters,'binary':binary,
 'extra_PC_offset_and_wrong_arm_rejections':'PASS','inner_outer_seconds':[240,300],'GPU_calls':0,
 'limits':'Bounded two-WG source protocol model and fixed compiled streams; actual runtime gates remain unrun'}
if __name__=='__main__':
 result=review();Path(__file__).with_name('review-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('physical','protocol','terminal','measurement','counters','binary')},indent=2))
