"""Exercise identities, all canonical binary gates, and adversarial timed-output gate CPU only."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reverse_tma_x_stmatrix'))
import native_x_stmatrix_support as support
import timed_output_gate_proof

def binary_negatives():
    import itertools
    negatives={}
    canonical={label:(support.CANDIDATES/folder/'sass.txt').read_text() for label,folder in support.FOLDERS.items()}
    for expected,actual in itertools.permutations(canonical,2):
        key='wrong_label_'+expected+'_receives_'+actual
        try:support.actual_binary_gate(expected,canonical[actual])
        except AssertionError as error:negatives[key]=str(error)
        else:raise AssertionError('admitted '+key)
    for label,sass in canonical.items():
        # Preserve function identity while changing an actual instruction operand.
        assert 'HGMMA.64x64x16.F32.BF16 R24,' in sass
        tamper=sass.replace('HGMMA.64x64x16.F32.BF16 R24,','HGMMA.64x64x16.F32.BF16 R25,',1)
        try:support.actual_binary_gate(label,tamper)
        except AssertionError as error:negatives['tampered_operand_'+label]=str(error)
        else:raise AssertionError('admitted tamper '+label)
    assert len(negatives)==9
    return negatives


def schedule_proof():
    import ast
    from types import SimpleNamespace
    import statistics
    text=(HERE.parents[1]/'tools/native_x_stmatrix_lab.py').read_text()
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure')
    class Buffer:
        seed=None
        def copy_(self,value):self.seed=value
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
        def elapsed_time(self,other):return .001
    torch=SimpleNamespace(empty=lambda *a,**kw:Buffer(),sum=lambda *a,**kw:None,float32=None,
                          cuda=SimpleNamespace(Event=Event,synchronize=lambda:None))
    ns={'torch':torch,'statistics':statistics}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'native_x_stmatrix_lab.py','exec'),ns)
    labels=('incumbent','ds','native','new');x=Buffer();observed={};boundaries=[]
    class Graph:
        def __init__(self,label):self.label=label
        def replay(self):observed[self.label]=x.seed
    fixtures={s:s for s in [239999]+list(range(240000,240147))+list(range(250000,250032))}
    def strict(fixtures,seeds,final):
        assert observed==seeds and x.seed==final['input_seed']
        boundaries.append(dict(seeds));return dict(own_fixture_outputs_checked=4)
    result=ns['measure']({label:Graph(label) for label in labels},x,fixtures,strict)
    calls=result['all_timed_and_warmup_calls']
    counts={mode:sum(c['mode']==mode for c in calls) for mode in ('warmup','F1','paired')}
    assert counts=={'warmup':20,'F1':147,'paired':128}
    assert len(boundaries)==2 and all(set(b)==set(labels) for b in boundaries)
    assert len(set(boundaries[0].values()))==4
    paired=[c for c in calls if c['mode']=='paired']
    for position in range(4):
        assert all(sum(paired[4*r+position]['label']==label for r in range(32))==8 for label in labels)
    return dict(status='PASS_EXTRACTED_MEASURE_CPU_FAKE_EVENTS',call_counts=counts,
                terminal_outputs=8,F1_distinct_own_fixtures=True,paired_each_arm_each_position=8,
                no_GPU_context=True,limits='Exercises exact measurement control flow with CPU fake events; no timing evidence')

def proof():
    identities=support.verify_frozen()
    modules={label:support.load(folder,'prototype',label) for label,folder in support.FOLDERS.items()}
    for label,mod in modules.items():
        assert mod.source()==(support.CANDIDATES/support.FOLDERS[label]/'extension_source.cu').read_text()
    result={'status':'CPU_LAB_GATES_PASS_GPU_UNRUN','identities':identities,
            'canonical_binary_gates':{label:support.actual_binary_gate(label,(support.CANDIDATES/folder/'sass.txt').read_text()) for label,folder in support.FOLDERS.items()},
            'timed_output_adversarial_gate':timed_output_gate_proof.proof(),
            'canonical_wrong_label_and_tamper_negatives':binary_negatives(),
            'measurement_schedule':schedule_proof(),
            'GPU':'UNRUN_NO_CONTEXT_NO_EXTENSION_NO_PREPARE',
            'raw_native_X_bit_mover':'UNPROBED_NOT_AN_XP_PACK_GATE',
            'limits':'CPU proofs bind frozen sources and local cubin; actual JIT and all CUDA runtime gates remain required'}
    (HERE/'cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('canonical_binary_gates','timed_output_adversarial_gate')},indent=2))
if __name__=='__main__':proof()
