"""Root-only actual serving comparison. --cpu-check never imports torch.
GPU entry requires independent review; no source/dispatch edits or forced picks.
"""
import argparse, ast, collections, hashlib, importlib.util, json, os
from pathlib import Path
import re, statistics, subprocess, sys, time, traceback
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'proj_rsi'
TRIAL = BASE / 'candidates/smallm_increment_trial'
PINS = {
 'pw36': ('pw36','2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf'),
 'rs32': ('rs32','a7b6e01080197bd6568f0ad88dd62630486ac28661ac7d5c2c502d020f5a35ad'),
 'increment': ('smallm_increment','0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737')}
LABELS = tuple(PINS)
SHAPES = {'qkv32':('qkv_gate',32), 'qkv16':('qkv_gate',16), 'pooler32':('pooler',32),
          'unpooler32':('unpooler',32), 'o32':('o',32)}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data,indent=2,default=str)+'\n')
def verify():
    for _,(folder,pin) in PINS.items(): assert sha(BASE/'candidates'/folder/'submission.py') == pin
    manifest=json.loads((TRIAL/'manifest.json').read_text())
    for rel,pin in manifest['files'].items(): assert sha(ROOT/rel)==pin, rel
    return manifest

def ops(sass):
    result={}
    for segment in sass.split('Function : ')[1:]:
        name=segment.splitlines()[0].strip()
        result[name]=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',segment))
    return result

def selected_attributes(ext, rp, x, out, state):
    # rqrun(v,W,x,out,esc,cnt,KS,Tb,walias,prof,sink,info): final arg is info.
    return [int(ext.rqrun(rp['i'], rp['lay'], x, out, rp['esc'], rp['cnt'],
                          rp['KS'], rp['tab'], 0, state['rq_e'], state['rq_sink'], info))
            for info in (1, 2, 3)]

def routing_proof():
    source = (BASE/'candidates/smallm_increment/submission.py').read_text()
    signature = re.search(r'int64_t rqrun\(([^)]*)\)', source)
    assert signature
    assert re.search(r'if \(info\) \{\s*cudaFuncAttributes', source)
    assert re.search(r'if \(!info\) \{\s*if \(W.dim', source)
    names = [part.strip().split()[-1] for part in signature.group(1).split(',')]
    assert names == ['v','W','x','out','esc','cnt','KS','Tb','walias','prof','sink','info'], names
    class FakeExtension:
        def __init__(self): self.queries=[]; self.launches=0
        def rqrun(self, v, W, x, out, esc, cnt, KS, Tb, walias, prof, sink, info):
            assert walias == 0, 'walias is not the query selector'
            if info == 0: self.launches+=1; return 0
            assert info in (1,2,3)
            self.queries.append(info)
            return {1:168,2:0,3:83080}[info]
    rp={'i':7,'lay':'W','esc':'esc','cnt':'cnt','KS':7,'tab':123}
    state={'rq_e':'prof','rq_sink':'sink'}
    fake=FakeExtension()
    assert selected_attributes(fake,rp,'x','out',state)==[168,0,83080]
    assert fake.queries==[1,2,3] and fake.launches==0
    rejected=False
    try: fake.rqrun(7,'W','x','out','esc','cnt',7,123,1,'prof','sink',0)
    except AssertionError: rejected=True
    assert rejected, 'old info-in-walias call must fail the CPU contract'
    return {'signature':names,'positive_queries':[1,2,3],'positive_math_launches':fake.launches,
            'negative_old_info_in_walias_rejected':rejected}

def cpu_check():
    verify(); cases=0
    for st in (5,6):
        for kcs in range(1,129):
            slot=phase=0
            for g in range(kcs):
                assert (slot,phase)==(g%st,(g//st)&1)
                slot+=1
                if slot==st: slot=0;phase^=1
                cases+=1
    before=ops((BASE/'candidates/smallm_increment/before.sass').read_text())
    after=ops((BASE/'candidates/smallm_increment/after.sass').read_text())
    assert before.keys()==after.keys()
    for symbol in before:
        for op in set(before[symbol])|set(after[symbol]):
            if op.startswith(('LDS','SYNCS','HGMMA','WARPGROUP')):
                assert before[symbol][op]==after[symbol][op],(symbol,op)
        assert after[symbol]['UIMAD.WIDE.U32']==before[symbol]['UIMAD.WIDE.U32']-1
    return {'CPU_ONLY':True,'rqrun_argument_routing':routing_proof(),'cursor_events':cases,'all_selected_body_barrier_math_LDS_counts_unchanged':True,
            'physical_register_budget':{'producer':40,'math':232,'threads':384,'dynamic':40*128+232*256,'static':168*384},
            'scope_GPU_pending':list(SHAPES)}

def load(label):
    folder,_=PINS[label]; path=BASE/'candidates'/folder/'submission.py'
    spec=importlib.util.spec_from_file_location('smallm_frozen_'+label,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def jit_gate(mod,label,outdir,registry):
    """Build + inspect rq32 before prepare can bake candidate math. No launch."""
    vs=[mod.RQV[mod.RQI[n]] for n in mod.RQPROD2]
    source=mod._rqsrc(vs);tag='rq32_%x'%(hash(source)&0xffffffff)
    digest=hashlib.sha256(source.encode()).hexdigest()
    assert tag not in registry or registry[tag]==digest,'32-bit extension name collision'
    registry[tag]=digest
    ext=mod._rqext(vs,slot=mod._rqmod2)
    assert mod._rqmod2[1]==source
    so=Path(ext.__file__); (outdir/(label+'.rq32.cu')).write_text(source)
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(so)],text=True)
    resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(so)],text=True)
    (outdir/(label+'.sass')).write_text(sass);(outdir/(label+'.resources')).write_text(resource)
    counts=ops(sass);expected=ops((BASE/'candidates/smallm_increment'/('after.sass' if label=='increment' else 'before.sass')).read_text())
    # Exact device opcode counts bind the runtime JIT recurrence to CPU-admitted bodies.
    # pw36 is a comparator; its resources are recorded without asserting rs32 identity.
    if label in ('increment','rs32'):
        for symbol,want in expected.items():
            assert counts[symbol]==want,(label,'actual JIT opcode drift',symbol)
            block=resource.split('Function '+symbol)[-1].split('Function ')[0]
            if block==resource: block=resource.split(symbol)[-1].split('Function')[0]
            assert re.search(r'REG\s*[:=]\s*168\b',block),('register gate',symbol,block[:500])
            assert re.search(r'LOCAL\s*[:=]\s*0\b',block),('local/spill gate',symbol,block[:500])
    return {'source_sha256':digest,'extension_name':ext.__name__,'so':str(so),'so_sha256':sha(so),
            'SASS_sha256':sha(outdir/(label+'.sass')),'gate':'before prepare / first candidate math',
            'resource_file':label+'.resources'}

def tensors(value, prefix=''):
    import torch
    if isinstance(value,torch.Tensor): yield prefix,value
    elif isinstance(value,dict):
        for k,v in value.items(): yield from tensors(v,prefix+'/'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value): yield from tensors(v,prefix+'/'+str(i))

def run(args):
    import torch
    sys.path.insert(0,str(ROOT/'muse_bf16_proj'));import reference
    verify(); assert args.root_review_token, 'root must supply independent critic artifact path'
    review=ROOT/args.root_review_token; assert review.is_file()
    result={'status':'RUNNING','scope':args.shapes,'review':str(review),'review_sha256':sha(review),
            'source_pins':PINS,'phases':[],'raw':[],'jit':{},'shapes':{},'started':time.time()}
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output/'result.json'
    def phase(s):
        result['phases'].append({'phase':s,'time':time.time()});dump(output,result);print(s,flush=True)
    try:
        weights=reference.generate_weights(reference.WEIGHTS_SEED)
        modules={};states={};registry={}
        for label in LABELS:
            phase('compile_and_gate_'+label); mod=modules[label]=load(label)
            result['jit'][label]=jit_gate(mod,label,args.output,registry)
            # Reset only import-time budget clock after compile; production prepare semantics otherwise unchanged.
            # Build is normally overlapped with prepare; removing that overlap is disclosed and all picks recorded.
            mod._T0=time.time();phase('actual_full_prepare_'+label)
            states[label]=mod.prepare(weights);torch.cuda.synchronize()
        immutable=[];counters=[];seen=set()
        for label,state in states.items():
            for name,t in tensors(state):
                key=t.data_ptr()
                if key in seen or not t.numel(): continue
                seen.add(key)
                if 'cnt' in name or 'sink' in name:
                    counters.append((label+name,t,t.clone()));continue
                if 'ws' not in name:
                    immutable.append((label+name,t,t.clone()))
        def invariants():
            assert all(torch.equal(t,s) for _,t,s in immutable),'W/pack/escape modified'
            assert all(torch.equal(t,baseline) if 'rw_cnt' in name else not int(t.abs().sum())
                       for name,t,baseline in counters if 'cnt' in name),'counter not reset / unrelated monotonic counter changed'
        for shape in args.shapes:
            layer,m=SHAPES[shape];n,k=reference.LAYERS[layer]
            config={'m':m,'n':n,'k':k,'layer':layer}
            entry=result['shapes'][shape]={'picks':{},'validation':[]}
            for label in LABELS:
                state=states[label];mod=modules[label];rp=state.get('rq',{}).get((layer,m))
                entry['picks'][label]={'rq':{key:value for key,value in (rp or {}).items() if isinstance(value,(int,float,str))},
                    'rq_name':mod.RQPROD2[rp['i']] if rp else None,
                    'sel':str(state.get('sel',{}).get((layer,mod._bk(m)))),
                    'gv':str(state.get('gv',{}).get((layer,mod._bk(m)))), 'lt':str(state.get('lt',{}).get((layer,m)))}
                def fn(x,out,mod=mod,state=state):out.copy_(mod.kernel_fn({'x':x,'config':config},state));return 0
                phase(shape+'_basis_FP64_'+label)
                ok,msg=mod._rqval(fn,weights[layer],m,n,torch.Generator(device='cuda').manual_seed(10001),4e-3)
                assert ok,(label,msg);entry['validation'].append({'label':label,'full_K_basis_FP64':msg})
                if rp:
                    ext=rp.get('mod',mod._rqmod[0]);x=torch.zeros((m,k),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                    attrs=selected_attributes(ext,rp,x,out,state)
                    entry['picks'][label]['selected_attrs_regs_local_shared']=attrs
                    entry['picks'][label]['cursor_affected_selected']=name_affected = mod.RQPROD2[rp['i']] in ('t32s5hze','t32s6hze','t16s5hze')
                    name=entry['picks'][label]['rq_name']
                    if label=='increment' and name in ('t32s5hze','t32s6hze','t16s5hze'):
                        assert attrs==[168,0,72840 if m==16 else 99488 if 's6' in name else 83080],attrs
            x=torch.empty((m,k),dtype=torch.bfloat16,device='cuda');graphs={};outs={}
            fixture=lambda seed,dist='normal':reference.generate_input(m,layer,seed,dist=dist)['x']
            fixtures={seed:fixture(seed) for seed in range(20000,20100)}
            for label in LABELS:
                x.copy_(fixtures[20000]);mod,state=modules[label],states[label]
                for _ in range(2):mod.kernel_fn({'x':x,'config':config},state)
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=mod.kernel_fn({'x':x,'config':config},state)
                graphs[label]=g;outs[label]=out
            def check(label,fresh,kind):
                assert bool(torch.isfinite(outs[label]).all()),(label,kind,'nonfinite')
                ok,msg=reference.check_implementation({'x':fresh,'config':config},outs[label],weights)
                assert ok,(label,kind,msg);invariants()
            probes=torch.zeros_like(x);rows=torch.arange(m,device='cuda');probes[rows,(rows*37+19)%k]=8
            for kind,fresh in [('normal',fixtures[20000]),('outliers',fixture(20999,'outliers')),('zero',torch.zeros_like(x)),('all_probes',probes),('fresh_normal',fixtures[20001])]:
                x.copy_(fresh)
                for label in LABELS:
                    outs[label].fill_(float('nan'))
                    for name,t in tensors(states[label]):
                        if name in ('/ws','/gvws') and t.is_floating_point():t.fill_(float('nan'))
                    graphs[label].replay();torch.cuda.synchronize();check(label,fresh,kind)
                    assert torch.equal(x,fresh),'X changed'
                    if kind=='zero':assert not int(torch.count_nonzero(outs[label]))
                    entry['validation'].append({'label':label,'graph_mutation':kind,'NaN_output_fresh_workspace':'PASS'})
            flush=torch.empty(64<<20,dtype=torch.float32,device='cuda');sink=torch.empty((),device='cuda');last={}
            def one(label,seed,mode):
                fresh=fixtures[seed];x.copy_(fresh);torch.sum(flush,0,out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[label].replay();b.record();torch.cuda.synchronize()
                us=a.elapsed_time(b)*1000;assert us>0 and us<float('inf')
                record={'shape':shape,'label':label,'seed':seed,'mode':mode,'us':us}
                result['raw'].append(record)
                with (args.output/'raw.jsonl').open('a') as journal:journal.write(json.dumps(record)+'\n');journal.flush()
                last[label]=seed
                return us
            def terminal(mode):
                # First operation after final timed launch is read-only output oracle, before any replay/mutation.
                assert torch.equal(x,fixtures[result['raw'][-1]['seed']])
                for label in LABELS:check(label,fixtures[last[label]],'immediate_'+mode)
                entry['validation'].append({'immediate_final_timed_output':mode,'last_fixtures':dict(last),'PASS':True})
                dump(output,result)
            phase(shape+'_timing')
            for label in LABELS:
                for _ in range(3):one(label,20000,'warmup')
            f1={}
            for label in LABELS[:-1]: f1[label+'_before']=[one(label,20001+i,'F1') for i in range(11)]
            f1['increment']=[one('increment',20001+i,'F1') for i in range(11)]
            for label in reversed(LABELS[:-1]):f1[label+'_after']=[one(label,20001+i,'F1') for i in range(11)]
            terminal('F1');entry['F1_medians_us']={key:statistics.median(v) for key,v in f1.items()}
            entry['F1_control_over_increment']={label:min(statistics.median(f1[label+'_before']),statistics.median(f1[label+'_after']))/statistics.median(f1['increment']) for label in LABELS[:-1]}
            orders=[(0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1)];pairs=[]
            for i in range(30):
                row={}
                for j in orders[i%6]:row[LABELS[j]]=one(LABELS[j],20040+i,'paired')
                pairs.append(row)
            terminal('paired');entry['paired_control_over_increment_median_ratio']={label:statistics.median(row[label]/row['increment'] for row in pairs) for label in LABELS[:-1]}
            entry['paired_all_samples']=pairs
        result['status']='VALID_FOCUSED_DEVELOPMENT_ONLY_FULL_SCOPE_REQUIRED';phase('complete')
    except BaseException:
        result['status']='INVALID_NO_PERFORMANCE_ADMISSION';result['failure']=traceback.format_exc();dump(output,result);raise
    finally:
        result['ended']=time.time();result['cleanup']='process exit releases CUDA; root independently verifies container/PID/GPU absence';dump(output,result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');ap.add_argument('--output',type=Path)
    ap.add_argument('--shapes',nargs='+',choices=SHAPES,default=['qkv32']);ap.add_argument('--root-review-token')
    args=ap.parse_args()
    if args.cpu_check:print(json.dumps(cpu_check(),indent=2));return
    assert args.output;run(args)
if __name__=='__main__':main()
