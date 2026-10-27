"""CPU-only descriptive RQ32 actual-policy/fixed contrasts; no admission claim.
Use: python3 analyze.py /collected/run/result.json --output /new/analysis.json
Checks exact frozen timing plans and summary consistency, retaining every sign.
Runtime source/storage/cleanup admission needs a separate collected-artifact audit.
"""
import argparse, collections, hashlib, importlib.util, json, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
TRIAL=ROOT/'proj_rsi/candidates/rq32_predxor_trial'
PINS={'pw45':'835acb90b47bacff3899c1c599ee60a24ca510801868da7971a85b490f9025f3','rq32_predxor':'cb7df808539190dac72c4aef85589f72bb1849465933871c0ef8febfe3a26210'}
LOADS=('decode1','decode16','decode32','verify1','verify16','verify32')
LAYERS=('qkv_gate','o','pooler','unpooler','router')
def load(name):
    spec=importlib.util.spec_from_file_location('offline_'+name,TRIAL/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
S=load('support');M=load('mechanism')
def median(values):return statistics.median(values)
def contrast(pairs,a,b,fa,fb):
    ds=[r[a]-r[b] for r in pairs]
    return {'F1_endpoint_average_gain_us':fa-fb,'paired_median_gain_us':median(ds),'paired_median_ratio':median(r[a]/r[b] for r in pairs),'wins_ties_losses':[sum(d>0 for d in ds),sum(d==0 for d in ds),sum(d<0 for d in ds)]}
def checked_rows(rows,plan):
    assert len(rows)==len(plan)
    assert [{k:r[k] for k in ('label','mode','seed')} for r in rows]==plan
    assert all(type(r['us']) in (int,float) and math.isfinite(r['us']) and r['us']>0 for r in rows)
def analyze(d):
    assert d['state']=='FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED'
    assert d['phase']=='complete' and d['sources']==PINS and d['official_score_claim'] is False
    assert not d['graph_cleanup'] and not d['jit_errors'] and not d.get('invalid_measurements')
    assert set(d['shapes'])=={l+'/'+s for l in LOADS for s in LAYERS}
    assert len(d['raw'])==3810
    result={'state':'DESCRIPTIVE_SIGNED_COMPARISON_RUNTIME_ADMISSION_SEPARATE','shapes':{},'fixed':{},'selected_target_admission':d['selected_target_admission'],'prepare':d['prepare']}
    loads={mode:{arm:{l:0. for l in LOADS} for arm in PINS} for mode in ('F1','paired')}
    for name,e in d['shapes'].items():
        rows=[r for r in d['raw'] if r['shape']==name];checked_rows(rows,S.policy_plan())
        pairs=[{r['label']:r['us'] for r in rows[63+2*i:65+2*i]} for i in range(32)]
        assert pairs==e['paired_all_samples']
        fm={'pw45':(median(r['us'] for r in rows[:21])+median(r['us'] for r in rows[42:63]))/2,'rq32_predxor':median(r['us'] for r in rows[21:42])}
        c=contrast(pairs,'pw45','rq32_predxor',fm['pw45'],fm['rq32_predxor'])
        assert c['paired_median_gain_us']==e['paired_control_minus_rq32_predxor_us']['pw45']
        assert c['paired_median_ratio']==e['paired_control_over_rq32_predxor_ratio']['pw45']
        assert e['graphs']=='RESET_AFTER_DRAIN' and len(e['terminal_oracles'])==4 and len(e['mutations'])==8
        result['shapes'][name]=dict(c,picks=e['picks'],F1_arm_us=fm)
        for arm in PINS:
            loads['F1'][arm][name.split('/')[0]]+=fm[arm]
            loads['paired'][arm][name.split('/')[0]]+=median(p[arm] for p in pairs)
    for layer,e in d['fixed_matched'].items():
        checked_rows(e['raw'],M.plan());rows=e['raw']
        med=[median(r['us'] for r in rows[20+21*i:41+21*i]) for i in range(7)]
        assert med==e['F1_block_medians_us']
        fm={'parent5':(med[0]+med[6])/2,'new5':(med[1]+med[5])/2,'parent6':(med[2]+med[4])/2,'new6':med[3]}
        pairs=[{r['label']:r['us'] for r in rows[167+4*i:171+4*i]} for i in range(32)]
        assert pairs==e['paired_all_samples']
        cs={a+'_minus_'+b:contrast(pairs,a,b,fm[a],fm[b]) for a,b in [('parent5','new5'),('parent6','new6'),('parent5','parent6'),('new5','new6')]}
        assert cs==e['contrasts']
        assert e['state']=='VALID_MATCHED_FIXED_NO_SELECTED_POLICY_INFERENCE' and e['cleanup']=='DRAINED_RESET_ALL_CREATED'
        result['fixed'][layer]={'contrasts':cs,'F1_arm_us':fm,'validation':e['validation'],'edge_validation':e['edge_validation']}
    assert set(result['fixed'])=={'pooler','unpooler'}
    result['six_load_component_sums_us']=loads
    # Ratio of geometric means of actual load sums; no historical/official score rescale.
    result['actual_policy_geometric_load_speedup']={mode:math.exp(statistics.mean(math.log(v['pw45'][l]/v['rq32_predxor'][l]) for l in LOADS)) for mode,v in loads.items()}
    result['signed_shape_counts']={mode:{'positive':sum(e[key]>0 for e in result['shapes'].values()),'ties':sum(e[key]==0 for e in result['shapes'].values()),'negative':sum(e[key]<0 for e in result['shapes'].values())} for mode,key in [('F1','F1_endpoint_average_gain_us'),('paired','paired_median_gain_us')]}
    result['opposite_sign_shapes']=[name for name,e in result['shapes'].items() if e['F1_endpoint_average_gain_us']*e['paired_median_gain_us']<0]
    result['limits']=['Every tiny signed point retained; no significance or minimum gain gate.','Fixed SO and allocation placement effects remain; matched wins alone do not isolate cycle attribution.','Actual full30 selected policy and fixed mechanism remain distinct.','One prepare per policy; no selection-frequency inference.','Historical official PW45 score 3179.285623687654 is context, not this local-run score.','No source, numerical, storage, cleanup or official promotion admission from this analyzer alone.']
    return result

def selfcheck():
    # Opposite signed estimators and small deltas must survive unchanged.
    pairs=[{'a':1.+1e-10,'b':1.}]*17+[{'a':.9,'b':1.}]*15
    c=contrast(pairs,'a','b',.999,1.)
    assert 0<c['paired_median_gain_us']<1e-9 and c['F1_endpoint_average_gain_us']<0 and c['wins_ties_losses']==[17,0,15]
    p=S.policy_plan();assert len(p)==127 and len(M.plan())==295
    try:checked_rows([dict(r,us=float('nan')) for r in p],p)
    except AssertionError:pass
    else:raise AssertionError('NaN admitted')
    print('PASS tiny gain/opposite signs/invalid elapsed; full runtime result remains required')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('result',type=Path,nargs='?');ap.add_argument('--output',type=Path);ap.add_argument('--selfcheck',action='store_true');args=ap.parse_args()
    if args.selfcheck:selfcheck();return
    assert args.result and args.output
    data=args.result.read_bytes();out=analyze(json.loads(data));out['result_sha256']=hashlib.sha256(data).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:out[k] for k in ('state','actual_policy_geometric_load_speedup','signed_shape_counts','opposite_sign_shapes')},indent=2))
if __name__=='__main__':main()
