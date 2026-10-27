import hashlib,importlib.util,json,statistics,sys,collections
from pathlib import Path
root=Path('/home/ubuntu/okc');base=root/'proj_rsi/candidates';out=root/'proj_rsi/audits/sol03-kernel-validation-data'
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
new=base/'operand_reverse_group8_stmatrix';sys.path.insert(0,str(new));import preflight,liveness
m=json.loads((new/'manifest.json').read_text());assert hashlib.sha256((new/'manifest.json').read_bytes()).hexdigest()=='80dae9122d1f964af99cdf586fbf0d4b5e9ac28db4c7998f253069ae6889f24b'
for f,h in m['files'].items():assert hashlib.sha256((new/f).read_bytes()).hexdigest()==h,f
r={'composition_manifest_all_files':len(m['files']),'sourceproof':preflight.sourceproof(),'schedule':preflight.proof(),'composition_physical':liveness.check((new/'sass.txt').read_text()),'negative_controls':preflight.physical_negative_controls((new/'sass.txt').read_text())}
store=module(base/'operand_reverse_dual_stmatrix/liveness.py','sol03_ds_store')
r['composition_store']=store.store_form((new/'sass.txt').read_text(),'rq_px_g8_stmx',True,dual=True)
import layout_proof;r['layout']=layout_proof.wproof()
d=json.loads((out/'operand-reverse-group8-01.json').read_text());r['g8_status']=d['status'];r['g8_keys']=list(d)
for label,folder in [('dp','operand_reverse_dual_producer'),('ds','operand_reverse_dual_stmatrix'),('g8','operand_reverse_group8')]:
 checker=module(base/folder/'liveness.py','sol03_'+label)
 data=(out/f'operand-reverse-group8-01.{label}.jit.sass').read_text();assert hashlib.sha256(data.encode()).hexdigest()==d['actual_JIT_SASS'][label]['sha256']
 r['g8_actual_'+label]=checker.check(data)
 r['g8_actual_store_'+label]=store.store_form(data,{'dp':'rq_px_dp','ds':'rq_px_dp_stmx','g8':'rq_px_g8'}[label],label=='ds',dual=True)
t=d['timing'];raw=t['F1_samples_us'];med={k:statistics.median(v) for k,v in raw.items()};assert med==t['F1_medians_us'];assert all(len(v)==21 for v in raw.values())
f1={p:min(med[p+'_before'],med[p+'_after'])-med['g8'] for p in ('incumbent','dp','ds')};assert f1==t['F1_g8_gain_us']
pairs=t['paired_graph_us'];assert len(pairs)==21
pg={p:statistics.median(x[p]-x['g8'] for x in pairs) for p in f1};assert pg==t['paired_g8_median_gain_us'];wins={p:sum(x[p]>x['g8'] for x in pairs) for p in f1};assert wins==t['paired_g8_wins']
r['g8_recomputed']={'F1':med,'gains_F1':f1,'gains_paired':pg,'wins':wins,'order_counts':dict(collections.Counter(','.join(x) for x in pairs)),'sample_count_F1':sum(map(len,raw.values())),'paired_count':len(pairs),'mutations':len(d['actual_timed_graph_checks']),'XP_poison_checks':sum(x['packed_poison_overwritten_byte_exact'] for x in d['actual_timed_graph_checks'])}
r['g8_runtime_evidence']={k:d[k] for k in ('attrs','packer_attrs','GPU_pack_all65536_BF16_storage_bits','fresh_workspace_and_output','profile','weights_inputs_counters','graph_cleanup')}
(out/'cpu-review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'manifest_files':r['composition_manifest_all_files'],'composition_source':'PASS','composition_layout':'PASS','all_actual_JIT':'PASS','recomputed':r['g8_recomputed']},indent=2))
