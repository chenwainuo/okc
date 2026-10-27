"""Bind PW44 design discriminators to immutable recorded lessons, not transferable gains."""
import hashlib,json
from pathlib import Path
from proj_rsi import learning
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
APPLIED={
'eh-tiny-context-bound':'Keep EPfalse and current zero/trap/Dwait0; preserve tiny mixed effects and require actual serving timings.',
'composition-not-additivity':'Private layout and barrier removal remain separate equal-resource source arms; no additive gain or source-only shipping claim.',
'wring-mechanism-policy-separation':'Measure reused-X EH, private+bar and private-free matched arms separately from actual prepared PW38/new policies.',
'native-profile-x-handoff-accounting':'Profile0x3480 dependency association is not intrinsic bar latency or convertible sample fraction; removal needs storage ownership proof.',
'qkv-split4-full-pipeline-regression':'Require complete selected graph and actual current policy, no main-only/parent-only performance admission.'}
def proof():
 data=json.loads((HERE/'learning-registry-snapshot.json').read_text());validation=learning.validate(data,ROOT)
 lessons={l['id']:l for l in data['lessons']};rows=[]
 for id,applied in APPLIED.items():
  r=lessons[id];assert r['version']==1 and r['record_sha256']==learning.digest(r)
  rows.append({'id':id,'version':r['version'],'record_sha256':r['record_sha256'],'scope':r['scope'],'transfer_limits':r['transfer_limits'],'applied_to':applied,'observations_retained':[{'id':o['id'],'outcome':o['outcome'],'statement':o['statement']} for o in r['observations']]})
 source=(HERE/'submission.py').read_text();assert 'EP = false' in source and 'if constexpr (!(FL & 524288))' in source
 return {'state':'PW44_LESSON_SCOPE_PASS_NO_GAIN_TRANSFER','registry_validation':validation,'registry_snapshot_sha256':hashlib.sha256((HERE/'learning-registry-snapshot.json').read_bytes()).hexdigest(),'query_sha256':hashlib.sha256((HERE/'learning-query.json').read_bytes()).hexdigest(),'applied':rows,'performance_transfer':'NONE; no measurement or cost fraction is transferred into attainable gain'}
if __name__=='__main__':
 p=proof();(HERE/'learning-proof.json').write_text(json.dumps(p,indent=2)+'\n');print(p['state'])
