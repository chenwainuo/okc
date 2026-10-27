"""Read-only immutable lesson retrieval proof for PW40 provenance."""
import hashlib
import json
from pathlib import Path
from proj_rsi import learning
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
APPLIED={
 'wring-mechanism-policy-separation@1':'8aed17de00967041aa8d8f3de9d75ba2adf3f9f8cadb3a2315c610cc2067b3ab',
 'composition-not-additivity@1':'0e3ae3fc3b9be194f5afb595c0b3bd065aeafeccfdcd3f4f344d401b59404ee0',
 'eh-tiny-context-bound@1':'6b58697000771856c5c3ef11cbc68049f172bcaa898e63e3e8fadfbb5f8cd8f9'}
def proof():
    registry=json.loads((ROOT/'proj_rsi/learnings.json').read_text());retained=json.loads((HERE/'learning-query.json').read_text());records=[]
    for ref,digest in APPLIED.items():
        name,version=ref.rsplit('@',1);record=next(r for r in registry['lessons'] if r['id']==name and r['version']==int(version))
        assert record['record_sha256']==learning.digest(record)==digest
        assert any(r['kind']=='lessons' and r['id']==name and r['version']==int(version) for r in retained)
        records.append(dict(ref=ref,record_sha256=digest,scope=record['scope'],transfer_limits=record['transfer_limits'],falsification=record['falsification'],next_test=record['next_test']))
    return dict(state='PASS_READONLY_LESSON_RETRIEVAL_AND_DIGEST_BINDING',applied=records,registry_version_observed=registry['registry_version'],
                registry_sha256_at_retrieval=hashlib.sha256((ROOT/'proj_rsi/learnings.json').read_bytes()).hexdigest(),
                command="python3 -m proj_rsi.learning query 'wring separate composition overlap' --hardware H100",API_GPU_queue_writes=0,
                scope='Only optional qkv128 x6w4 contender over PW38; no transfer/additivity/promotion, equal-depth negative retained')
if __name__=='__main__':
    result=proof();(HERE/'learning-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
