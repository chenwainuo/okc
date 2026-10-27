"""Read-only applied-lesson binding; no registry mutation or promotion."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REFS=['prepare-selection-is-experiment@2','counter-contract-not-name@1','composition-not-additivity@1','eh-tiny-context-bound@1','wring-mechanism-policy-separation@1','qkv-split4-full-pipeline-regression@1']
def proof():
    from proj_rsi.learning import digest,validate
    data=json.loads((ROOT/'proj_rsi/learnings.json').read_text());validate(data)
    records={r['id']+'@'+str(r['version']):r for r in data['lessons']}
    selected=[]
    for ref in REFS:
        record=records[ref];assert digest(record)==record['record_sha256']
        selected.append({'ref':ref,'record_sha256':record['record_sha256'],'applied':'Keep actual PW38 serving/prepare comparator separate from fixed o32 mechanism arms, preserve all controls/EH/counter storage semantics and all tiny signed outcomes; source composition/grid/resource changes are not measured/additive speed gains.'})
    return {'state':'PASS_APPLIED_HASHED_LESSONS','applied':selected,'counter_contract':'New fullK plain-store path does not touch counters; compare entire inactive storage to actual prepare baseline rather than demand unrelated zeros. Original ZI control active/dormant semantics stay unchanged.','evidence_transfer':'Only invariants and experiment design transfer; no old latency or CPU-to-GPU/H100-to-H200 speed assumption.','runtime':'UNRUN_NO_SOURCE_PROMOTION'}
