"""Read-only bound-prefix/last-event cost model; no QueueStore instance or live cache."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def run():
    attestation=json.loads((HERE/'live-readonly-profile.json').read_text());path=ROOT/attestation['path']
    started=time.perf_counter();digest=hashlib.sha256();remaining=attestation['complete_bytes']
    with path.open('rb') as f:
        while remaining:
            block=f.read(min(1<<20,remaining));assert block
            digest.update(block);remaining-=len(block)
        assert digest.hexdigest()==attestation['initial_sha256'],'previously strict-validated byte prefix changed'
        f.seek(attestation['last_record_offset']);line=f.read(attestation['last_record_bytes'])
    entry=json.loads(line);p=entry['payload'];canonical=json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==entry['sha256'];assert p['sequence']==p['snapshot']['sequence']==attestation['events']
    assert p['snapshot']['version']==1 and isinstance(p['snapshot']['jobs'],dict)
    return {'status':'READ_ONLY_VERIFIED_PREFIX_LAST_EVENT_MODEL_PASS','prefix_bytes':attestation['complete_bytes'],'events':attestation['events'],'seconds':time.perf_counter()-started,'original_strict_streaming_seconds':attestation['strict_streaming_replay_seconds'],'full_payload_records_parsed':1,'live_cache_written':False,'live_store_instantiated':False,'limits':'Independent byte-read model, not deployed QueueStore/CLI benchmark; root has not activated proposal','GPU_calls':0}
if __name__=='__main__':
    result=run();(HERE/'live-readonly-bound-model.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
