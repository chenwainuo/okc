"""Read a fixed initial WAL byte prefix only. Never instantiate QueueStore/recover/write WAL."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def profile(path):
    size=path.stat().st_size;start=time.perf_counter();h=hashlib.sha256();remaining=size
    with path.open('rb') as f:
        while remaining:
            block=f.read(min(1<<20,remaining))
            if not block:raise RuntimeError('WAL shortened during read-only profile')
            h.update(block);remaining-=len(block)
    hash_seconds=time.perf_counter()-start
    start=time.perf_counter();remaining=size;sequence=0;complete=0;tail=0;last_start=0;last_bytes=0
    with path.open('rb') as f:
        while remaining:
            line=f.readline(remaining)
            if not line:raise RuntimeError('WAL shortened during read-only profile')
            remaining-=len(line)
            if not line.endswith(b'\n'):
                tail=len(line);break
            entry=json.loads(line);p=entry['payload'];canonical=json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            assert hashlib.sha256(canonical).hexdigest()==entry['sha256']
            assert p['sequence']==sequence+1
            state=p['snapshot'];assert state['version']==1 and state['sequence']==p['sequence'] and isinstance(state['jobs'],dict)
            sequence=p['sequence'];last_start=complete;last_bytes=len(line);complete+=len(line)
    return dict(status='READ_ONLY_PROFILE_VALID_INITIAL_PREFIX',path=str(path.relative_to(ROOT)),initial_bytes=size,
                initial_sha256=h.hexdigest(),complete_bytes=complete,incomplete_initial_tail_bytes=tail,events=sequence,
                final_snapshot_jobs=len(state['jobs']) if sequence else 0,last_record_offset=last_start,last_record_bytes=last_bytes,
                prefix_sha256_seconds=hash_seconds,strict_streaming_replay_seconds=time.perf_counter()-start,
                final_path_bytes=path.stat().st_size,method='fixed initial size, open rb, streaming hash and independent original checksum/sequence/envelope replay; no locks/store/recovery/queue writes',GPU_calls=0)
if __name__=='__main__':
    result=profile(ROOT/'results/proj-rsi/scheduler/events.jsonl');Path(__file__).with_name('live-readonly-profile.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
