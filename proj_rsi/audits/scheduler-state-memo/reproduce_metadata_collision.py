"""Show same-tick corruption can be metadata-identical; temporary queues only."""
import importlib
import json
import tempfile
from proj_rsi.scheduling import store as baseline
from proj_rsi.scheduling.test_store import job

candidate = importlib.import_module('proj_rsi.audits.scheduler-state-memo.test_proposal').candidate


def run():
    for attempt in range(100):
        with tempfile.TemporaryDirectory(prefix='memo-metadata-collision-') as directory:
            store = candidate.QueueStore(directory)
            store.register(job('one')); store.register(job('two'))
            before = store._wal_fingerprint()
            raw = store.wal_path.read_bytes()
            changed = raw.replace(b'"register"', b'"tampered"', 1)
            store.wal_path.write_bytes(changed)
            after = store._wal_fingerprint()
            if before != after:
                continue
            with store._locked():
                stale = store._read()
            old = baseline.QueueStore(directory)
            try:
                with old._locked():
                    old._read()
            except baseline.CorruptWAL as error:
                detected = str(error)
            else:
                raise AssertionError('baseline unexpectedly missed corruption')
            return {'attempt': attempt, 'fingerprint_before': before, 'fingerprint_after': after,
                    'actual_bytes_changed': changed != raw,
                    'candidate_returned_unverified_sequence': stale['sequence'],
                    'baseline_refused': detected,
                    'conclusion': 'ordinary same-tick same-size corruption bypasses exact stat-fingerprint memo; DO NOT ACTIVATE'}
    raise RuntimeError('could not reproduce metadata collision in 100 attempts')


if __name__ == '__main__':
    from pathlib import Path
    result = run()
    Path(__file__).with_name('metadata-collision.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
