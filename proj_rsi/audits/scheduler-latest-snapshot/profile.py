"""CPU-only synthetic WAL; all reads/writes confined to a TemporaryDirectory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import time
from proj_rsi.scheduling import store as baseline

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('proj_rsi.scheduling._latest_profile', HERE/'store_proposal.py')
proposal = importlib.util.module_from_spec(spec); spec.loader.exec_module(proposal)


def timed(store, count):
    times = []
    for _ in range(count):
        start = time.perf_counter()
        with store._locked():
            state = store._read()
        times.append(time.perf_counter() - start)
    return {'seconds': times, 'median_seconds': statistics.median(times), 'sequence': state['sequence']}


def run():
    with tempfile.TemporaryDirectory(prefix='scheduler-latest-profile-') as temporary:
        directory = Path(temporary)
        # Large audit data, small final state: isolates the repeated full-WAL hash.
        padding = 'x' * (1 << 20)
        with (directory/'events.jsonl').open('wb') as handle:
            for sequence in range(1, 257):
                payload = {'sequence': sequence, 'timestamp': 1, 'action': 'synthetic',
                           'data': padding, 'snapshot': {'version': 1, 'sequence': sequence, 'jobs': {}}}
                handle.write(baseline._json({'payload': payload, 'sha256': hashlib.sha256(baseline._json(payload)).hexdigest()}) + b'\n')
        reference = baseline.QueueStore(directory)
        start = time.perf_counter()
        candidate = proposal.QueueStore(directory, latest_snapshot_reads=True)
        startup_seconds = time.perf_counter() - start
        result = {'source': 'synthetic temporary WAL; no live queue access',
                  'wal_bytes': (directory/'events.jsonl').stat().st_size,
                  'baseline_cold': timed(reference, 1), 'proposal_first_operational': timed(candidate, 1), 'proposal_strict_startup_seconds': startup_seconds,
                  'baseline_repeated': timed(reference, 7), 'proposal_repeated': timed(candidate, 7)}
        with candidate._locked():
            state = candidate._read()
            candidate._commit(state, 'synthetic-own-commit')
        result['proposal_after_own_commit'] = timed(candidate, 7)
        # Cooperating separate instance writer invalidates the warm memo.
        with reference._locked():
            state = reference._read()
            reference._commit(state, 'synthetic-external-commit')
        result['proposal_after_external_commit'] = timed(candidate, 1)
        result['proposal_repeated_after_external'] = timed(candidate, 7)
        result['repeated_speedup'] = result['baseline_repeated']['median_seconds']/result['proposal_repeated']['median_seconds']
        return result


if __name__ == '__main__':
    result = run()
    (HERE/'synthetic-profile.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
