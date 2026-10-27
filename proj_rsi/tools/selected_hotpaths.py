"""Export observed scored dispatch and local graph identities, without CUDA.

Score sensitivity is not a profiler time share or a speedup prediction.
Run from repository root: python3 -m proj_rsi.tools.selected_hotpaths
"""
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def identity(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    stack_path = ROOT / 'proj_rsi/current_leaderboard_stack.json'
    stack = json.loads(stack_path.read_text())
    profile_path = ROOT / stack['routes'][0]['profile']['path']
    profile = json.loads(profile_path.read_text())
    assert identity(ROOT / stack['source']['path'])['sha256'] == stack['source']['sha256']
    assert identity(profile_path)['sha256'] == stack['routes'][0]['profile']['sha256']
    assert profile['job_id'] == stack['official_job_id']
    benchmarks = {b['index']: b for b in profile['benchmarks']}
    profiles = {p['index']: p for p in profile['profiles']}
    assert len(benchmarks) == len(stack['routes']) == 30
    totals, baselines = collections.defaultdict(float), collections.defaultdict(float)
    for b in benchmarks.values():
        assert b['status'] == 'pass' and b['case']['weight'] == 62
        totals[b['case']['load']] += b['median_ns'] / 1000
        baselines[b['case']['load']] += b['baseline_ns'] / 1000
    assert len(totals) == 6
    score = 4830.8 / math.exp(sum(math.log(baselines[l] / totals[l]) for l in totals) / 6)
    assert math.isclose(score, profile['summary']['score_us'], abs_tol=1e-8)
    rows, kernel_rows = [], []
    for r in stack['routes']:
        b = benchmarks[r['index']]
        observed = profiles[r['index']]
        assert all(b['case'][k] == r[k] for k in ('load', 'layer', 'm'))
        assert observed['case'] == b['case']
        assert observed['report'] == r['kernel_report']
        assert r['profile']['path'] == str(profile_path.relative_to(ROOT))
        assert r['profile']['sha256'] == identity(profile_path)['sha256']
        matched = 0
        for line in observed['report'].splitlines()[1:]:
            if line.startswith('total'):
                continue
            match = re.fullmatch(r'(.+?)\s+([0-9.]+) µs\s+([0-9.]+)%\s+(\d+)', line)
            assert match, ('unparsed actual kernel report', line)
            matched += 1
            kernel_rows.append(dict(index=r['index'], load=r['load'], m=r['m'], layer=r['layer'],
                                    kernel_display_name=match[1],
                                    profile_gpu_us_rounded=float(match[2]),
                                    profile_share_pct_rounded=float(match[3]), calls=int(match[4])))
        assert matched
        t = b['median_ns'] / 1000
        rows.append(dict(index=r['index'], load=r['load'], m=r['m'], layer=r['layer'],
                         family=r['family'], scored_latency_us=t,
                         score_elasticity=t / totals[r['load']] / 6,
                         scored_kernel_report=r['kernel_report']))
    rows.sort(key=lambda r: -r['score_elasticity'])
    groups = {}
    for key in ('family', 'layer'):
        by = collections.defaultdict(list)
        for r in rows:
            by[r[key]].append(r)
        groups[key] = []
        for name, members in by.items():
            cuts = collections.defaultdict(float)
            for r in members:
                cuts[r['load']] += .1 * r['scored_latency_us']
            hypothetical = score * math.exp(sum(math.log1p(-cuts[l] / totals[l]) for l in totals) / 6)
            groups[key].append(dict(name=name, selected_cases=len(members),
                                   score_elasticity=sum(r['score_elasticity'] for r in members),
                                   conditional_10pct_latency_cut_score_gain_us=score-hypothetical))
        groups[key].sort(key=lambda r: -r['score_elasticity'])
        assert math.isclose(sum(r['score_elasticity'] for r in groups[key]), 1)

    # This immutable validated capture records actual kernels but not CUDA
    # event durations. Do not fabricate a per-kernel runtime decomposition.
    local_path = ROOT / ('results/proj-rsi/scheduler/artifacts/research.pw45.gpu.01/'
                         'collection-1793111000996134576/logs/scheduler/pw45-stack-01/run/result.json')
    local = json.loads(local_path.read_text())
    assert local['state'] == 'FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED'
    assert local['sources']['pw38'] == stack['source']['sha256']
    local_rows = []
    for shape, data in local['shapes'].items():
        graph = data['actual_graph_profiles']['pw38']
        assert graph['nodes'] == len(graph['kernels']) > 0
        samples = [x['pw38'] for x in data['paired_all_samples']]
        local_rows.append(dict(shape=shape, actual_pick=data['picks']['pw38'],
                               actual_graph_profile=graph,
                               measured_paired_replays=len(samples),
                               per_kernel_duration_us=None,
                               whole_graph_samples_us=samples))
    assert len(local_rows) == 30
    report = dict(official_job_id=profile['job_id'], score_us=score,
                  evidence=[identity(stack_path), identity(profile_path), identity(local_path)],
                  scope='Projection benchmark only; not a full model serving trace.',
                  ranking='Mean within-load latency fraction across six equal log-weight loads.',
                  limitations=['Conditional cuts are arithmetic scenarios, not measured gains.',
                               'Selected case counts are not production request frequencies.',
                               'H100 local picks and H200 scored picks may differ.',
                               'No sampled inner-loop bottleneck is established by this export.',
                               'Local graph identities lack per-kernel event durations.'],
                  load_component_sums_us=dict(totals), groups=groups,
                  scored_routes=rows, scored_kernel_events=kernel_rows,
                  observed_local_graphs=local_rows)
    out = ROOT / 'proj_rsi/audits/selected-hotpaths-pw38'
    out.mkdir(exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'routes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'scored-kernel-events.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(kernel_rows[0]))
        writer.writeheader()
        writer.writerows(kernel_rows)
    print(json.dumps({'output': str(out), 'score_us': score, 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
