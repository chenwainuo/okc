"""Read-only verification of the current selected-hotpath export; no GPU imports."""
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report = json.loads((HERE / 'report.json').read_text())
    for item in report['evidence']:
        assert sha(ROOT / item['path']) == item['sha256']
    stack = json.loads((ROOT / report['evidence'][0]['path']).read_text())
    source = ROOT / stack['source']['path']
    assert sha(source) == stack['source']['sha256']
    text = source.read_text();tree = ast.parse(text)
    profile_path = ROOT / stack['routes'][0]['profile']['path']
    profile = json.loads(profile_path.read_text())
    assert Path(profile['submission']).resolve() == source.resolve()
    assert report['official_job_id'] == profile['job_id'] == stack['official_job_id']
    bench = {b['index']: b for b in profile['benchmarks']}
    profiles = {p['index']: p for p in profile['profiles']}
    routes = {r['index']: r for r in stack['routes']}
    exported = {r['index']: r for r in report['scored_routes']}
    assert set(bench) == set(profiles) == set(routes) == set(exported) == set(range(30))
    times = collections.defaultdict(float);base = collections.defaultdict(float)
    cells = set();bindings = 0
    for index, route in routes.items():
        b,p,out = bench[index],profiles[index],exported[index]
        assert route['profile'] == stack['routes'][0]['profile']
        assert sha(profile_path) == route['profile']['sha256']
        assert route['kernel_report'] == p['report'] == out['scored_kernel_report']
        assert all(b['case'][k] == p['case'][k] == route[k] == out[k] for k in ('load','layer','m'))
        assert b['case']['weight'] == 62 and b['status'] == 'pass'
        assert math.isfinite(b['median_ns']) and b['median_ns'] > 0
        times[route['load']] += b['median_ns'] * 62
        base[route['load']] += b['baseline_ns'] * 62
        assert (route['load'],route['layer']) not in cells
        cells.add((route['load'],route['layer']))
        for binding in route['bindings']:
            assert binding['path'] == stack['source']['path'] and binding['source_sha256'] == sha(source)
            node = tree.body[binding['ordinal']]
            assert (node.lineno,node.end_lineno) == (binding['line'],binding['end_line'])
            assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() == binding['AST_sha256']
            segment = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
            assert hashlib.sha256(segment.encode()).hexdigest() == binding['source_segment_sha256']
            bindings += 1
    assert len(times) == 6 and all(sum(load == x for load,layer in cells) == 5 for x in times)
    score = 4830.8 * math.prod(times[load]/base[load] for load in times)**(1/6)
    assert math.isclose(score,report['score_us'],abs_tol=1e-9)
    assert math.isclose(score,profile['summary']['score_us'],abs_tol=1e-9)
    for index, out in exported.items():
        elasticity = bench[index]['median_ns'] * 62 / times[out['load']] / 6
        assert math.isclose(out['scored_latency_us'],bench[index]['median_ns']/1000,abs_tol=1e-12)
        assert math.isclose(out['score_elasticity'],elasticity,abs_tol=1e-15)
        eps = 1e-5
        finite_difference = math.log((1 + 6*elasticity*eps)**(1/6))/math.log1p(eps)
        assert math.isclose(finite_difference,elasticity,rel_tol=1e-5)
    for key, groups in report['groups'].items():
        for group in groups:
            members = [r for r in exported.values() if r[key] == group['name']]
            fractions = [sum(bench[r['index']]['median_ns']*62 for r in members if r['load']==load)/times[load] for load in times]
            changed = score*math.prod(1-.1*f for f in fractions)**(1/6)
            assert math.isclose(group['conditional_10pct_latency_cut_score_gain_us'],score-changed,abs_tol=1e-9)
            assert math.isclose(group['score_elasticity'],sum(fractions)/6,abs_tol=1e-15)
    with (HERE/'routes.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==30
    for row,out in zip(csv_rows,report['scored_routes']):
        assert all(row[k] == str(v) for k,v in out.items())
    # Independently parse the printed profile from the right: names may contain
    # spaces, commas or a deliberate display ellipsis. These rounded numbers
    # stay separate from calibrated scored medians and local graph events.
    events = []
    for route in stack['routes']:
        for line in profiles[route['index']]['report'].splitlines()[1:]:
            if line.startswith('total'):
                continue
            name,time,unit,share,calls = line.rsplit(None,4)
            assert unit == 'µs' and share.endswith('%')
            observed_family = ('gv' if name.startswith('void gv<') else 'rq' if name.startswith('void rq<')
                               else 'rr' if name.startswith('void rr<') else 'rtk' if name.startswith('void rtk<')
                               else 'Lt/vendor' if name.startswith('nvjet_') else None)
            assert observed_family == route['family'], (route['index'],name,route['family'])
            events.append(dict(index=route['index'],load=route['load'],m=route['m'],layer=route['layer'],
                               kernel_display_name=name,profile_gpu_us_rounded=float(time),
                               profile_share_pct_rounded=float(share[:-1]),calls=int(calls)))
    assert events == report['scored_kernel_events']
    with (HERE/'scored-kernel-events.csv').open(newline='') as f:event_csv=list(csv.DictReader(f))
    assert len(event_csv) == len(events)
    for row,event in zip(event_csv,events):
        assert all(row[k] == str(v) for k,v in event.items())
    local = json.loads((ROOT / report['evidence'][2]['path']).read_text())
    assert local['sources']['pw38'] == sha(source) and local['hardware']['SM_count'] == 132
    assert local['hardware']['name'].startswith('NVIDIA H100')
    assert local['state'] == 'FULL_LOCAL_STACK_VALIDATION_COMPLETE_ROOT_REVIEW_REQUIRED'
    for out in report['observed_local_graphs']:
        data = local['shapes'][out['shape']]
        assert out['actual_pick'] == data['picks']['pw38']
        assert out['actual_graph_profile'] == data['actual_graph_profiles']['pw38']
        assert out['whole_graph_samples_us'] == [r['pw38'] for r in data['paired_all_samples']]
        assert out['measured_paired_replays'] == len(out['whole_graph_samples_us']) == 30
        assert out['per_kernel_duration_us'] is None
    assert len(report['observed_local_graphs']) == 30
    return dict(state='PASS_CURRENT_EXPORT_CPU_ONLY',score_us=score,source_sha256=sha(source),
                scored_case_profile_joins=30,source_AST_and_segment_bindings=bindings,
                weighted_six_load_elasticities_and_finite_differences=30,
                actual_rounded_scored_kernel_event_joins=len(events),
                local_graph_identity_and_whole_graph_sample_joins=30,
                per_kernel_duration_invented=False,full_serving_frequency_or_share_claim=False,
                tool_sha256=sha(ROOT/'proj_rsi/tools/selected_hotpaths.py'),
                report_sha256=sha(HERE/'report.json'),CSV_sha256=sha(HERE/'routes.csv'),
                kernel_events_CSV_sha256=sha(HERE/'scored-kernel-events.csv'))

if __name__ == '__main__':print(json.dumps(main(),indent=2))
