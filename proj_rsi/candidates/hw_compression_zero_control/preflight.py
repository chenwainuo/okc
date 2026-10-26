"""CPU-only dependency, reused-parser/unit, poison-oracle and profiler-policy gates."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

import torch

HERE = Path(__file__).resolve().parent
REAL = HERE.parent / "hw_compression_counters"


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for p in HERE.glob("*.py"): ast.parse(p.read_text())
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    D = imported(HERE / "driver.py", "hw_zero_cpu_driver")
    assert sha(REAL / "manifest.json") == D.REAL_MANIFEST_SHA
    dependency = json.loads((REAL / "manifest.json").read_text())
    assert all(sha(REAL / n) == h for n, h in dependency["sha256"].items())
    P = imported(HERE / "parse_report.py", "hw_zero_cpu_parser").parser_module()
    assert (HERE / "metrics.txt").read_bytes() == (REAL / "metrics.txt").read_bytes()
    assert len(P.METRICS) == 21
    def normalized_script(path):
        text = '\n'.join(l for l in path.read_text().splitlines() if l.strip() and not l.startswith('#'))
        return text.replace('hw_compression_zero_control', 'hw_compression_counters').replace('hw-zero-counters01', 'hw-native-counters01')
    assert normalized_script(HERE / "run.sh") == normalized_script(REAL / "run.sh"), "profile/cache/clock method changed"
    code = ast.parse((HERE / "driver.py").read_text())
    calls = [n.func.attr for n in ast.walk(code) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {"prepare", "validate", "_rqval", "_time_us", "_tune", "_ltsearch", "measure", "profile"}
    assert calls.count("lt_add") == 1 and "range_push" in calls
    output = torch.zeros((128, 8192), dtype=torch.bfloat16)
    D.exact_zero(output)
    oracle_rejections = 0
    for poison in (float('nan'), 1.0):
        output.fill_(poison)
        try: D.exact_zero(output)
        except AssertionError: oracle_rejections += 1
        else: raise AssertionError('zero oracle accepted poisoned/nonzero output')
    # Exercise the exact reused CSV parser with explicit base units and undefined ILC ratios.
    header = ['ID', 'Kernel Name', 'thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'] + P.METRICS
    units = dict.fromkeys(header, '')
    for direction in ['read', 'write']:
        units['dram__bytes_'+direction+'.sum'] = 'byte'
        units['dram__sectors_'+direction+'.sum'] = 'sector'
    rows = []
    for i, label in enumerate(['compression_generic', 'compression_none']):
        r = dict.fromkeys(header, '0'); r.update({'ID': str(i), 'Kernel Name': P.KERNEL, header[2]: '34 "<default domain>:'+label+':none:none:none:none:none:none"'})
        for direction in ['read', 'write']:
            sectors = 1000 if direction == 'read' else 100
            r['dram__sectors_'+direction+'.sum'] = str(sectors); r['dram__bytes_'+direction+'.sum'] = str(32*sectors)
        for name, value in zip(P.LAUNCH, [1, 128, 1, 384, 1, 1, 168, 213268]): r[name] = str(value)
        for name in P.ILC[3:]: r[name] = 'n/a'
        rows.append(r)
    def csv_text(u, rs):
        out = io.StringIO(); w = csv.DictWriter(out, fieldnames=header, quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerow(u); w.writerows(rs); return out.getvalue()
    assert P.parse_text(csv_text(units, rows))['observed_DRAM_read_bytes_saved'] == 0
    parser_rejections = 0
    for bad in [csv_text(dict(units, **{'dram__bytes_read.sum':'Mbyte'}), rows),
                csv_text(units, [dict(rows[0], **{'Kernel Name':'wrong'}), rows[1]]),
                csv_text(units, [rows[0], dict(rows[1], **{header[2]:rows[0][header[2]]})]),
                csv_text(units, [rows[0]])]:
        try: P.parse_text(bad)
        except (AssertionError, KeyError): parser_rejections += 1
        else: raise AssertionError('reused parser accepted invalid report')
    result = {'state':'CPU_DEPENDENCIES_PARSER_ZERO_POISON_POLICY_PASS_GPU_UNRUN', 'CUDA_API_calls':0,
              'same_supported_metrics':21, 'same_profiler_policy':True, 'zero_oracle_negative_controls':oracle_rejections,
              'raw_parser_negative_controls':parser_rejections, 'all_K_onehot':'NOT_RUN_ZERO_WEIGHT_UNINFORMATIVE',
              'normal_weight_rerun':False, 'real_counter_manifest_sha256':D.REAL_MANIFEST_SHA}
    (HERE / 'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    names = sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name != 'manifest.json')
    manifest = {'state':'ZERO_POSITIVE_CONTROL_FROZEN_FOR_REVIEW_GPU_UNRUN',
                'dependencies':{'real_counter_manifest_sha256':D.REAL_MANIFEST_SHA},
                'sha256':{n:sha(HERE/n) for n in names}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('MANIFEST',sha(HERE/'manifest.json'))


if __name__ == '__main__': main()
