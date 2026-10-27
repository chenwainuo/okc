"""Four-node raw-unit parser; ratios of allocation contrasts, no latency/wire claims."""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
from pathlib import Path

PHYSICAL = ["dram__bytes_read.sum", "dram__bytes_write.sum", "dram__sectors_read.sum", "dram__sectors_write.sum"]
ILC = ["lrc__ilc_input_sectors.sum", "lrc__ilc_input_sectors_compressible.sum", "lrc__ilc_output_sectors.sum",
       "lrc__average_ilc_input_sector_compression_rate.ratio", "lrc__average_ilc_input_sector_success_rate.pct",
       "lrc__average_ilc_output_sector_compression_achieved_rate.ratio"]
L2 = ["lts__t_sectors.sum", "lts__d_sectors_fill_device.sum", "lts__t_sector_hit_rate.pct"]
LAUNCH = ["launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z", "launch__block_dim_x",
          "launch__block_dim_y", "launch__block_dim_z", "launch__registers_per_thread", "launch__shared_mem_per_block"]
METRICS = PHYSICAL + ILC + L2 + LAUNCH
LABELS = ("raw_generic", "raw_none", "signlow_generic", "signlow_none")
KERNEL = "okc_hwc_copy64m"


def number(text):
    try:
        v = float(text.replace(',', '').strip())
        return v if math.isfinite(v) else None
    except (ValueError, AttributeError):
        return None


def parse_text(text, registers=32):
    lines = text.lstrip('\ufeff').splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    assert len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate column"
    assert set(['ID', 'Kernel Name'] + METRICS) <= set(reader.fieldnames), "missing required column"
    units, rows = None, []
    for row in reader:
        assert None not in row, "overwide row"
        if not row.get('ID'):
            assert units is None, "duplicate units row"
            units = row
        else:
            assert row['ID'].isdigit(), "unexpected row"
            rows.append(row)
    assert units is not None and len(rows) == 4 and len({r['ID'] for r in rows}) == 4, "exactly four distinct nodes"
    for direction in ('read', 'write'):
        assert units[f'dram__bytes_{direction}.sum'] == 'byte', "non-base byte units"
        assert units[f'dram__sectors_{direction}.sum'] == 'sector', "non-base sector units"
    assert all(units[k] == 'sector' for k in L2[:2]) and units[L2[2]] == '%'
    assert units[LAUNCH[6]] == 'register/thread' and units[LAUNCH[7]] == 'byte/block'
    assert all(units[k] in ('', 'block') for k in LAUNCH[:6])
    assert all(units[k] in ('', 'sector') for k in ILC[:3])
    assert units[ILC[3]] == '' and units[ILC[4]] == '%' and units[ILC[5]] == ''
    cases = {}
    for row in rows:
        ranges = ' '.join(v or '' for k, v in row.items() if 'Domain:' in k)
        matches = [label for label in LABELS if ':' + label + ':' in ranges]
        assert len(matches) == 1 and matches[0] not in cases, "missing/duplicate case range"
        assert row['Kernel Name'] == KERNEL, "different copy kernel"
        values = {k: number(row[k]) for k in METRICS}
        assert all(values[k] is not None and values[k] >= 0 for k in PHYSICAL + ILC[:3] + L2 + LAUNCH)
        assert all(values[k] is None or values[k] >= 0 for k in ILC[3:])
        assert values[ILC[4]] is None or values[ILC[4]] <= 100
        assert values[L2[2]] <= 100
        assert [values[k] for k in LAUNCH[:6]] == [128, 1, 1, 256, 1, 1]
        assert values[LAUNCH[6]] == registers
        # This kernel has no shared memory. NCU may include Hopper's1KiB driver reserve.
        assert values[LAUNCH[7]] in (0, 1024)
        for direction in ('read', 'write'):
            assert values[f'dram__bytes_{direction}.sum'] == 32 * values[f'dram__sectors_{direction}.sum'], "byte/sector mismatch"
        assert values['dram__bytes_read.sum'] > 0
        cases[matches[0]] = {'ID': row['ID'], 'kernel': row['Kernel Name'], 'values': values,
                            'unavailable_ILC_ratios': [k for k in ILC[3:] if values[k] is None]}
    assert set(cases) == set(LABELS)
    assert all(cases[label]['values'][k] == cases[LABELS[0]]['values'][k] for label in LABELS for k in LAUNCH)
    contrasts = {}
    for payload in ('raw', 'signlow'):
        g = cases[payload + '_generic']['values']; n = cases[payload + '_none']['values']
        contrasts[payload] = {
            'aggregate_DRAM_read_bytes_saved': n[PHYSICAL[0]] - g[PHYSICAL[0]],
            'GENERIC_over_NONE_observed_read_ratio': g[PHYSICAL[0]] / n[PHYSICAL[0]],
            'aggregate_DRAM_read_reduction_pct': 100 * (1 - g[PHYSICAL[0]] / n[PHYSICAL[0]]),
            'L2_total_sectors_GEN_minus_NONE': g[L2[0]] - n[L2[0]],
            'L2_device_fill_sectors_GEN_minus_NONE': g[L2[1]] - n[L2[1]],
        }
    raw_ratio = contrasts['raw']['GENERIC_over_NONE_observed_read_ratio']
    return {'purpose': 'AGGREGATE_READER_CONTROLLED_COUNTERS_NO_SPEED_OR_WIRE_FACTOR', 'raw_units': {k: units[k] for k in METRICS},
            'cases': cases, 'within_payload': contrasts,
            'ratio_of_allocation_ratios_signlow_over_raw': (contrasts['signlow']['GENERIC_over_NONE_observed_read_ratio'] / raw_ratio
                                                          if raw_ratio else None),
            'attribution_limit': 'Same full-span copy controls reader. Aggregate memory-unit counters include output/cache/metadata; not whole-W or certified wire ratio. One ordered four-node observation has no CI.',
            'ILC_limit': 'Input/evaluated/output fields are insufficient to observe every read benefit; zero success is not no compression.'}


def validate_run(run):
    assert run['status'] == 'DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED'
    assert run['profile_order'] == list(LABELS) and run['allocation_spans_disjoint']
    assert run['postprofile_sources_immutable'] and run['cleanup']['status'] == 'CHECKED_CLOSE_PASS'
    assert len(run['cleanup']['counts_after_close']) == 4
    assert all(c['Region_owners'] == 0 and not c['tensor_defined'] for c in run['cleanup']['counts_after_close'])
    assert run['copy_identity']['kernel'] == KERNEL and run['copy_identity']['bytes'] == 67108864
    assert run['copy_resources']['localSizeBytes'] == 0 and run['copy_resources']['sharedSizeBytes'] == 0
    assert run['copy_resources']['occupancy_blocks_per_SM'] >= 1
    assert run['copy_resources']['numRegs'] == 32
    assert {k: run['jit_resources'][k] for k in ['registers', 'stack', 'shared', 'local']} == {
        'registers': 32, 'stack': 0, 'shared': 0, 'local': 0}
    assert set(run['payloads']) == {'raw', 'signlow'} and run['payloads']['signlow']['inverse_to_raw_exact']
    for field in ('checks', 'preprofile_checks', 'fresh_output_checks'):
        assert len(run[field]) == 4 and {r['case'] for r in run[field]} == set(LABELS)
        assert all(r['complement_poison_overwritten'] for r in run[field])
    assert all(r['stage'] == 'IMMEDIATELY_AFTER_SELECTED_ACTUAL_GRAPH' and r['full64MiB_bytes_exact']
               and r['input_immutable'] and not r['repair_replay'] for r in run['checks'])
    assert all(r['eager_and_actual_graph_bytes_exact'] and r['input_immutable'] for r in run['preprofile_checks'])
    assert all(r['full64MiB_bytes_exact'] for r in run['fresh_output_checks'])
    assert len(run['allocations']) == 4
    for label, allocation in zip(LABELS, run['allocations']):
        expected = int(label.endswith('_generic'))
        assert allocation['label'] == label and allocation['info']['actual_compression'] == expected
        assert allocation['properties']['compression'] == expected and allocation['properties']['retain_result'] == 0
        assert allocation['info']['requested_compression'] == expected and allocation['info']['bytes'] == 67108864
    assert set(run['outputs']) == set(LABELS)
    # Unknown ordinary-output flags remain unknown; do not manufacture NONE.
    for out in list(run['outputs'].values()) + run['fresh_output_checks']:
        prop = out.get('properties', out.get('output_properties'))
        assert prop['compression'] in (None, 0, 1)
        if prop['compression'] is None: assert prop['retain_result'] != 0


def validate_signlow_metadata(run):
    assert run['reader_manifest_sha256'] == '8e3d6016f388d23616fec4fe4055ceefee99b2cba550ef3f437e3f481b52321e'
    assert run['CPU_patterns_manifest_sha256'] == '780fabc881ef06f5f3559742a4ce3bdefed3f93423bc99cb628e0c581e230f9d'
    assert run['format'] == {'name': 'SIGNLOW_XOR16', 'BASE': 0x3c00, 'arithmetic_precision_change': False}
    assert run['payloads']['signlow']['GPU_full_payload_inverse_to_raw_exact']
    assert run['payloads']['raw']['sha256'] == '09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120'
    assert run['copy_identity']['source_sha256'] == 'd5b5e86f7801d518d8192a9397f952666466e08fef86af964334c7644ecaded5'
    assert run['copy_identity']['bindings_sha256'] == 'e391a11e6af0b3a1287d1a19a963111fb85a2f5748e0b8da69c07ecad846a0cb'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True, type=Path)
    ap.add_argument('--run-json', required=True, type=Path); ap.add_argument('--json', required=True, type=Path)
    args = ap.parse_args(); run = json.loads(args.run_json.read_text()); validate_run(run); validate_signlow_metadata(run)
    result = parse_text(args.csv.read_text(), run['copy_resources']['numRegs'])
    result['outputs_properties'] = run['outputs']
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
