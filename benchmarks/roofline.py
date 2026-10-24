#!/usr/bin/env python3
"""Optimistic expert-weight streaming estimate; not measured HBM traffic."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('results/selected-validation.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/2026-10-24-weight-roofline.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    # gate+up: (2*4096,4096); down: (4096,4096); 4-bit values.
    packed_bytes_per_expert = 3 * 4096 * 4096 // 2
    peak_bytes_per_second = 4.8e12
    rows = []
    for row in report['results']:
        counts = [r['distinct_local_experts'] for r in row['routing_samples']]
        ideal_us = statistics.median(counts) * packed_bytes_per_expert / peak_bytes_per_second * 1e6
        rows.append(dict(bs=row['bs'],median_active_experts=statistics.median(counts),
                         incumbent_us=row['candidate_us'],ideal_weight_streaming_us=ideal_us,
                         optimistic_speedup_ceiling_under_assumptions=row['candidate_us']/ideal_us))
    result = dict(schema_version=1,source_report=str(args.report),
                  hardware_source='https://www.nvidia.com/en-au/data-center/h200/',
                  peak_hbm_bytes_per_second=peak_bytes_per_second,
                  assumptions=['Each active dense expert reads its packed weights once from HBM per invocation',
                               'Cold-cache benchmark; estimate ignores residual cached weight bytes',
                               'Assume peak HBM bandwidth; ignore scales, activation traffic, compute, synchronization and communication',
                               'This is a model of component headroom, not measured traffic or a proof about full serving'],
                  packed_bytes_per_expert=packed_bytes_per_expert,results=rows,
                  implication='If large-shape weight streaming is dominant, 3x overall will require serving-level reuse, overlap, or fewer equivalent iterations as well as kernel work; preserve exact model and draft semantics.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
