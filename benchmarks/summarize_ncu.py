#!/usr/bin/env python3
"""Normalize Nsight Compute CSV metrics without treating them as clean timings."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.reader(args.csv.open()))
    def number(value):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return value
    index = next(i for i,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r)
    header = rows[index]
    kernels = {}
    if 'Metric Name' in header:
        for values in rows[index+1:]:
            if len(values) != len(header):
                continue
            row = dict(zip(header,values))
            kernel = kernels.setdefault(row['ID'], dict(name=row['Kernel Name'], metrics={}))
            kernel['metrics'][row['Metric Name']] = dict(value=number(row['Metric Value']),unit=row.get('Metric Unit'),section=row.get('Section Name'))
    else:
        units = rows[index+1]
        for values in rows[index+2:]:
            if len(values) != len(header):
                continue
            row = dict(zip(header,values))
            kernels[row['ID']] = dict(name=row['Kernel Name'],metrics={
                name:dict(value=number(value),unit=unit) for name,value,unit in zip(header[11:],values[11:],units[11:])})
    if not kernels:
        raise SystemExit('No kernel metrics; inspect profiler permissions/logs')
    args.output.write_text(json.dumps(dict(source=str(args.csv),scope='hardware counters; profiler timing excluded from speedup claims',kernels=kernels),indent=2)+'\n')


if __name__ == '__main__':
    main()
