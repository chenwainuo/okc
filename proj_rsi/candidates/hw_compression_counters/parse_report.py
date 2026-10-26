"""CPU-only raw NCU parser. Counts, launch/NVTX identity and units; never a latency/speedup verdict."""
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
LAUNCH = ["launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z",
          "launch__block_dim_x", "launch__block_dim_y", "launch__block_dim_z",
          "launch__registers_per_thread", "launch__shared_mem_per_block"]
METRICS = PHYSICAL + ILC + L2 + LAUNCH
KERNEL = "nvjet_tst_128x64_64x8_1x1_v_bz_coopB_NNN"


def number(value):
    try:
        x = float(value.replace(",", "").strip())
        return x if math.isfinite(x) else None
    except (ValueError, AttributeError):
        return None


def parse_text(text):
    lines = text.lstrip("\ufeff").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    units, rows = None, []
    for row in reader:
        if not row.get("ID"):
            if units is None:
                units = row
            continue
        if row["ID"].isdigit():
            rows.append(row)
    assert units is not None and len(rows) == 2, "need exactly two graph-node results plus units row"
    assert all(name in rows[0] for name in METRICS), "requested raw metric missing"
    assert units["dram__bytes_read.sum"] == "byte" and units["dram__sectors_read.sum"] == "sector", "non-base units"
    cases = {}
    for row in rows:
        nvtx = " ".join(value or "" for key, value in row.items() if "Domain:" in key)
        matched = [name for name in ("compression_generic", "compression_none") if ":" + name + ":" in nvtx]
        assert len(matched) == 1 and matched[0] not in cases, "missing/duplicate case NVTX identity"
        assert row["Kernel Name"] == KERNEL, "changed shader identity"
        values = {name: number(row[name]) for name in METRICS}
        assert all(values[name] is not None and values[name] >= 0 for name in PHYSICAL + L2 + LAUNCH)
        assert [values[n] for n in LAUNCH[:3]] == [1, 128, 1]
        assert [values[n] for n in LAUNCH[3:6]] == [384, 1, 1]
        assert values["launch__registers_per_thread"] == 168
        for direction in ("read", "write"):
            assert math.isclose(values[f"dram__bytes_{direction}.sum"], 32 * values[f"dram__sectors_{direction}.sum"],
                                rel_tol=1e-6, abs_tol=.01), "DRAM byte/sector inconsistency"
        cases[matched[0]] = {"ID": row["ID"], "kernel": row["Kernel Name"], "values": values}
    g, n = cases["compression_generic"]["values"], cases["compression_none"]["values"]
    assert all(g[k] == n[k] for k in LAUNCH), "allocation control launch/resource identity changed"
    assert n["dram__bytes_read.sum"] > 0
    result = {"purpose": "AGGREGATE_TRAFFIC_OBSERVATION_ONLY_NO_SPEED_CLAIM", "cases": cases,
              "observed_DRAM_read_bytes_saved": n["dram__bytes_read.sum"] - g["dram__bytes_read.sum"],
              "observed_DRAM_read_reduction_pct": 100 * (1 - g["dram__bytes_read.sum"] / n["dram__bytes_read.sum"]),
              "L2_requested_sectors_delta": g["lts__t_sectors.sum"] - n["lts__t_sectors.sum"],
              "L2_hit_rate_pct_point_delta": g["lts__t_sector_hit_rate.pct"] - n["lts__t_sector_hit_rate.pct"],
              "ILC_counter_limit": "ILC input/output are not documented here as a dedicated weight decompression path. Zero during read-only Lt is not proof that population never compressed.",
              "attribution_limit": "All-kernel aggregate includes X/output/cache/metadata. One allocation pair is not a confidence interval or a whole-W compression ratio.",
              "runtime_unavailable_ILC_metrics": [k for k in ILC if g[k] is None or n[k] is None]}
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--run-json", required=True, type=Path); ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()
    run = json.loads(args.run_json.read_text())
    assert run["status"] == "DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED"
    assert run["cleanup"]["status"] == "CHECKED_CLOSE_PASS" and len(run["checks"]) == 2
    assert run["fixed_lt"]["layout"] == 2 and run["fixed_lt"]["cfg"] == [66, 18, 1, 0, 0, 4, 35, 0, 2]
    parsed = parse_text(args.csv.read_text())
    parsed["run_status"] = run["status"]
    args.json.write_text(json.dumps(parsed, indent=2) + "\n")
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__": main()
