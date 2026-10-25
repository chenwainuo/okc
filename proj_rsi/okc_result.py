"""Read okc job JSON into per-shape tables, and do the board arithmetic on them.

The score is the geomean over six loads of SGLang load time over ours. In log space
that is the mean of six ln(ratio)s, so a shape's leverage is d(ln score)/d(yours_us)
= 1 / (6 * load_sum_us): one microsecond on a decode load is worth about 0.4% of
the score; on verify 32 about 0.2%.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

from .score import LAYERS, LOADS

ROOT = Path(__file__).resolve().parents[1]
LOAD_ORDER = [load for load, _ in LOADS]
SGLANG_US = 4830.8


def load_job(path_or_id: str | Path) -> dict:
    """A job JSON, from a path or by job id (searched under candidates/*/profiles and results/proj-rsi/jobs)."""
    path = Path(path_or_id)
    if path.is_file():
        return json.loads(path.read_text())
    job = str(path_or_id)
    for pattern in (f"candidates/*/profiles/*{job}*.json", f"../results/proj-rsi/jobs/{job}.json"):
        for found in sorted((ROOT / "proj_rsi").glob(pattern)):
            return json.loads(found.read_text())
    raise FileNotFoundError(f"no job JSON for {job}")


def shapes(job: dict) -> dict[tuple[str, str], dict]:
    """(load, layer) -> {'yours', 'sglang', 'sol', 'speedup', 'm'} in microseconds."""
    out = {}
    for bench in job.get("benchmarks", []):
        case = bench["case"]
        key = (case["load"], case["layer"])
        out[key] = {
            "m": case["m"],
            "yours": bench["median_ns"] / 1e3,
            "sglang": bench["baseline_ns"] / 1e3,
            "sol": bench.get("speed_of_light_ns", float("nan")) / 1e3,
            "speedup": bench.get("speedup", float("nan")),
            "status": bench.get("status"),
        }
    return out


def profiles(job: dict) -> dict[tuple[str, str], str]:
    return {(p["case"]["load"], p["case"]["layer"]): p["report"] for p in job.get("profiles", [])}


def load_sums(table: dict[tuple[str, str], dict], field: str = "yours") -> dict[str, float]:
    sums = {}
    for load in LOAD_ORDER:
        values = [table[(load, layer)][field] for layer in LAYERS if (load, layer) in table]
        if len(values) == len(LAYERS):
            sums[load] = sum(values)
    return sums


def speedup_from_table(table: dict[tuple[str, str], dict]) -> float | None:
    ours, theirs = load_sums(table, "yours"), load_sums(table, "sglang")
    if len(ours) != 6 or len(theirs) != 6:
        return None
    return math.exp(sum(math.log(theirs[l] / ours[l]) for l in LOAD_ORDER) / 6)


def score_us(speedup: float) -> float:
    return SGLANG_US / speedup


def ln_gain_needed(current: float, target_us: float = 3000.0) -> float:
    """Sum over loads of the ln-ratio improvement needed to reach target_us."""
    return 6 * (math.log(SGLANG_US / target_us) - math.log(current))


def stitch(tables: Iterable[dict[tuple[str, str], dict]]) -> dict[tuple[str, str], dict]:
    """Best yours per shape across jobs, with the sglang column taken from the same job as that best
    (shapes are judged as paired ratios, so mixing sglang columns across jobs would be optimistic)."""
    best: dict[tuple[str, str], dict] = {}
    for table in tables:
        for key, row in table.items():
            if row.get("status") not in (None, "pass"):
                continue
            if key not in best or row["yours"] < best[key]["yours"]:
                best[key] = dict(row)
    return best


def whatif(table: dict[tuple[str, str], dict], overrides: dict[tuple[str, str], float]) -> dict:
    """Project the score if yours_us on the given shapes became the override values."""
    edited = {k: dict(v) for k, v in table.items()}
    for key, value in overrides.items():
        edited[key]["yours"] = value
    before, after = speedup_from_table(table), speedup_from_table(edited)
    return {"before_us": score_us(before), "after_us": score_us(after), "before": before, "after": after}


def leverage(table: dict[tuple[str, str], dict]) -> dict[str, float]:
    """Score microseconds gained per microsecond cut on one shape of each load."""
    sums = load_sums(table)
    speed = speedup_from_table(table)
    return {load: score_us(speed) / (6 * sums[load]) for load in sums}


def format_table(table: dict[tuple[str, str], dict], against: dict[tuple[str, str], dict] | None = None) -> str:
    lines = ["%-10s %-9s %4s %8s %8s %7s %8s %s" % ("load", "layer", "m", "yours", "sglang", "sol", "speedup", "vs incumbent")]
    for load in LOAD_ORDER:
        for layer in LAYERS:
            row = table.get((load, layer))
            if row is None:
                continue
            delta = ""
            if against and (load, layer) in against:
                d = row["yours"] - against[(load, layer)]["yours"]
                delta = "%+6.2f %s" % (d, "WIN" if d < -0.05 else "loss" if d > 0.05 else "tie")
            lines.append("%-10s %-9s %4d %8.2f %8.2f %7.2f %7.3fx %s" % (
                load, layer, row["m"], row["yours"], row["sglang"], row["sol"], row["speedup"], delta))
        if load in load_sums(table):
            sums_y, sums_s = load_sums(table)[load], load_sums(table, "sglang")[load]
            extra = ""
            if against and load in load_sums(against):
                extra = "  (incumbent %.2f)" % load_sums(against)[load]
            lines.append("%-10s %-9s %4s %8.2f %8.2f %7s %7.3fx%s" % (load, "SUM", "", sums_y, sums_s, "", sums_s / sums_y, extra))
    speed = speedup_from_table(table)
    if speed:
        lines.append("speedup %.4fx  score %.1f us  ln-gain still needed for 3000: %.3f" % (
            speed, score_us(speed), ln_gain_needed(speed)))
    return "\n".join(lines)


BAKE_LINE = re.compile(r"^BAKE\s+(\S+)\s+m=(\d+)\s+(\S+)\s+([0-9.]+)")


def parse_bake(log_text: str) -> list[dict]:
    """'BAKE <layer> m=<m> <path> <us>' lines printed by prepare()."""
    rows = []
    for line in log_text.splitlines():
        match = BAKE_LINE.match(line.strip())
        if match:
            rows.append({"layer": match.group(1), "m": int(match.group(2)), "path": match.group(3), "us": float(match.group(4))})
    return rows


PROBE_LINE = re.compile(r"^PROBE\s+(\S+)\s+(.*?)\s+([0-9.]+)(?:\s+([0-9.]+))?\s*$")


def parse_probe(log_text: str) -> list[dict]:
    """'PROBE <engine> <config> <us> [<TB/s>]' lines printed by prepare()'s streaming probe."""
    rows = []
    for line in log_text.splitlines():
        match = PROBE_LINE.match(line.strip())
        if match:
            rows.append({"engine": match.group(1), "config": match.group(2), "us": float(match.group(3)),
                         "tbps": float(match.group(4)) if match.group(4) else None})
    return rows
