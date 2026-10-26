"""Reuse the frozen real-counter parser; retain known-zero calibration scope."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parent / "hw_compression_counters"


def parser_module():
    path = REAL / "parse_report.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "95091dae6d46db4426cff213ab37e656f3c49799fcd5d51e9ee75eb412ae3575"
    spec = importlib.util.spec_from_file_location("hw_zero_frozen_raw_parser", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--run-json", required=True, type=Path); ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()
    run = json.loads(args.run_json.read_text())
    assert run["status"] == "DIAGNOSTIC_COMPLETE_COUNTERS_UNINTERPRETED"
    assert run["control_payload"] == "KNOWN_ZERO_BF16_WT" and not run["normal_weight_rerun"]
    assert run["cleanup"]["status"] == "CHECKED_CLOSE_PASS" and len(run["checks"]) == 2
    assert all(c["complete_output"] == "EXACT_ZERO" and c["NaN_poison_overwritten"] for c in run["checks"])
    assert run["fixed_lt"]["layout"] == 2 and run["fixed_lt"]["cfg"] == [66, 18, 1, 0, 0, 4, 35, 0, 2]
    result = parser_module().parse_text(args.csv.read_text())
    result["control_payload"] = "KNOWN_ZERO_BF16_WT"
    result["interpretation"] = "observer/path responsiveness control; flat stays ambiguous; not challenge-weight or latency evidence"
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
