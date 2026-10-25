"""Append-only experiment ledger for the muse_bf16_proj loop.

One JSON file. Every speed number in it carries an okc job id. Actor names are
labels for coordination, not authentication.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "results" / "proj-rsi" / "ledger.json"
STATES = ("proposed", "reviewed", "checking", "measured", "promoted", "rejected")


class Ledger:
    def __init__(self, path: Path = DEFAULT_LEDGER):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "problem": "muse_bf16_proj",
                "incumbent": None,
                "experiments": {},
                "jobs": {},
            }
        data = json.loads(self.path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"ledger schema {data.get('schema_version')} != {SCHEMA_VERSION}; move it aside")
        return data

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.path)

    def set_incumbent(self, job_id: str, candidate: str, score_us: float, speedup: float, ci: list[float], board: bool) -> dict:
        data = self.load()
        data["incumbent"] = {
            "job": job_id, "candidate": candidate, "score_us": score_us, "speedup": speedup,
            "ci": ci, "on_board": board, "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.save(data)
        return data["incumbent"]

    def propose(self, experiment_id: str, actor: str, candidate: str, hypothesis: dict) -> dict:
        data = self.load()
        if experiment_id in data["experiments"]:
            raise ValueError(f"experiment {experiment_id} exists")
        for key in ("claim", "target_shapes", "kill_if"):
            if key not in hypothesis:
                raise ValueError(f"hypothesis missing {key}")
        if not (ROOT / candidate).is_file():
            raise ValueError(f"candidate file missing: {candidate}")
        data["experiments"][experiment_id] = {
            "id": experiment_id, "state": "proposed", "author": actor, "candidate": candidate,
            "hypothesis": hypothesis, "review": None, "jobs": [], "decision": None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.save(data)
        return {"id": experiment_id, "state": "proposed"}

    def review(self, experiment_id: str, actor: str, approved: bool, notes: str) -> dict:
        data = self.load()
        exp = self._get(data, experiment_id)
        if actor == exp["author"]:
            raise ValueError("critic must differ from the author")
        exp["review"] = {"actor": actor, "approved": approved, "notes": notes}
        exp["state"] = "reviewed" if approved else "rejected"
        if not approved:
            exp["decision"] = {"action": "reject", "actor": actor, "reason": notes}
        self.save(data)
        return {"id": experiment_id, "state": exp["state"]}

    def attach_job(self, experiment_id: str, job_id: str, kind: str) -> None:
        data = self.load()
        exp = self._get(data, experiment_id)
        exp["jobs"].append({"job": job_id, "kind": kind, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        exp["state"] = "checking"
        self.save(data)

    def record_job(self, experiment_id: str | None, job_id: str, summary: dict, load_sums: dict, kind: str) -> None:
        data = self.load()
        data["jobs"][job_id] = {
            "experiment": experiment_id, "kind": kind, "status": summary.get("status"), "passed": summary.get("passed"),
            "score_us": summary.get("score_us"), "speedup": summary.get("speedup"), "ci": summary.get("speedup_ci"),
            "tests": f"{summary.get('tests_passed')}/{summary.get('tests')}", "error": summary.get("error"),
            "load_sums_us": load_sums, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if experiment_id and experiment_id in data["experiments"]:
            exp = data["experiments"][experiment_id]
            exp["state"] = "measured" if summary.get("passed") else "rejected"
            if not summary.get("passed"):
                exp["decision"] = {"action": "reject", "actor": "judge", "reason": summary.get("error") or "run failed"}
        self.save(data)

    def decide(self, experiment_id: str, actor: str, action: str, reason: str) -> dict:
        data = self.load()
        exp = self._get(data, experiment_id)
        if action not in ("promote", "reject", "refine"):
            raise ValueError(action)
        exp["decision"] = {"action": action, "actor": actor, "reason": reason}
        if action == "promote":
            exp["state"] = "promoted"
        elif action == "reject":
            exp["state"] = "rejected"
        self.save(data)
        return {"id": experiment_id, "state": exp["state"]}

    def status(self) -> dict:
        data = self.load()
        return {
            "ledger": str(self.path),
            "incumbent": data["incumbent"],
            "experiments": [
                {"id": e["id"], "state": e["state"], "candidate": e["candidate"], "jobs": [j["job"] for j in e["jobs"]],
                 "claim": e["hypothesis"]["claim"][:120]}
                for e in data["experiments"].values()
            ],
            "jobs": {k: {"score_us": v["score_us"], "passed": v["passed"], "experiment": v["experiment"]} for k, v in data["jobs"].items()},
        }

    @staticmethod
    def _get(data: dict, experiment_id: str) -> dict:
        if experiment_id not in data["experiments"]:
            raise ValueError(f"unknown experiment {experiment_id}")
        return data["experiments"][experiment_id]
