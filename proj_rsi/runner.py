"""The executor: quota guard, okc check / submit, wait, archive, record.

okc writes profiles/submission.<job>.{json,txt,log} beside the submitted file. This
module copies the JSON into results/proj-rsi/jobs/ as well so job ids resolve
without knowing the candidate directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .ledger import ROOT, Ledger
from .okc_result import format_table, load_sums, parse_bake, profiles, shapes, speedup_from_table
from .static_check import check_file

JOBS = ROOT / "results" / "proj-rsi" / "jobs"
PROBLEM = "muse_bf16_proj"


def okc(*args: str, timeout: int = 900) -> dict:
    cmd = ["okc", *args, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{' '.join(cmd)} returned non-JSON: {proc.stdout[:400]} {proc.stderr[:400]}")


def quota() -> dict:
    for item in okc("problems")["problems"]:
        if item["name"] == PROBLEM:
            return item["limits"]
    raise RuntimeError("problem not listed")


def candidate_file(candidate: str | Path) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path = path / "submission.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def submit(candidate: str | Path, kind: str = "check", experiment: str | None = None, reserve: int = 0) -> str:
    """Lint, quota-check, submit with --no-wait, record the job id. Returns the job id."""
    path = candidate_file(candidate)
    lint = check_file(path)
    if not lint["ok"]:
        raise RuntimeError("lint failed: " + json.dumps(lint["findings"]))
    limits = quota()
    left = limits["checks" if kind == "check" else "scored"]["left"]
    if left <= reserve:
        raise RuntimeError(f"{kind} quota: {left} left, reserve {reserve}; wait for 00:00 UTC")
    verb = "check" if kind == "check" else "submit"
    result = okc(verb, PROBLEM, str(path), "--no-wait")
    job_id = result.get("job_id") or result.get("job") or result.get("id")
    if not job_id:
        raise RuntimeError(f"no job id in {result}")
    ledger = Ledger()
    if experiment:
        ledger.attach_job(experiment, job_id, kind)
    JOBS.mkdir(parents=True, exist_ok=True)
    (JOBS / f"{job_id}.submitted.json").write_text(json.dumps({
        "job": job_id, "kind": kind, "candidate": str(path.relative_to(ROOT)), "experiment": experiment,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "quota_before": limits}, indent=2))
    return job_id


def wait(job_id: str, timeout: int = 1200) -> dict:
    return okc("status", job_id, "--wait", timeout=timeout)


def archive(job: dict, candidate: Path | None = None) -> Path:
    JOBS.mkdir(parents=True, exist_ok=True)
    out = JOBS / f"{job['job_id']}.json"
    out.write_text(json.dumps(job, indent=2))
    if candidate is not None:
        prof = candidate.parent / "profiles"
        for ext in ("json", "txt", "log"):
            src = prof / f"submission.{job['job_id']}.{ext}"
            if src.exists():
                shutil.copy(src, JOBS / f"{job['job_id']}.{ext}")
    return out


def record(job: dict, experiment: str | None, kind: str) -> dict:
    table = shapes(job)
    sums = load_sums(table)
    Ledger().record_job(experiment, job["job_id"], job["summary"], sums, kind)
    return {"job": job["job_id"], "summary": job["summary"], "load_sums_us": sums, "speedup": speedup_from_table(table)}


def report(job: dict, against: dict | None = None, log_text: str = "") -> str:
    table = shapes(job)
    summary = job["summary"]
    head = ["okc job %s  status=%s passed=%s tests=%s/%s score=%s speedup=%s ci=%s" % (
        job["job_id"], summary.get("status"), summary.get("passed"), summary.get("tests_passed"), summary.get("tests"),
        summary.get("score_us"), summary.get("speedup"), summary.get("speedup_ci"))]
    if summary.get("error"):
        head.append("error: %s" % summary["error"])
    body = format_table(table, against) if table else "(no timed shapes)"
    prof = profiles(job)
    kernels = []
    for key, text in prof.items():
        top = [l for l in text.splitlines()[1:] if l.strip() and not l.startswith("total")][:2]
        kernels.append("%-10s %-9s | " % key + " || ".join(t.strip() for t in top))
    bake = parse_bake(log_text)
    bake_lines = ["BAKE %-9s m=%-4d %-18s %8.2f" % (b["layer"], b["m"], b["path"], b["us"]) for b in bake]
    return "\n".join(head + [body, "", "profiles:"] + kernels + ([""] + bake_lines if bake_lines else []))


def run(candidate: str | Path, kind: str, experiment: str | None, reserve: int, against_job: str | None) -> str:
    path = candidate_file(candidate)
    job_id = submit(path, kind, experiment, reserve)
    print(f"submitted {kind} {job_id} for {path.relative_to(ROOT)}", flush=True)
    job = wait(job_id)
    archive(job, path)
    record(job, experiment, kind)
    log_path = path.parent / "profiles" / f"submission.{job_id}.log"
    log_text = log_path.read_text() if log_path.exists() else ""
    against = None
    if against_job:
        from .okc_result import load_job
        against = shapes(load_job(against_job))
    return report(job, against, log_text)
