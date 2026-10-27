"""The executor: quota guard, okc check / submit, wait, archive, record.

okc writes profiles/submission.<job>.{json,txt,log} beside the submitted file. This
module copies the JSON into results/proj-rsi/jobs/ as well so job ids resolve
without knowing the candidate directory.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .ledger import ROOT, Ledger
from .okc_result import format_table, load_sums, parse_bake, parse_probe, profiles, shapes, speedup_from_table
from .static_check import check_file

JOBS = ROOT / "results" / "proj-rsi" / "jobs"
PROBLEM = "muse_bf16_proj"


class OkcResponseError(RuntimeError):
    """Retain malformed CLI output for submission recovery, without retrying."""

    def __init__(self, message, proc):
        super().__init__(message)
        self.raw_response = {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


class SubmissionRecoveryError(RuntimeError):
    """An external request was issued: inspect the receipt, never blindly retry."""


def okc(*args: str, timeout: int = 900) -> dict:
    cmd = ["okc", *args, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise OkcResponseError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}", proc)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise OkcResponseError(f"{' '.join(cmd)} returned non-JSON: {proc.stdout[:400]} {proc.stderr[:400]}", proc) from None


def _atomic_receipt(path: Path, data: dict) -> None:
    """Commit JSON with file fsync, atomic replacement and directory fsync."""
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _job_id(response) -> str | None:
    if not isinstance(response, dict):
        return None
    value = response.get("job_id") or response.get("job") or response.get("id")
    # A returned identifier must never control receipt paths through traversal.
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) else None


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
    """Validate locally, submit once, durably receipt, then attach ledger metadata.

    Once the external call starts, failure is not permission to resubmit. The
    unique attempt receipt survives ledger failures; the legacy job-named receipt
    is also retained. Neither error path automatically repeats the request.
    """
    path = candidate_file(candidate)
    ledger = Ledger()
    if experiment:
        # This used to run only after an accepted external job, losing its ID
        # when a misspelled/unregistered experiment failed attach_job().
        ledger._get(ledger.load(), experiment)
    try:
        candidate_name = str(path.relative_to(ROOT))
    except ValueError:
        candidate_name = str(path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    lint = check_file(path)
    if not lint["ok"]:
        raise RuntimeError("lint failed: " + json.dumps(lint["findings"]))
    limits = quota()
    left = limits["checks" if kind == "check" else "scored"]["left"]
    if left <= reserve:
        raise RuntimeError(f"{kind} quota: {left} left, reserve {reserve}; wait for 00:00 UTC")
    verb = "check" if kind == "check" else "submit"
    JOBS.mkdir(parents=True, exist_ok=True)
    attempt = uuid.uuid4().hex
    intent_path = JOBS / f"attempt-{attempt}.intent.json"
    receipt_path = JOBS / f"attempt-{attempt}.response.json"
    metadata = {
        "receipt_schema_version": 1, "attempt_id": attempt, "problem": PROBLEM,
        "job": None, "kind": kind, "verb": verb,
        "candidate": candidate_name, "source_sha256": source_sha256, "experiment": experiment,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quota_before": limits, "quota_bucket": "checks" if kind == "check" else "scored",
    }
    # Establish a writable, durable recovery location BEFORE the external call.
    _atomic_receipt(intent_path, {**metadata, "status": "submission_intent",
                                 "recovery_note": "An intent alone does not prove whether the external request was issued; never retry blindly."})
    try:
        result = okc(verb, PROBLEM, str(path), "--no-wait")
    except Exception as error:
        response = getattr(error, "raw_response", None)
        try:
            _atomic_receipt(receipt_path, {**metadata, "status": "submission_outcome_unknown", "submit_response": response})
        except Exception:
            receipt_path = intent_path
        raise SubmissionRecoveryError(
            f"Submission outcome unknown; recovery receipt: {receipt_path}. "
            "The request may already have been accepted. Never retry blindly; recover its existing job ID first."
        ) from error
    job_id = _job_id(result)
    receipt = {**metadata, "job": job_id, "response_received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "status": "accepted" if job_id else "submission_outcome_unknown", "submit_response": result}
    try:
        # Do this before ANY ledger write or fallible legacy bookkeeping. The
        # unique receipt is never overwritten by another attempt with the same ID.
        _atomic_receipt(receipt_path, receipt)
        if not job_id:
            raise SubmissionRecoveryError(
                f"Submission response has no safe job ID; recovery receipt: {receipt_path}. "
                "The request may already have been accepted. Never retry blindly; recover its existing job ID first."
            )
        _atomic_receipt(JOBS / f"{job_id}.submitted.json", receipt)
        if experiment:
            ledger.attach_job(experiment, job_id, kind)
    except SubmissionRecoveryError:
        raise
    except Exception as error:
        identity = f"Accepted job {job_id}" if job_id else "Submission outcome unknown"
        raise SubmissionRecoveryError(
            f"{identity}; local bookkeeping failed. Recovery receipt: {receipt_path} "
            f"(pre-submit intent: {intent_path}). Never retry blindly; recover or wait for the existing job instead."
        ) from error
    return job_id


def wait(job_id: str, timeout: int = 1200, candidate: Path | None = None) -> dict:
    """Follow the job; return the normalized job JSON (benchmarks/profiles/summary).

    `okc status --json` is a wrapper (result/output/profile); okc also saves the normalized form beside the
    submitted file as profiles/submission.<job>.json, which is what the parser reads. Prefer that file."""
    status = okc("status", job_id, "--wait", timeout=timeout)
    if candidate is not None:
        saved = candidate.parent / "profiles" / f"submission.{job_id}.json"
        if saved.exists():
            return json.loads(saved.read_text())
    profile = status.get("profile") or {}
    if isinstance(profile.get("json"), dict) and "benchmarks" in profile["json"]:
        return profile["json"]
    result = status.get("result") or {}
    return {"job_id": job_id, "problem": status.get("board"), "benchmarks": [], "profiles": [],
            "summary": {"status": status.get("status"), "passed": result.get("passed"), "error": result.get("error"),
                        "score_us": result.get("score_us"), "speedup": None, "speedup_ci": None,
                        "tests": None, "tests_passed": None}}


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
    probe_lines = ["PROBE %-8s %-24s %8.2f us %s" % (b["engine"], b["config"], b["us"], ("%.2f TB/s" % b["tbps"]) if b["tbps"] else "")
                   for b in parse_probe(log_text)]
    return "\n".join(head + [body, "", "profiles:"] + kernels + ([""] + probe_lines if probe_lines else [])
                     + ([""] + bake_lines if bake_lines else []))


def run(candidate: str | Path, kind: str, experiment: str | None, reserve: int, against_job: str | None) -> str:
    path = candidate_file(candidate)
    job_id = submit(path, kind, experiment, reserve)
    print(f"submitted {kind} {job_id} for {path.relative_to(ROOT)}", flush=True)
    job = wait(job_id, candidate=path)
    archive(job, path)
    record(job, experiment, kind)
    log_path = path.parent / "profiles" / f"submission.{job_id}.log"
    log_text = log_path.read_text() if log_path.exists() else ""
    against = None
    if against_job:
        from .okc_result import load_job
        against = shapes(load_job(against_job))
    return report(job, against, log_text)
