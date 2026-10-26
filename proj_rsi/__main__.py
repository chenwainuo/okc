"""python3 -m proj_rsi <command>: the head's tools for the muse_bf16_proj loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .correctness import OFFICIAL_CASES, SMOKE_CASES, check_submission
from .ledger import ROOT, Ledger
from .okc_result import (format_table, leverage, ln_gain_needed, load_job, load_sums, parse_bake, score_us, shapes,
                         speedup_from_table, stitch, whatif)
from .static_check import check_file


def main() -> None:
    parser = argparse.ArgumentParser(description="muse_bf16_proj RSI loop tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    p = sub.add_parser("lint"); p.add_argument("file", type=Path)
    p = sub.add_parser("cpu-check"); p.add_argument("file", type=Path); p.add_argument("--official", action="store_true")
    p = sub.add_parser("result"); p.add_argument("job"); p.add_argument("--against", default=None); p.add_argument("--log", type=Path, default=None)
    p = sub.add_parser("board"); p.add_argument("jobs", nargs="*"); p.add_argument("--set", nargs="*", default=[], help="load/layer=us overrides")
    p = sub.add_parser("check"); p.add_argument("candidate"); p.add_argument("--experiment", default=None); p.add_argument("--reserve", type=int, default=0); p.add_argument("--against", default=None); p.add_argument("--scored", action="store_true")
    p = sub.add_parser("wait"); p.add_argument("job"); p.add_argument("--experiment", default=None); p.add_argument("--kind", default="check"); p.add_argument("--candidate", default=None); p.add_argument("--against", default=None)
    p = sub.add_parser("propose"); p.add_argument("id"); p.add_argument("--actor", required=True); p.add_argument("--candidate", required=True); p.add_argument("--hypothesis", type=Path, required=True)
    p = sub.add_parser("review"); p.add_argument("id"); p.add_argument("--actor", required=True); p.add_argument("--approve", action="store_true"); p.add_argument("--notes", required=True)
    p = sub.add_parser("decide"); p.add_argument("id"); p.add_argument("--actor", required=True); p.add_argument("--action", required=True); p.add_argument("--reason", required=True)
    p = sub.add_parser("incumbent"); p.add_argument("job"); p.add_argument("--candidate", required=True); p.add_argument("--board", action="store_true")
    sub.add_parser("quota")
    args = parser.parse_args()
    ledger = Ledger()

    if args.command == "status":
        print(json.dumps(ledger.status(), indent=2))
    elif args.command == "lint":
        result = check_file(args.file)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ok"] else 1)
    elif args.command == "cpu-check":
        print(json.dumps(check_submission(args.file, cases=OFFICIAL_CASES if args.official else SMOKE_CASES), indent=2))
    elif args.command == "result":
        job = load_job(args.job)
        against = shapes(load_job(args.against)) if args.against else None
        print(format_table(shapes(job), against))
        if args.log and args.log.exists():
            for b in parse_bake(args.log.read_text()):
                print("BAKE %-9s m=%-4d %-18s %8.2f" % (b["layer"], b["m"], b["path"], b["us"]))
    elif args.command == "board":
        jobs = args.jobs or ([ledger.load()["incumbent"]["job"]] if ledger.load()["incumbent"] else [])
        tables = [shapes(load_job(j)) for j in jobs]
        table = stitch(tables) if len(tables) > 1 else tables[0]
        overrides = {}
        for item in args.set:
            key, value = item.split("=")
            load, layer = key.split("/")
            overrides[(load.replace("_", " "), layer)] = float(value)
        print(format_table(table))
        print("leverage (score us per us cut on one shape):", {k: round(v, 1) for k, v in leverage(table).items()})
        if overrides:
            print("what-if:", whatif(table, overrides))
    elif args.command == "check":
        from .runner import run
        print(run(args.candidate, "scored" if args.scored else "check", args.experiment, args.reserve, args.against))
    elif args.command == "wait":
        from .runner import archive, candidate_file, record, report, wait
        cand = candidate_file(args.candidate) if args.candidate else None
        job = wait(args.job, candidate=cand)
        archive(job, cand)
        record(job, args.experiment, args.kind)
        log_text = ""
        if cand is not None:
            lp = cand.parent / "profiles" / f"submission.{job['job_id']}.log"
            log_text = lp.read_text() if lp.exists() else ""
        against = shapes(load_job(args.against)) if args.against else None
        print(report(job, against, log_text))
    elif args.command == "propose":
        print(json.dumps(ledger.propose(args.id, args.actor, args.candidate, json.loads(args.hypothesis.read_text())), indent=2))
    elif args.command == "review":
        print(json.dumps(ledger.review(args.id, args.actor, args.approve, args.notes), indent=2))
    elif args.command == "decide":
        print(json.dumps(ledger.decide(args.id, args.actor, args.action, args.reason), indent=2))
    elif args.command == "incumbent":
        job = load_job(args.job)
        s = job["summary"]
        print(json.dumps(ledger.set_incumbent(job["job_id"], args.candidate, s["score_us"], s["speedup"], s["speedup_ci"], args.board), indent=2))
    elif args.command == "quota":
        from .runner import quota
        print(json.dumps(quota(), indent=2))


if __name__ == "__main__":
    main()
