# muse_bf16_proj — handover (written 2026-10-26 19:10 UTC by the Fable head session)

> **Codex takeover, 2026-10-26:** Read `12-codex-takeover.json` first. Live board verified rs32 at **3200.615 µs / 1.5093×, rank 81**. pw36 passed the clean parent comparison and is frozen for a check at the 00:00 UTC reset. The GPU lock now lives inside the container; old outer-lock instructions below are superseded.

Goal: get the okc `muse_bf16_proj` board under **3000 µs** (speedup ≥ 1.6103×). Board now: **3278.6 µs / 1.4734×, rank 111**
(pw35 = pw34 + rtk router, scored job 5d880cd0aa504209, 19:02 UTC; previous pw34 846e25d9df09433f 3293.9). Leader 2951.5 (1.637×); top-10 cut 3121.5 (1.548×). Track closes 2026-11-04 23:59 PT.
Owner: jianc65@gmail.com. Roles the user set: head = Fable 5.1 (high/max effort), implementers & critics = Opus 5.5 medium via
the Agent tool. The user wants the loop to "follow through", a complete kernel rewrite tailored to the model's shapes,
step-back ceiling checks before drilling, the H100 box kept busy, and progress every day. Subagents do NOT survive a session
restart: re-brief fresh agents from these files (exact briefs in `11-agent-briefs.md`).

## Files in this folder
| file | what |
|---|---|
| `01-state-board-score.md` | board history, the judge's per-shape table, score arithmetic & leverage, where the remaining µs are |
| `02-infrastructure.md` | the H100 box, container, GPU lock, harness, local CLI, ledger, waiters, logs, quotas |
| `03-rules-and-lessons.md` | judge rules, lint rules, the transfer rule, bake/measurement caveats, pitfalls that cost us runs |
| `04-track-rs128-m128.md` | RS128 kernel (the m=128 winner): design, numbers, what is being decomposed now, m=256 closure |
| `05-track-rs32.md` | RS128 at m=32/16 for qkv/o (+ short-K o) — just started |
| `06-track-router-pw35.md` | P3's one-node router (rtk) and pw35 (scored run launched 19:08 UTC) |
| `07-track-p4-anatomy.md` | P4: decode cost findings and the per-stage anatomy of rq (rs128p) |
| `08-track-cleanroom.md` | the clean-room rewrite workflow (4 schools), the latency-loader follow-up, how to read the journal |
| `09-reset-plan.md` | what to submit at the 00:00 UTC reset and in what order; how to spend checks vs scored; the agents alive at handover |
| `10-closed-tracks.md` | dead ends with the evidence, so nobody re-drills them |
| `11-agent-briefs.md` | copy-paste briefs to re-spawn each track's agent |

## Resume checklist (first 30 minutes of a new session)
1. `cd /home/ubuntu/okc && git status --short && git log --oneline -15` — agents commit often; uncommitted edits in
   `proj_rsi/candidates/{rs128,rs128p,rs32}` and `proj_rsi/cleanroom/*` belong to agents that may be dead. Do not revert them.
2. `python3 -m proj_rsi quota` and `date -u`. Checks and scored slots both reset at 00:00 UTC and EXPIRE (not banked).
3. Read the tail of `proj_rsi/LOOP.md` (the head log; every iteration is there) and `results/proj-rsi/jobs/*.scored.txt`
   for any judge report that landed while nobody watched (`rs32.scored.txt` is the one in flight at handover: job fb548b5dc6f04696, submitted 19:09 UTC, waiter armed).
4. Box health: `ssh root@107.170.6.60 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -af "flock -w" | wc -l; docker ps'`.
   No soak container may run while agents measure (see 03). If the GPU is idle, start the tracks in 09.
5. Re-spawn the tracks from `11-agent-briefs.md` (one agent per track, Opus medium, each in its own candidate dir).
6. Persistent memory file: `/home/ubuntu/.claude/projects/-home-ubuntu-okc/memory/muse-bf16-proj-loop.md` (long; same facts as here).
