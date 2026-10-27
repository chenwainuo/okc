# Read-only operations dashboard

URL: https://okc-kernel-operations.brinyspice130.chatgpt.site

Owner-private Sites deployment. Visitors can read and filter observations; no
launch, queue mutation, submission, SSH, or process-control endpoint exists.
`POST /_ingest` is an internal publisher route requiring both the private Sites
service boundary and a separate runtime secret. Anonymous reads fail closed.

Data path:
```
history_catalog.json + scheduler authoritative WAL + agent observation
    + hardware observation derived from actual dispatcher probes
  -> sanitized snapshot -> authenticated HTTPS ingest -> private R2
  -> GET /api/snapshot -> browser (10-second polling)
```

Queue, history and hardware have independent observation timestamps. Queue state
does not prove a GPU process is live. Missing, busy, corrupt, stale and future
observations remain explicit. The viewer never constructs QueueStore, repairs a
WAL, claims work, or launches a GPU probe. Agent counts come from a timestamped
`collaboration.list_agents` capture by root, not inferred process counts. History
is regenerated after archived evidence changes; it is not a live leaderboard API.

Local processes and logs (ignored by Git):
- `results/proj-rsi/dashboard/publisher.pid` / `publisher.log`: ten-second uploader.
- `hardware-observer.pid` / `hardware-observer.log`: ten-second WAL observer.
- `agents.json`: latest actual root agent observation, with `observed_at`.
- `hardware.json`: original probe timestamp retained across observer iterations.

Credentials remain in publisher memory. The startup helper receives one JSON
line on hidden stdin: `url`, `service_token`, `ingest_token`. Never commit tokens,
write credential files, pass tokens as command arguments, or log exceptions that
contain them. A host restart requires root to obtain service access again and
restart the publisher; stale data remains visible with its age meanwhile.

```
python3 -m proj_rsi.dashboard.start_publisher --interval 10
python3 -m proj_rsi.dashboard.observe_hardware --gpu-name H100 --interval 10
python3 -m proj_rsi.dashboard.verify_deployment
```

The first and third commands read credentials on stdin; disable terminal echo
before providing them. Check recorded PIDs against their actual command lines
before stopping or restarting either process. Do not infer their death from a
dashboard timeout or scheduler lease. Repeated startup refuses a live publisher.

Code checks:
```
python3 -m unittest proj_rsi.dashboard.test_snapshot proj_rsi.dashboard.test_publish_snapshot proj_rsi.dashboard.test_observe_hardware proj_rsi.dashboard.test_independent -q
```

Site source has its own repository; `.openai/hosting.json` inside that checkout
owns its persistent identity. `deployment.json` pins the published source/version.
Copy final `ui.html` and `schema.json` into that checkout before composing,
building, checking and packaging through the Sites skill. Do not create a second
Site when updating this one. Current verification includes live access, methods,
fresh data, schema and source checks; browser visual QA was unavailable.
