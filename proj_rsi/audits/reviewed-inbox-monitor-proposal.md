# Root admissions-only monitor: inactive integration proposal

This wrapper never claims, dispatches, retries, changes existing jobs, launches GPU work, or calls official APIs. Import and default invocation do not construct a store. Only root's explicit `--run` opens the existing store, after verifying a separately pinned manifest covering monitor, reviewed helper, exact policy, production model/policy/dispatcher/store, and independent helper review. Controller activation remains a root action after independent integration GO.

The policy admits only NEW GPU BACKLOG descriptors with prefixes `pw41.`, `pw42.`, `pw44.`, `pw45.`, `pw38.selected.profile.`; explicit IDs are empty. It charges USD5/hour including full reservation and any linked followup. Normal inner/outer caps are3600/3720 seconds; profile600/720. Existing IDs in every state, including HELD, are skipped. Partial files are reconsidered on subsequent scans. A source-review annotation requires a separate complete independent GO receipt, or an explicit root policy binding. The empty bindings map never guesses abbreviated review hashes. DONE valid dependencies remain mandatory; root completes CPU readiness records separately.

Each cycle reads one authoritative discovery snapshot and prefilters existing IDs before invoking the helper. Only new allowed GPU descriptors incur fresh locked helper validation. The existing root worker owns subsequent prioritization and atomic claim; the monitor never requests READY or changes capacity. Shared admission locking and atomic register protect concurrent scans. A new process imports the root-approved cached store; this proposal pins cached store578613244b8f89fb4f3bdf624e2dde417a98f112cac775e5e52804b98c1fbe1f. No running process is patched.

The default scan interval is15 seconds, hard maximum30. The internal monotonic loop allows at most7200 seconds; nonfinite or backward clock readings fail closed. A blocking filesystem operation cannot be interrupted by an internal Python deadline, so root's activation command must also use an outer timeout. The following is documentation only and has not been executed:

```sh
# Replace ROOT_REVIEWED_MANIFEST_SHA with the exact independently reviewed SHA.
timeout --signal=TERM --kill-after=5s 7195s python3 -m proj_rsi.scheduling._reviewed_inbox_monitor_proposal --run --manifest-sha256 ROOT_REVIEWED_MANIFEST_SHA --poll-seconds 15 --max-wall-seconds 7140
```

The TERM/KILL envelope ends by7200 seconds. Root starts this admissions-only process alongside the existing fresh worker; no worker cleanup, kill, launch, or registration was performed by the author. Restart is safe because existing IDs remain immutable. The monitor exits on store corruption/shape errors rather than repair state.

Decision JSONL contains only allowlisted ID/status fields, timestamps, and SHA256 digests. Raw commands, source/review contents, hosts, environments and paths are excluded. Identical decision/status/hash records are logged once per process; WAL sequence-only heartbeat changes do not spam logs. Log files use append/fsync, mode600 and symlink rejection. Output directories are created only during an explicitly activated run.

Validation uses temporary directories and FakeStore exclusively. Author tests cover inactive defaults, exact policy/caps, new BACKLOG admission, existing HELD/RUNNING immutability, single-snapshot prefilter, partial-file reconsideration, disallowed CPU/prefixes, bounded polling and deduplication, invalid clock, compact redaction, and missing/wrong/incomplete/drifted manifest pins. Independent review is separate. No real scheduler instance, queue registration, SSH, API, CUDA context or monitor run was used.

A separate performance followup remains: dispatch.work currently ranks GPU then repeatedly gets each row, causing redundant WAL replay in the legacy implementation. The root-approved cached store reduces this cost; a future reviewed batching change could use one readiness snapshot plus atomic claim. This monitor does not edit dispatch.work.
