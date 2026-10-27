# Independent dashboard review

CPU-only review, 2026-10-27. Code GO for the reviewed sources, subject to the publication owner verifying the final composed artifact and strict-private hosting access. No GPU, SSH, scheduler mutation, process launch API, or Sites publication was performed in this review.

## Findings resolved

- Both sanitizers leaked the secret following `Authorization: Bearer` and quoted JSON credential keys. Reproduced with independent sentinel strings; source owners fixed whole-value credential redaction, including quoted values with spaces. Python now also redacts absolute paths and Windows paths.
- Missing scheduler observations appeared as an empty queue and no execution. UI now explicitly renders scheduler unavailable / queue unknown.
- UI inferred hardware activity from recorded RUNNING jobs. Hardware headline now preserves its separate observed status; queue activity is explicitly recorded state.
- Unknown/future agent observation timestamps now indicate current activity is unknown. Canonical task identities survive normalization without exporting filesystem paths.

## Independent evidence

`python3 -m unittest proj_rsi.dashboard.test_snapshot proj_rsi.dashboard.test_publish_snapshot proj_rsi.dashboard.test_independent`: 24 tests pass. The independent suite uses a real Python-generated missing-source snapshot with the actual strict worker schema, authenticated ingest, GET roundtrip, replay rejection, unknown-key rejection and malicious credential strings. It evaluates the actual UI model and verifies a recorded RUNNING job does not turn unknown hardware status into Running. UI uses text nodes rather than HTML string assignments.

`node --test proj_rsi/dashboard/site/test-worker.mjs`: 10 tests pass at the reviewed backend revision. Tests exercise method/query rejection, absent/short/wrong secrets, oversized authorization, two-MiB streamed body limits, malformed UTF-8/JSON, strict nested schema, replay/future/stale timestamps, queue revision rollback, conditional concurrent updates, and sanitized errors. Final hardening also passes: stored objects are size bounded before parsing, and R2 metadata preserves the known queue sequence high-water mark across snapshots that correctly expose their sequence as unknown.

Static review confirms UI fetches only GET `/api/snapshot`. The only server write route is POST `/_ingest`, gated by a configured >=32-character secret and exact Bearer authentication before body/storage work. Runtime token comparison uses fixed-length SHA-256 digests. CSP blocks default sources, framing, forms, and base changes; inline scripts/styles are used by the embedded self-contained UI. Dynamic text is inserted with `textContent`.

Publisher credentials are supplied through environment or one stdin JSON line, excluded from dataclass repr and diagnostics, and sent to HTTPS using separate service-token and ingest-token Bearer headers. Redirects cannot forward credentials. Snapshot construction reads the authoritative WAL without repair and leaves missing queue/agent/hardware information unknown.

## Publication boundary

Local worker code does not itself authenticate dashboard GETs: strict-private hosting must enforce access to `/` and `/api/snapshot`. The root publication owner must verify strict-private policy, unauthorized access rejection, service-token header behavior, final artifact freshness and that the embedded UI/schema match the reviewed source. A local `.openai/hosting.json` project ID and R2 binding do not prove the deployed access policy. No live-deployment claim is made by this audit.

## Reviewed source hashes

- snapshot.py: `037ddf9a7b5897027226e814700e1057665c703e2d86c53116afd0299b5d3349`
- schema.json: `ea6723a914694886478da14f49426ac94f20c113a7bafef4f50aeb6f6143184e`
- ui.html: `07c795a6664067e97b9e70656eb8bef63f4987d53eea11176e9f4360c1b7a6de`
- backend.mjs: `9352b9324e0b88fb0b5dee1f7e5f4397f57b695c4120418d08736f42cbfb1169`
- publish_snapshot.py: `e1fff58fd800bdb87f195e26dc6ed78d9d3b8266e7bc2d568946864501133dce`
