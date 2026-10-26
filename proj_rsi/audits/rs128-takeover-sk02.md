# RS128 stream-K takeover-sk02 audit

Snapshot: 2026-10-26T19:51:55.313417+00:00. Readonly harvest by `/root/proj_tracks`; root executed the exclusive run.

Decision: **REJECT** the 132-SM stream-K variant. It fails the >=0.3us gain bar on every complete production path. No port or further automatic sweep.

Run: root PID342942, container `okc-proj-rs128-takeover-sk02`; root reports terminal and container absent. Outer360s / inner300s. Exact command: `lab.py --plan sk --layers pooler,qkv_gate,unpooler,o --reps40 --graph --json logs/rs128/takeover-sk02.json`.

| layer m128 | original hz graph | SK hz graph | regression | original z graph | SK z graph | regression |
|---|---:|---:|---:|---:|---:|---:|
|pooler|24.096|26.752|+2.656|25.120|27.424|+2.304|
|qkv_gate|17.152|18.912|+1.760|17.664|18.880|+1.216|
|unpooler|24.320|26.592|+2.272|25.344|27.552|+2.208|
|o|12.128|14.560|+2.432|11.968|14.656|+2.688|

All complete variants validate one-hot exact over all K columns plus normal/outlier checker tolerance; counters reset,168registers/no local memory. Shared memory increases144520->178176bytes. The input pack/unpack is exact.

The faster-looking `s3s5` and `s3s5k` rows use PRB4 timing-only probes without the complete output epilogue and do not carry validation. In particular, `o s3s5k`10.56us versus cuBLAS11.20us is not a deployable win. Original o production rq already loses to the tuned cuBLAS path; SK loses more.

CUPTI medians agree with graph regressions. They are attribution evidence, not official judge timing. No official check or scored slot should be spent on this proposal.

Raw JSON SHA256: `205292ca619f7a1dfd8c923088aad70d5ed915c74d5163732045000414173abb`
Raw log SHA256: `8fbd2538808ffebd8b5841e0f927786be272c7b86cd18c011dfa092a3bed4f50`
Remote artifacts remain `/root/okc/logs/rs128/takeover-sk02.{log,json}`.

The stream-K path is the full source variant tested here. This audit does not close every possible future scheduling design, but a new design needs different evidence rather than repeating these configurations.
