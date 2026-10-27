# CPU exact-pattern study

`state=CPU_ONLY; CUDA_CALLS=0; hardware_ratio=UNMEASURED; device_prototype=NONE`

Run `python3 proj_rsi/candidates/hw_compression_patterns/cpu_patterns.py`.

Proves both bitwise transforms on every65536 u16 pattern, packed inverse halfword range/control checks, and byte-plane roundtrips. Records local byte-pattern summaries on4.19M CPU synthetic BF16 normal values. CPU RNG bytes are not the actual GPU weights. No software codec ratio, NVIDIA compression-format assumption, speed prediction or GPU/API call is included.

See [interpretation/cost/next-gate audit](../../audits/hw-compression-pattern-layout.md). Frozen prior allocator/native/real-counter/zero-counter packages are not edited.
