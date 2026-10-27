# One sign-low XOR representation / allocation diagnostic

`state=CPU_IMPLEMENTATION_FROZEN_FOR_REVIEW; GPU_UNRUN; new_CUDA_CPP=NONE; speed_claim=NONE`

Previous byteplanes01 is valid and flat ([terminal audit](../../audits/hw-compression-byteplanes-result.md)): raw GENERIC/NONE67,112,960/67,114,496 observed DRAM-read bytes, planes67,122,176/67,114,240. All gates passed; no matmul port is admitted. Preserve that source/result. This new version tests **one** different bit pattern with identical reader/allocation/profile controls; it does not reject the entire family or sweep formats.

Fixed exact sign-low XOR16, BASE0x3c00, on original BF16 bits u:

```text
e = ((((u & 0x7fff) ^ BASE) << 1) | (u >> 15)) & 0xffff
u = ((e >> 1) ^ BASE) | ((e & 1) << 15)
```

Move random sign to the least significant bit and XOR the magnitude prefix. This is integer reencoding into uint8 prepared storage; no floating rescale, precision relaxation or semantic reinterpretation as BF16 math occurs. CPU proof covers all65536patterns, both directions, NaN payloads/signed zeros/infinities and agreement with the frozen NumPy implementation in `hw_compression_patterns` manifest780f. BASE is fixed independent of inputs or weight values. Actual GPU preparation must decode **all64MiB back to the original native Wt snapshot**, and CPU/GPU encoded bytes must match, before any foreign allocation clone/capture.

CPU normal-distribution evidence was15.17%zero bytes and~3.98common upper-zero bits/u16 per32B sector, versus~0.20%/0native; **all-zero32B sectors remain0**. Whole-weight entropy is unchanged. No NVIDIA format, compression granularity or ratio is inferred. Future inverse~24logical ops/8weights versus packed12~9/8 is a material risk even if traffic responds; no backend is being built here.

Four selected cases only: `raw_generic`, `raw_none`, `signlow_generic`, `signlow_none`. Raw is the same contiguous prepared Wt[4096,8192],64MiB, actual GPU seed5151/Torch2.8.0+cu128, WtSHA09d1920b…168120. Seed is for reproducibility, with no kernel specialization. Each source owner is separate, query/grant1/0 exact, helper7f2/manifest8f32 D2D+streamsync unchanged. Source snapshots, four disjoint regular main outputs and four retained fresh outputs, complement poisoning, immediate full-byte output/source checks, no postprofile repair replay, all graph reset→alias drop→Region0 checks are inherited from frozen reader `hw_compression_byteplanes` manifest8e3d. Its CUDA d5b5/bindings e391/Case/copy_extension are imported verbatim; there is no new CUDA/C++ source or changed geometry. Runtime still requires exact32regs/0stack-spill-local-shared, occupancy≥1, CUDA attrs and actual JIT resource proof. Ordinary outputs remain UNKNOWN if VMM property query cannot identify them.

CPU gate:

```bash
python3 proj_rsi/candidates/hw_compression_signlow/preflight.py
```

No CUDA API or extension build runs locally: all65536Torch reencoding checks, frozen dependency hashes/proof, same21metrics/profile/cache/clock policy, parser core AST equivalence except fixed new identities, and2byte/12CSV/6graph-lifetime-format negative controls. The actual GPU full-payload inverse and all runtime gates remain unrun.

After critic GO, root alone may run one bounded exclusive diagnostic, proposed inner180/outer240, with existing profiler capability:

```bash
bash /okc/proj_rsi/candidates/hw_compression_signlow/run.sh
```

Same `okc_hwc_copy64m` volatile128bit reader,128blocks×256threads, full64MiB load/store coverage. Four NVTX nodes/launch-count4, kernel/node replay, cache-controlall/clock-controlnone/killno and same21supported memory/ILC/launch metrics. Transform, D2D, full snapshots/poison,256MiB flush, validation and cleanup remain outside selected ranges. Record actual passes/warnings/process cleanup. No GEMM, autotune, timer, zigzag, other encoding, population variation or serving claim.

CPU report import after capture:

```bash
/usr/local/cuda/bin/ncu --import /okc/logs/hw-signlow01.ncu-rep \
  --csv --page raw --print-units base > /okc/logs/hw-signlow01.raw.csv
python3 proj_rsi/candidates/hw_compression_signlow/parse_report.py \
  --csv /okc/logs/hw-signlow01.raw.csv \
  --run-json /okc/logs/hw-signlow01.json \
  --json /okc/logs/hw-signlow01.analysis.json
```

Compare GENERIC/NONE **within each payload**, then allocation ratios across raw/signlow under the identical reader. The new parser retains exact four-case labels, units/schema/resource guards and honest UNKNOWN/unavailable fields, and additionally requires correct fixed format/source identities and actual GPU full-byte inverse. Kilobyte-scale variation like byteplanes01 is not evidence to pay a new decode/backend cost. Even a clear read response is aggregate memory-unit evidence, not certified on-wire bytes, whole-W ratio, CI, matmul speed or direct portability to a different WGMMA-fragment pack. Root decides the next hypothesis from the result; no automatic matmul/official submission/rerun is authorized.
