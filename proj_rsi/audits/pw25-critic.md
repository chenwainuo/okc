# pw25 critic review (CUTLASS 3.9.2 sm90 bf16 GEMM at m=128/256) — 2026-10-26

**Verdict: APPROVE.** I found no blockers. One MAJOR item is a budget margin on the build guard.

What I ran:
- r6's sources are untouched. GVCU, GVI and CUDA are byte-identical to pw23 (checked by import), so r6's 78 instances
  and the router kernels are unchanged (I verified them for pw23).
- I decoded the bundle: 502 tar members, regular files and directories only, no absolute or `..` paths, with
  `./LICENSE.txt` included.
- I compiled CUTCU locally exactly as load_inline would: torch 2.13's COMMON_NVCC_FLAGS + `-std=c++17 -fPIC` + the
  file's CUTFLAGS + the two include dirs. Result:
  - 0 errors;
  - 3 kernels at 168 registers, 0 spills;
  - **30.5 s** and 3.4 GB peak memory, for the .cu alone.

## 1. Judge-side mechanics: OK

**What the extraction touches.**
- The bundle is extracted with `tempfile.mkdtemp(prefix="cutlass392_")` under the default temp dir. No absolute paths
  are hard-coded, and no network is used.
- The modules are stdlib: base64, io, lzma, tarfile, tempfile. lzma imports fine here (Python 3.10.12).
- `tf.extractall(d)` has no `filter`. That is fine on 3.10-3.13 (3.12+ only warns), and 3.14's default "data" filter
  also accepts this archive.
- The only processes are load_inline's ninja/nvcc, as in every earlier check.

**load_inline arguments.**
- Include dirs go through `extra_include_paths=[d/include, d/tools/util/include]`, flags through
  `extra_cuda_cflags=CUTFLAGS`.
- `-gencode` is present, so torch adds no default arch.
- CUTCPP (the torch glue) does not include CUTLASS.

**Rules (muse_bf16_proj/TASK.md, AGENTS.md).**
- TASK.md: "A submission is rejected if it patches timing, reads harness frames, touches harness I/O or secrets, edits
  loaded modules, imports the reference, starts processes or uses the network."
- TASK.md: "A run has 10 minutes in all, compiling included". prepare "runs once, untimed, for up to 300 s".
- TASK.md: "The last 16,000 characters of what a run prints come back with it".
- AGENTS.md: "submissions should follow the spirit of the challenge … Create real speedups that work", and "Final
  review reruns the top entries on the scored shapes and on nearby ones … an entry that's much slower on the nearby
  shapes gets a closer look."
- Neither file has a rule on file size or third-party code. The full rules page (openkernelchallenge.dev/rules) is not
  readable offline.
- CUTLASS is BSD-3, which requires keeping the copyright notice; LICENSE.txt ships inside the bundle.
- **Note.** The paths are served at exact m=128 and 256 only. Nearby m (130, …) stays on r6, the same structure as
  pw22-24.

## 2. Workspace and stream-K: OK

- `st["cut_ws"]` is a 64 MB torch tensor allocated once in prepare, after the build. kernel_fn passes it in, so there
  is no cudaMalloc in the timed path.
- `run_one` checks `get_workspace_size(args) > wsn` and returns -4 (declined in the bake).
- `gemm.initialize(args, ws, st)` performs CUTLASS's workspace initialisation on `st` (a memset node under capture) on
  every call, so the stream-K flags are reset per replay.
- Stream-K fixup sums fp32 partials. Any order effect is about 1e-7, far inside the 1% / 2e-5 checks. One-hot stays
  exact: only one split carries the single product, and the others add 0.

## 3. Swap-AB layout: OK

- A = W, RowMajor (M = n, K): K-major.
- B = x, ColumnMajor K × N (N = m): element (k, j) at j·K + k, which is x[j][k].
- D = ColumnMajor M × N with ld = M: out[j][i].
- The strides are `make_cute_packed_stride` over (M,K,1), (N,K,1) and (M,N,1). Alignment 8 lies on the contiguous
  dims (K, K, M), so odd N = 121 / 250 is legal and goes through TMA OOB.
- alpha = 1, beta = 0. C aliases out but is not read when beta = 0.
- The output is bf16 RN of fp32. 8·w (one-hot) is exactly representable, so it is bit-exact.

## 4. Host code: OK

- The sequence is `can_implement` → workspace check → `initialize(args, ws, st)` → `run(st)`, returning
  -1 / -4 / -2 / -3 / cudaGetLastError. The bake prints `ERROR launch <code>` and declines.
- The stream is `at::cuda::getCurrentCUDAStream()`, so the GEMM is captured with the graph.
- `hw_info` is left default (sm_count 0). CUTLASS then queries the multiprocessor count with
  `cudaDeviceGetAttribute`, which is legal during capture.
- kernel_fn ignores the return code. That is safe only because the served (shape, variant) is exactly the one
  validated in the bake, and the host checks are deterministic.

## 5. Budget

- The cut bake starts only if `time < 240 − 60 = 180 s`. Each path re-checks 240.
- **Expected path.** The r6 phase plus the router bake ends at about 120 s (pw23). On the judge the build is the .cu
  at about 30 s local × 1.5-1.8, in parallel with the torch-glue .cpp: about 45-60 s. The 8 (layer, m) × 3 path bake
  then takes about 15 s. Total about 185-200 s, matching the claim.
- **Worst case.** If the r6 phase ends at 179 s and the build takes 90 s, the build ends at 269 s. The paths are
  skipped after 240, and the 8 r6 baseline timings (about 2 s) still run. That finishes at about 272 s, under 300, and
  r6 keeps serving.
- **MAJOR (margin).** The build cannot be interrupted, so a slow judge compile is not covered by any guard. Lower the
  pre-bake guard to `PREPARE_BUDGET_S − 100` (140 s), so that a 90-120 s compile still ends before about 260 s.
- The 10-minute total run limit also has to absorb about 80 s more prepare than pw23.

## 6. PROBE format

- `PROBE cuthbm pooler m=128 <name> <x.xx> tbs` parses as config "pooler m=128 cut128" followed by the number. The
  trailing "tbs" is ignored.
- MINOR. The value is TB/s, but the runner labels the slot "us". Read it as TB/s.
