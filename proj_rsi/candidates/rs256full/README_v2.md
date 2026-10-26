# rs256full port probe v2 — host initialization fix only

`state=CLOSED_PORT02_VALID_SYNTHETIC_INCONCLUSIVE_PORT_MODEL; port01=INVALID_PRESERVED; full_matmul=NOT_IMPLEMENTED`

**port02 result:** exact source `fe55…2ade` ran on the H100 after independent review. All27 measured samples (3 modes×9 reps) passed every accumulator/snapshot/state check and used128 distinct SMs. Actual default max dynamic shared was49152B; v2 set164928B, and all3 modes reported occupancy1/local0/device opt-in maximum232448B. This confirms v2 addressed the setup error. RootPID349453 and container are terminal. Raw evidence/hashes/all samples are in `port02.json`.

| Median, clock64 cycles | Consumer/K256 | Producer/80KiB | Overlap | Completed writes/consumer round |
|---|---:|---:|---:|---:|
| RS_LDS_DEC | 1303.680 | — | — | — |
| TMA | — | 1460.758 | — | — |
| COMBINED | 1308.969 | 1648.902 | 0.99995 | 0.77344 |

Combined consumer/resident=1.004057, but completed-write coverage failed the preregistered0.95 gate in all9 samples (range0.76562–0.78125). Temporal overlap passed; it does not imply the intended traffic rate. Resident consumer also exceeded the preregistered1200clk gate. **Classification: INCONCLUSIVE on port sharing; STOP this experiment, no full matmul or geometry sweep.** Consumer+0.4% cannot establish separate read/write paths at one80KiB fill per math round. Producer slowing under combined load also includes pacing/issue/poll effects. These data establish neither an aggregate-port physical ceiling nor a universal geometry NO-GO. No r6 comparison, wall-time speedup, H200 transfer, score, or serving claim follows.

The remaining sections preserve the prelaunch source/build description and commands; they do not authorize another run. `v2-manifest.json` remains the unchanged prelaunch compiler/CPU/SASS manifest.

Source: `port_probe_v2.cu`, SHA256 `fe55f19ba18d6e213ec038f4d068a771aa6190766b5aa72926fe95966ffe2ade`.

port01's original source remains `cc46c06739966806e4bd86ca078824e6d768068e8bac9426937b0f2b5fbf1774`; `port01.json` remains unchanged. Its occupancy0 rejection occurred before the first probe launch because `info()` ran before the launch helper opted into164928B dynamic shared memory. port01 produced no correctness, overlap, write-count, or timing result. It provides no evidence for or against either shared-port model.

The v2 delta is restricted to host `info()` and the build-command comment. `info()` now prints the actual kernel max dynamic shared bytes, static shared bytes, device opt-in maximum, and requested bytes; sets `cudaFuncAttributeMaxDynamicSharedMemorySize` before the occupancy query; re-reads and prints actual attributes; rejects a missing opt-in, local memory, or occupancy≠1. The existing launch helper's identical opt-in remains. Runtime attribute values are still unmeasured until a fresh trial.

Device source, geometry, controls, pacing, operands, all-result stores, snapshots, timing windows, warmups, checks, and interpretation thresholds are unchanged. CPU proof passes. CUDA12.8/sm90a compile+link passes for both sources: all3 kernels168regs/0stack/0spill/no C75 warnings. Normalized SASS instructions match exactly for all three kernels, not merely the MMA counts:

| Mode | Instructions | Normalized instruction SHA256 |
|---|---:|---|
| RS_LDS_DEC | 760 | `70263293747dc6e2759e4d216de35b96c091b1d99faef9ed850ae79022eb50fe` |
| TMA | 600 | `302c58cb81acaf973b16835927e5a7f43f1a59c519b7016dd77c35b5e4e29216` |
| COMBINED | 992 | `a516ac7de53f43eecdadac4cbb4aa9f4c30c9806fa9f064707afaf89392393a3` |

The mathematical question remains aggregate128B read+write ⇒1536clk/K256 versus separate128B paths ⇒max(896read,640write)=896clk, against1024 tensor clk. These are conditional models, not proven hardware limits. As preregistered in `README.md`, inadequate overlap/write coverage is inconclusive; numeric/state/occupancy errors invalidate the run; a port diagnostic cannot establish a ≥1µs pooler-m256 improvement over tuned r6. No clock64→wall-time conversion using nvidia-smi clocks.

Reproduce the CPU, compiler, source-delta, and full SASS identity gate without any GPU launch:

```bash
python3 proj_rsi/candidates/rs256full/preflight_v2.py --json /tmp/rs256full-v2-manifest.json
```

`v2-manifest.json` records source/proof hashes, both compiler outputs, exact per-kernel instruction hashes/counts, and CPU map/codec checks. All source bytes outside the host `info()` replacement and build-name comment are checked identical.

Only after independent review, root schedules one fresh bounded trial under the exclusive container-owned lock:

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v proj_rsi/candidates/rs256full/port_probe_v2.cu -o /tmp/rs256full-port-probe-v2
timeout 30s /tmp/rs256full-port-probe-v2 128 9
```

Use fresh run/log names and check source hash before execution. No child GPU launch or remote mutation occurred. This is a separately authorized host harness correction; port01 remains invalid. No geometry/mechanism sweep or full matmul implementation is included.
