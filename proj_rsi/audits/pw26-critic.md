# pw26 critic (light) — 2026-10-26

**APPROVE, with one MAJOR fix recommended before the check.**

- **Build.** I built from the embedded bundle with torch's nvcc flags plus CUTFLAGS: 0 errors, 4 kernels at
  63 registers, 0 spills, 20 s locally.
- **MAJOR: wgmma serialisation.** ptxas emits C7510 on all four kernels: "wgmma.mma_async instructions are serialized
  due to wgmma pipeline crossing function boundary". pw25's three kernels had it too.
  - **Cause.** NDEBUG is not defined, so CUTLASS's device `assert` calls stay in the mainloop.
  - **Fix.** Add `-DNDEBUG` to CUTFLAGS. With it the build has 0 C7510, 64 registers and 0 spills.
- **Layouts.** The swap-AB strides and layouts are unchanged from pw25. M = n (2304, 8192, 4096, 8192) and N = m
  (128, 256) are multiples of 64.
- **Cluster.** cw7c2 is ClusterShape 1×2. A (W) uses SM90_TMA_LOAD_MULTICAST and B uses a plain TMA load. The cluster
  shape is static in the kernel type, so no runtime argument is needed. The token-tile count (2 or 4) is divisible by 2.
- **Shared memory.** SharedStorageSize, from `sizeof` in a host build:

  | Kernel | SharedStorageSize |
  |---|---|
  | cw13 | 214,016 B |
  | cw7 / cw7c2 | 115,712 B (exactly 113 KB) |
  | cwk128 | 132,096 B |

  2 × (115,712 + 1 KB reserve) = 233,472 B, exactly the 228 KB per SM, so cw7 fits 2 CTAs/SM with zero margin.
- **Python side.** Only the path table changed against pw25. The guard at budget − 100 is present. The file is
  817 KB.
