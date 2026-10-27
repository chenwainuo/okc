# Thin CUDA C ABI: bounded CPU-only proof of concept

A small per-kernel CUDA shared library can avoid rebuilding Torch headers and
pybind glue. The POC uses exact frozen GV53 target code and three C exports:
`gv53_run(raw pointers, table, M, explicit stream)`, `gv53_attrs`, and
`gv53_error`. Python continues to own and validate tensors. No current harness,
author source, worker or runtime library is migrated.

## What the saved timings establish

The first unpool16 trial records42.474650819s compile-and-load inside
45.705811622s run time. Its host log shows three steps: C++ Torch/pybind glue,
NVCC CUDA source that also includes Torch headers, then link against Torch and
cudart. Those commands lack individual timestamps. C++ parsing, NVCC parsing,
link, dlopen and scheduling shares are therefore **unknown**. Torch import
occurs before the compile-and-load timer; the3.231s residual includes other
work and cannot be called import time.

The robust repeat reuses its already verified SO: native load0.002178360s,
total3.653058080s. Reusing an existing immutable SO is already the simplest
saving when native code is unchanged. Separate lanebase target-only CPU compile
2.303506942s is a different translation unit; subtracting it from42.475s would
not measure a header-only saving.

This POC's actual local CUDA12.8 build measures:

| Phase | Seconds |
|---|---:|
| NVCC compile, target kernels plus C wrapper |1.541180|
| link |0.064482|
| sum |1.605662|

There is no remote/paired build comparison or measured reservation saving.
No CUDA DLL was loaded and no CUDA API/context/GPU call was made. ELF imports
only libcudart.so.12 and libc.so.6; no Torch/Python headers/libraries are used.
The two linked target SASS streams match the frozen pilot exactly, including
PCs, instructions, predicates, encodings and control lines (360instructions
per target). Both compile at64registers, no stack/spills. The inherited pilot
also contains p1zero; the wrapper adds no device kernel. Runtime queried
attributes/occupancy remain unmeasured.

## The small seam and its responsibilities

- `bridge.cuh` is appended after exact pilot.cu. It builds the same G structure,
  geometry, dynamic shared size and launch entry as the existing binding, using
  the supplied stream handle instead of ATen's stream lookup.
- `adapter.py` takes an explicitly supplied Torch module, pinned DLL and verified
  existing cleanup callable. It validates dtype, shape, size, contiguity, device
  and alignment before accepting pointers. No Torch import or CUDA call occurs
  at module import. Runtime instantiation loads the DLL only in an authorized
  future GPU process.
- Bind buffers once per graph fixture. The bound object retains all tensor and
  DLL owners. The harness must retain it alongside every captured graph and
  reset **all** such graphs before `close_after_graph_reset()`. That operation
  synchronizes the device, then releases owners. There is no automatic destructor,
  manual dlclose, guessed stream completion or early buffer release.
- Each launch enters the tensor's device context and queries Torch's current
  stream at that moment, including its capture stream. It passes the64bit
  handle explicitly; no default-stream assumption, new stream or per-launch
  synchronization. Nondefault-stream/graph behavior is designed, **not GPU-tested**.
- Query attrs for both arms before capture, including opt-in shared allocation,
  registers/local/static bytes and queried maximum CTA capacity. The latter is
  capacity, not measured resident scheduling. Existing resource expectations
  remain the harness's responsibility.
- Immediate native errors raise with numeric status and CUDA error text.
  Asynchronous failures still surface at existing mandatory Torch synchronize,
  own-output and terminal checks. The C launch preserves the original immediate
  error clear/return behavior; no success is fabricated.
- BLAS workspace cleanup is not part of the per-kernel C library. Python oracle
  calls can still create Torch-owned workspaces. Forward the original reviewed,
  version-matched `clear_blas_cache` callable after graph drain/reset. Reuse an
  existing verified SO or a once-built stable Torch cleanup bridge. Missing bridge
  fails; no mangled-symbol ctypes call or fake allocation-origin claim is added.

Eight tests load only a tiny CPU C stub:64bit stream/table forwarding, changing
current stream, strong ownership until explicit drain, native positive/negative
errors, attrs return, metadata/alignment/device/dtype validation, DLL hash pin
and existing cleanup forwarding. No Torch/CUDA runtime is involved in those
tests. They do not prove numerical output, graph capture or actual occupancy.

## Small next use, if root chooses it

Keep active authors/harnesses unchanged. On one next already justified fixed
comparison, adapt only its binding/import seam to the thin DLL and existing
cleanup bridge; retain every current-X/task/terminal/mutation/immutability and
resource/graph cleanup check. Capture per-phase CPU build/import timing. Check
that explicit-stream launches enter the real captured graph and that output
numerics/own-terminal checks match before using any speed result. No new full
inventory, benchmark campaign or universal migration gate is proposed.

A cold paired build in the same target image can then price actual saved
reservation time. Build with that image's toolkit/cudart ABI and architecture;
this local12.8 library is not a portable binary for arbitrary CUDA13 images.
New kernel families require their own small raw-parameter wrapper, not a plugin
registry or framework. Preserve source/library hashes and original build flags.

`build.py` is a reproduction recipe, not an instruction to regenerate this
frozen bundle in place. Copy to a separately named output/version for changes.
Binaries are local proof products, not activation artifacts; the source-only
commit list excludes .o/.so. No GPU/API/SSH/queue/git operation occurred.

Loop: profile → measure → hypothesis → optimize → test → validate → stack →
submit measured gains → learn → reprioritize → repeat; idle ≤10s.
[Contract](../../ITERATION.md).
