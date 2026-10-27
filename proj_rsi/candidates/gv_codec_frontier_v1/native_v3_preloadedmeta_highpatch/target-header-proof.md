# Target cleanup interface receipt

Root independently inspected the actual target container image using CPU-only
Docker `--runtime=runc`, with no GPU initialization. Root reported
CUDAContextLight.h lines90–92 declaring:

```
clearCublasWorkspaces();
std::map<std::tuple<void*,void*>,at::DataPtr>& cublas_handle_stream_to_workspace();
getChosenWorkspaceSize();
```

This is the plain-map Torch2.8 interface. The target has no
CublasHandlePool.h. Bindings therefore include ATen/cuda/CUDAContextLight.h and
use direct `.size()` before/after the declared clear operation. The independent
reviewer received the same target receipt. This memo records root's supplied
observation; it is not a newly collected header dump or a GPU test.

The newer local CPU wheel has a mutex-wrapped map. A draft compatibility adapter
was investigated but removed before final freeze because exact target evidence
resolves that concern. Draft bytes remain under history/draft-interface-adapter.
No adapter or version gate is added to the executable comparison.

The clear hook runs only after stream drain, graph reset and run-scope alias
release. It does not waive allocated0 or attribute the previous32MiB by size
alone. The new run must record cache entries/workspace size and allocator bytes
before/after clear; failure remains a cleanup failure.

Loop: profile -> measure -> hypothesis -> optimize/reimplement -> test ->
validate -> stack -> submit measured gains -> learn -> reprioritize -> repeat;
idle <=10s. [Contract](../../../ITERATION.md)
