# Two scored-selected projection RQ32 cells

Only `muse_bf16_proj`. Compare exact inc501d control body and descriptor-offset v2
at the RQ32/ST6/FL35 type actually selected in job9916d34357b745cf for pooler32 KS4
and unpooler32 KS2. This is a fixed scored-selected geometry test, not a claim of
fresh prepare selection or full-policy benefit. The copied actual profile records
are in `scored-selected.json`. Original Python selectors remain unchanged.

One extension compiles only these two device functions in private namespaces.
CUDA/SO identity and actual resources are saved before math. Original RQ host
launch/tensor maps, codec/escape helpers and matrix/protocol are reused. The local
Torch CPU installation lacks cusparse.h, so exact full-host offline compilation
was unavailable; four device types compile and the ordinary runtime JIT/resource
checks remain mandatory. This environment fact adds no extra GPU-admission gate.

Each cell uses shared immutable original weights/pack/escape arrays, separate
input/output/counter/sink backings, physical counter reset-zero and offset guards.
Normal/outlier/zero and32 onehot rows at changed ring/split boundaries precede
measurements. No full-K basis or codec campaign is repeated. F1 uses the same
fixture across all three blocks;16 paired seeds have8/8order. Outputs/current-X
are checked on every timed call; own block/paired terminals precede mutation.
Mutations poison output and change current inputs; all created graphs drain/reset.

CPU checks pass for exact codec AST extraction, scoped geometry, boundary rows,
measurement order and pins. `cpu-proof.json` is that executed readout. Bounds are
120s inner/180s outer. No full prepare/full30/profiler. Root owns GPU, queue and
subsequent actual policy/integration/official work only after a measured win.

Loop: profile -> measure -> hypothesis -> optimize/test -> validate -> stack/submit
-> learn -> reprioritize -> repeat; idle<=10s. Contract: proj_rsi/ITERATION.md.
