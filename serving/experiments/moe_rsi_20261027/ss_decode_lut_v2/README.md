# Bounded register-lifetime followup to measured full-LUT stage win

Parent actual CUDA13 result: up1.406x and down1.732x median paired speedup over
original SS, both8/8wins, bitwise exact current/terminal stage outputs. Its queried
resources reduce active CTA capacity from8→6 up and5→4 down. At9active expert
blocks, down has576valid CTAs, exceeding132SM×4=528. A48-register candidate may
restore5activeCTAs/SM and fit those CTAs in one idealized capacity wave. This is a
resource hypothesis; actual scheduling/wave service was not counter-measured.

Two small CPU-only codegen experiments preserve the exact8KBhelper/table and
matrix/protocol. No codec/matrix proof was repeated:

| Variant | Up DIV8 regs | Down DIV1 regs | Spills |
|---|---:|---:|---|
| Parent full-LUT CUDA12.8 |40|60|0|
| Fuse each lookup into its shared scatter |40|58|0|
| Fuse + producer step loop unroll1 |38|48|0|

Pure fusion misses the occupancy target and remains preserved as a signed CPU
experiment. The rolled producer-step variant potentially restores down capacity;
up still rounds to40registers and has no established occupancy improvement.
The unchanged logical work remains262144U16lookups/valid full-KCTA,512KB shared
read payload. Static LDS count8 versus32 reflects a four-iteration loop, not four
fewer dynamic loads. Static instruction reductions likewise do not establish a
cycle improvement. Rolling can lose instruction parallelism or add loop overhead.

`fused_rolled_step/adapter.py` is runnable with the exact existing Bank/project
API and normal zeroing. It instantiates only T8DIV1/DIV8. Proposed next screen is
parent v1full-table down versus this rolled down on the same captured real inputs,
using the exact existing representation/oracle/protocol evidence and actual-JIT
resource gate. Keep v1full-table up unless independently measured otherwise. A
mixed v1up/v2down whole-serving arm is a followup only if the down stage wins.

`build.py` reproduces both offline CUDA12.8 compilations. `codegen.json` preserves
both results and source hashes. CPU only; no GPU, remote, queue or source editing
outside this isolated directory was performed for these variants.
