# Paired register fence: CPU codegen failure retained

`state=FAILED_VERSION; GPU_CALLS=0; GPU_ADMISSION=HOLD; performance=UNMEASURED`

Parent is frozen dual-producer `281ff5a3…`, generated source `97dd5613…`.
This candidate retains exact X pack, weight codec/escapes, parity producers,
W6/B8 barrier protocol, output map, launch512 and ordinary numerical policy.
Only startup/math scheduling and512B ephemeral retirement scratch differ.

The proposed schedule fences two preloaded A banks once, loads future A only
after that fence, and uses wait1 before reusing an old bank. Both decoded-B
waits precede the batch; both Bempty releases follow its second wait1. This
changes X lookahead3→2 and decoded-B wait/release timing as well as fence count.
These are explicit confounds, even if a future compiler makes the schedule work.

`schedule_proof.py` checks64stage loads/issues,32fences/WG, every register in
each16u32 bank, covering retirement, final wait0, and the192-node wait-for DAG.
It is a symbolic source model, not a physical register or runtime proof.
`preflight.py` verifies exact restoration to the frozen parent source, retained
bit/layout/output models, scratch bounds, positive parent physical gate and old
premature-overwrite negative control, then compiles with the exact parent flags.

```sh
python3 proj_rsi/candidates/operand_reverse_fence_pair/preflight.py
```

Expected exit1 for the current preserved source. No extension/context/GPU is
created. `preflight.json` records the failure; exact compiler log and SASS remain
beside the source. No full timing lab or generalized candidate-JIT checker was
admitted after this hard failure.

|version|change|actual CPU result|
|---|---|---|
|v1|initial pair, runtime startup/tail branches|C7520,122regs,0spill;16ARRIVE/16HGMMA static|
|v2|separate startup/steady/tail|C7512+C7519,118regs,0spill;48/48|
|v3|both Bfull waits before fence, releases after second group, compile-time futures|C7520+C7519,106regs,0spill;48/48|
|v4|MOV-self retirement pins|eliminated; same106regs and48/48|
|v5|identity PRMT retirement pins|eliminated; same106regs and48/48|
|v6 current|real volatile shared retirement pins|C7512+C7519,126regs,48B stack/48B spill stores/144B spill loads;48/48|

v1–v5 are immutable local archives. v6 source SHA256:
`ad08afe67b1e7798146e702796dd48cf7affd38df7364890693a94e4c2933b77`.
Desired v6 static schedule was6ARRIVEs across48HGMMA instructions (startup,
one materialized steady body, tail). Actual48ARRIVEs serialize every HGMMA;
register count alone therefore does not establish the intended A4 pipeline.

v3 SASS directly shows future-A loads reusing each consumed current-A chunk
after compiler-inserted wait0/ARRIVE. Empty asm pins disappear from PTX; MOV
and identity-PRMT variants did not retain a real liveness operation. This is
observed allocation/scheduling behavior, not a complete model of ptxas's cost
heuristic. Real volatile stores survive in v6 but fail register/spill and fence
gates. No warning suppression, flag sweep or GPU test followed.

Scratch is one64B span per math warp, immediately after all Wfull/Bfull/Bempty
arrays: relative byte102576..103087 from aligned base. Worst alignment slack
makes the last byte104111 within104112B requested SMEM. Only lane0 executes
four predicated volatile vector stores per retired stage; all warp A registers
remain compiler-live to their corresponding asm operands. Scratch is never read
as model input and has CTA lifetime. At128CTAs/8mathwarps/64stages it adds4MiB
logical shared writes and four vector-store instructions per warp/stage, before
any extra compiler spills. This cost was accepted only as a CPU liveness test.

The current mechanism is parked after its explicit CPU stop gate. ISA-level
per-register batching remains a legal conditional idea; this compiler result
does not prove a hardware ceiling or reject other producer/fence mechanisms.
