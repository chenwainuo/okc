# PW37 v1: concrete cumulative production candidate

`submission.py` is the actual standalone273072-byte cumulative file. SHA256:
`7913564fe7912cc69eace3e60b595ef3865440d60acc25e7fe0af7fda67a0330`.
It contains both established main branches from PW36: RS32 small-token RQ and
PW35 TMA/WGMMA RTK router, plus the existing cursor microchange. It is ready for
independent review and full cumulative GPU validation. It is not yet shipped,
measured, officially submitted or promoted.

Construction is exact: parent `../pw36/submission.py` SHA256
`2a067059cc432daa00b367e70aa150672c33e9d25dfe196d8bbe6943c813bccf`,
then the four unchanged replacements from `../smallm_increment/delta.json`
SHA256 `9513bc03fc5288d670b8b3c2a519ef63e3d03b90391e5d8ebaf7419276ad4d17`.
Reversing those four replacements restores the entire PW36 file byte-for-byte.
No header, router, dispatch, fallback, bake, geometry, flag or prepare guard was
otherwise changed. Source identity of historical official uploads remains the
association documented in the leaderboard audit; this new local file has an
explicit frozen byte identity.

CPU gates prove:

- The only changed named AST node is `RQCU`; its literal equals the cursor
  candidate `../smallm_increment/submission.py` SHA256
  `0d0ef30a495de725139fd8a401ed0c7a239d0ef10e7c41a8b4f9a6c2d37c1737`.
- `RWCU`, `_rwsrc`, `_rwext`, `RSPALL`, `RSP`, `_rsbake` exactly match both
  PW35 and PW36 AST. Both main optimization branches are present in one file.
- `prepare`, `kernel_fn`, `_r6`, `_v12`, `_rqbake`, `_rqsbake`, production
  variant lists, RQ source generators and flags exactly match PW36 AST.
  All30 layer/M task paths and their original selection mechanisms remain.
- Cursor `(slot,phase)` equals `(G%ST,(G//ST)&1)` over ST1..64 and KCS1..128,
  including ragged split-K bounds. It applies only to NT16/32, GS1, PRB0,
  WM0, INS0, FL35. Other RQ instantiations retain the original expression.
- Static lint passes, with the inherited multiple-sm90a-build advisory.
  Production has no helper dependency, subprocess or network import and stays
  below1MB. The CPU gate does not import production or torch, create aCUDA
  context, invoke prepare, load an extension, compile or run GPU work.

Existing offline compilation evidence is reused narrowly. Reconstructing the
standalone RQ translation unit from this file produces exactly the existing
`smallm_increment/after.cu`, with identical RQFLAGS and all pinned cubin/SASS
hashes. That evidence covers NT32/ST5, NT32/ST6 and NT16/ST5 with FL35; it does
not certify an actual JIT binary, every production variant, GPU correctness or
performance. Actual combined JIT/resource and full-scope tests remain required.

Reproduce the CPU composition proof from repository root:

```sh
python3 -m proj_rsi.candidates.pw37.cpu_proof
```

`cpu-proof.json` records all gates and reused artifact pins. `manifest.json`
freezes this version. A new EH port or any other follow-up must receive a new
candidate/version; PW37 v1 source and attached results remain immutable.
