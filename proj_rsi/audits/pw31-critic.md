# pw31 critic review: the scored merge of pw27h and pw29h over pw23 (2026-10-26)

**Verdict: APPROVE.** The merge is exact. Both extensions build clean. Prepare fits the budget with guards on every
new phase. All 65 official CHECKs pass on the box, and no hazardous path can be picked.

The local file and the box copy are the same (md5 0be1318d…). The file is 196,211 B.

## 1. Merge: exact union
- `diff3 -m pw27h pw23 pw29h` merges with **0 conflicts**. Its output equals pw31 byte for byte, except the module
  docstring, where pw31 adds 2 lines describing the merge.
- So every hunk comes from exactly one parent, and no line is dropped or duplicated.
- The parents' hunks are disjoint:
  - pw27h changes GVCU (the P1 flag and trap, `prrun`), GRD/GXP, `_gvext`, `_gvcall` (`p1cnt`), `_gxbake`'s P1ALT,
    `_pr27` and RRB.
  - pw29h changes RWCU, RWV, `_rwwg`, `_rwsrc`, `_rwext`, RSPALL/RSP, `_rscall`'s `p[1] >= 100` branch, and `rw_cnt` in
    `_rsbake`.
- Counters are separate: `p1cnt` holds one 2048-int32 tensor per GVI instance, and `rw_cnt` is 8192 int32. Neither
  aliases `gvcnt`, `rs_cnt` or `rs_ws`.
- Generated sources:
  - pw31's gv source is byte-identical to pw27h's (91 GVI).
  - pw31's rw source is byte-identical to pw29h's.
  - r6's `CUDA` is identical across all four files.
  - pw31 has 7 RSP entries, which is pw29h's set.
- Both pw27h fixes are present:
  - `:1791`: rk-s16w8x is limited to m ∈ {8, 16} ("272 CTAs").
  - `:1078`: the flag wait traps after 2^24 spins.

## 2. Builds: OK
| Extension | Flags | Errors | Spills | Notes |
|---|---|---|---|---|
| gv (sm_90) | judge's flags, torch stub header | 0 | none new | only warning is the existing #1886 |
| rw (sm_90a) | RWFLAGS | 0 | 0 | 6 instances, 90/142/148/154 regs |

- **gv**
  - 116 kernels: all **90 of pw23's are SASS-identical**. The 30 new ones (13 P1 gv, rr<·,·,256/1024>, pr_*, p1zero)
    use 8–150 regs.
  - The 13 spilling kernels are the same set as pw23's.
  - Every P1 gv instance (V bits 27 and 29) contains exactly one `BPT.TRAP`.
- **rw**: this is pw29h's build exactly.

## 3. Prepare budget: OK
- **Box run (88.7 s):** r6 phase 84 s, which includes a cold gv build. The P1 bake took about 3 s. The rw build took
  0 s because it was cached, and the rs bake about 2 s.
- **Judge estimate**
  - r6 phase: about 110–118 s historically (pw17/pw22), plus about 10 s for the 13 extra gv instances.
  - P1 bake: about 10 s.
  - Cold rw build: about 37–45 s (box 37 s, pw29h.warm).
  - rs bake: about 6 s.
  - **Total about 175–190 s**, under the 240 s guard.
- **Guards**
  - P1ALT runs only before 210 s, and every gx path skips after 240 s.
  - `_rsbake` runs only before 220 s, and the rw build only before 180 s; each rw path skips after 240 s.
  - Worst case: the build starts at 179 s and the paths are skipped, ending at about 245 s, under 300 s.
  - PR27 is False.

## 4. lint / cpu-check: OK
- `lint`: ok; the only finding is the expected sm_90a multi-build warning.
- `cpu-check --official`: passed, 9/9.

## 5. Box log (logs/pw31.log, H100)
- 65 of 65 CHECK lines are ok.
- PICKs:
  - unpooler m=1 uk-s4z; m=8 uk-s4x
  - router m=8/16/32 rk-s16w8z
  - router m=128 rw64k32s8z; m=256 rw64k32r3x2z (107, which pw29h's critic stressed at 256 and 250)
  - router m=1 rr6 (r6 bake)
- No INVALID, ERROR, failed or skipped-budget lines.
- **rk-s16w8x is never baked at m=32.** It is not in the candidate list, so the judge cannot pick it.
- Soak against pw23 (10 rounds): router verify16 −2.0 µs and decode32 −0.8 µs. Everything else is within noise of
  ±0.5 µs.

## Notes (non-blocking)
- `p1cnt` is allocated lazily on first call. Every picked P1 path is called in the bake first, so the judge's captures
  never allocate it.
- The remaining pw29h note still applies: there is no `rwocc` co-residency guard. That is safe on a 132-SM H200.
