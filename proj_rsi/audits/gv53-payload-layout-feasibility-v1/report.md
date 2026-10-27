# GV53 M8 payload transport feasibility

The actual selected parent kernel is `gv_predxor<1,1,1,2,16,8,1,1,0,3>`, grid1×128×1/block1024. Its C12 payload is already planar `[rowtile,Kchunk,3,32,16]`: each lane loads three aligned16-byte vectors, not12-byte strided records.

Actual parent SASS has three payload loads at PCs0x3c0/0x3d0/0x3e0 with offsets0/512/1024; three loop prefetch sites supply the remaining seven chunks. This gives24 payload warp LDG128 instructions per KU8. Each1536-byte warp/chunk is covered exactly once. Across the selected grid:50,331,648 payload bytes,98,304 warp load instructions; no cross-warp/CTA payload duplication in the source address map. Escapes, X and output are separate traffic.

For a32-byte-aligned base, each warp vector load touches16 mathematical32-byte sectors,48 total. Actual W pointer alignment was not logged; if only16-byte alignment is assumed and base mod32=16, the model is17 per load/51 incidences,49 unique sectors. These are exact address sets, **not measured L1 requests, replays, L2 or DRAM transactions**.

One concrete alternative was checked: repack each lane's three planes into one48-byte AoS record, preserving the exact three vectors and thus codec/HMMA operands. Direct16-byte accesses at stride48 touch32 sectors per load:96 sector-instruction incidences versus current48, despite the same1536 unique bytes. A flat cooperative3-load read recovers48 incidences, but must redistribute92/96 vectors across lanes:368 remote scalar words, at least12 scalar SHFL instructions before input-round selection costs. The existing layout needs no payload gather shuffle. Exact byte-permutation inverse is verified by the CPU script.

The current three full-warp16-byte-per-lane loads already reach the1536/(32×16)=3 instruction bound for the existing primitive. No planar/AoS payload-layout compile is justified by a supposed12-byte-stride defect. This does not prove there is no memory-path issue: the parallel dynamic profile may expose request/replay, address dependency, X/header or cache-policy behavior that the static model cannot establish.

Bulk-copy counterevidence is preserved separately. On this geometry, adding the existing two-slot per-warp bulk path would request116,224 shared bytes versus17,408, preserve24 payload vector reads as LDS, and add eight copy issues, eight expect-tx operations and eight wait episodes per warp. Historical pw12 bulk variants lost their same-run bakes; its claimed bandwidth headroom was corrected for mixed timing units. Those old results are not current GV53 M8 bandwidth proof or a family-wide rejection.

**Next smallest falsifier:** bind the dynamic profile to the exact W load PCs and actual base alignment. If evidence contradicts the sector model or shows a specific memory-path issue, compile one changed load-stage implementation for that cause. Otherwise retain the existing planar transport and prioritize measured critical-path opportunities. No GPU/API/SSH/queue/git action or new GPU proposal.

Loop: profile → measure → hypothesis → optimize/test → validate → stack/submit → learn → reprioritize → repeat. [Contract](../../ITERATION.md).
