# Minimal M256 coverage port (CPU only until root/critic GO)

Parent: frozen G8 manifest 2a1500fafb5e9a61675d17e83b2e1cd49958463eaba3567fa42d39a51d60d8d7,
generated source e92744c61531ae99e9345d2f541d5f5dda685ecd9ab5a7753621edd7c9d7289e.

Keep NT128, fixed N8192/K4096, producer codec/CSR/W6/B8/math order and per-CTA
103600B/512 threads unchanged. Main grid (128,2) uses y as token tile, never
split K. Fresh pack grid (256,2) adds 262144 uint32 input and XP offsets per y.
The main XP pointer adds 1048576 bytes per y; output TMA row becomes 128*y.
Output descriptor remains box128 with complete outer M256. Host guard admits
exactly M256 and 524288 int32 XP. Both launches are included every invocation.

CPU admission: reverse the exact finite edits to restore frozen G8 body/source;
exhaustively check every packed uint32 source/destination and consumer vector,
all output scalar destinations, inherited weight mapping and unchanged schedule;
compile exact parent flags with local nvcc, require <=128 main registers, zero
spill/stack/local/C75, full physical A/D fence/retirement checker and negative
controls. Any failed version stays attached to its artifacts; no GPU work here.

Root-only proposed lab compares actual prepared pw36 M256, new full two-node
M256, and two full parent G8 M128 invocations (four nodes) on disjoint halves.
Control owns separate workspaces/output; both parent packs are timed. Require
actual JIT physical/resource gates before first candidate math, full4096 basis,
FP64/task normal/outlier checks, all storage-bit pack oracle, fresh output/XP,
20 actual timed graph mutation checks, immutable input/W/packed/CSR, zero
counters, and reset all graphs. Full timing validity cannot be repaired later.

This doubles CTAs to256 and W decode/ingress work; roughly two CTA waves and
more W/cache pressure can erase benefit. The duplicate-parent control also
changes launch count and scheduling serialization. No microsecond forecast.
A genuine M256 gain affects verify32; conditional score arithmetic uses rs32
fb548 only. Parent G8 first validation and root/critic GO are prerequisites.
Odd/nearby M stays with production fallback. No frozen parent/queue edits.
