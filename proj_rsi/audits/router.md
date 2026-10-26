# Router m=128/256 split-K (pw22/pw23): where the 7.5 µs go

Data: check 9e45ff0e1e78459d; the profile shows rsk2<64,256,4,128,0,1> at 7.5 µs plus a 0.9 µs memset node.

**(a) Reductions.** These are not scalar atomics: each fragment pair goes out as one `red.global.add.v2.f32`. That is 4,096 reds of 8 B per CTA, 128 CTAs, 0.5 M ops and 4 MB in total, all issued in the last microsecond of a one-wave grid. Each output address takes 32 adds (one per K split). At an estimated 200–300 reds/clk across the L2 slices, the burst costs about 1 µs. Pairing lanes with a shuffle into v4 reds halves the op count, about −0.4 µs.

**(b) Loads.** Each CTA ingests 128 KB (x 64 KB, re-read by the 4 n tiles mostly from L2; W 64 KB from HBM). At 1 CTA/SM that is about 2.5 µs at a ~53 GB/s L2→SM share, after about 1 µs to the first byte. The mma.sync work is 8.4 MFLOP per SM, about 2 µs, and mostly hides behind the stream. The 4 stages cut the exposed tail to the last 32 KB stage, about 0.5 µs; that is the −1.05 µs measured against rs64k32.

**(c) Why rp64k64x2 lost.** 256 CTAs halve the per-CTA bytes but double the partials: 1 M v2 reds, 8 MB. That adds about +1 µs of reduction burst, and the shorter K slices give each CTA only 2 stages to overlap.

**(d) m=256.** Beating cuBLAS's 10.9 judge-µs needs about 9.5 µs of kernel plus memset. The bytes are 192 KB/CTA (about 3.6 µs) and the mma.sync work is 16.8 MFLOP/SM (about 4 µs). That is at most 1 µs of overlap headroom before the 1 µs red burst, against a measured ~13.5 µs. Only wgmma (about 2× the tensor rate) makes it plausible.

**pw24: no-go.** v4 reds (−0.4) plus a cluster-2 DSMEM pre-reduction (−0.4, halving the reds, at the cost of cluster launch) give about 0.8 µs at m=128, below the 1.5 µs bar. m=256 needs a wgmma rewrite of the router for an uncertain win of about 1 µs.
