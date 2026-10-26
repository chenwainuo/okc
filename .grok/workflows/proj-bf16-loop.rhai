let meta = #{
    name: "proj-bf16-loop",
    description: "Two read-only researchers for the next muse_bf16_proj probe. The session lead decides. No critic, validator, or executor.",
    when_to_use: "When muse_bf16_proj needs the next H200 experiment chosen without a role ceremony",
    phases: [
        #{ title: "Research", detail: "schedule and board math in parallel" },
    ],
};

fn as_text(r) {
    if r == () {
        return "MISSING";
    }
    if r.success {
        if type_of(r.output) == "string" {
            return r.output;
        }
        return json_encode(r.output);
    }
    return "FAILED";
}

phase("Research");

let facts = "Problem: muse_bf16_proj. The official leaderboard number must be under 3200 us. Displayed time is 4830.8 divided by speedup, so speedup must exceed 1.5096. Best submission-quality point is r6 (public v20): local harness about 1.303x, calibrated official proxy about 3413 us using R=3.58085 us per shape. Do not recommend okc login or okc submit. An agent must not accept the rules. Do not edit any file. Do not run docker. Do not use the GPU.";
facts += " Fresh H200 numbers for pooler M=128 N=4096 K=8192, cold-L2 CUDA graph, printed as raw minus 4 us. Per-tile wgmma asm with a wait after every tile (mser): 86.53 us. One PTX loop of 128 tiles reusing the same A registers and the same B descriptor, accumulators live for the whole asm (mpipe): 21.22 us. The same 512 wgmmas fully unrolled (munroll): 21.41 us. torch.mm in that harness: 27.10 us. The 21 us loop does not load from HBM. An earlier TMA-only roof (load X and W, no math) was 26.11 us. r6's full GEMM on this shape has judge median 22.72 us. That probe used 64 CTAs of BN=64, so 64 of 132 SMs. The m64n64k16 fragment map is already exact. A lossless 12-bit weight pack is 1.5 bytes per value. r6 uses it only for m<=32.";

let p_sched = facts;
p_sched += " You own the MMA schedule only. Read /tmp/proj_r7_probe.py. Grep for tma_k and for 'PIPE == 1'. Do not read the long unrolled wgmma listing. Explain why one asm is 21 us and a per-tile asm is 86 us. Say whether ld.shared, an mbarrier wait, and a thread-0 TMA can sit inside that same asm without the compiler draining the 32 accumulators between K tiles. Answer yes or no: can a correct BN=64 GEMM beat 22.72 us if the load roof is 26 us? If no, name the one tile or grid change required. End with a kernel sketch under 40 lines and one pass/fail microseconds number for the next probe. Stay on pooler M=128.";

let p_board = facts;
p_board += " You own the board arithmetic only. Read /home/ubuntu/okc/results/proj-rsi/r6-v20-p20.json and /home/ubuntu/okc/proj_rsi/protocol.json. Recompute the gap from about 3413 us to under 3200 us. Assume only verify-16 (m=128) qkv, pooler, and unpooler can change, and give the judge-microsecond cuts that land under 3200. Say whether BN=32 (128 CTAs) or weight compression is mandatory for pooler, given a 26 us uncompressed load roof and a 21 us math loop on 64 SMs. End with submit: no, plus the reason. Under 400 words. Name the JSON fields you used.";

let results = parallel([
    #{ prompt: p_sched, label: "research:schedule", capability_mode: "read-only" },
    #{ prompt: p_board, label: "research:board", capability_mode: "read-only" },
]);

let sched = as_text(results[0]);
let board = as_text(results[1]);
let report = "# Research for the lead\n\nThe session lead decides. Critic, validator, executor, and a second supervisor were not run.\n\n## Schedule\n\n";
report += sched;
report += "\n\n## Board\n\n";
report += board;
let path = write_scratch_file("research.md", report);
complete(#{ path: path, report: report });
