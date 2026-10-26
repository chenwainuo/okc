"""CUTLASS 3.9.2 sm90 bf16 GEMM configuration sweep for muse_bf16_proj's m = 128 / 256 shapes (run inside the container).

Swap-AB like pw25/pw26: A = W [M = n rows, K] row-major, B = x as K x N (N = tokens) column-major, D [M x N] column-major
(ld = M) == out[tok][n] row-major; C is void (beta = 0, no C smem / loads).

Each configuration is its own translation unit compiled by nvcc straight to a plain .so (extern "C" entry points, no torch
headers, -cudart shared) and loaded with ctypes, so instantiations build in parallel and each gets its own compile time
and ptxas -v report (registers, spills, wgmma serialisation warnings).

  python proj_rsi/tools/cutlass_sweep.py build [--set shake|wide|stages|all|name,name] [--jobs 16] [--timeout 300]
  /root/gpu_lock.sh /root/okc_run.sh python proj_rsi/tools/cutlass_sweep.py bench [--set ...] [--reps 30]
  python proj_rsi/tools/cutlass_sweep.py report

bench: for every built config x the eight (layer, m) shapes x the config's runtime variants (raster order; stream-K
decomposition), validates against torch.mm (relative L2 <= 1e-3) and times with h100_bench.time_graph (CUDA graph, L2
flush, median), alongside torch.mm(x, W^T contiguous) (the harness baseline), F.linear(x, W) and F.linear with
torch.backends.cuda.preferred_blas_library("cublaslt"). Results accumulate in logs/cutlass_sweep.json (resumable; a
config whose bench crashed the process is marked and skipped on restart); report writes logs/cutlass_sweep.md.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUT = ROOT / "third_party" / "cutlass_dl" / "cutlass"
BUILD = ROOT / "logs" / "cutlass_sweep_build"
JSON = ROOT / "logs" / "cutlass_sweep.json"
MD = ROOT / "logs" / "cutlass_sweep.md"
FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-DNDEBUG", "-std=c++17"]
SHAPES = [("qkv_gate", 128), ("qkv_gate", 256), ("o", 128), ("o", 256), ("pooler", 128), ("pooler", 256),
          ("unpooler", 128), ("unpooler", 256)]
SMEM_MAX = 232448

KSCHED = {"ws": "cutlass::gemm::KernelTmaWarpSpecialized", "pp": "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
          "coop": "cutlass::gemm::KernelTmaWarpSpecializedCooperative"}
# runtime variants: 0 heuristic raster, 1 along M (W rows), 2 along N (tokens); stream-K: 3 forced stream-K,
# 4 split-K 2, 5 split-K 4, 6 data-parallel, 7 stream-K nondeterministic reduction
VARS_PERSIST = [0, 1, 2]
VARS_SK = [0, 3, 4, 5, 7]
VNAME = {0: "heur", 1: "rasM", 2: "rasN", 3: "sk", 4: "split2", 5: "split4", 6: "dp", 7: "sk-nd"}


def cfg_name(c):
    tm, tn, tk = c["tile"]
    s = "%s_%dx%dx%d_c%dx%d_s%s_%s" % (c["k"], tm, tn, tk, c["cl"][0], c["cl"][1], c["st"] or "A", c["epi"])
    return ("ns_" if c.get("o") == "ns" else "") + s + ("_sk" if c["sched"] == "sk" else "")


def stage_bytes(tile):
    tm, tn, tk = tile
    return (tm * tk + tn * tk) * 2 + 32


def mk(k, tile, cl=(1, 1), st=0, epi="tma", sched="p", o="s"):
    c = {"k": k, "tile": tile, "cl": cl, "st": st, "epi": epi, "sched": sched}
    if o == "ns":
        c["o"] = "ns"  # non-swapped: M = tokens, N = W rows (out row-major directly); allows N tiles of 72/96/48 like nvjet
    return c


TILES = [(64, 64, 64), (64, 128, 64), (128, 64, 64), (128, 128, 64), (64, 64, 128), (64, 256, 64)]
CLUSTERS = [(1, 1), (2, 1), (1, 2)]


def config_sets():
    shake = [mk("ws", (64, 64, 64), (1, 2), 7), mk("pp", (64, 64, 64)), mk("coop", (128, 64, 64), sched="sk"),
             mk("ws", (64, 128, 64))]
    wide = []
    for tile in TILES:
        kinds = ["ws", "pp"] + (["coop"] if tile[0] >= 128 else [])
        for k in kinds:
            for cl in CLUSTERS:
                for epi in ("tma", "nosmem"):
                    wide.append(mk(k, tile, cl, 0, epi))
                if k == "coop":
                    wide.append(mk(k, tile, cl, 0, "tma", "sk"))
    stages = []
    for tile in TILES:
        kinds = ["ws", "pp"] + (["coop"] if tile[0] >= 128 else [])
        for k in kinds:
            for cl in CLUSTERS:
                for st in (4, 8, 12):
                    if st * stage_bytes(tile) + 16384 <= SMEM_MAX:
                        stages.append(mk(k, tile, cl, st, "tma"))
    ns = []
    for tile in [(64, 64, 64), (64, 128, 64), (64, 72, 64), (64, 96, 64), (64, 48, 64), (128, 72, 64), (128, 96, 64), (64, 144, 64)]:
        for k in ["ws", "pp"]:
            for cl in CLUSTERS:
                for epi in ("tma", "nosmem"):
                    ns.append(mk(k, tile, cl, 0, epi, o="ns"))
        if tile[0] >= 128:
            ns.append(mk("coop", tile, (1, 1), 0, "tma", "sk", o="ns"))
    sets = {"shake": shake, "wide": wide, "stages": stages, "ns": ns}
    seen, allc = set(), []
    for v in sets.values():
        for c in v:
            if cfg_name(c) not in seen:
                seen.add(cfg_name(c))
                allc.append(c)
    sets["all"] = allc
    return sets


def select(spec):
    sets = config_sets()
    if spec in sets:
        return sets[spec]
    byname = {cfg_name(c): c for c in sets["all"]}
    out = []
    for part in spec.split(","):
        if part in sets:
            out += sets[part]
        elif part in byname:
            out.append(byname[part])
        else:
            raise SystemExit("unknown config/set %s" % part)
    return out


def gen_cu(c):
    tm, tn, tk = c["tile"]
    esched = {"tma": "cutlass::epilogue::TmaWarpSpecializedCooperative" if c["k"] == "coop" else "cutlass::epilogue::TmaWarpSpecialized",
              "nosmem": "cutlass::epilogue::NoSmemWarpSpecialized"}[c["epi"]]
    stagecount = ("cutlass::gemm::collective::StageCount<%d>" % c["st"]) if c["st"] else \
        "cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>"
    tsched = "cutlass::gemm::StreamKScheduler" if c["sched"] == "sk" else "cutlass::gemm::PersistentScheduler"
    return r"""// %(name)s (generated by proj_rsi/tools/cutlass_sweep.py)
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
namespace sw {
using EA = cutlass::bfloat16_t;
template <class TileShape, class ClusterShape, class KSched, class ESched, class TSched, int IsSK>
struct Cfg {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, %(lD)s, 8, EA, %(lD)s, 8, ESched>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, cutlass::layout::RowMajor, 8, EA, cutlass::layout::ColumnMajor, 8, float,
      TileShape, ClusterShape, %(stagecount)s, KSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, TSched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using C = Cfg<Shape<_%(tm)d, _%(tn)d, _%(tk)d>, Shape<_%(cx)d, _%(cy)d, _1>, %(ksched)s, %(esched)s, %(tsched)s, %(issk)d>;
using G = C::Gemm;
using Kernel = C::Kernel;
static int sm_count() {
  static int n = 0;
  if (!n) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}
template <class TS, class A>
static void set_sched(A& s, int var) {
  using RO = typename TS::RasterOrderOptions;
  s.raster_order = var == 1 ? RO::AlongM : var == 2 ? RO::AlongN : RO::Heuristic;
  if constexpr (%(issk)d) {
    if (var == 3) s.decomposition_mode = TS::DecompositionMode::StreamK;
    if (var == 4) { s.decomposition_mode = TS::DecompositionMode::SplitK; s.splits = 2; }
    if (var == 5) { s.decomposition_mode = TS::DecompositionMode::SplitK; s.splits = 4; }
    if (var == 6) s.decomposition_mode = TS::DecompositionMode::DataParallel;
    if (var == 7) { s.decomposition_mode = TS::DecompositionMode::StreamK; s.reduction_mode = TS::ReductionMode::Nondeterministic; }
  }
}
static typename G::Arguments make_args(const void* W, const void* x, void* out, int M, int N, int K, int var) {
  if (%(ns)d) { const void* t = W; W = x; x = t; int u = M; M = N; N = u; }  // non-swapped: A = x [tok][K], B = W [n][K], D = out [tok][n]
  using SA = typename G::GemmKernel::StrideA; using SB = typename G::GemmKernel::StrideB;
  using SC = typename G::GemmKernel::StrideC; using SD = typename G::GemmKernel::StrideD;
  SA sa = cutlass::make_cute_packed_stride(SA{}, make_shape(M, K, 1));
  SB sb = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
  SC sc = cutlass::make_cute_packed_stride(SC{}, make_shape(M, N, 1));
  SD sd = cutlass::make_cute_packed_stride(SD{}, make_shape(M, N, 1));
  typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {(const EA*)W, sa, (const EA*)x, sb}, {{1.f, 0.f}, nullptr, sc, (EA*)out, sd}};
  args.hw_info.device_id = 0;
  args.hw_info.sm_count = sm_count();
  set_sched<typename Kernel::TileScheduler>(args.scheduler, var);
  return args;
}
}  // namespace sw
extern "C" int cfg_run(const void* W, const void* x, void* out, int M, int N, int K, void* ws, size_t wsn, cudaStream_t st, int var) {
  auto args = sw::make_args(W, x, out, M, N, K, var);
  sw::G gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (sw::G::get_workspace_size(args) > wsn) return -4;
  if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) return -2;
  if (gemm.run(st) != cutlass::Status::kSuccess) return -3;
  return (int)cudaGetLastError();
}
extern "C" long long cfg_ws(int M, int N, int K, int var) {
  auto args = sw::make_args(nullptr, nullptr, nullptr, M, N, K, var);
  return (long long)sw::G::get_workspace_size(args);
}
extern "C" int cfg_info(long long* o) {
  auto fn = cutlass::device_kernel<sw::Kernel>;
  int smem = sw::Kernel::SharedStorageSize;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, fn) != cudaSuccess) return -1;
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, sw::Kernel::MaxThreadsPerBlock, smem);
  o[0] = a.numRegs; o[1] = a.localSizeBytes; o[2] = smem; o[3] = sw::C::Main::DispatchPolicy::Stages;
  o[4] = sw::Kernel::MaxThreadsPerBlock; o[5] = occ;
  return (int)cudaGetLastError();
}
""" % {"name": cfg_name(c), "tm": tm, "tn": tn, "tk": tk, "cx": c["cl"][0], "cy": c["cl"][1], "ksched": KSCHED[c["k"]],
       "esched": esched, "tsched": tsched, "issk": int(c["sched"] == "sk"), "stagecount": stagecount,
       "ns": int(c.get("o") == "ns"), "lD": "cutlass::layout::RowMajor" if c.get("o") == "ns" else "cutlass::layout::ColumnMajor"}


def parse_ptxas(log):
    regs = spill_st = spill_ld = stack = 0
    for m in re.finditer(r"Used (\d+) registers", log):
        regs = max(regs, int(m.group(1)))
    for m in re.finditer(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log):
        stack, spill_st, spill_ld = max(stack, int(m.group(1))), max(spill_st, int(m.group(2))), max(spill_ld, int(m.group(3)))
    return {"ptx_regs": regs, "spill_st": spill_st, "spill_ld": spill_ld, "stack": stack,
            "serialized": "serialized" in log or "C7510" in log}


def build_one(c, timeout):
    name = cfg_name(c)
    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    src = gen_cu(c)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("ok") and (d / "cfg.so").exists() and meta.get("src_hash") == hash_src(src):
            return meta
    (d / "cfg.cu").write_text(src)
    cmd = ["nvcc", *FLAGS, "-Xptxas", "-v", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared",
           "-I", str(CUT / "include"), "-I", str(CUT / "tools" / "util" / "include"), "-o", str(d / "cfg.so"), str(d / "cfg.cu")]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        log, _ = p.communicate()
        rc = "timeout"
    dt = time.time() - t0
    (d / "build.log").write_text(log or "")
    meta = {"name": name, "cfg": c, "ok": rc == 0, "rc": rc, "compile_s": round(dt, 1), "src_hash": hash_src(src)}
    meta.update(parse_ptxas(log or ""))
    if rc != 0:
        errs = [ln for ln in (log or "").splitlines() if "error" in ln]
        meta["err"] = (errs[0] if errs else (log or "")[-300:])[:400]
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def hash_src(s):
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def cmd_build(args):
    cfgs = select(args.set)
    print("building %d configs with %d jobs" % (len(cfgs), args.jobs), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(build_one, c, args.timeout) for c in cfgs]
        for f in futs:
            m = f.result()
            print("%-44s %s %6.1fs regs %3d spill %d/%d stack %d%s%s" % (
                m["name"], "ok  " if m["ok"] else "FAIL", m["compile_s"], m["ptx_regs"], m["spill_st"], m["spill_ld"],
                m["stack"], " SERIALIZED" if m["serialized"] else "", "" if m["ok"] else "  " + m.get("err", "")[:160]), flush=True)
    print("build wall %.0fs" % (time.time() - t0), flush=True)


def load_results():
    if JSON.exists():
        return json.loads(JSON.read_text())
    return {"configs": {}, "torch": {}, "env": {}}


def save_results(r):
    JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1))
    tmp.replace(JSON)


def cmd_bench(args):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
    sys.path.insert(0, str(ROOT / "proj_rsi" / "tools"))
    import reference
    from h100_bench import flush_buffer, time_graph

    res = load_results()
    torch.cuda.init()
    dev = torch.device("cuda")
    weights = reference.generate_weights(args.seed)
    flush = flush_buffer()
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    tiny = torch.zeros(32, device=dev)
    res["env"] = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "replay_overhead_us": time_graph(lambda: tiny.add_(1.0), flush, args.reps), "reps": args.reps}
    data = {}
    for layer, m in SHAPES:
        w = weights[layer]
        n = reference.out_features(layer, m)
        w = w[:n].contiguous()
        x = reference.generate_input(m, layer, args.seed + m, dist="normal")["x"].contiguous()
        wt = w.t().contiguous()
        ref = torch.mm(x, wt)
        key = "%s:%d" % (layer, m)
        data[key] = (w, x, ref, n, w.shape[1])
        if key not in res["torch"] or args.retorch:
            t = {}
            t["torch_mm"] = time_graph(lambda: torch.mm(x, wt), flush, args.reps)
            t["linear"] = time_graph(lambda: F.linear(x, w), flush, args.reps)
            prev = torch.backends.cuda.preferred_blas_library()
            try:
                torch.backends.cuda.preferred_blas_library("cublaslt")
                t["linear_lt"] = time_graph(lambda: F.linear(x, w), flush, args.reps)
                t["mm_lt"] = time_graph(lambda: torch.mm(x, wt), flush, args.reps)
            finally:
                torch.backends.cuda.preferred_blas_library(prev)
            res["torch"][key] = t
            print("TORCH %-12s %s" % (key, "  ".join("%s %.2f" % kv for kv in t.items())), flush=True)
    save_results(res)

    for c in select(args.set):
        name = cfg_name(c)
        meta_p = BUILD / name / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        ent = res["configs"].get(name)
        if ent and ent.get("done") and ent.get("src_hash") == meta.get("src_hash") and not args.force:
            continue
        if ent and ent.get("inprogress") and not args.force:
            ent.update({"inprogress": False, "done": True, "crashed": True})
            save_results(res)
            print("SKIP %s (crashed a previous bench)" % name, flush=True)
            continue
        ent = {"meta": meta, "src_hash": meta.get("src_hash"), "inprogress": True, "results": {}}
        res["configs"][name] = ent
        save_results(res)
        if not meta["ok"]:
            ent.update({"inprogress": False, "done": True})
            save_results(res)
            continue
        lib = ctypes.CDLL(str(BUILD / name / "cfg.so"))
        lib.cfg_run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
        lib.cfg_run.restype = ctypes.c_int
        lib.cfg_ws.argtypes = [ctypes.c_int] * 4
        lib.cfg_ws.restype = ctypes.c_longlong
        info = (ctypes.c_longlong * 8)()
        lib.cfg_info.argtypes = [ctypes.c_void_p]
        rc = lib.cfg_info(info)
        ent["info"] = {"rc": rc, "regs": info[0], "local": info[1], "smem": info[2], "stages": info[3], "threads": info[4], "occ": info[5]}
        variants = VARS_SK if c["sched"] == "sk" else VARS_PERSIST
        line = []
        for layer, m in SHAPES:
            key = "%s:%d" % (layer, m)
            w, x, ref, n, k = data[key]
            out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            per = {}
            for var in variants:
                out.zero_()

                def run():
                    return lib.cfg_run(w.data_ptr(), x.data_ptr(), out.data_ptr(), n, m, k, ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream, var)
                r = run()
                torch.cuda.synchronize()
                if r != 0:
                    per[VNAME[var]] = {"rc": r}
                    continue
                rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if not rel <= 1e-3:
                    per[VNAME[var]] = {"rc": 0, "rel": rel, "bad": True}
                    continue
                per[VNAME[var]] = {"rc": 0, "rel": rel, "us": time_graph(run, flush, args.reps)}
            ent["results"][key] = per
            good = [(v["us"], kv) for kv, v in per.items() if "us" in v]
            if good:
                b = min(good)
                line.append("%s %.1f(%s)" % (key, b[0], b[1]))
            else:
                line.append("%s -- %s" % (key, ",".join("%s" % v.get("rc") if not v.get("bad") else "rel%.1e" % v["rel"] for v in per.values())))
        ent.update({"inprogress": False, "done": True})
        save_results(res)
        print("CFG %-44s regs %d occ %d smem %d st %d | %s" % (name, info[0], info[5], info[2], info[3], "  ".join(line)), flush=True)
    cmd_report(args)


def cmd_confirm(args):
    """Re-time a short list of configs (every runtime variant) and the torch baselines in interleaved rounds with more
    reps, to remove the best-of-many selection bias of the sweep; also records cuBLAS's kernel names per shape."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / "muse_bf16_proj"))
    sys.path.insert(0, str(ROOT / "proj_rsi" / "tools"))
    import reference
    from h100_bench import flush_buffer, time_graph

    res = load_results()
    dev = torch.device("cuda")
    weights = reference.generate_weights(args.seed)
    flush = flush_buffer()
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    names = args.names.split(",")
    libs = {}
    for name in names:
        c = json.loads((BUILD / name / "meta.json").read_text())["cfg"]
        lib = ctypes.CDLL(str(BUILD / name / "cfg.so"))
        lib.cfg_run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
        libs[name] = (lib, VARS_SK if c["sched"] == "sk" else VARS_PERSIST)
    conf = {}
    kern = {}
    for layer, m in SHAPES:
        key = "%s:%d" % (layer, m)
        n = reference.out_features(layer, m)
        w = weights[layer][:n].contiguous()
        x = reference.generate_input(m, layer, args.seed + m, dist="normal")["x"].contiguous()
        wt = w.t().contiguous()
        ref = torch.mm(x, wt)
        out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
        fns = {"torch_mm": lambda: torch.mm(x, wt), "linear": lambda: F.linear(x, w)}
        for name, (lib, variants) in libs.items():
            for var in variants:
                def run(lib=lib, var=var):
                    return lib.cfg_run(w.data_ptr(), x.data_ptr(), out.data_ptr(), n, m, w.shape[1], ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream, var)
                out.zero_()
                if run() != 0:
                    continue
                torch.cuda.synchronize()
                if not ((out.float() - ref.float()).norm() / ref.float().norm()).item() <= 1e-3:
                    continue
                fns["%s/%s" % (name, VNAME[var])] = run
        prev = torch.backends.cuda.preferred_blas_library()
        times = {k: [] for k in list(fns) + ["linear_lt"]}
        for rnd in range(args.rounds):
            order = list(fns.items()) if rnd % 2 == 0 else list(fns.items())[::-1]
            for k, fn in order:
                times[k].append(time_graph(fn, flush, args.reps))
            try:
                torch.backends.cuda.preferred_blas_library("cublaslt")
                times["linear_lt"].append(time_graph(lambda: F.linear(x, w), flush, args.reps))
            finally:
                torch.backends.cuda.preferred_blas_library(prev)
        conf[key] = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        names_k = {}
        for k, use_lt in (("torch_mm", False), ("linear", False), ("linear_lt", True)):
            try:
                if use_lt:
                    torch.backends.cuda.preferred_blas_library("cublaslt")
                fn = fns["torch_mm"] if k == "torch_mm" else (lambda: F.linear(x, w))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                names_k[k] = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA][:3]
            finally:
                torch.backends.cuda.preferred_blas_library(prev)
        kern[key] = names_k
        tb = min(conf[key][k] for k in ("torch_mm", "linear", "linear_lt"))
        top = sorted((v, k) for k, v in conf[key].items() if "/" in k)[:4]
        print("CONF %-12s torch_mm %.2f linear %.2f linear_lt %.2f | %s | %s" % (key, conf[key]["torch_mm"], conf[key]["linear"],
              conf[key]["linear_lt"], "  ".join("%s %.2f (%.3f)" % (k, v, v / tb) for v, k in top), names_k), flush=True)
    res["confirm"] = {"names": names, "reps": args.reps, "rounds": args.rounds, "us": conf, "cublas_kernels": kern}
    save_results(res)
    cmd_report(args)


def cmd_report(args):
    res = load_results()
    tor = res.get("torch", {})
    keys = ["%s:%d" % s for s in SHAPES]
    rows = []
    for name, ent in res.get("configs", {}).items():
        meta = ent.get("meta", {})
        info = ent.get("info", {})
        cells = {}
        for key in keys:
            per = ent.get("results", {}).get(key, {})
            good = [(v["us"], kv) for kv, v in per.items() if "us" in v]
            if good:
                us, var = min(good)
                t = tor.get(key, {})
                cells[key] = {"us": us, "var": var, "torch_mm": t.get("torch_mm"), "torch_best": min(t.values()) if t else None,
                              "ratio": us / t["torch_mm"] if t else None, "ratio_best": us / min(t.values()) if t else None,
                              "rel": per[var]["rel"]}
        rows.append({"name": name, "ok": meta.get("ok"), "crashed": ent.get("crashed", False), "compile_s": meta.get("compile_s"),
                     "regs": info.get("regs", meta.get("ptx_regs")), "spill": "%s/%s" % (meta.get("spill_st"), meta.get("spill_ld")),
                     "serialized": meta.get("serialized"), "smem": info.get("smem"), "stages": info.get("stages"),
                     "occ": info.get("occ"), "cells": cells, "err": meta.get("err")})
    res["table"] = rows
    save_results(res)
    env = res.get("env", {})
    L = ["# CUTLASS 3.9.2 sm90 bf16 swap-AB sweep (H100, graph replay + L2 flush, median of %s)" % env.get("reps"), "",
         "Device %s, torch %s, CUDA %s. Times are graph-replay us incl. the box's empty-replay overhead (%.2f us); "
         "ratio = CUTLASS / torch.mm(x, W^T contiguous) (h100_bench baseline); 'best torch' = min of torch.mm, F.linear, "
         "and both under preferred_blas_library('cublaslt'). Each cell is the config's best runtime variant "
         "(raster heur/rasM/rasN; stream-K heur/sk/split2/split4/sk-nd). Validation: rel L2 vs torch.mm <= 1e-3."
         % (env.get("device"), env.get("torch"), env.get("cuda"), env.get("replay_overhead_us") or float("nan")), "",
         "## Torch baselines (us)", "", "| shape | torch.mm | F.linear | F.linear (cublaslt) | mm (cublaslt) |", "|---|---|---|---|---|"]
    for key in keys:
        t = tor.get(key, {})
        L.append("| %s | %s |" % (key, " | ".join("%.2f" % t[k] if k in t else "-" for k in ("torch_mm", "linear", "linear_lt", "mm_lt"))))
    L += ["", "## Best CUTLASS config per shape (top 5)", ""]
    for key in keys:
        cand = sorted([(r["cells"][key]["us"], r) for r in rows if key in r["cells"]], key=lambda z: z[0])[:5]
        t = tor.get(key, {})
        L.append("**%s** (torch.mm %.2f, best torch %.2f)" % (key, t.get("torch_mm", float("nan")), min(t.values()) if t else float("nan")))
        L.append("")
        L.append("| config | variant | us | ratio | ratio vs best torch | regs | spill | smem | occ |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for us, r in cand:
            c = r["cells"][key]
            L.append("| %s | %s | %.2f | %.3f | %.3f | %s | %s | %s | %s |" % (r["name"], c["var"], us, c["ratio"], c["ratio_best"],
                                                                         r["regs"], r["spill"], r["smem"], r["occ"]))
        L.append("")
    L += ["## Full table (us / ratio vs torch.mm, best variant)", "",
          "| config | regs | spill st/ld | smem | stages | occ | compile s | " + " | ".join(keys) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(keys)]
    for r in sorted(rows, key=lambda r: r["name"]):
        if not r["ok"]:
            L.append("| %s | build FAIL: %s |" % (r["name"], (r["err"] or "").replace("|", "/")[:150]))
            continue
        cells = []
        for key in keys:
            c = r["cells"].get(key)
            cells.append("%.1f / %.2f" % (c["us"], c["ratio"]) if c else ("crash" if r["crashed"] else "-"))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (r["name"], r["regs"], r["spill"] + (" SER" if r["serialized"] else ""),
                                                             r["smem"], r["stages"], r["occ"], r["compile_s"], " | ".join(cells)))
    cf = res.get("confirm")
    if cf:
        C = ["## Confirmation re-run (interleaved, %d rounds x %d reps; median of rounds)" % (cf["rounds"], cf["reps"]), "",
             "Configs: %s. Each cell: best variant us (ratio vs best torch = min of torch.mm / F.linear / F.linear cublaslt)." % ", ".join(cf["names"]), "",
             "| shape | torch.mm | F.linear | F.linear lt | " + " | ".join(cf["names"]) + " |", "|---|---|---|---|" + "---|" * len(cf["names"])]
        for key in keys:
            u = cf["us"].get(key)
            if not u:
                continue
            tb = min(u["torch_mm"], u["linear"], u["linear_lt"])
            cells = []
            for nm in cf["names"]:
                vs = [(v, k.split("/")[1]) for k, v in u.items() if k.split("/")[0] == nm and "/" in k]
                cells.append("%.2f %s (%.3f)" % (min(vs)[0], min(vs)[1], min(vs)[0] / tb) if vs else "-")
            C.append("| %s | %.2f | %.2f | %.2f | %s |" % (key, u["torch_mm"], u["linear"], u["linear_lt"], " | ".join(cells)))
        C += ["", "cuBLAS kernels per shape (torch.mm / F.linear / F.linear cublaslt):", ""]
        for key in keys:
            C.append("- %s: %s" % (key, cf.get("cublas_kernels", {}).get(key)))
        C.append("")
        i = L.index("## Best CUTLASS config per shape (top 5)")
        L[i:i] = C
    MD.write_text("\n".join(L) + "\n")
    print("wrote %s (%d configs) and %s" % (MD, len(rows), JSON), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "bench", "report", "list", "confirm"])
    ap.add_argument("--names", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--set", default="shake")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retorch", action="store_true")
    args = ap.parse_args()
    if args.cmd == "list":
        for c in select(args.set):
            print(cfg_name(c))
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "confirm":
        cmd_confirm(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
