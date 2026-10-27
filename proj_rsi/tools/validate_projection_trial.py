"""Read-only evidence gate for the four-arm projection prototype labs (not NCU)."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


PACKED_MAINS = {"px": "rq_px", "smx": "rq_stmx", "dp": "rq_px_dp", "ds": "rq_px_dp_stmx"}


def passing_recorded_statuses(record, label):
    """Check saved nested outcomes; this does not rerun physical SASS checks."""
    if isinstance(record, dict):
        for key, value in record.items():
            if key == "status":
                require(isinstance(value, str) and value.startswith("PASS"), "recorded subgate failed: " + label)
            passing_recorded_statuses(value, label + "." + key)
    elif isinstance(record, list):
        for i, value in enumerate(record):
            passing_recorded_statuses(value, label + "." + str(i))


def passing_oracle(value):
    return isinstance(value, str) and value.startswith(("relative L2 error ", "exact zero complete output"))


def validate(run, candidate_dir=None, artifact_dir=None):
    status = run.get("status", "")
    require(status.startswith(("PX_", "STMX_", "DP_", "DS_", "INCUMBENT_GAIN_")) and
            "INVALID" not in status, "not a completed valid prototype status")
    require(run.get("graph_cleanup") == "RESET_ALL_FOUR", "graph cleanup incomplete")
    require(run.get("weights_inputs_counters") == "UNCHANGED_OR_RESET", "immutable/counter gate incomplete")
    t = run["timing"]; raw = t["F1_samples_us"]
    candidates = [k for k in raw if not k.endswith(("_before", "_after"))]
    require(len(candidates) == 1, "ambiguous candidate F1 series")
    candidate = candidates[0]; rows = t["paired_graph_us"]
    require(len(rows) == 21, "expected all21 paired rounds")
    labels = set(rows[0]); controls = labels - {candidate}
    require(len(labels) == 4 and "incumbent" in controls, "four arms with actual incumbent required")
    require(candidate in PACKED_MAINS and labels <= set(PACKED_MAINS) | {"incumbent", "v1", "v4"}, "unknown prototype label")
    packed_labels = labels & set(PACKED_MAINS)
    expected_keys = {candidate} | {c + side for c in controls for side in ("_before", "_after")}
    require(set(raw) == expected_keys, "missing or extra before/after F1 series")
    for key, values in raw.items():
        require(len(values) == 21 and all(positive(v) for v in values), "invalid F1 samples: " + key)
        close(t["F1_medians_us"][key], statistics.median(values), "incorrect F1 median: " + key)
    require(all(set(row) == labels and all(positive(v) for v in row.values()) for row in rows), "invalid paired samples")
    gains = {}; paired = {}; wins = {}
    for c in sorted(controls):
        gains[c] = min(statistics.median(raw[c + "_before"]), statistics.median(raw[c + "_after"])) - statistics.median(raw[candidate])
        paired[c] = statistics.median(row[c] - row[candidate] for row in rows)
        wins[c] = sum(row[c] > row[candidate] for row in rows)
        close(t["F1_" + candidate + "_gain_us"][c], gains[c], "incorrect F1 gain: " + c)
        close(t["paired_" + candidate + "_median_gain_us"][c], paired[c], "incorrect paired gain: " + c)
        require(t["paired_" + candidate + "_wins"][c] == wins[c], "incorrect win count: " + c)
    jit = run["actual_JIT_SASS"]
    jit = {candidate: jit} if "artifact" in jit else jit
    require(set(jit) == packed_labels, "JIT identities missing/unknown")
    require(set(run["profile"]) == labels, "profile arm identities missing/unknown")
    for label, record in jit.items():
        require(str(record["retirement_gate"]["status"]).startswith("PASS"), "JIT physical gate failed: " + label)
        passing_recorded_statuses(record, label)
        if artifact_dir:
            path = Path(artifact_dir) / Path(record["artifact"]).name
            require(path.is_file(), "missing actual JIT artifact: " + str(path))
            require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], "actual JIT hash drift")
        events = run["profile"][label]
        require(len(events) == 2 and sum("rx_pack" in e["name"] for e in events) == 1 and
                sum(PACKED_MAINS[label] + "<" in e["name"] for e in events) == 1 and
                all(positive(e["duration_ns"]) for e in events), "full pack+main source identity: " + label)
    require("FRESH_PACK_PLUS_MAIN_2_NODES" in run["full_pipeline_in_timed_graph"], "incomplete timed pipeline")
    mutations = {"original", "zero", "fresh_normal", "fresh_outliers", "all_probes"}
    checks = run["actual_timed_graph_checks"]
    require(len(checks) == 20 and {(r["graph"], r["input"]) for r in checks} ==
            {(label, kind) for label in labels for kind in mutations}, "incomplete/duplicate real graph checks")
    for r in checks:
        require(r["NaN_poison_overwritten"] is True and r["input_immutable"] is True and
                passing_oracle(r["oracle"]), "failed real graph oracle")
        if r["graph"] in jit:
            require(r.get("packed_poison_overwritten_byte_exact") is True, "packed workspace poison/byte gate missing")
    for label in labels:
        proof = run["validation"][label]
        require(any(c.get("all4096_onehot_columns") == "EXACT" and c.get("random_FP64") == "PASS" for c in proof), "fullK/FP64 gate missing: " + label)
        require({c.get("official_input_dist") for c in proof} >= {"normal", "outliers"}, "normal/outlier gate missing")
    bits = run["GPU_pack_all65536_BF16_storage_bits"]
    bits = {candidate: bits} if isinstance(bits, str) else bits
    fresh = run["fresh_workspace_and_output"]
    fresh = {candidate: fresh} if "status" in fresh else fresh
    require(set(bits) == packed_labels == set(fresh), "packed freshness/bit coverage identities differ")
    require(all(str(v).startswith("PASS") for v in bits.values()) and
            all(v["status"] == "PASS" and passing_oracle(v["oracle"]) for v in fresh.values()), "packed bit/fresh workspace gate failed")
    attrs = run["attrs"]
    require(set(attrs) == labels - {"incumbent"}, "runtime resource identities missing/unknown")
    packer_attrs = run["packer_attrs"]
    packer_attrs = {candidate: packer_attrs} if "regs" in packer_attrs else packer_attrs
    require(set(packer_attrs) == packed_labels, "packer resource identities missing/unknown")
    for label, attr in packer_attrs.items():
        require(positive(attr["regs"]) and attr["regs"] <= 64 and attr["local"] == 0 and attr["max_threads"] >= 256, "packer resource gate failed: " + label)
    for label, attr in attrs.items():
        threads = attr.get("threads", 384)
        require(threads == (512 if label in {"dp", "ds"} else 384), "wrong main launch geometry: " + label)
        require(positive(attr["regs"]) and positive(attr["smem"]) and attr["local"] == 0 and attr["occupancy_after_optin"] >= 1 and
                attr["max_dynamic_after"] >= attr["smem"], "runtime resource gate failed")
        if label == "v1":
            require(attr["register_mode"] == "64/216 dynamic frozen v1" and threads == 384, "unknown dynamic register policy")
            pool = 64 * 128 + 216 * 256
            initial_pool = ((attr["regs"] + 7) // 8) * 8 * threads
            require(pool <= initial_pool <= 65536, "dynamic pool exceeds initial reservation or SM capacity")
        else:
            pool = ((attr["regs"] + 7) // 8) * 8 * threads
        require(pool <= 65536, "Hopper physical register pool overflow")
        if threads == 512:
            require(attr["regs"] <= 128 and attr["max_threads"] >= 512, "512thread actual maxThreads/register gate")
    if candidate_dir:
        root = Path(candidate_dir); manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "source drift: " + name)
        require(manifest["source_sha256"] == run["candidate_source_sha256"][candidate], "run source mismatch")
        source = (root / "extension_source.cu").read_text()
        names = re.findall(r"__global__ void __launch_bounds__\([^)]*\)\s*(\w+)\(", source)
        require(len(names) == 1, "ambiguous candidate main kernel declaration")
        require(sum(names[0] + "<" in e["name"] for e in run["profile"][candidate]) == 1,
                "profiled candidate main differs from frozen source")
    return {"status": "VALID_WHOLE_PIPELINE_EVIDENCE", "candidate": candidate,
            "F1_us": statistics.median(raw[candidate]), "F1_gains_us": gains,
            "paired_gains_us": paired, "paired_wins": wins,
            "eligible_for_incumbent_review": gains["incumbent"] >= .5 and paired["incumbent"] >= .5,
            "mixed_comparisons": [c for c in controls if (gains[c] > 0) != (paired[c] > 0)],
            "source_files_verified": bool(candidate_dir), "JIT_artifacts_verified": bool(artifact_dir),
            "limits": "Checks saved evidence consistency. Candidate manifest/source and recorded SASS hashes are checked when paths are supplied; control sources/harness at capture are not attested. Numerical kernels, physical checkers, host exit and GPU cleanup require separate evidence. H100 local evidence is not an official score."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--candidate-dir", type=Path)
    ap.add_argument("--artifact-dir", type=Path)
    ap.add_argument("--negative-controls", action="store_true")
    args = ap.parse_args(); run = json.loads(args.result.read_text())
    result = validate(run, args.candidate_dir, args.artifact_dir)
    if args.negative_controls:
        failures = []
        corruptions = [lambda d: d["actual_timed_graph_checks"].pop(),
                       lambda d: d.update(graph_cleanup="PENDING"),
                       lambda d: d["timing"]["paired_graph_us"][0].update(incumbent=float("nan")),
                       lambda d: d["timing"]["F1_medians_us"].update({result["candidate"]: -1})]
        packed_label = next(iter(run["packer_attrs"])) if "regs" not in run["packer_attrs"] else result["candidate"]
        def delete_packed_control(d):
            if "artifact" in d["actual_JIT_SASS"]:
                d["actual_JIT_SASS"] = {}
                d["GPU_pack_all65536_BF16_storage_bits"] = {}
                d["fresh_workspace_and_output"] = {}
                d["attrs"].pop(result["candidate"])
            else:
                for key in ("actual_JIT_SASS", "GPU_pack_all65536_BF16_storage_bits", "fresh_workspace_and_output", "attrs"):
                    d[key].pop(packed_label)
        def fail_nested_gate(d):
            jit = d["actual_JIT_SASS"]
            record = jit if "artifact" in jit else jit[packed_label]
            record["retirement_gate"]["binaries"][0]["status"] = "FAIL: injected physical mismatch"
        def fail_store_gate(d):
            jit = d["actual_JIT_SASS"]
            record = jit if "artifact" in jit else jit.get("smx", jit[packed_label])
            record["store_form_gate"] = {"status": "FAIL: injected store mismatch"}
        def wrong_profile_main(d):
            for event in d["profile"][packed_label]:
                if "rx_pack" not in event["name"]:
                    event["name"] = "wrong_main_shader"
        corruptions.extend([delete_packed_control, lambda d: d.update(attrs={}), fail_store_gate,
                            lambda d: d["actual_timed_graph_checks"][0].update(oracle="FAIL: deliberately bad output"),
                            wrong_profile_main, fail_nested_gate, lambda d: d.update(packer_attrs={})])
        def oversized_packer(d):
            attrs = d["packer_attrs"]
            (attrs if "regs" in attrs else attrs[packed_label])["regs"] = 1000
        corruptions.extend([oversized_packer,
                            lambda d: d["attrs"][result["candidate"]].update(threads=384 if result["candidate"] in {"dp", "ds"} else 512),
                            lambda d: d["attrs"][result["candidate"]].update(regs=1000)])
        if "v1" in run["attrs"]:
            corruptions.extend([lambda d: d["attrs"]["v1"].update(regs=160),
                                lambda d: d["attrs"]["v1"].update(regs=240)])
        for change in corruptions:
            damaged = copy.deepcopy(run); change(damaged)
            try:
                validate(damaged)
            except ValueError:
                failures.append(True)
            else:
                raise ValueError("negative control unexpectedly accepted")
        result["negative_controls_rejected"] = len(failures)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
