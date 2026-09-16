"""Post-measurement reporting-only repair for refused-window seal counters.

The frozen analyzer looks only at window.controller_metadata, whereas rejected
windows store the already-performed controller work in rejection.controller_metadata.
Preserve summary.json and every measured record. Write summary_corrected.json
plus a provenance sidecar enumerating every changed JSON leaf. No timing,
correctness, fault, integrity-gate, source-freeze or snapshot result is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


COUNTERS = ("seal_calls", "seal_logical_bytes", "seal_transfer_calls", "seal_transfer_bytes", "seal_peak_batch_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonstandard JSON number: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def case_key(row):
    return row["workload_id"], row["scenario"], row["seed"], row["variant"]


def distribution(values):
    values = sorted(float(value) for value in values)
    require(bool(values) and all(math.isfinite(v) for v in values), "Invalid counter distribution")
    def quantile(p):
        offset = p * (len(values) - 1)
        lo, hi = math.floor(offset), math.ceil(offset)
        return values[lo] + (values[hi] - values[lo]) * (offset - lo)
    return {"n": len(values), "min": values[0], "q25": quantile(.25), "median": median(values), "q75": quantile(.75), "max": values[-1], "mean": mean(values)}


def leaf_differences(left, right, path=""):
    require(type(left) is type(right) or isinstance(left, (int, float)) and isinstance(right, (int, float)), "Unexpected structural type change: " + path)
    if isinstance(left, dict):
        require(set(left) == set(right), "Unexpected dictionary structure change: " + path)
        return [entry for key in left for entry in leaf_differences(left[key], right[key], path + "/" + str(key))]
    if isinstance(left, list):
        require(len(left) == len(right), "Unexpected list length change: " + path)
        return [entry for i, (a, b) in enumerate(zip(left, right)) for entry in leaf_differences(a, b, path + "/" + str(i))]
    return [] if left == right else [{"json_pointer": path, "before": left, "after": right}]


def corrected(original, rows):
    result = deepcopy(original)
    raw_cases = defaultdict(list)
    for row in rows:
        raw_cases[case_key(row)].append(row)
    affected_cases, allowed_prefixes = [], []
    for ci, case in enumerate(result["case_medians"]):
        repeats = raw_cases[case_key(case)]
        require(len(repeats) == case["timing_repetitions"], "Case repetition coverage mismatch")
        for wi, window in enumerate(case["windows"]):
            raw_windows = [r["windows"][wi] for r in repeats]
            require(all(w["status"] == window["status"] for w in raw_windows), "Case/window status mismatch")
            if window["status"] != "rejected":
                continue
            require(case["scenario"] == "checkpoint_corruption" and all(w["valid"] is True and
                    w["rejection"]["reason"] == "Checkpoint integrity failure" for w in raw_windows), "Unexpected refusal case")
            values = []
            for w in raw_windows:
                counters = w["rejection"]["controller_metadata"]
                require(all(type(counters[k]) in (int, float) and math.isfinite(counters[k]) and counters[k] >= 0 for k in COUNTERS), "Missing/invalid raw refusal counters")
                values.append(counters)
            window["counter_medians"] = {k: median(v[k] for v in values) for k in COUNTERS}
            allowed_prefixes.append(f"/case_medians/{ci}/windows/{wi}/counter_medians/")
            affected_cases.append({"case": list(case_key(case)), "window_index": wi, "timing_repetitions": len(repeats)})
    # Regenerate only refused-window counter distributions. The grouping and
    # repetition hierarchy remain exactly those of the frozen descriptive analysis.
    for gi, group in enumerate(result["groups"]):
        selected = [c for c in result["case_medians"] if all(c[k] == group[k] for k in ("model", "prefix_tokens", "window", "scenario", "variant"))]
        for si, stratum in enumerate(group["window_strata"]):
            if stratum["status"] != "rejected":
                continue
            windows = [w for c in selected for w in c["windows"] if w["status"] == stratum["status"] and w["fault_affected"] == stratum["fault_affected"]]
            require(len(windows) == stratum["case_windows"], "Counter-stratum coverage mismatch")
            stratum["counters"] = {k: distribution(w["counter_medians"][k] for w in windows) for k in COUNTERS}
            allowed_prefixes.append(f"/groups/{gi}/window_strata/{si}/counters/")
    changes = leaf_differences(original, result)
    require(all(any(c["json_pointer"].startswith(prefix) for prefix in allowed_prefixes) for c in changes), "Correction altered an unauthorized field")
    return result, changes, affected_cases


def run(directory):
    targets = [directory / "summary_corrected.json", directory / "analysis_corrections.json"]
    require(not any(path.exists() for path in targets), "Refusing to overwrite correction artifacts")
    metadata = read(directory / "metadata.json")
    original = read(directory / "summary.json")
    require(metadata["status"] == "complete" and original["status"] == "complete_descriptive_stage3", "Incomplete source evidence")
    input_hashes = {"summary.json": sha(directory / "summary.json")}
    for label in ("metadata", "sessions", "references", "warmups", "profiles"):
        filename = label + (".json" if label == "metadata" else ".jsonl")
        input_hashes[filename] = sha(directory / filename)
        require(original["input_sha256"][label] == input_hashes[filename], "Summary/raw source binding mismatch: " + filename)
        if label != "metadata":
            require(metadata[label + "_sha256"] == input_hashes[filename], "Metadata/raw binding mismatch: " + filename)
    rows = [json.loads(line) for line in (directory / "sessions.jsonl").read_text().splitlines() if line.strip()]
    require(len(rows) == metadata["sessions_completed"] == original["coverage"]["measured_sessions"], "Raw session coverage mismatch")
    require(all(r["valid"] is True and r["source_sha256"] == metadata["source_sha256"] and
                r["config_sha256"] == metadata["config_sha256"] for r in rows), "Invalid or misbound raw row")
    result, changes, affected = corrected(original, rows)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    provenance = {"status": "reporting_only_correction", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "correction_script_sha256": sha(__file__), "model": metadata["model"], "input_sha256": input_hashes,
        "corrected_summary_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "explanation": "The frozen analyzer defaulted rejected-window seal counters to zero because it read top-level controller_metadata. The raw rejected window correctly retained performed work in rejection.controller_metadata. This sidecar recomputes only those case counter medians and their grouped counter distributions.",
        "unchanged": ["summary.json", "all raw sessions and reference evidence", "all measured timing values and ratios", "all correctness/refusal results", "all integrity gates", "all source-freeze files", "all tensor snapshot evidence"],
        "affected_case_windows": affected, "changed_json_leaf_count": len(changes), "changes": changes,
        "usage": "Use summary_corrected.json together with this sidecar for refusal counter displays. Original summary.json remains the immutable frozen-analyzer output. This is post-measurement reporting, not a rerun or an experiment amendment."}
    with targets[0].open("x") as stream:
        stream.write(serialized)
    with targets[1].open("x") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(all(sha(directory / filename) == value for filename, value in input_hashes.items()), "An input changed while correcting")
    return {"directory": str(directory), "model": metadata["model"], "affected_case_windows": len(affected), "changed_json_leaves": len(changes), "corrected_summary_sha256": provenance["corrected_summary_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([run(directory) for directory in args.directories], indent=2))


if __name__ == "__main__":
    main()
