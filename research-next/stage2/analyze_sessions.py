"""Derive stage-2 counts and paired timings from recorded transactional sessions.

This analysis checks record consistency, not saved GPU tensors or physical
fault resilience. No confidence interval treats repeated measurements of the
same controlled fault as independent faults. No general novelty or deployment
claim is derived from these compact-model, single-request experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, probability):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    values = list(values)
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "q25": quantile(values, .25), "q75": quantile(values, .75),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None}


def analyze(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "sessions.jsonl").read_text().splitlines() if line]
    refs = [json.loads(line) for line in (directory / "references.jsonl").read_text().splitlines() if line]
    if metadata.get("status") != "complete":
        raise ValueError("Refusing complete-run analysis of incomplete metadata")
    if sha(directory / "sessions.jsonl") != metadata["sessions_sha256"] or sha(directory / "references.jsonl") != metadata["references_sha256"]:
        raise ValueError("Evidence hash mismatch")
    if len(rows) != metadata["expected_sessions"] or len(rows) != metadata["sessions_completed"]:
        raise ValueError("Session cardinality mismatch")
    unique, grouped, window_groups = set(), defaultdict(list), defaultdict(list)
    fault_counts, target_counts, rejection_counts = Counter(), Counter(), Counter()
    for row in rows:
        key = (row["workload_id"], row["scenario"], row["method"], row["repetition"])
        if key in unique:
            raise ValueError("Duplicate session key")
        unique.add(key)
        if not row["valid"] or row["config_sha256"] != metadata["config_sha256"] or row["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("Invalid or misbound session")
        windows = row["windows"]
        if not windows or [w["window_index"] for w in windows] != list(range(len(windows))):
            raise ValueError("Noncontiguous windows")
        if row["scenario"] == "checkpoint_corruption":
            if row["status"] != "expected_abort" or len(windows) != 2 or row["windows_committed"] != 1 or not row["poisoned"]:
                raise ValueError("Checkpoint corruption did not abort exactly the second window")
        elif row["status"] != "complete" or row["windows_committed"] != row["n_windows"] or row["poisoned"]:
            raise ValueError("Unexpected session termination")
        if len(row["committed_tokens"]) != row["windows_committed"] * row["window"]:
            raise ValueError("Atomic commit token count mismatch")
        if not math.isclose(row["total_wall_seconds"], sum(w["wall_seconds"] for w in windows), rel_tol=1e-12):
            raise ValueError("Session timing sum mismatch")
        grouped[(row["model"], row["scenario"], row["method"])].append(row)
        for window in windows:
            if not window["valid"] or not window["no_early_commit"]:
                raise ValueError("Invalid window or premature token commit")
            for field in ("wall_seconds", "wall_minus_injection_seconds"):
                if not isinstance(window[field], (float, int)) or not math.isfinite(window[field]) or window[field] <= 0:
                    raise ValueError("Nonpositive/nonfinite timing")
            if window["injection_harness_seconds"] < 0 or not math.isclose(window["wall_minus_injection_seconds"],
                    window["wall_seconds"] - window["injection_harness_seconds"], rel_tol=1e-12):
                raise ValueError("Injection timing accounting mismatch")
            if window["status"] == "committed":
                if not window["checks"]["all_equal"] or len(window["tokens"]) != row["window"]:
                    raise ValueError("Commit oracle failed")
                if window["committed_after"] - window["committed_before"] != row["window"]:
                    raise ValueError("Partial window committed")
            elif window["status"] == "rejected":
                if not all(window["checks"].values()) or window["committed_after"] != window["committed_before"]:
                    raise ValueError("Rejection atomicity failed")
                rejection_counts[window["rejection"]["reason"]] += 1
            else:
                raise ValueError("Unknown window status")
            for fault in window["faults"]:
                changed = sum(a != b for a, b in zip(fault["before"], fault["after"]))
                if not changed or changed != fault["changed_elements"] or not all(math.isfinite(x) for x in fault["after"]):
                    raise ValueError("Recorded fault is unchanged/nonfinite")
                target_counts[fault["target"]] += 1
            fault_counts[len(window["faults"])] += 1
            window_groups[(row["model"], row["scenario"], row["method"])].append(window)
    expected = {(workload["id"], scenario, method, rep)
                for workload in metadata["workloads"] for scenario in metadata["config"]["scenarios"]
                for method in metadata["config"]["methods"] for rep in range(metadata["config"]["repetitions"])}
    if unique != expected or len(refs) != len(metadata["workloads"]):
        raise ValueError("Frozen matrix or reference coverage mismatch")
    for ref in refs:
        if not all(p["cache"]["equal"] and p["logits"]["equal"] and p["token_equal"] for p in ref["parity"]):
            raise ValueError("Native parity evidence failed")
    summaries = []
    for (model, scenario, method), group in sorted(grouped.items()):
        windows = window_groups[(model, scenario, method)]
        summaries.append({"model": model, "scenario": scenario, "method": method,
            "sessions": len(group), "windows_attempted": len(windows),
            "windows_committed": sum(w["status"] == "committed" for w in windows),
            "windows_rejected": sum(w["status"] == "rejected" for w in windows),
            "recovered_windows": sum(bool(w.get("recovered")) for w in windows),
            "session_wall_seconds": distribution(r["total_wall_seconds"] for r in group),
            "window_wall_seconds": distribution(w["wall_seconds"] for w in windows),
            "window_wall_minus_injection_seconds": distribution(w["wall_minus_injection_seconds"] for w in windows),
            "peak_increment_bytes": distribution(w["peak_increment_bytes"] for w in windows),
            "logical_checkpoint_bytes": distribution(w["logical_checkpoint_bytes"] for w in windows),
            "logical_journal_bytes": distribution(w["logical_journal_bytes"] for w in windows),
            "fallback_reasons": dict(Counter(w.get("fallback_reason") for w in windows if w.get("fallback_reason"))),
            "selected_cuts": dict(Counter(str(w.get("selected_cut")) for w in windows if w.get("recovered")))})
    paired, paired_groups = {}, defaultdict(list)
    for row in rows:
        key = (row["workload_id"], row["scenario"], row["repetition"])
        paired.setdefault(key, {})[row["method"]] = row
    pair_records = []
    for key, pair in sorted(paired.items()):
        if set(pair) != {"full", "sparse"}:
            raise ValueError("Unpaired policy rows")
        full, sparse = pair["full"], pair["sparse"]
        if len(full["windows"]) != len(sparse["windows"]):
            raise ValueError("Paired windows have different lengths")
        same_treatment = key[1] != "journal_corruption"
        if same_treatment and [w["faults"] for w in full["windows"]] != [w["faults"] for w in sparse["windows"]]:
            raise ValueError("Matched policies received different finite perturbations")
        item = {"workload_id": key[0], "scenario": key[1], "repetition": key[2], "model": full["model"],
                "same_policy_treatment": same_treatment,
                "session_full_over_sparse_wall_ratio": full["total_wall_seconds"] / sparse["total_wall_seconds"],
                "window_ratios": []}
        for fw, sw in zip(full["windows"], sparse["windows"]):
            ratio = {"window_index": fw["window_index"], "status": fw["status"],
                     "full_over_sparse_wall_ratio": fw["wall_seconds"] / sw["wall_seconds"],
                     "full_over_sparse_adjusted_ratio": fw["wall_minus_injection_seconds"] / sw["wall_minus_injection_seconds"],
                     "fault_affected": bool(fw["faults"]), "same_policy_treatment": same_treatment}
            item["window_ratios"].append(ratio)
            if same_treatment:
                paired_groups[(full["model"], key[1], ratio["fault_affected"], ratio["status"])].append(ratio)
        pair_records.append(item)
    timing_summary = [{"model": model, "scenario": scenario, "fault_affected": fault, "status": status,
                       "wall_full_over_sparse_ratio": distribution(x["full_over_sparse_wall_ratio"] for x in values),
                       "adjusted_full_over_sparse_ratio": distribution(x["full_over_sparse_adjusted_ratio"] for x in values)}
                      for (model, scenario, fault, status), values in sorted(paired_groups.items())]
    all_windows = [w for r in rows for w in r["windows"]]
    return {"status": "complete", "metadata_sha256": sha(directory / "metadata.json"),
            "sessions_sha256": sha(directory / "sessions.jsonl"), "references_sha256": sha(directory / "references.jsonl"),
            "analyzer_sha256": sha(__file__), "sessions": len(rows), "workloads": len(refs),
            "windows_attempted": len(all_windows), "windows_committed": sum(w["status"] == "committed" for w in all_windows),
            "windows_rejected": sum(w["status"] == "rejected" for w in all_windows),
            "recovered_windows": sum(bool(w.get("recovered")) for w in all_windows),
            "fault_target_counts": dict(target_counts), "faults_per_window_histogram": dict(fault_counts),
            "rejection_reasons": dict(rejection_counts), "all_recorded_checks_pass": True,
            "native_parity_positions": sum(len(r["parity"]) for r in refs),
            "groups": summaries, "paired_timing_summary": timing_summary, "pairs": pair_records,
            "interpretation_limits": ["Record-level audit only; gold tensors were not persisted.",
                "Repeated deterministic fault trials are not independent faults.",
                "Fault-injection work is included in raw latency; subtraction is diagnostic only.",
                "Journal-corruption treatment differs by policy and is excluded from matched timing aggregates.",
                "No-fault, fault-affected and rejection windows remain separate; no overall inference throughput claim.",
                "This does not establish production-serving feasibility, physical-fault coverage, or research novelty."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "sessions", "windows_attempted", "windows_committed",
                      "windows_rejected", "recovered_windows", "all_recorded_checks_pass")}, indent=2))


if __name__ == "__main__":
    main()
