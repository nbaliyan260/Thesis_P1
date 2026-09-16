"""Independent stage-3 report quantities reconstructed from raw session records.

This script imports no experiment/controller/analyzer modules. It calculates
from sessions.jsonl before optionally comparing the existing summary.json.
It audits record arithmetic, not saved GPU tensors or physical fault coverage.
The output is derived evidence; raw run directories are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


VARIANTS = ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched", "full_legacy", "sparse_legacy")
BATCHED = VARIANTS[2:5]
MEMORY = ("logical_input_cache_bytes", "logical_checkpoint_bytes", "logical_journal_bytes", "allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes")
COUNTERS = ("seal_calls", "seal_logical_bytes", "seal_transfer_calls", "seal_transfer_bytes", "seal_peak_batch_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(text):
    def reject(value):
        raise ValueError("Nonstandard JSON numeric value: " + value)
    return json.loads(text, parse_constant=reject)


def read(path):
    return parse(Path(path).read_text())


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-13)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "q25": None, "q75": None}
    require(all(math.isfinite(v) for v in values), "Nonfinite distribution input")
    def percentile(fraction):
        index = fraction * (len(values) - 1)
        whole = math.floor(index)
        following = min(whole + 1, len(values) - 1)
        return values[whole] + (index - whole) * (values[following] - values[whole])
    return {"n": len(values), "median": median(values), "min": values[0], "max": values[-1], "q25": percentile(.25), "q75": percentile(.75)}


def identity(row):
    return row["workload_id"], row["scenario"], row["seed"], row["variant"]


def load_run(directory):
    metadata = read(directory / "metadata.json")
    require(metadata["status"] == "complete", "Incomplete run: " + str(directory))
    provenance = {"directory": str(directory.resolve()), "metadata_sha256": sha(directory / "metadata.json")}
    datasets = {}
    for name in ("sessions", "references", "warmups", "profiles"):
        path = directory / (name + ".jsonl")
        provenance[name + "_sha256"] = sha(path)
        require(provenance[name + "_sha256"] == metadata[name + "_sha256"], "Recorded hash differs: " + name)
        datasets[name] = [parse(line) for line in path.read_text().splitlines() if line.strip()]
    cfg, model = metadata["config"], metadata["model"]
    workloads = {f"{model.replace('/', '--')}--{p['id']}--p{s['prefix_tokens']}--w{s['window']}"
                 for p in cfg["prompts"] for s in cfg["shapes"]}
    require({r["workload_id"] for r in datasets["references"]} == workloads
            and len(datasets["references"]) == len(workloads), "Reference workload coverage mismatch")
    expected = set()
    for workload in workloads:
        for rep in range(cfg["clean_repetitions"]):
            expected.update((workload, "clean", cfg["seeds"][0], v, rep) for v in VARIANTS)
        for scenario in cfg["fault_cases"]:
            variants = BATCHED + (VARIANTS[5:] if scenario in cfg["legacy_cases"] else ())
            for seed in cfg["seeds"]:
                for rep in range(cfg["repetitions"]):
                    expected.update((workload, scenario, seed, v, rep) for v in variants)
        for scenario in cfg["guard_cases"]:
            expected.update((workload, scenario, cfg["seeds"][0], v, 0) for v in BATCHED)
    keys = [identity(r) + (r["repetition"],) for r in datasets["sessions"]]
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, "Main matrix is not exactly complete")
    require(len(keys) == metadata["sessions_completed"] == metadata["expected_sessions"], "Metadata session count mismatch")
    for row in datasets["sessions"]:
        require(row["valid"] is True and row["profile"] is False, "Invalid/profiled main row")
        require(row["model"] == model and row["model_revision"] == metadata["model_revision"]
                and row["config_sha256"] == metadata["config_sha256"] and row["source_sha256"] == metadata["source_sha256"], "Raw row provenance mismatch")
        windows = row["windows"]
        require([w["window_index"] for w in windows] == list(range(len(windows))), "Window index mismatch")
        require(close(row["total_wall_seconds"], sum(w["wall_seconds"] for w in windows)), "Raw window/session sum mismatch")
        require(positive(row["session_initialization_seconds"]) and close(row["total_with_initialization_seconds"],
                row["total_wall_seconds"] + row["session_initialization_seconds"]), "Constructor sum mismatch")
        expected_abort = row["scenario"] == "checkpoint_corruption"
        require(row["status"] == ("expected_abort" if expected_abort else "complete")
                and len(windows) == (2 if expected_abort else cfg["n_windows"])
                and row["windows_committed"] == (1 if expected_abort else cfg["n_windows"]), "Unexpected session termination")
        for w in windows:
            require(w["valid"] is True and w["no_early_commit"] is True and positive(w["wall_seconds"]), "Invalid raw window")
            require(close(w["wall_minus_injection_seconds"], w["wall_seconds"] - w["injection_harness_seconds"]), "Injection arithmetic mismatch")
            require(all(v["equal"] is True if isinstance(v, dict) else v is True for v in w["checks"].values()), "Raw check failed")
            if w["status"] == "committed":
                require(w["committed_after"] - w["committed_before"] == row["window"] == len(w["tokens"]), "Token commitment accounting mismatch")
                require(w["release_seconds"] == w["wall_seconds"], "Release time mismatch")
            else:
                require(w["status"] == "rejected" and w["committed_after"] == w["committed_before"]
                        and w["release_seconds"] is None, "Invalid rejection")
    return metadata, datasets, provenance


def count_run(data):
    rows = data["sessions"]
    windows = [w for r in rows for w in r["windows"]]
    events = [f for w in windows for f in w["faults"]]
    distinct_divergence = {(r["workload_id"], r["scenario"], r["seed"], r["variant"], w["window_index"])
                           for r in rows for w in r["windows"] if w.get("first_divergence") is not None}
    counts = {"measured_sessions": len(rows), "workloads": len(data["references"]), "windows_attempted": len(windows),
              "committed_windows": sum(w["status"] == "committed" for w in windows),
              "rejected_windows": sum(w["status"] == "rejected" for w in windows),
              "gold_checked_commits": sum(w["status"] == "committed" and w["checks"]["all_equal"] for w in windows),
              "recovered_windows": sum(bool(w.get("recovered")) for w in windows),
              "replay_windows_with_token_divergence": sum(w.get("first_divergence") is not None for w in windows),
              "partial_to_full_replay_windows": sum(any(c > 0 for c in w.get("replay_starts", [])) and 0 in w.get("replay_starts", []) for w in windows),
              "divergent_case_policy_windows_without_timing_repetitions": len(distinct_divergence),
              "committed_token_checks": sum(len(w.get("tokens", [])) for w in windows),
              "native_custom_parity_positions": sum(len(r["parity"]) for r in data["references"]),
              "warmup_sessions_excluded": len(data["warmups"]), "profiled_sessions_excluded": len(data["profiles"]),
              "snapshot_declarations": sum("snapshot" in w for w in windows),
              "snapshot_declared_bytes": sum(w.get("snapshot", {}).get("bytes", 0) for w in windows),
              "fault_event_targets": dict(Counter(f["target"] for f in events)),
              "fault_event_kinds": dict(Counter(f.get("kind", "journal_hidden") for f in events)),
              "fault_event_types": dict(Counter(f["fault_type"] for f in events)),
              "refusal_reasons": dict(Counter(w["rejection"]["reason"] for w in windows if w["status"] == "rejected")),
              "fallback_reasons": dict(Counter(w["fallback_reason"] for w in windows if w.get("fallback_reason")))}
    counts["case_policy_cells_without_timing_repetitions"] = len({identity(r) for r in rows})
    counts["configured_case_cells_without_policy_or_timing_repetitions"] = len({identity(r)[:3] for r in rows})
    starts = [(r["num_layers"], c) for r in rows for w in r["windows"] for c in w.get("replay_starts", [])]
    counts["measured_replay_token_steps"] = len(starts)
    counts["measured_partial_replay_token_steps"] = sum(c > 0 for _, c in starts)
    counts["measured_full_replay_token_steps"] = sum(c == 0 for _, c in starts)
    counts["measured_replay_layer_steps"] = sum(n - c for n, c in starts)
    counts["measured_skipped_layer_steps_within_replay"] = sum(c for _, c in starts)
    for r in rows:
        for w in r["windows"]:
            require(window_metadata(w).get("replay_layer_steps", 0) == sum(r["num_layers"] - c for c in w.get("replay_starts", [])), "Replay-layer accounting mismatch")
    counts["seal_counter_totals_across_measured_policy_repetitions"] = {
        name: sum(window_metadata(w).get(name, 0) for w in windows)
        for name in COUNTERS if name != "seal_peak_batch_bytes"}
    counts["largest_recorded_seal_batch_bytes"] = max(window_metadata(w).get("seal_peak_batch_bytes", 0) for w in windows)
    return counts


def window_metadata(window):
    return window.get("controller_metadata", window.get("rejection", {}).get("controller_metadata", {}))


def reduce_repetitions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[identity(row)].append(row)
    cells = {}
    for key, repeats in grouped.items():
        row = repeats[0]
        require(all([w["faults"] for w in r["windows"]] == [w["faults"] for w in row["windows"]] for r in repeats), "Treatment changes across repetitions")
        cell = {k: row[k] for k in ("workload_id", "model", "prompt_id", "prefix_tokens", "window", "scenario", "seed", "variant")}
        cell.update(n_repetitions=len(repeats), session=median(r["total_wall_seconds"] for r in repeats),
                    session_with_initialization=median(r["total_with_initialization_seconds"] for r in repeats),
                    initialization=median(r["session_initialization_seconds"] for r in repeats),
                    adjusted_session=median(r["total_wall_seconds"] - r["total_injection_harness_seconds"] for r in repeats), windows=[])
        for wi, first in enumerate(row["windows"]):
            ws = [r["windows"][wi] for r in repeats]
            item = {"index": wi, "status": first["status"], "faults": first["faults"],
                    "affected": any(f["target"] == "working_prefix" for f in first["faults"]),
                    "wall": median(w["wall_seconds"] for w in ws), "adjusted": median(w["wall_minus_injection_seconds"] for w in ws),
                    "memory": {name: median(w[name] for w in ws) for name in MEMORY}, "availability": {},
                    "counters": {name: median(window_metadata(w).get(name, 0) for w in ws) for name in COUNTERS},
                    "replay_layer_steps": median(sum(row["num_layers"] - c for c in w.get("replay_starts", [])) for w in ws),
                    "skipped_layer_steps": median(sum(w.get("replay_starts", [])) for w in ws)}
            if first["status"] == "committed":
                def metrics(w):
                    times, steps = w["token_availability_seconds"], w["step_completion_seconds"]
                    return {"first_token_seconds": times[0], "last_token_seconds": times[-1], "mean_token_seconds": mean(times),
                            "mean_hold_after_step_seconds": mean(a - b for a, b in zip(times, steps)),
                            "first_step_seconds": steps[0], "last_step_seconds": steps[-1], "release_seconds": w["release_seconds"]}
                observations = [metrics(w) for w in ws]
                item["availability"] = {k: median(v[k] for v in observations) for k in observations[0]}
            cell["windows"].append(item)
        cells[key] = cell
    return cells


def calculate_pairs(cells, config):
    session_pairs, window_pairs, costs, seals, sensitivity = [], [], [], [], []
    comparisons = (("full_batched", "sparse_batched"), ("full_batched", "allcuts_batched"),
                   ("sparse_batched", "allcuts_batched"), ("full_legacy", "sparse_legacy"))
    for key, left in cells.items():
        common = {k: left[k] for k in ("model", "workload_id", "prompt_id", "prefix_tokens", "window", "scenario", "seed")}
        if left["scenario"] not in ("journal_corruption", "checkpoint_corruption"):
            for numerator, denominator in comparisons:
                if left["variant"] != numerator:
                    continue
                right = cells.get(key[:3] + (denominator,))
                if right is None:
                    continue
                require([w["faults"] for w in left["windows"]] == [w["faults"] for w in right["windows"]], "Matched policy fault mismatch")
                session_pairs.append(dict(common, numerator=numerator, denominator=denominator, wall_ratio=left["session"] / right["session"],
                    with_initialization_ratio=left["session_with_initialization"] / right["session_with_initialization"], adjusted_ratio=left["adjusted_session"] / right["adjusted_session"]))
                for lw, rw in zip(left["windows"], right["windows"]):
                    window_pairs.append(dict(common, numerator=numerator, denominator=denominator, window_index=lw["index"],
                        fault_affected=lw["affected"], status=lw["status"], wall_ratio=lw["wall"] / rw["wall"], adjusted_ratio=lw["adjusted"] / rw["adjusted"]))
                    if numerator.startswith("full_") and denominator.startswith("sparse_") and lw["affected"]:
                        clean_key = (left["workload_id"], "clean", config["seeds"][0])
                        clean_full = cells[clean_key + (numerator,)]["windows"][lw["index"]]
                        clean_sparse = cells[clean_key + (denominator,)]["windows"][lw["index"]]
                        sensitivity.append(dict(common, numerator=numerator, denominator=denominator, window_index=lw["index"],
                            raw=break_even(clean_full["wall"], clean_sparse["wall"], lw["wall"], rw["wall"]),
                            injection_subtracted=break_even(clean_full["adjusted"], clean_sparse["adjusted"], lw["adjusted"], rw["adjusted"])))
        if left["scenario"] == "clean" and left["variant"] not in ("bare", "native"):
            for baseline in ("bare", "native"):
                right = cells[key[:3] + (baseline,)]
                costs.append(dict(common, protected=left["variant"], unprotected_baseline=baseline,
                    session_protected_over_baseline=left["session"] / right["session"],
                    session_with_initialization_protected_over_baseline=left["session_with_initialization"] / right["session_with_initialization"],
                    session_extra_ms=1000 * (left["session"] - right["session"]),
                    session_with_initialization_extra_ms=1000 * (left["session_with_initialization"] - right["session_with_initialization"])))
        if left["variant"].endswith("_legacy"):
            batched = left["variant"].replace("_legacy", "_batched")
            right = cells[key[:3] + (batched,)]
            require([w["faults"] for w in left["windows"]] == [w["faults"] for w in right["windows"]], "Sealing treatment mismatch")
            seals.append(dict(common, legacy=left["variant"], batched=batched,
                session_legacy_over_batched_wall_ratio=left["session"] / right["session"],
                session_legacy_over_batched_with_initialization_ratio=left["session_with_initialization"] / right["session_with_initialization"],
                session_legacy_over_batched_adjusted_ratio=left["adjusted_session"] / right["adjusted_session"]))
    return {"matched_policy_sessions": session_pairs, "matched_policy_windows": window_pairs,
            "protection_overhead": costs, "sealing_ablation": seals, "break_even_sensitivity": sensitivity}


def break_even(full_clean, sparse_clean, full_fault, sparse_fault):
    overhead, saving = sparse_clean - full_clean, full_fault - sparse_fault
    probability = None
    if overhead > 0 and saving > 0:
        probability = overhead / (overhead + saving)
        label = "extra_clean_cost_and_positive_fault_saving"
    elif overhead <= 0 and saving > 0:
        label = "no_extra_clean_cost_with_positive_fault_saving"
    elif overhead < 0 and saving == 0:
        label = "lower_clean_cost_equal_fault_cost"
    elif overhead < 0 and saving < 0:
        label = "lower_clean_cost_but_higher_fault_cost"
    elif overhead == 0 and saving == 0:
        label = "equal_observed_clean_and_fault_cost"
    else:
        label = "no_positive_fault_saving_with_nonnegative_clean_extra_cost"
    return {"H_seconds": overhead, "R_seconds": saving, "p_star": probability, "classification": label}


def grouped(rows, dimensions, measures):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in dimensions)].append(row)
    return [{"dimensions": dict(zip(dimensions, key)), "case_records": len(values),
             **{m: distribution(row[m] for row in values) for m in measures}}
            for key, values in sorted(buckets.items(), key=lambda kv: str(kv[0]))]


def aggregate_pair_records(pairs):
    result = {}
    declarations = {
        "matched_policy_sessions": (("model", "prefix_tokens", "window", "scenario", "numerator", "denominator"), ("wall_ratio", "with_initialization_ratio", "adjusted_ratio")),
        "matched_policy_windows": (("model", "prefix_tokens", "window", "scenario", "numerator", "denominator", "fault_affected", "status"), ("wall_ratio", "adjusted_ratio")),
        "protection_overhead": (("model", "prefix_tokens", "window", "protected", "unprotected_baseline"), ("session_protected_over_baseline", "session_with_initialization_protected_over_baseline", "session_extra_ms", "session_with_initialization_extra_ms")),
        "sealing_ablation": (("model", "prefix_tokens", "window", "scenario", "legacy", "batched"), ("session_legacy_over_batched_wall_ratio", "session_legacy_over_batched_with_initialization_ratio", "session_legacy_over_batched_adjusted_ratio"))}
    for name, (dimensions, measures) in declarations.items():
        result[name] = grouped(pairs[name], dimensions, measures)
    buckets = defaultdict(list)
    dimensions = ("model", "prefix_tokens", "window", "scenario", "numerator", "denominator")
    for row in pairs["break_even_sensitivity"]:
        buckets[tuple(row[k] for k in dimensions)].append(row)
    result["break_even_sensitivity"] = []
    for key, observations in sorted(buckets.items()):
        item = {"dimensions": dict(zip(dimensions, key)), "matched_affected_windows": len(observations)}
        for mode in ("raw", "injection_subtracted"):
            values = [r[mode] for r in observations]
            valid = [v["p_star"] for v in values if v["p_star"] is not None]
            item[mode] = {"H_ms": distribution(v["H_seconds"] * 1000 for v in values),
                          "R_ms": distribution(v["R_seconds"] * 1000 for v in values),
                          "classifications": dict(Counter(v["classification"] for v in values)),
                          "defined_p_star_count": len(valid), "undefined_p_star_count": len(values) - len(valid),
                          "p_star_defined_subset": distribution(valid)}
        result["break_even_sensitivity"].append(item)
    return result


def resource_groups(cells):
    windows = []
    for cell in cells.values():
        for w in cell["windows"]:
            record = {k: cell[k] for k in ("model", "prefix_tokens", "window", "scenario", "variant")}
            record.update(status=w["status"], fault_affected=w["affected"], replay_layer_steps=w["replay_layer_steps"],
                          skipped_layer_steps=w["skipped_layer_steps"], **w["memory"], **w["availability"], **w["counters"])
            windows.append(record)
    dimensions = ("model", "prefix_tokens", "window", "scenario", "variant", "status", "fault_affected")
    memory = grouped(windows, dimensions, MEMORY)
    available = [w for w in windows if w["status"] == "committed"]
    availability = grouped(available, dimensions, ("first_token_seconds", "last_token_seconds", "mean_token_seconds", "mean_hold_after_step_seconds", "first_step_seconds", "last_step_seconds", "release_seconds"))
    counters = grouped(windows, dimensions, COUNTERS + ("replay_layer_steps", "skipped_layer_steps"))
    return {"memory_bytes": memory, "availability_seconds": availability, "seal_and_layerwork_counters": counters}


def compare_summary(result, directory, summary_name="summary.json"):
    # This is deliberately called only AFTER independent raw calculation.
    require(summary_name in ("summary.json", "summary_corrected.json"), "Unsupported summary selection")
    correction_binding = None
    if summary_name == "summary_corrected.json":
        sidecar_path = directory / "analysis_corrections.json"
        sidecar = read(sidecar_path)
        require(sidecar["status"] == "reporting_only_correction" and sidecar["corrected_summary_sha256"] == sha(directory / summary_name), "Correction/summary binding mismatch")
        allowed = {"summary.json", "metadata.json", "sessions.jsonl", "references.jsonl", "warmups.jsonl", "profiles.jsonl"}
        require(set(sidecar["input_sha256"]) == allowed and all(sha(directory / name) == value for name, value in sidecar["input_sha256"].items()), "Correction/raw source binding mismatch")
        correction_binding = {"sidecar_sha256": sha(sidecar_path), "corrected_summary_sha256": sidecar["corrected_summary_sha256"], "changed_json_leaf_count": sidecar["changed_json_leaf_count"]}
    summary = read(directory / summary_name)
    differences, comparisons, refusal_counter_displays = [], 0, []
    def compare(actual, expected, label):
        nonlocal comparisons
        comparisons += 1
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            ok = isinstance(expected, (int, float)) and close(actual, expected)
        else:
            ok = actual == expected
        if not ok:
            differences.append({"field": label, "independent": actual, "analyzer": expected})
    for field, value in result["counts"].items():
        if field in summary["coverage"]:
            compare(value, summary["coverage"][field], "coverage." + field)
    published_cells = {identity(c): c for c in summary["case_medians"]}
    for cell in result["case_medians"]:
        other = published_cells.get(identity(cell))
        require(other is not None, "Analyzer lacks independently reconstructed case")
        for our_key, their_key in (("session", "session_wall_seconds"), ("session_with_initialization", "session_with_initialization_seconds"),
                                  ("initialization", "session_initialization_seconds"), ("adjusted_session", "session_adjusted_seconds")):
            compare(cell[our_key], other[their_key], str(identity(cell)) + "." + our_key)
        for w, ow in zip(cell["windows"], other["windows"]):
            compare(w["wall"], ow["wall_seconds"], str(identity(cell)) + f".window{w['index']}.wall")
            for key, value in w["memory"].items():
                compare(value, ow["memory"][key], str(identity(cell)) + f".window{w['index']}.{key}")
            for key, value in w["availability"].items():
                compare(value, ow["availability"][key], str(identity(cell)) + f".window{w['index']}.{key}")
            # Analyzer records rejection counters as zero. Controller rejection
            # metadata actually contains work already performed. Compare only
            # successful windows here; retain actual refusal work in this report.
            if w["status"] == "committed":
                for key, value in w["counters"].items():
                    compare(value, ow["counter_medians"][key], str(identity(cell)) + f".window{w['index']}.{key}")
            elif w["counters"] != ow["counter_medians"]:
                refusal_counter_displays.append({"case": list(identity(cell)), "window_index": w["index"],
                    "raw_controller_counters": w["counters"], "analyzer_display": ow["counter_medians"]})
    for name, groups in result["paired_groups"].items():
        published = {json.dumps(g["dimensions"], sort_keys=True): g for g in summary["paired_groups"][name]}
        for group in groups:
            key = json.dumps(group["dimensions"], sort_keys=True)
            require(key in published, "Analyzer missing paired group: " + key)
            other = published[key]
            if name == "break_even_sensitivity":
                compare(group["matched_affected_windows"], other["matched_affected_windows"], name + key + ".matched_affected_windows")
                for mode in ("raw", "injection_subtracted"):
                    for metric, value in group[mode].items():
                        if metric in ("H_ms", "R_ms", "p_star_defined_subset"):
                            for statistic, number in value.items():
                                compare(number, other[mode][metric][statistic], name + key + "." + mode + "." + metric + "." + statistic)
                        else:
                            compare(value, other[mode][metric], name + key + "." + mode + "." + metric)
                continue
            compare(group["case_records"], other["case_records"], name + key + ".case_records")
            for measure, values in group.items():
                if measure in ("dimensions", "case_records"):
                    continue
                for statistic, value in values.items():
                    compare(value, other[measure][statistic], name + key + "." + measure + "." + statistic)
    status = "DISCREPANCY" if differences else "agree_except_refusal_counter_display" if refusal_counter_displays else "agree"
    return {"status": status, "comparisons": comparisons,
            "summary_name": summary_name, "summary_sha256": sha(directory / summary_name), "correction_binding": correction_binding, "differences": differences,
            "known_refusal_counter_display_discrepancies": refusal_counter_displays,
            "refusal_counter_explanation": "Frozen analyzer reads top-level controller_metadata for rejected windows and therefore displays zero seal counters. Raw records retain actual completed guard work under rejection.controller_metadata. This independent report uses the latter; headline timing/correctness quantities are unaffected.",
            "scope": "Raw-derived counts, per-case session/window/memory/availability medians, successful-window counters, grouped paired ratio and break-even distributions; refusal counter display differences are separately enumerated. Not entire analyzer output."}


def calculate(directory, comparison=False, summary_name="summary.json"):
    metadata, data, inputs = load_run(directory)
    cells = reduce_repetitions(data["sessions"])
    pairs = calculate_pairs(cells, metadata["config"])
    result = {"model": metadata["model"], "model_revision": metadata["model_revision"], "input_provenance": inputs,
              "counts": count_run(data), "case_medians": sorted(cells.values(), key=lambda c: str(identity(c))),
              "paired_records": pairs, "paired_groups": aggregate_pair_records(pairs), "resources": resource_groups(cells)}
    if comparison:
        result["analyzer_comparison"] = compare_summary(result, directory, summary_name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-summary", action="store_true")
    parser.add_argument("--summary-name", choices=("summary.json", "summary_corrected.json"), default="summary.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing derived report overwrite")
    runs = [calculate(directory, args.compare_summary, args.summary_name) for directory in args.directories]
    artifact = {"status": "complete_raw_recomputation", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha(__file__), "runs": runs,
                "aggregation": "Median timing within workload/scenario/seed/variant across repetitions first; ratios of matched medians next. Group distributions are across case-policy session cells or explicitly labeled case-window cells, not independent fault samples.",
                "ratio_direction": "Numerator/denominator; >1 favors denominator. Protection-overhead protected/bare-or-native >1 is extra cost.",
                "limitations": ["Record arithmetic audit; no independent model execution or tensor archive inspection.",
                    "Raw fault timing includes artificial injection; adjusted timing is diagnostic only.",
                    "Journal and checkpoint guard challenges excluded from matched timing.",
                    "Native/bare cost baselines do not provide the protection contract.",
                    "Session-inclusive time adds constructor only; common fork, prefill, reference generation and checking remain excluded.",
                    "GPU allocator peaks include live model and reference fixtures; host peak unmeasured.",
                    "Release and buffer-delay metrics are local API measurements, not concurrent/network service latency."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "runs": [{"model": r["model"], "counts": r["counts"],
                     "analyzer_comparison": r.get("analyzer_comparison", {}).get("status", "not_requested")} for r in runs]}, indent=2))
    require(not any(r.get("analyzer_comparison", {}).get("status") == "DISCREPANCY" for r in runs), "Analyzer discrepancy; see retained report")


if __name__ == "__main__":
    main()
