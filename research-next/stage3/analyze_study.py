"""Descriptive stage-3 protection-cost analysis; no field-rate/IID inference.

Raw timings, correctness records, guard refusals, profiling, and tensor-snapshot
coverage remain distinct. The independent audit, not this analyzer, establishes
the extent of independently checked archived tensor evidence.
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return dict(n=0, min=None, q25=None, median=None, q75=None, max=None, mean=None)
    require(all(math.isfinite(value) for value in values), "Nonfinite aggregate input")
    def at(probability):
        index = (len(values) - 1) * probability
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(n=len(values), min=values[0], q25=at(.25), median=at(.5),
                q75=at(.75), max=values[-1], mean=statistics.mean(values))


def break_even(clean_full, clean_sparse, fault_full, fault_sparse):
    """Conditional affected-window model, not a three-window session model."""
    h = clean_sparse - clean_full
    r = fault_full - fault_sparse
    probability = None
    if h > 0 and r > 0:
        classification = "extra_clean_cost_and_positive_fault_saving"
        probability = h / (h + r)
    elif h <= 0 and r > 0:
        classification = "no_extra_clean_cost_with_positive_fault_saving"
    elif h < 0 and r == 0:
        classification = "lower_clean_cost_equal_fault_cost"
    elif h < 0 and r < 0:
        classification = "lower_clean_cost_but_higher_fault_cost"
    elif h == 0 and r == 0:
        classification = "equal_observed_clean_and_fault_cost"
    else:
        classification = "no_positive_fault_saving_with_nonnegative_clean_extra_cost"
    return dict(clean_full_seconds=clean_full, clean_sparse_seconds=clean_sparse,
                affected_full_seconds=fault_full, affected_sparse_seconds=fault_sparse,
                H_seconds=h, R_seconds=r, p_star=probability, classification=classification)


LIMITS = [
    "Descriptive controlled workload/case results, not IID fault samples or estimates of hardware "
    "fault incidence. Timing repetitions are summarized within case first; no confidence intervals "
    "or sampling significance are inferred.",
    "Native and bare clean baselines are unprotected cost baselines, not resilience-equivalent "
    "alternatives to a guarded transactional controller.",
    "Clean, supported persistent-prefix faults, policy-specific journal damage and checkpoint "
    "refusals are separated. Journal damage differs by policy and is excluded from matched fair timing.",
    "Wall latency includes guard, capture and recovery work within the runner boundary. Raw wall "
    "and injection-subtracted sensitivity are both reported; subtraction is not a deployment measurement.",
    "Session initialization is reported separately and in total-with-initialization session comparisons. "
    "Common initial cache fork, prefill and verification remain excluded. Conditional-window break-even "
    "does not amortize the one-time session initialization cost.",
    "Break-even sensitivity compares matched full/sparse CLEAN and AFFECTED WINDOW costs, not "
    "three-window session totals: H=Ts_clean-Tf_clean, R=Tf_fault-Ts_fault, p*=H/(H+R) only if H,R>0. "
    "Other sign combinations are classified explicitly. These are observed noisy costs, not physical fault rates.",
    "Clean medians are shared across fault seeds only for the same model, authored prompt, prefix, "
    "window size and window index. No clean/fault pairing uses pooled unrelated workloads.",
    "Per-step availability and held-window release describe local synchronous API timing, not network "
    "streaming, end-user latency, concurrent service quality or a crash-durable commit protocol.",
    "Logical payload bytes omit protection/metadata/allocator overhead. Allocator peaks depend on "
    "resident fixtures and model state; they are not interchangeable with logical journal payload.",
    "Profiled component times come from separate profile-enabled runs with synchronization. They "
    "are not pooled into primary unprofiled timings or assumed to sum exhaustively to wall latency.",
    "Gold-check flags and snapshot declarations are reported separately. Consult the independent "
    "audit for saved-tensor inspection; this analyzer does not independently reload tensors or rerun inference.",
    "This bounded study does not establish production serving feasibility, broad transient/suffix/"
    "pre-checkpoint fault coverage, semantic task utility, novel prior-art clearance or an optimal journal placement.",
]

VARIANTS = ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched",
            "full_legacy", "sparse_legacy")
BATCHED = VARIANTS[2:5]
MEMORY_FIELDS = ("logical_input_cache_bytes", "logical_checkpoint_bytes", "logical_journal_bytes",
                 "allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes")
COUNTER_FIELDS = ("seal_calls", "seal_logical_bytes", "seal_transfer_calls", "seal_transfer_bytes",
                  "seal_peak_batch_bytes")


def read_json(path):
    def invalid(value):
        raise ValueError("Nonstandard JSON number: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def records(path):
    def invalid(value):
        raise ValueError("Nonstandard JSON number: " + value)
    return [json.loads(line, parse_constant=invalid)
            for line in Path(path).read_text().splitlines() if line.strip()]


def expected_keys(config, workloads):
    result = set()
    for workload in workloads:
        for rep in range(config["clean_repetitions"]):
            for variant in VARIANTS:
                result.add((workload["id"], "clean", config["seeds"][0], rep, variant))
        for scenario in config["fault_cases"]:
            variants = BATCHED + (VARIANTS[5:] if scenario in config["legacy_cases"] else ())
            for seed in config["seeds"]:
                for rep in range(config["repetitions"]):
                    for variant in variants:
                        result.add((workload["id"], scenario, seed, rep, variant))
        for scenario in config["guard_cases"]:
            for variant in BATCHED:
                result.add((workload["id"], scenario, config["seeds"][0], 0, variant))
    return result


def row_key(row):
    return (row["workload_id"], row["scenario"], row["seed"], row["repetition"], row["variant"])


def case_key(row):
    return (row["workload_id"], row["scenario"], row["seed"], row["variant"])


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_rows(rows, metadata, *, profile=False, warmup=False):
    workloads = {w["id"]: w for w in metadata["workloads"]}
    for row in rows:
        require(row["valid"] is True, "Invalid session: " + str(row_key(row)))
        require(row["model"] == metadata["model"] and row["model_revision"] == metadata["model_revision"]
                and row["config_sha256"] == metadata["config_sha256"]
                and row["source_sha256"] == metadata["source_sha256"], "Session provenance mismatch")
        workload = workloads[row["workload_id"]]
        require(all(row[field] == workload[field] for field in ("prompt_id", "prefix_tokens", "window"))
                and row["n_windows"] == metadata["config"]["n_windows"], "Workload shape mismatch")
        require(row["profile"] is profile, "Profile/measurement separation failed")
        require(row["repetition"] == -2 if profile else row["repetition"] == -1 if warmup
                else row["repetition"] >= 0, "Warmup/profile repetition leaked")
        require(row["same_policy_treatment"] is (row["scenario"] != "journal_corruption"),
                "Policy-treatment flag mismatch")
        windows = row["windows"]
        require([w["window_index"] for w in windows] == list(range(len(windows))), "Window order mismatch")
        require(math.isclose(row["total_wall_seconds"], sum(w["wall_seconds"] for w in windows),
                             rel_tol=1e-12, abs_tol=1e-15), "Session wall accounting mismatch")
        require(positive(row["session_initialization_seconds"]) and math.isclose(
            row["total_with_initialization_seconds"], row["total_wall_seconds"] + row["session_initialization_seconds"],
            rel_tol=1e-12, abs_tol=1e-15), "Initialization accounting mismatch")
        if row["scenario"] == "checkpoint_corruption":
            require(row["status"] == "expected_abort" and row["windows_committed"] == 1
                    and row["poisoned"] is True and len(windows) == 2, "Missing expected refusal")
        else:
            require(row["status"] == "complete" and row["windows_committed"] == row["n_windows"]
                    and len(windows) == row["n_windows"] and row["poisoned"] is False, "Unexpected termination")
        for window in windows:
            require(window["valid"] is True and window["no_early_commit"] is True, "Failed window/early commit")
            require(positive(window["wall_seconds"]) and positive(window["wall_minus_injection_seconds"]),
                    "Invalid wall timing")
            require(window["injection_harness_seconds"] >= 0 and math.isclose(
                window["wall_minus_injection_seconds"], window["wall_seconds"] - window["injection_harness_seconds"],
                rel_tol=1e-12, abs_tol=1e-15), "Injection subtraction mismatch")
            if window["status"] == "committed":
                require(window["checks"]["all_equal"] is True
                        and all(value["equal"] is True if isinstance(value, dict) else value is True
                                for value in window["checks"].values()), "Failed gold check")
                require(window["committed_after"] - window["committed_before"] == row["window"]
                        and len(window["tokens"]) == row["window"], "Partial commit")
                require(len(window["token_availability_seconds"]) == row["window"], "Availability length")
                expected_available = (window["step_completion_seconds"] if row["variant"] in ("native", "bare")
                                      else [window["wall_seconds"]] * row["window"])
                require(window["token_availability_seconds"] == expected_available
                        and window["release_seconds"] == window["wall_seconds"], "Availability scope mismatch")
            else:
                require(window["status"] == "rejected" and all(v is True for v in window["checks"].values())
                        and window["committed_before"] == window["committed_after"], "Failed refusal guard")
                require(window["release_seconds"] is None and window["token_availability_seconds"] == [],
                        "Rejected window released output")
            require(len(window["step_completion_seconds"]) == row["window"]
                    and all(positive(t) for t in window["step_completion_seconds"]), "Step timestamp coverage")
            require(window["step_completion_seconds"] == sorted(window["step_completion_seconds"])
                    and window["step_completion_seconds"][-1] <= window["wall_seconds"], "Nonmonotone step timing")
            require(all(window[field] >= 0 for field in MEMORY_FIELDS), "Negative payload/allocator count")


def load_complete(directory):
    metadata = read_json(directory / "metadata.json")
    require(metadata["status"] == "complete", "Refusing incomplete study")
    inputs = {"metadata": sha(directory / "metadata.json")}
    datasets = {}
    for name in ("sessions", "references", "warmups", "profiles"):
        inputs[name] = sha(directory / (name + ".jsonl"))
        require(inputs[name] == metadata[name + "_sha256"], "Raw hash mismatch: " + name)
        datasets[name] = records(directory / (name + ".jsonl"))
    rows, config = datasets["sessions"], metadata["config"]
    actual = [row_key(row) for row in rows]
    expected = expected_keys(config, metadata["workloads"])
    require(len(actual) == len(set(actual)) == len(expected) == metadata["sessions_completed"]
            == metadata["expected_sessions"] and set(actual) == expected, "Incomplete/duplicate frozen matrix")
    workload_ids = {w["id"] for w in metadata["workloads"]}
    require(len(datasets["references"]) == len(workload_ids)
            and {r["workload_id"] for r in datasets["references"]} == workload_ids, "Reference coverage mismatch")
    for reference in datasets["references"]:
        workload = next(w for w in metadata["workloads"] if w["id"] == reference["workload_id"])
        require([p["position"] for p in reference["parity"]] == list(range(
            workload["prefix_tokens"], workload["prefix_tokens"] + config["n_windows"] * workload["window"] + 2)),
            "Post-prefill parity position coverage mismatch")
        require(all(p["cache"]["equal"] is True and p["logits"]["equal"] is True and p["token_equal"] is True
                    for p in reference["parity"]), "Native/custom parity failure")
    require(len(datasets["warmups"]) == metadata["warmups_completed"]
            and len(datasets["profiles"]) == metadata["profiles_completed"], "Diagnostic coverage mismatch")
    warm_keys = {(wid, scenario, config["seeds"][0], -1, variant) for wid in workload_ids
                 for scenario, variants in (("clean", VARIANTS), ("value_late", VARIANTS[2:]))
                 for variant in variants} if config.get("warmup", True) else set()
    profile_keys = {(wid, scenario, config["seeds"][0], -2, variant) for wid in workload_ids
                    for scenario in ("clean", "value_late") for variant in ("full_batched", "sparse_batched")
                    } if config.get("profile", False) else set()
    for name, expected_diagnostics in (("warmups", warm_keys), ("profiles", profile_keys)):
        keys = [row_key(row) for row in datasets[name]]
        require(len(keys) == len(set(keys)) == len(expected_diagnostics) and set(keys) == expected_diagnostics,
                "Incomplete diagnostic matrix: " + name)
    validate_rows(rows, metadata)
    validate_rows(datasets["warmups"], metadata, warmup=True)
    validate_rows(datasets["profiles"], metadata, profile=True)
    expected_snapshots, observed_snapshots = set(), set()
    for row in rows:
        for window in row["windows"]:
            key = row_key(row) + (window["window_index"],)
            if "snapshot" in window:
                observed_snapshots.add(key)
            if (row["prompt_id"] == config.get("snapshot_prompt") and row["seed"] == config["seeds"][0]
                and row["repetition"] == 0 and window["window_index"] == 1
                and ((row["prefix_tokens"] == config.get("snapshot_short_prefix")
                      and row["scenario"] in config.get("snapshot_cases", []) and row["variant"] in BATCHED)
                     or (row["prefix_tokens"] == max(s["prefix_tokens"] for s in config["shapes"])
                         and row["scenario"] == config.get("snapshot_long_case") and row["variant"] == "sparse_batched"))):
                expected_snapshots.add(key)
    require(observed_snapshots == expected_snapshots, "Prespecified snapshot declaration coverage mismatch")
    return metadata, datasets, inputs


def availability(window):
    if window["status"] != "committed":
        return {}
    times, steps = window["token_availability_seconds"], window["step_completion_seconds"]
    return dict(first_token_seconds=times[0], last_token_seconds=times[-1],
                mean_token_seconds=statistics.mean(times),
                mean_hold_after_step_seconds=statistics.mean(a - b for a, b in zip(times, steps)),
                first_step_seconds=steps[0], last_step_seconds=steps[-1],
                release_seconds=window["release_seconds"])


def case_medians(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        row = group[0]
        require(len({r["repetition"] for r in group}) == len(group), "Duplicate case repetition")
        require(all([w["faults"] for w in r["windows"]] == [w["faults"] for w in row["windows"]]
                    for r in group), "Deterministic treatment changed across repetitions")
        result = {name: row[name] for name in ("workload_id", "model", "prompt_id", "prefix_tokens", "window",
                  "n_windows", "num_layers", "scenario", "seed", "variant", "policy", "seal_mode", "status")}
        result.update(timing_repetitions=len(group), repetitions=sorted(r["repetition"] for r in group),
                      session_wall_seconds=statistics.median(r["total_wall_seconds"] for r in group),
                      session_initialization_seconds=statistics.median(r["session_initialization_seconds"] for r in group),
                      session_with_initialization_seconds=statistics.median(r["total_with_initialization_seconds"] for r in group),
                      session_adjusted_seconds=statistics.median(r["total_wall_seconds"] -
                                                                r["total_injection_harness_seconds"] for r in group),
                      session_adjusted_with_initialization_seconds=statistics.median(r["total_with_initialization_seconds"] -
                                                                                    r["total_injection_harness_seconds"] for r in group),
                      windows=[])
        for wi, original in enumerate(row["windows"]):
            ws = [r["windows"][wi] for r in group]
            require(all(w["status"] == original["status"] for w in ws), "Status differs across repetitions")
            available = [availability(w) for w in ws]
            result["windows"].append(dict(
                window_index=wi, status=original["status"], faults=original["faults"],
                fault_affected=any(f["target"] == "working_prefix" for f in original["faults"]),
                wall_seconds=statistics.median(w["wall_seconds"] for w in ws),
                adjusted_seconds=statistics.median(w["wall_minus_injection_seconds"] for w in ws),
                wall_repetition_range=distribution(w["wall_seconds"] for w in ws),
                adjusted_repetition_range=distribution(w["wall_minus_injection_seconds"] for w in ws),
                memory={field: statistics.median(w[field] for w in ws) for field in MEMORY_FIELDS},
                availability={field: statistics.median(v[field] for v in available)
                              for field in available[0]},
                counter_medians={field: statistics.median(w.get("controller_metadata", {}).get(field, 0)
                                                         for w in ws) for field in COUNTER_FIELDS},
                selected_cuts=sorted({w.get("selected_cut", 0) for w in ws}),
                divergence_positions=sorted({w["first_divergence"] for w in ws
                                             if w.get("first_divergence") is not None}),
                recovered=any(w.get("recovered", False) for w in ws),
                fallback_reasons=sorted({w["fallback_reason"] for w in ws if w.get("fallback_reason")}),
            ))
        output.append(result)
    return output


def aggregate_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["model"], case["prefix_tokens"], case["window"], case["scenario"], case["variant"])].append(case)
    result = []
    for key, group in sorted(groups.items()):
        windows = [w for case in group for w in case["windows"]]
        item = dict(zip(("model", "prefix_tokens", "window", "scenario", "variant"), key))
        faults = [fault for window in windows for fault in window["faults"]]
        item.update(case_cells=len(group), raw_timing_repetitions=sum(c["timing_repetitions"] for c in group),
                    actual_working_prefix_layers=sorted({f["layer"] for f in faults if f["target"] == "working_prefix"}),
                    actual_fault_kinds=sorted({f.get("kind", "journal_hidden") for f in faults}),
                    actual_fault_types=sorted({f["fault_type"] for f in faults}),
                    session_wall_ms=distribution(c["session_wall_seconds"] * 1000 for c in group),
                    session_initialization_ms=distribution(c["session_initialization_seconds"] * 1000 for c in group),
                    session_with_initialization_ms=distribution(c["session_with_initialization_seconds"] * 1000 for c in group),
                    session_adjusted_ms=distribution(c["session_adjusted_seconds"] * 1000 for c in group),
                    window_strata=[])
        for status, affected in sorted({(w["status"], w["fault_affected"]) for w in windows}):
            subset = [w for w in windows if (w["status"], w["fault_affected"]) == (status, affected)]
            item["window_strata"].append(dict(
                status=status, fault_affected=affected, case_windows=len(subset),
                wall_ms=distribution(w["wall_seconds"] * 1000 for w in subset),
                adjusted_ms=distribution(w["adjusted_seconds"] * 1000 for w in subset),
                memory={field: distribution(w["memory"][field] for w in subset) for field in MEMORY_FIELDS},
                availability_ms={field: distribution(w["availability"][field] * 1000 for w in subset)
                                 for field in subset[0]["availability"]},
                counters={field: distribution(w["counter_medians"][field] for w in subset) for field in COUNTER_FIELDS},
            ))
        result.append(item)
    return result


def paired_records(cases, config):
    by_case = {case_key(c): c for c in cases}
    policy_pairs, window_pairs, ablations, overheads, sensitivities = [], [], [], [], []
    comparison_variants = (("full_batched", "sparse_batched"), ("full_batched", "allcuts_batched"),
                           ("sparse_batched", "allcuts_batched"), ("full_legacy", "sparse_legacy"))
    for left in cases:
        common = {k: left[k] for k in ("model", "workload_id", "prompt_id", "prefix_tokens", "window", "scenario", "seed")}
        if left["scenario"] in ("clean", *config["fault_cases"]):
            for numerator, denominator in comparison_variants:
                if left["variant"] != numerator:
                    continue
                right = by_case.get((left["workload_id"], left["scenario"], left["seed"], denominator))
                if right is None:
                    continue
                require([w["faults"] for w in left["windows"]] == [w["faults"] for w in right["windows"]],
                        "Policies received different actual treatment")
                pair = dict(common, numerator=numerator, denominator=denominator,
                            wall_ratio=left["session_wall_seconds"] / right["session_wall_seconds"],
                            adjusted_ratio=left["session_adjusted_seconds"] / right["session_adjusted_seconds"],
                            with_initialization_ratio=left["session_with_initialization_seconds"] / right["session_with_initialization_seconds"],
                            adjusted_with_initialization_ratio=left["session_adjusted_with_initialization_seconds"] / right["session_adjusted_with_initialization_seconds"])
                policy_pairs.append(pair)
                for lw, rw in zip(left["windows"], right["windows"]):
                    window_pairs.append(dict(common, numerator=numerator, denominator=denominator,
                        window_index=lw["window_index"], fault_affected=lw["fault_affected"], status=lw["status"],
                        wall_ratio=lw["wall_seconds"] / rw["wall_seconds"],
                        adjusted_ratio=lw["adjusted_seconds"] / rw["adjusted_seconds"]))
                    if numerator.startswith("full_") and denominator.startswith("sparse_") and lw["fault_affected"]:
                        clean_l = by_case[(left["workload_id"], "clean", config["seeds"][0], numerator)]["windows"][lw["window_index"]]
                        clean_r = by_case[(left["workload_id"], "clean", config["seeds"][0], denominator)]["windows"][lw["window_index"]]
                        sensitivities.append(dict(common, numerator=numerator, denominator=denominator,
                            window_index=lw["window_index"],
                            raw=break_even(clean_l["wall_seconds"], clean_r["wall_seconds"],
                                           lw["wall_seconds"], rw["wall_seconds"]),
                            injection_subtracted=break_even(clean_l["adjusted_seconds"], clean_r["adjusted_seconds"],
                                                            lw["adjusted_seconds"], rw["adjusted_seconds"])))
        if left["variant"].endswith("_legacy"):
            batched = left["variant"].replace("_legacy", "_batched")
            right = by_case[(left["workload_id"], left["scenario"], left["seed"], batched)]
            require([w["faults"] for w in left["windows"]] == [w["faults"] for w in right["windows"]],
                    "Sealing ablation treatment mismatch")
            ablations.append(dict(common, legacy=left["variant"], batched=batched,
                session_legacy_over_batched_wall_ratio=left["session_wall_seconds"] / right["session_wall_seconds"],
                session_legacy_over_batched_adjusted_ratio=left["session_adjusted_seconds"] / right["session_adjusted_seconds"],
                session_legacy_over_batched_with_initialization_ratio=left["session_with_initialization_seconds"] / right["session_with_initialization_seconds"],
                window_pairs=[dict(window_index=lw["window_index"], fault_affected=lw["fault_affected"],
                    legacy_over_batched_wall_ratio=lw["wall_seconds"] / rw["wall_seconds"],
                    legacy_over_batched_adjusted_ratio=lw["adjusted_seconds"] / rw["adjusted_seconds"])
                    for lw, rw in zip(left["windows"], right["windows"])]))
        if left["scenario"] == "clean" and left["variant"] not in ("native", "bare"):
            for baseline in ("bare", "native"):
                right = by_case[(left["workload_id"], "clean", config["seeds"][0], baseline)]
                ratio = left["session_wall_seconds"] / right["session_wall_seconds"]
                initialized = left["session_with_initialization_seconds"] / right["session_with_initialization_seconds"]
                overheads.append(dict(common, protected=left["variant"], unprotected_baseline=baseline,
                    session_protected_over_baseline=ratio, session_overhead_fraction=ratio - 1,
                    session_extra_ms=(left["session_wall_seconds"] - right["session_wall_seconds"]) * 1000,
                    session_with_initialization_protected_over_baseline=initialized,
                    session_with_initialization_overhead_fraction=initialized - 1,
                    session_with_initialization_extra_ms=(left["session_with_initialization_seconds"] -
                                                          right["session_with_initialization_seconds"]) * 1000,
                    window_pairs=[dict(window_index=lw["window_index"],
                        protected_over_baseline=lw["wall_seconds"] / rw["wall_seconds"],
                        overhead_fraction=lw["wall_seconds"] / rw["wall_seconds"] - 1,
                        extra_ms=(lw["wall_seconds"] - rw["wall_seconds"]) * 1000)
                        for lw, rw in zip(left["windows"], right["windows"])]))
    return dict(matched_policy_sessions=policy_pairs, matched_policy_windows=window_pairs,
                sealing_ablation=ablations, protection_overhead=overheads, break_even_sensitivity=sensitivities)


def group_distributions(rows, dimensions, measures):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in dimensions)].append(row)
    return [dict(dimensions=dict(zip(dimensions, key)), case_records=len(group),
                 **{measure: distribution(row[measure] for row in group) for measure in measures})
            for key, group in sorted(grouped.items(), key=lambda item: str(item[0]))]


def aggregate_pairs(pairs):
    groups = dict(
        matched_policy_sessions=group_distributions(pairs["matched_policy_sessions"],
            ("model", "prefix_tokens", "window", "scenario", "numerator", "denominator"),
            ("wall_ratio", "adjusted_ratio", "with_initialization_ratio", "adjusted_with_initialization_ratio")),
        matched_policy_windows=group_distributions(pairs["matched_policy_windows"],
            ("model", "prefix_tokens", "window", "scenario", "numerator", "denominator", "fault_affected", "status"),
            ("wall_ratio", "adjusted_ratio")),
        sealing_ablation=group_distributions(pairs["sealing_ablation"],
            ("model", "prefix_tokens", "window", "scenario", "legacy", "batched"),
            ("session_legacy_over_batched_wall_ratio", "session_legacy_over_batched_adjusted_ratio",
             "session_legacy_over_batched_with_initialization_ratio")),
        protection_overhead=group_distributions(pairs["protection_overhead"],
            ("model", "prefix_tokens", "window", "protected", "unprotected_baseline"),
            ("session_protected_over_baseline", "session_overhead_fraction", "session_extra_ms",
             "session_with_initialization_protected_over_baseline", "session_with_initialization_overhead_fraction",
             "session_with_initialization_extra_ms")),
    )
    sensitivity_groups = defaultdict(list)
    for row in pairs["break_even_sensitivity"]:
        sensitivity_groups[(row["model"], row["prefix_tokens"], row["window"], row["scenario"],
                            row["numerator"], row["denominator"])].append(row)
    groups["break_even_sensitivity"] = []
    for key, rows in sorted(sensitivity_groups.items()):
        cell = dict(dimensions=dict(zip(("model", "prefix_tokens", "window", "scenario", "numerator", "denominator"), key)),
                    matched_affected_windows=len(rows))
        for timing in ("raw", "injection_subtracted"):
            values = [row[timing] for row in rows]
            cell[timing] = dict(H_ms=distribution(v["H_seconds"] * 1000 for v in values),
                                R_ms=distribution(v["R_seconds"] * 1000 for v in values),
                                classifications=dict(sorted(Counter(v["classification"] for v in values).items())),
                                defined_p_star_count=sum(v["p_star"] is not None for v in values),
                                undefined_p_star_count=sum(v["p_star"] is None for v in values),
                                p_star_defined_subset=distribution(v["p_star"] for v in values if v["p_star"] is not None))
        groups["break_even_sensitivity"].append(cell)
    return groups


def profile_summary(rows):
    cells = defaultdict(list)
    for row in rows:
        for window in row["windows"]:
            cells[(row["model"], row["prefix_tokens"], row["window"], row["scenario"], row["variant"],
                   bool(window["faults"]))].append(window)
    output = []
    for key, windows in sorted(cells.items()):
        components = sorted({name for w in windows for name in w["controller_metadata"].get("profile_seconds", {})})
        output.append(dict(dimensions=dict(zip(("model", "prefix_tokens", "window", "scenario", "variant", "fault_affected"), key)),
            separately_profiled_windows=len(windows),
            profiled_wall_ms=distribution(w["wall_seconds"] * 1000 for w in windows),
            components_ms={name: distribution(w["controller_metadata"]["profile_seconds"].get(name, 0) * 1000
                                              for w in windows) for name in components},
            seal_counters={name: distribution(w["controller_metadata"].get(name, 0) for w in windows)
                           for name in COUNTER_FIELDS}))
    return output


def coverage(rows, datasets, metadata):
    windows = [w for r in rows for w in r["windows"]]
    snapshots = [dict(model=r["model"], workload_id=r["workload_id"], scenario=r["scenario"],
                      seed=r["seed"], repetition=r["repetition"], variant=r["variant"],
                      window_index=w["window_index"], **w["snapshot"])
                 for r in rows for w in r["windows"] if "snapshot" in w]
    faults = [f for w in windows for f in w["faults"]]
    strata = defaultdict(list)
    for row in rows:
        strata[(row["scenario"], row["variant"])].append(row)
    return dict(
        measured_sessions=len(rows), expected_sessions=metadata["expected_sessions"],
        workloads=len(datasets["references"]), warmup_sessions_excluded=len(datasets["warmups"]),
        profiled_sessions_excluded=len(datasets["profiles"]), windows_attempted=len(windows),
        committed_windows=sum(w["status"] == "committed" for w in windows),
        rejected_windows=sum(w["status"] == "rejected" for w in windows),
        gold_checked_commits=sum(w["status"] == "committed" and w["checks"]["all_equal"] for w in windows),
        valid_refusal_guards=sum(w["status"] == "rejected" and all(w["checks"].values()) for w in windows),
        native_custom_parity_positions=sum(len(r["parity"]) for r in datasets["references"]),
        recovered_windows=sum(w.get("recovered", False) for w in windows),
        replay_windows_with_token_divergence=sum(w.get("first_divergence") is not None for w in windows),
        partial_to_full_replay_windows=sum(any(c > 0 for c in w.get("replay_starts", []))
                                          and 0 in w.get("replay_starts", []) for w in windows),
        fault_event_targets=dict(sorted(Counter(f["target"] for f in faults).items())),
        fault_event_kinds=dict(sorted(Counter(f.get("kind", "journal_hidden") for f in faults).items())),
        fault_event_types=dict(sorted(Counter(f["fault_type"] for f in faults).items())),
        working_prefix_layer_event_counts=dict(sorted(Counter(str(f["layer"]) for f in faults
                                                              if f["target"] == "working_prefix").items())),
        # Counts include repeated timing executions and policy variants by design.
        strata=[dict(scenario=scenario, variant=variant, sessions=len(group),
                     controlled_case_cells=len({case_key(row) for row in group}),
                     attempted_windows=sum(len(row["windows"]) for row in group))
                for (scenario, variant), group in sorted(strata.items())],
        snapshot_declarations=len(snapshots), snapshot_declared_bytes=sum(s["bytes"] for s in snapshots),
        snapshots=snapshots, independently_loaded_snapshots_by_this_analyzer=0,
        all_recorded_checks_pass=True)


def analyze(directory):
    metadata, datasets, inputs = load_complete(directory)
    rows, config = datasets["sessions"], metadata["config"]
    cases = case_medians(rows)
    pairs = paired_records(cases, config)
    guard_rows = [r for r in rows if r["scenario"] in config["guard_cases"]]
    return dict(
        status="complete_descriptive_stage3", schema_version=1, model=metadata["model"],
        model_revision=metadata["model_revision"], input_sha256=inputs,
        config_sha256=metadata["config_sha256"], manifest_sha256=metadata["manifest_sha256"],
        source_sha256=metadata["source_sha256"], analyzer_sha256=sha(__file__),
        aggregation="Within each workload/scenario/seed/variant, median across timing repetitions first. "
                    "Ratios are ratios of matched case medians, not medians of repetition ratios. "
                    "Group distributions are descriptive across the explicitly identified case cells.",
        ratio_direction="Numerator time / denominator time: above one means denominator was faster. "
                        "Protection-overhead ratios invert that interpretation: protected / unprotected above one is added cost.",
        coverage=coverage(rows, datasets, metadata),
        case_medians=cases, groups=aggregate_cases(cases), paired_records=pairs, paired_groups=aggregate_pairs(pairs),
        guard_diagnostics=dict(
            journal_cases_are_policy_specific=True,
            checkpoint_refusal_reasons=dict(sorted(Counter(w["rejection"]["reason"] for r in guard_rows
                                                         for w in r["windows"] if w["status"] == "rejected").items())),
            journal_fallback_reasons=dict(sorted(Counter(w["fallback_reason"] for r in guard_rows
                                                       for w in r["windows"] if w.get("fallback_reason")).items())),
            excluded_from_matched_timing_scenarios=list(config["guard_cases"])),
        profiling=dict(separate_from_measured_rows=True, exhaustive_wall_decomposition=False,
                       component_times_include_trailing_profile_sync=True,
                       preceding_profile_sync_outside_component_interval=True,
                       groups=profile_summary(datasets["profiles"])),
        interpretation_limits=LIMITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    counts = result["coverage"]
    print(json.dumps(dict(status=result["status"], model=result["model"],
                          **{name: counts[name] for name in ("measured_sessions", "workloads", "committed_windows",
                              "rejected_windows", "gold_checked_commits", "snapshot_declarations")}), indent=2))


if __name__ == "__main__":
    main()
