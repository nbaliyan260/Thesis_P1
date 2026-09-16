"""Post-hoc, descriptive RECUT trade-off analysis of one complete frozen run.

Independent of the runner, engine, and primary analyzer. This does not establish
a Pareto frontier, statistical independence, a field fault rate, or novelty.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


VERSION = "RECUT-posthoc-frontier-v1"
REPAIR = ("full", "sparse", "all")
NORMAL = ("none", "sparse", "all")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    """Exact descriptive counts and interpolated quantiles; no uncertainty CI."""
    values = sorted(values)
    if not values:
        return {"n": 0}

    def at(p):
        index = (len(values) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return dict(n=len(values), min=values[0], q25=at(.25), median=at(.5),
                q75=at(.75), max=values[-1], mean=statistics.mean(values))


def finite_seconds(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def check_trials(row, key, label, methods, repetitions):
    trials = row[key]
    expected = {(method, rep) for method in methods for rep in range(repetitions)}
    actual = [(t[label], t["repetition"]) for t in trials]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Missing/duplicate timing repetitions: " + row["case_id"] + "/" + key)
    require(all(finite_seconds(t["seconds"]) for t in trials), "Invalid timing")
    for trial in trials:
        order = list(methods)
        offset = trial["repetition"] % len(methods)
        require(trial["order"] == order[offset:] + order[:offset], "Order mismatch")
        verified = (trial["checks"]["all_equal"] if key == "timings"
                    else trial["verified_equal"])
        require(verified is True, "Unverified trial: " + row["case_id"])


def load_complete(run):
    meta_raw = (run / "metadata.json").read_bytes()
    metadata = json.loads(meta_raw)
    require(metadata["status"] == "complete", "Refusing incomplete run")
    raw = (run / "episodes.jsonl").read_bytes()
    require(sha(raw) == metadata["episodes_sha256"], "Episode SHA mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    config = metadata["config"]
    cases = {c["id"]: c for c in config["cases"]}
    require(len(cases) == len(config["cases"]), "Duplicate configured case")
    models = {m["model"]: m for m in metadata["models"]}
    require(len(models) == len(metadata["models"]), "Duplicate configured model")
    expected = {(model, case) for model in models for case in cases}
    actual = [(r["model"], r["case_id"]) for r in rows]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Incomplete/duplicate model-by-case coverage")
    require(set(metadata["models_completed"]) == set(models), "Model incomplete")
    for row in rows:
        require(row["case"] == cases[row["case_id"]], "Per-row config mismatch")
        require(row["config_sha256"] == metadata["config_sha256"], "Config SHA mismatch")
        require(row["revision"] == models[row["model"]]["revision"], "Revision mismatch")
        require(row["valid_repairs_equal"] is True, "Run contains failed repair")
        require(all(row["methods"][m]["all_equal"] is True for m in REPAIR),
                "Run contains failed repair flag")
        check_trials(row, "timings", "method", REPAIR, config["repetitions"])
        check_trials(row, "overhead_trials", "mode", NORMAL,
                     config["overhead_repetitions"])
        require(row["sparse_journal_payload_bytes"] <= row["all_journal_payload_bytes"],
                "Unexpected payload ordering")
    return metadata, rows, dict(metadata=sha(meta_raw), episodes=sha(raw))


def repetition_comparison(left, right):
    """Same repetition is a descriptive pairing, not an independent sample."""
    differences = [left[rep] - right[rep] for rep in sorted(left)]
    return dict(
        n_pairs=len(differences),
        left_lower=sum(d < 0 for d in differences),
        exactly_equal=sum(d == 0 for d in differences),
        left_higher=sum(d > 0 for d in differences),
        left_minus_right_ms=distribution([d * 1000 for d in differences]),
        observed_ranges_overlap=max(min(left.values()), min(right.values()))
                                <= min(max(left.values()), max(right.values())),
        all_pairs_same_strict_sign=(all(d < 0 for d in differences)
                                    or all(d > 0 for d in differences)),
    )


def case_metrics(row):
    recovery_samples = {m: {t["repetition"]: t["seconds"] for t in row["timings"]
                            if t["method"] == m} for m in REPAIR}
    normal_samples = {m: {t["repetition"]: t["seconds"] for t in row["overhead_trials"]
                          if t["mode"] == m} for m in NORMAL}
    recovery = {m: statistics.median(v.values()) for m, v in recovery_samples.items()}
    normal = {m: statistics.median(v.values()) for m, v in normal_samples.items()}
    divergence = row["first_divergence"]
    divergence_class = ("none" if divergence is None else
                        "immediate" if divergence == 1 else "delayed")
    work = {m: row["methods"][m]["layer_steps"] for m in REPAIR}
    denominator = row["num_layers"] * row["case"]["window"]
    methods = {}
    for method in ("sparse", "all"):
        saving = recovery["full"] - recovery[method]
        cost = normal[method] - normal["none"]
        positive_cost = max(cost, 0.0)
        ratio = positive_cost / saving if saving > 0 else None
        methods[method] = dict(
            full_over_method_recovery_ratio=recovery["full"] / recovery[method],
            recovery_saving_ms=1000 * saving,
            signed_normal_cost_ms=1000 * cost,
            nonnegative_normal_cost_ms=1000 * positive_cost,
            normal_overhead_fraction=normal[method] / normal["none"] - 1,
            layer_steps=work[method], layer_step_fraction=work[method] / denominator,
            cut=row["methods"][method]["cut"],
            journal_payload_bytes=row[method + "_journal_payload_bytes"],
            journal_over_checkpoint_fraction=(row[method + "_journal_payload_bytes"]
                                              / row["checkpoint_payload_bytes"]),
            nonnegative_cost_sensitivity_ratio=ratio,
            break_even_status=("no_positive_recovery_saving" if ratio is None else
                               "above_one" if ratio > 1 else "within_zero_one"),
        )
    return dict(
        model=row["model"], case_id=row["case_id"], case=row["case"],
        num_layers=row["num_layers"], layer=row["layer"],
        population="noop_control" if row["case"]["fault"] == "noop" else "injected_fault",
        first_divergence=divergence, divergence_class=divergence_class,
        changed_elements=row["fault"]["changed_elements"],
        checkpoint_payload_bytes=row["checkpoint_payload_bytes"],
        recovery_seconds=recovery, normal_seconds=normal,
        recovery_samples_seconds=recovery_samples, normal_samples_seconds=normal_samples,
        recovery_sample_spread_ms={m: distribution([1000 * x for x in v.values()])
                                  for m, v in recovery_samples.items()},
        normal_sample_spread_ms={m: distribution([1000 * x for x in v.values()])
                                for m, v in normal_samples.items()},
        sparse_vs_all=dict(
            sparse_over_all_recovery_ratio=recovery["sparse"] / recovery["all"],
            sparse_minus_all_recovery_ms=1000 * (recovery["sparse"] - recovery["all"]),
            sparse_minus_all_normal_ms=1000 * (normal["sparse"] - normal["all"]),
            sparse_over_all_journal_payload_ratio=(row["sparse_journal_payload_bytes"]
                                                  / row["all_journal_payload_bytes"]),
            saved_journal_payload_bytes=(row["all_journal_payload_bytes"]
                                         - row["sparse_journal_payload_bytes"]),
            extra_layer_steps=work["sparse"] - work["all"],
            repair_repetitions=repetition_comparison(recovery_samples["sparse"],
                                                     recovery_samples["all"]),
            normal_repetitions=repetition_comparison(normal_samples["sparse"],
                                                     normal_samples["all"]),
        ),
        correctness={m: {key: row["methods"][m][key] for key in
                          ("all_equal", "tokens_equal", "continuation_tokens_equal")}
                     | {"cache_equal": row["methods"][m]["cache"]["equal"],
                        "logits_equal": row["methods"][m]["logits"]["equal"]}
                     for m in ("none", "slotonly") + REPAIR},
        methods=methods,
    )


def summarize(cases):
    """All case rows in a specified cell get equal descriptive weight."""
    result = dict(
        n=len(cases), case_ids=[c["case_id"] for c in cases],
        distinct_prompts=len({c["case"]["prompt"] for c in cases}),
        no_actual_change_cases=sum(c["changed_elements"] == 0 for c in cases),
        divergence_counts=dict(sorted(Counter(c["divergence_class"] for c in cases).items())),
        first_divergence_counts=dict(sorted(Counter(
            "none" if c["first_divergence"] is None else str(c["first_divergence"])
            for c in cases).items())),
        correctness={m: {flag: sum(c["correctness"][m][flag] for c in cases)
                          for flag in cases[0]["correctness"][m]}
                     for m in ("none", "slotonly") + REPAIR},
        methods={},
    )
    for method in ("sparse", "all"):
        values = [c["methods"][method] for c in cases]
        saving = statistics.mean(v["recovery_saving_ms"] for v in values)
        signed_cost = statistics.mean(v["signed_normal_cost_ms"] for v in values)
        positive_cost = statistics.mean(v["nonnegative_normal_cost_ms"] for v in values)
        ratio = positive_cost / saving if saving > 0 else None
        result["methods"][method] = dict(
            **{key: distribution(v[key] for v in values) for key in (
                "full_over_method_recovery_ratio", "recovery_saving_ms",
                "signed_normal_cost_ms", "normal_overhead_fraction", "layer_step_fraction",
                "journal_payload_bytes", "journal_over_checkpoint_fraction")},
            recovery_ms=distribution(c["recovery_seconds"][method] * 1000 for c in cases),
            normal_ms=distribution(c["normal_seconds"][method] * 1000 for c in cases),
            cases_with_negative_observed_normal_cost=sum(v["signed_normal_cost_ms"] < 0
                                                        for v in values),
            cases_with_nonpositive_recovery_saving=sum(v["recovery_saving_ms"] <= 0
                                                       for v in values),
            nonnegative_cost_sensitivity=dict(
                numerator_mean_nonnegative_normal_cost_ms=positive_cost,
                denominator_mean_signed_recovery_saving_ms=saving,
                mean_signed_normal_cost_ms=signed_cost,
                ratio=ratio, denominator_case_count=len(cases),
                status=("no_positive_mean_recovery_saving" if ratio is None else
                        "above_one" if ratio > 1 else "within_zero_one"),
                signed_mean_ratio=signed_cost / saving if saving > 0 else None,
                original_primary_ratio=(signed_cost / saving
                                        if saving > 0 and signed_cost >= 0 else None),
            ),
        )
    comparisons = [c["sparse_vs_all"] for c in cases]
    difference_key = "sparse_minus_all_recovery_ms"
    result["sparse_vs_all"] = dict(
        **{key: distribution(c[key] for c in comparisons) for key in (
            "sparse_over_all_recovery_ratio", difference_key,
            "sparse_minus_all_normal_ms", "sparse_over_all_journal_payload_ratio",
            "saved_journal_payload_bytes", "extra_layer_steps")},
        cases_sparse_repair_median_lower=sum(c[difference_key] < 0 for c in comparisons),
        cases_equal_repair_median=sum(c[difference_key] == 0 for c in comparisons),
        cases_sparse_repair_median_higher=sum(c[difference_key] > 0 for c in comparisons),
        cases_repair_sample_ranges_overlap=sum(c["repair_repetitions"]["observed_ranges_overlap"]
                                               for c in comparisons),
        cases_repair_pair_signs_mixed_or_equal=sum(
            not c["repair_repetitions"]["all_pairs_same_strict_sign"] for c in comparisons),
        cases_sparse_payload_strictly_lower=sum(c["saved_journal_payload_bytes"] > 0
                                                for c in comparisons),
    )
    result["full_recovery_ms"] = distribution(c["recovery_seconds"]["full"] * 1000 for c in cases)
    result["normal_nojournal_ms"] = distribution(c["normal_seconds"]["none"] * 1000 for c in cases)
    return result


def grouped(cases, fields):
    cells = defaultdict(list)
    for case in cases:
        values = tuple(case[field] if field in case else case["case"][field] for field in fields)
        cells[values].append(case)
    return [dict(dimensions=dict(zip(fields, key)), **summarize(value))
            for key, value in sorted(cells.items(), key=lambda item: str(item[0]))]


def render_markdown(report):
    lines = ["# RECUT post-hoc exploratory trade-offs", "",
             "This is descriptive analysis of the complete frozen run, not a preregistered "
             "test or evidence of a timing Pareto frontier. All cases remain in the JSON outputs; "
             "noops are summarized separately. Quantiles describe case medians, not independent trials.",
             "", "## Injected-fault cases, by model", "",
             "Recovery ratios below are full replay / journal recovery (larger is faster). "
             "Nonnegative-cost sensitivity is a hypothetical ratio, not a measured fault probability.",
             "", "| Model | Cases | Policy | Median recovery ratio | Median journal KiB | "
             "Mean nonnegative normal cost (ms) | Mean signed recovery saving (ms) | "
             "Clipped sensitivity | Original primary ratio |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for cell in report["groups"]["per_model"]:
        if cell["dimensions"]["population"] != "injected_fault":
            continue
        for method in ("sparse", "all"):
            info = cell["methods"][method]
            diagnostic = info["nonnegative_cost_sensitivity"]
            ratio = diagnostic["ratio"]
            ratio_text = "undefined" if ratio is None else format(ratio, ".4f")
            original = diagnostic["original_primary_ratio"]
            original_text = "undefined" if original is None else format(original, ".4f")
            lines.append("| {} | {} | {} | {:.3f} | {:.1f} | {:.3f} | {:.3f} | {} | {} |".format(
                cell["dimensions"]["model"], cell["n"], method,
                info["full_over_method_recovery_ratio"]["median"],
                info["journal_payload_bytes"]["median"] / 1024,
                diagnostic["numerator_mean_nonnegative_normal_cost_ms"],
                diagnostic["denominator_mean_signed_recovery_saving_ms"], ratio_text, original_text))
    lines.extend(["", "## Sparse versus all-layer journals", "",
                  "Positive timing differences mean sparse is slower. Observed repeat-range overlap "
                  "and sign counts are noise diagnostics only; neither supplies a confidence interval.", ""])
    for cell in report["groups"]["per_model"]:
        if cell["dimensions"]["population"] != "injected_fault":
            continue
        info = cell["sparse_vs_all"]
        lines.append("- {}: sparse-minus-all median recovery difference {:.3f} ms; "
                     "median payload ratio {:.3f}; sparse repair medians lower/equal/higher "
                     "in {}/{}/{} cases; observed repair ranges overlap in {}/{} cases.".format(
                         cell["dimensions"]["model"],
                         info["sparse_minus_all_recovery_ms"]["median"],
                         info["sparse_over_all_journal_payload_ratio"]["median"],
                         info["cases_sparse_repair_median_lower"],
                         info["cases_equal_repair_median"],
                         info["cases_sparse_repair_median_higher"],
                         info["cases_repair_sample_ranges_overlap"], cell["n"]))
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend("- " + text for text in report["caveats"])
    lines.extend(["", "Per-model/layer/window/fault-shape/divergence tables are in "
                  "`frontier_exploratory.json`; every case, raw timing sample, negative saving, "
                  "and noop control is retained in `frontier_cases.json`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata, rows, hashes = load_complete(run)
    cases = sorted((case_metrics(row) for row in rows), key=lambda c: (c["model"], c["case_id"]))
    audit_path = run / "independent_audit.json"
    audit = {"status": "not_present", "certification": "This analysis is not a raw-results audit."}
    if audit_path.exists():
        audit_raw = audit_path.read_bytes()
        source_audit = json.loads(audit_raw)
        audit = dict(status=source_audit.get("status"), file_sha256=sha(audit_raw),
                     matches_episode_and_metadata_hashes=all(
                         source_audit.get("input_sha256", {}).get(key) == hashes[key]
                         for key in ("episodes", "metadata")))
    dimensions = {
        "per_model": (), "per_layer": ("layer",), "per_window": ("window",),
        "per_fault_shape": ("fault",), "per_kind_and_fault_shape": ("kind", "fault"),
        "per_divergence": ("divergence_class",), "per_layer_window": ("layer", "window"),
        "per_layer_window_shape_divergence": ("layer", "window", "fault", "divergence_class"),
    }
    report = dict(
        status="post_hoc_exploratory_complete", version=VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        input_sha256=hashes, analysis_source_sha256=sha(Path(__file__).read_bytes()),
        protocol=metadata["config"]["protocol"], model_revisions=metadata["models"],
        independent_audit=audit,
        coverage=dict(rows=len(cases), models=len(metadata["models"]),
                      configured_cases_per_model=len(metadata["config"]["cases"]),
                      injected_faults=sum(c["population"] == "injected_fault" for c in cases),
                      noop_controls=sum(c["population"] == "noop_control" for c in cases),
                      repair_repetitions_per_method=metadata["config"]["repetitions"],
                      normal_repetitions_per_mode=metadata["config"]["overhead_repetitions"]),
        selection="No rows excluded. Noops separated. Failed/incomplete/unverified runs rejected, not filtered.",
        aggregation="Case medians first; equal case weighting within each explicitly enumerated cell. "
                    "No cross-model pooled performance or sampling-uncertainty calculation.",
        nonnegative_cost_sensitivity_definition="H_i=max(median(normal_method_i)-median(normal_none_i),0); "
            "S_i=median(repair_full_i)-median(repair_method_i); ratio=mean(H_i)/mean(S_i) only "
            "if mean(S_i)>0. All cell cases, including S_i<=0, enter both means. "
            "Original primary ratio uses mean(signed normal cost)/mean(S_i) only when "
            "mean(S_i)>0 and mean(signed normal cost)>=0; also report the unrestricted signed ratio. "
            "Assumed accounting is normal window cost plus p times one recovery cost. "
            "Above-one ratios cannot break even under at most one incident per window.",
        groups={name: grouped(cases, ("model", "population") + fields)
                for name, fields in dimensions.items()},
        caveats=[
            "Post-hoc exploratory groupings were selected after the debug pilot, before primary completion; "
            "they do not change the frozen experiment or primary analyzer.",
            "Constructed prompts, seeds, finite injections, and repeated policies/timings are not IID "
            "deployment trials; no field fault or divergence rate is inferred.",
            "Nonnegative-cost sensitivity is a noise-sensitive cost/saving diagnostic for this artificial case mix. "
            "Per-case positive clipping prevents negative observed overhead from funding the ratio, "
            "but is not an estimator of true cost, is not a statistical upper bound, and can bias "
            "the numerator upward. Ratios above one give no feasible p in the toy interval [0,1].",
            "Conditional divergence groups describe an observed outcome, not a prospective policy or "
            "causal comparison. Missing factorial cells are absent, not imputed as zero.",
            "Timing repetitions are too few to establish a Pareto frontier. Median orderings, range "
            "overlap and paired signs are descriptive only; no categorical timing dominance is claimed.",
            "Oracle fault detection/localization, trusted checkpoint availability, commit delay, full "
            "reference verification, and production protection costs are not all charged. No system "
            "break-even or deployment recommendation follows from this ratio.",
            "Journal bytes are logical tensor payload, not protected total storage or allocator peaks. "
            "Normal journal capture retains every configured cut even when selected repair cut is zero.",
            "All-layer journals use every nonzero decoder-layer input; sparse journals use the fixed "
            "configured cuts. No placement search or optimized serving-kernel baseline is evaluated.",
            "This analysis covers only the frozen primary/debug runner schema; any supplemental "
            "optimized restoration/cropping baseline is a separate experiment and is not combined here.",
            "Equality counts reuse recorded verification flags. Consult the independent raw-results "
            "audit and its snapshot limits; this script does not reload models or replay inference.",
        ],
    )
    cases_bytes = (json.dumps(cases, indent=2, allow_nan=False) + "\n").encode()
    report["case_output_sha256"] = sha(cases_bytes)
    (run / "frontier_cases.json").write_bytes(cases_bytes)
    (run / "frontier_exploratory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (run / "frontier_exploratory.md").write_text(render_markdown(report))
    print(json.dumps(dict(status=report["status"], coverage=report["coverage"],
                         input_sha256=hashes,
                         analysis_source_sha256=report["analysis_source_sha256"],
                         outputs=[str(run / name) for name in (
                             "frontier_exploratory.json", "frontier_cases.json", "frontier_exploratory.md")]),
                     indent=2))


if __name__ == "__main__":
    main()
