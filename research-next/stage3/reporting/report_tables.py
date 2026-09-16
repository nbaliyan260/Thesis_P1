"""Render report tables only from completed, audited stage-3 main evidence."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ORDER = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B")
LABELS = ("Smol-135M", "Qwen-0.5B", "Qwen-7B")
SLUGS = ("smol135m", "qwen05b", "qwen7b")
FAULTS = ("key_early", "value_early", "key_mid", "value_mid", "key_late", "value_late", "multi_prefix", "repeated_prefix")


def parse(text):
    def invalid(value):
        raise ValueError("Nonstandard JSON numeric value: " + value)
    return json.loads(text, parse_constant=invalid)


def read(path):
    return parse(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def choose(items, **terms):
    return [item for item in items if all(item.get(k) == v for k, v in terms.items())]


def metric(rows, field):
    require(bool(rows), "Missing table stratum for " + field)
    require(all(type(row[field]) in (int, float) and math.isfinite(row[field]) for row in rows),
            "Nonfinite/nonnumeric table quantity: " + field)
    return median(row[field] for row in rows)


def validate_inputs(project, directory, run, model):
    """Fail closed on incomplete, stale, or mismatched report dependencies."""
    meta = read(directory / "metadata.json")
    require(meta["status"] == "complete" and meta["model"] == run["model"] == model, "Incomplete/misbound model")
    raw_hashes = {}
    for name in ("metadata", "sessions", "references", "warmups", "profiles"):
        path = directory / (name + (".json" if name == "metadata" else ".jsonl"))
        raw_hashes[name] = sha(path)
        require(run["input_provenance"][name + "_sha256"] == raw_hashes[name], "Raw/recomputation binding: " + name)
        if name != "metadata":
            require(meta[name + "_sha256"] == raw_hashes[name], "Raw/metadata binding: " + name)
    audits = {}
    for name in ("audit.json", "audit_local.json"):
        audit = read(directory / name)
        require(audit["status"] == "passed" and audit["tensor_reload_requested"] is True
                and audit["model"] == model, "Incomplete/misbound tensor audit: " + name)
        for field, digest in raw_hashes.items():
            key = field if field == "metadata" else field + ".jsonl"
            require(audit["hashes"][key] == digest, "Audit/raw binding: " + name + "/" + field)
        require(audit["hashes"]["config"] == meta["config_sha256"]
                and audit["hashes"]["manifest"] == meta["manifest_sha256"], "Audit/input binding: " + name)
        require(audit["hashes"]["auditor"] == sha(project / "stage3/audit_study.py"), "Audit/source binding: " + name)
        require(type(audit["checks"]) is int and audit["checks"] > 0
                and audit["counts"]["snapshot_archives_reloaded"] == 7
                and len(audit["snapshot_results"]) == 7, "Incomplete audit coverage: " + name)
        for snapshot in audit["snapshot_results"]:
            require(snapshot["status"] == "passed" and snapshot["identity"]["model"] == model
                    and type(snapshot["tensor_pairs"]) is int and snapshot["tensor_pairs"] > 0
                    and type(snapshot["bytewise_equal_tensor_pairs"]) is int
                    and 0 <= snapshot["bytewise_equal_tensor_pairs"] <= snapshot["tensor_pairs"],
                    "Invalid audited snapshot: " + name)
        audits[name] = audit
    sidecar_path = directory / "analysis_corrections.json"
    correction = read(sidecar_path)
    require(correction["status"] == "reporting_only_correction" and correction["model"] == model,
            "Invalid correction identity")
    require(correction["correction_script_sha256"] == sha(project / "stage3/reporting/correct_refusal_counters.py"),
            "Correction/source binding")
    allowed = {"summary.json", "metadata.json", "sessions.jsonl", "references.jsonl", "warmups.jsonl", "profiles.jsonl"}
    require(set(correction["input_sha256"]) == allowed
            and all(sha(directory / name) == digest for name, digest in correction["input_sha256"].items()),
            "Correction/raw binding")
    corrected_sha = sha(directory / "summary_corrected.json")
    require(correction["corrected_summary_sha256"] == corrected_sha, "Correction/summary binding")
    comparison = run["analyzer_comparison"]
    require(comparison["status"] == "agree" and comparison["summary_name"] == "summary_corrected.json"
            and comparison["summary_sha256"] == corrected_sha, "Recomputation/corrected-summary binding")
    binding = comparison["correction_binding"]
    require(binding["sidecar_sha256"] == sha(sidecar_path) and binding["corrected_summary_sha256"] == corrected_sha
            and binding["changed_json_leaf_count"] == correction["changed_json_leaf_count"], "Recomputation/correction binding")
    inputs = {str((directory / name).relative_to(project)): sha(directory / name)
              for name in sorted(allowed | {"summary_corrected.json", "analysis_corrections.json", "audit.json", "audit_local.json"})}
    return meta, audits["audit_local.json"], inputs


def main():
    report_path = PROJECT / "notes/stage3_main_recomputed.json"
    report = read(report_path)
    require(report["status"] == "complete_raw_recomputation", "Incomplete raw recomputation")
    require(report["script_sha256"] == sha(PROJECT / "notes/independent_stage3_report.py"), "Recomputation/source binding")
    by_model = {run["model"]: run for run in report["runs"]}
    require(len(report["runs"]) == len(by_model) == len(MODEL_ORDER) and set(by_model) == set(MODEL_ORDER),
            "Missing/duplicate main model")
    matrices = {key: [] for key in ("clean_ms", "clean_ratios", "fault_ratios", "session_ratios", "batch_ratios", "memory", "memory_increment", "availability", "break_even", "validation")}
    audit_total = tensor_pairs = byte_pairs = 0
    main_counts = Counter()
    inputs, headline = {}, {}
    for model, label, slug in zip(MODEL_ORDER, LABELS, SLUGS):
        run = by_model[model]
        directory = PROJECT / "runs/stage3-main-v1" / slug
        meta, audit, bound_inputs = validate_inputs(PROJECT, directory, run, model)
        inputs.update(bound_inputs)
        raw = [parse(line) for line in (directory / "sessions.jsonl").read_text().splitlines() if line.strip()]
        counts = run["counts"]
        require(counts["measured_sessions"] == len(raw) == 786 and counts["committed_windows"] == 2322
                and counts["rejected_windows"] == 18, "Incomplete measured main matrix")
        require(counts["native_custom_parity_positions"] == 228 and counts["snapshot_declarations"] == 7,
                "Incomplete main validation matrix")
        require(audit["counts"]["sessions"]["sessions"] == counts["measured_sessions"]
                and audit["counts"]["sessions"]["windows_committed"] == counts["committed_windows"]
                and audit["counts"]["sessions"]["windows_rejected"] == counts["rejected_windows"],
                "Recomputation/audit count disagreement")
        case_keys = [(c["workload_id"], c["scenario"], c["seed"], c["variant"]) for c in run["case_medians"]]
        require(len(case_keys) == len(set(case_keys)), "Duplicate case medians")
        require(all(c["model"] == model and (c["prefix_tokens"], c["window"]) in ((128, 8), (2048, 16))
                    for c in run["case_medians"]), "Case model/shape separation")
        # Count only the local audit once. The HPC audit repeats the same
        # evidence checks; it is not another set of independent observations.
        audit_total += audit["checks"]
        pairs = sum(s["tensor_pairs"] for s in audit["snapshot_results"])
        tensor_pairs += pairs
        byte_pairs += sum(s["bytewise_equal_tensor_pairs"] for s in audit["snapshot_results"])
        matrices["validation"].append([label, f'{audit["checks"]:,}', pairs, counts["partial_to_full_replay_windows"]])
        for k, value in counts.items():
            if type(value) is int:
                if k == "largest_recorded_seal_batch_bytes":
                    main_counts[k] = max(main_counts[k], value)
                else:
                    main_counts[k] += value
        for prefix in (128, 2048):
            ident = label + " / " + str(prefix)
            cells = choose(run["case_medians"], prefix_tokens=prefix, scenario="clean")
            ms = [metric(choose(cells, variant=v), "session_with_initialization") * 1000
                  for v in ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched")]
            matrices["clean_ms"].append([ident] + [f"{v:.1f}" for v in ms])
            overhead = choose(run["paired_records"]["protection_overhead"], prefix_tokens=prefix, protected="sparse_batched")
            ratios = [metric(choose(overhead, unprotected_baseline=b), "session_with_initialization_protected_over_baseline") for b in ("bare", "native")]
            matrices["clean_ratios"].append([ident] + [f"{v:.2f}x" for v in ratios])
            matched = choose(run["paired_records"]["matched_policy_windows"], prefix_tokens=prefix, numerator="full_batched", denominator="sparse_batched", fault_affected=True)
            fault_ratios = [metric(choose(matched, scenario=s), "wall_ratio") for s in FAULTS]
            matrices["fault_ratios"].append([ident] + [f"{v:.2f}" for v in fault_ratios])
            session = [r for r in run["paired_records"]["matched_policy_sessions"] if r["prefix_tokens"] == prefix and r["scenario"] in FAULTS]
            ratios = [metric(choose(session, numerator=a, denominator=b), "with_initialization_ratio")
                      for a, b in (("full_batched", "sparse_batched"), ("full_batched", "allcuts_batched"), ("sparse_batched", "allcuts_batched"))]
            matrices["session_ratios"].append([ident] + [f"{v:.2f}" for v in ratios])
            ablation = choose(run["paired_records"]["sealing_ablation"], prefix_tokens=prefix)
            ratios = [metric(choose(ablation, legacy=v, scenario=s), "session_legacy_over_batched_with_initialization_ratio")
                      for s in ("clean", "value_late") for v in ("full_legacy", "sparse_legacy")]
            matrices["batch_ratios"].append([ident] + [f"{v:.2f}" for v in ratios])
            shape_rows = choose(raw, prefix_tokens=prefix)
            peak = max(w["peak_allocated_bytes"] for r in shape_rows for w in r["windows"]) / 1024**3
            journals = [median(w["logical_journal_bytes"] for r in choose(shape_rows, scenario="clean", variant=v) for w in r["windows"]) / 1024
                        for v in ("sparse_batched", "allcuts_batched")]
            matrices["memory"].append([ident, f"{peak:.2f}", f"{journals[0]:.0f}", f"{journals[1]:.0f}"])
            increments = [median(w["memory"]["peak_increment_bytes"] for c in choose(cells, variant=v) for w in c["windows"]) / 1024**2
                          for v in ("bare", "sparse_batched")]
            checkpoint = median(w["memory"]["logical_checkpoint_bytes"] for c in choose(cells, variant="sparse_batched") for w in c["windows"]) / 1024**2
            matrices["memory_increment"].append([ident] + [f"{v:.3f}" for v in increments + [checkpoint]])
            available = []
            for v in ("native", "bare", "sparse_batched"):
                available.append(median(w["availability"]["first_token_seconds"] * 1000 for c in choose(cells, variant=v) for w in c["windows"]))
            delay = median(w["availability"]["mean_hold_after_step_seconds"] * 1000 for c in choose(cells, variant="sparse_batched") for w in c["windows"])
            matrices["availability"].append([ident] + [f"{v:.1f}" for v in available + [delay]])
            sensitivity = choose(run["paired_records"]["break_even_sensitivity"], prefix_tokens=prefix, numerator="full_batched", denominator="sparse_batched")
            values = [row["raw"]["p_star"] for row in sensitivity if row["raw"]["p_star"] is not None]
            signs = Counter(row["raw"]["classification"] for row in sensitivity)
            matrices["break_even"].append([ident, f"{len(values)}/{len(sensitivity)}", f"{100 * median(values):.1f}%" if values else "n/a",
                signs["no_extra_clean_cost_with_positive_fault_saving"], len(sensitivity) - len(values) - signs["no_extra_clean_cost_with_positive_fault_saving"]])
            headline[ident] = {"native_session_ms": ms[0], "bare_session_ms": ms[1], "full_session_ms": ms[2], "sparse_session_ms": ms[3],
                              "sparse_over_bare": matrices["clean_ratios"][-1][1], "peak_allocated_GiB": peak,
                              "full_over_sparse_fault_window": dict(zip(FAULTS, fault_ratios))}
        inputs[str((directory / "metadata.json").relative_to(PROJECT))] = sha(directory / "metadata.json")
        inputs[str((directory / "summary_corrected.json").relative_to(PROJECT))] = sha(directory / "summary_corrected.json")
    headers = {
        "clean_ms": ["Model / prefix", "Native ms", "Bare ms", "Full ms", "Sparse ms", "All-cuts ms"],
        "clean_ratios": ["Model / prefix", "Sparse / bare", "Sparse / native"],
        "fault_ratios": ["Model / prefix", "E-K", "E-V", "M-K", "M-V", "L-K", "L-V", "Multi", "Repeat"],
        "session_ratios": ["Model / prefix", "Full / sparse", "Full / all-cuts", "Sparse / all-cuts"],
        "batch_ratios": ["Model / prefix", "Clean full", "Clean sparse", "Late-V full", "Late-V sparse"],
        "memory": ["Model / prefix", "Peak GiB", "Sparse KiB", "All-cuts KiB"],
        "memory_increment": ["Model / prefix", "Bare increase MiB", "Sparse increase MiB", "Checkpoint MiB"],
        "availability": ["Model / prefix", "Native ms", "Bare ms", "Sparse ms", "Median mean hold ms"],
        "break_even": ["Model / prefix", "Defined / all", "Median p*", "No extra H", "Other signs"],
        "validation": ["Model", "Audit checks", "Tensor pairs", "Partial-to-full"]}
    output = {"input_recomputation_sha256": sha(report_path), "generator_sha256": sha(__file__), "input_sha256": inputs,
              "tables": {k: table(headers[k], rows) for k, rows in matrices.items()}, "numeric_rows": matrices,
              "totals": dict(main_counts), "local_audit_checks": audit_total, "snapshot_tensor_pairs": tensor_pairs,
              "snapshot_bytewise_equal_pairs": byte_pairs, "headline": headline}
    output["aggregation_notes"] = {
        "hierarchy": "Median timing repetitions within a workload/scenario/seed/variant first; then median across matched case ratios, separately by model and prefix/window shape.",
        "sessions": "All session-cost tables include the separately timed constructor; common initial fork, prefill and reference checks remain excluded.",
        "fault_columns": "All eight configured fault strata are shown separately. Fault-session aggregate weights each configured case-policy cell once; guard challenges are excluded.",
        "memory": "Peak GiB is the maximum recorded GPU allocation over all measured variants/scenarios at that model/shape; journal KiB are clean logical payload medians, not GPU peaks.",
        "memory_increment": "Clean case-window medians after within-case repetition reduction: nine prompt/window cells per model/shape. Increase is allocator peak minus that window's starting allocation, not total GPU allocation or a logical protection-only payload; experimental allocation effects and temporary batching can contribute. Checkpoint MiB is logical sparse-policy payload. Host peak is not measured.",
        "availability": "Per-window first-token availability medians; hold column is the median of per-case-window mean token hold durations, not end-to-end network TTFT.",
        "break_even": "Defined/all uses all matched affected case-windows. Repeated-prefix contributes three affected windows per case; the seven other fault strata contribute one each. Undefined sign combinations remain in the denominator.",
        "audits": "Only local audit checks and tensor pairs are totaled; the HPC rerun of the same audit is not added."
    }
    path = PROJECT / "stage3/reporting/report_tables.json"
    with path.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(path), "totals": dict(main_counts), "audit_checks": audit_total}, indent=2))


if __name__ == "__main__":
    main()
