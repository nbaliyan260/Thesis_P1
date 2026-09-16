"""Synthetic delivery-table regressions; no scientific run or GPU is performed.

The fixture deliberately supplies compact scalar records and bound provenance
relationships expected by the delivery layer, not a substitute for the frozen
experiment's complete independent evidence audit.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tables_under_test", HERE / "report_tables.py")
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def table_cases(model):
    cells, overheads, fault_windows, sessions, ablations, sensitivity = [], [], [], [], [], []
    values = {"native": [1., 3., 100.], "bare": [2., 4., 200.], "full_batched": [5., 9., 300.],
              "sparse_batched": [7., 11., 400.], "allcuts_batched": [13., 15., 500.]}
    for prefix, window, scale in ((128, 8, 1), (2048, 16, 3)):
        for prompt in range(3):
            for variant, durations in values.items():
                availability = {"first_token_seconds": scale * (prompt + 1) * (.1 if variant in ("native", "bare") else 1.),
                                "mean_hold_after_step_seconds": scale * (.3 + .1 * prompt)}
                increment = {"native": 1, "bare": 2, "full_batched": 7, "sparse_batched": 9, "allcuts_batched": 11}[variant]
                memory = {"peak_increment_bytes": (increment + prompt) * scale * 1024**2,
                          "logical_checkpoint_bytes": (4 + prompt) * scale * 1024**2 if variant not in ("native", "bare") else 0}
                cells.append({"model": model, "workload_id": f"{model}-{prefix}-{prompt}", "prefix_tokens": prefix, "window": window,
                              "scenario": "clean", "seed": 1, "variant": variant,
                              "session_with_initialization": durations[prompt] * scale,
                              "windows": [{"availability": availability.copy(), "memory": memory.copy()} for _ in range(3)]})
            for baseline in ("bare", "native"):
                overheads.append({"prefix_tokens": prefix, "protected": "sparse_batched", "unprotected_baseline": baseline,
                                  "session_with_initialization_protected_over_baseline": values["sparse_batched"][prompt] / values[baseline][prompt]})
        for index, scenario in enumerate(tables.FAULTS):
            for number, ratio in enumerate((1., 1., 1., 3., 9., 100.)):
                count = 3 if scenario == "repeated_prefix" else 1
                for wi in range(count):
                    fault_windows.append({"prefix_tokens": prefix, "scenario": scenario, "numerator": "full_batched",
                                          "denominator": "sparse_batched", "fault_affected": True,
                                          "wall_ratio": ratio * scale * (index + 1)})
                    # 60 affected windows per shape, with 20 defined, 30
                    # nonpositive-H/positive-R, and 10 other sign patterns.
                    ordinal = sum(row["prefix_tokens"] == prefix for row in sensitivity)
                    category = ("extra_clean_cost_and_positive_fault_saving" if ordinal < 20 else
                                "no_extra_clean_cost_with_positive_fault_saving" if ordinal < 50 else
                                "lower_clean_cost_but_higher_fault_cost")
                    sensitivity.append({"prefix_tokens": prefix, "scenario": scenario, "numerator": "full_batched",
                                        "denominator": "sparse_batched", "raw": {
                                            "classification": category, "p_star": .1 if ordinal < 20 else None}})
                for left, right, value in (("full_batched", "sparse_batched", 1.2),
                                           ("full_batched", "allcuts_batched", 1.08),
                                           ("sparse_batched", "allcuts_batched", .9)):
                    sessions.append({"prefix_tokens": prefix, "scenario": scenario, "numerator": left,
                                     "denominator": right, "with_initialization_ratio": value * scale})
        # A guard challenge must not enter the fair session aggregate.
        sessions.append({"prefix_tokens": prefix, "scenario": "journal_corruption", "numerator": "full_batched",
                         "denominator": "sparse_batched", "with_initialization_ratio": 999999.})
        for scenario, offset in (("clean", 0), ("value_late", 1)):
            for legacy, ratio in (("full_legacy", 1.3), ("sparse_legacy", 1.5)):
                ablations.append({"prefix_tokens": prefix, "scenario": scenario, "legacy": legacy,
                                  "session_legacy_over_batched_with_initialization_ratio": ratio + offset})
    return cells, {"protection_overhead": overheads, "matched_policy_windows": fault_windows,
                   "matched_policy_sessions": sessions, "sealing_ablation": ablations, "break_even_sensitivity": sensitivity}


def synthetic_project(project):
    for name in ("notes/independent_stage3_report.py", "stage3/audit_study.py", "stage3/reporting/correct_refusal_counters.py"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Explicit synthetic provenance sentinel, not an executable experiment.\n")
    runs = []
    for index, (model, slug) in enumerate(zip(tables.MODEL_ORDER, tables.SLUGS)):
        directory = project / "runs/stage3-main-v1" / slug
        directory.mkdir(parents=True)
        rows = []
        for number in range(786):
            prefix = 128 if number < 393 else 2048
            variant = ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched")[number % 5]
            refused = number >= 768
            windows = []
            for wi in range(2 if refused else 3):
                journal = {"sparse_batched": 16, "allcuts_batched": 128}.get(variant, 0) * 1024 * (1 if prefix == 128 else 2)
                peak = (1 if prefix == 128 else 2) * 1024**3
                if number == 392:
                    peak = 9 * 1024**3
                if number == 785:
                    peak = 30 * 1024**3
                windows.append({"status": "rejected" if refused and wi == 1 else "committed",
                                "peak_allocated_bytes": peak, "logical_journal_bytes": journal})
            rows.append({"prefix_tokens": prefix, "scenario": "checkpoint_corruption" if refused else "clean",
                         "variant": variant, "windows": windows})
        (directory / "sessions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        for phase in ("references", "warmups", "profiles"):
            (directory / (phase + ".jsonl")).write_text("{}\n")
        meta = {"status": "complete", "model": model, "config_sha256": "fixture-config", "manifest_sha256": "fixture-manifest"}
        meta.update({phase + "_sha256": tables.sha(directory / (phase + ".jsonl")) for phase in ("sessions", "references", "warmups", "profiles")})
        write(directory / "metadata.json", meta)
        raw_hashes = {phase + "_sha256": tables.sha(directory / (phase + (".json" if phase == "metadata" else ".jsonl")))
                      for phase in ("metadata", "sessions", "references", "warmups", "profiles")}
        for name, checks in (("audit.json", 900000 + index), ("audit_local.json", 100 + index)):
            audit_hashes = {(phase if phase == "metadata" else phase + ".jsonl"): digest
                            for phase, digest in ((key[:-7], value) for key, value in raw_hashes.items())}
            audit_hashes.update(config=meta["config_sha256"], manifest=meta["manifest_sha256"], auditor=tables.sha(project / "stage3/audit_study.py"))
            audit = {"status": "passed", "model": model, "tensor_reload_requested": True, "checks": checks, "hashes": audit_hashes,
                     "counts": {"snapshot_archives_reloaded": 7, "sessions": {"sessions": 786, "windows_committed": 2322, "windows_rejected": 18}},
                     "snapshot_results": [{"status": "passed", "identity": {"model": model}, "tensor_pairs": 8,
                                           "bytewise_equal_tensor_pairs": 8} for _ in range(7)]}
            write(directory / name, audit)
        write(directory / "summary.json", {"fixture": "original"})
        write(directory / "summary_corrected.json", {"fixture": "corrected"})
        corrected_sha = tables.sha(directory / "summary_corrected.json")
        correction = {"status": "reporting_only_correction", "model": model,
                      "correction_script_sha256": tables.sha(project / "stage3/reporting/correct_refusal_counters.py"),
                      "corrected_summary_sha256": corrected_sha, "changed_json_leaf_count": 1,
                      "input_sha256": {name: tables.sha(directory / name) for name in
                                      ("summary.json", "metadata.json", "sessions.jsonl", "references.jsonl", "warmups.jsonl", "profiles.jsonl")}}
        write(directory / "analysis_corrections.json", correction)
        cells, pairs = table_cases(model)
        runs.append({"model": model, "input_provenance": raw_hashes, "case_medians": cells, "paired_records": pairs,
                     "counts": {"measured_sessions": 786, "committed_windows": 2322, "rejected_windows": 18,
                                "native_custom_parity_positions": 228, "snapshot_declarations": 7,
                                "partial_to_full_replay_windows": index, "largest_recorded_seal_batch_bytes": (index + 1) * 1024},
                     "analyzer_comparison": {"status": "agree", "summary_name": "summary_corrected.json", "summary_sha256": corrected_sha,
                         "correction_binding": {"sidecar_sha256": tables.sha(directory / "analysis_corrections.json"),
                                                "corrected_summary_sha256": corrected_sha, "changed_json_leaf_count": 1}}})
    report = {"status": "complete_raw_recomputation", "script_sha256": tables.sha(project / "notes/independent_stage3_report.py"), "runs": runs}
    write(project / "notes/stage3_main_recomputed.json", report)
    return report


@pytest.fixture
def project(tmp_path, monkeypatch):
    synthetic_project(tmp_path)
    monkeypatch.setattr(tables, "PROJECT", tmp_path)
    return tmp_path


def test_tables_preserve_case_hierarchy_shapes_units_and_denominators(project):
    tables.main()
    result = tables.read(project / "stage3/reporting/report_tables.json")
    rows = result["numeric_rows"]
    assert rows["clean_ms"][0][1:] == ["3000.0", "4000.0", "9000.0", "11000.0", "15000.0"]
    assert rows["clean_ms"][1][1:] == ["9000.0", "12000.0", "27000.0", "33000.0", "45000.0"]
    assert rows["clean_ratios"][0][1:] == ["2.75x", "4.00x"]  # Median of paired case ratios, not 11/3.
    assert rows["fault_ratios"][0][1:] == [f"{2 * n:.2f}" for n in range(1, 9)]
    assert rows["fault_ratios"][1][1:] == [f"{6 * n:.2f}" for n in range(1, 9)]
    assert rows["session_ratios"][0][1:] == ["1.20", "1.08", "0.90"]
    assert rows["batch_ratios"][0][1:] == ["1.30", "1.50", "2.30", "2.50"]
    assert rows["memory"][0][1:] == ["9.00", "16", "128"]  # Raw maximum, not the 1 GiB median.
    assert rows["memory"][1][1:] == ["30.00", "32", "256"]
    assert rows["memory_increment"][0][1:] == ["3.000", "10.000", "5.000"]
    assert rows["memory_increment"][1][1:] == ["9.000", "30.000", "15.000"]
    assert len(rows["memory_increment"]) == 6
    assert rows["availability"][0][1:] == ["200.0", "200.0", "2000.0", "400.0"]
    assert rows["break_even"][0][1:] == ["20/60", "10.0%", 30, 10]
    assert result["local_audit_checks"] == 303  # Never add the HPC repetition's 2,700,003 checks.
    assert result["snapshot_tensor_pairs"] == result["snapshot_bytewise_equal_pairs"] == 168
    assert result["totals"]["largest_recorded_seal_batch_bytes"] == 3072  # Max, not a sum.
    assert result["totals"]["measured_sessions"] == 2358
    assert "Median mean hold ms" in result["tables"]["availability"]
    assert "Repeated-prefix contributes three" in result["aggregation_notes"]["break_even"]
    assert "not total GPU allocation" in result["aggregation_notes"]["memory_increment"]
    assert len(result["input_sha256"]) == 30  # All ten dependencies for each model.


@pytest.mark.parametrize("target", ["sessions.jsonl", "references.jsonl", "warmups.jsonl", "profiles.jsonl",
                                   "metadata.json", "summary.json", "summary_corrected.json", "analysis_corrections.json"])
def test_stale_bound_evidence_blocks_delivery(project, target):
    path = project / "runs/stage3-main-v1/smol135m" / target
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="binding"):
        tables.main()
    assert not (project / "stage3/reporting/report_tables.json").exists()


def test_audit_raw_binding_and_corrected_summary_selection_are_required(project):
    directory = project / "runs/stage3-main-v1/smol135m"
    path = directory / "audit_local.json"
    audit = tables.read(path)
    audit["hashes"]["sessions.jsonl"] = "stale"
    write(path, audit)
    with pytest.raises(ValueError, match="Audit/raw binding"):
        tables.main()


def test_duplicates_and_uncorrected_comparisons_are_rejected(project):
    path = project / "notes/stage3_main_recomputed.json"
    report = tables.read(path)
    original = deepcopy(report)
    report["runs"].append(deepcopy(report["runs"][0]))
    write(path, report)
    with pytest.raises(ValueError, match="duplicate main model"):
        tables.main()
    original["runs"][0]["analyzer_comparison"]["status"] = "agree_except_refusal_counter_display"
    write(path, original)
    with pytest.raises(ValueError, match="corrected-summary binding"):
        tables.main()


def test_missing_fault_stratum_and_duplicate_case_are_not_silently_pooled(project):
    path = project / "notes/stage3_main_recomputed.json"
    report = tables.read(path)
    original = deepcopy(report)
    report["runs"][0]["paired_records"]["matched_policy_windows"] = [
        row for row in report["runs"][0]["paired_records"]["matched_policy_windows"] if row["scenario"] != "value_mid"]
    write(path, report)
    with pytest.raises(ValueError, match="Missing table stratum"):
        tables.main()
    original["runs"][0]["case_medians"].append(deepcopy(original["runs"][0]["case_medians"][0]))
    write(path, original)
    with pytest.raises(ValueError, match="Duplicate case medians"):
        tables.main()


def test_finite_json_and_nonoptimized_provenance_gates():
    with pytest.raises(ValueError, match="Nonstandard"):
        tables.parse('{"value": NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        tables.metric([{"value": float("inf")}], "value")
    code = "import runpy; runpy.run_path(" + repr(str(HERE / "report_tables.py")) + ")['require'](False, 'optimization-safe gate')"
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0 and "ValueError: optimization-safe gate" in result.stderr


def test_refuses_overwriting_derived_tables(project):
    tables.main()
    target = project / "stage3/reporting/report_tables.json"
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        tables.main()
    assert target.read_bytes() == before
