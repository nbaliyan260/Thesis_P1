"""Offline reporting regressions; synthetic scalar records, no models/HPC.

Run from the project root:
research-next/.venv/bin/python -m pytest -q research-next/stage3/reporting/test_reporting.py
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


correction = load("reporting_correction_under_test", HERE / "correct_refusal_counters.py")
independent = load("independent_report_under_test", HERE.parents[1] / "notes" / "independent_stage3_report.py")


def synthetic_correction_data():
    counters = dict(zip(correction.COUNTERS, (2, 8192, 2, 8192, 4096)))
    zero = {key: 0 for key in correction.COUNTERS}
    row = {"workload_id": "fixture", "model": "synthetic", "prefix_tokens": 16, "window": 4,
           "scenario": "checkpoint_corruption", "seed": 7, "variant": "full_batched", "repetition": 0,
           "valid": True, "source_sha256": {"fake.py": "synthetic"}, "config_sha256": "synthetic-config",
           "windows": [{"status": "committed", "valid": True},
                       {"status": "rejected", "valid": True, "rejection": {
                           "reason": "Checkpoint integrity failure", "controller_metadata": counters}}]}
    case = {key: row[key] for key in ("workload_id", "model", "prefix_tokens", "window", "scenario", "seed", "variant")}
    case.update(timing_repetitions=1, windows=[{"status": "committed", "fault_affected": False, "counter_medians": counters.copy()},
                                              {"status": "rejected", "fault_affected": False, "counter_medians": zero.copy()}])
    group = {key: row[key] for key in ("model", "prefix_tokens", "window", "scenario", "variant")}
    group["window_strata"] = [{"status": "committed", "fault_affected": False, "case_windows": 1,
                               "counters": {k: correction.distribution([v]) for k, v in counters.items()}},
                              {"status": "rejected", "fault_affected": False, "case_windows": 1,
                               "counters": {k: correction.distribution([0]) for k in counters}}]
    summary = {"status": "complete_descriptive_stage3", "case_medians": [case], "groups": [group],
               "coverage": {"measured_sessions": 1}, "paired_groups": {"sentinel_timing": [1.25, 2.0]}}
    return summary, [row], counters


def write_run(directory):
    summary, rows, counters = synthetic_correction_data()
    directory.mkdir()
    (directory / "sessions.jsonl").write_text(json.dumps(rows[0]) + "\n")
    for name in ("references", "warmups", "profiles"):
        (directory / (name + ".jsonl")).write_text("")
    metadata = {"status": "complete", "model": "synthetic", "sessions_completed": 1,
                "source_sha256": rows[0]["source_sha256"], "config_sha256": rows[0]["config_sha256"]}
    for name in ("sessions", "references", "warmups", "profiles"):
        metadata[name + "_sha256"] = correction.sha(directory / (name + ".jsonl"))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    summary["input_sha256"] = {"metadata": correction.sha(directory / "metadata.json")}
    summary["input_sha256"].update({name: metadata[name + "_sha256"] for name in ("sessions", "references", "warmups", "profiles")})
    (directory / "summary.json").write_text(json.dumps(summary))
    return summary, counters


def test_correction_is_minimal_and_does_not_mutate_arguments():
    summary, rows, counters = synthetic_correction_data()
    summary_before, rows_before = deepcopy(summary), deepcopy(rows)
    fixed, changes, affected = correction.corrected(summary, rows)
    assert summary == summary_before and rows == rows_before
    assert len(affected) == 1
    assert fixed["case_medians"][0]["windows"][1]["counter_medians"] == counters
    assert fixed["case_medians"][0]["windows"][0] == summary["case_medians"][0]["windows"][0]
    assert fixed["coverage"] == summary["coverage"] and fixed["paired_groups"] == summary["paired_groups"]
    assert len(changes) == 35
    prefixes = ("/case_medians/0/windows/1/counter_medians/", "/groups/0/window_strata/1/counters/")
    assert all(entry["json_pointer"].startswith(prefixes) for entry in changes)
    assert all(entry["before"] == 0 and entry["after"] > 0 for entry in changes)


def test_correction_writes_bound_sidecar_preserves_inputs_and_refuses_overwrite(tmp_path):
    directory = tmp_path / "run"
    write_run(directory)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = correction.run(directory)
    assert result["affected_case_windows"] == 1 and result["changed_json_leaves"] == 35
    assert all((directory / name).read_bytes() == contents for name, contents in original.items())
    provenance = correction.read(directory / "analysis_corrections.json")
    assert provenance["corrected_summary_sha256"] == correction.sha(directory / "summary_corrected.json")
    assert all(correction.sha(directory / name) == value for name, value in provenance["input_sha256"].items())
    assert len(provenance["changes"]) == 35
    with pytest.raises(ValueError, match="overwrite"):
        correction.run(directory)


@pytest.mark.parametrize("target", ["sessions.jsonl", "metadata.json", "references.jsonl", "warmups.jsonl", "profiles.jsonl"])
def test_correction_rejects_tampered_bound_input(tmp_path, target):
    directory = tmp_path / "run"
    write_run(directory)
    path = directory / target
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="binding"):
        correction.run(directory)
    assert not (directory / "summary_corrected.json").exists()


def test_correction_rejects_unexpected_refusal_and_missing_counter():
    summary, rows, _ = synthetic_correction_data()
    rows[0]["windows"][1]["rejection"]["reason"] = "Different failure"
    with pytest.raises(ValueError, match="Unexpected refusal"):
        correction.corrected(summary, rows)
    summary, rows, _ = synthetic_correction_data()
    del rows[0]["windows"][1]["rejection"]["controller_metadata"]["seal_calls"]
    with pytest.raises((KeyError, ValueError)):
        correction.corrected(summary, rows)


def test_correction_detects_an_unrelated_accidental_edit(monkeypatch):
    summary, rows, _ = synthetic_correction_data()
    def faulty_copy(value):
        result = deepcopy(value)
        result["coverage"]["measured_sessions"] += 1
        return result
    monkeypatch.setattr(correction, "deepcopy", faulty_copy)
    with pytest.raises(ValueError, match="unauthorized field"):
        correction.corrected(summary, rows)


def aggregation_row(variant, repetition, seconds):
    window = {"status": "committed", "faults": [], "wall_seconds": seconds,
              "wall_minus_injection_seconds": seconds, "token_availability_seconds": [seconds],
              "step_completion_seconds": [seconds], "release_seconds": seconds}
    window.update({key: 100 for key in independent.MEMORY})
    return {"workload_id": "case-a", "model": "synthetic", "prompt_id": "p", "prefix_tokens": 16,
            "window": 1, "num_layers": 4, "scenario": "clean", "seed": 1, "variant": variant,
            "repetition": repetition, "total_wall_seconds": seconds, "total_with_initialization_seconds": seconds + .1,
            "session_initialization_seconds": .1, "total_injection_harness_seconds": 0, "windows": [window]}


def test_ratios_use_medians_within_case_before_pairing():
    rows = [aggregation_row(variant, rep, value) for variant, values in
            (("full_batched", (1., 9.)), ("sparse_batched", (1., 3.)), ("bare", (1., 1.)), ("native", (2., 2.)))
            for rep, value in enumerate(values)]
    cells = independent.reduce_repetitions(rows)
    pairs = independent.calculate_pairs(cells, {"seeds": [1]})
    ratio = pairs["matched_policy_sessions"][0]["wall_ratio"]
    assert ratio == 2.5  # median(1,9)/median(1,3), not median(1/1,9/3)=2.
    assert ratio != 2.
    assert pairs["matched_policy_sessions"][0]["with_initialization_ratio"] == pytest.approx(5.1 / 2.1)
    assert all(cell["n_repetitions"] == 2 for cell in cells.values())


def test_case_medians_reject_repetitions_with_different_fault_treatments():
    rows = [aggregation_row("full_batched", 0, 1.), aggregation_row("full_batched", 1, 2.)]
    rows[1]["windows"][0]["faults"] = [{"target": "working_prefix"}]
    with pytest.raises(ValueError, match="Treatment changes"):
        independent.reduce_repetitions(rows)


def test_break_even_uses_conditional_complete_window_formula():
    result = independent.break_even(10, 12, 20, 15)
    assert result["H_seconds"] == 2 and result["R_seconds"] == 5
    assert result["p_star"] == pytest.approx(2 / 7)
    assert independent.break_even(10, 9, 20, 15)["p_star"] is None
    assert independent.break_even(10, 12, 20, 21)["p_star"] is None


def test_corrected_summary_selection_requires_bound_sidecar(tmp_path):
    directory = tmp_path / "run"
    write_run(directory)
    correction.run(directory)
    corrected_path = directory / "summary_corrected.json"
    corrected_path.write_text(corrected_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Correction/summary binding"):
        independent.compare_summary({}, directory, "summary_corrected.json")
