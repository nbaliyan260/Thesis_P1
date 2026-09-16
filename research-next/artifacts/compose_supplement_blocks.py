"""Derive short manuscript blocks from audited supplemental evidence only."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "research-next"
MODELS = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")
LABELS = {MODELS[0]: "SmolLM2-135M", MODELS[1]: "Qwen2.5-0.5B"}
METHODS = (("full", "Conservative full replay"),
           ("layer_view_full", "View / one-layer restore"),
           ("checkpoint_alias_full", "Checkpoint-alias full replay"),
           ("sparse", "Sparse RECUT"), ("all", "All-layer journal"))


def qratio(q):
    return f"{q['median']:.2f}x [{q['q25']:.2f}, {q['q75']:.2f}]"


def qtime(q):
    return f"{q['median']:.3f} ms [{q['q25']:.3f}, {q['q75']:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=PROJECT / "runs/sensitivity-v1")
    args = parser.parse_args()
    audit = json.loads((args.run / "audit.json").read_text())
    summary_bytes = (args.run / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    assert audit["status"] == "passed" and audit["failures"] == []
    assert audit["summary_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert summary["rows"] == 192 and audit["counts"]["detection_cases"] == 216
    lines = ["| Method | Alias-full / method: SmolLM2 | Alias-full / method: Qwen |", "| --- | --- | --- |"]
    for method, label in METHODS:
        ratios = [qratio(summary["models"][m]["comparisons"]["checkpoint_alias_full"][method]["speedup"]) for m in MODELS]
        lines.append(f"| {label} | {ratios[0]} | {ratios[1]} |")
    lines += ["", "Ratios are paired per-case medians with interquartile ranges. Above 1 means the listed method is faster than checkpoint-alias full replay; below 1 means slower. All 96 perturbed cases per model are included.", ""]
    all_equal = all(summary["models"][m]["correctness"][method]["cases_all_repetitions_equal_and_immutable"] == 96
                    for m in MODELS for method, _ in METHODS)
    assert all_equal
    lines.append(f"Every method recovered all 192 reused cases, with exact state/output/continuation checks and immutable-source guards in all {audit['counts']['trials']:,} measured recovery trials. A separate evidence audit passed {audit['checks']:,} consistency checks. It does not independently reload supplemental tensors: none were saved. The final expanded unit suite contains 76 passing tests, distinct from the original 58-test primary gate.")
    lines += ["", "Median checkpoint-alias replay times were " + "; ".join(
        f"{LABELS[m]}: {summary['models'][m]['timing_ms']['checkpoint_alias_full']['median']:.2f} ms" for m in MODELS) + "."]
    lines += ["", "At layer zero, conservative full and sparse invoke the same replay algorithm. Their same-algorithm control ratios were " + "; ".join(
        f"{LABELS[m]}: {summary['models'][m]['c0_controls']['faulty_layer_zero']['comparisons']['full']['sparse']['speedup']['median']:.2f}x"
        for m in MODELS) + ". The alias-full/sparse ratios at that layer were " + "; ".join(
        f"{LABELS[m]}: {summary['models'][m]['c0_controls']['faulty_layer_zero']['comparisons']['checkpoint_alias_full']['sparse']['speedup']['median']:.2f}x"
        for m in MODELS) + ", reflecting restoration differences as well as timing variability. These cases save no decoder-layer work. Per-layer/window strata remain in the summary; a pooled gain is not a gain at every layer."]
    supplemental = "\n".join(lines) + "\n"

    lines = ["| Detector endpoint | SmolLM2-135M | Qwen2.5-0.5B |", "| --- | --- | --- |"]
    for label, function in (
        ("Actually changed prefix cases localized", lambda d: f"{d['perturbed']['unique_layer_routes']}/{d['perturbed']['actual_changed_cases']}"),
        ("No-op controls with an alarm", lambda d: f"{d['noops']['observed_alarm_cases']}/{d['noops']['n']}"),
        ("Comparison time, all cases: median [IQR]", lambda d: qtime(d['all_cases']['timing_ms'])),
        ("Logical comparison reads per window", lambda d: f"{d['all_cases']['logical_bytes_read']['min']/1024**2:.2f}-{d['all_cases']['logical_bytes_read']['max']/1024**2:.2f} MiB"),
    ):
        values = [function(summary["detection"][m]) for m in MODELS]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    actual = sum(summary["detection"][m]["perturbed"]["actual_changed_cases"] for m in MODELS)
    alarms = sum(summary["detection"][m]["perturbed"]["unique_layer_routes"] for m in MODELS)
    noops = sum(summary["detection"][m]["noops"]["observed_alarm_cases"] for m in MODELS)
    assert alarms == actual and noops == 0
    lines += ["", f"The comparison localized {alarms}/{actual} actually changed prefix cases and raised {noops} alarms on 24 no-op controls. There were {192-actual} intended perturbations with no actual rounded change. These are controlled-case results, not sensitivity/specificity estimates for hardware faults. All {audit['counts']['detection_timing_samples']:,} recorded detection timings are separate from recovery timings; no new normal-journal overhead experiment was run.", "",
        "The supplemental repair policy used the detector's unique layer result, not the injected layer. Ground truth was checked only to evaluate the controlled experiment. Exact comparison consumes the already-required trusted snapshot and at least two copies of its logical prefix payload in reads; the table does not count all temporary-flag traffic or protected-storage cost. No claim of low-overhead deployment follows from these short-context measurements."]
    detector = "\n".join(lines) + "\n"
    assert supplemental.isascii() and detector.isascii()
    (PROJECT / "artifacts/supplement_results.md").write_text(supplemental)
    (PROJECT / "artifacts/detector_results.md").write_text(detector)
    provenance = dict(summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={name: hashlib.sha256((PROJECT / "artifacts" / name).read_bytes()).hexdigest()
                 for name in ("supplement_results.md", "detector_results.md")})
    (PROJECT / "artifacts/supplement_blocks_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"supplement": "artifacts/supplement_results.md", "detector": "artifacts/detector_results.md"}))


if __name__ == "__main__":
    main()
