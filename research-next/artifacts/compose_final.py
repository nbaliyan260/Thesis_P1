"""Fill the professor draft only from completed, audited primary evidence.

The supplemental-results prose is separately reviewed and supplied as a Markdown
file; it must never be silently replaced by the original weaker-baseline result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "research-next"
MODELS = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")
SHORT = {MODELS[0]: "SmolLM2-135M", MODELS[1]: "Qwen2.5-0.5B"}


def load(path):
    return json.loads(path.read_text())


def ratio_range(q):
    return f"{q['median']:.2f}x [{q['q25']:.2f}, {q['q75']:.2f}]"


def pct_range(q):
    return f"{100*q['median']:.2f}% [{100*q['q25']:.2f}, {100*q['q75']:.2f}]"


def kib_range(q):
    return f"{q['min']/1024:,.1f}-{q['max']/1024:,.1f} KiB"


def boundary_block(primary_raw_hash):
    run = PROJECT / 'runs/boundary-extension-v1'
    metadata = load(run / 'metadata.json')
    assert metadata['primary_raw_sha256'] == primary_raw_hash
    assert metadata['new_fault_samples'] is False and metadata['performance_evidence'] is False
    intro = ("A separately declared, post-hoc branch test reused Qwen primary-063's exact prefix, seed and "
             "recorded fault, extending W from 12 to 14 without searching alternative faults. ")
    if metadata['status'] != 'complete':
        assert metadata['status'] in ('failed', 'skipped')
        return intro + ("The targeted capture did not complete successfully. Its preserved metadata reports "
                        f"status '{metadata['status']}'. It supplies no positive fallback-validation evidence.")
    audit = load(run / 'independent_boundary_audit_hpc.json')
    local_audit = load(run / 'independent_boundary_audit.json')
    assert audit['status'] == 'passed' and not audit['failures']
    assert local_audit['status'] == 'passed' and not local_audit['failures']
    assert audit['diagnostic_metadata_sha256'] == hashlib.sha256((run / 'metadata.json').read_bytes()).hexdigest()
    assert local_audit['diagnostic_metadata_sha256'] == audit['diagnostic_metadata_sha256']
    row = load(run / 'diagnostic_row.json')
    assert metadata['diagnostic_row_sha256'] == hashlib.sha256((run / 'diagnostic_row.json').read_bytes()).hexdigest()
    archive = metadata['archive']
    assert archive['sha256'] == hashlib.sha256((run / archive['file']).read_bytes()).hexdigest()
    assert row['first_divergence'] == 12 and row['case']['window'] == 14 and row['valid_repairs_equal']
    for method, cut in (('full', 0), ('sparse', 6), ('all', 11)):
        assert row['methods'][method]['start_layers'] == [cut]*12+[0]*2
    checks = sum(group['passed'] for group in audit['groups'].values())
    return intro + (
        "Its first 12 clean/faulty decisions and complete injected-fault record matched the primary. "
        "At output 12 the histories diverged; sparse replay used cut 6 for transitions 1-12 and layer 0 for 13-14. "
        "All three valid methods exactly recovered the extended state and continuation. Sparse/all/full decoder-layer "
        f"work was 264/204/336, respectively. A separate CPU audit passed {checks:,} checks including the saved "
        f"{archive['bytes']/1024**2:.2f} MiB state/logit/token archive and traces. This is one selected branch diagnostic, "
        "not an additional independent fault or a performance result. Archived tensors omit continuation; its equality "
        "remains live runner evidence."
    )


def render_blocks(run, supplemental_run, supplemental_prose, detector_prose):
    metadata = load(run / "metadata.json")
    summary = load(run / "summary.json")
    audit = load(run / "independent_audit_hpc.json")
    local_audit = load(run / "independent_audit.json")
    assert metadata["status"] == "complete"
    assert audit["status"] == local_audit["status"] == "passed"
    raw_hash = hashlib.sha256((run / "episodes.jsonl").read_bytes()).hexdigest()
    assert metadata["episodes_sha256"] == summary["episode_sha256"] == raw_hash
    assert audit["input_sha256"]["episodes"] == local_audit["input_sha256"]["episodes"] == raw_hash
    primary_summary_hash = hashlib.sha256((run / "summary.json").read_bytes()).hexdigest()
    assert primary_summary_hash == audit["input_sha256"]["summary"] == local_audit["input_sha256"]["summary"]
    primary_metadata_hash = hashlib.sha256((run / "metadata.json").read_bytes()).hexdigest()
    assert primary_metadata_hash == audit["input_sha256"]["metadata"] == local_audit["input_sha256"]["metadata"]
    assert not audit["failures"] and not local_audit["failures"]
    supplemental_metadata = load(supplemental_run / "metadata.json")
    supplemental_audit = load(supplemental_run / "audit.json")
    supplemental_audit_hpc = load(supplemental_run / "audit_hpc.json")
    supplemental_summary = load(supplemental_run / "summary.json")
    assert supplemental_metadata["status"] == "complete"
    assert supplemental_audit["status"] == "passed"
    assert supplemental_audit_hpc['status'] == 'passed' and not supplemental_audit_hpc['failures']
    assert supplemental_audit["failures"] == []
    assert supplemental_metadata["primary_raw_sha256"] == raw_hash
    assert hashlib.sha256((supplemental_run / "summary.json").read_bytes()).hexdigest() == supplemental_audit["summary_sha256"]
    assert supplemental_audit_hpc['summary_sha256'] == supplemental_audit['summary_sha256']
    block_provenance = load(PROJECT / 'artifacts/supplement_blocks_provenance.json')
    figure_path = PROJECT / 'artifacts/figures/recovery_by_layer.png'
    figure_provenance = load(figure_path.with_suffix('.json'))
    assert block_provenance['summary_sha256'] == figure_provenance['summary_sha256'] == supplemental_audit['summary_sha256']
    for prose in (supplemental_prose, detector_prose):
        assert hashlib.sha256(prose.read_bytes()).hexdigest() == block_provenance['outputs'][prose.name]
    assert hashlib.sha256(figure_path.read_bytes()).hexdigest() == figure_provenance['image_sha256']
    for filename, audit_key, metadata_key in (("episodes.jsonl", "episodes", "episodes_sha256"),
                                              ("detection.jsonl", "detection", "detection_sha256")):
        actual = hashlib.sha256((supplemental_run / filename).read_bytes()).hexdigest()
        assert actual == supplemental_metadata[metadata_key] == supplemental_audit["input_sha256"][audit_key]
        assert actual == supplemental_audit_hpc['input_sha256'][audit_key]
    total, models = summary["total"], summary["models"]
    assert (total["n"], total["faults"], total["noops"]) == (216, 192, 24)
    assert all(models[m]["n"] == 108 for m in MODELS)
    assert all(total["correctness"][m]["all_equal_faults"] == 192 for m in ("full", "sparse", "all"))
    assert all(total["correctness"][m]["all_equal_controls"] == 24 for m in ("full", "sparse", "all"))

    sparse = [models[m]["performance"]["sparse"]["paired_speedup"]["median"] for m in MODELS]
    overhead = [models[m]["performance"]["sparse"]["paired_normal_overhead_fraction"]["median"] for m in MODELS]
    alias_ratios = [supplemental_summary["models"][m]["comparisons"]["checkpoint_alias_full"]["sparse"]["speedup"]["median"] for m in MODELS]
    detector_actual = sum(supplemental_summary["detection"][m]["perturbed"]["actual_changed_cases"] for m in MODELS)
    detector_localized = sum(supplemental_summary["detection"][m]["perturbed"]["unique_layer_routes"] for m in MODELS)
    blocks = {}
    blocks['BOUNDARY_RESULTS'] = boundary_block(raw_hash)
    blocks["ABSTRACT_RESULTS"] = (
        f"On an RTX 5000 Ada GPU, the frozen primary study exercised {total['n']} scenarios "
        f"across SmolLM2-135M and Qwen2.5-0.5B: {total['faults']} finite perturbations and {total['noops']} no-op controls. "
        "Full replay, sparse RECUT and all-layer activation recovery each matched the clean reference "
        "exactly on every tested state/output/continuation check. "
        f"Only {total['token_divergences']} perturbed trajectories changed a greedy token within the W-step window, both at its final step. "
        f"Against the conservative full-replay implementation, median paired sparse recovery speedups were "
        f"{sparse[0]:.2f}x and {sparse[1]:.2f}x, with measured median normal-path overheads of "
        f"{100*overhead[0]:.2f}% and {100*overhead[1]:.2f}%, respectively. "
        f"In the separate all-case sensitivity study, sparse recovery ratios against the stronger checkpoint-alias full replay were "
        f"{alias_ratios[0]:.2f}x and {alias_ratios[1]:.2f}x (above 1 means faster). "
        f"Snapshot comparison localized {detector_localized}/{detector_actual} actually changed old-prefix cases; its broader blind spots remain unaddressed. "
        "Recovery speedup is not an end-to-end serving-throughput claim."
    )
    blocks["ENVIRONMENT"] = (
        f"The primary run used one {metadata['gpu']}, BF16, batch size one and eager attention. "
        f"The pinned runtime was Python {metadata['python']}, PyTorch {metadata['torch']}, "
        f"Transformers {metadata['transformers']} and CUDA {metadata['cuda']}. "
        "Deterministic algorithms were enabled, TF32 disabled, and the execution configuration retained in metadata. "
        "The then-current 58-test CPU suite passed in the allocated HPC environment before the primary run. "
        "The eight-case debug run used separate inputs and is excluded from every primary result below."
    )
    lines = ["| Method / endpoint | Perturbed cases | No-op controls |", "| --- | --- | --- |"]
    labels = {"none": "No recovery", "slotonly": "Original-slot restore", "full": "Full suffix replay", "sparse": "Sparse RECUT", "all": "All-layer journal"}
    for method, label in labels.items():
        c = total["correctness"][method]
        lines.append(f"| {label}: all checks equal | {c['all_equal_faults']}/192 | {c['all_equal_controls']}/24 |")
    for method in ("none", "slotonly"):
        c = total["correctness"][method]
        lines.append(f"| {labels[method]}: tokens only | {c['tokens_equal_faults']}/192 | Not the full-state criterion |")
    blocks["CORRECTNESS_TABLE"] = "\n".join(lines)
    model_counts = "; ".join(f"{SHORT[m]}: {models[m]['token_divergences']}/96 token-changing faults" for m in MODELS)
    slot = total["correctness"]["slotonly"]
    blocks["CORRECTNESS_INTERPRETATION"] = (
        f"Every valid recovery method passed {models[MODELS[0]]['n']}/108 cases on SmolLM2 and "
        f"{models[MODELS[1]]['n']}/108 on Qwen. {model_counts}. "
        f"Original-slot restoration matched the cache in {slot['cache_equal_faults']}/192 faults and final logits in "
        f"{slot['logits_equal_faults']}/192; these narrower endpoints do not replace the joint recovery criterion. "
        f"There were {total['no_actual_change_faults']} perturbed cases with no actual rounded tensor change. "
        "These counts are descriptive of the fixed matrix, not estimates of service fault impact. "
        "Multiple recovery policies evaluated on the same corrupted trajectory do not multiply the sample size."
    )
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_bytes().splitlines()]
    faults = [r for r in rows if r["case"]["fault"] != "noop"]
    divergent = [r for r in faults if r['first_divergence'] is not None]
    assert len(divergent) == 2
    assert all(r['first_divergence'] == r['case']['window'] == 12 for r in divergent)
    blocks['CORRECTNESS_INTERPRETATION'] += (
        " Both token divergences occurred at j=W=12: Qwen primary-052 at layer 0/cut 0 and "
        "primary-063 at layer 11/sparse cut 6. Thus no primary case executed a replay step "
        "after an observed token divergence; the primary alone does not empirically validate that branch."
    )
    nonfinite = sum(any(not r["methods"]["none"][field]["finite"] for field in
                       ("cache", "logits", "continuation_cache", "continuation_logits")) for r in faults)
    same_tokens_wrong_cache = sum(r["methods"]["slotonly"]["tokens_equal"] and
                                 not r["methods"]["slotonly"]["cache"]["equal"] for r in faults)
    same_cache_wrong_logits = sum(r["methods"]["slotonly"]["cache"]["equal"] and
                                 not r["methods"]["slotonly"]["logits"]["equal"] for r in faults)
    uncorrected_continuation_changes = sum(not r['methods']['none']['continuation_tokens_equal'] for r in faults)
    blocks["CORRECTNESS_INTERPRETATION"] += (
        f" The faulty/no-repair state, logits or continuation were non-finite in {nonfinite}/192 cases. "
        f"After slot-only restoration, {same_tokens_wrong_cache} cases kept the same W tokens but had a different cache; "
        f"{same_cache_wrong_logits} had the same cache but different final logits. Thus not every failed repair is descendant KV corruption. "
        f"The two additional uncorrected continuation decisions differed in {uncorrected_continuation_changes}/192 cases; "
        "the two within-window divergences are not the count of all observed downstream token effects. "
        "These are numerical-state findings, not measured harmful-content outcomes."
    )
    counts = audit["counts"]
    warn = (f"The original-runtime audit retained {len(audit['warnings'])} advisory warning(s), fully listed in its JSON report. "
            if audit["warnings"] else "The original-runtime audit reported no advisory warnings. ")
    local_warn = ("The second audit on macOS retained an explicitly recorded cross-runtime advisory; "
                  "it does not replace the original-runtime result. " if local_audit["warnings"] else
                  "The second audit on macOS also reported no advisory warnings. ")
    blocks["AUDIT_RESULTS"] = (
        f"A separate audit program passed {counts['checks_passed']:,} evidence-consistency checks with "
        f"{counts['checks_failed']} failures in the pinned HPC CPU runtime. It checked all {counts['rows']} rows, "
        f"{counts['repair_timing_samples']:,} repair timing samples, {counts['overhead_samples']:,} normal-path timing samples, "
        f"and {counts['snapshot_files_checked']} selected complete-state snapshot files "
        f"({counts['snapshot_bytes']/1024**2:.2f} MiB). Each model's native-forward gate covered 142 positions. "
        "The auditor independently recomputed summary statistics and checked stored state tensors using weights-only CPU loading. "
        "Snapshots do not contain continuation tensors: those checks are recorded runner evidence, not independently rechecked snapshot tensors. "
        + warn + local_warn
    )
    lines = ["| Model / method | Full-replay speedup: median [IQR] | Normal overhead: median [IQR] |", "| --- | --- | --- |"]
    for m in MODELS:
        for method, label in (("sparse", "sparse"), ("all", "all-layer")):
            p = models[m]["performance"][method]
            lines.append(f"| {SHORT[m]} / {label} | {ratio_range(p['paired_speedup'])} | {pct_range(p['paired_normal_overhead_fraction'])} |")
    blocks["PERFORMANCE_TABLE"] = "\n".join(lines)
    base = "; ".join(f"{SHORT[m]}: {models[m]['full_recovery_ms']['median']:.2f} ms" for m in MODELS)
    blocks["PERFORMANCE_INTERPRETATION"] = (
        "Each entry is computed across 96 perturbed cases per model using paired per-case medians, not a ratio of pooled medians. "
        f"Median conservative full-replay times were {base}. "
        "IQR denotes the 25th-75th percentiles, not a confidence interval. "
        "Five normal-path rotations across three modes do not completely balance every order position; "
        "the small observed overhead differences are noise-sensitive, not overhead guarantees. "
        "The original comparison combines reduced decoder work with fewer checkpoint-prefix copies; "
        "it must not attribute all speedup to the layer-work identity. The next section tests a stronger restoration baseline. "
        "The all-layer method is not hidden: it can recover faster while paying substantially more activation storage."
    )
    lines = ["| Model | Shared full checkpoint | Extra journal payload: sparse / all |", "| --- | --- | --- |"]
    for m in MODELS:
        sm = models[m]
        lines.append(f"| {SHORT[m]} | {kib_range(sm['checkpoint_bytes'])} | {kib_range(sm['performance']['sparse']['journal_bytes'])} / {kib_range(sm['performance']['all']['journal_bytes'])} |")
    blocks["MEMORY_TABLE"] = "\n".join(lines)
    break_even = []
    for m in MODELS:
        p = models[m]["performance"]["sparse"]
        threshold = p["diagnostic_break_even_probability"]
        detail = (f"H/S = {100*threshold:.2f}% of windows" if threshold is not None else
                  "no interpretable nonnegative H/S threshold from these means")
        break_even.append(f"{SHORT[m]} sparse: mean H={p['mean_normal_cost_ms']:.3f} ms and S={p['mean_recovery_saving_ms']:.3f} ms; {detail}.")
    blocks["BREAK_EVEN"] = " ".join(break_even) + (
        " No observed deployment rate is available to show that either threshold would be met. "
        "These primary ratios use conservative copy-and-replay, not the stronger alias baseline from a separate run. "
        "The companion exploratory frontier analysis retains every case and reports layer/window/fault-shape/divergence strata; "
        "its nonnegative-cost sensitivity is not a statistical bound or a confirmed Pareto frontier."
    )
    signal = "met for both models" if all(x >= 1.2 for x in sparse) else "not met for both models"
    if all(x >= 1.2 for x in alias_ratios):
        stronger = "The sparse median recovery-only signal also survived checkpoint-alias full replay on both models."
    elif all(x > 1 for x in alias_ratios):
        stronger = "Sparse median recovery was faster than checkpoint-alias replay on both models, but the 1.2x screening signal did not survive on both."
    else:
        stronger = "The stronger checkpoint-alias baseline removed the sparse median recovery advantage on at least one model. This is adverse evidence for the proposed performance story."
    blocks["DECISION"] = (
        "The predeclared correctness gate passed on the supported frozen matrix. "
        f"The 1.2x recovery-only median screening signal against the original baseline was {signal}. "
        + stronger + " Neither result establishes the predeclared performance-thesis gate. "
        "Only the restricted snapshot-comparison cost was measured; broader end-to-end detection/commit economics, "
        "a production baseline, an acceptable holdback budget and a material advance over optimized prior-work adaptations remain unestablished. "
        "The late self-speculative-decoding comparison also leaves the novelty gate unmet; the general reuse-and-rollback rule is already known. "
        "The defensible status is a completed, auditable pilot worth a skeptical supervisor discussion, "
        "not a selected perfect thesis, a demonstrated high-impact contribution or an A/A*-ready submission."
    )
    blocks["SUPPLEMENTAL"] = supplemental_prose.read_text().strip()
    blocks["DETECTOR_RESULTS"] = detector_prose.read_text().strip()
    blocks["ARTIFACT"] = (
        f"Primary episodes SHA-256: {raw_hash[:16]}... (full digest in metadata and the artifact manifest). "
        f"The primary configuration begins {metadata['config_sha256'][:16]}..., and engine source begins "
        f"{metadata['engine_sha256'][:16]}.... Primary Slurm job 195057 and the separate audit/sensitivity jobs "
        "are recorded with timestamps and logs in notes/execution_log.md. "
        "The README provides immutable model-fetch, test, run, analysis and audit instructions. "
        "The first two PDFs preserve the prospective idea and plan; this document reports measured outcomes and limitations. "
        "One plan wording is resolved conservatively: a missing journal raises an error (fail closed), "
        "rather than implementing automatic fallback. Normal-path window computation was measured, "
        "but an end-user/private-commit latency SLO was not."
    )
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=PROJECT / "runs/primary-v1")
    parser.add_argument("--supplemental-prose", type=Path, required=True)
    parser.add_argument("--detector-prose", type=Path, required=True)
    parser.add_argument("--supplemental-run", type=Path, default=PROJECT / "runs/sensitivity-v1")
    args = parser.parse_args()
    template = PROJECT / "artifacts/RECUT_03_Professor_Draft.template.md"
    text = template.read_text()
    blocks = render_blocks(args.run, args.supplemental_run, args.supplemental_prose, args.detector_prose)
    required = set(re.findall(r"@@([A-Z_]+)@@", text))
    assert required == set(blocks), (required, set(blocks))
    for name, block in blocks.items():
        text = text.replace(f"@@{name}@@", block)
    assert "@@" not in text and text.isascii()
    output = PROJECT / "artifacts/RECUT_03_Professor_Draft_and_Results.md"
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
