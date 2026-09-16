"""Assemble the final manuscript after audited main results and interpretation.

The template and narrative are authored separately. Numerical tables come from
the independent raw recomputation, with input bindings checked again here.
This script does not create a PDF or mutate experimental evidence.
"""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from verify_fallback_diagnostic import verify as verify_selected_diagnostic

PROJECT = Path(__file__).resolve().parents[2]
STEM = "RECUT_04_Complete_Prototype_and_Validation"
MODELS = ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B")
LABELS = ("Smol-135M", "Qwen-0.5B", "Qwen-7B")
SLUGS = ("smol135m", "qwen05b", "qwen7b")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonstandard numeric value: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def main():
    tables_path = PROJECT / "stage3/reporting/report_tables.json"
    recomputed_path = PROJECT / "notes/stage3_main_recomputed.json"
    narrative_path = PROJECT / "stage3/reporting/interpretation.json"
    template_path = PROJECT / "artifacts" / (STEM + ".template.md")
    data, recomputed, narrative = read(tables_path), read(recomputed_path), read(narrative_path)
    require(data["generator_sha256"] == sha(PROJECT / "stage3/reporting/report_tables.py"), "Changed table generator")
    require(data["input_recomputation_sha256"] == sha(recomputed_path), "Changed recomputation")
    require(recomputed["script_sha256"] == sha(PROJECT / "notes/independent_stage3_report.py"), "Changed independent calculator")
    for relative, expected in data["input_sha256"].items():
        require(sha(PROJECT / relative) == expected, "Changed report input: " + relative)
    require(narrative["report_tables_sha256"] == sha(tables_path), "Narrative not bound to final numbers")
    diagnostic_dir = PROJECT / "runs/stage3-fallback-diagnostic-v1/smol135m"
    diagnostic_path = diagnostic_dir / "validation_only.json"
    diagnostic = read(diagnostic_path)
    require(diagnostic == verify_selected_diagnostic(
        PROJECT / "runs/stage3-main-v1/smol135m", diagnostic_dir,
        PROJECT / "stage3/reporting/fallback_config.json", PROJECT / "stage3/model_manifest.json"),
        "Selected-case verification is stale")
    diagnostic_log = PROJECT / "logs/stage3-fallback-195986.out"
    log_text = diagnostic_log.read_text()
    pre_run_records = [json.loads(line) for line in log_text.splitlines()
                       if line.startswith('{"status": "original_case_and_reduction_verified"')]
    require(len(pre_run_records) == 1 and pre_run_records[0]["prospective_diagnostic_binding"]
            == diagnostic["prospective_diagnostic_binding"], "Missing pre-run amendment binding in scheduler log")
    require(log_text.index('"original_case_and_reduction_verified"') < log_text.index('"session": 1,'),
            "Diagnostic binding was not logged before execution")
    freeze = read(PROJECT / "stage3/protocol_freeze.json")
    for relative, expected in freeze["files_sha256"].items():
        require(sha(PROJECT / relative) == expected, "Changed frozen source: " + relative)
    suites = ET.parse(PROJECT / "logs/stage3-local-final-tests.xml").getroot().findall(".//testsuite")
    require(suites and all(int(s.attrib.get("failures", 0)) == int(s.attrib.get("errors", 0)) == 0 for s in suites), "Local regression failures")
    tests = sum(int(s.attrib["tests"]) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    require(tests == 313 and skipped == 17, "Unexpected regression coverage")
    for task in range(3):
        log = PROJECT / "logs" / f"stage3-195926_{task}.out"
        require("286 passed" in log.read_text(), "Missing allocated GPU unit gate")

    runs = {run["model"]: run for run in recomputed["runs"]}
    require(len(runs) == len(recomputed["runs"]) == 3 and set(runs) == set(MODELS), "Wrong model coverage")
    for model, slug in zip(MODELS, SLUGS):
        metadata = read(PROJECT / "runs/stage3-main-v1" / slug / "metadata.json")
        expected = {"model": model, "python": "3.13.9", "torch": "2.6.0+cu124", "transformers": "4.51.3",
                    "device": "cuda", "dtype": "bfloat16", "gpu": "NVIDIA RTX 5000 Ada Generation",
                    "deterministic_algorithms": True, "tf32": False, "attention": "eager", "batch_size": 1}
        require(all(metadata.get(key) == value for key, value in expected.items()), "Runtime differs from manuscript: " + model)
    totals = data["totals"]
    blocks = {}
    blocks["EXECUTIVE"] = (
        "### Measured outcome\n\n" + narrative["executive"] + "\n\n" + table(["Main-study evidence", "Completed result"], [
            ["Models / workloads", "3 models, 18 model-workload combinations"],
            ["Measured sessions", f'{totals["measured_sessions"]:,}; warmups and profiles excluded'],
            ["Window outcomes", f'{totals["committed_windows"]:,} exact-checked commits; {totals["rejected_windows"]:,} specified safe refusals'],
            ["State archives", f'21 main archives; {data["snapshot_tensor_pairs"]:,} recovered/reference tensor pairs rechecked on CPU'],
        ]))
    blocks["CLEAN_COST"] = (
        "Times below are milliseconds for a complete three-window clean session, including its separately measured initialization. "
        "Rows identify the initial prefix: 128 uses W=8; 2,048 uses W=16. Later windows have longer prefixes. "
        "Each entry is the median of three prompt-level case medians; each case first summarizes three timing repetitions.\n\n"
        + data["tables"]["clean_ms"] + "\n\n"
        "Protected sparse cost relative to ordinary inference is shown separately. These are medians of three matched ratios, "
        "not ratios obtained by dividing the cross-prompt milliseconds displayed above.\n\n" + data["tables"]["clean_ratios"]
        + "\n\n" + narrative["clean_cost"])
    fault_rows = data["numeric_rows"]["fault_ratios"]
    fault_labels = ("Early K", "Early V", "Middle K", "Middle V", "Late K", "Late V", "Two-layer", "Every-window")
    fault_tables = []
    for prefix, offset in ((128, 0), (2048, 1)):
        values = [fault_rows[2 * model + offset][1:] for model in range(3)]
        require(all(len(row) == 8 for row in values), "Missing fault stratum")
        fault_tables.append(f"### Initial prefix {prefix:,}\n\n" + table(["Fault stratum", *LABELS],
            [[label] + [values[model][index] for model in range(3)] for index, label in enumerate(fault_labels)]))
    blocks["RECOVERY_COST"] = (
        "**Full / sparse complete affected-window time.** Above 1 favors sparse; below 1 favors full. "
        "Both use batched seals and the same protection contract. All eight prespecified fault strata are shown. "
        "Each ordinary stratum summarizes six matched case-window ratios (three prompts x two seeds); every-window faults summarize 18. "
        "Each ratio is formed after collapsing two timing repetitions. Guard challenges are excluded.\n\n"
        + "\n\n".join(fault_tables) + "\n\n" + narrative["recovery_cost"])
    divergence, debug_divergence, debug_partial = [], 0, 0
    for model, label, slug in zip(MODELS, LABELS, SLUGS):
        count = runs[model]["counts"]
        divergence.append([label, count["replay_windows_with_token_divergence"], count["partial_to_full_replay_windows"]])
        directory = PROJECT / "runs/stage3-debug-v1" / slug
        audit, metadata = read(directory / "audit_local.json"), read(directory / "metadata.json")
        require(audit["status"] == "passed" and audit["tensor_reload_requested"] and metadata["status"] == "complete", "Missing full debug re-audit")
        require(audit["hashes"]["metadata"] == sha(directory / "metadata.json"), "Stale debug audit")
        require(audit["hashes"]["auditor"] == sha(PROJECT / "stage3/audit_study.py"), "Changed debug auditor")
        for phase in ("sessions", "references", "warmups", "profiles"):
            actual = sha(directory / (phase + ".jsonl"))
            require(actual == metadata[phase + "_sha256"] == audit["hashes"][phase + ".jsonl"], "Changed debug stream: " + phase)
        rows = [read_line for read_line in map(json.loads, (directory / "sessions.jsonl").read_text().splitlines())]
        windows = [w for r in rows for w in r["windows"]]
        debug_divergence += sum(w.get("first_divergence") is not None for w in windows)
        debug_partial += sum(any(c > 0 for c in w.get("replay_starts", [])) and 0 in w.get("replay_starts", []) for w in windows)
    blocks["DIVERGENCE"] = (
        "Main-study replay windows with a changed corrected token / with subsequent partial-to-full replay: "
        + "; ".join(f"{row[0]} {row[1]} / {row[2]}" for row in divergence) + ". "
        + f"The separate tiny pretrained debug matrix had {debug_divergence} divergent replay windows and {debug_partial} partial-to-full windows. "
        "These are window executions including policies/repetitions, not independent faults. Forced-divergence regression tests also cover permanent fallback and later token reconvergence. "
        + narrative["divergence"])
    blocks["SESSION_COST"] = (
        "The following ratios include all three windows and session initialization. Each row summarizes 48 matched case-session ratios "
        "(three prompts x two seeds x eight fault strata), after two timing repetitions per case. This equal-case summary is not a deployment incidence mixture. "
        "All-cuts uses every nonzero layer cut; it is an internal denser-journal comparator, not a published-system reproduction.\n\n"
        + data["tables"]["session_ratios"] + "\n\n" + narrative["allcuts"])
    blocks["BREAK_EVEN"] = (
        data["tables"]["break_even"] + "\n\n"
        "Defined/all counts positive-H, positive-R thresholds out of all 60 affected case-windows per model/shape. "
        "The every-window stratum contributes three windows per case; the other seven strata contribute one each. "
        "Median p* uses only that defined subset. 'No extra H' counts nonpositive H with positive R; 'Other signs' retains all remaining cases. "
        + narrative["break_even"])
    blocks["SEALING"] = (
        "**Legacy / batched initialization-inclusive session time.** Above 1 favors batching. Clean columns summarize three matched prompt cases; "
        "late-V columns summarize six prompt/seed cases.\n\n" + data["tables"]["batch_ratios"] + "\n\n" + narrative["sealing"])
    blocks["MEMORY"] = (
        "### GPU allocation and journal payload\n\n"
        "Peak GiB is the largest window allocation recorded across all measured variants/scenarios at that shape, not a whole-job startup/prefill peak. "
        "Journal KiB is a clean case-window payload median, not an allocator peak.\n\n" + data["tables"]["memory"]
        + "\n\nThe next table shows the median of nine clean case-window cells after reducing timing repetitions. "
        "GPU increase is the window peak minus its starting allocated memory; it includes temporary copies/staging and harness effects, "
        "not just the journal or a production-only protection footprint. Checkpoint is logical sparse-policy payload. Host peak is unmeasured.\n\n"
        + data["tables"]["memory_increment"])
    blocks["AVAILABILITY"] = (
        "### First-token availability within clean windows\n\n"
        "Milliseconds below are medians across nine prompt/window case cells after within-case repetition reduction. "
        "Protected sparse releases its window at commitment. Native/bare step-completion times are used as availability proxies; the harness still buffers those tokens locally. "
        "The final column is the median of each window's mean hold-after-step duration, not a pooled-token average or network latency.\n\n"
        + data["tables"]["availability"])
    comparisons = sum(run["analyzer_comparison"]["comparisons"] for run in runs.values())
    blocks["VALIDATION"] = (
        f"All {totals['windows_attempted']:,} main attempted windows are accounted for: {totals['committed_windows']:,} committed with recorded exact cache/logit/token and continuation agreement; "
        f"{totals['rejected_windows']} checkpoint challenges refused with the specified integrity reason and no additional output. "
        f"There were {totals['recovered_windows']:,} recovered window executions and {totals['native_custom_parity_positions']} native/custom post-prefill parity positions. "
        f"The {totals['warmup_sessions_excluded']} warmup and {totals['profiled_sessions_excluded']} profile sessions are excluded from those measured totals.\n\n"
        "Commit totals include the clean native/bare reference baselines. Recovery counts include journal-integrity challenges; the paired performance tables exclude both guard strata.\n\n"
        + data["tables"]["validation"] + "\n\n"
        f"Local audits passed {data['local_audit_checks']:,} checks in total. The HPC audits check the same evidence; they are not counted as additional fault observations. "
        f"The 21 main archives contain {data['snapshot_tensor_pairs']:,} actual/reference tensor pairs, all numerically equal; "
        f"{data['snapshot_bytewise_equal_pairs']:,} also passed the stricter raw-byte comparison. "
        f"Independent raw recomputation agreed on {comparisons:,} compared report quantities.\n\n"
        "The core/controller suite passed all 286 tests on each allocated GPU. The final local combined suite passed 296 tests and skipped 17 CUDA-only tests, "
        "including 27 separate offline reporting tests. Repeating a test suite on several GPUs does not multiply its number of distinct test cases.")
    blocks["REPORTING_CORRECTION"] = (
        "### Preserved reporting correction\n\n"
        "The frozen analyzer originally displayed zero seal counters for refused windows: those counters are nested under rejection metadata. "
        "A separate reporting-only tool corrected those counter medians and grouped counter displays. Original summary.json and every raw record remain unchanged. "
        "Use summary_corrected.json with analysis_corrections.json, which lists the changed fields and binds all relevant hashes. "
        "No measured time, correctness outcome, performance ratio, source freeze or experimental integrity gate was changed.")
    blocks["SELECTED_DIAGNOSTIC"] = (
        "### Separate post-hoc fallback validation\n\n"
        "Smol's code/prefix-128/W8 multi-prefix case, seed 7431, repetition 0 exercised natural partial-to-full fallback but was outside the main predeclared snapshot subset. "
        "After observing that gap, a written amendment fixed one validation-only replay of the same workload, fault, model revision and frozen source. "
        "Its 13 sessions produced 33 commits and three expected refusals. These outcomes are excluded from all primary totals and timings.\n\n"
        "The replay reproduced the original applied faults, speculative/corrected token traces and native-reference fingerprints. "
        "Sparse started at cut 7 for seven steps, then at layer 0; all-cuts used cut 14 for seven steps, then layer 0. "
        "The recorded first_divergence=7 is one-based: the seventh predicted token differs, and only the eighth step starts at layer zero. "
        "Full replay used layer 0 throughout. Three separately saved policy archives passed HPC and local CPU audits, totaling 366 recovered/reference tensor pairs, including two-step continuation. "
        "This closes the selected case's state-archive gap; it is not another independent fault sample, an additional primary performance result or new novelty evidence.")
    blocks["PUBLICATION"] = narrative["publication"]
    manifest = read(PROJECT / "stage3/model_manifest.json")
    blocks["REPRODUCTION"] = (
        "The measured environment was Python 3.13.9, PyTorch 2.6.0+cu124, Transformers 4.51.3, BF16, deterministic algorithms, TF32 disabled, "
        "and NVIDIA RTX 5000 Ada 32 GB GPUs. Slurm allocated one GPU, four CPUs and 48 GB host memory per task. "
        "The account allowed two simultaneous jobs; the third model waited normally. The main matrix was gated on all three debug tasks succeeding.\n\n"
        + table(["Model", "Immutable revision"], [[r["model"], "`" + r["revision"] + "`"] for r in manifest]) + "\n\n"
        "Start with research-next/stage3/DELIVERY.md. It gives CPU-only evidence-audit commands, the combined tests, localized reproduction rules and PDF rebuilding. "
        "The portable ZIP includes the main/debug raw records and tensor archives, frozen dependencies, corrected summaries, reporting tools and reviews. "
        "It deliberately excludes model weights, environments and authentication material.\n\n"
        "Source freeze: stage3/protocol_freeze.json, written before pretrained measurements. Main data: runs/stage3-main-v1/{smol135m,qwen05b,qwen7b}. "
        "Audits: audit.json (HPC) and audit_local.json (local) per model. Numerical source: notes/stage3_main_recomputed.json. "
        "Main scheduler array: 195927; debug array: 195926. The failed initial model-fetch setup log is preserved; its successful CPU-only retry preceded measurement.")
    blocks["PROFESSOR_SUMMARY"] = narrative["professor_summary"]
    template = template_path.read_text()
    markers = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))
    require(markers == set(blocks), "Missing or surplus manuscript blocks")
    text = template
    for marker, content in blocks.items():
        require(isinstance(content, str) and content.strip(), "Empty block: " + marker)
        text = text.replace("{{" + marker + "}}", content)
    require(text.isascii() and "{{" not in text and "}}" not in text, "Unresolved marker/non-ASCII typography")
    output = PROJECT / "artifacts" / (STEM + ".md")
    provenance_path = PROJECT / "stage3/reporting/manuscript_provenance.json"
    require(not any(path.exists() for path in (output, PROJECT / "stage3/RESULTS.md", provenance_path)), "Refusing completed manuscript overwrite")
    with output.open("x") as stream:
        stream.write(text)
    with (PROJECT / "stage3/RESULTS.md").open("x") as stream:
        stream.write(text.replace("<!-- PAGE -->", ""))
    provenance = {"manuscript_sha256": sha(output), "template_sha256": sha(template_path), "tables_sha256": sha(tables_path),
                  "interpretation_sha256": sha(narrative_path), "independent_recomputation_sha256": sha(recomputed_path),
                  "selected_diagnostic_sha256": sha(diagnostic_path),
                  "selected_diagnostic_amendment_sha256": sha(PROJECT / "notes/stage3_fallback_diagnostic_amendment.md"),
                  "selected_diagnostic_pre_run_log_sha256": sha(diagnostic_log),
                  "composer_sha256": sha(__file__), "blocks": sorted(blocks), "independent_numeric_comparisons": comparisons}
    with provenance_path.open("x") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), **provenance}, indent=2))


if __name__ == "__main__":
    main()
