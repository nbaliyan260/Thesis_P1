"""Validate a declared post-hoc replay of one observed natural-fallback case.

This tool compares records and verified snapshot-audit bindings; the frozen
auditor performs the independent CPU tensor reload. It never adds this selected
replay to main counts, timing distributions, novelty evidence or fault samples.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
MODEL = "HuggingFaceTB/SmolLM2-135M"
WORKLOAD = "HuggingFaceTB--SmolLM2-135M--code--p128--w8"
VARIANTS = ("full_batched", "sparse_batched", "allcuts_batched")
ORIGINAL_FILES = {
    "metadata.json": "8a1695ba05c046af03fb7e1a6a7910c53b53efc594a94ab9cfe596f3220633e4",
    "sessions.jsonl": "fe8b1c0c4d785590c66d8582bfff6b8bc23f1115e35255ab55923d1fc41fd403",
    "references.jsonl": "0f1f36d1d0ea802d49e0a32e46d6d11362e56f3818018edc2cd8086fa5a2d7f5",
}
SOURCE_CANONICAL = "b0726201ade8978a20daeb7edaaec192f5eadea864ede656e4ad40c83bfdb51d"
REFERENCE_CANONICAL = "4940d983e78f347cf57ac5133d4d07db61fafac6d647816da6a0a0d427bea149"
FAULTS_CANONICAL = "1f95e81b06757c50fedc9f2f6fdc23ee320a547fff52fceb0f955766ffde4e35"
CASE_TRACES = {
    "full_batched": "0eb0f62ed2c415964bcd43ed6ad1f9343f9928fa9c1075442a9624bfca2bfe4e",
    "sparse_batched": "4b2733420fc00bfc2b6da799b37430d27029cc30d41d27117a91770a92957c6f",
    "allcuts_batched": "f9a82a731e53bab958581e0c55896be031048e74ebd273109204e37f55592942",
}
TRACE_FIELDS = ("window_index", "status", "faults", "speculative_tokens", "tokens", "detected_layers",
                "selected_cut", "first_divergence", "replay_starts", "fallback_reason", "recovered")
EXPECTED_STARTS = {"full_batched": [0] * 8, "sparse_batched": [7] * 7 + [0], "allcuts_batched": [14] * 7 + [0]}


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
    def invalid(value):
        raise ValueError("Nonstandard JSON value: " + value)
    return json.loads(text, parse_constant=invalid)


def read(path):
    return parse(Path(path).read_text())


def rows(path):
    return [parse(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected(records):
    matched = [row for row in records if row["workload_id"] == WORKLOAD and row["scenario"] == "multi_prefix"
               and row["seed"] == 7431 and row["repetition"] == 0 and row["variant"] in VARIANTS]
    result = {row["variant"]: row for row in matched}
    require(len(matched) == len(result) == 3 and set(result) == set(VARIANTS), "Selected three-policy case coverage")
    return result


def trace(row):
    return [{key: win[key] for key in TRACE_FIELDS} for win in row["windows"]]


def check_case(row, expected_source):
    require(row["model"] == MODEL and row["prompt_id"] == "code" and row["prefix_tokens"] == 128
            and row["window"] == 8 and row["n_windows"] == 3 and row["num_layers"] == 30
            and row["source_sha256"] == expected_source and row["profile"] is False
            and row["status"] == "complete" and row["valid"] is True, "Selected case identity/source/status")
    require(canonical(trace(row)) == CASE_TRACES[row["variant"]], "Changed canonical original-case trace")
    win = row["windows"][1]
    require(win["first_divergence"] == 7 and win["detected_layers"] == [14, 22]
            and win["replay_starts"] == EXPECTED_STARTS[row["variant"]], "Missing declared natural fallback trace")
    require(canonical(win["faults"]) == FAULTS_CANONICAL, "Changed applied fault vectors/coordinates")
    require(all(w["valid"] is True and w["no_early_commit"] is True and w["checks"]["all_equal"] is True
                for w in row["windows"]), "Selected window correctness/no-early-commit failure")


def verify_original(directory, config_path, manifest_path):
    verify_amendment()
    for name, expected in ORIGINAL_FILES.items():
        require(sha(directory / name) == expected, "Original main evidence changed: " + name)
    meta = read(directory / "metadata.json")
    require(meta["status"] == "complete" and meta["model"] == MODEL, "Original main run incomplete")
    require(canonical(meta["source_sha256"]) == SOURCE_CANONICAL, "Original source map changed")
    for relative, expected in meta["source_sha256"].items():
        require(sha(PROJECT / "stage3" / relative) == expected, "Frozen source changed: " + relative)
    require(sha(manifest_path) == meta["manifest_sha256"], "Original manifest changed")
    cases = selected(rows(directory / "sessions.jsonl"))
    for row in cases.values():
        check_case(row, meta["source_sha256"])
    refs = [ref for ref in rows(directory / "references.jsonl") if ref["workload_id"] == WORKLOAD]
    require(len(refs) == 1 and canonical(refs[0]) == REFERENCE_CANONICAL, "Original native reference changed")
    expected = deepcopy(meta["config"])
    expected.update(study="RECUT stage3 post-hoc natural-fallback validation only", models=[MODEL],
        prompts=[p for p in expected["prompts"] if p["id"] == "code"], shapes=[{"prefix_tokens": 128, "window": 8}],
        n_windows=3, seeds=[7431], repetitions=1, clean_repetitions=1, fault_cases=["multi_prefix"],
        guard_cases=["checkpoint_corruption"], legacy_cases=["clean"], snapshot_cases=["multi_prefix"],
        snapshot_prompt="code", snapshot_short_prefix=128, snapshot_long_case="multi_prefix", profile=False, warmup=False)
    require(read(config_path) == expected, "Diagnostic configuration deviates from declared reduction")
    return meta, cases, refs[0]


def verify_amendment():
    path = PROJECT / "stage3/reporting/fallback_protocol_freeze.json"
    frozen = read(path)
    expected = {"notes/stage3_fallback_diagnostic_amendment.md",
                "stage3/reporting/fallback_config.json",
                "stage3/reporting/run_fallback_diagnostic.sbatch",
                "stage3/reporting/verify_fallback_diagnostic.py"}
    require(set(frozen["files_sha256"]) == expected, "Incomplete selected diagnostic freeze")
    require(frozen["scope"] == "post_hoc_validation_only_before_diagnostic_submission", "Wrong amendment scope")
    for relative, expected_sha in frozen["files_sha256"].items():
        require(sha(PROJECT / relative) == expected_sha, "Changed declared diagnostic input: " + relative)
    return {"freeze_sha256": sha(path),
            "amendment_sha256": sha(PROJECT / "notes/stage3_fallback_diagnostic_amendment.md")}


def verify(original, diagnostic, config_path, manifest_path):
    original_meta, original_cases, reference = verify_original(original, config_path, manifest_path)
    meta = read(diagnostic / "metadata.json")
    require(meta["status"] == "complete" and meta["model"] == MODEL and meta["schema_version"] == 1, "Diagnostic incomplete")
    require(meta["source_sha256"] == original_meta["source_sha256"] and meta["model_revision"] == original_meta["model_revision"],
            "Diagnostic source/model differs")
    require(meta["config"] == read(config_path) and meta["config_sha256"] == sha(config_path)
            and meta["manifest_sha256"] == sha(manifest_path) and meta["manifest"] == read(manifest_path), "Diagnostic input binding")
    for field in ("dtype", "device", "transformers", "torch", "python", "attention", "deterministic_algorithms", "tf32", "batch_size"):
        require(meta[field] == original_meta[field], "Changed controlled runtime: " + field)
    datasets = {}
    hashes = {"metadata": sha(diagnostic / "metadata.json")}
    for name in ("sessions", "references", "warmups", "profiles"):
        hashes[name + ".jsonl"] = sha(diagnostic / (name + ".jsonl"))
        require(hashes[name + ".jsonl"] == meta[name + "_sha256"], "Diagnostic raw binding: " + name)
        datasets[name] = rows(diagnostic / (name + ".jsonl"))
    require(meta["expected_sessions"] == meta["sessions_completed"] == len(datasets["sessions"]) == 13
            and meta["warmups_completed"] == meta["profiles_completed"] == 0
            and datasets["warmups"] == datasets["profiles"] == [], "Diagnostic matrix/exclusion counts")
    windows = [w for row in datasets["sessions"] for w in row["windows"]]
    require(len(windows) == 36 and sum(w["status"] == "committed" for w in windows) == 33
            and sum(w["status"] == "rejected" for w in windows) == 3, "Diagnostic outcome counts")
    require(len(datasets["references"]) == 1 and canonical(datasets["references"][0]) == REFERENCE_CANONICAL
            and datasets["references"][0] == reference and len(reference["parity"]) == 26, "Changed native KV/logit/token reference")
    cases = selected(datasets["sessions"])
    snapshots = {}
    for variant, row in cases.items():
        check_case(row, meta["source_sha256"])
        require(trace(row) == trace(original_cases[variant]), "Rerun trace differs from original rep0")
        require(row["model_revision"] == original_cases[variant]["model_revision"], "Case revision mismatch")
        snapshot = row["windows"][1]["snapshot"]
        name = snapshot["file"]
        require(name.startswith("snapshots/") and ".." not in Path(name).parts and not Path(name).is_absolute(), "Snapshot path")
        path = diagnostic / name
        require(sha(path) == snapshot["sha256"] and path.stat().st_size == snapshot["bytes"], "Snapshot changed")
        snapshots[path.name] = snapshot
    require(len(snapshots) == 3 and sum("snapshot" in w for w in windows) == 3, "Snapshot selection/count")
    require({p.name for p in (diagnostic / "snapshots").glob("*.pt")} == set(snapshots), "Unexpected snapshot files")
    audits = {}
    for name in ("audit.json", "audit_local.json"):
        audit = read(diagnostic / name)
        require(audit["status"] == "passed" and audit["model"] == MODEL and audit["tensor_reload_requested"] is True,
                "Required full independent audit missing: " + name)
        require(all(audit["hashes"][key] == value for key, value in hashes.items()), "Stale audit raw binding: " + name)
        require(audit["hashes"]["config"] == sha(config_path) and audit["hashes"]["manifest"] == sha(manifest_path)
                and audit["hashes"]["auditor"] == sha(PROJECT / "stage3/audit_study.py"), "Audit input/source binding: " + name)
        counts = audit["counts"]
        require(counts["snapshot_archives_declared"] == counts["snapshot_archives_reloaded"] == 3
                and counts["native_parity_positions"] == 26 and counts["sessions"]["sessions"] == 13
                and counts["sessions"]["windows_committed"] == 33 and counts["sessions"]["windows_rejected"] == 3,
                "Audit coverage mismatch: " + name)
        require(len(audit["snapshot_results"]) == 3 and {r["file"] for r in audit["snapshot_results"]} == set(snapshots),
                "Audit archive coverage: " + name)
        for result in audit["snapshot_results"]:
            declared = snapshots[result["file"]]
            identity = result["identity"]
            require(result["status"] == "passed" and result["sha256"] == declared["sha256"]
                    and result["bytes"] == declared["bytes"] and result["tensor_pairs"] == 122, "Audit snapshot result: " + name)
            require(identity["workload_id"] == WORKLOAD and identity["model"] == MODEL and identity["variant"] in VARIANTS
                    and identity["scenario"] == "multi_prefix" and identity["seed"] == 7431
                    and identity["repetition"] == 0 and identity["window_index"] == 1, "Audit snapshot identity: " + name)
        audits[name] = {"sha256": sha(diagnostic / name), "checks": audit["checks"]}
    return {"status": "passed_validation_only", "post_hoc_selected": True,
        "prospective_diagnostic_binding": verify_amendment(),
        "excluded_from_main_performance_counts_novelty_and_iid_claims": True,
        "original_main_file_sha256": ORIGINAL_FILES, "original_source_canonical_sha256": SOURCE_CANONICAL,
        "native_reference_canonical_sha256": REFERENCE_CANONICAL, "faults_canonical_sha256": FAULTS_CANONICAL,
        "case_trace_sha256": CASE_TRACES, "diagnostic_input_sha256": hashes,
        "config_sha256": sha(config_path), "manifest_sha256": sha(manifest_path),
        "verifier_sha256": sha(__file__), "audit_bindings": audits,
        "counts": {"sessions": 13, "windows_attempted": 36, "committed_windows": 33,
                   "specified_refusals": 3, "snapshots_reloaded": 3, "native_parity_positions": 26,
                   "snapshot_tensor_pairs": 366},
        "observed_replay_starts": {variant: cases[variant]["windows"][1]["replay_starts"] for variant in VARIANTS},
        "scope": "Validation-only replay selected after observing the original main case. No independent fault sample, additional main performance evidence, new mechanism, or novelty evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=PROJECT / "runs/stage3-main-v1/smol135m")
    parser.add_argument("--diagnostic", type=Path, default=PROJECT / "runs/stage3-fallback-diagnostic-v1/smol135m")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("fallback_config.json"))
    parser.add_argument("--manifest", type=Path, default=PROJECT / "stage3/model_manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-original-only", action="store_true")
    args = parser.parse_args()
    if args.check_original_only:
        verify_original(args.original, args.config, args.manifest)
        print(json.dumps({"status": "original_case_and_reduction_verified", "original_sessions_sha256": ORIGINAL_FILES["sessions.jsonl"],
                          "config_sha256": sha(args.config), "verifier_sha256": sha(__file__),
                          "prospective_diagnostic_binding": verify_amendment()}))
        return
    output = args.output or args.diagnostic / "validation_only.json"
    require(not output.exists(), "Refusing validation report overwrite")
    result = verify(args.original, args.diagnostic, args.config, args.manifest)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
