"""Independent source/record audit and descriptive supplemental timing summary.

Does not import either experiment runner, its engine, or another analyzer.
Supplemental tensors are not saved: correctness remains recorded-runner evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

METHODS = ("full", "layer_view_full", "checkpoint_alias_full", "sparse", "all")
BASELINES = METHODS[:3]
VERSION = "RECUT-supplement-audit-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def at(p):
        index = (len(values) - 1) * p
        lo, hi = int(index), min(int(index) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(n=len(values), min=values[0], q25=at(.25), median=at(.5),
                q75=at(.75), max=values[-1], mean=statistics.mean(values))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.counts = {}

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
            raise ValueError(label)

    def equality(self, value, label, cache=False):
        self.check(value["equal"] is True and value["finite"] is True
                   and value["max_abs_error"] == 0, label)
        if cache:
            self.check(value["eq"] is True and value["mismatches"] == [], label + "/cache-detail")


def validate(audit, run, primary_dir, config_path, manifest_path):
    meta, primary = read_json(run / "metadata.json"), read_json(primary_dir / "metadata.json")
    config, manifest = read_json(config_path), read_json(manifest_path)
    raw_path, primary_raw_path = run / "episodes.jsonl", primary_dir / "episodes.jsonl"
    hashes = dict(episodes=digest(raw_path), metadata=digest(run / "metadata.json"),
                  detection=digest(run / "detection.jsonl"),
                  primary_episodes=digest(primary_raw_path),
                  primary_metadata=digest(primary_dir / "metadata.json"),
                  config=digest(config_path), manifest=digest(manifest_path),
                  auditor=digest(Path(__file__)))
    audit.check(meta["status"] == primary["status"] == "complete", "completion")
    audit.check(meta["study"] == "posthoc_baseline_sensitivity"
                and meta["not_preregistered"] is True, "study-label")
    audit.check(meta["episodes_sha256"] == hashes["episodes"], "supplement-raw-hash")
    audit.check(meta["detection_sha256"] == hashes["detection"], "detector-raw-hash")
    audit.check(meta["primary_raw_sha256"] == primary["episodes_sha256"]
                == hashes["primary_episodes"], "primary-raw-binding")
    audit.check(meta["primary_metadata_sha256"] == hashes["primary_metadata"], "primary-metadata-binding")
    for key in ("config", "manifest"):
        audit.check(meta[key + "_sha256"] == primary[key + "_sha256"] == hashes[key], key + "-binding")
    audit.check(primary["config"] == config, "primary-exact-config")
    audit.check(meta["models"] == primary["models"] == manifest, "exact-model-manifest")
    audit.check(meta["methods"] == list(METHODS) and meta["repetitions"] == 5
                and meta["warmups"] == 1, "five-policies-repetitions-warmup")
    audit.check(meta["expected_cases"] == meta["cases_completed"] == 192, "declared-192-coverage")
    audit.check(meta["expected_detection_cases"] == meta["detection_cases_completed"] == 216,
                "declared-216-detection-coverage")
    for key in ("python", "torch", "transformers", "cuda", "gpu", "equality", "timing_boundary",
                "memory_scope", "checkpoint_assumption", "sparse_all_scope", "detector_scope",
                "detector_timing", "no_alarm_policy"):
        audit.check(bool(meta[key]), "metadata/" + key)
    audit.check(meta["transformers"] == "4.51.3", "pinned-transformers")
    for key in ("torch", "transformers", "cuda", "gpu", "deterministic_algorithms", "tf32",
                "attention", "dtype", "allow_bf16_reduced_precision_reduction",
                "float32_matmul_precision", "cublas_workspace_config"):
        audit.check(meta[key] == primary[key], "primary-numeric-environment/" + key)
    source_dir = Path(__file__).resolve().parent
    expected_sources = ("run_optimized_replay.py", "prefix_detector.py", "recut_engine.py", "run_recut.py")
    audit.check(set(meta["source_sha256"]) == set(expected_sources), "source-coverage")
    for name in expected_sources:
        current = digest(source_dir / name)
        hashes[name] = current
        audit.check(meta["source_sha256"][name] == current, "current-source/" + name)
    audit.check(hashes["recut_engine.py"] == primary["engine_sha256"]
                and hashes["run_recut.py"] == primary["runner_sha256"], "primary-source-binding")
    models = {m["model"]: m for m in manifest}
    cases = {c["id"]: c for c in config["cases"]}
    audit.check(len(models) == len(manifest) == 2 and len(cases) == len(config["cases"]) == 108,
                "frozen-two-model-108-config")
    for model, record in models.items():
        audit.check(re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is not None,
                    model + "/immutable-revision")
        audit.check(Path(record["local_path"]).name == record["revision"], model + "/revision-path")
    primary_rows = [json.loads(line) for line in primary_raw_path.read_bytes().splitlines()]
    prior = {(r["model"], r["case_id"]): r for r in primary_rows}
    expected_primary = {(model, case_id) for model in models for case_id in cases}
    audit.check(len(primary_rows) == len(prior) == 216 and set(prior) == expected_primary,
                "complete-primary-216-coverage")
    for key, row in prior.items():
        audit.check(row["case"] == cases[key[1]] and row["revision"] == models[key[0]]["revision"]
                    and row["config_sha256"] == hashes["config"]
                    and row["valid_repairs_equal"] is True, "primary-case/" + str(key))
    expected = {key for key, row in prior.items() if row["case"]["fault"] != "noop"}
    rows = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
    keys = [(r["model"], r["case_id"]) for r in rows]
    audit.check(len(rows) == len(set(keys)) == len(expected) == 192 and set(keys) == expected,
                "exact-192-perturbed-case-coverage")
    recovery_by_key = {(r["model"], r["case_id"]): r for r in rows}
    detections = [json.loads(line) for line in (run / "detection.jsonl").read_bytes().splitlines()]
    detection_keys = [(r["model"], r["case_id"]) for r in detections]
    audit.check(len(detections) == len(set(detection_keys)) == 216
                and set(detection_keys) == expected_primary, "exact-216-detection-case-coverage")
    audit.check(set(meta["native_parity"]) == set(models), "native-parity-model-coverage")
    for model in models:
        parity = meta["native_parity"][model]
        expected_prefix = max(r["case"]["prefix_tokens"] for key, r in prior.items() if key[0] == model)
        expected_decode = max(r["case"]["window"] for key, r in prior.items() if key[0] == model) + 2
        audit.check(parity["passed"] is True and parity["prefix_tokens"] == expected_prefix
                    and parity["decode_steps"] == expected_decode
                    and expected_prefix + expected_decode == 142, model + "/parity-length")
        audit.check([t["position"] for t in parity["trace"]] == list(range(142)), model + "/parity-positions")
        for position in parity["trace"]:
            audit.equality(position["cache"], model + "/parity-cache", cache=True)
            audit.equality(position["logits"], model + "/parity-logits")
    for detector_row in detections:
        key = (detector_row["model"], detector_row["case_id"])
        row = recovery_by_key.get(key, detector_row)
        label = "/".join(key)
        ref, case = prior[key], cases[row["case_id"]]
        if key in recovery_by_key:
            audit.check({k: v for k, v in row.items() if k != "timings"} == detector_row,
                        label + "/recovery-detector-record-identity")
        audit.check("timings" not in detector_row, label + "/separate-detector-record")
        audit.check(row["case"] == case and row["revision"] == models[row["model"]]["revision"]
                    and row["primary_raw_sha256"] == hashes["primary_episodes"], label + "/binding")
        audit.check(row["valid"] is True and row["reconstruction_matches_primary"] is True
                    and "failure" not in row, label + "/valid-reconstruction")
        for field in ("fault", "reference_tokens", "faulty_tokens", "num_layers", "layer",
                      "prefix_ids", "initial_token"):
            audit.check(row[field] == ref[field], label + "/primary-" + field)
        n, w, layer = row["num_layers"], case["window"], row["layer"]
        changed = ref["fault"]["changed_elements"] > 0
        detected_layers = [layer] if changed else []
        localized = layer if changed else None
        sparse_cut = max([0] + [c for c in (n // 4, n // 2, 3 * n // 4)
                               if 0 < c < n and localized is not None and c <= localized])
        audit.check(layer == int(case["layer_fraction"] * (n - 1))
                    and row["selected_sparse_cut"] == sparse_cut <= layer, label + "/layer-cut")
        audit.check(len(row["reference_tokens"]) == len(row["faulty_tokens"]) == w, label + "/token-length")
        divergence = next((i + 1 for i, (x, y) in enumerate(zip(row["reference_tokens"], row["faulty_tokens"]))
                           if x != y), None)
        audit.check(divergence == ref["first_divergence"], label + "/actual-divergence")
        fault = row["fault"]
        model_config = primary["model_configs"][row["model"]]
        head_dim = model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]
        audit.check(fault["kind"] == case["kind"] and fault["fault"] == case["fault"]
                    and fault["layer"] == layer and fault["batch"] == 0
                    and 0 <= fault["head"] < model_config["num_key_value_heads"]
                    and 0 <= fault["prefix_position"] < case["prefix_tokens"]
                    and 0 <= fault["scalar_dimension"] < head_dim, label + "/fault-coordinates")
        for field in ("before", "requested_delta", "after"):
            audit.check(len(fault[field]) == head_dim and all(math.isfinite(x) for x in fault[field]),
                        label + "/finite-fault-" + field)
        audit.check(fault["changed_elements"] == sum(x != y for x, y in zip(fault["before"], fault["after"]))
                    and math.isfinite(fault["rms"]) and fault["rms"] >= 0, label + "/fault-change-count")
        detector = row["detection"]
        audit.check(row["detection_only"] is (case["fault"] == "noop"), label + "/detection-only")
        audit.check(row["detected_layer"] == localized and detector["detected_layers"] == detected_layers,
                    label + "/changed-prefix-localization")
        audit.check(detector["route"] == ("unique_layer" if changed else "no_alarm_full"),
                    label + "/routing")
        audit.check(detector["checkpoint_unchanged"] is True
                    and detector["initial_faulty_unchanged"] is True, label + "/detector-immutability")
        audit.check(detector["checkpoint_payload_bytes"] == ref["checkpoint_payload_bytes"]
                    and detector["logical_bytes_read"] == 2 * ref["checkpoint_payload_bytes"]
                    and detector["prefix_tokens"] == case["prefix_tokens"]
                    and bool(detector["semantics"]), label + "/detector-payload-scope")
        audit.check([t["repetition"] for t in detector["timings"]] == list(range(5)),
                    label + "/five-detector-repetitions-no-warmup")
        for trial in detector["timings"]:
            audit.check(trial["detected_layers"] == detected_layers
                        and math.isfinite(trial["seconds"]) and trial["seconds"] > 0,
                        label + "/detector-stable-finite-trial")
        if row["detection_only"]:
            audit.check(not changed and detector["detected_layers"] == [], label + "/noop-no-alarm")
            continue
        expected_order = [(method, rep) for rep in range(5)
                          for method in METHODS[rep:] + METHODS[:rep]]
        audit.check([(t["method"], t["repetition"]) for t in row["timings"]] == expected_order,
                    label + "/25-rotated-trials-no-warmup")
        for trial in row["timings"]:
            method, rep, checks = trial["method"], trial["repetition"], trial["checks"]
            trial_label = label + "/" + method + "/" + str(rep)
            audit.check(trial["order"] == list(METHODS[rep:] + METHODS[:rep]), trial_label + "/order")
            audit.check(math.isfinite(trial["seconds"]) and trial["seconds"] > 0, trial_label + "/timing")
            for flag in ("all_equal", "tokens_equal", "continuation_tokens_equal",
                         "checkpoint_unchanged", "initial_faulty_unchanged"):
                audit.check(checks[flag] is True, trial_label + "/" + flag)
            for field in ("cache", "continuation_cache", "logits", "continuation_logits"):
                audit.equality(checks[field], trial_label + "/" + field, cache="cache" in field)
            audit.check(checks["tokens"] == row["reference_tokens"]
                        and checks["continuation_tokens"] == ref["methods"]["full"]["continuation_tokens"]
                        and len(checks["continuation_tokens"]) == 2, trial_label + "/exact-tokens")
            cut = sparse_cut if method == "sparse" else (localized or 0) if method == "all" else 0
            expected_starts = [cut if i < (divergence or w) else 0 for i in range(w)]
            audit.check(checks["first_divergence"] == divergence and checks["start_layers"] == expected_starts,
                        trial_label + "/irreversible-start-trace")
            audit.check(checks["layer_steps"] == n * w - cut * min(divergence or w, w)
                        == sum(n - c for c in checks["start_layers"]), trial_label + "/work-formula")
            for field in ("allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes"):
                audit.check(isinstance(trial[field], int) and trial[field] >= 0, trial_label + "/" + field)
    audit.counts = dict(rows=len(rows), models=len(models), per_model=dict(Counter(r["model"] for r in rows)),
                       perturbed_cases=192, noop_cases=0, trials=192 * 5 * 5,
                       methods=5, repetitions_per_method=5, recorded_warmups=0,
                       declared_excluded_warmups_per_method=1,
                       native_parity_positions_per_model=142, independently_loaded_tensor_snapshots=0,
                       detection_cases=216, detection_perturbed_cases=192, detection_noops=24,
                       detection_timing_samples=1080)
    return meta, rows, detections, hashes


def summarize(rows):
    medians = [{"case_id": row["case_id"], "model": row["model"],
                "seconds": {method: statistics.median(t["seconds"] for t in row["timings"]
                                                    if t["method"] == method) for method in METHODS}}
               for row in rows]
    return dict(n=len(rows), case_ids=[r["case_id"] for r in rows],
        correctness={m: dict(cases_all_repetitions_equal_and_immutable=len(rows),
                             verified_trials=len(rows) * 5) for m in METHODS},
        timing_ms={m: quantiles(p["seconds"][m] * 1000 for p in medians) for m in METHODS},
        comparisons={baseline: {method: dict(
            speedup=quantiles(p["seconds"][baseline] / p["seconds"][method] for p in medians),
            baseline_minus_method_ms=quantiles((p["seconds"][baseline] - p["seconds"][method]) * 1000
                                              for p in medians),
            cases_method_median_lower=sum(p["seconds"][method] < p["seconds"][baseline] for p in medians),
            cases_equal_median=sum(p["seconds"][method] == p["seconds"][baseline] for p in medians))
                                for method in METHODS} for baseline in BASELINES},
        paired_case_medians=medians)


def model_summary(rows):
    result = summarize(rows)
    result["c0_controls"] = {name: summarize([r for r in rows if r[field] == 0])
                             for name, field in (("selected_sparse_cut_zero", "selected_sparse_cut"),
                                                 ("faulty_layer_zero", "layer"))}
    result["strata"] = {}
    for name, accessor in (("prefix_tokens", lambda r: r["case"]["prefix_tokens"]),
                           ("window", lambda r: r["case"]["window"]), ("layer", lambda r: r["layer"])):
        result["strata"][name] = {str(value): summarize([r for r in rows if accessor(r) == value])
                                  for value in sorted({accessor(r) for r in rows})}
    return result


def detection_summary(rows):
    def subset_summary(subset):
        return dict(n=len(subset),
            actual_changed_cases=sum(r["fault"]["changed_elements"] > 0 for r in subset),
            observed_alarm_cases=sum(bool(r["detection"]["detected_layers"]) for r in subset),
            unique_layer_routes=sum(r["detection"]["route"] == "unique_layer" for r in subset),
            no_alarm_full_routes=sum(r["detection"]["route"] == "no_alarm_full" for r in subset),
            timing_ms=quantiles(statistics.median(t["seconds"] for t in r["detection"]["timings"]) * 1000
                                for r in subset),
            logical_bytes_read=quantiles(r["detection"]["logical_bytes_read"] for r in subset))
    return dict(all_cases=subset_summary(rows),
                perturbed=subset_summary([r for r in rows if r["case"]["fault"] != "noop"]),
                noops=subset_summary([r for r in rows if r["case"]["fault"] == "noop"]))


LIMITS = [
    "Supplemental correctness and immutability checks are recorded runner assertions. No supplemental "
    "tensor snapshots were saved or independently loaded; no model inference was rerun by this auditor.",
    "Native parity flags and coordinate/vector reconstruction are checked and bound to primary records; "
    "this is not independent hardware/model replication or authentication of every model-weight byte.",
    "The single warmup per method is declared in metadata/source and excluded from recorded trials. "
    "Its actual execution cannot be independently reconstructed from the saved timing rows.",
    "Five repeated timings per policy are not independent fault trials; paired case medians describe "
    "these constructed cases, not an IID population, field fault rates, or statistically proven dominance.",
    "No new normal-decoding overhead measurements were made. Detector comparison time is measured "
    "separately, outside recovery timing. No end-to-end break-even, commit-delay, protected-memory "
    "cost or serving-system performance claim follows.",
    "Checkpoint-alias full replay assumes an immutable trusted resident GPU checkpoint and ordinary "
    "DynamicCache out-of-place append. Prefix views retain backing allocations until replaced.",
    "The common working-cache clone is excluded even when unused by checkpoint-alias replay. "
    "Policy-specific restoration and replay are timed; correctness diagnostics are excluded.",
    "Sparse/all retain their original conservative restoration, not their optimized attainable frontier. "
    "This post-hoc baseline sensitivity does not replace or pool primary timing measurements.",
    "c0 controls are perturbed cases with zero selected sparse cut or fault layer, not no-op injections.",
    "The naive prefix detector covers persistent post-checkpoint changes in the old K/V prefix only. "
    "Suffix-only changes, transient/reverted faults, pre-checkpoint errors, and dirty/corrupt trusted "
    "checkpoints are not covered. It is a bounded feasibility mechanism, not a production detector.",
    "Detector bytes are logical reads of current and trusted old-prefix tensor payload, not measured "
    "physical memory traffic. One consolidated decision transfer is supported by source inspection, "
    "not independently profiled here. No-alarm forces conservative full replay; multiple layers fail closed.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "primary", "config", "manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    audit = Audit()
    output = dict(status="failed", version=VERSION, audited_utc=datetime.now(timezone.utc).isoformat(),
                  run=str(args.run.resolve()), limitations=LIMITS)
    try:
        metadata, rows, detections, hashes = validate(audit, args.run, args.primary, args.config, args.manifest)
        summary = dict(status="posthoc_supplement_complete_audited", input_sha256=hashes,
                       rows=len(rows), methods=list(METHODS),
                       ratio_definition="paired per-case median baseline time / method time; >1 means method faster",
                       models={model: model_summary([r for r in rows if r["model"] == model])
                               for model in sorted({r["model"] for r in rows})},
                       detection={model: detection_summary([r for r in detections if r["model"] == model])
                                  for model in sorted({r["model"] for r in detections})}, caveats=LIMITS)
        summary_path = args.run / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        output.update(status="passed", input_sha256=hashes, summary_sha256=digest(summary_path),
                      source_sha256=metadata["source_sha256"])
    except Exception as error:
        if not audit.failures:
            audit.failures.append(type(error).__name__ + ": " + str(error))
        output["summary_written"] = False
    output.update(checks=audit.checks, counts=audit.counts, failures=audit.failures)
    (args.run / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
