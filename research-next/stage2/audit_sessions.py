"""Independent standard-library record audit for RECUT transactional sessions.

Does not import the controller, runner, analyzer, Torch, or Transformers. Checks
the frozen matrix, source binding, arithmetic, route/trace consistency, native
comparison verdicts, and paired perturbations. Native gold tensors and actual
recovered tensors were not persisted, so this is NOT tensor re-execution or an
external replication. No false-negative guarantee follows from recorded flags.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import struct


HERE = Path(__file__).resolve().parent
VERSION = "RECUT-stage2-independent-record-audit-v1"
SCENARIOS = {"clean", "single_prefix", "multi_prefix", "repeated_prefix",
             "journal_corruption", "checkpoint_corruption"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_loads(text):
    def invalid(value):
        raise ValueError("Nonstandard JSON constant: " + value)
    return json.loads(text, parse_constant=invalid)


def load(path):
    return strict_loads(Path(path).read_text())


def records(path):
    return [strict_loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def stored_sum(before, delta, dtype):
    """Independent IEEE addition/store check; does not regenerate Gaussian RNG."""
    result = float32(before + delta)
    if dtype == "float32":
        return result
    if dtype != "bfloat16":
        raise ValueError("Unsupported recorded arithmetic dtype")
    bits = struct.unpack("<I", struct.pack("<f", result))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def distribution(values):
    values = sorted(values)
    if not values:
        return {key: 0 if key == "n" else None
                for key in ("n", "median", "q25", "q75", "min", "max", "mean")}
    def quantile(p):
        index = (len(values) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(n=len(values), median=statistics.median(values), q25=quantile(.25),
                q75=quantile(.75), min=values[0], max=values[-1], mean=statistics.mean(values))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
            raise ValueError(label)

    def equal(self, actual, expected, label):
        """Compare an independently reconstructed expected subset recursively."""
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label + "/dict")
            for key, value in expected.items():
                self.check(str(key) in actual, label + "/key/" + str(key))
                self.equal(actual[str(key)], value, label + "/" + str(key))
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(actual) == len(expected), label + "/length")
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, label + "/" + str(index))
        elif type(expected) is float:
            self.check(finite(actual) and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-15), label)
        else:
            self.check(type(actual) is type(expected) and actual == expected, label)

    def comparison(self, value, label, cache=False):
        self.check(value["equal"] is True and value["finite"] is True
                   and value["max_abs_error"] == 0, label + "/exact-finite")
        if cache:
            self.check(value["eq"] is True and value["mismatches"] == [], label + "/cache-detail")


def tensor_fingerprint(audit, record, shape, dtype, label):
    size = 2 if dtype == "bfloat16" else 4
    audit.check(record["shape"] == list(shape) and record["dtype"] == "torch." + dtype,
                label + "/layout")
    audit.check(record["bytes"] == math.prod(shape) * size, label + "/payload")
    audit.check(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, label + "/sha-format")


def cache_fingerprint(audit, record, config, length, dtype, label):
    layers = config["num_hidden_layers"]
    shape = (1, config["num_key_value_heads"], length,
             config["hidden_size"] // config["num_attention_heads"])
    audit.check(record["seen_tokens"] == length, label + "/seen-tokens")
    for field in ("keys", "values"):
        audit.check(len(record[field]) == layers, label + "/" + field + "/layers")
        for index, fingerprint in enumerate(record[field]):
            tensor_fingerprint(audit, fingerprint, shape, dtype, label + "/" + field + "/" + str(index))


def expected_events(scenario, method, index, layers, seed):
    if scenario == "clean" or (index != 1 and scenario != "repeated_prefix"):
        return []
    seed += 1009 * index
    later = min(layers - 1, 3 * layers // 4)
    events = []
    if scenario in ("single_prefix", "repeated_prefix", "journal_corruption"):
        events.append(("working_prefix", 1, seed, later))
    if scenario == "multi_prefix":
        events += [("working_prefix", 1, seed, layers - 2),
                   ("working_prefix", 2, seed + 1, max(0, layers // 2 - 1))]
    if scenario == "checkpoint_corruption":
        events.append(("checkpoint", 2, seed, later))
    if scenario == "journal_corruption" and method == "sparse":
        events.append(("journal", 2, seed + 2, None))
    return events


def audit_faults(audit, row, window, config, dtype, reference, label):
    layers, wi = config["num_hidden_layers"], window["window_index"]
    prefix = row["prefix_tokens"] + wi * row["window"]
    events = expected_events(row["scenario"], row["method"], wi, layers, row["seed"])
    faults = window["faults"]
    audit.check(len(faults) == len(events), label + "/event-count")
    for index, (fault, event) in enumerate(zip(faults, events)):
        tag = label + "/event/" + str(index)
        target, step, seed, layer = event
        audit.check((fault["target"], fault["after_step"], fault["seed"], fault.get("layer"))
                    == event and fault["window_index"] == wi, tag + "/schedule")
        audit.check(fault["batch"] == 0 and finite(fault["rms"]) and fault["rms"] > 0, tag + "/rms-batch")
        width = config["hidden_size"] if target == "journal" else config["hidden_size"] // config["num_attention_heads"]
        for field in ("before", "requested_delta", "after"):
            audit.check(len(fault[field]) == width and all(finite(v) for v in fault[field]), tag + "/" + field)
        changed = sum(before != after for before, after in zip(fault["before"], fault["after"]))
        audit.check(changed == fault["changed_elements"] and changed > 0, tag + "/actual-changed-elements")
        for number, (before, delta, after) in enumerate(zip(fault["before"], fault["requested_delta"], fault["after"])):
            audit.check(stored_sum(before, delta, dtype) == after, tag + "/stored-addition/" + str(number))
        if target == "journal":
            later = min(layers - 1, 3 * layers // 4)
            eligible = max(c for c in row["sparse_cuts"] if c <= later)
            pending = reference["initial_token"] if wi == 0 else reference["windows"][wi - 1]["tokens"][-1]
            audit.check(fault["cut"] == eligible and fault["journal_step"] == 0
                        and fault["journal_position"] == prefix and fault["journal_token"] == pending
                        and fault["sequence_position"] == 0, tag + "/journal-binding")
        else:
            audit.check(fault["kind"] == "V" and 0 <= fault["head"] < config["num_key_value_heads"]
                        and 0 <= fault["prefix_position"] < prefix, tag + "/old-prefix-coordinates")


def audit_session(audit, row, reference, model_config, metadata, label, warmup=False):
    config, dtype = metadata["config"], metadata["dtype"]
    n, w = model_config["num_hidden_layers"], row["window"]
    cuts = sorted({c for c in (n // 4, n // 2, 3 * n // 4) if 0 < c < n})
    audit.check(row["num_layers"] == n and row["sparse_cuts"] == cuts, label + "/layers-cuts")
    audit.check(row["valid"] is True and row["same_policy_treatment"] is (row["scenario"] != "journal_corruption"),
                label + "/valid-treatment-label")
    audit.check(row["n_windows"] == (1 if warmup else config["n_windows"]), label + "/window-count-contract")
    abort = row["scenario"] == "checkpoint_corruption"
    count = 2 if abort else row["n_windows"]
    committed_count = 1 if abort else row["n_windows"]
    audit.check(len(row["windows"]) == count and [v["window_index"] for v in row["windows"]] == list(range(count)),
                label + "/exact-window-coverage")
    audit.check(row["status"] == ("expected_abort" if abort else "complete")
                and row["poisoned"] is abort and row["windows_committed"] == committed_count,
                label + "/termination")
    expected_tokens = [token for reference_window in reference["windows"][:committed_count]
                       for token in reference_window["tokens"]]
    audit.check(row["committed_tokens"] == expected_tokens, label + "/public-transcript")
    audit.equal(row["total_wall_seconds"], sum(v["wall_seconds"] for v in row["windows"]), label + "/wall-sum")
    audit.equal(row["total_injection_harness_seconds"], sum(v["injection_harness_seconds"] for v in row["windows"]),
                label + "/injection-sum")
    for wi, window in enumerate(row["windows"]):
        tag = label + "/window/" + str(wi)
        prefix = row["prefix_tokens"] + wi * w
        payload = 2 * n * model_config["num_key_value_heads"] * prefix * (model_config["hidden_size"] // model_config["num_attention_heads"]) * (2 if dtype == "bfloat16" else 4)
        journal_payload = w * len(cuts) * model_config["hidden_size"] * (2 if dtype == "bfloat16" else 4) if row["method"] == "sparse" else 0
        audit.check(window["valid"] is True and window["no_early_commit"] is True, tag + "/valid-no-early-output")
        audit.check(window["logical_input_cache_bytes"] == window["logical_checkpoint_bytes"] == payload
                    and window["logical_journal_bytes"] == journal_payload, tag + "/logical-payload")
        for field in ("wall_seconds", "wall_minus_injection_seconds"):
            audit.check(finite(window[field]) and window[field] > 0, tag + "/" + field)
        audit.check(finite(window["injection_harness_seconds"]) and window["injection_harness_seconds"] >= 0,
                    tag + "/injection-time")
        audit.equal(window["wall_minus_injection_seconds"], window["wall_seconds"] - window["injection_harness_seconds"],
                    tag + "/diagnostic-subtraction")
        for field in ("allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes"):
            audit.check(type(window[field]) is int and window[field] >= 0, tag + "/" + field)
        audit.check(window["peak_increment_bytes"] == window["peak_allocated_bytes"] - window["allocated_before_bytes"],
                    tag + "/peak-accounting")
        if metadata["device"] == "cpu":
            audit.check(window["peak_allocated_bytes"] == window["allocated_before_bytes"] == 0, tag + "/no-cpu-allocator-claim")
        audit_faults(audit, row, window, model_config, dtype, reference, tag)
        audit.check(window["committed_before"] == wi * w, tag + "/committed-before")
        checks = window["checks"]
        if abort and wi == 1:
            audit.check(window["status"] == "rejected" and window["committed_after"] == wi * w, tag + "/no-abort-release")
            expected_checks = {key: True for key in ("expected_rejection", "poisoned", "committed_tokens_unchanged",
                "committed_windows_unchanged", "committed_cache_unchanged", "committed_logits_unchanged",
                "next_token_unchanged", "no_early_commit")}
            audit.equal(checks, expected_checks, tag + "/abort-checks")
            audit.check(window["rejection"]["type"] == "WindowRejected"
                        and window["rejection"]["reason"] == "Checkpoint integrity failure", tag + "/guard-refusal")
            abort_metadata = window["rejection"]["controller_metadata"]
            audit.check(abort_metadata["route"] == "rejected"
                        and abort_metadata["replay_layer_steps"] == 0, tag + "/abort-no-replay")
            audit.check("tokens" not in window and "selected_cut" not in window, tag + "/no-abort-success-result")
            continue
        audit.check(window["status"] == "committed" and window["committed_after"] == (wi + 1) * w,
                    tag + "/whole-window-release")
        audit.check(window["tokens"] == reference["windows"][wi]["tokens"] and len(window["tokens"]) == w,
                    tag + "/gold-tokens")
        for field in ("cache", "logits", "continuation_cache", "continuation_logits"):
            audit.comparison(checks[field], tag + "/" + field, cache="cache" in field)
        for field in ("tokens_equal", "committed_tokens_equal", "next_token_equal", "continuation_tokens_equal",
                      "not_poisoned", "all_equal", "detected_layers_equal_injection", "recovery_expected",
                      "committed_windows_equal", "no_early_commit", "unexpected_checkpoint_commit"):
            audit.check(checks[field] is True, tag + "/native-verdict/" + field)
        detected = sorted({fault["layer"] for fault in window["faults"] if fault["target"] == "working_prefix"})
        audit.check(window["detected_layers"] == detected and window["recovered"] is bool(detected), tag + "/observed-prefix-route")
        eligible_cut = max([0] + [cut for cut in cuts if detected and cut <= min(detected)]) if row["method"] == "sparse" else 0
        journal_challenge = row["scenario"] == "journal_corruption" and wi == 1 and row["method"] == "sparse"
        selected = 0 if journal_challenge else eligible_cut
        fallback = "journal_integrity_failure" if journal_challenge else None
        audit.check(window["selected_cut"] == selected and window["fallback_reason"] == fallback, tag + "/detector-derived-cut")
        speculative = window["speculative_tokens"]
        audit.check(len(speculative) == w and all(type(token) is int and 0 <= token < model_config["vocab_size"] for token in speculative),
                    tag + "/speculative-token-record")
        divergence = next((index + 1 for index, pair in enumerate(zip(window["tokens"], speculative)) if pair[0] != pair[1]), None)
        audit.check(window["first_divergence"] == divergence, tag + "/independent-first-divergence")
        starts = ([selected if divergence is None or index < divergence else 0 for index in range(w)] if detected else [])
        audit.check(window["replay_starts"] == starts, tag + "/irreversible-history-gate")
        cm = window["controller_metadata"]
        route = "journal_fallback" if fallback else ("sparse" if selected else "full") if detected else "clean"
        audit.equal(cm, {"window_index": wi, "prefix_tokens": prefix, "window_tokens": w,
            "checkpoint_bytes": payload, "preserved_committed_cache_bytes": payload, "journal_bytes": journal_payload,
            "replay_layer_steps": sum(n - cut for cut in starts), "route": route,
            "journal_integrity_checked": bool(eligible_cut and detected)}, tag + "/controller-accounting")
        audit.check(finite(cm["replay_seconds"]) and 0 <= cm["replay_seconds"] <= window["wall_seconds"], tag + "/replay-time-subset")
        audit.check((cm["replay_seconds"] > 0) is bool(detected), tag + "/replay-time-presence")
        if journal_challenge:
            audit.check(checks["journal_corruption_fell_back"] is True, tag + "/journal-fallback-verdict")


def reconstructed_summary(rows, references):
    groups, pairmap, timing_groups = defaultdict(list), defaultdict(dict), defaultdict(list)
    fault_targets, fault_histogram, rejected = Counter(), Counter(), Counter()
    all_windows = []
    for row in rows:
        groups[(row["model"], row["scenario"], row["method"])].append(row)
        pairmap[(row["workload_id"], row["scenario"], row["repetition"])][row["method"]] = row
        for window in row["windows"]:
            all_windows.append(window)
            fault_histogram[str(len(window["faults"]))] += 1
            fault_targets.update(fault["target"] for fault in window["faults"])
            if window["status"] == "rejected":
                rejected[window["rejection"]["reason"]] += 1
    group_summaries = []
    for (model, scenario, method), group in sorted(groups.items()):
        windows = [window for row in group for window in row["windows"]]
        item = dict(model=model, scenario=scenario, method=method, sessions=len(group),
            windows_attempted=len(windows), windows_committed=sum(w["status"] == "committed" for w in windows),
            windows_rejected=sum(w["status"] == "rejected" for w in windows),
            recovered_windows=sum(bool(w.get("recovered")) for w in windows),
            session_wall_seconds=distribution(row["total_wall_seconds"] for row in group),
            fallback_reasons=dict(Counter(w["fallback_reason"] for w in windows if w.get("fallback_reason"))),
            selected_cuts=dict(Counter(str(w["selected_cut"]) for w in windows if w.get("recovered"))))
        for key, field in (("window_wall_seconds", "wall_seconds"),
                           ("window_wall_minus_injection_seconds", "wall_minus_injection_seconds"),
                           ("peak_increment_bytes", "peak_increment_bytes"),
                           ("logical_checkpoint_bytes", "logical_checkpoint_bytes"),
                           ("logical_journal_bytes", "logical_journal_bytes")):
            item[key] = distribution(w[field] for w in windows)
        group_summaries.append(item)
    pairs = []
    for (workload, scenario, rep), pair in sorted(pairmap.items()):
        full, sparse = pair["full"], pair["sparse"]
        treatment = scenario != "journal_corruption"
        item = dict(workload_id=workload, scenario=scenario, repetition=rep, model=full["model"],
                    same_policy_treatment=treatment,
                    session_full_over_sparse_wall_ratio=full["total_wall_seconds"] / sparse["total_wall_seconds"],
                    window_ratios=[])
        for left, right in zip(full["windows"], sparse["windows"]):
            record = dict(window_index=left["window_index"], status=left["status"],
                full_over_sparse_wall_ratio=left["wall_seconds"] / right["wall_seconds"],
                full_over_sparse_adjusted_ratio=left["wall_minus_injection_seconds"] / right["wall_minus_injection_seconds"],
                fault_affected=bool(left["faults"]), same_policy_treatment=treatment)
            item["window_ratios"].append(record)
            if treatment:
                timing_groups[(full["model"], scenario, record["fault_affected"], record["status"])].append(record)
        pairs.append(item)
    paired_timing = []
    for (model, scenario, fault, status), group in sorted(timing_groups.items()):
        paired_timing.append(dict(model=model, scenario=scenario, fault_affected=fault, status=status,
            wall_full_over_sparse_ratio=distribution(record["full_over_sparse_wall_ratio"] for record in group),
            adjusted_full_over_sparse_ratio=distribution(record["full_over_sparse_adjusted_ratio"] for record in group)))
    return dict(status="complete", sessions=len(rows), workloads=len(references), windows_attempted=len(all_windows),
        windows_committed=sum(w["status"] == "committed" for w in all_windows),
        windows_rejected=sum(w["status"] == "rejected" for w in all_windows),
        recovered_windows=sum(bool(w.get("recovered")) for w in all_windows),
        fault_target_counts=dict(fault_targets), faults_per_window_histogram=dict(fault_histogram),
        rejection_reasons=dict(rejected), all_recorded_checks_pass=True,
        native_parity_positions=sum(len(r["parity"]) for r in references), groups=group_summaries,
        paired_timing_summary=paired_timing, pairs=pairs)


def validate(audit, directory, config_path, manifest_path, summary_path):
    meta, config, manifest = load(directory / "metadata.json"), load(config_path), load(manifest_path)
    rows, refs = records(directory / "sessions.jsonl"), records(directory / "references.jsonl")
    hashes = {name: sha(directory / name) for name in ("metadata.json", "sessions.jsonl", "references.jsonl")}
    hashes.update(config=sha(config_path), manifest=sha(manifest_path), auditor=sha(__file__))
    audit.check(meta["status"] == "complete", "completed-metadata")
    audit.check(meta["config"] == config and meta["manifest"] == manifest, "exact-config-manifest")
    audit.check(meta["config_sha256"] == hashes["config"] and meta["manifest_sha256"] == hashes["manifest"], "input-bindings")
    for field in ("sessions", "references"):
        audit.check(meta[field + "_sha256"] == hashes[field + ".jsonl"], field + "/raw-hash")
    for field in ("timing_scope", "adjusted_timing_scope", "memory_scope", "equality", "oracle_storage", "python", "torch"):
        audit.check(bool(meta[field]), "metadata/" + field)
    audit.check(meta["transformers"] == "4.51.3" and meta["dtype"] in ("bfloat16", "float32")
                and meta["device"] in ("cuda", "cpu") and meta["deterministic_algorithms"] is True
                and meta["tf32"] is False and meta["attention"] == "eager" and meta["batch_size"] == 1,
                "bounded-numeric-environment")
    audit.check(finite(meta["started_unix_seconds"]) and finite(meta["finished_unix_seconds"])
                and meta["finished_unix_seconds"] > meta["started_unix_seconds"], "completion-time")
    for field in ("models", "prefixes", "windows", "seeds", "scenarios", "methods"):
        audit.check(len(config[field]) == len(set(config[field])) and bool(config[field]), "config/unique/" + field)
    audit.check(set(config["scenarios"]) <= SCENARIOS and set(config["methods"]) == {"full", "sparse"}, "config/scope")
    audit.check(type(config["n_windows"]) is int and config["n_windows"] >= 2
                and type(config["repetitions"]) is int and config["repetitions"] > 0
                and config["warmups"] in (0, 1), "config/repetitions-windows-warmups")
    required_sources = {"controller.py", "run_sessions.py", "analyze_sessions.py",
                        "../experiments/recut_engine.py", "../experiments/prefix_detector.py", "../experiments/run_recut.py"}
    audit.check(required_sources <= set(meta["source_sha256"]), "source/dependency-coverage")
    for relative, expected in meta["source_sha256"].items():
        path = (HERE / relative).resolve()
        audit.check(path.parent in (HERE, HERE.parent / "experiments") and path.suffix == ".py", "source/path/" + relative)
        audit.check(sha(path) == expected, "source/hash/" + relative)
    models = {record["model"]: record for record in manifest}
    audit.check(len(models) == len(manifest) and set(config["models"]) <= set(models), "model-manifest-coverage")
    audit.check(set(meta["models_completed"]) == set(config["models"]) and len(meta["models_completed"]) == len(config["models"]),
                "models-completed")
    for model in config["models"]:
        record = models[model]
        audit.check(re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is not None, "revision/" + model)
        files = meta["model_files"][model]
        audit.check(bool(files) and len({f["file"] for f in files}) == len(files), "model-file-records/" + model)
        for file in files:
            audit.check(not Path(file["file"]).is_absolute() and ".." not in Path(file["file"]).parts
                        and type(file["bytes"]) is int and file["bytes"] > 0
                        and re.fullmatch(r"[0-9a-f]{64}", file["sha256"]) is not None, "model-file-shape/" + model)
    workloads = {}
    for model, prefix, window, seed in itertools.product(config["models"], config["prefixes"], config["windows"], config["seeds"]):
        identifier = f"{model.replace('/', '--')}-p{prefix}-w{window}-s{seed}"
        workloads[identifier] = (model, prefix, window, seed)
    refmap = {record["workload_id"]: record for record in refs}
    audit.check(len(refmap) == len(refs) == len(workloads) and set(refmap) == set(workloads), "complete-reference-matrix")
    audit.check(len(meta["workloads"]) == len(workloads) and {w["id"] for w in meta["workloads"]} == set(workloads),
                "metadata-workload-matrix")
    for identifier, reference in refmap.items():
        model, prefix, window, seed = workloads[identifier]
        model_config = meta["model_configs"][model]
        audit.check(reference["model"] == model and reference["revision"] == models[model]["revision"], identifier + "/model-binding")
        audit.check(len(reference["prefix_ids"]) == prefix and all(type(t) is int and 0 <= t < model_config["vocab_size"]
                    for t in reference["prefix_ids"] + [reference["initial_token"]]), identifier + "/prefix")
        positions = prefix + window * config["n_windows"] + 2
        audit.check([p["position"] for p in reference["parity"]] == list(range(positions)), identifier + "/parity-coverage")
        marker = next(w for w in meta["workloads"] if w["id"] == identifier)
        audit.check(marker["passed"] is True and marker["parity_positions"] == positions, identifier + "/parity-metadata")
        for record in reference["parity"]:
            audit.comparison(record["cache"], identifier + "/parity-cache", cache=True)
            audit.comparison(record["logits"], identifier + "/parity-logits")
            audit.check(record["token_equal"] is True, identifier + "/parity-token")
        cache_fingerprint(audit, reference["checkpoint"], model_config, prefix, meta["dtype"], identifier + "/checkpoint-fingerprint")
        audit.check([w["window_index"] for w in reference["windows"]] == list(range(config["n_windows"])), identifier + "/reference-windows")
        for wi, record in enumerate(reference["windows"]):
            tag = identifier + "/reference/" + str(wi)
            audit.check(len(record["tokens"]) == window and len(record["continuation_tokens"]) == 2
                        and all(type(t) is int and 0 <= t < model_config["vocab_size"] for t in record["tokens"] + record["continuation_tokens"]),
                        tag + "/tokens")
            for continuation in (False, True):
                prefix_key = "continuation_" if continuation else ""
                cache_fingerprint(audit, record[prefix_key + "cache"], model_config,
                    prefix + (wi + 1) * window + (2 if continuation else 0), meta["dtype"], tag + "/" + prefix_key + "cache")
                tensor_fingerprint(audit, record[prefix_key + "logits"], (1, 1, model_config["vocab_size"]),
                                   meta["dtype"], tag + "/" + prefix_key + "logits")
    keys = [(r["workload_id"], r["scenario"], r["method"], r["repetition"]) for r in rows]
    expected = set(itertools.product(workloads, config["scenarios"], config["methods"], range(config["repetitions"])))
    audit.check(len(keys) == len(set(keys)) == len(expected) == meta["expected_sessions"] == meta["sessions_completed"]
                and set(keys) == expected, "complete-unique-session-matrix")
    pairs, repeated = defaultdict(dict), {}
    for row in rows:
        identifier, scenario, method, rep = row["workload_id"], row["scenario"], row["method"], row["repetition"]
        label = "/".join((identifier, scenario, method, str(rep)))
        model, prefix, window, seed = workloads[identifier]
        audit.check((row["model"], row["prefix_tokens"], row["window"], row["seed"]) == (model, prefix, window, seed), label + "/workload-binding")
        audit.check(row["model_revision"] == models[model]["revision"] and row["config_sha256"] == hashes["config"]
                    and row["source_sha256"] == meta["source_sha256"], label + "/provenance")
        methods = config["methods"]
        offset = rep % len(methods)
        audit.check(row["method_order"] == methods[offset:] + methods[:offset], label + "/rotated-order")
        audit_session(audit, row, refmap[identifier], meta["model_configs"][model], meta, label)
        pairs[(identifier, scenario, rep)][method] = row
        fault_records = [w["faults"] for w in row["windows"]]
        rep_key = (identifier, scenario, method)
        if rep_key in repeated:
            audit.check(repeated[rep_key] == fault_records, label + "/repeated-identical-perturbations")
        else:
            repeated[rep_key] = fault_records
    for key, pair in pairs.items():
        left, right = pair["full"], pair["sparse"]
        for lw, rw in zip(left["windows"], right["windows"]):
            common_left = [f for f in lw["faults"] if f["target"] != "journal"]
            common_right = [f for f in rw["faults"] if f["target"] != "journal"]
            audit.check(common_left == common_right, str(key) + "/paired-identical-common-faults")
            audit.check(lw["status"] == rw["status"], str(key) + "/paired-status")
            if key[1] != "journal_corruption":
                audit.check(lw["faults"] == rw["faults"], str(key) + "/paired-identical-entire-treatment")
    expected_warmups = {(model, method) for model in config["models"] for method in config["methods"]} if config["warmups"] else set()
    warmups = meta["warmup_results"]
    audit.check(len(warmups) == len(expected_warmups) and {(r["model"], r["method"]) for r in warmups} == expected_warmups,
                "excluded-warmup-coverage")
    for index, row in enumerate(warmups):
        audit.check(row["scenario"] == "clean" and row["repetition"] == -1
                    and row["method_order"] == [row["method"]], "warmup/identity/" + str(index))
        audit_session(audit, row, refmap[row["workload_id"]], meta["model_configs"][row["model"]], meta,
                      "warmup/" + str(index), warmup=True)
    expected_summary = reconstructed_summary(rows, refs)
    summary = load(summary_path)
    expected_summary.update(metadata_sha256=hashes["metadata.json"], sessions_sha256=hashes["sessions.jsonl"],
                            references_sha256=hashes["references.jsonl"], analyzer_sha256=sha(HERE / "analyze_sessions.py"))
    audit.equal(summary, expected_summary, "independently-recomputed-summary")
    hashes["summary"] = sha(summary_path)
    return {"hashes": hashes, "counts": {key: expected_summary[key] for key in ("sessions", "workloads", "windows_attempted",
            "windows_committed", "windows_rejected", "recovered_windows", "native_parity_positions", "fault_target_counts")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config_sessions.json")
    parser.add_argument("--manifest", type=Path, default=HERE.parent / "experiments/model_manifest.json")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit, result = Audit(), {}
    try:
        result = validate(audit, args.directory, args.config, args.manifest,
                          args.summary or args.directory / "summary.json")
    except Exception as error:
        if not audit.failures:
            audit.failures.append(type(error).__name__ + ": " + str(error))
    result.update(version=VERSION, status="passed" if not audit.failures else "failed", checks=audit.checks,
                  failures=audit.failures, audited_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=["Independent record/source audit, not reload of recovered or native gold tensors.",
                    "Native comparisons and no-early-output claims remain recorded live-run verdicts.",
                    "Route consistency is checked against observed layers; no-oracle API was separately code-reviewed.",
                    "Floating-point applications are checked; Gaussian draws are not independently regenerated.",
                    "Recorded model-file hashes are checked for structure, not remote model-weight reload.",
                    "No external replication, production service, hardware-fault coverage, or novelty guarantee."])
    output = args.output or args.directory / "audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failures")}, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
