"""Independent RECUT stage-3 record and saved-tensor evidence audit.

The record audit uses only Python's standard library and does not import the
runner, controller, or analyzer. The optional saved-tensor audit imports Torch
only to load primitive/tensor snapshots with weights_only=True on CPU. Reloading
saved actual/reference states is stronger than trusting equality flags, but is
not model re-execution, external replication, or proof of general resilience.
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
VERSION = "RECUT-stage3-independent-evidence-audit-v1"
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    result = float32(before + delta)
    if dtype == "float32":
        return result
    if dtype != "bfloat16":
        raise ValueError("Unsupported arithmetic dtype: " + str(dtype))
    bits = struct.unpack("<I", struct.pack("<f", result))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def stored_bits(value, dtype):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "bfloat16":
        if bits & 0xffff:
            raise ValueError("Recorded value is not exactly BF16")
        return bits >> 16
    if dtype == "float32":
        return bits
    raise ValueError("Unsupported arithmetic dtype: " + str(dtype))


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
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label + "/dict")
            for key, value in expected.items():
                self.check(str(key) in actual, label + "/key/" + str(key))
                self.equal(actual[str(key)], value, label + "/" + str(key))
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(actual) == len(expected), label + "/length")
            for i, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, label + "/" + str(i))
        elif type(expected) is float:
            self.check(finite(actual) and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-15), label)
        else:
            self.check(type(actual) is type(expected) and actual == expected, label)

    def comparison(self, value, label, cache=False):
        self.check(value["equal"] is True and value["finite"] is True
                   and value["max_abs_error"] == 0, label + "/exact-finite")
        if cache:
            self.check(value["eq"] is True and value["mismatches"] == [], label + "/cache-detail")


def relative_file(audit, directory, name, label):
    rel = Path(name)
    audit.check(not rel.is_absolute() and ".." not in rel.parts, label + "/relative-path")
    path = (directory / rel).resolve()
    audit.check(directory.resolve() in path.parents and path.is_file(), label + "/contained-file")
    return path


def tensor_fingerprint(audit, record, shape, dtype, label):
    audit.check(record["shape"] == list(shape) and record["dtype"] == "torch." + dtype, label + "/layout")
    audit.check(record["bytes"] == math.prod(shape) * (2 if dtype == "bfloat16" else 4), label + "/payload")
    audit.check(SHA256.fullmatch(record["sha256"]) is not None, label + "/sha-format")


def cache_fingerprint(audit, record, config, length, dtype, label):
    layers = config["num_hidden_layers"]
    shape = (1, config["num_key_value_heads"], length, config["hidden_size"] // config["num_attention_heads"])
    audit.check(record["seen_tokens"] == length, label + "/seen-tokens")
    for field in ("keys", "values"):
        audit.check(len(record[field]) == layers, label + "/" + field + "/layers")
        for i, tensor in enumerate(record[field]):
            tensor_fingerprint(audit, tensor, shape, dtype, label + "/" + field + "/" + str(i))


def expected_events(scenario, policy, wi, layers, seed, window):
    """Protocol schedule reconstructed without calling the injector."""
    if scenario == "clean" or (wi != 1 and scenario != "repeated_prefix"):
        return []
    base = seed + 1009 * wi
    late = 3 * layers // 4
    cases = {
        "key_early": ("K", 0, "additive", 1),
        "value_early": ("V", 0, "bitflip", window // 2),
        "key_mid": ("K", layers // 2, "additive", 1),
        "value_mid": ("V", layers // 2, "bitflip", window // 2),
        "key_late": ("K", late, "bitflip", 1),
        "value_late": ("V", late, "additive", window // 2),
        "repeated_prefix": ("V", late, "additive", 1),
    }
    if scenario in cases:
        kind, layer, mode, step = cases[scenario]
        return [("working_prefix", step, base, layer, kind, mode)]
    if scenario == "multi_prefix":
        return [("working_prefix", 1, base, late, "V", "additive"),
                ("working_prefix", 2, base + 1, max(0, layers // 2 - 1), "K", "additive")]
    if scenario == "checkpoint_corruption":
        return [("checkpoint", 2, base, late, "V", "additive")]
    if scenario == "journal_corruption":
        events = [("working_prefix", 1, base, late, "V", "additive")]
        if policy != "full":
            events.append(("journal", 2, base + 1, None, None, "additive"))
        return events
    raise ValueError("Unknown scenario: " + scenario)


VARIANTS = ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched", "full_legacy", "sparse_legacy")
BATCHED = ("full_batched", "sparse_batched", "allcuts_batched")


def expected_parameters(variant, layers):
    if variant in ("native", "bare"):
        return "unprotected", None, []
    name, seal = variant.split("_")
    if name == "full":
        return "full", seal, []
    cuts = list(range(1, layers)) if name == "allcuts" else sorted({c for c in (layers // 4, layers // 2, 3 * layers // 4) if 0 < c < layers})
    return "sparse", seal, cuts


def expected_matrix(config, workloads, phase):
    expected = {}
    first_seed = config["seeds"][0]
    for workload in workloads:
        if phase == "warmups":
            groups = [("clean", first_seed, -1, VARIANTS), ("value_late", first_seed, -1, VARIANTS[2:])] if config.get("warmup", True) else []
        elif phase == "profiles":
            groups = [(scenario, first_seed, -2, ("full_batched", "sparse_batched")) for scenario in ("clean", "value_late")] if config.get("profile", False) else []
        else:
            groups = [("clean", first_seed, rep, VARIANTS) for rep in range(config["clean_repetitions"])]
            for scenario, seed, rep in itertools.product(config["fault_cases"], config["seeds"], range(config["repetitions"])):
                variants = BATCHED + (("full_legacy", "sparse_legacy") if scenario in config.get("legacy_cases", []) else ())
                groups.append((scenario, seed, rep, variants))
            groups.extend((scenario, first_seed, 0, BATCHED) for scenario in config["guard_cases"])
        for scenario, seed, rep, variants in groups:
            offset = rep % len(variants)
            order = list(variants[offset:] + variants[:offset])
            for variant in variants:
                expected[(workload, scenario, variant, seed, rep)] = order if phase == "sessions" else [variant]
    return expected


def expected_snapshot(config, row, wi):
    return (row["prompt_id"] == config.get("snapshot_prompt") and row["seed"] == config["seeds"][0]
            and row["repetition"] == 0 and wi == 1
            and ((row["prefix_tokens"] == config.get("snapshot_short_prefix")
                  and row["scenario"] in config.get("snapshot_cases", []) and row["variant"] in BATCHED)
                 or (row["prefix_tokens"] == max(shape["prefix_tokens"] for shape in config["shapes"])
                     and row["scenario"] == config.get("snapshot_long_case") and row["variant"] == "sparse_batched")))


def audit_faults(audit, row, window, model_config, dtype, reference, label):
    n, wi = model_config["num_hidden_layers"], window["window_index"]
    prefix = row["prefix_tokens"] + wi * row["window"]
    events = expected_events(row["scenario"], row["policy"], wi, n, row["seed"], row["window"])
    audit.check(len(window["faults"]) == len(events), label + "/fault-count")
    for i, (fault, event) in enumerate(zip(window["faults"], events)):
        target, step, seed, layer, kind, mode = event
        tag = label + "/fault/" + str(i)
        actual = (fault["target"], fault["after_step"], fault["seed"], fault.get("layer"), fault.get("kind"))
        audit.check(actual == event[:5] and fault["window_index"] == wi, tag + "/schedule")
        audit.check(fault["batch"] == 0 and fault["dtype"] == "torch." + dtype, tag + "/batch-dtype")
        width = model_config["hidden_size"] if target == "journal" else model_config["hidden_size"] // model_config["num_attention_heads"]
        for field in ("before", "after"):
            audit.check(len(fault[field]) == (1 if mode == "bitflip" else width)
                        and all(finite(v) for v in fault[field]), tag + "/" + field)
        changed = sum(left != right for left, right in zip(fault["before"], fault["after"]))
        audit.check(changed > 0 and fault["changed_elements"] == changed, tag + "/changed-finite-values")
        if mode == "additive":
            audit.check(fault["fault_type"] == "additive_8rms" and finite(fault["rms"]) and fault["rms"] > 0, tag + "/additive-model")
            audit.check(len(fault["requested_delta"]) == width and all(finite(v) for v in fault["requested_delta"]), tag + "/delta-width-finite")
            for j, (before, delta, after) in enumerate(zip(fault["before"], fault["requested_delta"], fault["after"])):
                audit.check(stored_sum(before, delta, dtype) == after, tag + "/addition/" + str(j))
        else:
            bits = 16 if dtype == "bfloat16" else 32
            audit.check(fault["fault_type"] == "fraction_bit_flip" and fault["width"] == bits
                        and fault["bit"] == 3 and type(fault["feature"]) is int and 0 <= fault["feature"] < width, tag + "/bit-model")
            before, after = stored_bits(fault["before"][0], dtype), stored_bits(fault["after"][0], dtype)
            audit.check(fault["before_bits"] == [before] and fault["after_bits"] == [after]
                        and before ^ after == 1 << 3, tag + "/independent-unsigned-xor")
        if target == "journal":
            eligible = max(c for c in row["cuts"] if c <= 3 * n // 4)
            pending = reference["initial_token"] if wi == 0 else reference["windows"][wi - 1]["tokens"][-1]
            audit.check(fault["cut"] == eligible and fault["journal_step"] == 0
                        and fault["journal_position"] == prefix and fault["journal_token"] == pending
                        and fault["sequence_position"] == 0, tag + "/saved-activation-binding")
        else:
            audit.check(type(fault["head"]) is int and 0 <= fault["head"] < model_config["num_key_value_heads"]
                        and type(fault["prefix_position"]) is int and 0 <= fault["prefix_position"] < prefix, tag + "/old-prefix-coordinate")


def audit_controller_metadata(audit, cm, row, window, model_config, metadata, rejected, label):
    n, w, wi = model_config["num_hidden_layers"], row["window"], window["window_index"]
    size = 2 if metadata["dtype"] == "bfloat16" else 4
    prefix = row["prefix_tokens"] + wi * w
    payload = 2 * n * model_config["num_key_value_heads"] * prefix * (model_config["hidden_size"] // model_config["num_attention_heads"]) * size
    journal_row = len(row["cuts"]) * model_config["hidden_size"] * size
    journal_challenge = row["scenario"] == "journal_corruption" and wi == 1 and row["policy"] == "sparse"
    eligible = max([0] + [cut for cut in row["cuts"] if window.get("detected_layers") and cut <= min(window["detected_layers"])])
    validation_rows = (1 if journal_challenge else w) if eligible else 0
    cache_calls = 2 if rejected else 3
    journal_calls = w + validation_rows if row["policy"] == "sparse" else 0
    audit.equal(cm, {"window_index": wi, "prefix_tokens": prefix, "window_tokens": w,
        "checkpoint_bytes": payload, "preserved_committed_cache_bytes": payload, "journal_bytes": w * journal_row,
        "seal_mode": row["seal_mode"], "profile_enabled": row["profile"],
        "seal_calls": cache_calls + journal_calls, "seal_logical_bytes": cache_calls * payload + journal_calls * journal_row,
        "journal_integrity_checked": bool(eligible)}, label + "/accounting")
    gpu = metadata["device"] == "cuda"
    transfers = ((cache_calls * 2 * n + journal_calls * len(row["cuts"])) if row["seal_mode"] == "legacy"
                 else cache_calls + journal_calls) if gpu else 0
    batch = max(payload if row["seal_mode"] == "batched" else payload // (2 * n),
                journal_row if row["seal_mode"] == "batched" else model_config["hidden_size"] * size if row["cuts"] else 0) if gpu else 0
    audit.equal(cm, {"seal_transfer_calls": transfers, "seal_transfer_bytes": cache_calls * payload + journal_calls * journal_row if gpu else 0,
                     "seal_peak_batch_bytes": batch}, label + "/transfer-counts")
    audit.check(finite(cm["replay_seconds"]) and 0 <= cm["replay_seconds"] <= window["wall_seconds"], label + "/replay-time")
    if row["profile"]:
        timings = cm["profile_seconds"]
        required = {"working_clone", "checkpoint_clone", "checkpoint_initial_seal", "speculation", "checkpoint_validation_seal"}
        if not rejected:
            required |= {"prefix_detection", "precommit_finite", "checkpoint_final_seal"}
        audit.check(required <= set(timings) and all(finite(v) and v >= 0 for v in timings.values()), label + "/profile-components")
    else:
        audit.check("profile_seconds" not in cm, label + "/unprofiled-primary-path")


def audit_session(audit, row, reference, meta, label):
    config, mc, dtype = meta["config"], meta["model_config"], meta["dtype"]
    n, w = mc["num_hidden_layers"], row["window"]
    policy, seal, cuts = expected_parameters(row["variant"], n)
    protected = policy != "unprotected"
    audit.check((row["policy"], row["seal_mode"], row["cuts"]) == (policy, seal, cuts) and row["num_layers"] == n, label + "/variant-parameters")
    audit.check(row["valid"] is True and row["same_policy_treatment"] is (row["scenario"] != "journal_corruption"), label + "/valid-treatment")
    audit.check(row["n_windows"] == config["n_windows"], label + "/window-contract")
    audit.check(protected or row["scenario"] == "clean", label + "/unprotected-is-clean-cost-only")
    abort = row["scenario"] == "checkpoint_corruption"
    attempted, commits = (2, 1) if abort else (row["n_windows"], row["n_windows"])
    audit.check(len(row["windows"]) == attempted and [v["window_index"] for v in row["windows"]] == list(range(attempted)), label + "/window-coverage")
    audit.check(row["status"] == ("expected_abort" if abort else "complete") and row["poisoned"] is abort
                and row["windows_committed"] == commits, label + "/termination")
    audit.check(row["committed_tokens"] == [t for win in reference["windows"][:commits] for t in win["tokens"]], label + "/public-transcript")
    audit.equal(row["total_wall_seconds"], sum(v["wall_seconds"] for v in row["windows"]), label + "/wall-sum")
    audit.check(finite(row["session_initialization_seconds"]) and row["session_initialization_seconds"] > 0, label + "/session-initialization-time")
    audit.equal(row["total_with_initialization_seconds"], row["total_wall_seconds"] + row["session_initialization_seconds"], label + "/including-initialization-sum")
    audit.equal(row["total_injection_harness_seconds"], sum(v["injection_harness_seconds"] for v in row["windows"]), label + "/harness-sum")
    for wi, window in enumerate(row["windows"]):
        tag = label + "/window/" + str(wi)
        prefix = row["prefix_tokens"] + wi * w
        payload = 2 * n * mc["num_key_value_heads"] * prefix * (mc["hidden_size"] // mc["num_attention_heads"]) * (2 if dtype == "bfloat16" else 4)
        journal_payload = w * len(cuts) * mc["hidden_size"] * (2 if dtype == "bfloat16" else 4)
        rejected = abort and wi == 1
        audit.check(window["valid"] is True and window["no_early_commit"] is True, tag + "/valid-buffered")
        audit.check(window["logical_input_cache_bytes"] == payload and window["logical_checkpoint_bytes"] == (payload if protected else 0)
                    and window["logical_journal_bytes"] == journal_payload, tag + "/payload-accounting")
        for field in ("wall_seconds", "wall_minus_injection_seconds"):
            audit.check(finite(window[field]) and window[field] > 0, tag + "/" + field)
        audit.check(finite(window["injection_harness_seconds"]) and window["injection_harness_seconds"] >= 0, tag + "/harness-time")
        audit.equal(window["wall_minus_injection_seconds"], window["wall_seconds"] - window["injection_harness_seconds"], tag + "/diagnostic-subtraction")
        steps = window["step_completion_seconds"]
        audit.check(len(steps) == w and all(finite(v) and 0 < v <= window["wall_seconds"] for v in steps)
                    and all(a <= b for a, b in zip(steps, steps[1:])), tag + "/step-times")
        if rejected:
            audit.check(window["release_seconds"] is None and window["token_availability_seconds"] == [], tag + "/no-rejected-release")
        else:
            audit.equal(window["release_seconds"], window["wall_seconds"], tag + "/window-release")
            audit.equal(window["token_availability_seconds"], [window["wall_seconds"]] * w if protected else steps, tag + "/availability-semantics")
        for field in ("allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes"):
            audit.check(type(window[field]) is int and window[field] >= 0, tag + "/" + field)
        audit.check(window["peak_increment_bytes"] == window["peak_allocated_bytes"] - window["allocated_before_bytes"], tag + "/allocator-arithmetic")
        if meta["device"] == "cpu":
            audit.check(window["peak_allocated_bytes"] == window["allocated_before_bytes"] == 0, tag + "/no-cpu-allocator-claim")
        audit_faults(audit, row, window, mc, dtype, reference, tag)
        audit.check(window["committed_before"] == wi * w, tag + "/committed-before")
        if rejected:
            audit.check(window["status"] == "rejected" and window["committed_after"] == wi * w, tag + "/abort-commit-state")
            audit.equal(window["checks"], {key: True for key in ("expected_rejection", "poisoned", "committed_tokens_unchanged",
                "committed_windows_unchanged", "committed_cache_unchanged", "committed_logits_unchanged", "next_token_unchanged", "no_early_commit")}, tag + "/abort-verdicts")
            rejection = window["rejection"]
            audit.check(rejection["type"] == "WindowRejected" and rejection["reason"] == "Checkpoint integrity failure", tag + "/exact-refusal")
            cm = rejection["controller_metadata"]
            audit.check(cm["route"] == "rejected" and cm["replay_layer_steps"] == 0, tag + "/no-abort-replay")
            audit.check("tokens" not in window and "selected_cut" not in window, tag + "/no-abort-result")
            audit_controller_metadata(audit, cm, row, window, mc, meta, True, tag)
            continue
        audit.check(window["status"] == "committed" and window["committed_after"] == (wi + 1) * w, tag + "/whole-window-commit")
        audit.check(window["tokens"] == reference["windows"][wi]["tokens"] and len(window["tokens"]) == w, tag + "/gold-tokens")
        checks = window["checks"]
        for field in ("cache", "logits", "continuation_cache", "continuation_logits"):
            audit.comparison(checks[field], tag + "/" + field, cache="cache" in field)
        for field in ("tokens_equal", "committed_tokens_equal", "next_token_equal", "continuation_tokens_equal", "not_poisoned", "all_equal",
                      "no_early_commit", "committed_windows_equal", "detected_layers_equal_injection", "recovery_expected", "unexpected_checkpoint_commit"):
            audit.check(checks[field] is True, tag + "/verdict/" + field)
        detected = sorted({f["layer"] for f in window["faults"] if f["target"] == "working_prefix"})
        audit.check(window["detected_layers"] == detected and window["recovered"] is bool(detected), tag + "/detected-layers")
        eligible = max([0] + [cut for cut in cuts if detected and cut <= min(detected)])
        journal_challenge = row["scenario"] == "journal_corruption" and wi == 1 and policy == "sparse"
        selected, fallback = (0, "journal_integrity_failure") if journal_challenge else (eligible, None)
        audit.check(window["selected_cut"] == selected and window["fallback_reason"] == fallback, tag + "/detector-cut")
        speculative = window["speculative_tokens"]
        audit.check(len(speculative) == w and all(type(t) is int and 0 <= t < mc["vocab_size"] for t in speculative), tag + "/speculation-record")
        divergence = next((i + 1 for i, (a, b) in enumerate(zip(window["tokens"], speculative)) if a != b), None)
        audit.check(window["first_divergence"] == divergence, tag + "/first-divergence")
        starts = [selected if divergence is None or i < divergence else 0 for i in range(w)] if detected else []
        audit.check(window["replay_starts"] == starts, tag + "/irreversible-replay-trace")
        cm = window["controller_metadata"]
        if protected:
            route = "journal_fallback" if fallback else ("sparse" if selected else "full") if detected else "clean"
            audit.check(cm["route"] == route and cm["replay_layer_steps"] == sum(n - cut for cut in starts)
                        and (cm["replay_seconds"] > 0) is bool(detected), tag + "/route-layer-work")
            audit_controller_metadata(audit, cm, row, window, mc, meta, False, tag)
        else:
            audit.equal(cm, {"route": "unprotected", "profile_enabled": False}, tag + "/unprotected-cost-baseline")
        if journal_challenge:
            audit.check(checks["journal_corruption_fell_back"] is True, tag + "/guard-fallback-verdict")


def audit_snapshot(audit, path, identity, reference, model_config, dtype, label):
    import torch
    archive = torch.load(path, map_location="cpu", weights_only=True)
    audit.check(isinstance(archive, dict) and archive["schema_version"] == 1, label + "/safe-schema")
    audit.equal(archive["identity"], identity, label + "/identity")
    recovered, gold = archive["recovered"], archive["reference"]
    wi = identity["window_index"]
    expected = reference["windows"][wi]
    checked_tensors, checked_bytes, bytewise_pairs = 0, 0, 0

    def tensor(actual, native, fingerprint, tag):
        nonlocal checked_tensors, checked_bytes, bytewise_pairs
        digests = []
        for role, value in (("recovered", actual), ("reference", native)):
            audit.check(type(value) is torch.Tensor and value.device.type == "cpu"
                        and list(value.shape) == fingerprint["shape"]
                        and str(value.dtype) == fingerprint["dtype"], tag + "/" + role + "/layout")
            audit.check(bool(torch.isfinite(value).all()), tag + "/" + role + "/finite")
            payload = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            audit.check(len(payload) == fingerprint["bytes"], tag + "/" + role + "/byte-count")
            if role == "reference":
                audit.check(digest == fingerprint["sha256"], tag + "/reference-byte-hash")
            digests.append(digest)
            checked_tensors += 1
            checked_bytes += len(payload)
        audit.check(torch.equal(actual, native), tag + "/independent-loaded-exact-equality")
        # torch.equal intentionally accepts +0 == -0 under the declared finite
        # elementwise contract. Record stricter raw-byte equality descriptively,
        # but do not silently strengthen that contract only for archived states.
        bytewise_pairs += digests[0] == digests[1]

    def cache(actual, native, fingerprints, tag):
        audit.check(actual["seen_tokens"] == native["seen_tokens"] == fingerprints["seen_tokens"], tag + "/seen-tokens")
        for field in ("keys", "values"):
            audit.check(len(actual[field]) == len(native[field]) == len(fingerprints[field]) == model_config["num_hidden_layers"], tag + "/" + field + "/layers")
            for i, (left, right, digest) in enumerate(zip(actual[field], native[field], fingerprints[field])):
                tensor(left, right, digest, tag + "/" + field + "/" + str(i))

    for field in ("tokens", "continuation_tokens"):
        audit.check(recovered[field] == gold[field] == expected[field], label + "/" + field)
    audit.check(recovered["next_token"] == gold["next_token"] == expected["tokens"][-1], label + "/next-token")
    for prefix in ("", "continuation_"):
        cache(recovered[prefix + "cache"], gold[prefix + "cache"], expected[prefix + "cache"], label + "/" + prefix + "cache")
        tensor(recovered[prefix + "logits"], gold[prefix + "logits"], expected[prefix + "logits"], label + "/" + prefix + "logits")
    return {"file": str(path.name), "identity": identity, "tensors_reloaded": checked_tensors,
            "tensor_payload_bytes_checked": checked_bytes, "bytewise_equal_tensor_pairs": bytewise_pairs,
            "tensor_pairs": checked_tensors // 2, "status": "passed"}


def validate(audit, directory, config_path, manifest_path, with_snapshots=True):
    meta, config, manifest = load(directory / "metadata.json"), load(config_path), load(manifest_path)
    raw = {phase: records(directory / (phase + ".jsonl")) for phase in ("sessions", "references", "warmups", "profiles")}
    hashes = {name + ".jsonl": sha(directory / (name + ".jsonl")) for name in raw}
    hashes.update(metadata=sha(directory / "metadata.json"), config=sha(config_path), manifest=sha(manifest_path), auditor=sha(__file__))
    audit.check(meta["schema_version"] == 1 and meta["status"] == "complete", "completed-schema")
    audit.check(meta["config"] == config and meta["manifest"] == manifest, "exact-config-manifest")
    audit.check(meta["config_sha256"] == hashes["config"] and meta["manifest_sha256"] == hashes["manifest"], "input-hash-binding")
    for phase in raw:
        audit.check(meta[phase + "_sha256"] == hashes[phase + ".jsonl"], phase + "/file-hash")
    for field in ("timing_scope", "adjusted_timing_scope", "availability_scope", "memory_scope", "oracle_storage",
                  "prefill_scope", "initialization_timing_scope", "python", "torch"):
        audit.check(bool(meta[field]), "metadata/" + field)
    audit.check(meta["transformers"] == "4.51.3" and meta["dtype"] in ("bfloat16", "float32")
                and meta["device"] in ("cuda", "cpu") and meta["deterministic_algorithms"] is True
                and meta["tf32"] is False and meta["attention"] == "eager" and meta["batch_size"] == 1, "numeric-environment")
    audit.check(finite(meta["started_unix_seconds"]) and finite(meta["finished_unix_seconds"])
                and meta["finished_unix_seconds"] > meta["started_unix_seconds"], "completed-time")
    required_sources = {"recut_controller.py", "run_study.py", "audit_study.py", "analyze_study.py",
                        "../experiments/recut_engine.py", "../experiments/prefix_detector.py", "../experiments/run_recut.py"}
    audit.check(required_sources <= set(meta["source_sha256"]), "source/dependency-coverage")
    for relative, expected in meta["source_sha256"].items():
        path = (HERE / relative).resolve()
        audit.check(path.parent in (HERE, HERE.parent / "experiments") and path.suffix == ".py", "source/safe-path/" + relative)
        audit.check(sha(path) == expected, "source/frozen-hash/" + relative)
    audit.check(config["policies"] == ["full", "sparse", "allcuts"]
                and set(config["legacy_cases"]) <= {"clean", "value_late"} and "clean" in config["legacy_cases"], "config/variant-protocol")
    for field in ("models", "seeds", "fault_cases", "guard_cases"):
        audit.check(bool(config[field]) and len(config[field]) == len(set(config[field])), "config/unique/" + field)
    audit.check(type(config["n_windows"]) is int and config["n_windows"] >= 2
                and all(type(config[k]) is int and config[k] > 0 for k in ("clean_repetitions", "repetitions")), "config/counts")
    model = meta["model"]
    models = {item["model"]: item for item in manifest}
    audit.check(len(models) == len(manifest) and model in config["models"] and model in models, "selected-model-binding")
    audit.check(re.fullmatch(r"[0-9a-f]{40}", models[model]["revision"]) is not None
                and meta["model_revision"] == models[model]["revision"], "immutable-model-revision")
    files = meta["model_files"]
    audit.check(bool(files) and len({f["file"] for f in files}) == len(files), "model-files/coverage")
    for record in files:
        path = Path(record["file"])
        audit.check(not path.is_absolute() and ".." not in path.parts and type(record["bytes"]) is int and record["bytes"] > 0
                    and SHA256.fullmatch(record["sha256"]) is not None, "model-files/structure/" + record["file"])
    mc, dtype = meta["model_config"], meta["dtype"]
    workloads = {}
    for prompt, shape in itertools.product(config["prompts"], config["shapes"]):
        prefix, window = shape["prefix_tokens"], shape["window"]
        identifier = f"{model.replace('/', '--')}--{prompt['id']}--p{prefix}--w{window}"
        audit.check(identifier not in workloads, "unique-workload/" + identifier)
        workloads[identifier] = (prompt["id"], prefix, window)
    refs = {record["workload_id"]: record for record in raw["references"]}
    audit.check(set(refs) == set(workloads) and len(refs) == len(raw["references"]), "complete-reference-matrix")
    markers = {record["id"]: record for record in meta["workloads"]}
    audit.check(set(markers) == set(workloads) and len(markers) == len(meta["workloads"]), "complete-metadata-workloads")
    for identifier, ref in refs.items():
        prompt, prefix, window = workloads[identifier]
        audit.check(ref["model"] == model and ref["revision"] == models[model]["revision"] and ref["prompt_id"] == prompt, identifier + "/reference-binding")
        audit.check(len(ref["prefix_ids"]) == prefix and all(type(t) is int and 0 <= t < mc["vocab_size"]
                    for t in ref["prefix_ids"] + [ref["initial_token"]]), identifier + "/prefix-layout")
        positions = list(range(prefix, prefix + config["n_windows"] * window + 2))
        audit.check([p["position"] for p in ref["parity"]] == positions, identifier + "/post-prefill-parity-coverage")
        audit.equal(markers[identifier], {"prompt_id": prompt, "prefix_tokens": prefix, "window": window,
                                        "passed": True, "parity_positions": len(positions)}, identifier + "/workload-marker")
        for record in ref["parity"]:
            audit.comparison(record["cache"], identifier + "/parity-cache", cache=True)
            audit.comparison(record["logits"], identifier + "/parity-logits")
            audit.check(record["token_equal"] is True, identifier + "/parity-token")
        cache_fingerprint(audit, ref["checkpoint"], mc, prefix, dtype, identifier + "/checkpoint")
        audit.check([win["window_index"] for win in ref["windows"]] == list(range(config["n_windows"])), identifier + "/reference-window-coverage")
        for wi, record in enumerate(ref["windows"]):
            tag = identifier + "/reference/" + str(wi)
            audit.check(len(record["tokens"]) == window and len(record["continuation_tokens"]) == 2
                        and all(type(t) is int and 0 <= t < mc["vocab_size"] for t in record["tokens"] + record["continuation_tokens"]), tag + "/tokens")
            for continuation in (False, True):
                pre = "continuation_" if continuation else ""
                cache_fingerprint(audit, record[pre + "cache"], mc, prefix + (wi + 1) * window + (2 if continuation else 0), dtype, tag + "/" + pre + "cache")
                tensor_fingerprint(audit, record[pre + "logits"], (1, 1, mc["vocab_size"]), dtype, tag + "/" + pre + "logits")
    snapshots, phases, target_counts, type_counts, divergences = [], {}, Counter(), Counter(), Counter()
    for phase in ("sessions", "warmups", "profiles"):
        expected = expected_matrix(config, workloads, phase)
        keys = [(r["workload_id"], r["scenario"], r["variant"], r["seed"], r["repetition"]) for r in raw[phase]]
        audit.check(len(keys) == len(set(keys)) == len(expected) and set(keys) == set(expected), phase + "/complete-unique-matrix")
        audit.check(meta[phase + "_completed"] == len(expected), phase + "/completed-count")
        if phase == "sessions":
            audit.check(meta["expected_sessions"] == len(expected), "sessions/prespecified-count")
        pairs, repeated = defaultdict(list), {}
        for key, row in zip(keys, raw[phase]):
            identifier, scenario, variant, seed, rep = key
            label = phase + "/" + "/".join(map(str, key))
            prompt, prefix, window = workloads[identifier]
            audit.check((row["model"], row["prompt_id"], row["prefix_tokens"], row["window"]) == (model, prompt, prefix, window), label + "/workload-binding")
            audit.check(row["model_revision"] == models[model]["revision"] and row["source_sha256"] == meta["source_sha256"]
                        and row["config_sha256"] == hashes["config"], label + "/source-input-binding")
            audit.check(row["method_order"] == expected[key] and row["profile"] is (phase == "profiles"), label + "/order-and-profile-separation")
            audit_session(audit, row, refs[identifier], meta, label)
            pairs[(identifier, scenario, seed, rep)].append(row)
            faults = [win["faults"] for win in row["windows"]]
            rep_key = (identifier, scenario, variant, seed)
            if rep_key in repeated:
                audit.check(repeated[rep_key] == faults, label + "/repeated-identical-perturbations")
            else:
                repeated[rep_key] = faults
            for wi, win in enumerate(row["windows"]):
                want_snapshot = phase == "sessions" and expected_snapshot(config, row, wi)
                audit.check(("snapshot" in win) is want_snapshot, label + "/snapshot-selection/" + str(wi))
                if want_snapshot:
                    snapshot = win["snapshot"]
                    path = relative_file(audit, directory, snapshot["file"], label + "/snapshot-path")
                    audit.check(path.stat().st_size == snapshot["bytes"] and sha(path) == snapshot["sha256"], label + "/snapshot-file-hash")
                    identity = {field: row[field] for field in ("workload_id", "model", "scenario", "seed", "repetition", "variant")}
                    identity["window_index"] = wi
                    snapshots.append((path, identity, refs[identifier], snapshot))
                if phase == "sessions":
                    target_counts.update(f["target"] for f in win["faults"])
                    type_counts.update(f["fault_type"] for f in win["faults"])
                    if win.get("recovered"):
                        divergences[str(win["first_divergence"])] += 1
        for key, group in pairs.items():
            first = group[0]
            for row in group[1:]:
                for left, right in zip(first["windows"], row["windows"]):
                    audit.check([f for f in left["faults"] if f["target"] != "journal"] ==
                                [f for f in right["faults"] if f["target"] != "journal"], phase + "/" + str(key) + "/fair-common-faults")
                    audit.check(left["status"] == right["status"], phase + "/" + str(key) + "/paired-status")
                    if key[1] != "journal_corruption":
                        audit.check(left["faults"] == right["faults"], phase + "/" + str(key) + "/identical-full-treatment")
        windows = [win for row in raw[phase] for win in row["windows"]]
        phases[phase] = {"sessions": len(raw[phase]), "windows_attempted": len(windows),
                         "windows_committed": sum(win["status"] == "committed" for win in windows),
                         "windows_rejected": sum(win["status"] == "rejected" for win in windows),
                         "recovered_windows": sum(bool(win.get("recovered")) for win in windows)}
    audit.check(len({str(path) for path, _, _, _ in snapshots}) == len(snapshots), "snapshot/unique-files")
    actual_files = set((directory / "snapshots").glob("*.pt")) if (directory / "snapshots").exists() else set()
    audit.check({p.resolve() for p in actual_files} == {path for path, _, _, _ in snapshots}, "snapshot/exact-file-coverage")
    snapshot_results = []
    if with_snapshots:
        for path, identity, reference, declaration in snapshots:
            item = audit_snapshot(audit, path, identity, reference, mc, dtype, "snapshot/" + path.name)
            item.update(sha256=declaration["sha256"], bytes=declaration["bytes"])
            snapshot_results.append(item)
    return {"model": model, "hashes": hashes, "counts": {"workloads": len(workloads), **phases,
            "native_parity_positions": sum(len(ref["parity"]) for ref in refs.values()),
            "snapshot_archives_declared": len(snapshots), "snapshot_archives_reloaded": len(snapshot_results),
            "measured_fault_targets": dict(target_counts), "measured_fault_types": dict(type_counts),
            "measured_first_divergence_histogram": dict(divergences)},
            "snapshot_results": snapshot_results, "tensor_reload_requested": with_snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config_study.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "model_manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--records-only", action="store_true", help="Explicitly skip Torch tensor reload; report is record-only.")
    args = parser.parse_args()
    audit, result = Audit(), {}
    try:
        result = validate(audit, args.directory, args.config, args.manifest, not args.records_only)
    except Exception as error:
        if not audit.failures:
            audit.failures.append(type(error).__name__ + ": " + str(error))
    result.update(version=VERSION, status="passed" if not audit.failures else "failed", checks=audit.checks,
                  failures=audit.failures, audited_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=["No model re-execution or external replication.",
                    "Unsaved trials rely on recorded live native-comparison and no-early-commit verdicts.",
                    "Snapshot reload checks only the prespecified subset; hashes bind saved tensors to native-reference records.",
                    "Gaussian draws and tokenizer outputs are not independently regenerated.",
                    "Fault arithmetic, placement, timing, and same-treatment pairing are checked, not physical fault realism.",
                    "Model-file hash records are checked for structure, not reloaded against remote model weights.",
                    "Prefix detector's bounded contract excludes suffix-only, reverted, computation-origin, and pre-checkpoint faults.",
                    "No production-service, field-failure-rate, throughput, external novelty, or publication-readiness guarantee."])
    output = args.output or args.directory / "audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks", "failures")}, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
