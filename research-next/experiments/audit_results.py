"""Independent audit of RECUT raw records; never imports the runner/analyzer.

Usage: python audit_results.py RUN --config CONFIG --manifest MODEL_MANIFEST
       [--source-dir EXPERIMENTS] [--require-summary]

Only RUN/independent_audit.json is generated. Raw inputs are never modified.
Tensor archives must be declared, hash-matched local files, and are loaded
using torch.load(weights_only=True, map_location='cpu'). The audit checks
record consistency and saved-state equality, not physical fault incidence or
independent model execution. Continuation tensors are not in the pilot archive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import torch


VALID = ("full", "sparse", "all")
ALL_METHODS = ("none", "slotonly", *VALID)
NORMAL_MODES = ("none", "sparse", "all")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first_difference(left, right):
    return next((i + 1 for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty fault subset")
    result = {}
    for name, fraction in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1)):
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


class Audit:
    def __init__(self):
        self.groups = defaultdict(lambda: {"passed": 0, "failed": 0})
        self.failures = []
        self.warnings = []

    def check(self, condition, group, location, detail=""):
        success = bool(condition)
        self.groups[group]["passed" if success else "failed"] += 1
        if not success:
            self.failures.append({"check": group, "location": location, "detail": str(detail)})
        return success

    def equal(self, actual, expected, group, location):
        return self.check(actual == expected, group, location,
                          f"actual={actual!r}; expected={expected!r}")

    def compare_subset(self, actual, expected, location):
        """Independently calculated numerical summary, with explicit roundoff slack."""
        if isinstance(expected, dict):
            if not self.check(isinstance(actual, dict), "analysis_crosscheck", location, "Missing mapping"):
                return
            for key, value in expected.items():
                self.compare_subset(actual.get(key), value, location + "/" + str(key))
        elif isinstance(expected, float):
            self.check(finite_number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                       "analysis_crosscheck", location, f"actual={actual!r}; recomputed={expected!r}")
        else:
            self.equal(actual, expected, "analysis_crosscheck", location)


def check_exact_record(audit, checks, location, require_exact=False):
    component_keys = ("cache", "logits", "continuation_cache", "continuation_logits")
    flags = []
    for key in component_keys:
        record = checks[key]
        audit.check(type(record.get("equal")) is bool, "verification_schema", location + "/" + key)
        audit.check(type(record.get("finite")) is bool, "verification_schema", location + "/" + key + "/finite")
        flags.append(record["equal"])
        if record["equal"]:
            audit.equal(record["finite"], True, "exactness_flags", location + "/" + key + "/finite")
            audit.equal(record["max_abs_error"], 0, "exactness_flags", location + "/" + key + "/error")
            if "mismatches" in record:
                audit.equal(record["mismatches"], [], "exactness_flags", location + "/" + key + "/mismatches")
        if "eq" in record:
            audit.equal(record["eq"], record["equal"], "exactness_flags", location + "/" + key + "/alias")
    for key in ("tokens_equal", "continuation_tokens_equal", "all_equal"):
        audit.check(type(checks.get(key)) is bool, "verification_schema", location + "/" + key)
    flags.extend((checks["tokens_equal"], checks["continuation_tokens_equal"]))
    audit.equal(checks["all_equal"], all(flags), "exactness_flags", location + "/aggregate")
    if require_exact:
        audit.equal(checks["all_equal"], True, "valid_recovery_exact", location)


def audit_work(audit, check, row, method, location):
    n, window = row["num_layers"], row["case"]["window"]
    cut = {"full": 0, "sparse": row["selected_sparse_cut"], "all": row["layer"]}[method]
    divergence = first_difference(row["reference_tokens"], row["faulty_tokens"])
    count = divergence if divergence is not None else window
    starts = [cut] * count + [0] * (window - count)
    audit.equal(check["cut"], cut, "recovery_cut", location)
    audit.check(0 <= cut <= row["layer"], "recovery_cut", location + "/clean")
    audit.equal(check["first_divergence"], divergence, "divergence", location)
    audit.equal(check["start_layers"], starts, "irreversible_replay_trace", location)
    expected_work = n * window - cut * min(count, window)
    audit.equal(check["layer_steps"], expected_work, "layer_work", location + "/formula")
    audit.equal(check["layer_steps"], sum(n - s for s in check["start_layers"]),
                "layer_work", location + "/trace")


def audit_injection(audit, row, model_config, location):
    case, fault = row["case"], row["fault"]
    heads = model_config["num_key_value_heads"]
    head_dim = model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]
    for key, expected in (("kind", case["kind"]), ("fault", case["fault"]), ("layer", row["layer"]), ("batch", 0)):
        audit.equal(fault[key], expected, "injection_metadata", location + "/" + key)
    before, after, delta = fault["before"], fault["after"], fault["requested_delta"]
    for name, vector in (("before", before), ("after", after), ("delta", delta)):
        audit.equal(len(vector), head_dim, "injection_shape", location + "/" + name)
        audit.check(all(finite_number(v) for v in vector), "injection_finite", location + "/" + name)
    audit.check(finite_number(fault["rms"]) and fault["rms"] >= 0, "injection_finite", location + "/rms")
    changed = sum(a != b for a, b in zip(before, after))
    audit.equal(fault["changed_elements"], changed, "injection_changed_count", location)
    rng = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
    expected_coordinates = {
        "head": int(torch.randint(heads, (), generator=rng)),
        "prefix_position": int(torch.randint(case["prefix_tokens"], (), generator=rng)),
        "scalar_dimension": int(torch.randint(head_dim, (), generator=rng)),
    }
    for key, value in expected_coordinates.items():
        audit.equal(fault[key], value, "injection_seed_coordinates", location + "/" + key)
    expected_delta = torch.zeros(head_dim, dtype=torch.float32)
    if case["fault"] == "scalar":
        expected_delta[expected_coordinates["scalar_dimension"]] = 8 * fault["rms"] * torch.randn((), generator=rng)
        audit.check(changed <= 1, "injection_changed_count", location + "/scalar")
    elif case["fault"] == "vector":
        expected_delta = 8 * fault["rms"] * torch.randn(head_dim, generator=rng)
    elif case["fault"] == "noop":
        audit.equal(changed, 0, "noop_injection", location)
        audit.equal(before, after, "noop_injection", location + "/values")
    else:
        audit.check(False, "injection_metadata", location, "Unknown fault family")
    # Gaussian RNG kernels can differ across PyTorch versions/CPU platforms.
    # The actual recorded add/round operation below is an exact audit target;
    # cross-platform regeneration of random normals is only corroboration.
    if delta != expected_delta.tolist():
        warning = ("CPU Gaussian-RNG replay differed from recorded delta on this auditor runtime; "
                   "this is not treated as an experiment failure. Recorded delta shape, finite values, "
                   "fault-family constraints and exact add/BF16-rounding are still checked.")
        if warning not in audit.warnings:
            audit.warnings.append(warning)
    if case["fault"] == "scalar":
        audit.check(all(v == 0 for i, v in enumerate(delta) if i != fault["scalar_dimension"]),
                    "injection_delta_scope", location)
    elif case["fault"] == "noop":
        audit.check(all(v == 0 for v in delta), "injection_delta_scope", location)
    # GPU recorded operation is a float32 add then BF16 rounding, not a GEMM.
    rounded = (torch.tensor(before, dtype=torch.float32) + torch.tensor(delta, dtype=torch.float32)).to(torch.bfloat16).float()
    audit.equal(after, rounded.tolist(), "injection_rounding", location)


def audit_row(audit, row, expected_case, config, record, model_config):
    location = row["model"] + "/" + row["case_id"]
    audit.equal(row["case"], expected_case, "case_config", location)
    audit.equal(row["revision"], record["revision"], "model_revision", location)
    n, window, prefix = row["num_layers"], expected_case["window"], expected_case["prefix_tokens"]
    audit.equal(n, model_config["num_hidden_layers"], "model_layers", location)
    layer = int(float(expected_case["layer_fraction"]) * (n - 1))
    sparse = sorted({cut for cut in (n // 4, n // 2, 3 * n // 4) if 0 < cut < n})
    selected = max([0] + [cut for cut in sparse if cut <= layer])
    audit.equal(row["layer"], layer, "layer_selection", location)
    audit.equal(row["sparse_cuts"], sparse, "layer_selection", location + "/sparse")
    audit.equal(row["selected_sparse_cut"], selected, "layer_selection", location + "/selected")
    audit.equal(len(row["prefix_ids"]), prefix, "token_lengths", location + "/prefix")
    for name in ("reference_tokens", "faulty_tokens"):
        audit.equal(len(row[name]), window, "token_lengths", location + "/" + name)
    tokens = row["prefix_ids"] + row["reference_tokens"] + row["faulty_tokens"] + [row["initial_token"]]
    audit.check(all(type(t) is int and 0 <= t < model_config["vocab_size"] for t in tokens),
                "token_values", location)
    divergence = first_difference(row["reference_tokens"], row["faulty_tokens"])
    audit.equal(row["first_divergence"], divergence, "divergence", location + "/row")
    audit_injection(audit, row, model_config, location)
    audit.equal(set(row["methods"]), set(ALL_METHODS), "method_coverage", location)
    for name in ALL_METHODS:
        check = row["methods"][name]
        check_exact_record(audit, check, location + "/" + name,
                           require_exact=name in VALID or expected_case["fault"] == "noop")
        audit.equal(check["tokens_equal"], check["tokens"] == row["reference_tokens"],
                    "token_verification", location + "/" + name)
        audit.equal(check["continuation_tokens_equal"],
                    check["continuation_tokens"] == row["methods"]["full"]["continuation_tokens"],
                    "token_verification", location + "/" + name + "/continuation")
        audit.equal(len(check["continuation_tokens"]), 2, "token_lengths", location + "/continuation")
        if name in VALID:
            audit_work(audit, check, row, name, location + "/" + name)
    audit.equal(row["methods"]["none"]["tokens"], row["faulty_tokens"], "negative_control", location + "/none")
    audit.equal(row["methods"]["slotonly"]["tokens"], row["faulty_tokens"], "negative_control", location + "/slot")
    audit.equal(row["methods"]["slotonly"]["logits"], row["methods"]["none"]["logits"],
                "negative_control", location + "/slot_stale_logits")
    audit.equal(row["valid_repairs_equal"], True, "valid_recovery_exact", location + "/aggregate")
    audit.check("failure" not in row, "run_completion", location, "Failure record present")

    repetitions = int(config.get("repetitions", 3))
    timings = row["timings"]
    expected_schedule = [(rep, method, list(VALID[rep % 3:] + VALID[:rep % 3]))
                         for rep in range(repetitions)
                         for method in VALID[rep % 3:] + VALID[:rep % 3]]
    audit.equal(len(timings), len(expected_schedule), "timing_repetitions", location)
    for index, (timing, (rep, method, order)) in enumerate(zip(timings, expected_schedule)):
        loc = location + f"/timing/{index}"
        audit.equal((timing["repetition"], timing["method"], timing["order"]),
                    (rep, method, order), "timing_schedule", loc)
        audit.check(finite_number(timing["seconds"]) and timing["seconds"] > 0, "timing_values", loc)
        check_exact_record(audit, timing["checks"], loc, require_exact=True)
        audit_work(audit, timing["checks"], row, method, loc)
        audit.equal(timing["checks"], row["methods"][method], "timing_repeated_checks", loc)
        audit_memory(audit, timing, loc)
    overhead = row["overhead_trials"]
    overhead_reps = int(config.get("overhead_repetitions", 5))
    expected_overhead = [(rep, mode, list(NORMAL_MODES[rep % 3:] + NORMAL_MODES[:rep % 3]))
                         for rep in range(overhead_reps)
                         for mode in NORMAL_MODES[rep % 3:] + NORMAL_MODES[:rep % 3]]
    audit.equal(len(overhead), len(expected_overhead), "overhead_repetitions", location)
    width = model_config["hidden_size"]
    payloads = {"none": 0, "sparse": window * len(sparse) * width * 2,
                "all": window * (n - 1) * width * 2}
    for index, (trial, (rep, mode, order)) in enumerate(zip(overhead, expected_overhead)):
        loc = location + f"/overhead/{index}"
        audit.equal((trial["repetition"], trial["mode"], trial["order"]),
                    (rep, mode, order), "overhead_schedule", loc)
        audit.equal(trial["verified_equal"], True, "overhead_exactness", loc)
        audit.check(finite_number(trial["seconds"]) and trial["seconds"] > 0, "timing_values", loc)
        audit.equal(trial["journal_payload_bytes"], payloads[mode], "payload_bytes", loc)
        audit_memory(audit, trial, loc)
    for mode in ("sparse", "all"):
        audit.equal(row[mode + "_journal_payload_bytes"], payloads[mode], "payload_bytes", location + "/" + mode)
    dimension = model_config.get("head_dim") or width // model_config["num_attention_heads"]
    expected_checkpoint = 2 * n * model_config["num_key_value_heads"] * prefix * dimension * 2
    audit.equal(row["checkpoint_payload_bytes"], expected_checkpoint, "payload_bytes", location + "/checkpoint")
    audit.equal(len(row["checkpoint_copy_seconds"]), 3, "checkpoint_timing", location)
    audit.check(all(finite_number(t) and t > 0 for t in row["checkpoint_copy_seconds"]), "checkpoint_timing", location)


def audit_memory(audit, trial, location):
    before, peak, increment = (trial[k] for k in ("allocated_before_bytes", "peak_allocated_bytes", "peak_increment_bytes"))
    audit.check(all(type(v) is int and v >= 0 for v in (before, peak, increment)), "memory_accounting", location)
    audit.equal(increment, peak - before, "memory_accounting", location + "/difference")


def tensor_relation(left, right):
    layout = left.shape == right.shape and left.dtype == right.dtype
    finite = bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
    equal = layout and finite and torch.equal(left, right)
    error = float((left.double() - right.double()).abs().max()) if layout and finite and left.numel() else (0.0 if equal else None)
    return {"equal": equal, "finite": finite, "max_abs_error": error}


def state_relation(left, right):
    relations = [tensor_relation(a, b) for key in ("keys", "values") for a, b in zip(left[key], right[key])]
    equal = (len(left["keys"]) == len(right["keys"]) and len(left["values"]) == len(right["values"])
             and left["seen_tokens"] == right["seen_tokens"] and all(r["equal"] for r in relations))
    errors = [r["max_abs_error"] for r in relations]
    return {"equal": equal, "finite": all(r["finite"] for r in relations),
            "max_abs_error": max(errors, default=0.0) if all(e is not None for e in errors) else None}


def audit_snapshot(audit, path, archive_record, row, record, model_config):
    location = path.name
    payload = torch.load(path, map_location="cpu", weights_only=True)
    audit.equal(payload["model"], record, "snapshot_metadata", location + "/model")
    expected_episode = {k: v for k, v in row.items() if k not in ("audit_file", "audit_skipped")}
    audit.equal(payload["episode"], expected_episode, "snapshot_metadata", location + "/episode")
    states = payload["states"]
    for key in ("prefix_ids", "initial_token", "fault"):
        audit.equal(states[key], row[key], "snapshot_metadata", location + "/" + key)
    n, prefix, window = row["num_layers"], len(row["prefix_ids"]), row["case"]["window"]
    kv_heads = model_config["num_key_value_heads"]
    head_dim = model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]
    reference = states["reference"]
    for name in ("reference", *ALL_METHODS):
        state = states[name]
        expected_tokens = row["reference_tokens"] if name == "reference" else row["methods"][name]["tokens"]
        audit.equal(state["tokens"], expected_tokens, "snapshot_tokens", location + "/" + name)
        audit.equal(state["seen_tokens"], prefix + window, "snapshot_cache_length", location + "/" + name)
        for kind in ("keys", "values"):
            audit.equal(len(state[kind]), n, "snapshot_layout", location + "/" + name + "/" + kind)
            for layer, tensor in enumerate(state[kind]):
                audit.check(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                            and tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == (1, kv_heads, prefix + window, head_dim),
                            "snapshot_layout", location + f"/{name}/{kind}/{layer}")
        logits = state["logits"]
        audit.check(isinstance(logits, torch.Tensor) and logits.device.type == "cpu" and logits.dtype == torch.bfloat16
                    and tuple(logits.shape) == (1, 1, model_config["vocab_size"]),
                    "snapshot_layout", location + "/" + name + "/logits")
        audit.equal(int(logits[0, -1].argmax()), state["tokens"][-1], "snapshot_logits_decision", location + "/" + name)
        if name in VALID:
            audit.equal(state["start_layers"], row["methods"][name]["start_layers"], "snapshot_replay_trace", location + "/" + name)
            audit.equal(state["first_divergence"], row["first_divergence"], "snapshot_replay_trace", location + "/" + name + "/divergence")
        elif name in ("reference", "none"):
            audit.equal(state["start_layers"], [0] * window, "snapshot_replay_trace", location + "/" + name)
        else:
            audit.equal(state["start_layers"], [], "snapshot_replay_trace", location + "/slotonly")
        cache_relation = state_relation(state, reference)
        logits_relation = tensor_relation(logits, reference["logits"])
        if name == "reference":
            audit.equal(cache_relation["finite"], True, "snapshot_reference_finite", location)
            audit.equal(logits_relation["finite"], True, "snapshot_reference_finite", location + "/logits")
        else:
            for field, relation in (("cache", cache_relation), ("logits", logits_relation)):
                reported = row["methods"][name][field]
                audit.equal(relation["equal"], reported["equal"], "snapshot_vs_row", location + "/" + name + "/" + field)
                audit.equal(relation["finite"], reported["finite"], "snapshot_vs_row", location + "/" + name + "/" + field + "/finite")
                if relation["max_abs_error"] is not None:
                    audit.equal(relation["max_abs_error"], reported["max_abs_error"], "snapshot_vs_row", location + "/" + name + "/" + field + "/error")
            if name in VALID:
                audit.check(cache_relation["equal"] and logits_relation["equal"] and state["tokens"] == reference["tokens"],
                            "snapshot_recovery_exact", location + "/" + name)
    checkpoint_bytes = 0
    for kind in ("keys", "values"):
        checkpoint = states["checkpoint_" + kind]
        audit.equal(len(checkpoint), n, "snapshot_checkpoint", location + "/" + kind)
        for layer, tensor in enumerate(checkpoint):
            checkpoint_bytes += tensor.numel() * tensor.element_size()
            audit.check(tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == (1, kv_heads, prefix, head_dim)
                        and bool(torch.isfinite(tensor).all()), "snapshot_checkpoint", location + f"/{kind}/{layer}")
            for name in ("reference", "slotonly", *VALID):
                audit.check(torch.equal(tensor, states[name][kind][layer][..., :prefix, :]),
                            "snapshot_prefix_preserved", location + f"/{name}/{kind}/{layer}")
            # Reconstruct the sole prefix perturbation from the serialized record.
            expected = tensor.clone()
            fault = row["fault"]
            if layer == row["layer"] and kind == ("keys" if fault["kind"] == "K" else "values"):
                before = tensor[0, fault["head"], fault["prefix_position"]].float().tolist()
                audit.equal(before, fault["before"], "snapshot_injection", location + "/before")
                expected[0, fault["head"], fault["prefix_position"]] = torch.tensor(fault["after"], dtype=tensor.dtype)
            audit.check(torch.equal(expected, states["none"][kind][layer][..., :prefix, :]),
                        "snapshot_injection", location + f"/{kind}/{layer}")
    audit.equal(checkpoint_bytes, row["checkpoint_payload_bytes"], "snapshot_payload", location)
    return {"file": path.name, "model": row["model"], "case_id": row["case_id"],
            "sha256": archive_record["sha256"], "bytes": path.stat().st_size,
            "states_checked": ["reference", *ALL_METHODS], "continuation_tensors_present": False}


def recalculate(rows):
    """Recompute the published descriptive estimands directly from raw samples."""
    faults = [r for r in rows if r["case"]["fault"] != "noop"]
    controls = [r for r in rows if r["case"]["fault"] == "noop"]
    result = {"n": len(rows), "faults": len(faults), "noops": len(controls),
              "distinct_prompts": len({r["case"]["prompt"] for r in rows}),
              "token_divergences": sum(first_difference(r["reference_tokens"], r["faulty_tokens"]) is not None for r in faults),
              "no_actual_change_faults": sum(r["fault"]["changed_elements"] == 0 for r in faults),
              "correctness": {}, "performance": {}, "layers": {}}
    for method in ALL_METHODS:
        result["correctness"][method] = {
            "all_equal_faults": sum(r["methods"][method]["all_equal"] for r in faults),
            "cache_equal_faults": sum(r["methods"][method]["cache"]["equal"] for r in faults),
            "tokens_equal_faults": sum(r["methods"][method]["tokens_equal"] for r in faults),
            "logits_equal_faults": sum(r["methods"][method]["logits"]["equal"] for r in faults),
            "all_equal_controls": sum(r["methods"][method]["all_equal"] for r in controls),
        }
    paired = []
    for row in faults:
        repair = {m: statistics.median([t["seconds"] for t in row["timings"] if t["method"] == m]) for m in VALID}
        normal = {m: statistics.median([t["seconds"] for t in row["overhead_trials"] if t["mode"] == m]) for m in NORMAL_MODES}
        pair = {"case_id": row["case_id"], "model": row["model"], "layer": row["layer"],
                "divergence": row["first_divergence"], "recovery": repair, "normal": normal}
        for method in ("sparse", "all"):
            pair[method + "_speedup"] = repair["full"] / repair[method]
            pair[method + "_overhead"] = normal[method] / normal["none"] - 1
        paired.append(pair)
    if not paired:
        return result, paired
    for method in ("sparse", "all"):
        saved = math.fsum(p["recovery"]["full"] - p["recovery"][method] for p in paired) / len(paired)
        cost = math.fsum(p["normal"][method] - p["normal"]["none"] for p in paired) / len(paired)
        result["performance"][method] = {
            "paired_speedup": percentiles([p[method + "_speedup"] for p in paired]),
            "paired_normal_overhead_fraction": percentiles([p[method + "_overhead"] for p in paired]),
            "recovery_ms": percentiles([p["recovery"][method] * 1000 for p in paired]),
            "mean_recovery_saving_ms": 1000 * saved, "mean_normal_cost_ms": 1000 * cost,
            "diagnostic_break_even_probability": cost / saved if saved > 0 and cost >= 0 else None,
            "journal_bytes": percentiles([r[method + "_journal_payload_bytes"] for r in faults]),
            "journal_over_checkpoint_fraction": percentiles([r[method + "_journal_payload_bytes"] / r["checkpoint_payload_bytes"] for r in faults]),
            "layer_step_fraction": percentiles([r["methods"][method]["layer_steps"] / (r["num_layers"] * r["case"]["window"]) for r in faults]),
        }
    result["full_recovery_ms"] = percentiles([p["recovery"]["full"] * 1000 for p in paired])
    result["normal_nojournal_ms"] = percentiles([p["normal"]["none"] * 1000 for p in paired])
    result["checkpoint_bytes"] = percentiles([r["checkpoint_payload_bytes"] for r in faults])
    result["checkpoint_copy_ms"] = percentiles([statistics.median(r["checkpoint_copy_seconds"]) * 1000 for r in faults])
    for layer in sorted({p["layer"] for p in paired}):
        group = [p for p in paired if p["layer"] == layer]
        result["layers"][str(layer)] = {"n": len(group), "divergences": sum(p["divergence"] is not None for p in group),
            "sparse_speedup": percentiles([p["sparse_speedup"] for p in group]),
            "all_speedup": percentiles([p["all_speedup"] for p in group])}
    return result, paired


def inspect_run(run, config_path, manifest_path, source_dir, require_summary=False):
    audit = Audit()
    files = {"config": config_path, "manifest": manifest_path, "metadata": run / "metadata.json",
             "episodes": run / "episodes.jsonl", "runner": source_dir / "run_recut.py",
             "engine": source_dir / "recut_engine.py", "auditor": Path(__file__).resolve()}
    hashes = {name: sha256(path) for name, path in files.items()}
    config, records, metadata = (read_json(p) for p in (config_path, manifest_path, files["metadata"]))
    raw_lines = files["episodes"].read_text().splitlines()
    audit.check(all(line.strip() for line in raw_lines), "raw_jsonl", "episodes", "Blank lines found")
    rows = [json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))) for line in raw_lines if line.strip()]
    audit.equal(metadata.get("status"), "complete", "run_completion", "metadata")
    audit.check("active_case" not in metadata and "error" not in metadata and "traceback" not in metadata,
                "run_completion", "metadata", "Incomplete/failure fields remain")
    for name in ("config", "manifest", "runner", "engine"):
        audit.equal(metadata.get(name + "_sha256"), hashes[name], "input_sha256", name)
    audit.equal(metadata.get("episodes_sha256"), hashes["episodes"], "input_sha256", "episodes")
    audit.equal(metadata.get("config"), config, "metadata_config", "full_config")
    audit.equal(metadata.get("models"), records, "metadata_manifest", "full_manifest")
    for key in ("python", "torch", "transformers", "cuda", "gpu", "timing_boundary", "memory_scope",
                "allow_bf16_reduced_precision_reduction", "float32_matmul_precision", "cublas_workspace_config"):
        audit.check(key in metadata and metadata[key] is not None, "metadata_fields", key)
    for key, expected in (("transformers", "4.51.3"), ("dtype", "bfloat16"), ("attention", "eager"),
                          ("deterministic_algorithms", True), ("tf32", False),
                          ("equality", "exact elementwise, not byte/bitwise")):
        audit.equal(metadata.get(key), expected, "execution_contract", key)
    case_by_id = {case["id"]: case for case in config["cases"]}
    record_by_model = {r["model"]: r for r in records}
    audit.equal(len(case_by_id), len(config["cases"]), "case_coverage", "unique_config_ids")
    audit.equal(len(record_by_model), len(records), "model_coverage", "unique_manifest_models")
    requested = config.get("models", list(record_by_model))
    audit.check(set(requested) <= set(record_by_model), "model_coverage", "requested")
    expected_pairs = {(model, c["id"]) for model in requested for c in config["cases"] if c.get("model", model) == model}
    expected_models = {m for m, _ in expected_pairs}
    observed = [(row["model"], row["case_id"]) for row in rows]
    audit.equal(len(set(observed)), len(observed), "case_coverage", "unique_observed_pairs")
    audit.equal(set(observed), expected_pairs, "case_coverage", "complete_model_case_grid")
    audit.equal(len(rows), len(expected_pairs), "case_coverage", "row_count")
    audit.equal(set(metadata.get("models_completed", [])), expected_models, "model_coverage", "completed")
    audit.equal(len(metadata.get("models_completed", [])), len(expected_models), "model_coverage", "completed_unique")
    audit.equal(set(metadata.get("model_configs", {})), expected_models, "model_coverage", "configs")
    audit.equal(set(metadata.get("native_parity", {})), expected_models, "model_coverage", "native_parity")
    if "frozen-primary" in config.get("protocol", ""):
        audit.equal(config.get("repetitions"), 3, "primary_protocol", "repair_repetitions")
        audit.equal(config.get("overhead_repetitions"), 5, "primary_protocol", "overhead_repetitions")
        audit.equal(len(config["cases"]), 108, "primary_protocol", "cases")
    parity_positions = {}
    for model in sorted(expected_models):
        record = record_by_model[model]
        audit.check(bool(re.fullmatch(r"[0-9a-fA-F]{40}", record["revision"])), "model_revision", model)
        audit.equal(Path(record["local_path"]).name, record["revision"], "model_revision", model + "/snapshot_path")
        cases = [c for c in config["cases"] if c.get("model", model) == model]
        prefix, tail = max(c["prefix_tokens"] for c in cases), max(c["window"] for c in cases) + 2
        parity = metadata["native_parity"][model]
        parity_positions[model] = prefix + tail
        audit.equal(parity.get("passed"), True, "native_parity", model)
        audit.equal(parity.get("prefix_tokens"), prefix, "native_parity", model + "/prefix")
        audit.equal(parity.get("decode_steps"), tail, "native_parity", model + "/tail")
        audit.equal([t["position"] for t in parity["trace"]], list(range(prefix + tail)), "native_parity", model + "/positions")
        for item in parity["trace"]:
            for kind in ("cache", "logits"):
                check = item[kind]
                audit.check(check["equal"] is True and check["finite"] is True and check["max_abs_error"] == 0,
                            "native_parity", model + f"/{item['position']}/{kind}")
    row_by_pair = dict(zip(observed, rows))
    for row in rows:
        location = row["model"] + "/" + row["case_id"]
        try:
            audit.equal(row.get("config_sha256"), hashes["config"], "row_sha256", location)
            audit_row(audit, row, case_by_id[row["case_id"]], config, record_by_model[row["model"]],
                      metadata["model_configs"][row["model"]])
        except Exception as error:
            audit.check(False, "row_schema_exception", location, f"{type(error).__name__}: {error}")

    snapshot_results, snapshot_bytes = [], 0
    archives = metadata.get("audits", [])
    listed_names = [entry["file"] for entry in archives]
    audit.equal(len(set(listed_names)), len(listed_names), "snapshot_coverage", "unique_files")
    audit.equal({p.name for p in run.glob("*.pt")}, set(listed_names), "snapshot_coverage", "all_owned_pt_files")
    row_audits = {r["audit_file"]: (r["model"], r["case_id"]) for r in rows if "audit_file" in r}
    audit.equal(set(row_audits), set(listed_names), "snapshot_coverage", "row_links")
    budget = config.get("audit_max_bytes", 50 * 1024 * 1024)
    for entry in archives:
        name = entry["file"]
        try:
            path = run / name
            if not audit.check(Path(name).name == name and path.resolve().parent == run.resolve(), "snapshot_safety", name):
                continue
            length = path.stat().st_size
            snapshot_bytes += length
            if not audit.check(0 < length <= budget, "snapshot_safety", name, "Size exceeds configured archive budget"):
                continue
            matched = audit.equal(sha256(path), entry["sha256"], "snapshot_sha256", name)
            audit.equal(length, entry["bytes"], "snapshot_bytes", name)
            pair = (entry["model"], entry["case_id"])
            audit.equal(row_audits.get(name), pair, "snapshot_coverage", name)
            if matched:
                snapshot_results.append(audit_snapshot(audit, path, entry, row_by_pair[pair], record_by_model[pair[0]], metadata["model_configs"][pair[0]]))
        except Exception as error:
            audit.check(False, "snapshot_exception", name, f"{type(error).__name__}: {error}")
    audit.equal(snapshot_bytes, metadata.get("audit_total_bytes"), "snapshot_bytes", "total")
    audit.check(snapshot_bytes <= budget, "snapshot_bytes", "budget")
    if not archives:
        audit.warnings.append("No tensor snapshots saved; tensor checks could not independently corroborate row flags.")

    total, total_paired = recalculate(rows)
    models, all_paired = {}, []
    for model in sorted(expected_models):
        models[model], pairs = recalculate([r for r in rows if r["model"] == model])
        all_paired.extend(pairs)
    analysis_status = "not_available"
    summary_path, paired_path = run / "summary.json", run / "paired_analysis.json"
    if summary_path.exists() and paired_path.exists():
        hashes["summary"], hashes["paired_analysis"] = sha256(summary_path), sha256(paired_path)
        summary, published_pairs = read_json(summary_path), read_json(paired_path)
        audit.equal(summary.get("status"), "complete_bounded_pilot", "analysis_crosscheck", "summary/status")
        audit.equal(summary.get("episode_sha256"), hashes["episodes"], "analysis_crosscheck", "summary/raw_hash")
        audit.compare_subset(summary.get("total"), total, "summary/total")
        audit.equal(set(summary.get("models", {})), expected_models, "analysis_crosscheck", "summary/models")
        for model, result in models.items():
            audit.compare_subset(summary.get("models", {}).get(model), result, "summary/" + model)
        actual_pairs = {(p["model"], p["case_id"]): p for p in published_pairs}
        expected_pairs_map = {(p["model"], p["case_id"]): p for p in all_paired}
        audit.equal(len(actual_pairs), len(published_pairs), "analysis_crosscheck", "paired/unique")
        audit.equal(set(actual_pairs), set(expected_pairs_map), "analysis_crosscheck", "paired/coverage")
        for pair, expected in expected_pairs_map.items():
            audit.compare_subset(actual_pairs.get(pair), expected, "paired/" + "/".join(pair))
        analysis_status = "checked"
    elif require_summary:
        audit.check(False, "analysis_crosscheck", "summary", "summary.json and paired_analysis.json are both required")
    else:
        audit.warnings.append("Summary artifacts not both available; rerun with --require-summary after analysis.")
    return {
        "status": "passed" if not audit.failures else "failed",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_version": "RECUT-independent-audit-v1",
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__},
        "run": str(run.resolve()), "input_sha256": hashes,
        "source_dir": str(source_dir.resolve()),
        "counts": {"rows": len(rows), "expected_rows": len(expected_pairs),
                   "models": len(expected_models), "rows_per_model": dict(Counter(r["model"] for r in rows)),
                   "fault_rows": total["faults"], "noop_rows": total["noops"],
                   "repair_timing_samples": sum(len(r["timings"]) for r in rows),
                   "overhead_samples": sum(len(r["overhead_trials"]) for r in rows),
                   "snapshot_files_declared": len(archives), "snapshot_files_checked": len(snapshot_results),
                   "snapshot_bytes": snapshot_bytes, "native_parity_positions": parity_positions,
                   "checks_passed": sum(g["passed"] for g in audit.groups.values()),
                   "checks_failed": sum(g["failed"] for g in audit.groups.values())},
        "checks": dict(audit.groups), "failures": audit.failures, "warnings": audit.warnings,
        "snapshot_checks": snapshot_results, "analysis_crosscheck": analysis_status,
        "independently_recomputed": {"total": total, "models": models},
        "scope_limits": [
            "No runner, engine or analyzer function was imported to verify raw results or recalculate summaries.",
            "Audit author wrote the engine; this is a separate checking implementation, not an independent-team replication.",
            "Saved tensors corroborate only the archived cases; unsaved cases and native parity are checked from recorded assertions.",
            "Snapshots contain no continuation tensors; recorded continuation flags/tokens are checked but cannot be independently tensor-compared.",
            "No pretrained model was re-executed; archived reference correctness is not proven by agreement alone.",
            "Commit SHA/path checks do not independently authenticate every local model weight file against the upstream repository.",
            "Exact elementwise equality is not bytewise identity; signed-zero representations may differ.",
            "Repeated prompts, timing repetitions, methods and continuations are not independent correctness trials.",
            "Descriptive injected-case measurements and break-even ratios do not estimate real-world fault rates or establish statistical independence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model_manifest.json"))
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-summary", action="store_true")
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("Run directory must already exist; no raw data will be created")
    try:
        report = inspect_run(args.run, args.config, args.manifest, args.source_dir, args.require_summary)
    except Exception as error:
        report = {"status": "failed", "audited_utc": datetime.now(timezone.utc).isoformat(),
                  "auditor_sha256": sha256(__file__), "fatal_error": f"{type(error).__name__}: {error}",
                  "scope_limits": ["Audit could not complete; this is not evidence of a successful experiment."]}
    destination = args.run / "independent_audit.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "counts": report.get("counts"), "fatal_error": report.get("fatal_error")}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
