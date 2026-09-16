"""RECUT bounded experiment runner; raw measurements, not deployment claims."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from recut_engine import (HFIncrementalEngine, cache_nbytes, clone_cache,
                          compare_caches, journal_nbytes, restore_layers,
                          truncate_cache)


@dataclass
class Trajectory:
    cache: DynamicCache
    tokens: list[int]                 # W selected outputs; last is not yet cached.
    logits: torch.Tensor
    journals: list[dict]
    starts: list[int]
    first_divergence: int | None = None  # One-based OUTPUT transition.


def sync(engine):
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)


def tensor_check(a, b):
    same = a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    equal = same and finite and torch.equal(a, b)
    error = float((a.double() - b.double()).abs().max()) if same and finite else math.inf
    return {"equal": equal, "finite": finite, "max_abs_error": error}


@torch.no_grad()
def decode(engine, cache, initial_token, window, capture_cuts=()):
    tokens, journals, starts = [], [], []
    token = initial_token
    for _ in range(window):
        result = engine.step(token, cache, capture_cuts=capture_cuts)
        token = result.next_token
        tokens.append(token)
        journals.append(result.journals)
        starts.append(0)
    return Trajectory(cache, tokens, result.logits, journals, starts)


@torch.no_grad()
def repair(engine, working, checkpoint, initial_token, faulty, fault_layer, cut):
    """Restore INSIDE this function; caller clones the working input before timing."""
    if not 0 <= cut <= fault_layer < engine.num_layers:
        raise ValueError("Untrusted cut/localization")
    prefix = int(checkpoint.get_seq_length())
    restore_layers(working, checkpoint, start_layer=cut)
    token, divergence = initial_token, None
    tokens, starts = [], []
    for i, old_output in enumerate(faulty.tokens):
        start = cut if divergence is None else 0
        entry = faulty.journals[i][cut] if start else None
        result = engine.step(token, working, position=prefix + i,
                             start_layer=start, journal=entry,
                             fault_layer=fault_layer if start else None)
        token = result.next_token
        tokens.append(token)
        starts.append(start)
        if divergence is None and token != old_output:
            divergence = i + 1
            if cut:
                truncate_cache(working, prefix + i + 1, start_layer=0)
    expected = engine.num_layers * len(tokens) - cut * (divergence or len(tokens))
    executed = sum(engine.num_layers - start for start in starts)
    if expected != executed:
        raise AssertionError("Layer-step accounting disagrees with actual start trace")
    return Trajectory(working, tokens, result.logits, [], starts, divergence)


@torch.no_grad()
def inject_fault(cache, layer, case):
    """Benign predeclared tensor perturbation; never searches model outputs."""
    collection = cache.key_cache if case["kind"] == "K" else cache.value_cache
    data = collection[layer]
    rng = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
    head = int(torch.randint(data.shape[1], (), generator=rng))
    position = int(torch.randint(data.shape[2], (), generator=rng))
    dimension = int(torch.randint(data.shape[3], (), generator=rng))
    original = data[0, head, position].detach().clone()
    rms = float(data.float().square().mean().sqrt())
    if not math.isfinite(rms):
        raise ValueError("Non-finite clean injection tensor")
    delta = torch.zeros(data.shape[3], dtype=torch.float32)
    if case["fault"] == "scalar":
        delta[dimension] = 8 * rms * torch.randn((), generator=rng)
    elif case["fault"] == "vector":
        delta = 8 * rms * torch.randn(data.shape[3], generator=rng)
    elif case["fault"] != "noop":
        raise ValueError("Unknown perturbation")
    data[0, head, position] = (original.float() + delta.to(data.device)).to(data.dtype)
    record = {"kind": case["kind"], "fault": case["fault"], "layer": layer,
              "batch": 0, "head": head, "prefix_position": position,
              "scalar_dimension": dimension, "rms": rms,
              "before": original.float().cpu().tolist(),
              "requested_delta": delta.tolist(),
              "after": data[0, head, position].float().cpu().tolist()}
    record["changed_elements"] = int((original != data[0, head, position]).sum())
    return record, original


@torch.no_grad()
def verify(engine, actual, reference, reference_continuation):
    cache = compare_caches(actual.cache, reference.cache)
    logits = tensor_check(actual.logits, reference.logits)
    tokens = actual.tokens == reference.tokens
    continuation = decode(engine, clone_cache(actual.cache), actual.tokens[-1], 2)
    cont_cache = compare_caches(continuation.cache, reference_continuation.cache)
    cont_logits = tensor_check(continuation.logits, reference_continuation.logits)
    cont_tokens = continuation.tokens == reference_continuation.tokens
    result = {"cache": cache, "logits": logits, "tokens_equal": tokens,
              "continuation_cache": cont_cache, "continuation_logits": cont_logits,
              "continuation_tokens_equal": cont_tokens,
              "tokens": actual.tokens, "continuation_tokens": continuation.tokens}
    result["all_equal"] = all((cache["equal"], logits["equal"], tokens,
                               cont_cache["equal"], cont_logits["equal"], cont_tokens))
    return result


@torch.no_grad()
def native_parity(engine, prefix, decode_steps=4):
    """Independent library forward at every token, including beyond position eight."""
    native, custom = DynamicCache(), engine.empty_cache()
    trace, token = [], None
    for i in range(len(prefix) + decode_steps):
        token = prefix[i] if i < len(prefix) else token
        expected = engine.model(input_ids=torch.tensor([[token]], device=engine.device),
                                past_key_values=native, use_cache=True, return_dict=True)
        native = expected.past_key_values
        actual = engine.step(token, custom)
        item = {"position": i, "logits": tensor_check(actual.logits, expected.logits),
                "cache": compare_caches(custom, native)}
        trace.append(item)
        if not item["logits"]["equal"] or not item["cache"]["equal"]:
            return {"passed": False, "prefix_tokens": len(prefix), "trace": trace}
        token = int(expected.logits[0, -1].argmax())
    return {"passed": True, "prefix_tokens": len(prefix),
            "decode_steps": decode_steps, "trace": trace}


def prefix_ids(tokenizer, case):
    ids = tokenizer.encode(case["prompt"], add_special_tokens=True)
    if not ids:
        raise ValueError("Empty tokenized prompt")
    length = int(case["prefix_tokens"])
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def memory_start(engine):
    if engine.device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats(engine.device)
    return torch.cuda.memory_allocated(engine.device)


def memory_end(engine, baseline):
    peak = torch.cuda.max_memory_allocated(engine.device) if engine.device.type == "cuda" else 0
    return {"allocated_before_bytes": baseline, "peak_allocated_bytes": peak,
            "peak_increment_bytes": peak - baseline}


@torch.no_grad()
def overhead_trials(engine, checkpoint, initial, reference, cuts, repetitions):
    rows, names = [], list(cuts)
    for rep in range(-1, repetitions):  # One excluded warmup for each mode.
        order = names if rep < 0 else names[rep % len(names):] + names[:rep % len(names)]
        for name in order:
            working = clone_cache(checkpoint)  # Excluded equally, as for repair.
            sync(engine)
            baseline = memory_start(engine)
            start = time.perf_counter()
            trajectory = decode(engine, working, initial, len(reference.tokens), cuts[name])
            sync(engine)
            elapsed = time.perf_counter() - start
            memory = memory_end(engine, baseline)
            equal = (trajectory.tokens == reference.tokens and
                     torch.equal(trajectory.logits, reference.logits) and
                     compare_caches(trajectory.cache, reference.cache)["equal"])
            if not equal:
                raise RuntimeError(f"Journal mode {name} changed clean execution")
            if rep >= 0:
                rows.append({"mode": name, "repetition": rep, "order": order,
                             "seconds": elapsed, "verified_equal": equal,
                             "journal_payload_bytes": journal_nbytes(trajectory.journals),
                             **memory})
            del working, trajectory
    return rows


def cpu_state(trajectory):
    return {"keys": [t.detach().cpu() for t in trajectory.cache.key_cache],
            "values": [t.detach().cpu() for t in trajectory.cache.value_cache],
            "seen_tokens": int(trajectory.cache._seen_tokens),
            "tokens": trajectory.tokens, "logits": trajectory.logits.detach().cpu(),
            "start_layers": trajectory.starts,
            "first_divergence": trajectory.first_divergence}


@torch.no_grad()
def run_case(engine, tokenizer, case, config, capture_audit=False):
    N, W = engine.num_layers, int(case["window"])
    layer = int(float(case["layer_fraction"]) * (N - 1))
    sparse = sorted({c for c in (N // 4, N // 2, 3 * N // 4) if 0 < c < N})
    selected = max([0] + [c for c in sparse if c <= layer])
    ids = prefix_ids(tokenizer, case)
    checkpoint, prefix_logits = engine.prefill(ids)
    initial = int(prefix_logits[0, -1].argmax())
    checkpoint_times = []
    for _ in range(3):
        sync(engine)
        start = time.perf_counter()
        copied = clone_cache(checkpoint)
        sync(engine)
        checkpoint_times.append(time.perf_counter() - start)
        del copied
    reference = decode(engine, clone_cache(checkpoint), initial, W)
    reference_cont = decode(engine, clone_cache(reference.cache), reference.tokens[-1], 2)
    faulty_cache = clone_cache(checkpoint)
    fault, original = inject_fault(faulty_cache, layer, case)
    faulty = decode(engine, faulty_cache, initial, W, range(1, N))
    first_divergence = next((i + 1 for i, (a, b) in enumerate(zip(reference.tokens, faulty.tokens))
                             if a != b), None)
    row = {"case_id": case["id"], "case": case, "num_layers": N, "layer": layer,
           "sparse_cuts": sparse, "selected_sparse_cut": selected,
           "prefix_ids": ids, "initial_token": initial, "fault": fault,
           "reference_tokens": reference.tokens, "faulty_tokens": faulty.tokens,
           "first_divergence": first_divergence, "checkpoint_payload_bytes": cache_nbytes(checkpoint),
           "checkpoint_copy_seconds": checkpoint_times,
           "all_journal_payload_bytes": journal_nbytes(faulty.journals),
           "sparse_journal_payload_bytes": journal_nbytes([{c: d[c] for c in sparse} for d in faulty.journals]),
           "methods": {}, "timings": [], "overhead_trials": []}
    # Caller may request first-divergence capture without knowing it in advance.
    save = capture_audit or (config.get("_want_divergence_audit", False) and first_divergence is not None)
    audit = {"prefix_ids": ids, "initial_token": initial, "fault": fault,
             "checkpoint_keys": [t.detach().cpu() for t in checkpoint.key_cache],
             "checkpoint_values": [t.detach().cpu() for t in checkpoint.value_cache],
             "reference": cpu_state(reference), "none": cpu_state(faulty)} if save else None
    row["methods"]["none"] = verify(engine, faulty, reference, reference_cont)
    slot_cache = clone_cache(faulty.cache)
    collection = slot_cache.key_cache if case["kind"] == "K" else slot_cache.value_cache
    collection[layer][0, fault["head"], fault["prefix_position"]] = original
    slot = Trajectory(slot_cache, faulty.tokens, faulty.logits, [], [])
    row["methods"]["slotonly"] = verify(engine, slot, reference, reference_cont)
    if audit is not None:
        audit["slotonly"] = cpu_state(slot)
    del slot, slot_cache
    methods = {"full": 0, "sparse": selected, "all": layer}
    names = list(methods)
    valid = True
    for rep in range(-1, int(config.get("repetitions", 3))):
        order = names if rep < 0 else names[rep % 3:] + names[:rep % 3]
        for name in order:
            working = clone_cache(faulty.cache)  # OUTSIDE timed restore/replay.
            sync(engine)
            baseline = memory_start(engine)
            start = time.perf_counter()
            repaired = repair(engine, working, checkpoint, initial, faulty, layer, methods[name])
            sync(engine)
            elapsed = time.perf_counter() - start
            memory = memory_end(engine, baseline)
            checks = verify(engine, repaired, reference, reference_cont)
            checks.update(cut=methods[name], first_divergence=repaired.first_divergence,
                          start_layers=repaired.starts,
                          layer_steps=sum(N - c for c in repaired.starts))
            valid = valid and checks["all_equal"]
            if rep >= 0:
                row["timings"].append({"method": name, "repetition": rep, "order": order,
                                       "seconds": elapsed, "checks": checks, **memory})
                row["methods"][name] = checks
            if audit is not None and rep == 0:
                audit[name] = cpu_state(repaired)
            del repaired, working
            if not checks["all_equal"]:
                row["valid_repairs_equal"] = False
                row["failure"] = {"method": name, "repetition": rep, "checks": checks}
                return row, audit
    row["overhead_trials"] = overhead_trials(engine, checkpoint, initial, reference,
        {"none": [], "sparse": sparse, "all": list(range(1, N))},
        int(config.get("overhead_repetitions", 5)))
    row["valid_repairs_equal"] = valid
    return row, audit


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Explicit non-finite diagnostics, valid strict JSON.
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "starting", "config_sha256": digest(args.config),
                "manifest_sha256": digest(args.manifest), "runner_sha256": digest(__file__),
                "engine_sha256": digest(Path(__file__).with_name("recut_engine.py")),
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "equality": "exact elementwise, not byte/bitwise",
                "timing_boundary": "working-cache clone excluded equally; restore and replay included; diagnostics excluded",
                "memory_scope": "peaks include resident correctness fixtures; payload bytes exclude allocator/metadata overhead",
                "native_parity": {}, "models_completed": []}
    write_json(args.output / "metadata.json", metadata)
    try:
        config = json.loads(args.config.read_text())
        records = json.loads(args.manifest.read_text())
        cases = config["cases"]
        if not cases or len({c["id"] for c in cases}) != len(cases):
            raise ValueError("Need nonempty cases with unique IDs")
        for case in cases:
            if (int(case["prefix_tokens"]) < 8 or int(case["window"]) < 1 or
                not 0 <= float(case["layer_fraction"]) <= 1 or case["kind"] not in ("K", "V") or
                case["fault"] not in ("scalar", "vector", "noop")):
                raise ValueError(f"Invalid case: {case['id']}")
        if min(int(config.get("repetitions", 3)), int(config.get("overhead_repetitions", 5))) < 1:
            raise ValueError("Repetitions must be positive")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This real-model runner requires one BF16-capable CUDA GPU")
        torch.set_grad_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        metadata.update(status="running", config=config, models=records, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0),
                        deterministic_algorithms=True, tf32=False, attention="eager", dtype="bfloat16",
                        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        write_json(args.output / "metadata.json", metadata)
        audit_used = 0
        budget = int(config.get("audit_max_bytes", 50 * 1024 * 1024))
        requested = config.get("models", [r["model"] for r in records])
        if set(requested) - {r["model"] for r in records}:
            raise ValueError("Configured model is missing from pinned manifest")
        with (args.output / "episodes.jsonl").open("w") as output:
            for record in records:
                if record["model"] not in requested:
                    continue
                model_cases = [c for c in cases if c.get("model", record["model"]) == record["model"]]
                if not model_cases:
                    continue
                saved_first, saved_divergence = False, False  # Keep global byte budget.
                revision = record["revision"]
                if len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision.lower()):
                    raise ValueError("Manifest revision must be an immutable 40-character commit SHA")
                source = record["local_path"]
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                    attn_implementation="eager").to("cuda").eval()
                engine = HFIncrementalEngine(model)
                metadata.setdefault("model_configs", {})[record["model"]] = model.config.to_dict()
                longest = max(model_cases, key=lambda c: c["prefix_tokens"])
                parity = native_parity(engine, prefix_ids(tokenizer, longest),
                                       decode_steps=max(int(c["window"]) for c in model_cases) + 2)
                metadata["native_parity"][record["model"]] = parity
                write_json(args.output / "metadata.json", metadata)
                if not parity["passed"]:
                    raise RuntimeError("Native incremental-forward parity failed")
                for case in model_cases:
                    metadata["active_case"] = {"model": record["model"], "case_id": case["id"]}
                    local = dict(config, _want_divergence_audit=not saved_divergence and audit_used < budget)
                    row, audit = run_case(engine, tokenizer, case, local,
                                          capture_audit=not saved_first and audit_used < budget)
                    row.update(model=record["model"], revision=revision,
                               config_sha256=metadata["config_sha256"])
                    if audit is not None:
                        buffer = io.BytesIO()
                        torch.save({"model": record, "episode": serializable(row), "states": audit}, buffer)
                        blob = buffer.getvalue()
                        if audit_used + len(blob) <= budget:
                            name = f"audit_{len(metadata.get('audits', [])):02d}.pt"
                            (args.output / name).write_bytes(blob)
                            audit_used += len(blob)
                            saved_first = True
                            saved_divergence |= row["first_divergence"] is not None
                            row["audit_file"] = name
                            metadata.setdefault("audits", []).append({"file": name, "bytes": len(blob),
                                "sha256": hashlib.sha256(blob).hexdigest(), "model": record["model"], "case_id": case["id"]})
                        else:
                            row["audit_skipped"] = "global byte budget"
                        del buffer, blob, audit
                    elif not saved_first or (not saved_divergence and row["first_divergence"] is not None):
                        row["audit_skipped"] = "global byte budget"
                    output.write(json.dumps(serializable(row), allow_nan=False) + "\n")
                    output.flush()
                    print(json.dumps({"model": record["model"], "case": case["id"],
                                      "valid_repairs_equal": row["valid_repairs_equal"],
                                      "first_divergence": row["first_divergence"]}), flush=True)
                    if not row["valid_repairs_equal"]:
                        raise RuntimeError("Valid repair failed; stopping rather than continuing primary claims")
                metadata["models_completed"].append(record["model"])
                write_json(args.output / "metadata.json", metadata)
                del engine, model, tokenizer
                torch.cuda.empty_cache()
        metadata.pop("active_case", None)
        metadata.update(status="complete", audit_total_bytes=audit_used)
        metadata["episodes_sha256"] = digest(args.output / "episodes.jsonl")
        write_json(args.output / "metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(args.output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
