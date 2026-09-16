"""Frozen-matrix RECUT protection-cost study; not a production serving benchmark.

Native batched prefill is common setup. Measured transaction windows include
guards, copies, journals, injection, detection and repair. Native/bare clean
decoding are cost baselines, not resilience-equivalent implementations. Separate
profiling and warmups never enter measured session records.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from recut_engine import HFIncrementalEngine, cache_nbytes, clone_cache, compare_caches, journal_nbytes
from recut_controller import TransactionalSession, WindowRejected

VARIANTS = ("native", "bare", "full_batched", "sparse_batched", "allcuts_batched", "full_legacy", "sparse_legacy")
BATCHED = ("full_batched", "sparse_batched", "allcuts_batched")
PROTECTED = VARIANTS[2:]
FAULTS = ("key_early", "value_early", "key_mid", "value_mid", "key_late", "value_late", "multi_prefix", "repeated_prefix")
GUARDS = ("journal_corruption", "checkpoint_corruption")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def sources():
    paths = list(HERE.glob("*.py"))
    paths += [HERE.parent / "experiments" / name for name in ("recut_engine.py", "prefix_detector.py", "run_recut.py")]
    return {os.path.relpath(path, HERE): sha(path) for path in sorted(paths)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def append_json(stream, data):
    stream.write(json.dumps(data, allow_nan=False) + "\n")
    stream.flush()


def sync(engine):
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)


def tensor_check(left, right):
    layout = left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
    finite = bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
    equal = layout and finite and torch.equal(left, right)
    error = float((left.double() - right.double()).abs().max()) if layout and finite else None
    return {"equal": equal, "finite": finite, "max_abs_error": error}


def tensor_digest(tensor):
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def cache_digest(cache):
    return {"keys": [tensor_digest(t) for t in cache.key_cache], "values": [tensor_digest(t) for t in cache.value_cache], "seen_tokens": int(cache._seen_tokens)}


def cache_cpu(cache):
    return {"keys": [t.detach().cpu().clone() for t in cache.key_cache], "values": [t.detach().cpu().clone() for t in cache.value_cache], "seen_tokens": int(cache._seen_tokens)}


@dataclass
class Reference:
    cache: DynamicCache
    logits: torch.Tensor
    tokens: tuple
    continuation_cache: DynamicCache
    continuation_logits: torch.Tensor
    continuation_tokens: tuple


@torch.no_grad()
def native_step(engine, cache, token):
    output = engine.model(input_ids=torch.tensor([[token]], device=engine.device), past_key_values=cache, use_cache=True, return_dict=True)
    return output.past_key_values, output.logits, int(output.logits[0, -1].argmax())


@torch.no_grad()
def continuation(engine, cache, token, length=2):
    tokens = []
    for _ in range(length):
        cache, logits, token = native_step(engine, cache, token)
        tokens.append(token)
    return cache, logits, tuple(tokens)


def make_prefix(tokenizer, passage, length):
    parts, ids, index = [], [], 0
    while len(ids) < length:
        index += 1
        parts.append(f"Section {index}\n{passage}\n")
        ids = tokenizer.encode("\n".join(parts), add_special_tokens=True)
        if index > length + 1:
            raise RuntimeError("Tokenizer produced insufficient prefix tokens")
    return ids[:length]


@torch.no_grad()
def build_references(engine, prefix, window, n_windows):
    # Retain only the final prefill decision, not [prefix,vocabulary] logits.
    output = engine.model(input_ids=torch.tensor([prefix], device=engine.device), past_key_values=DynamicCache(), use_cache=True, return_dict=True)
    checkpoint, initial = output.past_key_values, int(output.logits[0, -1].argmax())
    del output
    native, custom = clone_cache(checkpoint), clone_cache(checkpoint)
    token, refs, parity = initial, [], []

    def checked_step(pending):
        nonlocal native
        position = int(native.get_seq_length())
        native, logits, next_token = native_step(engine, native, pending)
        actual = engine.step(pending, custom)
        check = {"position": position, "cache": compare_caches(custom, native), "logits": tensor_check(actual.logits, logits), "token_equal": actual.next_token == next_token}
        parity.append(check)
        if not (check["cache"]["equal"] and check["logits"]["equal"] and check["token_equal"]):
            raise RuntimeError(f"Native/custom post-prefill parity failed at {position}")
        return logits, next_token

    for _ in range(n_windows):
        tokens = []
        for _ in range(window):
            logits, token = checked_step(token)
            tokens.append(token)
        extra = continuation(engine, clone_cache(native), token)
        refs.append(Reference(clone_cache(native), logits.detach().clone(), tuple(tokens), *extra))
    for _ in range(2):
        logits, token = checked_step(token)
    return checkpoint, initial, refs, parity


@torch.no_grad()
def perturb(data, seed, target, *, prefix=None, layer=None, kind=None, bit=False):
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    if data.ndim == 4:
        if not prefix or prefix > data.shape[2]:
            raise ValueError("Fault must target an existing old prefix")
        head = int(torch.randint(data.shape[1], (), generator=rng))
        position = int(torch.randint(prefix, (), generator=rng))
        vector = data[0, head, position]
        rms_source = data[..., :prefix, :]
        coordinate = {"batch": 0, "head": head, "prefix_position": position, "layer": layer, "kind": kind}
    elif data.ndim == 3 and tuple(data.shape[:2]) == (1, 1):
        vector, rms_source = data[0, 0], data
        coordinate = {"batch": 0, "sequence_position": 0}
    else:
        raise ValueError("Unsupported injection layout")
    if bit:
        feature = int(torch.randint(vector.numel(), (), generator=rng))
        scalar = vector[feature:feature + 1]
        width = scalar.element_size() * 8
        if scalar.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Bit diagnostic requires BF16 or FP32")
        integer_dtype = torch.int16 if width == 16 else torch.int32
        integer = scalar.view(integer_dtype)
        before = scalar.detach().float().cpu().tolist()
        before_bits = int(integer.item()) & ((1 << width) - 1)
        after_bits = before_bits ^ (1 << 3)
        signed = after_bits if after_bits < (1 << (width - 1)) else after_bits - (1 << width)
        integer.fill_(signed)
        after = scalar.detach().float().cpu().tolist()
        result = {"fault_type": "fraction_bit_flip", "feature": feature, "bit": 3, "width": width,
                  "dtype": str(scalar.dtype), "before_bits": [before_bits], "after_bits": [after_bits], "before": before, "after": after}
    else:
        before = vector.detach().clone()
        rms = float(rms_source.float().square().mean().sqrt())
        if not math.isfinite(rms) or rms <= 0:
            raise RuntimeError("Fault requires positive finite source RMS")
        delta = 8 * rms * torch.randn(vector.numel(), generator=rng, dtype=torch.float32)
        vector.copy_((before.float() + delta.to(vector.device)).to(vector.dtype))
        result = {"fault_type": "additive_8rms", "rms": rms, "dtype": str(vector.dtype), "before": before.float().cpu().tolist(),
                  "requested_delta": delta.tolist(), "after": vector.float().cpu().tolist()}
    changed = sum(a != b for a, b in zip(result["before"], result["after"]))
    if not changed or not all(math.isfinite(x) for x in result["after"]):
        raise RuntimeError("Intended fault did not yield changed finite values")
    return {"target": target, "seed": int(seed), "changed_elements": changed, **coordinate, **result}


@torch.no_grad()
def inject(tx, engine, scenario, wi, completed_steps, seed, variant, cuts):
    if scenario == "clean" or (wi != 1 and scenario != "repeated_prefix"):
        return []
    late, mid = 3 * engine.num_layers // 4, engine.num_layers // 2
    schedule = {
        "key_early": [(1, "K", 0, False)], "value_early": [(tx.window // 2, "V", 0, True)],
        "key_mid": [(1, "K", mid, False)], "value_mid": [(tx.window // 2, "V", mid, True)],
        "key_late": [(1, "K", late, True)], "value_late": [(tx.window // 2, "V", late, False)],
        "multi_prefix": [(1, "V", late, False), (2, "K", max(0, mid - 1), False)],
        "repeated_prefix": [(1, "V", late, False)], "journal_corruption": [(1, "V", late, False)], "checkpoint_corruption": []}
    result = []
    for event, (when, kind, layer, bit) in enumerate(schedule[scenario]):
        if completed_steps == when:
            tensor = (tx.working.key_cache if kind == "K" else tx.working.value_cache)[layer]
            result.append(perturb(tensor, seed + 1009 * wi + event, "working_prefix", prefix=tx.prefix, layer=layer, kind=kind, bit=bit))
    if completed_steps == 2 and scenario == "checkpoint_corruption":
        result.append(perturb(tx.checkpoint.value_cache[late], seed + 1009 * wi, "checkpoint", prefix=tx.prefix, layer=late, kind="V"))
    if completed_steps == 2 and scenario == "journal_corruption" and not variant.startswith("full_"):
        cut = max(c for c in cuts if c <= late)
        entry = tx.journals[0][cut]
        record = perturb(entry.hidden, seed + 1009 * wi + 1, "journal")
        record.update(cut=cut, journal_step=0, journal_position=entry.position, journal_token=entry.token)
        result.append(record)
    for record in result:
        record.update(window_index=wi, after_step=completed_steps)
    return result


def variant_parameters(variant, layers):
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    if variant in ("native", "bare"):
        return "unprotected", None, ()
    label, seal = variant.split("_")
    cuts = tuple(range(1, layers)) if label == "allcuts" else tuple(sorted({c for c in (layers // 4, layers // 2, 3 * layers // 4) if 0 < c < layers})) if label == "sparse" else ()
    return "full" if label == "full" else "sparse", seal, cuts


def should_snapshot(config, prompt_id, prefix, scenario, seed, rep, variant, wi):
    return (prompt_id == config.get("snapshot_prompt") and seed == config["seeds"][0] and rep == 0 and wi == 1 and
            ((prefix == config.get("snapshot_short_prefix") and scenario in config.get("snapshot_cases", []) and variant in BATCHED)
             or (prefix == max(s["prefix_tokens"] for s in config["shapes"]) and scenario == config.get("snapshot_long_case") and variant == "sparse_batched")))


@torch.no_grad()
def verify_and_snapshot(engine, session, tokens, reference, expected_tokens, snapshot_path=None, identity=None):
    extra = continuation(engine, clone_cache(session.cache), session.next_token)
    checks = {"cache": compare_caches(session.cache, reference.cache), "logits": tensor_check(session.last_logits, reference.logits),
              "tokens_equal": tuple(tokens) == reference.tokens, "committed_tokens_equal": tuple(session.committed_tokens) == tuple(expected_tokens),
              "next_token_equal": session.next_token == reference.tokens[-1], "continuation_cache": compare_caches(extra[0], reference.continuation_cache),
              "continuation_logits": tensor_check(extra[1], reference.continuation_logits), "continuation_tokens_equal": extra[2] == reference.continuation_tokens,
              "not_poisoned": not session.poisoned}
    checks["all_equal"] = all(v["equal"] if isinstance(v, dict) else v for v in checks.values())
    snapshot = None
    if snapshot_path is not None:
        def pack(cache, logits, out, pending, continuation_values):
            return {"cache": cache_cpu(cache), "logits": logits.detach().cpu().clone(), "tokens": list(out), "next_token": pending,
                    "continuation_cache": cache_cpu(continuation_values[0]), "continuation_logits": continuation_values[1].detach().cpu().clone(),
                    "continuation_tokens": list(continuation_values[2])}
        archive = {"schema_version": 1, "identity": identity,
                   "recovered": pack(session.cache, session.last_logits, tokens, session.next_token, extra),
                   "reference": pack(reference.cache, reference.logits, reference.tokens, reference.tokens[-1],
                                     (reference.continuation_cache, reference.continuation_logits, reference.continuation_tokens))}
        snapshot_path.parent.mkdir(exist_ok=True)
        if snapshot_path.exists():
            raise RuntimeError("Refusing snapshot overwrite")
        torch.save(archive, snapshot_path)
        snapshot = {"file": "snapshots/" + snapshot_path.name, "sha256": sha(snapshot_path), "bytes": snapshot_path.stat().st_size}
    return checks, snapshot


@torch.no_grad()
def run_session(engine, checkpoint, initial, references, *, variant, scenario, window, seed, repetition,
                workload_id, model_name, order, profile=False, snapshot_directory=None, snapshot_predicate=None):
    policy, seal, cuts = variant_parameters(variant, engine.num_layers)
    protected = policy != "unprotected"
    if not protected and scenario != "clean":
        raise ValueError("Unprotected variants are clean cost baselines only")
    cache = clone_cache(checkpoint)
    # The common input fork is setup, not a protection cost. Report the actual
    # constructor cost separately because protected construction validates the
    # initial state, whereas a plain baseline only builds its bookkeeping.
    sync(engine)
    initialization_started = time.perf_counter()
    session = (TransactionalSession(engine, cache, initial, policy=policy, cuts=cuts, seal_mode=seal, profile=profile) if protected else
               SimpleNamespace(cache=cache, next_token=initial, last_logits=None, committed_tokens=(), windows_committed=0, poisoned=False))
    sync(engine)
    initialization_seconds = time.perf_counter() - initialization_started
    del cache
    row = {"workload_id": workload_id, "model": model_name, "prefix_tokens": int(checkpoint.get_seq_length()), "num_layers": engine.num_layers,
           "window": window, "n_windows": len(references), "seed": seed, "scenario": scenario, "variant": variant, "policy": policy,
           "seal_mode": seal, "repetition": repetition, "method_order": list(order), "cuts": list(cuts), "profile": profile,
           "session_initialization_seconds": initialization_seconds,
           "windows": [], "valid": True, "same_policy_treatment": scenario != "journal_corruption"}
    expected_tokens = []
    for wi, reference in enumerate(references):
        before_tokens, before_windows = tuple(session.committed_tokens), session.windows_committed
        prefix_payload = cache_nbytes(session.cache)
        faults, injection_seconds, step_times, tx, result, rejection = [], 0.0, [], None, None, None
        no_early_commit = True
        sync(engine)
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            allocated_before = torch.cuda.memory_allocated(engine.device)
        else:
            allocated_before = 0
        started = time.perf_counter()
        generated = []
        try:
            if protected:
                tx = session.begin(window)
            for step in range(window):
                if protected:
                    tx.step()
                elif variant == "native":
                    session.cache, session.last_logits, session.next_token = native_step(engine, session.cache, session.next_token)
                    generated.append(session.next_token)
                else:
                    output = engine.step(session.next_token, session.cache)
                    session.last_logits, session.next_token = output.logits, output.next_token
                    generated.append(session.next_token)
                # All variants already synchronize greedy argmax; no new CUDA
                # synchronization is introduced by this timestamp alone.
                step_times.append(time.perf_counter() - started)
                no_early_commit &= tuple(session.committed_tokens) == before_tokens
                if protected and scenario != "clean" and (wi == 1 or scenario == "repeated_prefix"):
                    # Time the deterministic harness including empty schedule
                    # calls; subtracting it is only a secondary diagnostic.
                    sync(engine)
                    injection_started = time.perf_counter()
                    faults.extend(inject(tx, engine, scenario, wi, step + 1, seed, variant, cuts))
                    sync(engine)
                    injection_seconds += time.perf_counter() - injection_started
            if protected:
                result = tx.finish()
                generated = list(result.tokens)
            else:
                session.committed_tokens += tuple(generated)
                session.windows_committed += 1
        except WindowRejected as error:
            rejection = {"type": type(error).__name__, "reason": error.reason, "controller_metadata": error.metadata}
        sync(engine)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(engine.device) if engine.device.type == "cuda" else 0
        measured = {"window_index": wi, "wall_seconds": elapsed, "injection_harness_seconds": injection_seconds,
                    "wall_minus_injection_seconds": elapsed - injection_seconds, "step_completion_seconds": step_times,
                    "release_seconds": elapsed if rejection is None else None,
                    "token_availability_seconds": ([elapsed] * window if protected else list(step_times)) if rejection is None else [],
                    "availability_semantics": "held until commit" if protected else "per-step completion, locally buffered cost baseline; not network streaming",
                    "faults": faults, "no_early_commit": no_early_commit, "committed_before": len(before_tokens),
                    "committed_after": len(session.committed_tokens), "logical_input_cache_bytes": prefix_payload,
                    "logical_checkpoint_bytes": cache_nbytes(tx.checkpoint) if tx is not None else 0,
                    "logical_journal_bytes": journal_nbytes(tx.journals) if tx is not None else 0,
                    "allocated_before_bytes": allocated_before, "peak_allocated_bytes": peak, "peak_increment_bytes": peak - allocated_before,
                    "speculative_tokens": list(tx._outputs) if tx is not None else generated}
        if rejection:
            previous = references[wi - 1] if wi else None
            checks = {"expected_rejection": scenario == "checkpoint_corruption" and wi == 1 and rejection["reason"] == "Checkpoint integrity failure" and any(f["target"] == "checkpoint" for f in faults),
                      "poisoned": session.poisoned, "committed_tokens_unchanged": tuple(session.committed_tokens) == before_tokens,
                      "committed_windows_unchanged": session.windows_committed == before_windows,
                      "committed_cache_unchanged": bool(previous and compare_caches(session.cache, previous.cache)["equal"]),
                      "committed_logits_unchanged": bool(previous and tensor_check(session.last_logits, previous.logits)["equal"]),
                      "next_token_unchanged": bool(previous and session.next_token == previous.tokens[-1]), "no_early_commit": no_early_commit}
            measured.update(status="rejected", rejection=rejection, checks=checks, valid=all(checks.values()))
            row["windows"].append(measured)
            row.update(status="expected_abort" if measured["valid"] else "invalid", valid=measured["valid"])
            break
        expected_tokens.extend(reference.tokens)
        snapshot_path = None
        identity = {"workload_id": workload_id, "model": model_name, "scenario": scenario, "seed": seed, "repetition": repetition, "variant": variant, "window_index": wi}
        if snapshot_directory is not None and snapshot_predicate is not None and snapshot_predicate(wi):
            filename = f"{workload_id}--{scenario}--s{seed}--r{repetition}--{variant}--w{wi}.pt"
            snapshot_path = snapshot_directory / filename
        checks, snapshot = verify_and_snapshot(engine, session, generated, reference, expected_tokens, snapshot_path, identity)
        layers = sorted({f["layer"] for f in faults if f["target"] == "working_prefix"})
        checks.update(no_early_commit=no_early_commit, committed_windows_equal=session.windows_committed == wi + 1,
                      detected_layers_equal_injection=list(result.detected_layers if protected else ()) == layers,
                      recovery_expected=bool(result.recovered if protected else False) == bool(layers),
                      unexpected_checkpoint_commit=not (scenario == "checkpoint_corruption" and wi == 1))
        if scenario == "journal_corruption" and wi == 1 and policy == "sparse":
            checks["journal_corruption_fell_back"] = result.fallback_reason == "journal_integrity_failure" and result.selected_cut == 0
        checks["all_equal"] = all(v["equal"] if isinstance(v, dict) else v for v in checks.values())
        measured.update(status="committed", valid=checks["all_equal"], checks=checks, tokens=generated,
                        detected_layers=list(result.detected_layers) if protected else [], selected_cut=result.selected_cut if protected else 0,
                        replay_starts=list(result.replay_starts) if protected else [], first_divergence=result.first_divergence if protected else None,
                        fallback_reason=result.fallback_reason if protected else None, recovered=result.recovered if protected else False,
                        controller_metadata=result.metadata if protected else {"route": "unprotected", "profile_enabled": False})
        if snapshot:
            measured["snapshot"] = snapshot
        row["windows"].append(measured)
        del tx, result
        if not measured["valid"]:
            row.update(status="invalid", valid=False)
            break
    row.setdefault("status", "complete")
    row.update(committed_tokens=list(session.committed_tokens), windows_committed=session.windows_committed, poisoned=session.poisoned,
               total_wall_seconds=sum(w["wall_seconds"] for w in row["windows"]),
               total_injection_harness_seconds=sum(w["injection_harness_seconds"] for w in row["windows"]))
    row["total_with_initialization_seconds"] = row["total_wall_seconds"] + initialization_seconds
    return row


def measured_cases(config):
    cases = []
    first_seed = config["seeds"][0]
    for rep in range(config["clean_repetitions"]):
        cases.append(("clean", first_seed, rep, VARIANTS))
    for scenario, seed, rep in itertools.product(config["fault_cases"], config["seeds"], range(config["repetitions"])):
        variants = BATCHED + (("full_legacy", "sparse_legacy") if scenario in config.get("legacy_cases", []) else ())
        cases.append((scenario, seed, rep, variants))
    for scenario in config["guard_cases"]:
        cases.append((scenario, first_seed, 0, BATCHED))
    return cases


def validate_config(config, records, selected):
    if config.get("policies") != ["full", "sparse", "allcuts"]:
        raise ValueError("This fixed protocol requires policies ['full', 'sparse', 'allcuts']")
    legacy = config.get("legacy_cases", [])
    if (not isinstance(legacy, list) or len(legacy) != len(set(legacy))
            or set(legacy) - {"clean", "value_late"} or "clean" not in legacy):
        raise ValueError("Legacy cases must include clean and may additionally include value_late")
    if selected not in config["models"]:
        raise ValueError("Selected model is not configured")
    records = [record for record in records if record["model"] == selected]
    if len(records) != 1:
        raise ValueError("Selected model must have exactly one manifest record")
    record = records[0]
    if len(record["revision"]) != 40 or any(c not in "0123456789abcdef" for c in record["revision"].lower()):
        raise ValueError("Model revision must be an immutable 40-character commit")
    if not Path(record["local_path"]).is_dir():
        raise ValueError("Pinned offline model directory does not exist")
    if set(config["fault_cases"]) - set(FAULTS) or set(config["guard_cases"]) - set(GUARDS):
        raise ValueError("Unknown scenario")
    for key in ("seeds", "fault_cases", "guard_cases", "models"):
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f"{key} must be nonempty and unique")
    if any(type(v) is not int or v < 0 for v in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("repetitions", "clean_repetitions")) or config["n_windows"] < 2:
        raise ValueError("Invalid repetitions/window count")
    if len({p["id"] for p in config["prompts"]}) != len(config["prompts"]) or not config["prompts"]:
        raise ValueError("Prompts require unique IDs")
    for prompt in config["prompts"]:
        if not prompt["text"] or not prompt["id"].replace("_", "").isalnum():
            raise ValueError("Unsafe/empty prompt ID or text")
    if len({(s["prefix_tokens"], s["window"]) for s in config["shapes"]}) != len(config["shapes"]) or not config["shapes"]:
        raise ValueError("Shapes must be nonempty and unique")
    if any(type(s[k]) is not int or s[k] < 2 for s in config["shapes"] for k in ("prefix_tokens", "window")):
        raise ValueError("Shape sizes must be integers >=2")
    return record


def model_file_hashes(path):
    root = Path(path)
    suffixes = {".safetensors", ".json", ".model", ".txt", ".tiktoken"}
    return [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in suffixes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise RuntimeError("Refusing nonempty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    initial_sources = sources()
    metadata = {"schema_version": 1, "status": "starting", "started_unix_seconds": time.time(), "model": args.model,
                "config_sha256": sha(args.config), "manifest_sha256": sha(args.manifest), "source_sha256": initial_sources,
                "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "device": args.device, "dtype": args.dtype, "cuda_runtime": torch.version.cuda,
                "timing_scope": "BEGIN-to-COMMIT for protected; decode-window start-to-finish for native/bare. Includes artificial injection. Window times exclude constructor, common initial fork, prefill, native reference and verification.",
                "initialization_timing_scope": "session_initialization_seconds measures protected constructor including initial finite/layout validation or plain baseline namespace construction, synchronized after the common initial cache fork. total_with_initialization_seconds adds that cost to summed window wall time. Common input fork and prefill remain excluded.",
                "adjusted_timing_scope": "Wall minus separately measured injection harness, diagnostic only; not measured deployment latency.",
                "availability_scope": "Per-step completion versus guarded window release, no client/network; plain outputs locally buffered only for checking.",
                "memory_scope": "Allocator peak includes resident model and correctness references. Payload counters exclude Python/allocator; host peak not measured.",
                "oracle_storage": "Live native equality for every commit; full tensor archives for prespecified subset only.",
                "prefill_scope": "Common native batched prefill; exact custom/native parity at subsequent decode positions only.",
                "workloads": []}
    write_json(args.output / "metadata.json", metadata)
    try:
        config = json.loads(args.config.read_text())
        manifest = json.loads(args.manifest.read_text())
        record = validate_config(config, manifest, args.model)
        if args.device == "cuda" and (not torch.cuda.is_available() or (args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported())):
            raise RuntimeError("Requested CUDA/BF16 capability unavailable")
        torch.set_grad_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        expected = len(config["prompts"]) * len(config["shapes"]) * sum(len(case[3]) for case in measured_cases(config))
        metadata.update(status="running", config=config, manifest=manifest, model_revision=record["revision"], expected_sessions=expected,
                        deterministic_algorithms=True, tf32=False, attention="eager", batch_size=1,
                        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        float32_matmul_precision=torch.get_float32_matmul_precision(), cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        if args.device == "cuda":
            metadata.update(gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        metadata["model_files"] = model_file_hashes(record["local_path"])
        write_json(args.output / "metadata.json", metadata)
        tokenizer = AutoTokenizer.from_pretrained(record["local_path"], local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(record["local_path"], local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()
        engine = HFIncrementalEngine(model)
        metadata["model_config"] = model.config.to_dict()
        completed, warmups, profiles = 0, 0, 0
        with (args.output / "sessions.jsonl").open("x") as output, (args.output / "references.jsonl").open("x") as refs_output, (args.output / "warmups.jsonl").open("x") as warm_output, (args.output / "profiles.jsonl").open("x") as profile_output:
            for prompt, shape in itertools.product(config["prompts"], config["shapes"]):
                prefix_length, window = shape["prefix_tokens"], shape["window"]
                if prefix_length + config["n_windows"] * window + 2 > model.config.max_position_embeddings:
                    raise ValueError("Configured session and continuation exceed context")
                workload_id = f"{args.model.replace('/', '--')}--{prompt['id']}--p{prefix_length}--w{window}"
                prefix = make_prefix(tokenizer, prompt["text"], prefix_length)
                checkpoint, initial, refs, parity = build_references(engine, prefix, window, config["n_windows"])
                reference_record = {"workload_id": workload_id, "model": args.model, "revision": record["revision"], "prompt_id": prompt["id"],
                                    "prefix_ids": prefix, "initial_token": initial, "parity": parity, "checkpoint": cache_digest(checkpoint), "windows": []}
                for wi, ref in enumerate(refs):
                    reference_record["windows"].append({"window_index": wi, "tokens": ref.tokens, "cache": cache_digest(ref.cache), "logits": tensor_digest(ref.logits),
                        "continuation_tokens": ref.continuation_tokens, "continuation_cache": cache_digest(ref.continuation_cache), "continuation_logits": tensor_digest(ref.continuation_logits)})
                append_json(refs_output, reference_record)
                metadata["workloads"].append({"id": workload_id, "prompt_id": prompt["id"], "prefix_tokens": prefix_length, "window": window, "parity_positions": len(parity), "passed": True})

                def execute(variant, scenario, seed, rep, order, profile=False, snapshots=False):
                    row = run_session(engine, checkpoint, initial, refs, variant=variant, scenario=scenario, window=window, seed=seed,
                                      repetition=rep, workload_id=workload_id, model_name=args.model, order=order, profile=profile,
                                      snapshot_directory=args.output / "snapshots" if snapshots else None,
                                      snapshot_predicate=lambda wi: should_snapshot(config, prompt["id"], prefix_length, scenario, seed, rep, variant, wi))
                    row.update(prompt_id=prompt["id"], model_revision=record["revision"], config_sha256=metadata["config_sha256"], source_sha256=initial_sources)
                    return row

                if config.get("warmup", True):
                    for scenario, variants in (("clean", VARIANTS), ("value_late", PROTECTED)):
                        for variant in variants:
                            row = execute(variant, scenario, config["seeds"][0], -1, (variant,))
                            append_json(warm_output, row)
                            warmups += 1
                            if not row["valid"]:
                                raise RuntimeError("Warmup correctness failure")
                for scenario, seed, rep, variants in measured_cases(config):
                    offset = rep % len(variants)
                    order = variants[offset:] + variants[:offset]
                    for variant in order:
                        metadata["active_session"] = {"workload_id": workload_id, "scenario": scenario, "seed": seed, "repetition": rep, "variant": variant}
                        write_json(args.output / "metadata.json", metadata)
                        row = execute(variant, scenario, seed, rep, order, snapshots=True)
                        append_json(output, row)
                        completed += 1
                        print(json.dumps({"session": completed, "of": expected, **metadata["active_session"], "status": row["status"], "valid": row["valid"]}), flush=True)
                        if not row["valid"]:
                            raise RuntimeError("Measured correctness failure; preserved evidence")
                if config.get("profile", False):
                    for scenario, variant in itertools.product(("clean", "value_late"), ("full_batched", "sparse_batched")):
                        row = execute(variant, scenario, config["seeds"][0], -2, (variant,), profile=True)
                        append_json(profile_output, row)
                        profiles += 1
                        if not row["valid"]:
                            raise RuntimeError("Profile correctness failure")
                del refs, parity, checkpoint, reference_record
        if sources() != initial_sources or sha(args.config) != metadata["config_sha256"] or sha(args.manifest) != metadata["manifest_sha256"]:
            raise RuntimeError("Source/config/manifest changed during run")
        if completed != expected:
            raise RuntimeError("Frozen session coverage mismatch")
        metadata.pop("active_session", None)
        metadata.update(status="complete", sessions_completed=completed, warmups_completed=warmups, profiles_completed=profiles,
                        finished_unix_seconds=time.time(), **{name + "_sha256": sha(args.output / (name + ".jsonl")) for name in ("sessions", "references", "warmups", "profiles")})
        write_json(args.output / "metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc(), finished_unix_seconds=time.time())
        write_json(args.output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
