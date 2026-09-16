"""Fresh RECUT stage-2 transactional sessions; bounded evidence, not a service.

The controller receives no injected fault coordinates. Native HF forward is the
correctness oracle. Measured windows include BEGIN, checkpoint and journal
guards, speculative execution, delayed prefix comparison, repair and COMMIT.
Artificial injection is included in wall time and measured separately; its
subtraction is a diagnostic, not a measured production-service latency.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from recut_engine import HFIncrementalEngine, cache_nbytes, clone_cache, compare_caches, journal_nbytes
from controller import TransactionalSession, WindowRejected


SCENARIOS = ("clean", "single_prefix", "multi_prefix", "repeated_prefix",
             "journal_corruption", "checkpoint_corruption")
DEFAULT_PROMPT = "A blue bird sat beside a quiet river. The bird looked at the water and"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def append_json(stream, value):
    stream.write(json.dumps(json_safe(value), allow_nan=False) + "\n")
    stream.flush()


def sync(engine):
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)


def tensor_check(left, right):
    layout = left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
    finite = bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
    equal = layout and finite and torch.equal(left, right)
    error = float((left.double() - right.double()).abs().max()) if layout and finite else math.inf
    return {"equal": equal, "finite": finite, "max_abs_error": error}


def tensor_digest(tensor):
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def cache_digest(cache):
    return {"keys": [tensor_digest(x) for x in cache.key_cache],
            "values": [tensor_digest(x) for x in cache.value_cache],
            "seen_tokens": int(cache._seen_tokens)}


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
    output = engine.model(input_ids=torch.tensor([[token]], device=engine.device),
                          past_key_values=cache, use_cache=True, return_dict=True)
    return output.past_key_values, output.logits, int(output.logits[0, -1].argmax())


@torch.no_grad()
def native_continuation(engine, cache, initial, length=2):
    tokens, token = [], initial
    for _ in range(length):
        cache, logits, token = native_step(engine, cache, token)
        tokens.append(token)
    return cache, logits, tuple(tokens)


@torch.no_grad()
def build_references(engine, prefix, window, n_windows):
    """Independent native gold plus exact custom/native parity at every position."""
    native, custom, parity, refs = DynamicCache(), engine.empty_cache(), [], []

    def checked_step(token):
        nonlocal native
        position = int(native.get_seq_length())
        native, logits, next_token = native_step(engine, native, token)
        actual = engine.step(token, custom)
        checks = {"position": position, "cache": compare_caches(custom, native),
                  "logits": tensor_check(actual.logits, logits),
                  "token_equal": actual.next_token == next_token}
        parity.append(checks)
        if not (checks["cache"]["equal"] and checks["logits"]["equal"] and checks["token_equal"]):
            raise RuntimeError(f"Native/custom parity failed at position {position}")
        return logits, next_token

    for token in prefix:
        logits, token = checked_step(token)
    checkpoint, initial = clone_cache(native), token
    for _ in range(n_windows):
        outputs = []
        for _ in range(window):
            logits, token = checked_step(token)
            outputs.append(token)
        continuation = native_continuation(engine, clone_cache(native), token)
        refs.append(Reference(clone_cache(native), logits.detach().clone(), tuple(outputs), *continuation))
    # Exercise the continuation shapes in the custom/native parity gate as well.
    for _ in range(2):
        logits, token = checked_step(token)
    return checkpoint, initial, refs, parity


@torch.no_grad()
def perturb_vector(data, seed, target, *, prefix_length=None, layer=None):
    """Fixed finite 8-RMS Gaussian vector, sampled on CPU; no outcome search."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    if data.ndim == 4:
        if not prefix_length or prefix_length > data.shape[2]:
            raise ValueError("Injection must target a nonempty old prefix")
        head = int(torch.randint(data.shape[1], (), generator=rng))
        position = int(torch.randint(prefix_length, (), generator=rng))
        vector = data[0, head, position]
        rms_source = data[..., :prefix_length, :]
        coordinates = {"batch": 0, "head": head, "prefix_position": position, "layer": layer, "kind": "V"}
    elif data.ndim == 3 and tuple(data.shape[:2]) == (1, 1):
        vector, rms_source, coordinates = data[0, 0], data, {"batch": 0, "sequence_position": 0}
    else:
        raise ValueError("Unsupported controlled-injection tensor")
    before = vector.detach().clone()
    rms = float(rms_source.float().square().mean().sqrt())
    if not math.isfinite(rms) or rms <= 0:
        raise RuntimeError("Controlled fault requires a finite, nonzero source RMS")
    delta = 8 * rms * torch.randn(vector.numel(), generator=rng, dtype=torch.float32)
    vector.copy_((before.float() + delta.to(vector.device)).to(vector.dtype))
    changed = int((before != vector).sum())
    if not changed or not bool(torch.isfinite(vector).all()):
        raise RuntimeError("Fault did not produce a finite changed vector")
    return {"target": target, "seed": int(seed), "rms": rms,
            "before": before.float().cpu().tolist(), "requested_delta": delta.tolist(),
            "after": vector.float().cpu().tolist(), "changed_elements": changed, **coordinates}


@torch.no_grad()
def injections_after_step(tx, engine, scenario, window_index, step_index, seed, cuts, method):
    """Fault coordinates remain inside this harness, never passed to controller."""
    if scenario == "clean" or (window_index != 1 and scenario != "repeated_prefix"):
        return []
    prefix = int(tx.checkpoint.get_seq_length())
    later = min(engine.num_layers - 1, 3 * engine.num_layers // 4)
    seed = int(seed) + 1009 * window_index
    faults = []
    if step_index == 0 and scenario in ("single_prefix", "repeated_prefix", "journal_corruption"):
        faults.append(perturb_vector(tx.working.value_cache[later], seed, "working_prefix",
                                     prefix_length=prefix, layer=later))
    if scenario == "multi_prefix" and step_index in (0, 1):
        layer = engine.num_layers - 2 if step_index == 0 else max(0, engine.num_layers // 2 - 1)
        faults.append(perturb_vector(tx.working.value_cache[layer], seed + step_index,
                                     "working_prefix", prefix_length=prefix, layer=layer))
    if step_index == 1 and scenario == "checkpoint_corruption":
        faults.append(perturb_vector(tx.checkpoint.value_cache[later], seed, "checkpoint",
                                     prefix_length=prefix, layer=later))
    if step_index == 1 and scenario == "journal_corruption" and method == "sparse":
        cut = max(c for c in cuts if c <= later)
        entry = tx.journals[0][cut]
        record = perturb_vector(entry.hidden, seed + 2, "journal")
        record.update(cut=cut, journal_step=0, journal_position=entry.position, journal_token=entry.token)
        faults.append(record)
    for record in faults:
        record.update(after_step=step_index + 1, window_index=window_index)
    return faults


@torch.no_grad()
def verify_commit(engine, session, result, reference, expected_committed):
    continuation = native_continuation(engine, clone_cache(session.cache), session.next_token)
    checks = {"cache": compare_caches(session.cache, reference.cache),
              "logits": tensor_check(session.last_logits, reference.logits),
              "tokens_equal": tuple(result.tokens) == reference.tokens,
              "committed_tokens_equal": tuple(session.committed_tokens) == tuple(expected_committed),
              "next_token_equal": session.next_token == reference.tokens[-1],
              "continuation_cache": compare_caches(continuation[0], reference.continuation_cache),
              "continuation_logits": tensor_check(continuation[1], reference.continuation_logits),
              "continuation_tokens_equal": continuation[2] == reference.continuation_tokens,
              "not_poisoned": not session.poisoned}
    checks["all_equal"] = all(value["equal"] if isinstance(value, dict) else value for value in checks.values())
    return checks


@torch.no_grad()
def run_session(engine, checkpoint, initial, references, *, method, scenario, window, seed,
                cuts, repetition, workload_id, model_name, order):
    # The common input fork and Session constructor are excluded, explicitly.
    session = TransactionalSession(engine, clone_cache(checkpoint), initial, policy=method, cuts=cuts)
    row = {"workload_id": workload_id, "model": model_name, "prefix_tokens": int(checkpoint.get_seq_length()),
           "num_layers": engine.num_layers,
           "window": window, "n_windows": len(references), "seed": seed, "scenario": scenario,
           "method": method, "repetition": repetition, "method_order": list(order),
           "sparse_cuts": list(cuts), "windows": [], "valid": True,
           "same_policy_treatment": scenario != "journal_corruption"}
    expected_committed = []
    for wi, reference in enumerate(references):
        before_tokens = tuple(session.committed_tokens)
        before_windows = session.windows_committed
        prefix_payload = cache_nbytes(session.cache)
        faults, injection_seconds = [], 0.0
        sync(engine)
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            allocated_before = torch.cuda.memory_allocated(engine.device)
        else:
            allocated_before = 0
        start = time.perf_counter()
        tx, result, rejection = None, None, None
        no_early_commit = True
        try:
            tx = session.begin(window)
            for step in range(window):
                tx.step()
                # Atomicity is observable without reading a model's working state.
                no_early_commit &= tuple(session.committed_tokens) == before_tokens
                if scenario != "clean" and (wi == 1 or scenario == "repeated_prefix") and step < 2:
                    sync(engine)
                    injection_start = time.perf_counter()
                    injected = injections_after_step(tx, engine, scenario, wi, step, seed, cuts, method)
                    sync(engine)
                    injection_seconds += time.perf_counter() - injection_start
                    faults.extend(injected)
            result = tx.finish()
        except WindowRejected as error:
            rejection = {"type": type(error).__name__, "reason": error.reason,
                         "controller_metadata": error.metadata}
        sync(engine)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(engine.device) if engine.device.type == "cuda" else 0
        measured = {"window_index": wi, "wall_seconds": elapsed,
                    "injection_harness_seconds": injection_seconds,
                    "wall_minus_injection_seconds": elapsed - injection_seconds,
                    "faults": faults, "no_early_commit": no_early_commit,
                    "committed_before": len(before_tokens), "committed_after": len(session.committed_tokens),
                    "logical_input_cache_bytes": prefix_payload,
                    "logical_checkpoint_bytes": cache_nbytes(tx.checkpoint) if tx is not None else None,
                    "logical_journal_bytes": journal_nbytes(tx.journals) if tx is not None else None,
                    "allocated_before_bytes": allocated_before, "peak_allocated_bytes": peak,
                    "peak_increment_bytes": peak - allocated_before}
        # Harness-only post-timing evidence. These speculative decisions were
        # never returned by step() or appended to the committed transcript.
        measured["speculative_tokens"] = list(tx._outputs) if tx is not None else []
        if rejection:
            expected = (scenario == "checkpoint_corruption" and wi == 1
                        and rejection["reason"] == "Checkpoint integrity failure"
                        and any(f["target"] == "checkpoint" for f in faults))
            previous = references[wi - 1] if wi else None
            checks = {"expected_rejection": expected, "poisoned": session.poisoned,
                      "committed_tokens_unchanged": tuple(session.committed_tokens) == before_tokens,
                      "committed_windows_unchanged": session.windows_committed == before_windows,
                      "committed_cache_unchanged": bool(previous and compare_caches(session.cache, previous.cache)["equal"]),
                      "committed_logits_unchanged": bool(previous and tensor_check(session.last_logits, previous.logits)["equal"]),
                      "next_token_unchanged": bool(previous and session.next_token == previous.tokens[-1]),
                      "no_early_commit": no_early_commit}
            measured.update(status="rejected", rejection=rejection, checks=checks, valid=all(checks.values()))
            row["windows"].append(measured)
            row.update(status="expected_abort" if measured["valid"] else "invalid", valid=measured["valid"])
            break
        expected_committed.extend(reference.tokens)
        checks = verify_commit(engine, session, result, reference, expected_committed)
        injected_layers = sorted({f["layer"] for f in faults if f["target"] == "working_prefix"})
        checks["detected_layers_equal_injection"] = list(result.detected_layers) == injected_layers
        checks["recovery_expected"] = bool(result.recovered) == bool(injected_layers)
        checks["committed_windows_equal"] = session.windows_committed == wi + 1
        checks["no_early_commit"] = no_early_commit
        checks["unexpected_checkpoint_commit"] = not (scenario == "checkpoint_corruption" and wi == 1)
        if scenario == "journal_corruption" and wi == 1 and method == "sparse":
            checks["journal_corruption_fell_back"] = bool(result.fallback_reason) and result.selected_cut == 0
        checks["all_equal"] = all(v["equal"] if isinstance(v, dict) else v for v in checks.values())
        measured.update(status="committed", valid=checks["all_equal"], checks=checks,
                        tokens=list(result.tokens), detected_layers=list(result.detected_layers),
                        selected_cut=result.selected_cut, replay_starts=list(result.replay_starts),
                        first_divergence=result.first_divergence, fallback_reason=result.fallback_reason,
                        recovered=result.recovered, controller_metadata=result.metadata)
        row["windows"].append(measured)
        # Keep no previous window's checkpoint/journals alive during the next
        # window's peak-memory baseline or timing. Only committed state persists.
        del tx, result
        if not measured["valid"]:
            row.update(status="invalid", valid=False)
            break
    row.setdefault("status", "complete")
    row.update(committed_tokens=list(session.committed_tokens), windows_committed=session.windows_committed,
               poisoned=session.poisoned, total_wall_seconds=sum(w["wall_seconds"] for w in row["windows"]),
               total_injection_harness_seconds=sum(w["injection_harness_seconds"] for w in row["windows"]))
    return row


def validate_config(config, records):
    for key in ("models", "prefixes", "windows", "seeds", "scenarios", "methods"):
        if not isinstance(config.get(key), list) or not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be a nonempty list with unique entries")
    if set(config["models"]) - {r["model"] for r in records}:
        raise ValueError("Configured model absent from manifest")
    if set(config["scenarios"]) - set(SCENARIOS) or set(config["methods"]) != {"full", "sparse"}:
        raise ValueError("Unknown scenarios or missing paired policies")
    if any(not isinstance(v, int) or v < 2 for v in config["prefixes"] + config["windows"]):
        raise ValueError("Prefixes and windows must be integers >= 2")
    if any(not isinstance(v, int) or v < 0 for v in config["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if int(config.get("n_windows", 0)) < 2 or int(config.get("repetitions", 0)) < 1:
        raise ValueError("Need at least two windows and one repetition")
    if config.get("warmups", 1) not in (0, 1):
        raise ValueError("Warmup setting must be 0 or 1")
    for record in records:
        if record["model"] in config["models"]:
            revision = record["revision"]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError("Model revisions must be immutable 40-character commits")
            if not Path(record["local_path"]).is_dir():
                raise ValueError(f"Missing offline model directory for {record['model']}")


def model_file_hashes(path):
    root = Path(path)
    suffixes = {".safetensors", ".json", ".model", ".txt", ".tiktoken"}
    return [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in suffixes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise RuntimeError("Refusing to overwrite a nonempty output path")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {p.name: digest(p) for p in sorted(HERE.glob("*.py"))}
    for dependency in ("recut_engine.py", "prefix_detector.py", "run_recut.py"):
        sources["../experiments/" + dependency] = digest(HERE.parent / "experiments" / dependency)
    metadata = {"status": "starting", "started_unix_seconds": time.time(),
                "config_sha256": digest(args.config), "manifest_sha256": digest(args.manifest),
                "source_sha256": sources, "python": platform.python_version(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "device": args.device, "dtype": args.dtype, "cuda_runtime": torch.version.cuda,
                "timing_scope": "BEGIN through finish, including artificial fault injection, guards, checkpoint, journals, delayed detection, recovery and COMMIT; common session input fork, constructor, prefill, parity and oracle checks excluded",
                "adjusted_timing_scope": "Wall time minus separately measured injection harness; residual harness bookkeeping remains. Not measured deployment latency.",
                "memory_scope": "Logical payloads exclude Python metadata/allocator; GPU peaks include resident native correctness fixtures and model",
                "equality": "Exact finite elementwise tensors, greedy tokens, and native two-step continuation",
                "oracle_storage": "Native gold tensors held live; persisted hashes and per-window comparison records, not full gold tensors",
                "models_completed": [], "warmup_results": [], "workloads": []}
    write_json(args.output / "metadata.json", metadata)
    try:
        config = json.loads(args.config.read_text())
        records = json.loads(args.manifest.read_text())
        validate_config(config, records)
        if args.device == "cuda" and (not torch.cuda.is_available() or
                (args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported())):
            raise RuntimeError("Requested CUDA/BF16 capability unavailable")
        torch.set_grad_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        dtype = getattr(torch, args.dtype)
        metadata.update(status="running", config=config, manifest=records, deterministic_algorithms=True,
                        tf32=False, attention="eager", batch_size=1,
                        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        if args.device == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name(0)
        expected = len(config["models"]) * len(config["prefixes"]) * len(config["windows"]) * len(config["seeds"])
        metadata["expected_sessions"] = expected * len(config["scenarios"]) * len(config["methods"]) * config["repetitions"]
        completed = 0
        with (args.output / "sessions.jsonl").open("w") as output, (args.output / "references.jsonl").open("w") as ref_output:
            for record in records:
                name = record["model"]
                if name not in config["models"]:
                    continue
                source = record["local_path"]
                metadata.setdefault("model_files", {})[name] = model_file_hashes(source)
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
                engine = HFIncrementalEngine(model)
                metadata.setdefault("model_configs", {})[name] = model.config.to_dict()
                cuts = tuple(sorted({c for c in (engine.num_layers // 4, engine.num_layers // 2,
                                                 3 * engine.num_layers // 4) if 0 < c < engine.num_layers}))
                if not cuts:
                    raise RuntimeError("Stage-2 pretrained protocol needs nonzero sparse cuts")
                warmed = set()
                for prefix_length, window, seed in itertools.product(config["prefixes"], config["windows"], config["seeds"]):
                    workload_id = f"{name.replace('/', '--')}-p{prefix_length}-w{window}-s{seed}"
                    prompt = config.get("prompt", DEFAULT_PROMPT)
                    ids = tokenizer.encode(prompt, add_special_tokens=True)
                    if not ids:
                        raise ValueError("Empty encoded prompt")
                    prefix = (ids * ((prefix_length + len(ids) - 1) // len(ids)))[:prefix_length]
                    checkpoint, initial, refs, parity = build_references(engine, prefix, window, config["n_windows"])
                    ref_record = {"workload_id": workload_id, "model": name, "revision": record["revision"],
                                  "prefix_ids": prefix, "initial_token": initial, "parity": parity,
                                  "checkpoint": cache_digest(checkpoint), "windows": []}
                    for index, ref in enumerate(refs):
                        ref_record["windows"].append({"window_index": index, "tokens": ref.tokens,
                            "cache": cache_digest(ref.cache), "logits": tensor_digest(ref.logits),
                            "continuation_tokens": ref.continuation_tokens,
                            "continuation_cache": cache_digest(ref.continuation_cache),
                            "continuation_logits": tensor_digest(ref.continuation_logits)})
                    append_json(ref_output, ref_record)
                    metadata["workloads"].append({"id": workload_id, "parity_positions": len(parity), "passed": True})
                    for method in config["methods"]:
                        if method not in warmed and config.get("warmups", 1):
                            warmup = run_session(engine, checkpoint, initial, refs[:1], method=method,
                                scenario="clean", window=window, seed=seed, cuts=cuts, repetition=-1,
                                workload_id=workload_id, model_name=name, order=(method,))
                            metadata["warmup_results"].append(warmup)
                            if not warmup["valid"]:
                                raise RuntimeError("Untimed warmup correctness failed")
                            warmed.add(method)
                    for scenario in config["scenarios"]:
                        for rep in range(config["repetitions"]):
                            methods = list(config["methods"])
                            offset = rep % len(methods)
                            order = methods[offset:] + methods[:offset]
                            for method in order:
                                metadata["active_session"] = {"workload": workload_id, "scenario": scenario,
                                                              "repetition": rep, "method": method}
                                write_json(args.output / "metadata.json", metadata)
                                row = run_session(engine, checkpoint, initial, refs, method=method,
                                    scenario=scenario, window=window, seed=seed, cuts=cuts, repetition=rep,
                                    workload_id=workload_id, model_name=name, order=order)
                                row.update(model_revision=record["revision"], config_sha256=metadata["config_sha256"],
                                           source_sha256=sources)
                                append_json(output, row)
                                completed += 1
                                print(json.dumps({"session": completed, "of": metadata["expected_sessions"],
                                      **metadata["active_session"], "status": row["status"], "valid": row["valid"]}), flush=True)
                                if not row["valid"]:
                                    raise RuntimeError("Session correctness failed; preserving evidence and stopping")
                    del refs, parity, checkpoint, ref_record
                metadata["models_completed"].append(name)
                write_json(args.output / "metadata.json", metadata)
                del engine, model, tokenizer
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        final_sources = {p.name: digest(p) for p in sorted(HERE.glob("*.py"))}
        for dependency in ("recut_engine.py", "prefix_detector.py", "run_recut.py"):
            final_sources["../experiments/" + dependency] = digest(HERE.parent / "experiments" / dependency)
        if sources != final_sources or digest(args.config) != metadata["config_sha256"] or digest(args.manifest) != metadata["manifest_sha256"]:
            raise RuntimeError("Source/config/manifest changed during run")
        if completed != metadata["expected_sessions"]:
            raise RuntimeError("Session count differs from frozen matrix")
        metadata.pop("active_session", None)
        metadata.update(status="complete", sessions_completed=completed, finished_unix_seconds=time.time(),
                        sessions_sha256=digest(args.output / "sessions.jsonl"),
                        references_sha256=digest(args.output / "references.jsonl"))
        write_json(args.output / "metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc(), finished_unix_seconds=time.time())
        write_json(args.output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
