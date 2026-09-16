"""Post-hoc restoration-baseline sensitivity; never replaces frozen primary data.

Five policies share the excluded working-cache clone; alias-full does not use
that clone. Timed boundaries include policy-specific restoration and replay.
Prefix views retain their original backing allocation until append replaces
them. Checkpoint aliases require resident trusted immutable GPU tensors and
ordinary DynamicCache's out-of-place torch.cat update. Prefix comparison is a
separate narrowly scoped feasibility check, not a general online detector.
"""
from __future__ import annotations
import argparse
import hashlib
import json
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
from recut_engine import HFIncrementalEngine, clone_cache
from prefix_detector import measure_prefix_detection
from run_recut import (Trajectory, decode, digest, inject_fault, memory_end,
    memory_start, native_parity, prefix_ids, repair, serializable, sync, verify,
    write_json)

METHODS = ("full", "layer_view_full", "checkpoint_alias_full", "sparse", "all")


def unchanged(a, b):
    """Raw snapshot immutability, including stable non-finite bit patterns."""
    if len(a) != len(b) or a._seen_tokens != b._seen_tokens:
        return False
    return all(x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
        and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        for x, y in zip(a.key_cache + a.value_cache, b.key_cache + b.value_cache))


def prefix_views(working, checkpoint, fault_layer):
    """Crop without copying, then restore whole known layer; no coordinate input."""
    if type(working) is not DynamicCache or type(checkpoint) is not DynamicCache:
        raise TypeError("Only ordinary DynamicCache is supported")
    if len(working) != len(checkpoint) or not 0 <= fault_layer < len(working):
        raise ValueError("Layer/count mismatch")
    length = int(checkpoint.get_seq_length())
    if any(checkpoint.get_seq_length(i) != length or working.get_seq_length(i) < length
           for i in range(len(working))):
        raise ValueError("Prefix length mismatch")
    owners = (tuple(working.key_cache), tuple(working.value_cache))
    working.key_cache = [x[..., :length, :] for x in owners[0]]
    working.value_cache = [x[..., :length, :] for x in owners[1]]
    working.key_cache[fault_layer] = checkpoint.key_cache[fault_layer].detach().clone()
    working.value_cache[fault_layer] = checkpoint.value_cache[fault_layer].detach().clone()
    working._seen_tokens = length
    return owners


@torch.no_grad()
def layer_view_full(engine, working, checkpoint, initial, faulty, layer):
    owners = prefix_views(working, checkpoint, layer)
    token, tokens, starts = initial, [], []
    for index in range(len(faulty.tokens)):
        result = engine.step(token, working)
        token = result.next_token
        if index == 0:
            owners = None  # Every prefix view has now been replaced by cat.
        tokens.append(token)
        starts.append(0)
    divergence = next((i + 1 for i, (a, b) in enumerate(zip(tokens, faulty.tokens)) if a != b), None)
    return Trajectory(working, tokens, result.logits, [], starts, divergence)


def alias_checkpoint(checkpoint):
    """Fresh cache lists alias immutable checkpoint tensors until first append."""
    if type(checkpoint) is not DynamicCache:
        raise TypeError("Only ordinary DynamicCache is supported")
    length = int(checkpoint.get_seq_length())
    if any(checkpoint.get_seq_length(i) != length for i in range(len(checkpoint))):
        raise ValueError("Inconsistent checkpoint")
    cache = DynamicCache()
    cache.key_cache = list(checkpoint.key_cache)
    cache.value_cache = list(checkpoint.value_cache)
    cache._seen_tokens = length
    return cache


@torch.no_grad()
def checkpoint_alias_full(engine, checkpoint, initial, faulty):
    result = decode(engine, alias_checkpoint(checkpoint), initial, len(faulty.tokens))
    result.first_divergence = next((i + 1 for i, (a, b) in
        enumerate(zip(result.tokens, faulty.tokens)) if a != b), None)
    return result


@torch.no_grad()
def run_sensitivity_case(engine, tokenizer, case, primary_row, repetitions=5, detection_only=False):
    count, window = engine.num_layers, int(case["window"])
    layer = int(float(case["layer_fraction"]) * (count - 1))
    cuts = sorted({c for c in (count // 4, count // 2, 3 * count // 4) if 0 < c < count})
    ids = prefix_ids(tokenizer, case)
    if ids != primary_row["prefix_ids"]:
        raise RuntimeError("Reconstructed prefix differs from primary evidence")
    checkpoint, logits = engine.prefill(ids)
    initial = int(logits[0, -1].argmax())
    reference = decode(engine, clone_cache(checkpoint), initial, window)
    continuation = decode(engine, clone_cache(reference.cache), reference.tokens[-1], 2)
    faulty_cache = clone_cache(checkpoint)
    fault, _ = inject_fault(faulty_cache, layer, case)
    faulty = decode(engine, faulty_cache, initial, window, range(1, count))
    if (fault != primary_row["fault"] or reference.tokens != primary_row["reference_tokens"]
            or faulty.tokens != primary_row["faulty_tokens"] or initial != primary_row["initial_token"]):
        raise RuntimeError("Reconstructed fault/trajectory differs from primary evidence")
    checkpoint_guard, faulty_guard = clone_cache(checkpoint), clone_cache(faulty.cache)
    detection = measure_prefix_detection(engine, faulty.cache, checkpoint)
    detected = detection["detected_layers"]
    if len(detected) > 1:
        raise RuntimeError("Multiple changed layers: fail closed")
    # Ground truth is an audit check only; policy receives detector output below.
    if detected != ([layer] if fault["changed_elements"] else []):
        raise RuntimeError("Prefix detector disagrees with controlled injection")
    localized = detected[0] if detected else None
    cut = max([0] + [c for c in cuts if localized is not None and c <= localized])
    detection.update(checkpoint_unchanged=unchanged(checkpoint, checkpoint_guard),
                     initial_faulty_unchanged=unchanged(faulty.cache, faulty_guard))
    if not detection["checkpoint_unchanged"] or not detection["initial_faulty_unchanged"]:
        raise RuntimeError("Detector modified input snapshots")
    row = dict(case_id=case["id"], case=case, layer=layer, num_layers=count,
        selected_sparse_cut=cut, fault=fault, prefix_ids=ids, initial_token=initial,
        reconstruction_matches_primary=True,
        reference_tokens=reference.tokens, faulty_tokens=faulty.tokens, timings=[],
        detection=detection, detected_layer=localized, detection_only=detection_only)
    if detection_only:
        row["valid"] = True
        return row
    for rep in range(-1, repetitions):
        order = list(METHODS) if rep < 0 else list(METHODS[rep % 5:] + METHODS[:rep % 5])
        for method in order:
            working = clone_cache(faulty.cache)  # Same excluded setup, unused by alias.
            sync(engine)
            baseline = memory_start(engine)
            start = time.perf_counter()
            if method == "layer_view_full":
                recovered = (layer_view_full(engine, working, checkpoint, initial, faulty, localized)
                    if localized is not None else checkpoint_alias_full(engine, checkpoint, initial, faulty))
            elif method == "checkpoint_alias_full":
                recovered = checkpoint_alias_full(engine, checkpoint, initial, faulty)
            else:
                selected = {"full": 0, "sparse": cut, "all": localized or 0}[method]
                recovered = repair(engine, working, checkpoint, initial, faulty, localized or 0, selected)
            sync(engine)
            elapsed = time.perf_counter() - start
            memory = memory_end(engine, baseline)
            checks = verify(engine, recovered, reference, continuation)
            checks.update(checkpoint_unchanged=unchanged(checkpoint, checkpoint_guard),
                initial_faulty_unchanged=unchanged(faulty.cache, faulty_guard),
                start_layers=recovered.starts, first_divergence=recovered.first_divergence,
                layer_steps=sum(count - c for c in recovered.starts))
            valid = checks["all_equal"] and checks["checkpoint_unchanged"] and checks["initial_faulty_unchanged"]
            if rep >= 0 or not valid:
                row["timings"].append(dict(method=method, repetition=rep, order=order,
                    seconds=elapsed, checks=checks, **memory))
            del recovered, working
            if not valid:
                row.update(valid=False, failure=dict(method=method, repetition=rep))
                return row
    row["valid"] = True
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "primary-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty output directory")
    primary = json.loads((args.primary_run / "metadata.json").read_text())
    raw = (args.primary_run / "episodes.jsonl").read_bytes()
    config = json.loads(args.config.read_text())
    records = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for line in raw.splitlines()]
    primary_hash = hashlib.sha256(raw).hexdigest()
    if primary["status"] != "complete" or primary_hash != primary["episodes_sha256"]:
        raise RuntimeError("Primary run must be complete with matching raw hash")
    if digest(args.config) != primary["config_sha256"] or digest(args.manifest) != primary["manifest_sha256"]:
        raise RuntimeError("Use the exact primary config and model manifest")
    directory = Path(__file__).resolve().parent
    sources = {name: digest(directory / name) for name in
        ("run_optimized_replay.py", "prefix_detector.py", "recut_engine.py", "run_recut.py")}
    if sources["recut_engine.py"] != primary["engine_sha256"] or sources["run_recut.py"] != primary["runner_sha256"]:
        raise RuntimeError("Primary engine/runner source changed")
    model_names = config.get("models", [r["model"] for r in records])
    expected = {(r["model"], c["id"]) for r in records if r["model"] in model_names
                for c in config["cases"] if c.get("model", r["model"]) == r["model"]}
    observed = {(r["model"], r["case_id"]) for r in rows}
    if len(rows) != len(observed) or observed != expected or len(rows) != 216:
        raise RuntimeError("Expected all 216 unique frozen primary model/case pairs")
    if not all(r["valid_repairs_equal"] for r in rows):
        raise RuntimeError("Primary supported-path correctness did not pass")
    targets = {(r["model"], r["case_id"]): r for r in rows if r["case"]["fault"] != "noop"}
    by_key = {(r["model"], r["case_id"]): r for r in rows}
    if len(targets) != 192:
        raise RuntimeError("Expected exactly 192 primary perturbed cases")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires one BF16-capable CUDA GPU")
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    numeric = dict(deterministic_algorithms=True, tf32=False, attention="eager", dtype="bfloat16",
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    environment = dict(torch=torch.__version__, transformers=transformers.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    if any(primary.get(k) != v for k, v in {**numeric, **environment}.items()):
        raise RuntimeError("Numerical backend/environment differs from primary metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(status="starting", study="posthoc_baseline_sensitivity",
        not_preregistered=True, primary_raw_sha256=primary_hash,
        primary_metadata_sha256=digest(args.primary_run / "metadata.json"),
        config_sha256=digest(args.config), manifest_sha256=digest(args.manifest),
        source_sha256=sources, methods=list(METHODS), repetitions=5, warmups=1,
        expected_cases=192, models=records, python=platform.python_version(),
        **environment, **numeric,
        equality="exact elementwise; immutable-snapshot guards use raw bytes",
        timing_boundary="Common working-cache clone excluded, even when unused by alias; restoration/replay included; diagnostics excluded",
        memory_scope="Fixture-inclusive; views retain backing storage until replaced; no claim of early suffix-memory release",
        checkpoint_assumption="Trusted immutable resident GPU checkpoint; alias needs no layer/coordinate privilege",
        sparse_all_scope="Original conservative restoration retained; not their optimized attainable frontier",
        detector_scope="Persistent post-checkpoint old-prefix changes only; no suffix/transient/pre-checkpoint coverage",
        detector_timing="Separate 5 repetitions plus warmup; outside recovery timing; one consolidated layer-decision transfer",
        no_alarm_policy="Conservative cut0; layer_view_full falls back to checkpoint_alias_full",
        expected_detection_cases=216, native_parity={}, cases_completed=0, detection_cases_completed=0)
    write_json(args.output / "metadata.json", metadata)
    try:
        metadata["status"] = "running"
        with (args.output / "episodes.jsonl").open("w") as output, (args.output / "detection.jsonl").open("w") as detections:
            for record in records:
                cases = [c for c in config["cases"] if (record["model"], c["id"]) in by_key]
                if not cases:
                    continue
                if len(record["revision"]) != 40:
                    raise ValueError("Expected pinned immutable model revision")
                tokenizer = AutoTokenizer.from_pretrained(record["local_path"], local_files_only=True, trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(record["local_path"], local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                    attn_implementation="eager").to("cuda").eval()
                engine = HFIncrementalEngine(model)
                longest = max(cases, key=lambda c: c["prefix_tokens"])
                parity = native_parity(engine, prefix_ids(tokenizer, longest),
                                       decode_steps=max(c["window"] for c in cases) + 2)
                metadata["native_parity"][record["model"]] = parity
                write_json(args.output / "metadata.json", metadata)
                if not parity["passed"]:
                    raise RuntimeError("Native parity failed")
                for case in cases:
                    prior = by_key[(record["model"], case["id"])]
                    if prior["revision"] != record["revision"] or prior["case"] != case:
                        raise RuntimeError("Primary case/revision mismatch")
                    row = run_sensitivity_case(engine, tokenizer, case, prior, detection_only=case["fault"] == "noop")
                    row.update(model=record["model"], revision=record["revision"], primary_raw_sha256=primary_hash)
                    detections.write(json.dumps(serializable({k: v for k, v in row.items() if k != "timings"}), allow_nan=False) + "\n")
                    detections.flush()
                    metadata["detection_cases_completed"] += 1
                    if not row["detection_only"]:
                        output.write(json.dumps(serializable(row), allow_nan=False) + "\n")
                        output.flush()
                        metadata["cases_completed"] += 1
                    print(json.dumps(dict(model=record["model"], case=case["id"], valid=row["valid"])), flush=True)
                    if not row["valid"]:
                        raise RuntimeError("Supplemental correctness/immutability failed")
                del engine, model, tokenizer
                torch.cuda.empty_cache()
        if metadata["cases_completed"] != 192 or metadata["detection_cases_completed"] != 216:
            raise RuntimeError("Supplemental case coverage mismatch")
        metadata.update(status="complete", episodes_sha256=digest(args.output / "episodes.jsonl"),
                        detection_sha256=digest(args.output / "detection.jsonl"))
    except Exception as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(args.output / "metadata.json", metadata)
        raise
    write_json(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
