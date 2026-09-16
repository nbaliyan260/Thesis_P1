"""Post-hoc snapshot of an observed partial-replay/fallback boundary; not timing evidence."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from recut_engine import HFIncrementalEngine
from run_recut import digest, native_parity, prefix_ids, run_case, serializable, write_json

BUDGET = 20 * 1024 * 1024


def select_case(rows):
    eligible = [r for r in rows if r["first_divergence"] is not None
                and r["selected_sparse_cut"] > 0
                and r["case"]["window"] > r["first_divergence"]]
    return min(eligible, key=lambda r: (r["case_id"], r["model"])) if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary-run", "manifest", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--amendment", type=Path, help="Explicitly opt into the separately frozen W12-to-W14 diagnostic")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite an existing output path")
    primary = json.loads((args.primary_run / "metadata.json").read_text())
    raw = (args.primary_run / "episodes.jsonl").read_bytes()
    config, records = json.loads(args.config.read_text()), json.loads(args.manifest.read_text())
    source_dir = Path(__file__).resolve().parent
    sources = {name: digest(source_dir / name) for name in
               ("capture_boundary_case.py", "recut_engine.py", "run_recut.py", "audit_results.py", "audit_boundary_snapshot.py")}
    raw_hash = hashlib.sha256(raw).hexdigest()
    if primary["status"] != "complete" or raw_hash != primary["episodes_sha256"]:
        raise RuntimeError("Primary must be complete with a matching raw digest")
    if (digest(args.config) != primary["config_sha256"] or
            digest(args.manifest) != primary["manifest_sha256"] or
            sources["recut_engine.py"] != primary["engine_sha256"] or
            sources["run_recut.py"] != primary["runner_sha256"]):
        raise RuntimeError("Primary config, manifest, engine or runner changed")
    if any(len(r["revision"]) != 40 or any(c not in "0123456789abcdef" for c in r["revision"])
           for r in records):
        raise RuntimeError("Every model must have a pinned immutable revision")
    rows = [json.loads(line) for line in raw.splitlines()]
    pairs = {(r["model"], r["case_id"]) for r in rows}
    names = config.get("models", [r["model"] for r in records])
    expected = {(r["model"], c["id"]) for r in records if r["model"] in names
                for c in config["cases"] if c.get("model", r["model"]) == r["model"]}
    if len(pairs) != len(rows) or pairs != expected or not all(r["valid_repairs_equal"] for r in rows):
        raise RuntimeError("Primary coverage or supported-path correctness failed")
    selected = select_case(rows)
    amendment = json.loads(args.amendment.read_text()) if args.amendment else None
    if amendment is not None:
        required = dict(primary_raw_sha256=raw_hash, model="Qwen/Qwen2.5-0.5B", case_id="primary-063",
            original_window=12, diagnostic_window=14, expected_first_divergence=12,
            expected_sparse_cut=6, expected_fault_layer=11, new_fault_samples=False, performance_evidence=False)
        if any(amendment.get(k) != v for k, v in required.items()):
            raise RuntimeError("Amendment differs from the authorized one-case diagnostic")
        selected = next(r for r in rows if (r["model"], r["case_id"]) == (amendment["model"], amendment["case_id"]))
        if (selected["case"]["window"], selected["first_divergence"], selected["selected_sparse_cut"], selected["layer"]) != (12, 12, 6, 11):
            raise RuntimeError("The designated observed primary boundary no longer matches")
    metadata = dict(status="starting", study="posthoc_observed_boundary_snapshot",
        performance_evidence=False, new_fault_samples=False, snapshot_max_bytes=BUDGET,
        selection="first lexicographic (case_id, model) with divergence, sparse cut>0 and W>j",
        primary_raw_sha256=raw_hash, primary_metadata_sha256=digest(args.primary_run / "metadata.json"),
        config_sha256=digest(args.config), manifest_sha256=digest(args.manifest), source_sha256=sources,
        observed_divergences=[{k: r[k] for k in ("model", "case_id", "first_divergence", "selected_sparse_cut")}
                              | {"window": r["case"]["window"]} for r in rows if r["first_divergence"] is not None],
        diagnostic_overrides={"repetitions": 1, "overhead_repetitions": 1},
        timings="Diagnostic only: ignored for all performance claims; frozen config is not modified")
    if amendment is not None:
        metadata.update(study=amendment["study"], amendment=amendment, amendment_sha256=digest(args.amendment),
                        selection="Explicit post-hoc primary-063/Qwen W12-to-W14 amendment, not a search")
    args.output.mkdir(parents=True)
    write_json(args.output / "metadata.json", metadata)
    try:
        if selected is None:
            metadata.update(status="skipped", reason="No primary case meets the declared actual-fallback criterion")
        else:
            record = next(r for r in records if r["model"] == selected["model"])
            case = next(c for c in config["cases"] if c["id"] == selected["case_id"])
            if case != selected["case"] or record["revision"] != selected["revision"]:
                raise RuntimeError("Selected primary case/revision mismatch")
            original_window = case["window"]
            case = dict(case, window=amendment["diagnostic_window"]) if amendment else case
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Capture requires a BF16-capable CUDA GPU")
            torch.set_grad_enabled(False)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            numeric = dict(torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), deterministic_algorithms=True,
                tf32=False, attention="eager", dtype="bfloat16",
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
            if any(primary.get(k) != v for k, v in numeric.items()):
                raise RuntimeError("Numerical backend differs from the primary")
            metadata.update(numeric=numeric, model=record, case_id=case["id"])
            tokenizer = AutoTokenizer.from_pretrained(record["local_path"], local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(record["local_path"], local_files_only=True,
                trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                attn_implementation="eager").to("cuda").eval()
            engine = HFIncrementalEngine(model)
            metadata["model_config"] = model.config.to_dict()
            metadata["native_parity"] = native_parity(engine, prefix_ids(tokenizer, case), case["window"] + 2)
            if not metadata["native_parity"]["passed"]:
                raise RuntimeError("Native parity failed")
            diagnostic_config = dict(config, **metadata["diagnostic_overrides"])
            row, states = run_case(engine, tokenizer, case, diagnostic_config, capture_audit=True)
            for key in ("fault", "prefix_ids", "initial_token",
                        "first_divergence", "selected_sparse_cut", "layer", "num_layers"):
                if row[key] != selected[key]:
                    raise RuntimeError("Reconstruction differs from primary: " + key)
            if any(row[k][:original_window] != selected[k] for k in ("reference_tokens", "faulty_tokens")):
                raise RuntimeError("Original-window decisions differ from primary")
            if amendment and any(row["methods"][m]["start_layers"] != [c] * 12 + [0] * 2
                                 for m, c in (("full", 0), ("sparse", 6), ("all", 11))):
                raise RuntimeError("Expected post-divergence fallback at transition 13 was not observed")
            if not row["valid_repairs_equal"] or states is None:
                raise RuntimeError("Diagnostic repair or capture failed")
            row.update(model=record["model"], revision=record["revision"], config_sha256=metadata["config_sha256"])
            write_json(args.output / "diagnostic_row.json", row)
            metadata["diagnostic_row_sha256"] = digest(args.output / "diagnostic_row.json")
            buffer = io.BytesIO()
            torch.save({"model": record, "episode": serializable(row), "states": states}, buffer)
            blob = buffer.getvalue()
            if len(blob) > BUDGET:
                metadata.update(status="skipped", reason="Serialized snapshot exceeds 20 MiB", requested_bytes=len(blob))
            else:
                path = args.output / "boundary_snapshot.pt"
                path.write_bytes(blob)
                metadata.update(status="complete", archive={"file": path.name, "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(), "model": record["model"], "case_id": case["id"]},
                    reconstruction_matches_primary=True, independent_cpu_audit="pending")
    except Exception as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(args.output / "metadata.json", metadata)
        raise
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("status", "study")} | {"reason": metadata.get("reason")}), flush=True)


if __name__ == "__main__":
    main()
