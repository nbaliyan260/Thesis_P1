"""CPU tensor audit of a trusted local targeted archive; never imports the runner."""
import argparse
import json
from pathlib import Path
from audit_results import Audit, audit_row, audit_snapshot, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    for name in ("primary-run", "manifest", "config"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--amendment", type=Path, default=Path(__file__).with_name("boundary_extension_amendment.json"))
    args = parser.parse_args()
    metadata = read_json(args.run / "metadata.json")
    if metadata["status"] != "complete":
        raise RuntimeError("Only a completed captured archive can be audited")
    checks = [(args.primary_run / "episodes.jsonl", metadata["primary_raw_sha256"]),
                           (args.primary_run / "metadata.json", metadata["primary_metadata_sha256"]),
                           (args.manifest, metadata["manifest_sha256"]), (args.config, metadata["config_sha256"]),
                           (args.run / "diagnostic_row.json", metadata["diagnostic_row_sha256"])]
    checks.extend((Path(__file__).with_name(name), expected) for name, expected in metadata["source_sha256"].items())
    for path, expected in checks:
        if sha256(path) != expected:
            raise RuntimeError("Input digest mismatch: " + str(path))
    entry = metadata["archive"]
    if Path(entry["file"]).name != entry["file"]:
        raise RuntimeError("Archive must be an explicitly declared local filename")
    path = args.run / entry["file"]
    if path.is_symlink() or path.stat().st_size != entry["bytes"] or entry["bytes"] > 20 * 1024 * 1024 or sha256(path) != entry["sha256"]:
        raise RuntimeError("Archive digest, type or byte budget mismatch")
    row = read_json(args.run / "diagnostic_row.json")
    record = next(r for r in read_json(args.manifest) if r["model"] == row["model"])
    config = read_json(args.config)
    case = next(c for c in config["cases"] if c["id"] == row["case_id"])
    primary_rows = [json.loads(line) for line in (args.primary_run / "episodes.jsonl").read_bytes().splitlines()]
    original = next(r for r in primary_rows if (r["model"], r["case_id"]) == (row["model"], row["case_id"]))
    if metadata.get("amendment"):
        amendment = metadata["amendment"]
        if sha256(args.amendment) != metadata["amendment_sha256"] or read_json(args.amendment) != amendment:
            raise RuntimeError("Premeasurement amendment digest/content mismatch")
        if (row["model"], row["case_id"], case["window"], amendment["diagnostic_window"]) != ("Qwen/Qwen2.5-0.5B", "primary-063", 12, 14):
            raise RuntimeError("Unexpected amended boundary")
        case = dict(case, window=14)
    diagnostic_config = dict(config, **metadata["diagnostic_overrides"])
    audit = Audit()
    primary_metadata = read_json(args.primary_run / "metadata.json")
    audit.equal(metadata["model_config"], primary_metadata["model_configs"][row["model"]], "model_config", row["model"])
    numeric_keys = ("torch", "transformers", "cuda", "gpu", "deterministic_algorithms", "tf32", "attention", "dtype",
                    "allow_bf16_reduced_precision_reduction", "float32_matmul_precision", "cublas_workspace_config")
    audit.equal(metadata.get("numeric"), {k: primary_metadata[k] for k in numeric_keys}, "numeric_binding", "primary")
    parity = metadata["native_parity"]
    audit.equal(parity.get("passed"), True, "native_parity", "passed")
    audit.equal(parity.get("prefix_tokens"), case["prefix_tokens"], "native_parity", "prefix")
    audit.equal(parity.get("decode_steps"), case["window"] + 2, "native_parity", "tail")
    audit.equal([item["position"] for item in parity["trace"]],
                list(range(case["prefix_tokens"] + case["window"] + 2)), "native_parity", "positions")
    for item in parity["trace"]:
        for kind in ("cache", "logits"):
            check = item[kind]
            audit.equal((check.get("equal"), check.get("finite"), check.get("max_abs_error")),
                        (True, True, 0), "native_parity", str(item["position"]) + "/" + kind)
    for key in ("fault", "prefix_ids", "initial_token", "first_divergence", "selected_sparse_cut", "layer", "num_layers"):
        audit.equal(row[key], original[key], "primary_reconstruction", key)
    for key in ("reference_tokens", "faulty_tokens"):
        audit.equal(row[key][:original["case"]["window"]], original[key], "primary_reconstruction", key)
    if metadata.get("amendment"):
        audit.equal(row["first_divergence"], 12, "amended_boundary", "divergence")
        for method, cut in (("full", 0), ("sparse", 6), ("all", 11)):
            audit.equal(row["methods"][method]["start_layers"], [cut] * 12 + [0] * 2, "amended_boundary", method)
    audit_row(audit, row, case, diagnostic_config, record, metadata["model_config"])
    snapshot = audit_snapshot(audit, path, entry, row, record, metadata["model_config"])
    report = dict(status="passed" if not audit.failures else "failed", groups=dict(audit.groups),
        failures=audit.failures, warnings=audit.warnings, snapshot=snapshot,
        diagnostic_metadata_sha256=sha256(args.run / "metadata.json"),
        auditor_sha256=sha256(__file__), independent_helper_sha256=sha256(Path(__file__).with_name("audit_results.py")),
        scope="CPU saved KV/logit/tokens and trace audit; no independent model execution or archived continuation tensors")
    output = args.run / "independent_boundary_audit.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "failures": len(audit.failures)}), flush=True)
    if audit.failures:
        raise RuntimeError("Independent saved-state audit failed")


if __name__ == "__main__":
    main()
