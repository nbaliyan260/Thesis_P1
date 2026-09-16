"""Package the completed stage-3 artifact without weights or environments.

This is post-measurement delivery tooling, deliberately outside the frozen
experimental-source glob. It never rewrites raw evidence or protocol files.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
TITLE = "RECUT_04_Complete_Prototype_and_Validation"
MODELS = ("smol135m", "qwen05b", "qwen7b")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text())


def select_files():
    provenance = load(PROJECT / "stage3/reporting/manuscript_provenance.json")
    report = PROJECT / "artifacts" / (TITLE + ".md")
    qa = load(PROJECT / "artifacts/qa" / TITLE / "build.json")
    pdf = ROOT / "output/pdf" / (TITLE + ".pdf")
    if provenance["manuscript_sha256"] != digest(report) or qa["source_sha256"] != digest(report):
        raise ValueError("Stale manuscript/PDF source binding")
    if qa["pdf_sha256"] != digest(pdf) or qa["visual_review"] != "passed_all_pages":
        raise ValueError("PDF has not passed complete visual review")
    if (qa["builder_sha256"] != digest(PROJECT / "artifacts/build_pdf.py")
            or qa["layout_sha256"] != digest(PROJECT / "artifacts/pdf_layout.py")):
        raise ValueError("PDF builder/layout changed after rendering")
    validation = PROJECT / "runs/stage3-fallback-diagnostic-v1/smol135m/validation_only.json"
    if (load(validation)["status"] != "passed_validation_only"
            or provenance["selected_diagnostic_sha256"] != digest(validation)):
        raise ValueError("Unverified selected diagnostic")
    amendment = PROJECT / "notes/stage3_fallback_diagnostic_amendment.md"
    if provenance["selected_diagnostic_amendment_sha256"] != digest(amendment):
        raise ValueError("Changed selected diagnostic amendment")
    freeze = load(PROJECT / "stage3/protocol_freeze.json")
    selected = {PROJECT / relative for relative in freeze["files_sha256"]}
    for relative, expected in freeze["files_sha256"].items():
        if digest(PROJECT / relative) != expected:
            raise ValueError("Frozen source mismatch: " + relative)
    for phase, models in (("debug", MODELS), ("main", MODELS), ("fallback-diagnostic", ("smol135m",))):
        for model in models:
            directory = PROJECT / "runs" / ("stage3-" + phase + "-v1") / model
            metadata = load(directory / "metadata.json")
            if metadata["status"] != "complete":
                raise ValueError("Incomplete evidence: " + str(directory))
            for stream in ("sessions", "references", "warmups", "profiles"):
                if metadata[stream + "_sha256"] != digest(directory / (stream + ".jsonl")):
                    raise ValueError("Changed raw evidence: " + str(directory / (stream + ".jsonl")))
            for name in ("audit.json", "audit_local.json"):
                audit = load(directory / name)
                if audit["status"] != "passed" or not audit["tensor_reload_requested"]:
                    raise ValueError("Missing full audit: " + str(directory / name))
                if audit["hashes"]["metadata"] != digest(directory / "metadata.json"):
                    raise ValueError("Stale audit: " + str(directory / name))
                for stream in ("sessions", "references", "warmups", "profiles"):
                    if audit["hashes"][stream + ".jsonl"] != metadata[stream + "_sha256"]:
                        raise ValueError("Stale raw-audit binding: " + str(directory / name))
                for snapshot in audit["snapshot_results"]:
                    path = directory / "snapshots" / snapshot["file"]
                    if digest(path) != snapshot["sha256"] or path.stat().st_size != snapshot["bytes"]:
                        raise ValueError("Changed snapshot: " + str(path))
            selected.update(p for p in directory.rglob("*") if p.is_file())
    for directory in (PROJECT / "stage3",):
        selected.update(p for p in directory.rglob("*") if p.is_file())
    selected.update((PROJECT / "notes").glob("stage3_*.md"))
    selected.update((PROJECT / "notes").glob("stage3_*recomputed.json"))
    selected.add(PROJECT / "notes/independent_stage3_report.py")
    selected.update((PROJECT / "logs").glob("stage3-*"))
    selected.update(PROJECT / "experiments" / name for name in
                    ("requirements.txt", "environment_freeze.txt", "run_optimized_replay.py", "fetch_pinned_models.py"))
    selected.update(PROJECT / "artifacts" / name for name in
                    (TITLE + ".md", TITLE + ".template.md", "build_pdf.py", "pdf_layout.py"))
    selected.update((PROJECT / "artifacts/figures").glob("stage3_*"))
    selected.update((PROJECT / "artifacts/qa" / TITLE).glob("*"))
    selected.add(ROOT / "output/pdf" / (TITLE + ".pdf"))
    selected.add(PROJECT / "stage3/RESULTS.md")
    # Keep a separate entry point: earlier-stage package README describes an
    # earlier experiment and is not a stage-3 reproduction guide.
    selected.add(PROJECT / "stage3/DELIVERY.md")
    excluded = {"delivery_manifest.json", "package_summary.json"}
    allowed = {".py", ".json", ".jsonl", ".md", ".txt", ".out", ".err",
               ".sbatch", ".sh", ".pt", ".pdf", ".png", ".xml"}
    files = sorted(p for p in selected if p.name not in excluded and p.suffix in allowed
                   and not any(part.startswith(".") or part == "__pycache__"
                               for part in p.relative_to(ROOT).parts))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing/nonordinary selected file: " + str(path))
        if path.suffix in {".py", ".json", ".jsonl", ".md", ".txt", ".out", ".err", ".sbatch", ".sh"}:
            # Pattern screen, not a claim of exhaustive secret detection.
            content = path.read_text()
            if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)", content):
                raise ValueError("Potential credential pattern in " + str(path))
    return files


def main():
    output = ROOT / "output/RECUT_Stage3_Validated_Artifact.zip"
    manifest = PROJECT / "stage3/reporting/delivery_manifest.json"
    summary_path = PROJECT / "stage3/reporting/package_summary.json"
    for path in (output, manifest, summary_path):
        if path.exists():
            raise FileExistsError("Refusing to replace an existing delivery output: " + str(path))
    files = select_files()
    records = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                "sha256": digest(p)} for p in files]
    manifest_text = json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "RECUT stage-3 bounded experiment; not model weights, environments or credentials",
        "entry_point": "research-next/stage3/DELIVERY.md", "records": records}, indent=2) + "\n"
    with manifest.open("x") as stream:
        stream.write(manifest_text)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest]:
            archive.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        for record in records + [{"path": str(manifest.relative_to(ROOT)), "sha256": digest(manifest)}]:
            result = hashlib.sha256()
            with archive.open(record["path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    result.update(block)
            if result.hexdigest() != record["sha256"]:
                raise ValueError("Archived bytes differ: " + record["path"])
    summary = {"output": str(output), "files": len(files) + 1,
               "bytes": output.stat().st_size, "sha256": digest(output),
               "all_archived_member_hashes_verified": True}
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
