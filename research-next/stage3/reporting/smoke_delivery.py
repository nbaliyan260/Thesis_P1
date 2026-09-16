"""Extract the new delivery in temporary storage and test its source closure.

This is a CPU-only portability check, not fresh model execution. It does not
modify the ZIP, original evidence, or any user's existing working directory.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[2]


def main():
    archive_path = PROJECT.parent / "output/RECUT_Stage3_Validated_Artifact.zip"
    result_path = PROJECT / "stage3/reporting/delivery_smoke.json"
    if result_path.exists():
        raise FileExistsError("Refusing to replace delivery verification")
    commands = [
        ["stage3/freeze_study.py", "verify"],
        ["-m", "pytest", "-q", "stage3/test_stage3_controller.py", "stage2/test_transactional.py",
         "experiments/test_recut_engine.py", "experiments/tests_independent.py",
         "stage3/reporting/test_reporting.py", "stage3/reporting/test_report_tables.py"],
        ["stage3/reporting/verify_fallback_diagnostic.py", "--output", "validation_reader.json"],
    ]
    for model in ("smol135m", "qwen05b", "qwen7b"):
        commands.append(["stage3/audit_study.py", "runs/stage3-main-v1/" + model,
                         "--output", "runs/stage3-main-v1/" + model + "/audit_reader.json"])
    results = []
    with tempfile.TemporaryDirectory(prefix="recut3-delivery-") as temporary:
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts or item.is_dir():
                    raise ValueError("Unexpected archive member: " + item.filename)
            archive.extractall(temporary)
        cwd = Path(temporary) / "research-next"
        for arguments in commands:
            completed = subprocess.run([sys.executable, *arguments], cwd=cwd,
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            result = {"arguments": arguments, "exit_code": completed.returncode, "output": completed.stdout}
            results.append(result)
            print(json.dumps(result), flush=True)
            if completed.returncode:
                raise RuntimeError("Extracted artifact failed: " + " ".join(arguments))
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with result_path.open("x") as stream:
        json.dump({"status": "passed", "archive_sha256": digest.hexdigest(),
                   "scope": "Temporary extracted CPU tests and full tensor audits, not model re-execution",
                   "commands": results}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "output": str(result_path)}))


if __name__ == "__main__":
    main()
