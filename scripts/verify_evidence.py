"""Verify the original research-package manifest without third-party dependencies."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research-next/artifacts/artifact_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    failures = []
    seen = set()
    for record in records:
        name = record["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            failures.append(f"Invalid or duplicate manifest path: {name}")
            continue
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or root not in path.resolve().parents:
            failures.append(f"Unsafe manifest target: {name}")
            continue
        if not path.is_file():
            failures.append(f"Missing: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["bytes"]:
            failures.append(f"Size mismatch: {name}")
        if digest.hexdigest() != record["sha256"]:
            failures.append(f"SHA-256 mismatch: {name}")
    if failures:
        print("Evidence integrity verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"PASS: {len(records)} original research-artifact files match size and SHA-256.")
    print("This verifies file integrity, not scientific correctness or publication readiness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
