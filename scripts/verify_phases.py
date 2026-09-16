"""Verify both RECUT publication phases using Python's standard library.

Phase 1 is read from its immutable Git commit, never from the current checkout.
Phase 2 checks the repository manifest. An optional release ZIP is checked in
place, without extraction or tensor deserialization. Integrity is not a rerun
of the experiments and does not establish scientific correctness.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile


PHASE1_COMMIT = "c542f51d4a8711da4d14628f47f81e998afdba3d"
PHASE1_MANIFEST = "research-next/artifacts/artifact_manifest.json"
PHASE2_MANIFEST = "docs/phases/phase-2-repository-manifest.json"
DELIVERY_MANIFEST = "research-next/stage3/reporting/delivery_manifest.json"
ARCHIVE_MEMBERS = 207
EXTERNAL_ARTIFACT = {
    "file": "RECUT_Stage3_Validated_Artifact.zip",
    "bytes": 990799407,
    "sha256": "535241cdc8e12f1a1a850ef1efc763b004e8975c235d2624a0c3e2003457e253",
    "url": "https://github.com/nbaliyan260/Thesis_P1/releases/download/phase-2-validated-prototype/RECUT_Stage3_Validated_Artifact.zip",
}


class VerificationError(ValueError):
    """An input failed an evidence-integrity gate."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def parse_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise VerificationError("Nonstandard JSON number: " + value)

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError("Invalid JSON: " + str(error)) from error


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Empty or non-string path")
    require(not any(ord(c) < 32 or ord(c) == 127 for c in name), "Control character in path")
    require("\\" not in name and ":" not in name, "Nonportable path: " + name)
    require(all(part not in ("", ".", "..") for part in name.split("/")), "Unsafe path: " + name)
    return name


def record_map(records, label):
    require(isinstance(records, list) and bool(records), label + ": empty or invalid records")
    result = {}
    for record in records:
        require(isinstance(record, dict), label + ": record must be an object")
        require(set(record) == {"path", "bytes", "sha256"}, label + ": invalid record fields")
        name = safe_name(record["path"])
        require(name not in result, label + ": duplicate path: " + name)
        require(type(record["bytes"]) is int and record["bytes"] >= 0, label + ": invalid size: " + name)
        require(isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                label + ": invalid SHA-256: " + name)
        result[name] = record
    return result


def file_path(root, name):
    root = Path(root).resolve()
    current = root
    for part in safe_name(name).split("/"):
        current = current / part
        require(not current.is_symlink(), "Symlink in repository path: " + name)
    require(current.is_file(), "Missing regular file: " + name)
    try:
        current.resolve().relative_to(root)
    except ValueError as error:
        raise VerificationError("Path escapes repository: " + name) from error
    return current


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def check_digest(size, digest, record, label):
    require(size == record["bytes"], "Size mismatch: " + label)
    require(digest == record["sha256"], "SHA-256 mismatch: " + label)


def verify_file(path, record, label):
    with Path(path).open("rb") as stream:
        check_digest(*digest_stream(stream), record, label)


def git(root, *arguments):
    process = subprocess.run(["git", "-C", str(root), *arguments], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    require(process.returncode == 0, "Git history unavailable or unreadable: "
            + process.stderr.decode("utf-8", errors="replace").strip()
            + ". Use a clone containing the phase-1 commit; a shallow clone may require fetching its history.")
    return process.stdout


def historical_tree(root, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "Invalid historical commit")
    result = {}
    for entry in git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        name = raw_name.decode("utf-8")
        require(name not in result, "Duplicate historical tree path")
        result[name] = (mode, kind, object_id)
    return result


def historical_blob(root, tree, name):
    safe_name(name)
    require(name in tree, "Missing historical file: " + name)
    mode, kind, object_id = tree[name]
    require(mode in ("100644", "100755") and kind == "blob", "Non-regular historical file: " + name)
    return git(root, "cat-file", "blob", object_id)


def verify_phase1(root, commit=PHASE1_COMMIT):
    tree = historical_tree(root, commit)
    manifest = parse_json(historical_blob(root, tree, PHASE1_MANIFEST))
    require(isinstance(manifest, dict) and "records" in manifest, "Invalid phase-1 manifest")
    records = record_map(manifest["records"], "phase 1")
    for name, record in records.items():
        payload = historical_blob(root, tree, name)
        check_digest(len(payload), hashlib.sha256(payload).hexdigest(), record, "phase 1: " + name)
    return len(records)


def verify_phase2(root):
    manifest = parse_json(file_path(root, PHASE2_MANIFEST).read_bytes())
    require(isinstance(manifest, dict) and manifest.get("phase") == "phase-2", "Invalid phase-2 manifest")
    require(manifest.get("external_artifact") == EXTERNAL_ARTIFACT, "Changed external artifact binding")
    records = record_map(manifest.get("records"), "phase 2")
    require(DELIVERY_MANIFEST in records, "Phase-2 manifest must bind the delivery manifest")
    require(PHASE2_MANIFEST not in records, "Repository manifest must not include itself")
    for name, record in records.items():
        verify_file(file_path(root, name), record, "phase 2: " + name)
    return records, manifest["external_artifact"]


def verify_artifact(root, artifact, external, repository_records, expected_members=ARCHIVE_MEMBERS):
    """Check the entire archive, anchored to a verified repository manifest.

    ``expected_members`` is injectable for tiny unit fixtures. The CLI always
    requires the published 207-member artifact and its fixed size/digest.
    """
    artifact = Path(artifact)
    require(not artifact.is_symlink() and artifact.is_file(), "Artifact is missing or is a symlink")
    verify_file(artifact, external, "release ZIP")
    require(DELIVERY_MANIFEST in repository_records, "Unbound delivery manifest")
    local_manifest = file_path(root, DELIVERY_MANIFEST)
    verify_file(local_manifest, repository_records[DELIVERY_MANIFEST], DELIVERY_MANIFEST)
    payload = local_manifest.read_bytes()
    manifest = parse_json(payload)
    require(isinstance(manifest, dict) and "records" in manifest, "Invalid delivery manifest")
    records = record_map(manifest["records"], "delivery")
    require(DELIVERY_MANIFEST not in records, "Delivery manifest must not include itself")
    expected = dict(records)
    expected[DELIVERY_MANIFEST] = repository_records[DELIVERY_MANIFEST]
    require(len(expected) == expected_members, "Unexpected declared archive member count")
    with zipfile.ZipFile(artifact) as archive:
        infos = archive.infolist()
        require(len(infos) == expected_members, "Unexpected ZIP member count")
        seen = set()
        for info in infos:
            name = safe_name(info.filename)
            require(name not in seen, "Duplicate ZIP member: " + name)
            seen.add(name)
            require(name in expected, "Unexpected ZIP member: " + name)
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(not info.is_dir() and mode in (0, stat.S_IFREG), "Non-regular ZIP member: " + name)
            require(not (info.flag_bits & 1), "Encrypted ZIP member: " + name)
            require(info.file_size == expected[name]["bytes"], "ZIP header size mismatch: " + name)
            with archive.open(info) as stream:
                check_digest(*digest_stream(stream), expected[name], "ZIP member: " + name)
        require(seen == set(expected), "Missing ZIP members")
    return len(expected)


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="Path to the separately downloaded validated release ZIP")
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        first = verify_phase1(root)
        second, external = verify_phase2(root)
        print(f"PASS: phase 1: {first} historical files match commit {PHASE1_COMMIT}.")
        print(f"PASS: phase 2: {len(second)} repository files match size and SHA-256.")
        if args.artifact is None:
            print("WARNING: external release archive NOT PROVIDED; full snapshot/tensor files were NOT CHECKED.")
            print("Run again with --artifact PATH to verify all 207 archive members; download: " + external["url"])
        else:
            members = verify_artifact(root, args.artifact, external, second)
            print(f"PASS: release ZIP size/SHA-256 and all {members} members (including the manifest) match.")
            print("Tensor bytes were checked, not deserialized or re-executed.")
        print("Integrity verification is not scientific validation, external replication, or publication-readiness certification.")
        return 0
    except (VerificationError, OSError, UnicodeError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        print("Phase integrity verification FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
