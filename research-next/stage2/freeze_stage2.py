"""Seal/verify the prespecified stage-2 source and protocol before GPU runs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "stage2/controller.py", "stage2/run_sessions.py", "stage2/analyze_sessions.py",
    "stage2/audit_sessions.py", "stage2/test_transactional.py", "stage2/freeze_stage2.py",
    "stage2/config_debug.json", "stage2/config_sessions.json", "stage2/PLAN.md", "stage2/README.md",
    "stage2/run_hpc.sbatch", "experiments/recut_engine.py", "experiments/run_recut.py",
    "experiments/prefix_detector.py", "experiments/model_manifest.json",
    "experiments/test_recut_engine.py", "experiments/tests_independent.py",
]


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "verify"))
    parser.add_argument("--file", type=Path, default=ROOT / "stage2/protocol_freeze.json")
    args = parser.parse_args()
    hashes = fingerprints()
    if args.action == "write":
        record = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "purpose": "Stage-2 prespecified code/config freeze, after synthetic tests and before pretrained measurement",
                  "files_sha256": hashes,
                  "expected_main": {"sessions": 192, "attempted_windows": 544,
                                    "committed_windows": 512, "expected_aborts": 32,
                                    "recovered_windows": 192, "native_parity_positions": 944},
                  "interpretation": "Controlled software injections only; repeated timing trials are not independent faults."}
        with args.file.open("x") as output:
            json.dump(record, output, indent=2)
            output.write("\n")
    else:
        expected = json.loads(args.file.read_text())["files_sha256"]
        if hashes != expected:
            changed = sorted(name for name in set(hashes) | set(expected) if hashes.get(name) != expected.get(name))
            raise RuntimeError("Freeze mismatch: " + ", ".join(changed))
    print(json.dumps({"action": args.action, "files": len(hashes), "passed": True}))


if __name__ == "__main__":
    main()
