"""Write once, then verify stage-3 sources and prespecified inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "stage3"
FIXED = [
    "stage3/config_study.json", "stage3/config_debug.json", "stage3/model_manifest.json",
    "stage3/PLAN.md", "stage3/README.md", "stage3/run_hpc.sbatch", "stage3/setup_hpc.sbatch", "stage3/submit_study.sh",
    "stage2/controller.py", "stage2/test_transactional.py", "stage2/protocol_freeze.json",
    "experiments/recut_engine.py", "experiments/prefix_detector.py", "experiments/run_recut.py",
    "experiments/test_recut_engine.py", "experiments/tests_independent.py",
]
REQUIRED = {"recut_controller.py", "test_stage3_controller.py", "run_study.py",
            "analyze_study.py", "audit_study.py", "freeze_study.py", "fetch_models.py"}


def hashes():
    paths = sorted(STAGE.glob("*.py"))
    if not REQUIRED <= {p.name for p in paths}:
        raise RuntimeError("Required source is not ready")
    paths += [ROOT / name for name in FIXED]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "verify"))
    parser.add_argument("--file", type=Path, default=STAGE / "protocol_freeze.json")
    args = parser.parse_args()
    actual = hashes()
    if args.action == "write":
        data = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Stage3 sources frozen after synthetic tests, before any pretrained measurement",
                "files_sha256": actual,
                "expected_main_per_model": {"sessions": 786, "attempted_windows": 2340,
                    "committed_windows": 2322, "expected_refusals": 18, "snapshots": 7,
                    "native_post_prefill_parity_positions": 228, "excluded_warmups": 72,
                    "excluded_profiles": 24},
                "expected_models": 3,
                "scope": "Bounded software injection and protection cost; no physical faults, production throughput or publication guarantee"}
        with args.file.open("x") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
    else:
        expected = json.loads(args.file.read_text())["files_sha256"]
        if expected != actual:
            differences = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
            raise RuntimeError("Freeze mismatch: " + ", ".join(differences))
    print(json.dumps({"action": args.action, "files": len(actual), "passed": True}))


if __name__ == "__main__":
    main()
