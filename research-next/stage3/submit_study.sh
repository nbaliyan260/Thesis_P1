#!/bin/bash
# Submit only after pinned public downloads and source transfer are complete.
# Main waits for the ENTIRE three-model debug array, not just its own model.
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
.venv313/bin/python stage3/freeze_study.py verify
test -f stage3/downloaded_manifest.json
if test -e runs/stage3-debug-v1 || test -e runs/stage3-main-v1; then
  echo 'Refusing submission into existing evidence directories.' >&2
  exit 1
fi
recut_debug_job=$(sbatch --parsable --export=ALL,RECUT_PHASE=debug stage3/run_hpc.sbatch)
recut_main_job=$(sbatch --parsable --dependency="afterok:${recut_debug_job}" --export=ALL,RECUT_PHASE=main stage3/run_hpc.sbatch)
echo "Debug array: ${recut_debug_job}"
echo "Main array, gated on all debug tasks: ${recut_main_job}"
