#!/bin/bash
#SBATCH --job-name=recut-audit
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/audit-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/audit-%j.err
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
export OMP_NUM_THREADS=2
date -Is
hostname
.venv313/bin/python experiments/analyze_recut.py runs/primary-v1
.venv313/bin/python experiments/audit_results.py runs/primary-v1 \
  --config experiments/config_primary.json --manifest experiments/model_manifest.json --require-summary
date -Is
