#!/bin/bash
#SBATCH --job-name=recut-boundary
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/boundary-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/boundary-%j.err
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
export HF_HOME=/l/users/nazish.baliyan/recut-20260916/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
hostname
.venv313/bin/python -u experiments/capture_boundary_case.py \
  --manifest experiments/model_manifest.json --config experiments/config_primary.json \
  --primary-run runs/primary-v1 --amendment experiments/boundary_extension_amendment.json \
  --output runs/boundary-extension-v1
.venv313/bin/python experiments/audit_boundary_snapshot.py runs/boundary-extension-v1 \
  --primary-run runs/primary-v1 --manifest experiments/model_manifest.json \
  --config experiments/config_primary.json
date -Is
