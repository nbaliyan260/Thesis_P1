#!/bin/bash
#SBATCH --job-name=recut-eval
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/eval-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/eval-%j.err
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
nvidia-smi
.venv313/bin/python -m pytest -q experiments/test_recut_engine.py experiments/tests_independent.py
.venv313/bin/python -u experiments/run_recut.py \
  --manifest experiments/model_manifest.json \
  --config "experiments/${RECUT_CONFIG:-config_debug.json}" \
  --output "runs/${RECUT_RUN:-debug-v1}"
date -Is
