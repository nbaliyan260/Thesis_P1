#!/bin/bash
#SBATCH --job-name=recut-stage2
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/stage2-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/stage2-%j.err
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
.venv313/bin/python stage2/freeze_stage2.py verify
.venv313/bin/python -m pytest -q stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
.venv313/bin/python -u stage2/run_sessions.py --manifest experiments/model_manifest.json --config stage2/config_debug.json --output runs/stage2-debug-v1
.venv313/bin/python stage2/analyze_sessions.py runs/stage2-debug-v1
.venv313/bin/python stage2/audit_sessions.py runs/stage2-debug-v1 --config stage2/config_debug.json
.venv313/bin/python -u stage2/run_sessions.py --manifest experiments/model_manifest.json --config stage2/config_sessions.json --output runs/stage2-sessions-v1
.venv313/bin/python stage2/analyze_sessions.py runs/stage2-sessions-v1
.venv313/bin/python stage2/audit_sessions.py runs/stage2-sessions-v1
.venv313/bin/python stage2/freeze_stage2.py verify
date -Is
