#!/bin/bash
#SBATCH --job-name=recut3-fallback-check
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/stage3-fallback-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/stage3-fallback-%j.err
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
.venv313/bin/python stage3/freeze_study.py verify
.venv313/bin/python stage3/reporting/verify_fallback_diagnostic.py --check-original-only
.venv313/bin/python -u stage3/run_study.py --config stage3/reporting/fallback_config.json --manifest stage3/model_manifest.json --model HuggingFaceTB/SmolLM2-135M --output runs/stage3-fallback-diagnostic-v1/smol135m
.venv313/bin/python stage3/analyze_study.py runs/stage3-fallback-diagnostic-v1/smol135m
.venv313/bin/python stage3/audit_study.py runs/stage3-fallback-diagnostic-v1/smol135m --config stage3/reporting/fallback_config.json --manifest stage3/model_manifest.json
.venv313/bin/python stage3/freeze_study.py verify
date -Is
# After retrieval, run the frozen CPU auditor to audit_local.json, then the
# validation-only verifier. Main performance tables must never read this run.
